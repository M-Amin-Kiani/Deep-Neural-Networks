
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score

from common import (
    ImageFolderWithoutTarget,
    ImageFolderWithPath,
)

OUT_CHANNELS = 384
IMAGE_SIZE = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

default_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

transform_ae = transforms.RandomChoice([
    transforms.ColorJitter(brightness=0.2),
    transforms.ColorJitter(contrast=0.2),
    transforms.ColorJitter(saturation=0.2),
])

def train_transform(image):
    return default_transform(image), default_transform(transform_ae(image))

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

@torch.no_grad()
def teacher_normalization(teacher, train_loader):
    mean_outputs = []
    for train_image, _ in tqdm(train_loader, desc="Computing mean of features"):
        train_image = train_image.to(device, non_blocking=True)
        teacher_output = teacher(train_image)
        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        mean_outputs.append(mean_output)
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)[None, :, None, None]

    mean_distances = []
    for train_image, _ in tqdm(train_loader, desc="Computing std of features"):
        train_image = train_image.to(device, non_blocking=True)
        teacher_output = teacher(train_image)
        distance = (teacher_output - channel_mean) ** 2
        mean_distance = torch.mean(distance, dim=[0, 2, 3])
        mean_distances.append(mean_distance)
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)[None, :, None, None]
    channel_std = torch.sqrt(channel_var + 1e-12)
    return channel_mean, channel_std

@torch.no_grad()
def predict_base(image, teacher, student, autoencoder, teacher_mean, teacher_std):
    teacher_output = teacher(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std

    student_output = student(image)
    autoencoder_output = autoencoder(image)

    map_st = torch.mean((teacher_output - student_output[:, :OUT_CHANNELS]) ** 2, dim=1, keepdim=True)
    map_ae = torch.mean((autoencoder_output - student_output[:, OUT_CHANNELS:]) ** 2, dim=1, keepdim=True)

    return map_st, map_ae, teacher_output

@torch.no_grad()
def map_norm_base(validation_loader, teacher, student, autoencoder, teacher_mean, teacher_std, desc="Base map normalization"):
    maps_st, maps_ae = [], []
    for image, _ in tqdm(validation_loader, desc=desc):
        image = image.to(device, non_blocking=True)
        map_st, map_ae, _ = predict_base(image, teacher, student, autoencoder, teacher_mean, teacher_std)
        maps_st.append(map_st)
        maps_ae.append(map_ae)

    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae)

    q_st_start = torch.quantile(maps_st, q=0.9)
    q_st_end = torch.quantile(maps_st, q=0.995)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end = torch.quantile(maps_ae, q=0.995)
    return q_st_start, q_st_end, q_ae_start, q_ae_end

def normalize_map(map_x, q_start, q_end):
    return 0.1 * (map_x - q_start) / (q_end - q_start + 1e-12)

@torch.no_grad()
def build_dcc_state(
    teacher,
    teacher_mean,
    teacher_std,
    train_loader,
    k: int = 128,
    pool: int = 2,
    max_train_images: int = 250,
    samples_per_image: int = 256,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    teacher.eval()

    sampled_feats = []
    n_imgs = 0
    for img_st, _ in tqdm(train_loader, desc="DCC: sampling teacher features"):
        img_st = img_st.to(device, non_blocking=True)
        teacher_feat = teacher(img_st)
        teacher_feat = (teacher_feat - teacher_mean) / teacher_std
        feat = torch.nn.functional.avg_pool2d(teacher_feat, kernel_size=pool, stride=pool)

        b, c, h, w = feat.shape
        flat = feat.permute(0, 2, 3, 1).reshape(-1, c)

        take = min(samples_per_image, flat.shape[0])
        idx = rng.choice(flat.shape[0], size=take, replace=False)
        sampled_feats.append(flat[idx].detach().cpu().numpy().astype(np.float32))

        n_imgs += 1
        if n_imgs >= max_train_images:
            break

    X = np.concatenate(sampled_feats, axis=0)
    try:
        kmeans = MiniBatchKMeans(n_clusters=k, batch_size=4096, random_state=seed, n_init="auto")
    except TypeError:
        kmeans = MiniBatchKMeans(n_clusters=k, batch_size=4096, random_state=seed, n_init=10)
    kmeans.fit(X)
    centers = torch.from_numpy(kmeans.cluster_centers_.astype(np.float32)).to(device)

    counts = torch.zeros((k, k), dtype=torch.float32, device="cpu")

    def assign_tokens(feat_map: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat_map.shape
        flat = feat_map.permute(0, 2, 3, 1).reshape(-1, c)
        x2 = (flat ** 2).sum(dim=1, keepdim=True)
        c2 = (centers ** 2).sum(dim=1)[None, :]
        dist = x2 + c2 - 2.0 * flat @ centers.t()
        tok = torch.argmin(dist, dim=1).reshape(h, w)
        return tok

    n_imgs2 = 0
    for img_st, _ in tqdm(train_loader, desc="DCC: building co-occurrence"):
        img_st = img_st.to(device, non_blocking=True)
        teacher_feat = teacher(img_st)
        teacher_feat = (teacher_feat - teacher_mean) / teacher_std
        feat = torch.nn.functional.avg_pool2d(teacher_feat, kernel_size=pool, stride=pool)

        tok = assign_tokens(feat).detach().cpu().numpy().astype(np.int64)

        center = tok[1:-1, 1:-1].ravel()
        up = tok[:-2, 1:-1].ravel()
        down = tok[2:, 1:-1].ravel()
        left = tok[1:-1, :-2].ravel()
        right = tok[1:-1, 2:].ravel()

        for nb in (up, down, left, right):
            pair = nb * k + center
            bc = np.bincount(pair, minlength=k * k)
            counts += torch.from_numpy(bc.reshape(k, k).astype(np.float32))

        n_imgs2 += 1
        if n_imgs2 >= max_train_images:
            break

    counts_np = counts.numpy()
    counts_np += 1.0
    row_sum = counts_np.sum(axis=1, keepdims=True)
    cond = counts_np / (row_sum + 1e-12)

    return {
        "k": int(k),
        "pool": int(pool),
        "centers": centers.detach().cpu().numpy().astype(np.float32),
        "cond": cond.astype(np.float32),
        "max_train_images_used": int(min(n_imgs, n_imgs2)),
        "samples_per_image": int(samples_per_image),
    }

def _assign_tokens(feat: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    b, c, h, w = feat.shape
    flat = feat.permute(0, 2, 3, 1).reshape(-1, c)
    x2 = (flat ** 2).sum(dim=1, keepdim=True)
    c2 = (centers ** 2).sum(dim=1)[None, :]
    dist = x2 + c2 - 2.0 * flat @ centers.t()
    tok = torch.argmin(dist, dim=1).reshape(h, w)
    return tok

@torch.no_grad()
def dcc_map_from_teacherfeat(teacher_feat_norm: torch.Tensor, dcc_state: dict) -> torch.Tensor:
    k = int(dcc_state["k"])
    pool = int(dcc_state["pool"])
    centers = torch.from_numpy(dcc_state["centers"]).to(device)
    cond = torch.from_numpy(dcc_state["cond"]).to(device)

    feat = torch.nn.functional.avg_pool2d(teacher_feat_norm, kernel_size=pool, stride=pool)
    tok = _assign_tokens(feat, centers)
    h, w = tok.shape

    if h < 3 or w < 3:
        return torch.zeros((1, 1, h, w), device=device, dtype=torch.float32)

    c = tok[1:-1, 1:-1]
    up = tok[:-2, 1:-1]
    down = tok[2:, 1:-1]
    left = tok[1:-1, :-2]
    right = tok[1:-1, 2:]

    eps = 1e-12
    def nll(nb, c):
        p = cond[nb.reshape(-1), c.reshape(-1)]
        return (-torch.log(p + eps)).reshape(c.shape)

    nll_map = (nll(up, c) + nll(down, c) + nll(left, c) + nll(right, c)) / 4.0
    nll_map = torch.nn.functional.pad(nll_map[None, None, :, :], (1, 1, 1, 1), mode="replicate")
    return nll_map

@torch.no_grad()
def map_norm_dcc(validation_loader, teacher, student, autoencoder, teacher_mean, teacher_std, dcc_state, desc="DCC map normalization"):
    dcc_maps = []
    for image, _ in tqdm(validation_loader, desc=desc):
        image = image.to(device, non_blocking=True)
        _, _, teacher_feat = predict_base(image, teacher, student, autoencoder, teacher_mean, teacher_std)
        md = dcc_map_from_teacherfeat(teacher_feat, dcc_state)
        dcc_maps.append(md)
    dcc_maps = torch.cat(dcc_maps, dim=0)
    q_start = torch.quantile(dcc_maps, q=0.9)
    q_end = torch.quantile(dcc_maps, q=0.995)
    return q_start, q_end

@torch.no_grad()
def run_test_image_auc(
    test_set,
    teacher,
    student,
    autoencoder,
    teacher_mean,
    teacher_std,
    q_st_start,
    q_st_end,
    q_ae_start,
    q_ae_end,
    dcc_state=None,
    q_dcc_start=None,
    q_dcc_end=None,
    dcc_weight: float = 0.3,
    save_maps_dir: Path | None = None,
    desc: str = "Inference",
):
    y_true, y_score = [], []

    for image, target, path in tqdm(test_set, desc=desc):
        orig_w, orig_h = image.width, image.height
        image_t = default_transform(image)[None].to(device)

        map_st, map_ae, teacher_feat = predict_base(image_t, teacher, student, autoencoder, teacher_mean, teacher_std)
        base_map = 0.5 * normalize_map(map_st, q_st_start, q_st_end) + 0.5 * normalize_map(map_ae, q_ae_start, q_ae_end)

        final_map = base_map
        if dcc_state is not None:
            md = dcc_map_from_teacherfeat(teacher_feat, dcc_state)
            md = normalize_map(md, q_dcc_start, q_dcc_end)
            if md.shape[-2:] != base_map.shape[-2:]:
                md = torch.nn.functional.interpolate(md, size=base_map.shape[-2:], mode="bilinear", align_corners=False)
            final_map = (1.0 - dcc_weight) * base_map + dcc_weight * md

        final_map = torch.nn.functional.pad(final_map, (4, 4, 4, 4))
        final_map = torch.nn.functional.interpolate(final_map, (orig_h, orig_w), mode="bilinear", align_corners=False)
        final_map_np = final_map[0, 0].detach().cpu().numpy().astype(np.float32)

        defect_class = os.path.basename(os.path.dirname(path))
        y_true_image = 0 if defect_class == "good" else 1
        y_score_image = float(np.max(final_map_np))
        y_true.append(y_true_image)
        y_score.append(y_score_image)

        if save_maps_dir is not None:
            img_nm = os.path.split(path)[1].split(".")[0]
            out_dir = save_maps_dir / defect_class
            out_dir.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite((out_dir / f"{img_nm}.tiff").as_posix(), final_map_np)

    auc = roc_auc_score(y_true=y_true, y_score=y_score) * 100.0 if len(set(y_true)) > 1 else float("nan")
    return float(auc)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["mvtec_ad", "mvtec_loco"])
    p.add_argument("--subdataset", required=True)
    p.add_argument("--mvtec_ad_path", default="/content/datasets/mvtec_anomaly_detection")
    p.add_argument("--mvtec_loco_path", default="/content/datasets/mvtec_loco_anomaly_detection")
    p.add_argument("--baseline_train_dir", required=True)
    p.add_argument("--output_dir", default="/content/outputs/efficientad_dcc_post")
    p.add_argument("--train_steps", type=int, default=7000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--dcc_k", type=int, default=128)
    p.add_argument("--dcc_pool", type=int, default=2)
    p.add_argument("--dcc_weight", type=float, default=0.3)
    p.add_argument("--dcc_max_train_images", type=int, default=250)
    p.add_argument("--dcc_samples_per_image", type=int, default=256)
    p.add_argument("--save_maps", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    dataset_root = Path(args.mvtec_ad_path) if args.dataset == "mvtec_ad" else Path(args.mvtec_loco_path)
    baseline_dir = Path(args.baseline_train_dir)

    teacher = torch.load((baseline_dir / "teacher_final.pth").as_posix(), map_location=device)
    student = torch.load((baseline_dir / "student_final.pth").as_posix(), map_location=device)
    autoencoder = torch.load((baseline_dir / "autoencoder_final.pth").as_posix(), map_location=device)

    teacher.eval(); student.eval(); autoencoder.eval()
    teacher.to(device); student.to(device); autoencoder.to(device)

    full_train_set = ImageFolderWithoutTarget(
        (dataset_root / args.subdataset / "train").as_posix(),
        transform=transforms.Lambda(train_transform),
    )
    test_set = ImageFolderWithPath((dataset_root / args.subdataset / "test").as_posix())

    if args.dataset == "mvtec_ad":
        train_size = int(0.9 * len(full_train_set))
        validation_size = len(full_train_set) - train_size
        rng = torch.Generator().manual_seed(args.seed)
        train_set, validation_set = torch.utils.data.random_split(full_train_set, [train_size, validation_size], rng)
    else:
        train_set = full_train_set
        validation_set = ImageFolderWithoutTarget(
            (dataset_root / args.subdataset / "validation").as_posix(),
            transform=transforms.Lambda(train_transform),
        )

    train_loader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(validation_set, batch_size=1, shuffle=False, num_workers=0)

    out_root = Path(args.output_dir)
    train_out = out_root / "trainings" / args.dataset / args.subdataset
    train_out.mkdir(parents=True, exist_ok=True)

    maps_base_dir = out_root / "anomaly_maps_base" / args.dataset / args.subdataset / "test"
    maps_dcc_dir = out_root / "anomaly_maps_dcc" / args.dataset / args.subdataset / "test"
    if args.save_maps:
        maps_base_dir.mkdir(parents=True, exist_ok=True)
        maps_dcc_dir.mkdir(parents=True, exist_ok=True)

    tb_dir = train_out / "tensorboard"
    writer = SummaryWriter(log_dir=str(tb_dir))
    writer.add_text("run/config", json.dumps(vars(args), indent=2))

    for fn in ["teacher_final.pth", "student_final.pth", "autoencoder_final.pth"]:
        src = baseline_dir / fn
        if src.exists():
            shutil.copy2(src, train_out / fn)

    teacher_mean, teacher_std = teacher_normalization(teacher, train_loader)

    q_st_start, q_st_end, q_ae_start, q_ae_end = map_norm_base(
        validation_loader=val_loader,
        teacher=teacher,
        student=student,
        autoencoder=autoencoder,
        teacher_mean=teacher_mean,
        teacher_std=teacher_std,
        desc="Base map normalization",
    )

    auc_base = run_test_image_auc(
        test_set=test_set,
        teacher=teacher, student=student, autoencoder=autoencoder,
        teacher_mean=teacher_mean, teacher_std=teacher_std,
        q_st_start=q_st_start, q_st_end=q_st_end,
        q_ae_start=q_ae_start, q_ae_end=q_ae_end,
        dcc_state=None,
        save_maps_dir=(maps_base_dir if args.save_maps else None),
        desc="Baseline inference",
    )
    writer.add_scalar("test/image_auc_baseline", auc_base, args.train_steps)

    dcc_state = build_dcc_state(
        teacher=teacher,
        teacher_mean=teacher_mean,
        teacher_std=teacher_std,
        train_loader=train_loader,
        k=args.dcc_k,
        pool=args.dcc_pool,
        max_train_images=args.dcc_max_train_images,
        samples_per_image=args.dcc_samples_per_image,
        seed=args.seed,
    )

    np.save((train_out / "dcc_centers.npy").as_posix(), dcc_state["centers"])
    np.save((train_out / "dcc_cond.npy").as_posix(), dcc_state["cond"])
    with open((train_out / "dcc_meta.json").as_posix(), "w") as f:
        json.dump({k: v for k, v in dcc_state.items() if k not in ["centers", "cond"]}, f, indent=2)

    q_dcc_start, q_dcc_end = map_norm_dcc(
        validation_loader=val_loader,
        teacher=teacher, student=student, autoencoder=autoencoder,
        teacher_mean=teacher_mean, teacher_std=teacher_std,
        dcc_state=dcc_state,
        desc="DCC map normalization",
    )

    auc_dcc = run_test_image_auc(
        test_set=test_set,
        teacher=teacher, student=student, autoencoder=autoencoder,
        teacher_mean=teacher_mean, teacher_std=teacher_std,
        q_st_start=q_st_start, q_st_end=q_st_end,
        q_ae_start=q_ae_start, q_ae_end=q_ae_end,
        dcc_state=dcc_state, q_dcc_start=q_dcc_start, q_dcc_end=q_dcc_end,
        dcc_weight=args.dcc_weight,
        save_maps_dir=(maps_dcc_dir if args.save_maps else None),
        desc="DCC inference",
    )

    writer.add_scalar("test/image_auc_dcc", auc_dcc, args.train_steps)
    writer.add_scalar("test/image_auc_delta", auc_dcc - auc_base, args.train_steps)

    metrics = {
        "dataset": args.dataset,
        "subdataset": args.subdataset,
        "train_steps_ref": args.train_steps,
        "baseline_train_dir": str(baseline_dir),
        "output_dir": str(out_root),
        "auc_image_baseline": float(auc_base),
        "auc_image_dcc": float(auc_dcc),
        "auc_image_delta": float(auc_dcc - auc_base),
        "dcc_weight": float(args.dcc_weight),
        "dcc_k": int(args.dcc_k),
        "dcc_pool": int(args.dcc_pool),
        "dcc_max_train_images_used": int(dcc_state["max_train_images_used"]),
    }
    with open((train_out / "final_metrics_dcc.json").as_posix(), "w") as f:
        json.dump(metrics, f, indent=2)

    writer.flush()
    writer.close()
    print("DONE. Saved:", train_out)

if __name__ == "__main__":
    main()
