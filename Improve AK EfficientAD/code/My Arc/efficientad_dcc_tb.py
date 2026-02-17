
#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import itertools
import os
import random
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.cluster import MiniBatchKMeans

from common import (
    get_autoencoder,
    get_pdn_small,
    get_pdn_medium,
    ImageFolderWithoutTarget,
    ImageFolderWithPath,
    InfiniteDataloader,
)

# --------------------------
# Args
# --------------------------
def get_argparse():
    p = argparse.ArgumentParser()

    # base
    p.add_argument("-d", "--dataset", default="mvtec_ad", choices=["mvtec_ad", "mvtec_loco"])
    p.add_argument("-s", "--subdataset", default="bottle")
    p.add_argument("-o", "--output_dir", default="output/efficientad_dcc")
    p.add_argument("-m", "--model_size", default="small", choices=["small", "medium"])
    p.add_argument("-w", "--weights", default="models/teacher_small.pth")
    p.add_argument("-i", "--imagenet_train_path", default="none")
    p.add_argument("-a", "--mvtec_ad_path", default="./mvtec_anomaly_detection")
    p.add_argument("-b", "--mvtec_loco_path", default="./mvtec_loco_anomaly_detection")
    p.add_argument("-t", "--train_steps", type=int, default=70000)

    # colab friendly
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--tb_log_every", type=int, default=10)
    p.add_argument("--tb_eval_every", type=int, default=10000)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)

    # DCC params (idea)
    p.add_argument("--dcc_enable", action="store_true")
    p.add_argument("--dcc_k", type=int, default=128)
    p.add_argument("--dcc_pool", type=int, default=2)  # avgpool on feature map (reduce cost)
    p.add_argument("--dcc_samples_per_image", type=int, default=256)  # for kmeans fit
    p.add_argument("--dcc_max_train_images", type=int, default=400)   # cap for speed
    p.add_argument("--dcc_weight", type=float, default=0.20)         # mix weight in final map

    return p.parse_args()

# --------------------------
# constants / transforms
# --------------------------
on_gpu = torch.cuda.is_available()
out_channels = 384
image_size = 256

default_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_ae = transforms.RandomChoice([
    transforms.ColorJitter(brightness=0.2),
    transforms.ColorJitter(contrast=0.2),
    transforms.ColorJitter(saturation=0.2)
])

def train_transform(image):
    return default_transform(image), default_transform(transform_ae(image))

# --------------------------
# Teacher normalization
# --------------------------
@torch.no_grad()
def teacher_normalization(teacher, train_loader):
    mean_outputs = []
    for train_image, _ in tqdm(train_loader, desc="Computing mean of features"):
        if on_gpu:
            train_image = train_image.cuda(non_blocking=True)
        teacher_output = teacher(train_image)
        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        mean_outputs.append(mean_output)
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)[None, :, None, None]

    mean_distances = []
    for train_image, _ in tqdm(train_loader, desc="Computing std of features"):
        if on_gpu:
            train_image = train_image.cuda(non_blocking=True)
        teacher_output = teacher(train_image)
        dist = (teacher_output - channel_mean) ** 2
        mean_dist = torch.mean(dist, dim=[0, 2, 3])
        mean_distances.append(mean_dist)
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)[None, :, None, None]
    channel_std = torch.sqrt(channel_var + 1e-12)
    return channel_mean, channel_std

# --------------------------
# Base EfficientAD predict
# --------------------------
@torch.no_grad()
def predict_base(image, teacher, student, autoencoder, teacher_mean, teacher_std,
                 q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None):
    teacher_out = teacher(image)
    teacher_out = (teacher_out - teacher_mean) / teacher_std

    student_out = student(image)
    ae_out = autoencoder(image)

    map_st = torch.mean((teacher_out - student_out[:, :out_channels]) ** 2, dim=1, keepdim=True)
    map_ae = torch.mean((ae_out - student_out[:, out_channels:]) ** 2, dim=1, keepdim=True)

    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start + 1e-12)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start + 1e-12)

    map_base = 0.5 * map_st + 0.5 * map_ae
    return map_base, map_st, map_ae, teacher_out

@torch.no_grad()
def map_normalization_base(validation_loader, teacher, student, autoencoder, teacher_mean, teacher_std, desc="Map norm"):
    maps_st, maps_ae = [], []
    for image, _ in tqdm(validation_loader, desc=desc):
        if on_gpu:
            image = image.cuda(non_blocking=True)
        _, map_st, map_ae, _ = predict_base(image, teacher, student, autoencoder, teacher_mean, teacher_std)
        maps_st.append(map_st)
        maps_ae.append(map_ae)
    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae)

    q_st_start = torch.quantile(maps_st, q=0.9)
    q_st_end   = torch.quantile(maps_st, q=0.995)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end   = torch.quantile(maps_ae, q=0.995)
    return q_st_start, q_st_end, q_ae_start, q_ae_end

# --------------------------
# DCC: codebook + co-occurrence
# --------------------------
def _pool_feat(feat, pool):
    if pool <= 1:
        return feat
    return F.avg_pool2d(feat, kernel_size=pool, stride=pool)

@torch.no_grad()
def assign_tokens(feat, centers_t):
    
    # feat: (B,C,H,W) torch (GPU)
    # centers_t: (K,C) torch (GPU)
    # returns tokens: (B,H,W) long
    
    B, C, H, W = feat.shape
    x = feat.permute(0,2,3,1).reshape(-1, C)  # (B*H*W, C)
    x2 = (x * x).sum(dim=1, keepdim=True)     # (N,1)
    c2 = (centers_t * centers_t).sum(dim=1).unsqueeze(0)  # (1,K)
    # dist2 = ||x||^2 + ||c||^2 - 2 x c^T
    dist2 = x2 + c2 - 2.0 * (x @ centers_t.t())  # (N,K)
    tok = torch.argmin(dist2, dim=1).reshape(B, H, W)
    return tok

def build_dcc(train_loader, teacher, teacher_mean, teacher_std, dcc_k, dcc_pool, samples_per_image, max_train_images, seed):
    rng = np.random.default_rng(seed)

    # 1) collect samples for kmeans fit
    X_list = []
    n_img = 0
    for (img, _ ) in tqdm(train_loader, desc="DCC: sampling teacher features for KMeans"):
        n_img += 1
        if n_img > max_train_images:
            break
        if on_gpu:
            img = img.cuda(non_blocking=True)
        feat = teacher(img)
        feat = (feat - teacher_mean) / teacher_std
        feat = _pool_feat(feat, dcc_pool)  # (1,C,h,w)
        _, C, h, w = feat.shape
        n = h * w
        take = min(samples_per_image, n)
        idx = rng.choice(n, size=take, replace=False)
        vec = feat[0].permute(1,2,0).reshape(-1, C)[idx].detach().cpu().numpy().astype(np.float32)
        X_list.append(vec)

    X = np.concatenate(X_list, axis=0)
    kmeans = MiniBatchKMeans(
        n_clusters=dcc_k,
        batch_size=2048,
        random_state=seed,
        n_init="auto",
        max_iter=200
    )
    kmeans.fit(X)
    centers = kmeans.cluster_centers_.astype(np.float32)  # (K,C)

    # 2) build co-occurrence counts: counts[neighbor, center]
    counts = np.zeros((dcc_k, dcc_k), dtype=np.int64)
    offsets = [(0,1),(1,0),(0,-1),(-1,0)]  # 4-neighborhood

    # move centers to GPU for tokenization
    centers_t = torch.from_numpy(centers)
    if on_gpu:
        centers_t = centers_t.cuda()

    n_img = 0
    for (img, _) in tqdm(train_loader, desc="DCC: building co-occurrence matrix"):
        n_img += 1
        if n_img > max_train_images:
            break
        if on_gpu:
            img = img.cuda(non_blocking=True)
        feat = teacher(img)
        feat = (feat - teacher_mean) / teacher_std
        feat = _pool_feat(feat, dcc_pool)  # (1,C,h,w)

        tok = assign_tokens(feat, centers_t)[0].detach().cpu().numpy().astype(np.int32)  # (h,w)
        # pad
        tok_p = np.pad(tok, ((1,1),(1,1)), mode="edge")
        center = tok_p[1:-1,1:-1]

        for dy, dx in offsets:
            nb = tok_p[1+dy:1+dy+center.shape[0], 1+dx:1+dx+center.shape[1]]
            pair = (nb.reshape(-1) * dcc_k + center.reshape(-1)).astype(np.int64)
            bc = np.bincount(pair, minlength=dcc_k*dcc_k).reshape(dcc_k, dcc_k)
            counts += bc

    # 3) convert to log prob with smoothing
    alpha = 1.0  # Laplace smoothing
    row_sum = counts.sum(axis=1, keepdims=True).astype(np.float64)
    prob = (counts + alpha) / (row_sum + alpha * dcc_k + 1e-12)
    logp = np.log(prob + 1e-12).astype(np.float32)          # log P(center | neighbor)
    neg_logp = (-logp).astype(np.float32)                   # anomaly score = -logP

    return {
        "centers": centers.astype(np.float32),
        "neg_logp": neg_logp.astype(np.float32),
        "offsets": offsets,
        "dcc_k": int(dcc_k),
        "dcc_pool": int(dcc_pool),
    }

@torch.no_grad()
def dcc_map_from_teacherfeat(teacher_feat_norm, dcc_state):
    
    # teacher_feat_norm: (B,C,H,W) teacher feature after normalization
    # returns map_dcc: (B,1,H,W) in feature-map resolution (upsampled if pooled)
    
    centers = torch.from_numpy(dcc_state["centers"])
    neg_logp = torch.from_numpy(dcc_state["neg_logp"])
    dcc_k = dcc_state["dcc_k"]
    pool = dcc_state["dcc_pool"]
    offsets = dcc_state["offsets"]

    if on_gpu:
        centers = centers.cuda()
        neg_logp = neg_logp.cuda()

    feat = _pool_feat(teacher_feat_norm, pool)  # (B,C,h,w)
    B, C, h, w = feat.shape

    tok = assign_tokens(feat, centers)  # (B,h,w)
    tok_p = F.pad(tok, (1,1,1,1), mode="replicate")
    center = tok_p[:,1:-1,1:-1]  # (B,h,w)

    score = 0.0
    for dy, dx in offsets:
        nb = tok_p[:,1+dy:1+dy+h, 1+dx:1+dx+w]
        # score_dir = neg_logp[nb, center]
        score = score + neg_logp[nb, center]

    score = score / float(len(offsets))  # (B,h,w)
    score = score.unsqueeze(1)           # (B,1,h,w)

    # upsample back to teacher_feat_norm resolution if pooled
    if pool > 1:
        score = F.interpolate(score, size=(teacher_feat_norm.shape[-2], teacher_feat_norm.shape[-1]), mode="bilinear", align_corners=False)
    return score

@torch.no_grad()
def map_normalization_dcc(validation_loader, teacher, student, autoencoder, teacher_mean, teacher_std, dcc_state, desc="DCC map norm"):
    maps_dcc = []
    for image, _ in tqdm(validation_loader, desc=desc):
        if on_gpu:
            image = image.cuda(non_blocking=True)
        _, _, _, teacher_feat = predict_base(image, teacher, student, autoencoder, teacher_mean, teacher_std)
        md = dcc_map_from_teacherfeat(teacher_feat, dcc_state)  # (B,1,H,W)
        maps_dcc.append(md)
    maps_dcc = torch.cat(maps_dcc)
    q_start = torch.quantile(maps_dcc, q=0.9)
    q_end   = torch.quantile(maps_dcc, q=0.995)
    return q_start, q_end

# --------------------------
# Testing
# --------------------------
@torch.no_grad()
def test_image_auc(test_set, teacher, student, autoencoder, teacher_mean, teacher_std,
                   q_st_start, q_st_end, q_ae_start, q_ae_end,
                   save_dir=None,
                   dcc_state=None, q_dcc_start=None, q_dcc_end=None, dcc_weight=0.2,
                   desc="Inference"):
    
    # If dcc_state is None -> baseline map
    # else -> baseline + DCC combined
    
    y_true, y_score = [], []

    for image, target, path in tqdm(test_set, desc=desc):
        ow, oh = image.width, image.height
        x = default_transform(image)[None]
        if on_gpu:
            x = x.cuda(non_blocking=True)

        map_base, _, _, teacher_feat = predict_base(
            x, teacher, student, autoencoder, teacher_mean, teacher_std,
            q_st_start, q_st_end, q_ae_start, q_ae_end
        )

        if dcc_state is not None:
            md = dcc_map_from_teacherfeat(teacher_feat, dcc_state)
            md = 0.1 * (md - q_dcc_start) / (q_dcc_end - q_dcc_start + 1e-12)
            map_final = (1.0 - dcc_weight) * map_base + dcc_weight * md
        else:
            map_final = map_base

        # upsample to original size like baseline code
        map_final = F.pad(map_final, (4,4,4,4))
        map_final = F.interpolate(map_final, (oh, ow), mode="bilinear", align_corners=False)
        amap = map_final[0,0].detach().cpu().numpy()

        defect_class = os.path.basename(os.path.dirname(path))
        if save_dir is not None:
            out_dir = Path(save_dir) / defect_class
            out_dir.mkdir(parents=True, exist_ok=True)
            img_nm = os.path.split(path)[1].split(".")[0]
            tifffile.imwrite((out_dir / f"{img_nm}.tiff").as_posix(), amap.astype(np.float32))

        y_true.append(0 if defect_class == "good" else 1)
        y_score.append(float(np.max(amap)))

    auc = roc_auc_score(y_true=y_true, y_score=y_score) * 100.0
    return auc

# --------------------------
# Main
# --------------------------
def main():
    cfg = get_argparse()

    # seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # dataset path
    if cfg.dataset == "mvtec_ad":
        dataset_path = cfg.mvtec_ad_path
    else:
        dataset_path = cfg.mvtec_loco_path

    # output dirs
    train_output_dir = os.path.join(cfg.output_dir, "trainings", cfg.dataset, cfg.subdataset)
    tb_dir = os.path.join(train_output_dir, "tensorboard")
    os.makedirs(train_output_dir, exist_ok=True)

    # save maps separately for baseline and DCC
    test_output_base = os.path.join(cfg.output_dir, "anomaly_maps_base", cfg.dataset, cfg.subdataset, "test")
    test_output_dcc  = os.path.join(cfg.output_dir, "anomaly_maps_dcc",  cfg.dataset, cfg.subdataset, "test")
    os.makedirs(test_output_base, exist_ok=True)
    os.makedirs(test_output_dcc, exist_ok=True)

    writer = SummaryWriter(log_dir=tb_dir)
    writer.add_text("run/config", str(vars(cfg)))

    # data
    full_train_set = ImageFolderWithoutTarget(
        os.path.join(dataset_path, cfg.subdataset, "train"),
        transform=transforms.Lambda(train_transform)
    )
    test_set = ImageFolderWithPath(os.path.join(dataset_path, cfg.subdataset, "test"))

    # validation split
    if cfg.dataset == "mvtec_ad":
        train_size = int(0.9 * len(full_train_set))
        val_size = len(full_train_set) - train_size
        rng = torch.Generator().manual_seed(cfg.seed)
        train_set, val_set = torch.utils.data.random_split(full_train_set, [train_size, val_size], rng)
    else:
        train_set = full_train_set
        val_set = ImageFolderWithoutTarget(
            os.path.join(dataset_path, cfg.subdataset, "validation"),
            transform=transforms.Lambda(train_transform)
        )

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    train_loader_inf = InfiniteDataloader(train_loader)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=0)

    # penalty dataloader
    pretrain_penalty = (cfg.imagenet_train_path != "none")
    if pretrain_penalty:
        penalty_transform = transforms.Compose([
            transforms.Resize((2 * image_size, 2 * image_size)),
            transforms.RandomGrayscale(0.3),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        penalty_set = ImageFolderWithoutTarget(cfg.imagenet_train_path, transform=penalty_transform)
        penalty_loader = DataLoader(penalty_set, batch_size=1, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
        penalty_loader_inf = InfiniteDataloader(penalty_loader)
    else:
        penalty_loader_inf = itertools.repeat(None)

    # models
    if cfg.model_size == "small":
        teacher = get_pdn_small(out_channels)
        student = get_pdn_small(2*out_channels)
    else:
        teacher = get_pdn_medium(out_channels)
        student = get_pdn_medium(2*out_channels)

    teacher.load_state_dict(torch.load(cfg.weights, map_location="cpu"))
    autoencoder = get_autoencoder(out_channels)

    teacher.eval()
    student.train()
    autoencoder.train()

    if on_gpu:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()

    teacher_mean, teacher_std = teacher_normalization(teacher, train_loader)

    optimizer = torch.optim.Adam(itertools.chain(student.parameters(), autoencoder.parameters()), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.95*cfg.train_steps), gamma=0.1)

    # --------------------------
    # train loop (same as baseline)
    # --------------------------
    pbar = tqdm(range(cfg.train_steps))
    for it, (img_st, img_ae), img_pen in zip(pbar, train_loader_inf, penalty_loader_inf):
        if on_gpu:
            img_st = img_st.cuda(non_blocking=True)
            img_ae = img_ae.cuda(non_blocking=True)
            if img_pen is not None:
                img_pen = img_pen.cuda(non_blocking=True)

        # ST loss (hard mining)
        with torch.no_grad():
            t_out = teacher(img_st)
            t_out = (t_out - teacher_mean) / teacher_std

        s_out = student(img_st)[:, :out_channels]
        dist = (t_out - s_out) ** 2
        d_hard = torch.quantile(dist, q=0.999)
        loss_hard = torch.mean(dist[dist >= d_hard])

        if img_pen is not None:
            s_pen = student(img_pen)[:, :out_channels]
            loss_pen = torch.mean(s_pen ** 2)
            loss_st = loss_hard + loss_pen
        else:
            loss_pen = torch.tensor(0.0, device=loss_hard.device)
            loss_st = loss_hard

        # AE + STAE
        ae_out = autoencoder(img_ae)
        with torch.no_grad():
            t_ae = teacher(img_ae)
            t_ae = (t_ae - teacher_mean) / teacher_std
        s_ae = student(img_ae)[:, out_channels:]
        loss_ae = torch.mean((t_ae - ae_out) ** 2)
        loss_stae = torch.mean((ae_out - s_ae) ** 2)

        loss_total = loss_st + loss_ae + loss_stae

        optimizer.zero_grad(set_to_none=True)
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        if it % cfg.tb_log_every == 0:
            writer.add_scalar("train/loss_total", loss_total.item(), it)
            writer.add_scalar("train/loss_st", loss_st.item(), it)
            writer.add_scalar("train/loss_hard", loss_hard.item(), it)
            writer.add_scalar("train/loss_penalty", loss_pen.item(), it)
            writer.add_scalar("train/loss_ae", loss_ae.item(), it)
            writer.add_scalar("train/loss_stae", loss_stae.item(), it)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], it)

        if it % 10 == 0:
            pbar.set_description(f"loss={loss_total.item():.4f} st={loss_st.item():.4f} ae={loss_ae.item():.4f} stae={loss_stae.item():.4f}")

        if it % cfg.save_every == 0 and it > 0:
            torch.save(teacher, os.path.join(train_output_dir, "teacher_tmp.pth"))
            torch.save(student, os.path.join(train_output_dir, "student_tmp.pth"))
            torch.save(autoencoder, os.path.join(train_output_dir, "autoencoder_tmp.pth"))

    # final save
    teacher.eval(); student.eval(); autoencoder.eval()
    torch.save(teacher, os.path.join(train_output_dir, "teacher_final.pth"))
    torch.save(student, os.path.join(train_output_dir, "student_final.pth"))
    torch.save(autoencoder, os.path.join(train_output_dir, "autoencoder_final.pth"))

    # --------------------------
    # Baseline eval (same pipeline)
    # --------------------------
    q_st_s, q_st_e, q_ae_s, q_ae_e = map_normalization_base(
        val_loader, teacher, student, autoencoder, teacher_mean, teacher_std, desc="Baseline map normalization"
    )
    auc_base = test_image_auc(
        test_set, teacher, student, autoencoder, teacher_mean, teacher_std,
        q_st_s, q_st_e, q_ae_s, q_ae_e,
        save_dir=test_output_base,
        dcc_state=None,
        desc="Baseline inference"
    )
    writer.add_scalar("test/final_image_auc_base", auc_base, cfg.train_steps)
    print(f"[BASELINE] Final image auc: {auc_base:.4f}")

    # --------------------------
    # DCC build + eval
    # --------------------------
    auc_dcc = None
    if cfg.dcc_enable:
        dcc_state = build_dcc(
            train_loader=train_loader,
            teacher=teacher,
            teacher_mean=teacher_mean,
            teacher_std=teacher_std,
            dcc_k=cfg.dcc_k,
            dcc_pool=cfg.dcc_pool,
            samples_per_image=cfg.dcc_samples_per_image,
            max_train_images=cfg.dcc_max_train_images,
            seed=cfg.seed,
        )

        # save dcc artifacts
        np.save(os.path.join(train_output_dir, "dcc_centers.npy"), dcc_state["centers"])
        np.save(os.path.join(train_output_dir, "dcc_neg_logp.npy"), dcc_state["neg_logp"])
        with open(os.path.join(train_output_dir, "dcc_config.txt"), "w") as f:
            f.write(str(dcc_state))

        q_dcc_s, q_dcc_e = map_normalization_dcc(
            val_loader, teacher, student, autoencoder, teacher_mean, teacher_std, dcc_state, desc="DCC map normalization"
        )

        auc_dcc = test_image_auc(
            test_set, teacher, student, autoencoder, teacher_mean, teacher_std,
            q_st_s, q_st_e, q_ae_s, q_ae_e,
            save_dir=test_output_dcc,
            dcc_state=dcc_state, q_dcc_start=q_dcc_s, q_dcc_end=q_dcc_e, dcc_weight=cfg.dcc_weight,
            desc="DCC inference"
        )
        writer.add_scalar("test/final_image_auc_dcc", auc_dcc, cfg.train_steps)
        print(f"[DCC] Final image auc: {auc_dcc:.4f} (dcc_weight={cfg.dcc_weight})")

    # metrics json
    import json
    out = {
        "dataset": cfg.dataset,
        "subdataset": cfg.subdataset,
        "model_size": cfg.model_size,
        "train_steps": int(cfg.train_steps),
        "pretrain_penalty": bool(pretrain_penalty),
        "tb_dir": tb_dir,
        "train_output_dir": train_output_dir,
        "test_output_base": test_output_base,
        "test_output_dcc": test_output_dcc,
        "final_image_auc_base": float(auc_base),
        "final_image_auc_dcc": (float(auc_dcc) if auc_dcc is not None else None),
        "dcc_enable": bool(cfg.dcc_enable),
        "dcc_k": int(cfg.dcc_k),
        "dcc_pool": int(cfg.dcc_pool),
        "dcc_weight": float(cfg.dcc_weight),
    }
    with open(os.path.join(train_output_dir, "final_metrics_dcc.json"), "w") as f:
        json.dump(out, f, indent=2)

    writer.flush()
    writer.close()

if __name__ == "__main__":
    main()
