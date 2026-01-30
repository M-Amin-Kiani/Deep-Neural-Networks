# Efficient CNN for Malaria Cell Classification (Parasitized vs. Uninfected)

A lightweight and computation-aware convolutional neural network (CNN) project for classifying red blood cell images into **Parasitized** (infected with _Plasmodium falciparum_) vs. **Uninfected** (healthy).  
The focus is **efficient inference on weak/low-power hardware** by reducing learnable parameters and **FLOPs** while preserving accuracy.

---

## ✨ What’s inside?

This repository provides:

- ✅ A PyTorch pipeline to **download, preprocess, and load** the malaria cell dataset.
- ✅ Three CNN variants with efficiency/accuracy trade-offs:
  1. **Model A (BaseCNN)**: standard 4-layer convolutional network.
  2. **Model B (SpecialCNN)**: replaces the 4th convolution with a **special partial convolution** (compute-saving).
  3. **Model C (Special + Pointwise)**: adds a **1×1 pointwise convolution** after the special layer to improve channel mixing.
- ✅ Training & evaluation for each model (**8 epochs**) + TensorBoard logs.
- ✅ FLOPs computation **exactly based on the project PDF definition** (MAC=2 FLOPs, bias add=1 FLOP per output element).
- ✅ Reporting of learnable parameters and FLOPs for all models.

---

## 📌 Background (why efficiency?)

Convolution layers can be expensive, especially on embedded devices or older GPUs/CPUs.  
This project explores a technique inspired by efficient architectures (e.g., MobileNet) to reduce compute while keeping classification performance.

Key ideas:

- **Parameter sharing** in convolution reduces parameters compared to dense layers.
- **Partial channel processing**: apply heavy convolution only to part of the channels.
- **Pointwise (1×1) convolution**: cheap but powerful channel mixing.

---

## 🗂️ Dataset

We use the public malaria dataset:

- Source (ZIP): `https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip`
- Total images: **27,558**
- Classes:
  - `Parasitized` (infected)
  - `Uninfected` (healthy)
- Preprocessing:
  - Resize to **28×28**
  - Normalize pixel values to **[0, 1]** (via `ToTensor()`)

✅ This dataset version is **perfectly balanced**:

- 13,779 parasitized
- 13,779 uninfected

---

## 🧠 Models

### Model A — BaseCNN (4× Conv)

- 4 convolution blocks with **3×3 kernels** (padding=1 to preserve spatial size)
- Adaptive average pooling → linear classifier

### Model B — SpecialCNN (compute-saving 4th layer)

Instead of a full 32→32 conv on all channels, we:

1. Split channels: 32 → 16 + 16
2. Apply a **3×3 conv (16→16)** only on the first half
3. Concatenate with the untouched second half

This reduces both **parameters** and **FLOPs**.

### Model C — SpecialCNN + Pointwise (accuracy boost)

After the special layer, we add a **1×1 pointwise conv** to improve channel mixing:

- Example: 32 → 64 with 1×1 conv  
  Then pool + classifier.

---

## 🧮 FLOPs definition (exact)

This repo reports **FLOPs exactly as defined in the project PDF**:

- **1 add = 1 FLOP**
- **1 multiply + 1 add (MAC) = 2 FLOPs**
- For Conv/Linear:
  - FLOPs = `2 × (#MACs)`
  - - bias adds: `+ 1 × (#output elements)` if bias is enabled

> FLOPs are computed by running a forward pass on a dummy input and accumulating FLOPs for **Conv2d** and **Linear** layers only (matching typical layer-FLOPs reporting).  
> If you want to include activations/pooling too, you can extend the counter.

---

## ⚙️ Installation

### 1) Clone

```bash
git clone https://github.com/<your-username>/<your-repo>.git
consider replacing placeholders above
cd <your-repo>
```

### 2) Create environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, install manually:

```bash
pip install torch torchvision tensorboard pillow matplotlib
```

---

## ▶️ Run (training + evaluation + TensorBoard)

### Option A — Google Colab

Open the notebook/script in Colab and run all cells. CUDA will be used automatically if available.

### Option B — Local machine

Run:

```bash
python train.py
```

After training, launch TensorBoard:

```bash
tensorboard --logdir runs
```

Then open the printed local URL.

---

## 📊 Example results (from one run)

Your results will vary slightly due to randomness, but a typical run may look like:

|             Model | Params | FLOPs (per 1 image) | Test Accuracy (after 8 epochs) |
| ----------------: | -----: | ------------------: | -----------------------------: |
|          A (Base) | 16,689 |          13,008,192 |                          ~0.68 |
|       B (Special) |  9,761 |           7,589,184 |                          ~0.67 |
| C (Special + 1×1) | 11,905 |           9,219,968 |                          ~0.79 |

✅ In the shared run, **adding pointwise conv improved accuracy significantly** compared to Model A/B.

---

## 🧩 Recommended project structure

```
.
├── train.py                 # main training script
├── models.py                # model definitions (A/B/C)
├── flops.py                 # FLOPs counter (PDF-style)
├── runs/                    # TensorBoard logs
└── README.md
```

> If your repo currently has everything in one notebook/script, that’s fine too.  
> Splitting into modules just makes it cleaner for GitHub.

---

## 🧪 Notes & Tips

- **BCEWithLogitsLoss** is used:
  - The model outputs **logits** (no sigmoid inside).
  - Sigmoid is only applied for accuracy computation.
- If you want more accuracy:
  - increase epochs
  - add data augmentation
  - use learning rate scheduling
- If you want faster inference:
  - reduce channels (e.g., 16→24→32 instead of 16→16→32→32)
  - try depthwise separable conv (MobileNet-style)

---

## 📚 References

1. G. Menghani, “Efficient deep learning: A survey on making deep learning models smaller, faster, and better,” _ACM Computing Surveys_, 55(12), 2023.
2. M. Sandler et al., “MobileNetV2: Inverted residuals and linear bottlenecks,” _CVPR_, 2018.
3. S. Rajaraman et al., “Pretrained convolutional neural networks as feature extractors toward improved malaria parasite detection…,” _PeerJ_, 2018.

---

## 📝 License

ui.ac.ir

---

## 🙌 Acknowledgements

Dataset provided by the National Library of Medicine (NLM) / NIH (public malaria cell images).

---

### Contact

- GitHub: `@M-Amin-Kiani`
