# Bean Leaf Disease Classification with Deep Neural Networks

This repository contains a full **end-to-end PyTorch pipeline** for classifying bean leaf diseases using the [AI-Lab-Makerere/beans](https://huggingface.co/datasets/AI-Lab-Makerere/beans) dataset on Google Colab.

The project covers:

- Data loading & preprocessing from Hugging Face
- A 3-layer fully connected neural network (MLP) implemented in PyTorch
- Multiple controlled experiments to study:
  - Normalization
  - Dropout
  - Noisy labels
  - Gaussian noise on test images
  - Test-Time Augmentation (TTA)
- Logging with TensorBoard
- Robust training with checkpoints saved to Google Drive

If you just clone this repo and follow the instructions below, you can **reproduce all experiments and plots** without needing prior deep learning expertise.

---

## 1. Problem Overview

We want to automatically classify bean leaf images into **three classes**:

- `0` → *Angular Leaf Spot* (diseased)
- `1` → *Bean Rust* (diseased)
- `2` → *Healthy* (no visible disease)

Given an image `x ∈ ℝ^{500×500×3}`, we resize it to `224×224×3`, then feed it into a neural network `f(x)` that outputs a probability distribution over the three classes.

The main goal is to:

1. Build a working end-to-end classifier.
2. Understand how **normalization, dropout, label quality, and noise** affect training and generalization.
3. Make the training pipeline robust to interruptions using checkpoints on Google Drive.

---

## 2. Dataset

We use the **Beans dataset** hosted on Hugging Face:

- Name: `AI-Lab-Makerere/beans`
- Type: Image classification
- Splits: `train`, `validation`, `test`
- Each example:
  - `image`: RGB image of a bean leaf (PIL Image)
  - `labels`: integer in `{0, 1, 2}`

### 2.1. Label Mapping

The project uses the following label mapping:

- `0` → `angular_leaf_spot`  (Angular Leaf Spot disease)
- `1` → `bean_rust`          (Bean Rust disease)
- `2` → `healthy`            (Healthy leaf)

This mapping is consistent with the official Hugging Face dataset card.

---

## 3. Model Architecture

We implement a **3-layer MLP (Multi-Layer Perceptron)** in PyTorch:

1. **Input**:
   - Image resized to `224×224×3`
   - Flattened to a vector of length `150528 = 3×224×224`

2. **Layer 1**:
   - `Linear(150528 → 512)`
   - `ReLU`
   - `Dropout(p = p1)`

3. **Layer 2**:
   - `Linear(512 → 256)`
   - `ReLU`
   - `Dropout(p = p2)`

4. **Output Layer**:
   - `Linear(256 → 3)` (logits for 3 classes)
   - `CrossEntropyLoss` is used, so the softmax is applied implicitly inside the loss.

In code:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

FLATTEN_DIM = 3 * 224 * 224
NUM_CLASSES = 3

class BeanMLP(nn.Module):
    def __init__(self, input_dim=FLATTEN_DIM,
                 hidden1=512, hidden2=256,
                 num_classes=NUM_CLASSES,
                 p1=0.3, p2=0.3):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)
        self.dropout1 = nn.Dropout(p=p1)
        self.dropout2 = nn.Dropout(p=p2)

    def forward(self, x):
        x = self.flatten(x)
        o1 = F.relu(self.fc1(x))
        o1 = self.dropout1(o1)
        o2 = F.relu(self.fc2(o1))
        o2 = self.dropout2(o2)
        logits = self.fc3(o2)
        return logits
