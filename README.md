# 🐄🐘🐴🕷️ Animal Image Classification — TensorFlow / Keras

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Kaggle](https://img.shields.io/badge/Kaggle-28th%20place-20BEFF)
![Public LB](https://img.shields.io/badge/Public%20Leaderboard-0.8475-green)

> Train a custom CNN **from scratch** (no pretrained weights) for multi-class image classification on 4 animal classes: **cow, elephant, horse, spider**.
>
> **0.8475 Public Leaderboard accuracy — 28th place on Kaggle** 🏆

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Public Leaderboard accuracy | **0.8475** |
| Kaggle ranking | **28th place** |
| Project target (≥ 80%) | ✅ Achieved |

---

## 🧠 Model

- Custom CNN built from scratch with TensorFlow/Keras
- 4 convolutional blocks (32 → 64 → 128 → 256 filters)
- Batch Normalization + Dropout + Global Average Pooling
- Data augmentation: flip, rotation, zoom, brightness, contrast
- Trained on Kaggle GPU

---

## 📁 Dataset

- **3,997** training images (`Train/`)
- **786** validation images (`Valid/`)
- **4 classes:** cow, elephant, horse, spider
- Source: [Kaggle Competition](https://www.kaggle.com/competitions/open-cv-tf-project-1-image-classification-r-4)

> ⚠️ The dataset is not included in this repository (too large for Git). Download it from Kaggle.

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/Nasreddine-EL-MAMMERI/Image-Classifier.git
cd Image-Classifier

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate      # Windows
# source .venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt