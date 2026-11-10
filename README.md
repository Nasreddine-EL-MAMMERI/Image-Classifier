# Animal Image Classification — CNN from scratch

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Kaggle](https://img.shields.io/badge/Kaggle-28th%2F54-20BEFF)
![Params](https://img.shields.io/badge/params-651k-lightgrey)

A convolutional neural network built **from scratch — no pretrained weights** —
to classify four animal classes: **cow, elephant, horse, spider**.

| Metric | Value |
|---|---|
| Kaggle leaderboard accuracy | **0.8476** |
| Training accuracy | 0.866 |
| Validation accuracy | 0.852 (best epoch) |
| Parameters | 651,812 (2.5 MB) |
| Training time | ~26 min on 2×T4 (Kaggle) |

> The top of this leaderboard reaches 0.988, almost certainly with transfer
> learning. Training from scratch was a deliberate constraint: the point of
> the exercise was to understand what happens inside the layers, not to
> maximise the score.

---

## Results
in progress ... 

---

## Model

Custom CNN, 651,812 parameters:

- `Rescaling(1/255)` on the input — normalisation is part of the model, so
  training and inference see identically prepared images
- 3 convolutional blocks (32 → 64 → 128), each: two 3×3 convs, BatchNorm,
  max-pooling, dropout (0.15 / 0.25 / 0.35)
- Final block: one 256-filter conv + BatchNorm + GlobalAveragePooling
- Head: `Dense(256)` + dropout 0.4 + softmax over 4 classes
- Data augmentation: horizontal flip, rotation 0.05, zoom 0.1, contrast 0.2,
  brightness 0.1
- Optimiser: Adam with exponential decay schedule
- Early stopping on `val_accuracy` (patience 8), checkpoint on `val_loss`

---

## Dataset

- **3,997** training images, **786** validation images
- 4 classes, near-balanced: cow 997, elephant 1000, horse 1000, spider 1000
- Source: [Kaggle competition](https://www.kaggle.com/competitions/open-cv-tf-project-1-image-classification-r-4)

The dataset is not included. Download it and lay it out as:

```
data/
├── Train/
│   ├── cow/
│   ├── elephant/
│   ├── horse/
│   └── spider/
├── Valid/
│   └── ...same 4 folders...
├── Test/            # flat folder of test images
└── test.csv         # ID column
```

---

## Quickstart

```bash
git clone https://github.com/Nasreddine-EL-MAMMERI/Image-Classifier.git
cd Image-Classifier

python -m venv .venv
.\.venv\Scripts\activate        # Windows
# source .venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

Then:

```bash
python -m src.train       # train, writes models/best_model.keras + history.json
python -m src.evaluate    # writes the 3 files in results/
python -m src.predict     # writes submission.csv (ID, CLASS)
```

On Kaggle:

```bash
DATA_ROOT=/kaggle/input/open-cv-tf-project-1-image-classification-r-4/dataset \
TEST_CSV=/kaggle/input/open-cv-tf-project-1-image-classification-r-4/test.csv \
python -m src.train
```

---

## What I would improve next

Measured on my own training run, not guessed:

1. **The learning-rate schedule never fired.** `decay_steps` was 10,000 steps
   while the run totalled 8,000 (250 batches × 32 epochs). The rate stayed at
   0.001 throughout, which is visible as a saw-tooth validation accuracy in the
   last epochs. Fix: `decay_steps=2000` or a cosine decay.
2. **Inconsistent monitoring.** Early stopping watched `val_accuracy` while the
   checkpoint watched `val_loss`, so the two could pick different epochs. Pick
   one metric.
3. **BatchNorm placement.** The original orders `Conv → ReLU → BatchNorm`; the
   more common order is `Conv → BatchNorm → ReLU`. Worth measuring.
4. **The validation set was used twice** — for model selection and for final
   evaluation. The reported 0.852 is therefore slightly optimistic. A 5-fold
   cross-validation would remove the selection bias.

---

## Repository structure

```
├── src/
│   ├── config.py     paths, class names, hyperparameters
│   ├── data.py       dataset loading + augmentation
│   ├── model.py      CNN architecture
│   ├── train.py      training loop, callbacks, history export
│   ├── evaluate.py   metrics, confusion matrix, training curves
│   └── predict.py    batched inference, validated submission file
├── docs/decisions.md technical decisions and known limitations
├── results/          figures and classification report
└── requirements.txt
```

## License

MIT
