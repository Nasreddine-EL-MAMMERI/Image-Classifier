import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src.data import CLASS_NAMES, IMG_SIZE

MODEL_PATH = "models/best_model.keras"
VAL_DIR = "data/Valid"


def main():
    model = tf.keras.models.load_model(MODEL_PATH)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMG_SIZE,
        batch_size=32,
        label_mode="categorical",
        class_names=CLASS_NAMES,
        shuffle=False,
    )

    y_true = np.concatenate([y.numpy() for _, y in val_ds], axis=0)
    y_pred = model.predict(val_ds)
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    print(classification_report(y_true_labels, y_pred_labels,
                                target_names=CLASS_NAMES))
    print("Confusion matrix:\n", confusion_matrix(y_true_labels, y_pred_labels))
    acc = np.mean(y_true_labels == y_pred_labels)
    print(f"Validation accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()