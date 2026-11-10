"""Évaluation du modèle sur le jeu de validation.

Usage :
    python -m src.evaluate

Produit dans results/ :
    classification_report.txt  précision / rappel / F1 / support par classe
    confusion_matrix.png       matrice de confusion 4×4
    training_curves.png       perte et exactitude par époque 
"""

import json
import pathlib
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")  # Pas d'affichage : on écrit des fichiers.

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src.config import (
    BEST_MODEL_PATH,
    CLASS_NAMES,
    HISTORY_PATH,
    RESULTS_DIR,
)
from src.data import get_datasets


def plot_training_curves(history_path: pathlib.Path, out_path: pathlib.Path) -> None:
    """Trace perte et exactitude, entraînement vs validation."""
    with open(history_path, encoding="utf-8") as f:
        history = json.load(f)

    epochs = range(1, len(history["loss"]) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(12, 9))

    ax_loss.plot(epochs, history["loss"], label="Entraînement")
    ax_loss.plot(epochs, history["val_loss"], label="Validation")
    ax_loss.set_xlabel("Époque")
    ax_loss.set_ylabel("Perte (entropie croisée)")
    ax_loss.set_title("Perte par époque")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(epochs, history["accuracy"], label="Entraînement")
    ax_acc.plot(epochs, history["val_accuracy"], label="Validation")
    ax_acc.set_xlabel("Époque")
    ax_acc.set_ylabel("Exactitude")
    ax_acc.set_title("Exactitude par époque")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure enregistrée : {out_path}")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(str(BEST_MODEL_PATH))

    # augment=False : on évalue sur les images telles quelles.
    _, valid_ds = get_datasets(augment=False)

    y_true = np.concatenate([y.numpy() for _, y in valid_ds], axis=0)
    probs = model.predict(valid_ds, verbose=0)

    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(probs, axis=1)

    accuracy = float(np.mean(y_true_labels == y_pred_labels))

    report = classification_report(
        y_true_labels,
        y_pred_labels,
        target_names=CLASS_NAMES,
        digits=4,
    )

    print(f"Exactitude de validation : {accuracy:.4f}")
    print()
    print(report)

    report_path = RESULTS_DIR / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Exactitude de validation : {accuracy:.4f}\n\n")
        f.write(report)
    print(f"Rapport enregistré : {report_path}")

    # --- Matrice de confusion ------------------------------------------
    cm = confusion_matrix(y_true_labels, y_pred_labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Reds",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        annot_kws={"size": 13},
        ax=ax,
    )
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")
    ax.set_title(f"Matrice de confusion — validation ({len(y_true)} images)")
    fig.tight_layout()

    cm_path = RESULTS_DIR / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Matrice enregistrée : {cm_path}")

    # --- Courbes d'entraînement ----------------------------------------
    if HISTORY_PATH.exists():
        plot_training_curves(HISTORY_PATH, RESULTS_DIR / "training_curves.png")
    else:
        print(
            "history.json absent : lancez d'abord `python -m src.train` "
            "pour générer les courbes."
        )


if __name__ == "__main__":
    main()
