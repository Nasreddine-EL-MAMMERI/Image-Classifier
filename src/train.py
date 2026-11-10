"""Entraînement du CNN.

Usage (depuis la racine du projet) :
    python -m src.train

Produit :
    models/best_model.keras   meilleur modèle selon val_loss
    models/final_model.keras  état en fin d'entraînement
    models/history.json       courbes (loss/accuracy, train et validation)
"""

import json
import pathlib
import sys

# Permet aussi bien `python -m src.train` que `python src/train.py`.
if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import tensorflow as tf

from src.config import (
    BEST_MODEL_PATH,
    EPOCHS,
    HISTORY_PATH,
    MODELS_DIR,
    NUM_CLASSES,
    SEED,
)
from src.data import get_datasets
from src.model import build_model, compile_model


def main() -> None:
    # Graine globale : initialisation des poids, dropout, augmentation.
    # Sans cela, deux exécutions donnent deux scores différents.
    tf.keras.utils.set_random_seed(SEED)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Chargement des données…")
    train_ds, valid_ds = get_datasets(augment=True)

    model = build_model(NUM_CLASSES)
    compile_model(model)
    model.summary()

    callbacks = [
        # Le notebook arrêtait sur val_accuracy…
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        # …mais sauvegardait le modèle selon val_loss.
        # On conserve ce comportement d'origine, documenté dans
        # docs/decisions.md : c'est une incohérence à corriger en v2.
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=0,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    model.save(str(MODELS_DIR / "final_model.keras"))

    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)

    best_val_acc = max(history.history["val_accuracy"])
    best_epoch = history.history["val_accuracy"].index(best_val_acc) + 1
    print(
        f"\nEntraînement terminé. "
        f"Meilleure exactitude de validation : {best_val_acc:.4f} "
        f"(époque {best_epoch}/{len(history.history['loss'])})"
    )
    print(f"Courbes enregistrées dans {HISTORY_PATH}")


if __name__ == "__main__":
    main()
