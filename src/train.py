import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from src.data import get_datasets
from src.model import build_model, compile_model

EPOCHS = 30
NUM_CLASSES = 4


def main():
    train_ds, val_ds = get_datasets("data/Train", "data/Valid")

    model = build_model(NUM_CLASSES)
    compile_model(model)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=7, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "models/best_model.keras",
            monitor="val_accuracy",
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    model.save("models/final_model.keras")
    best_val = max(history.history.get("val_accuracy", [0]))
    print(f"\nTraining done. Best val accuracy: {best_val:.4f}")


if __name__ == "__main__":
    main()