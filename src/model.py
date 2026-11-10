"""Architecture du CNN.

Reproduction EXACTE du modèle du notebook Kaggle qui a produit 0,84756.

Vérification : le modèle construit ici totalise 651 812 paramètres
(650 404 entraînables, 1 408 non entraînables), comme l'annonçait
`model.summary()` dans le notebook.

"""

import tensorflow as tf

from src.config import (
    DECAY_RATE,
    DECAY_STEPS,
    IMG_SIZE,
    INITIAL_LEARNING_RATE,
    NUM_CLASSES,
)


def conv_block(x, filters: int, dropout_rate: float):
    """Bloc convolutif du notebook.

    Ordre d'origine : Conv(relu) → BatchNorm → Conv(relu) → BatchNorm
                      → MaxPool → Dropout.

    NB : l'activation AVANT la normalisation est l'ordre utilisé dans le
    notebook. L'ordre plus courant est Conv → BatchNorm → ReLU. On conserve
    l'original pour rester fidèle au modèle qui a produit le score ; le
    changement est identifié comme piste v2 dans docs/decisions.md.
    """
    x = tf.keras.layers.Conv2D(
        filters, 3, activation="relu", padding="same"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(
        filters, 3, activation="relu", padding="same"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x


def build_model(num_classes: int = NUM_CLASSES, input_shape=(*IMG_SIZE, 3)):
    """Construit le CNN (aucun poids pré-entraîné)."""
    inputs = tf.keras.Input(shape=input_shape)

    # Normalisation des pixels : [0, 255] -> [0, 1].
    # Elle fait partie du modèle, donc l'inférence reçoit les mêmes
    # images brutes que l'entraînement : aucune divergence train/production.
    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)

    x = conv_block(x, 32, dropout_rate=0.15)
    x = conv_block(x, 64, dropout_rate=0.25)
    x = conv_block(x, 128, dropout_rate=0.35)

    # Dernier bloc : une seule convolution, puis global average pooling.
    x = tf.keras.layers.Conv2D(
        256, 3, activation="relu", padding="same"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Tête de classification
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs, name="animal_classifier_scratch")


def compile_model(model):
    """Compile avec la décroissance exponentielle du notebook."""
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=INITIAL_LEARNING_RATE,
        decay_steps=DECAY_STEPS,
        decay_rate=DECAY_RATE,
        staircase=True,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        # Pas de label_smoothing : le notebook n'en utilisait pas.
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
