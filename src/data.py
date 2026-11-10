import tensorflow as tf

from src.config import (
    BATCH_SIZE,
    CLASS_NAMES,
    IMG_SIZE,
    SEED,
    TRAIN_DIR,
    VALID_DIR,
)


def get_augmentation() -> tf.keras.Sequential:
    """Pipeline d'augmentation identique à celui du notebook.

    flip horizontal, rotation 0,05, zoom 0,1, contraste 0,2, luminosité 0,1.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.2),
            tf.keras.layers.RandomBrightness(0.1),
        ],
        name="data_augmentation",
    )


def get_datasets(
    train_dir=TRAIN_DIR,
    valid_dir=VALID_DIR,
    batch_size: int = BATCH_SIZE,
    augment: bool = True,
):
    """Construit les jeux d'entraînement et de validation.

    Args:
        augment: applique l'augmentation au seul jeu d'entraînement.
            Mettre False pour l'évaluation (les images doivent rester
            telles quelles).

    Returns:
        (train_ds, valid_ds), tous deux prêts à l'itération.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        label_mode="categorical",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=IMG_SIZE,
        shuffle=True,
        seed=SEED,
        class_names=CLASS_NAMES,
    )

    valid_ds = tf.keras.utils.image_dataset_from_directory(
        valid_dir,
        label_mode="categorical",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=IMG_SIZE,
        shuffle=False,
        class_names=CLASS_NAMES,
    )

    if augment:
        augmentation = get_augmentation()
        # L'augmentation est appliquée après mise en cache nulle part :
        # chaque époque voit donc une version différente de chaque image.
        # (Si l'on ajoutait .cache() APRÈS ce .map(), chaque image serait
        # augmentée une seule fois et la même version resservirait à toutes les époques 
        train_ds = train_ds.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, valid_ds
