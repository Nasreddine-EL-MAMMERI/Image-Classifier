"""Génération du fichier de soumission.

Usage :
    python -m src.predict
    python -m src.predict --test-csv chemin/test.csv --test-dir chemin/Test

Produit submission.csv avec les colonnes ID et CLASS attendues par Kaggle
(mêmes noms et mêmes valeurs que sample_submission.csv).

"""

import argparse
import pathlib
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import (
    BEST_MODEL_PATH,
    CLASS_NAMES,
    IMG_SIZE,
    TEST_CSV,
    TEST_DIR,
)

BATCH_SIZE = 64
OUTPUT_CSV = "submission.csv"


def build_image_index(test_dir: pathlib.Path) -> dict:
    """Construit une fois l'index nom_de_fichier -> chemin.

    Parcourir le dossier pour CHAQUE image coûterait des millions de tests
    de fichiers sur un jeu de plusieurs milliers d'images.
    """
    index = {}
    for path in test_dir.rglob("*"):
        if path.is_file():
            index[path.name] = path
            index.setdefault(path.stem, path)
    return index


def resolve_path(image_id: str, index: dict, test_dir: pathlib.Path) -> pathlib.Path:
    """Retrouve le fichier correspondant à un identifiant de test.csv."""
    if image_id in index:
        return index[image_id]

    stem = pathlib.Path(image_id).stem
    if stem in index:
        return index[stem]

    for ext in (".jpg", ".jpeg", ".png"):
        candidate = test_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Image introuvable pour l'identifiant « {image_id} » dans {test_dir}"
    )


def load_batch(paths, target_size=IMG_SIZE) -> np.ndarray:
    """Charge une liste d'images en un tableau (N, H, W, 3)."""
    images = []
    for path in paths:
        img = tf.keras.utils.load_img(path, target_size=target_size)
        images.append(tf.keras.utils.img_to_array(img))
    return np.stack(images)


def validate(ids, classes, expected_rows: int) -> None:
    """Contrôle le fichier avant écriture."""
    if len(ids) != expected_rows:
        raise ValueError(
            f"{len(ids)} prédictions pour {expected_rows} lignes attendues."
        )
    if any(c is None or (isinstance(c, float) and np.isnan(c)) for c in classes):
        raise ValueError("Des prédictions sont manquantes.")

    unknown = set(classes) - set(CLASS_NAMES)
    if unknown:
        raise ValueError(f"Labels inattendus : {unknown}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", default=str(TEST_CSV))
    parser.add_argument("--test-dir", default=str(TEST_DIR))
    parser.add_argument("--output", default=OUTPUT_CSV)
    args = parser.parse_args()

    test_csv = pathlib.Path(args.test_csv)
    test_dir = pathlib.Path(args.test_dir)

    test_df = pd.read_csv(test_csv)
    if "ID" not in test_df.columns:
        raise ValueError(
            f"Colonne « ID » absente de {test_csv} "
            f"(colonnes trouvées : {list(test_df.columns)})"
        )

    ids = test_df["ID"].astype(str).tolist()
    print(f"{len(ids)} images à prédire.")

    index = build_image_index(test_dir)
    print(f"{len(index)} entrées indexées dans {test_dir}")

    model = tf.keras.models.load_model(str(BEST_MODEL_PATH))

    predictions = []
    for start in range(0, len(ids), BATCH_SIZE):
        chunk = ids[start : start + BATCH_SIZE]
        paths = [resolve_path(i, index, test_dir) for i in chunk]
        images = load_batch(paths)

        probs = model.predict(images, verbose=0)
        predictions.extend(CLASS_NAMES[int(np.argmax(p))] for p in probs)

        done = min(start + BATCH_SIZE, len(ids))
        print(f"  {done}/{len(ids)}", end="\r")

    print()

    validate(ids, predictions, expected_rows=len(test_df))

    submission = pd.DataFrame({"ID": ids, "CLASS": predictions})
    submission.to_csv(args.output, index=False)

    print(f"Fichier écrit : {args.output} ({len(submission)} lignes)")
    print(submission.head(10))


if __name__ == "__main__":
    main()
