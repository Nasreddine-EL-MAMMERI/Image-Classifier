"""Configuration centrale du projet.

"""

import os
from pathlib import Path

# --- Chemins -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Sur Kaggle, définir DATA_ROOT vers /kaggle/input/.../dataset
# En local, laisser la valeur par défaut : <racine du projet>/data
DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(PROJECT_ROOT / "data")))

TRAIN_DIR = DATA_ROOT / "Train"
VALID_DIR = DATA_ROOT / "Valid"
TEST_DIR = DATA_ROOT / "Test"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Sur Kaggle, test.csv se trouve à côté du dossier dataset, pas dedans.
# On peut donc le redéfinir : TEST_CSV="chemin" python -m src.predict
# ou passer --test-csv / --test-dir en ligne de commande.
TEST_CSV = Path(os.environ.get("TEST_CSV", str(DATA_ROOT / "test.csv")))

BEST_MODEL_PATH = MODELS_DIR / "best_model.keras"
HISTORY_PATH = MODELS_DIR / "history.json"

# --- Données -------------------------------------------------------------
# L'ordre est alphabétique : c'est celui imposé par
# image_dataset_from_directory par défaut. On le fixe explicitement pour
# qu'un ajout de classe ne décale pas silencieusement les étiquettes.
CLASS_NAMES = ["cow", "elephant", "horse", "spider"]
NUM_CLASSES = len(CLASS_NAMES)

IMG_SIZE = (224, 224)

# --- Entraînement --------------------------------------------------------
SEED = 42

# Le notebook utilisait 8 × nombre de GPU (16 sur les 2×T4 de Kaggle).
BATCH_SIZE = 16

EPOCHS = 32

# Décroissance exponentielle du taux d'apprentissage.
# ATTENTION : avec 250 lots/époque et 32 époques (8 000 pas), un
# decay_steps de 10 000 signifie que le taux n'a JAMAIS décru.

DECAY_STEPS = 10000
DECAY_RATE = 0.9
INITIAL_LEARNING_RATE = 1e-3