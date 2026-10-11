from pathlib import Path

# ml/ directory
ML_DIR = Path(__file__).resolve().parents[2]

# Main directories
DATA_DIR = ML_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
POISONED_DATA_DIR = DATA_DIR / "poisoned"

MODEL_DIR = ML_DIR / "models"
RESULTS_DIR = ML_DIR / "experiments" / "results"

# Reproducibility
RANDOM_SEED = 42