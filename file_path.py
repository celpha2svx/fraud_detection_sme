import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


# --- Common Directory Paths ---
FILES_DIR = PROJECT_ROOT / "Files"
MODEL_DIR = PROJECT_ROOT / "DATA_PRE_FEATURE_ENG"

TRAIN_ID = FILES_DIR / "train_identity.csv"
TRAIN_TRANS = FILES_DIR / "train_transaction.csv"
MODEL_PATH = MODEL_DIR / "fraud_model_final.pth"

print(TRAIN_ID)
