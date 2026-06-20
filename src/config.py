# config.py
# Central place for all paths and hyperparameters

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
# Update DATA_ROOT to point to your local COVIDx dataset
DATA_ROOT   = "data/"
TRAIN_DIR   = os.path.join(DATA_ROOT, "train")
VAL_DIR     = os.path.join(DATA_ROOT, "val")
TEST_DIR    = os.path.join(DATA_ROOT, "test")
TRAIN_TXT   = os.path.join(DATA_ROOT, "train.txt")
VAL_TXT     = os.path.join(DATA_ROOT, "val.txt")
TEST_TXT    = os.path.join(DATA_ROOT, "test.txt")

# ── Output ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR      = "outputs/"
WEIGHTS_PATH    = os.path.join(OUTPUT_DIR, "best_model.weights.h5")
FINAL_MODEL     = os.path.join(OUTPUT_DIR, "final_model.h5")
SUMMARY_PATH    = os.path.join(OUTPUT_DIR, "model_summary.txt")

# ── Model ──────────────────────────────────────────────────────────────────────
INPUT_SHAPE = (224, 224, 1)

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE      = 64
EPOCHS          = 50
LEARNING_RATE   = 1e-5
RANDOM_SEED     = 42

# ── Callbacks ──────────────────────────────────────────────────────────────────
EARLY_STOP_PATIENCE = 5
LR_REDUCE_PATIENCE  = 6
LR_REDUCE_FACTOR    = 0.5
MIN_LR              = 1e-7
