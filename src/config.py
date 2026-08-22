# Member 2 - model engineer
# central configuration

from pathlib import Path

# paths are based on repository root
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

RAW_CSV = DATA_DIR / "labelled_full.csv"
SPLITS_JSON = DATA_DIR / "splits.json"

PREPROCESSOR_PKL = MODELS_DIR / "preprocessor.joblib"
LABEL_ENCODER_PKL = MODELS_DIR / "label_encoder.joblib"
FEATURE_COLS_JSON = MODELS_DIR / "feature_columns.json"
BACKGROUND_CSV = RESULTS_DIR / "background_sample.csv"

COMPARISON_CSV = RESULTS_DIR / "model_comparison.csv"
EXPERIMENT_LOG_CSV = RESULTS_DIR / "experiment_log.csv"
PREDICTIONS_CSV = RESULTS_DIR / "test_predictions.csv"

for _d in (DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# experiment settings
# keep this fixed because results depend on it
PRIMARY_SEED = 42

# same split is used for every model run
# helps compare model stability
MODEL_SEEDS = [42, 7, 123, 2024, 999]

VAL_SIZE = 0.15
TEST_SIZE = 0.15

# order used for labels and evaluation outputs
CLASS_NAMES = ["At-Risk", "Distressed", "Healthy"]

# columns removed before training
LEAK_COLS = ["z_score", "label", "company_id", "year"]

# remove incomplete filing year
DROP_YEARS = [2025]

# neural model training defaults
MAX_EPOCHS = 200
PATIENCE = 20            # early stopping on validation F1-macro
BATCH_SIZE = 256
WEIGHT_DECAY = 1e-4
LR_DEFAULT = 1e-3

# Optuna search limits
N_TRIALS_XGB = 40
N_TRIALS_MLP = 25
N_TRIALS_TABTRANSFORMER = 25

DEVICE = "cpu"           # changed automatically when GPU is available