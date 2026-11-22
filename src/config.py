"""Central settings. Change things here, not inside the notebooks."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SPLIT_DIR = DATA_DIR / "splits"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
PLOT_DIR = OUT_DIR / "plots"
METRIC_DIR = OUT_DIR / "metrics"

RAW_CSV = DATA_DIR / "mobile_phones_raw.csv"
CLEAN_CSV = DATA_DIR / "mobile_phones_clean.csv"

TARGET = "price"
CATEGORICAL = ["brand", "model", "condition", "location", "source"]
NUMERICAL = ["storage_gb", "ram_gb"]
FEATURES = CATEGORICAL + NUMERICAL

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15          # of the whole dataset (so 70 / 15 / 15)

# Categories that appear fewer times than this in the training data are grouped
# together by the one-hot encoder (stops 1,000+ one-off model names).
MIN_CATEGORY_FREQ = 3

# Rows whose ad_url is blank cannot be assigned to ikman or Dialog automatically.
# Look at them in notebook 01 first. If you KNOW they are all Dialog rows set this to "Dialog".
BLANK_URL_SOURCE = None   # None -> label them "Unknown" (and warn)

# Suspicious prices are only FLAGGED by default. After checking them by hand set True to drop.
DROP_SUSPICIOUS = False

# Display names (order = order used in tables, charts and the dashboard)
MODEL_NAMES = {
    "ridge": "Ridge Regression (baseline)",
    "random_forest": "Random Forest (Bagging)",
    "extra_trees": "Extra Trees (Bagging)",
    "xgboost": "XGBoost (Boosting)",
    "lightgbm": "LightGBM (Boosting)",
    "catboost": "CatBoost (Boosting)",
    "stacking": "Stacking (Ridge meta-model)",
    "hybrid": "Hybrid (RF + ANN)",
}
MODEL_FILES = {k: ("hybrid_rf_ann.pkl" if k == "hybrid" else f"{k}.pkl") for k in MODEL_NAMES}

CV_FOLDS = 5
