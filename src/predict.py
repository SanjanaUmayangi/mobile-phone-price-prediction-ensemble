"""Helpers used by the Streamlit dashboard (kept out of the app so they can be tested)."""
import joblib
import numpy as np
import pandas as pd

from . import config

VERDICTS = ("Good deal", "Fair price", "Overpriced")


def load_models():
    """{display name: fitted model} for every model file that exists, in the standard order."""
    out = {}
    for key, label in config.MODEL_NAMES.items():
        path = config.MODEL_DIR / config.MODEL_FILES[key]
        if path.exists():
            out[label] = joblib.load(path)
    return out


def build_options(clean):
    """Dropdown options for the dashboard, derived from the clean dataset."""
    by_brand = {b: sorted(g["model"].unique()) for b, g in clean.groupby("brand")}
    return {
        "brands": sorted(by_brand),
        "models_by_brand": by_brand,
        "conditions": sorted(clean["condition"].unique()),
        "locations": sorted(clean["location"].unique()),
        "sources": sorted(clean["source"].unique()),
        "ram": sorted(clean["ram_gb"].dropna().unique()),
        "storage": sorted(clean["storage_gb"].dropna().unique()),
    }


def make_input(brand, model, condition, storage_gb, ram_gb, location, source):
    return pd.DataFrame([{
        "brand": brand, "model": model, "condition": condition,
        "storage_gb": storage_gb, "ram_gb": ram_gb, "location": location, "source": source,
    }])[config.FEATURES]


def predict_all(models, row):
    return {name: float(m.predict(row)[0]) for name, m in models.items()}


# ----------------------------------------------------------------------------
# Innovation: "Fair Price Checker" - compare an advertised price with the model's fair-price estimate
# ----------------------------------------------------------------------------
def deal_verdict(asking, predicted, tolerance=0.15):
    """Return (verdict, difference) where difference = asking / predicted - 1.

    asking more than `tolerance` above the predicted price -> 'Overpriced'
    asking more than `tolerance` below                      -> 'Good deal'
    otherwise                                               -> 'Fair price'
    """
    diff = float(asking) / float(predicted) - 1.0
    if diff > tolerance:
        return "Overpriced", diff
    if diff < -tolerance:
        return "Good deal", diff
    return "Fair price", diff


def scan_listings(preds_df, model_col, tolerance=0.15):
    """Apply the Fair Price Checker to every listing in a predictions table (needs 'price' + model_col)."""
    d = preds_df.copy()
    d["fair_price"] = d[model_col]
    d["difference_%"] = (d["price"] / d["fair_price"] - 1.0) * 100
    d["verdict"] = np.where(d["difference_%"] > tolerance * 100, "Overpriced",
                            np.where(d["difference_%"] < -tolerance * 100, "Good deal", "Fair price"))
    return d
