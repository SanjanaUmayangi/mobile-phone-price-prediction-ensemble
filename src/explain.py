"""Explainability: permutation importance (all models) and SHAP (tree models, optional)."""
import numpy as np
import pandas as pd


def shap_summary(fitted_model, X_sample, out_path=None, max_display=15):
    """SHAP summary for a fitted Random Forest / XGBoost model made by src.models.
    Returns the shap values or None if the shap package is not installed."""
    try:
        import matplotlib.pyplot as plt
        import shap
    except ImportError:
        print("shap is not installed - run: pip install shap")
        return None
    pipe = fitted_model.regressor_              # Pipeline(prep, model) inside the log-target wrapper
    prep, tree = pipe.named_steps["prep"], pipe.named_steps["model"]
    Xt = prep.transform(X_sample)
    names = prep.get_feature_names_out()
    sv = shap.TreeExplainer(tree).shap_values(Xt)   # values are in log-price units
    shap.summary_plot(sv, Xt, feature_names=names, max_display=max_display, show=False)
    if out_path:
        plt.gcf().savefig(out_path, dpi=150, bbox_inches="tight")
    return sv


def tree_feature_importance(fitted_model, top=15):
    pipe = fitted_model.regressor_
    names = pipe.named_steps["prep"].get_feature_names_out()
    imp = pipe.named_steps["model"].feature_importances_
    return pd.DataFrame({"feature": names, "importance": imp}).sort_values("importance", ascending=False).head(top)


def grouped_importance(fitted_model):
    """Built-in importance of a fitted tree model, summed back onto the RAW features
    (brand, model, condition, ...) and normalised to 100%. Works for RF, ExtraTrees, XGBoost, LightGBM, CatBoost."""
    from . import config
    pipe = fitted_model.regressor_
    names = list(pipe.named_steps["prep"].get_feature_names_out())
    imp = np.asarray(pipe.named_steps["model"].feature_importances_, dtype=float)
    totals = {}
    for n, v in zip(names, imp):
        raw = None
        body = n.split("__", 1)[1] if "__" in n else n
        for col in config.CATEGORICAL + config.NUMERICAL:
            if body == col or body.startswith(col + "_") or body.endswith("_" + col):
                raw = col
                if body.startswith(col + "_") and col in config.CATEGORICAL:
                    break
        totals[raw or body] = totals.get(raw or body, 0.0) + v
    s = pd.Series(totals)
    s = s / s.sum() * 100 if s.sum() > 0 else s
    return s.sort_values(ascending=False).rename("importance_%").reset_index().rename(columns={"index": "feature"})
