"""Metrics, segment errors, importance and the auto-generated findings text."""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

from . import config


def regression_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "MAPE_%": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        "MedianAE": median_absolute_error(y_true, y_pred),
    }


def segment_errors(df, y_col, pred_col, by):
    """MAE / MAPE per category (source, condition, brand ...)."""
    d = df.copy()
    d["abs_err"] = (d[y_col] - d[pred_col]).abs()
    d["ape"] = d["abs_err"] / d[y_col] * 100
    out = d.groupby(by).agg(rows=(y_col, "size"), MAE=("abs_err", "mean"), MAPE=("ape", "mean"),
                            mean_price=(y_col, "mean")).reset_index()
    return out.sort_values("rows", ascending=False)


def permutation_table(model, X, y, n_repeats=5):
    """Model-agnostic importance on the RAW columns (works for RF, XGB, stacking and the hybrid)."""
    r = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=config.RANDOM_STATE,
                               scoring="neg_mean_absolute_error", n_jobs=1)
    return (pd.DataFrame({"feature": X.columns, "mae_increase": r.importances_mean, "std": r.importances_std})
            .sort_values("mae_increase", ascending=False))


def key_findings(test_df, val_df=None, gap_df=None):
    """Facts computed from the results, so the dashboard text always matches the numbers."""
    t = test_df.sort_values("MAE").reset_index(drop=True)
    best, worst = t.iloc[0], t.iloc[-1]
    lines = [f"Best model on the untouched test set: **{best['Model']}** "
             f"(MAE Rs. {best['MAE']:,.0f}, RMSE Rs. {best['RMSE']:,.0f}, R² {best['R2']:.3f})."]
    lines.append(f"Weakest model: **{worst['Model']}** (MAE Rs. {worst['MAE']:,.0f}); the best model's MAE is "
                 f"{(1 - best['MAE'] / worst['MAE']) * 100:.1f}% lower.")

    def row(key):
        r = t[t["Model"].str.contains(key, regex=False)]
        return r.iloc[0] if len(r) else None

    base = row("baseline")
    if base is not None and best["Model"] != base["Model"]:
        lines.append(f"The best ensemble cuts the MAE of the Ridge baseline by "
                     f"{(1 - best['MAE'] / base['MAE']) * 100:.1f}%.")
    rf = row("Random Forest")
    boosters = [r for r in (row("XGBoost"), row("LightGBM"), row("CatBoost")) if r is not None]
    if rf is not None and boosters:
        bb = min(boosters, key=lambda r: r["MAE"])
        diff = (1 - bb["MAE"] / rf["MAE"]) * 100
        txt = (f"Random Forest vs gradient boosting: best booster is **{bb['Model']}** "
               f"(MAE {abs(diff):.1f}% {'lower' if diff >= 0 else 'higher'} than Random Forest)")
        if "Train_time_s" in t:
            txt += f"; training time RF {rf['Train_time_s']:.1f}s vs {bb['Train_time_s']:.1f}s"
        lines.append(txt + ".")
    st = row("Stacking")
    singles = [r for r in (rf, *boosters, row("Extra Trees")) if r is not None]
    if st is not None and singles:
        base_mae = min(r["MAE"] for r in singles)
        diff = (1 - st["MAE"] / base_mae) * 100
        lines.append(f"Stacking vs its best single model: MAE {'improved' if diff > 0 else 'did not improve'} "
                     f"by {abs(diff):.1f}%.")
    hy = row("Hybrid")
    if hy is not None and rf is not None:
        diff = (1 - hy["MAE"] / rf["MAE"]) * 100
        lines.append(f"Hybrid RF + ANN vs Random Forest alone: MAE {abs(diff):.1f}% {'lower' if diff >= 0 else 'higher'}.")
    if gap_df is not None and len(gap_df):
        g = gap_df.sort_values("R2_gap", ascending=False).iloc[0]
        lines.append(f"Largest train-to-test R² gap (a sign of over-fitting): **{g['Model']}** ({g['R2_gap']:.3f}).")
    return lines


def mark_best(test_df, best_validation_name=None):
    """Add a 'Marker' column: trophy = lowest test MAE, tick = chosen on the validation set."""
    d = test_df.copy()
    lo = d["MAE"].idxmin()
    d.insert(0, "Marker", "")
    d.loc[lo, "Marker"] = "🏆 best test MAE"
    if best_validation_name is not None:
        m = d["Model"] == best_validation_name
        d.loc[m, "Marker"] = (d.loc[m, "Marker"] + " ✔ chosen on validation").str.strip()
    return d
