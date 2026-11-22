"""Train + evaluate all models from the command line.

    python -m src.train_models                # clean (if needed) -> validate -> refit -> test -> save
    python -m src.train_models --rebuild      # re-run cleaning + splitting from the raw CSV first
    python -m src.train_models --tune         # also tune RF / ExtraTrees / XGBoost / LightGBM / CatBoost (slower)
    python -m src.train_models --no-cv        # skip the 5-fold cross-validation (much faster)

Protocol (same as the notebooks):
  1. fit every model on TRAIN, score on VALIDATION   -> used to choose the best model / hybrid weight
  2. refit every model on TRAIN + VALIDATION         -> score once on the untouched TEST set
     (training time of this refit is reported) and run 5-fold CV on TRAIN + VALIDATION
  3. save models/*.pkl, outputs/model_comparison.csv and the supporting metric files
"""
import argparse
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_validate

from . import config, models as M, preprocess as P
from .evaluate import permutation_table, regression_metrics, segment_errors
from .explain import grouped_importance

HYBRID_GRID = [round(w, 2) for w in np.arange(0.1, 1.0, 0.1)]
TREE_MODELS = ["random_forest", "extra_trees", "xgboost", "lightgbm", "catboost"]


def build(key, best_w=0.5):
    return M.make_hybrid(best_w) if key == "hybrid" else M.available_factories()[key]()


def fit_timed(model, X, y):
    t = time.perf_counter()
    model.fit(X, y)
    return model, time.perf_counter() - t


def search_hybrid_weights(X_train, y_train, X_val, y_val):
    """Fit RF + ANN once on train, then try every RF/ANN weight on the validation set."""
    hyb = M.make_hybrid().fit(X_train, y_train)
    rf_pred, ann_pred = hyb.predict_components(X_val)
    rows = []
    for w in HYBRID_GRID:
        m = regression_metrics(y_val, w * rf_pred + (1 - w) * ann_pred)
        rows.append({"rf_weight": w, "ann_weight": round(1 - w, 2), "val_MAE": m["MAE"],
                     "val_RMSE": m["RMSE"], "val_R2": m["R2"]})
    table = pd.DataFrame(rows)
    best_w = float(table.sort_values("val_MAE").iloc[0]["rf_weight"])
    comp = {"RF alone": regression_metrics(y_val, rf_pred), "ANN alone": regression_metrics(y_val, ann_pred)}
    return best_w, table, hyb, comp


def main(rebuild=False, tune=False, tune_iter=12, cv=True):
    t0 = time.time()
    for d in (config.MODEL_DIR, config.METRIC_DIR, config.PLOT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    if rebuild or not (config.SPLIT_DIR / "train.csv").exists():
        print("Cleaning raw CSV and creating splits ...")
        clean, report, splits = P.run_preprocessing()
        print(report.to_string(index=False))
    else:
        splits = P.load_splits()
    tr, va, te = splits["train"], splits["val"], splits["test"]
    X_tr, y_tr = P.xy(tr)
    X_va, y_va = P.xy(va)
    X_te, y_te = P.xy(te)
    print(f"train/val/test = {len(tr)}/{len(va)}/{len(te)} rows")

    keys = list(M.available_factories())
    print("models:", ", ".join(keys))

    if tune:
        for name in [k for k in TREE_MODELS if k in keys]:
            best, cv_mae, _ = M.tune(name, X_tr, y_tr, n_iter=tune_iter)
            print(f"tuned {name}: CV MAE {cv_mae:,.0f}  params={best}")

    # ---------- 1. validation round ----------
    best_w, w_table, _, comps = search_hybrid_weights(X_tr, y_tr, X_va, y_va)
    w_table.to_csv(config.METRIC_DIR / "hybrid_weight_search.csv", index=False)
    print(f"hybrid: best RF weight on validation = {best_w}")

    val_rows = []
    for key in keys:
        model, secs = fit_timed(build(key, best_w), X_tr, y_tr)
        val_rows.append({"Model": config.MODEL_NAMES[key], **regression_metrics(y_va, model.predict(X_va)),
                         "Train_time_s": secs})
        print(f"  validation {val_rows[-1]['Model']}: MAE={val_rows[-1]['MAE']:,.0f}  ({secs:.1f}s)")
    val_df = pd.DataFrame(val_rows)
    val_df.to_csv(config.OUT_DIR / "validation_comparison.csv", index=False)
    best_name = val_df.sort_values("MAE").iloc[0]["Model"]

    # ---------- 2. final round: refit on train+val, score on test, CV ----------
    X_full, y_full = pd.concat([X_tr, X_va]), pd.concat([y_tr, y_va])
    kf = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    cols = ["id", "title", "brand", "model", "condition", "location", "source", "price", "ad_url"]
    preds = te[[c for c in cols if c in te.columns]].copy()
    test_rows, gap_rows, fitted, imp_tables = [], [], {}, []
    for key in keys:
        label = config.MODEL_NAMES[key]
        model, secs = fit_timed(build(key, best_w), X_full, y_full)
        fitted[key] = model
        p_te, p_fit = model.predict(X_te), model.predict(X_full)
        m_te, m_fit = regression_metrics(y_te, p_te), regression_metrics(y_full, p_fit)
        row = {"Model": label, **m_te, "Train_time_s": secs}
        if cv:
            r = cross_validate(build(key, best_w), X_full, y_full, cv=kf, n_jobs=1, error_score="raise",
                               scoring={"mae": "neg_mean_absolute_error", "r2": "r2"})
            row.update(CV_MAE_mean=-r["test_mae"].mean(), CV_MAE_std=r["test_mae"].std(),
                       CV_R2_mean=r["test_r2"].mean())
        test_rows.append(row)
        gap_rows.append({"Model": label, "R2_train": m_fit["R2"], "R2_test": m_te["R2"],
                         "R2_gap": m_fit["R2"] - m_te["R2"], "MAE_train": m_fit["MAE"], "MAE_test": m_te["MAE"]})
        preds[label] = p_te
        joblib.dump(model, config.MODEL_DIR / config.MODEL_FILES[key], compress=3)
        if key in TREE_MODELS:
            g = grouped_importance(model)
            g.insert(0, "Model", label)
            imp_tables.append(g)
        print(f"  test {label}: MAE={m_te['MAE']:,.0f} R2={m_te['R2']:.3f} "
              f"time={secs:.1f}s" + (f" CV_MAE={row['CV_MAE_mean']:,.0f}±{row['CV_MAE_std']:,.0f}" if cv else ""))

    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(config.OUT_DIR / "model_comparison.csv", index=False)
    pd.DataFrame(gap_rows).to_csv(config.METRIC_DIR / "overfitting_gap.csv", index=False)
    preds.to_csv(config.OUT_DIR / "test_predictions.csv", index=False)
    if imp_tables:
        pd.concat(imp_tables).to_csv(config.METRIC_DIR / "feature_importance.csv", index=False)
    pd.DataFrame(comps).T.reset_index().rename(columns={"index": "component"}).to_csv(
        config.METRIC_DIR / "hybrid_components_validation.csv", index=False)

    seg = []
    for label in test_df["Model"]:
        for by in ("source", "condition"):
            s = segment_errors(preds, "price", label, by)
            s.insert(0, "Model", label)
            s.insert(1, "segment_by", by)
            seg.append(s.rename(columns={by: "segment"}))
    pd.concat(seg).to_csv(config.METRIC_DIR / "error_by_segment.csv", index=False)

    # best model = lowest VALIDATION MAE; saved as best_model.pkl
    inv = {v: k for k, v in config.MODEL_NAMES.items()}
    best_key = inv[best_name]
    joblib.dump(fitted[best_key], config.MODEL_DIR / "best_model.pkl", compress=3)
    (config.OUT_DIR / "best_model.txt").write_text(best_name)
    permutation_table(fitted[best_key], X_te, y_te).to_csv(
        config.METRIC_DIR / "permutation_importance_best.csv", index=False)
    print(f"best model (by validation MAE): {best_name}")
    print(f"done in {(time.time() - t0) / 60:.1f} min")
    return test_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--tune-iter", type=int, default=12)
    ap.add_argument("--no-cv", action="store_true")
    a = ap.parse_args()
    main(rebuild=a.rebuild, tune=a.tune, tune_iter=a.tune_iter, cv=not a.no_cv)
