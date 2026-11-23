"""Streamlit dashboard - Sri Lankan mobile phone price prediction with ensemble learning.

Run from the project root:   streamlit run app/streamlit_app.py
(Run `python -m src.train_models --rebuild` first so that models/ and outputs/ exist.)

Pages: Overview | Model Comparison | Price Prediction & Fair Price Checker | Deal Finder | Model Insights
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src import config, predict as PR
from src.evaluate import key_findings, mark_best

st.set_page_config(page_title="LK Mobile Phone Price Predictor", page_icon="📱", layout="wide")


# ----------------------------------------------------------------------------- data loading
@st.cache_data
def read_csv(path):
    return pd.read_csv(path) if Path(path).exists() else None


@st.cache_resource
def get_models():
    return PR.load_models()


clean = read_csv(config.CLEAN_CSV)
test_df = read_csv(config.OUT_DIR / "model_comparison.csv")
val_df = read_csv(config.OUT_DIR / "validation_comparison.csv")
gap_df = read_csv(config.METRIC_DIR / "overfitting_gap.csv")
seg_df = read_csv(config.METRIC_DIR / "error_by_segment.csv")
w_df = read_csv(config.METRIC_DIR / "hybrid_weight_search.csv")
pi_df = read_csv(config.METRIC_DIR / "permutation_importance_best.csv")
imp_df = read_csv(config.METRIC_DIR / "feature_importance.csv")
preds_df = read_csv(config.OUT_DIR / "test_predictions.csv")
best_txt = config.OUT_DIR / "best_model.txt"
best_name = best_txt.read_text() if best_txt.exists() else None

if clean is None or test_df is None:
    st.error("Cleaned data or results not found. Run `python -m src.train_models --rebuild` first.")
    st.stop()

PAGES = ["Overview", "Model Comparison", "Price Prediction & Fair Price Checker", "Deal Finder", "Model Insights"]
st.sidebar.title("📱 Phone Price Predictor")
page = st.sidebar.radio("Page", PAGES)
st.sidebar.caption("CM4371 Machine Learning & Pattern Recognition - Ensemble Learning assignment")
st.sidebar.caption("Innovation: **Fair Price Checker** - is an advertised price fair, a good deal or overpriced?")


def rs(x):
    return f"Rs. {x:,.0f}"


def verdict_box(verdict, diff, tol):
    msg = f"{verdict}: the asking price is {diff * 100:+.1f}% compared with the model's fair-price estimate (±{tol * 100:.0f}% = fair)."
    if verdict == "Good deal":
        st.success("✅ " + msg)
    elif verdict == "Overpriced":
        st.error("⚠️ " + msg)
    else:
        st.info("⚖️ " + msg)


# ----------------------------------------------------------------------------- Overview
if page == "Overview":
    st.title("Sri Lankan Mobile Phone Price Prediction")
    st.write("Predicts the **advertised / catalogue price (LKR)** of a phone from its brand, model, condition, "
             "RAM, storage, location and data source. It compares **8 models**: a Ridge baseline, "
             "**bagging** (Random Forest, Extra Trees), **boosting** (XGBoost, LightGBM, CatBoost), "
             "**stacking** and a **hybrid RF + ANN**. The Fair Price Checker turns the predictions into a "
             "buyer-friendly verdict: *good deal / fair price / overpriced*.")
    c = st.columns(4)
    c[0].metric("Records (after cleaning)", f"{len(clean):,}")
    c[1].metric("Average price", rs(clean["price"].mean()))
    c[2].metric("Brands", clean["brand"].nunique())
    c[3].metric("Data sources", clean["source"].nunique())

    left, right = st.columns(2)
    with left:
        st.subheader("Price distribution")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(clean["price"], bins=50, color="#4C72B0")
        ax.set_xlabel("Price (LKR)"); ax.set_ylabel("Listings")
        st.pyplot(fig)
    with right:
        st.subheader("Data source distribution")
        st.bar_chart(clean["source"].value_counts())
    st.subheader("Median price by brand (top 12 brands by listings)")
    top = clean["brand"].value_counts().head(12).index
    st.bar_chart(clean[clean["brand"].isin(top)].groupby("brand")["price"].median().sort_values())
    st.subheader("Sample rows")
    st.dataframe(clean.drop(columns=[c for c in ["ad_url", "suspicious_price"] if c in clean.columns]).head(10))
    st.caption("ikman.lk prices are sellers' asking prices (used and new); Dialog prices are retail catalogue prices "
               "for brand-new phones. Neither is a confirmed sale price.")

# ----------------------------------------------------------------------------- Model comparison
elif page == "Model Comparison":
    st.title("Model Comparison")
    st.caption("Final numbers are on the untouched 15% test set. Models were refitted on train + validation data; "
               "the best model and the hybrid weight were chosen on the validation set only. "
               "CV = 5-fold cross-validation on train + validation.")
    for line in key_findings(test_df, val_df, gap_df):
        st.markdown(f"- {line}")

    st.subheader("All models - one table")
    fmt = {"MAE": "{:,.0f}", "RMSE": "{:,.0f}", "R2": "{:.3f}", "MAPE_%": "{:.1f}", "MedianAE": "{:,.0f}",
           "Train_time_s": "{:.1f}", "CV_MAE_mean": "{:,.0f}", "CV_MAE_std": "{:,.0f}", "CV_R2_mean": "{:.3f}"}
    shown = mark_best(test_df, best_name)
    st.dataframe(shown.style.format({k: v for k, v in fmt.items() if k in shown.columns}))
    st.caption("🏆 = lowest test MAE. ✔ = model chosen on the validation set (saved as models/best_model.pkl). "
               "They can differ - that is normal, the test set is only used for the final report.")

    metric_options = [m for m in ["MAE", "RMSE", "R2", "MAPE_%", "Train_time_s", "CV_MAE_mean"] if m in test_df.columns]
    c = st.columns(3)
    for col, metric in zip(c, metric_options[:3]):
        col.markdown(f"**{metric}** " + ("(higher is better)" if metric == "R2" else "(lower is better)"))
        col.bar_chart(test_df.set_index("Model")[metric])
    c = st.columns(3)
    for col, metric in zip(c, metric_options[3:6]):
        col.markdown(f"**{metric}** (lower is better)")
        col.bar_chart(test_df.set_index("Model")[metric])

    if val_df is not None:
        with st.expander("Validation-set results (used for model selection)"):
            st.dataframe(val_df)

    st.subheader("Model selector: actual vs predicted and feature importance")
    model_choice = st.selectbox("Model", list(test_df["Model"]), key="avp")
    left, right = st.columns(2)
    if preds_df is not None and model_choice in preds_df.columns:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.scatter(preds_df["price"], preds_df[model_choice], alpha=0.35, s=10)
        lim = [0, max(preds_df["price"].max(), preds_df[model_choice].max())]
        ax.plot(lim, lim, "r--"); ax.set_xlabel("Actual (LKR)"); ax.set_ylabel("Predicted (LKR)")
        ax.set_title("Actual vs predicted (test set)")
        left.pyplot(fig)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.hist(preds_df["price"] - preds_df[model_choice], bins=40, color="#55A868")
        ax.set_xlabel("Residual = actual - predicted (LKR)"); ax.set_title("Residuals")
        right.pyplot(fig)
    if imp_df is not None and model_choice in set(imp_df["Model"]):
        st.markdown("**Feature importance (built-in, summed per feature, %)**")
        st.bar_chart(imp_df[imp_df["Model"] == model_choice].set_index("feature")["importance_%"])
    elif pi_df is not None:
        st.caption("Built-in importance exists for the tree models only. Permutation importance of the "
                   f"validation-chosen model ({best_name}) is shown on the Model Insights page.")

    if seg_df is not None:
        st.subheader("Where the errors are: MAE by data source and condition")
        by = st.radio("Group by", ["source", "condition"], horizontal=True)
        s = seg_df[seg_df["segment_by"] == by]
        st.dataframe(s.pivot(index="Model", columns="segment", values="MAE").style.format("{:,.0f}"))
        st.caption("Rows per segment (test set): " + ", ".join(
            f"{k}: {v}" for k, v in s.groupby("segment")["rows"].first().items()))
    if gap_df is not None:
        st.subheader("Over-fitting check (train vs test)")
        st.dataframe(gap_df)
    if w_df is not None:
        st.subheader("Hybrid RF + ANN: weight experiment (validation set)")
        st.dataframe(w_df)
        st.line_chart(w_df.set_index("rf_weight")["val_MAE"])

# ----------------------------------------------------------------------------- Prediction + Fair Price Checker
elif page == "Price Prediction & Fair Price Checker":
    st.title("Predict a price - and check if an advert is fair")
    models = get_models()
    if not models:
        st.error("No trained models in models/. Run `python -m src.train_models` first.")
        st.stop()
    opt = PR.build_options(clean)

    c1, c2, c3 = st.columns(3)
    brand = c1.selectbox("Brand", opt["brands"])
    phone_model = c2.selectbox("Model", opt["models_by_brand"][brand])
    condition = c3.selectbox("Condition", opt["conditions"])
    c4, c5, c6 = st.columns(3)
    ram = c4.selectbox("RAM (GB)", ["Unknown"] + [int(r) for r in opt["ram"]])
    storage = c5.selectbox("Storage (GB)", ["Unknown"] + [int(s) for s in opt["storage"]])
    location = c6.selectbox("Location", opt["locations"])
    c7, c8 = st.columns(2)
    source = c7.selectbox("Data source (price type)", opt["sources"],
                          help="ikman = seller asking price, Dialog = retail catalogue price")
    choice = c8.selectbox("Prediction model", ["All models"] + list(models))

    st.markdown("**Fair Price Checker (optional)** - enter the price you saw in an advert:")
    d1, d2 = st.columns(2)
    asking = d1.number_input("Advertised price (Rs.) - leave 0 to skip", min_value=0, value=0, step=1000)
    tol = d2.slider("Fair-price tolerance (±%)", 5, 40, 15) / 100

    if st.button("Predict price", type="primary"):
        row = PR.make_input(brand, phone_model, condition,
                            np.nan if storage == "Unknown" else storage,
                            np.nan if ram == "Unknown" else ram, location, source)
        results = PR.predict_all(models, row)
        if choice == "All models":
            table = pd.DataFrame({"Predicted price (Rs.)": results})
            table.index.name = "Model"
            st.dataframe(table.style.format("{:,.0f}"))
            st.bar_chart(table)
            st.caption(f"Spread between models: {rs(max(results.values()) - min(results.values()))}. "
                       "A wide spread means the models disagree - treat the estimate with caution.")
            ref_name = best_name if best_name in results else list(results)[0]
        else:
            st.metric(choice, rs(results[choice]))
            ref_name = choice
        st.caption(f"Fair price used for the verdict: **{ref_name}** = {rs(results[ref_name])}")
        if asking > 0:
            verdict, diff = PR.deal_verdict(asking, results[ref_name], tol)
            verdict_box(verdict, diff, tol)
        st.warning("These are estimates of advertised / catalogue prices, not guaranteed selling prices.")
        with st.expander("Input sent to the models"):
            st.dataframe(row)

# ----------------------------------------------------------------------------- Deal Finder
elif page == "Deal Finder":
    st.title("Deal Finder - scan real listings")
    st.write("Every listing in the held-out **test set** (never seen by the models while training) is compared "
             "with the model's fair-price estimate. Listings priced well below the estimate are potential "
             "**good deals**; listings well above it may be **overpriced** (or have something the model cannot "
             "see: damage, accessories, warranty, a typo).")
    if preds_df is None:
        st.error("test_predictions.csv not found - run `python -m src.train_models` first.")
        st.stop()
    model_cols = [m for m in test_df["Model"] if m in preds_df.columns]
    default_idx = model_cols.index(best_name) if best_name in model_cols else 0
    c1, c2 = st.columns(2)
    model_col = c1.selectbox("Fair-price model", model_cols, index=default_idx)
    tol = c2.slider("Tolerance (±%)", 5, 40, 15, key="dealtol") / 100
    scan = PR.scan_listings(preds_df, model_col, tol)
    counts = scan["verdict"].value_counts().reindex(list(PR.VERDICTS)).fillna(0).astype(int)
    m = st.columns(3)
    for col, v in zip(m, PR.VERDICTS):
        col.metric(v, int(counts[v]), f"{counts[v] / len(scan) * 100:.1f}% of listings")
    st.bar_chart(counts)
    mape = test_df.loc[test_df["Model"] == model_col, "MAPE_%"]
    if len(mape):
        st.caption(f"This model's typical error on the test set is about {mape.iloc[0]:.0f}% (MAPE). "
                   "A tolerance smaller than the typical error will flag many listings by chance.")

    f1, f2, f3 = st.columns(3)
    pick = f1.multiselect("Verdict", list(PR.VERDICTS), default=["Good deal", "Overpriced"])
    src = f2.multiselect("Source", sorted(scan["source"].unique()), default=sorted(scan["source"].unique()))
    cond = f3.multiselect("Condition", sorted(scan["condition"].unique()), default=sorted(scan["condition"].unique()))
    view = scan[scan["verdict"].isin(pick) & scan["source"].isin(src) & scan["condition"].isin(cond)]
    view = view.sort_values("difference_%")
    cols = [c for c in ["title", "source", "condition", "location", "price", "fair_price", "difference_%",
                        "verdict", "ad_url"] if c in view.columns]
    st.dataframe(view[cols].head(200).style.format({"price": "{:,.0f}", "fair_price": "{:,.0f}",
                                                    "difference_%": "{:+.1f}"}))
    st.caption("Sorted from the biggest discount to the biggest premium. Always open the advert and check the "
               "phone's real condition before trusting a verdict.")

# ----------------------------------------------------------------------------- Insights
else:
    st.title("Model Insights")
    st.subheader("The approaches")
    a, b = st.columns(2)
    a.markdown("**Baseline - Ridge regression.** A linear model on the same features. If the ensembles cannot beat "
               "it, their extra complexity is not worth it.")
    b.markdown("**Bagging - Random Forest / Extra Trees.** Many trees trained independently on different samples "
               "(Extra Trees also uses random split points); predictions are averaged. Cuts **variance**.")
    a.markdown("**Boosting - XGBoost / LightGBM / CatBoost.** Trees built one after another, each correcting the "
               "previous errors. Mainly cuts **bias**; too many rounds can over-fit.")
    b.markdown("**Stacking.** RF, XGBoost, LightGBM and CatBoost are base models; a Ridge meta-model learns how "
               "to combine their out-of-fold predictions (5-fold).")
    a.markdown("**Hybrid - Random Forest + ANN.** A neural network (MLP) and a Random Forest combined as "
               "`w·RF + (1-w)·ANN`; `w` is chosen on the validation set.")
    b.markdown("**Innovation - Fair Price Checker.** `asking / predicted - 1` compared with a tolerance gives "
               "*good deal / fair price / overpriced*.")

    st.subheader("Why did the models behave this way?")
    interp = config.OUT_DIR / "interpretation.md"
    if interp.exists():
        st.markdown(interp.read_text())
    else:
        st.info("Write your interpretation in `outputs/interpretation.md` (see the last cell of notebook 07) "
                "and it will appear here.")
    for line in key_findings(test_df, val_df, gap_df):
        st.markdown(f"- {line}")

    if pi_df is not None:
        st.subheader(f"What drives the validation-chosen model's predictions? ({best_name})")
        st.caption("Permutation importance on the test set: how much MAE increases when a feature is shuffled.")
        st.bar_chart(pi_df.set_index("feature")["mae_increase"])

    st.subheader("Limitations")
    st.table(pd.DataFrame({
        "Limitation": ["Asking prices", "Mixed sources", "Extracted specs", "Small Dialog sample",
                       "Time", "Limited data", "Fair Price Checker"],
        "Explanation": [
            "ikman prices are what sellers ask, not what buyers pay.",
            "ikman asking prices and Dialog catalogue prices follow different patterns; `source` is a model feature.",
            "RAM/storage are parsed from ad text; some ads omit or mis-state them.",
            "Dialog-specific error estimates rest on few rows.",
            "Phone prices change over time and the data is a snapshot.",
            "About 5,000 listings cannot cover every phone model in Sri Lanka.",
            "The verdict only compares with the model's estimate; it cannot see damage, accessories or warranty."]}))
    st.caption("AI assistance: code was generated with Claude (Anthropic) and reviewed/adapted by the student - "
               "see AI_DISCLOSURE.md.")
