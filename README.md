# Sri Lankan Mobile Phone Price Prediction: Ensemble Learning Dashboard

This project predicts the price (in LKR) of a mobile phone sold in Sri Lanka from its brand, model, condition, RAM, storage, location and data source. It trains and compares **8 models** on the same data split, shows them in a **Streamlit dashboard**, and adds one extra idea: a **Fair Price Checker** that tells you if an advert looks like a *good deal*, a *fair price* or *overpriced*.

## Key Features

* Cleaning pipeline for ikman.lk and Dialog phone prices (RAM and storage extracted from advert text)
* 8 models on the same 70 / 15 / 15 split: Ridge, Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost, Stacking, Hybrid (RF + ANN)
* Fair comparison: MAE, RMSE, R², MAPE, training time and 5-fold cross-validation
* Over-fitting check, errors by source and condition, feature and permutation importance
* Interactive Streamlit dashboard with a model selector and a **Fair Price Checker**

## Models

| Family | Models |
|---|---|
| Baseline | Ridge regression |
| Bagging | Random Forest, Extra Trees |
| Boosting | XGBoost, LightGBM, CatBoost |
| Stacking | RF + XGBoost + LightGBM + CatBoost, combined by a Ridge meta-model (5-fold out-of-fold predictions) |
| Hybrid | Random Forest + ANN (MLP): `w * RF + (1 - w) * ANN`, where `w` is chosen on the validation set |



## Dataset

The merged file `data/mobile_phones_raw.csv` has 5,000+ rows before cleaning. It contains:

* **ikman.lk** adverts: the price the seller asks (new and used phones).
* **Dialog** catalogue prices: retail prices of brand-new phones (under 1% of the data).

I started from the CSV in the reference repository https://github.com/mskanishkaudayanga/ML-Algorithems-model (used only as a starting point) and added the ikman.lk and Dialog data. 

After cleaning and preprocessing, the data was split into 70% training, 15% validation, and 15% testing sets. The target is an *asking or catalogue price*, not a confirmed sale price.

**Columns used:** brand, model, condition, location, source (categorical); storage_gb, ram_gb (numerical). Target: `price`.




## Dashboard

`streamlit run app/streamlit_app.py` opens five pages:

1. **Overview**: the data, the model families and the written interpretation.
2. **Model Comparison**: the table with training time and 5-fold CV score, the best model marked with a trophy, and charts.
3. **Price Prediction & Fair Price Checker**: choose any of the 8 models, enter a phone and an asking price, and get a prediction and a verdict.
4. **Deal Finder**: upload or scan many listings and find good deals and overpriced ones.
5. **Model Insights**: feature importance, permutation importance, key findings and limitations.



## Project Structure

Mobile Phone Price Prediction Ensemble/
│
├── data/
│   ├── mobile_phones_raw.csv          # Original dataset (unchanged)
│   ├── mobile_phones_clean.csv        # Cleaned dataset
│   └── splits/                        # Training, validation and test data
│
├── notebooks/
│   ├── 01_data_understanding.ipynb    # Data understanding and EDA
│   ├── 02_data_preprocessing.ipynb    # Data preprocessing and splitting
│   ├── 03_baseline_and_bagging.ipynb  # Baseline models and bagging
│   ├── 04_boosting.ipynb              # Boosting models
│   ├── 05_stacking.ipynb              # Stacking ensemble
│   ├── 06_hybrid_rf_ann.ipynb         # Random Forest + ANN hybrid
│   ├── 07_model_comparison.ipynb      # Final model comparison
│   ├── 08_fair_price_checker.ipynb    # Fair Price Checker
│   └── 09_research_discussion.ipynb   # Research discussion and conclusions
│
├── src/
│   ├── config.py                      # Project configuration
│   ├── preprocess.py                  # Data preprocessing functions
│   ├── features.py                    # Feature engineering
│   ├── models.py                      # Model definitions
│   ├── train_models.py                # Model training pipeline
│   ├── evaluate.py                    # Model evaluation
│   ├── explain.py                     # Model interpretation
│   └── predict.py                     # Prediction functions
│
├── models/                            # Saved trained models (.pkl)
│
├── outputs/
│   ├── metrics/                       # Additional evaluation metrics (CSV)
│   └── plots/                         # Generated charts and visualisations (PNG)
│
├── app/
│   └── streamlit_app.py               # Streamlit dashboard
│
├── README.md                          # Project documentation
└── .gitignore                         # Git ignore rules





## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv venv
venv\Scripts\activate            # Windows PowerShell: .\venv\Scripts\Activate.ps1
                                 # macOS / Linux: source venv/bin/activate
pip install -r requirements.txt
```


## How to run

```bash
python -m src.train_models --rebuild   # clean, split, train 8 models, 5-fold CV, evaluate, save everything
streamlit run app/streamlit_app.py     # start the dashboard
jupyter notebook                       # open notebooks 01 to 09, run them in order
```

## Technologies Used

Python 3.10+, pandas, NumPy, scikit-learn, XGBoost, LightGBM, CatBoost, Streamlit, joblib, Matplotlib, Seaborn, SHAP (optional), Jupyter
