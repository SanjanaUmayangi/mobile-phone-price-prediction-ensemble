"""Shared preprocessing (imputation + encoding). Fitted INSIDE every model pipeline,
so validation / test rows never influence the encoders (no leakage)."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def build_preprocessor():
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),  # + "was missing" flags
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=config.MIN_CATEGORY_FREQ,
                                 sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", numeric, config.NUMERICAL), ("cat", categorical, config.CATEGORICAL)],
        sparse_threshold=0,
    )
