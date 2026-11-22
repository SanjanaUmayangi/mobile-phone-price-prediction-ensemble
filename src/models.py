"""All models of the project. Every model takes the RAW feature dataframe and returns prices in LKR.

  ridge          baseline (linear model)
  random_forest  BAGGING
  extra_trees    BAGGING (randomised trees, averaged)
  xgboost        BOOSTING
  lightgbm       BOOSTING
  catboost       BOOSTING
  stacking       STACKING  (RF + XGBoost + LightGBM + CatBoost -> Ridge meta-model)
  hybrid         HYBRID    (RF + ANN weighted average)

Every model learns log(1 + price) internally, because phone prices are strongly right-skewed.
LightGBM and CatBoost are optional: if they are not installed they are skipped (with a warning)
and everything else still runs. Install them with:  pip install lightgbm catboost
"""
import importlib.util
import json
import warnings

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .features import build_preprocessor

try:
    from xgboost import XGBRegressor
except ImportError as e:  # pragma: no cover
    raise ImportError("xgboost is required: pip install xgboost") from e

PARAMS_FILE = config.OUT_DIR / "best_params.json"


def has_lightgbm():
    return importlib.util.find_spec("lightgbm") is not None


def has_catboost():
    return importlib.util.find_spec("catboost") is not None


# ----------------------------------------------------------------------------
# Saved hyper-parameters (from tuning)
# ----------------------------------------------------------------------------
def load_params():
    """Hyper-parameters found by tuning. Empty dict if not tuned yet."""
    if PARAMS_FILE.exists():
        return json.loads(PARAMS_FILE.read_text())
    return {}


def save_params(name, params):
    allp = load_params()
    allp[name] = params
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    PARAMS_FILE.write_text(json.dumps(allp, indent=2))


# ----------------------------------------------------------------------------
# Thin scikit-learn adapters for LightGBM / CatBoost (so they work inside Pipelines
# and StackingRegressor whatever the library version)
# ----------------------------------------------------------------------------
class LightGBMRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, n_estimators=400, learning_rate=0.05, num_leaves=31, max_depth=-1,
                 min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
                 random_state=config.RANDOM_STATE, n_jobs=-1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        from lightgbm import LGBMRegressor
        self.model_ = LGBMRegressor(
            n_estimators=self.n_estimators, learning_rate=self.learning_rate, num_leaves=self.num_leaves,
            max_depth=self.max_depth, min_child_samples=self.min_child_samples, subsample=self.subsample,
            subsample_freq=1, colsample_bytree=self.colsample_bytree, random_state=self.random_state,
            n_jobs=self.n_jobs, verbose=-1)
        self.model_.fit(np.asarray(X), np.asarray(y))
        self.feature_importances_ = np.asarray(self.model_.feature_importances_, dtype=float)
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X))


class CatBoostRegressorAdapter(RegressorMixin, BaseEstimator):
    def __init__(self, iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
                 random_state=config.RANDOM_STATE, thread_count=-1):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_state = random_state
        self.thread_count = thread_count

    def fit(self, X, y):
        from catboost import CatBoostRegressor
        self.model_ = CatBoostRegressor(
            iterations=self.iterations, learning_rate=self.learning_rate, depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg, random_seed=self.random_state, thread_count=self.thread_count,
            verbose=0, allow_writing_files=False)
        self.model_.fit(np.asarray(X), np.asarray(y))
        self.feature_importances_ = np.asarray(self.model_.feature_importances_, dtype=float)
        return self

    def predict(self, X):
        return np.asarray(self.model_.predict(np.asarray(X)), dtype=float)


# ----------------------------------------------------------------------------
# Base learners (without preprocessing)
# ----------------------------------------------------------------------------
def _p(name, defaults, override):
    p = dict(defaults)
    p.update(load_params().get(name, {}))
    p.update(override)
    return p


def rf_regressor(**override):
    return RandomForestRegressor(**_p("random_forest", dict(
        n_estimators=200, max_depth=20, min_samples_split=2, min_samples_leaf=2, max_features=1.0,
        random_state=config.RANDOM_STATE, n_jobs=-1), override))


def et_regressor(**override):
    return ExtraTreesRegressor(**_p("extra_trees", dict(
        n_estimators=300, max_depth=None, min_samples_split=2, min_samples_leaf=2, max_features=1.0,
        random_state=config.RANDOM_STATE, n_jobs=-1), override))


def xgb_regressor(**override):
    return XGBRegressor(**_p("xgboost", dict(
        n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", random_state=config.RANDOM_STATE, n_jobs=-1), override))


def lgbm_regressor(**override):
    return LightGBMRegressor(**_p("lightgbm", {}, override))


def cat_regressor(**override):
    return CatBoostRegressorAdapter(**_p("catboost", {}, override))


def ann_regressor(**override):
    return MLPRegressor(**_p("ann", dict(
        hidden_layer_sizes=(64, 32), activation="relu", solver="adam", learning_rate_init=0.001,
        alpha=1e-3, max_iter=800, early_stopping=True, n_iter_no_change=20,
        random_state=config.RANDOM_STATE), override))


def with_prep(model):
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def log_target(estimator, standardise=False):
    """Train on log1p(price) and return prices in LKR."""
    if standardise:
        return _LogStdTarget(estimator)
    return TransformedTargetRegressor(regressor=estimator, func=np.log1p, inverse_func=np.expm1,
                                      check_inverse=False)


class _LogStdTarget(RegressorMixin, BaseEstimator):
    """log1p then standardise the target (neural networks train better on a ~N(0,1) target)."""

    def __init__(self, regressor):
        self.regressor = regressor

    def fit(self, X, y):
        z = np.log1p(np.asarray(y, dtype=float))
        self.scaler_ = StandardScaler().fit(z.reshape(-1, 1))
        self.regressor_ = clone(self.regressor).fit(X, self.scaler_.transform(z.reshape(-1, 1)).ravel())
        return self

    def predict(self, X):
        z = self.regressor_.predict(X).reshape(-1, 1)
        return np.expm1(self.scaler_.inverse_transform(z).ravel())


# ----------------------------------------------------------------------------
# The models
# ----------------------------------------------------------------------------
def make_ridge():
    """Baseline: linear model on the same encoded features."""
    return log_target(with_prep(RidgeCV(alphas=np.logspace(-2, 3, 12))))


def make_random_forest(**override):
    return log_target(with_prep(rf_regressor(**override)))


def make_extra_trees(**override):
    return log_target(with_prep(et_regressor(**override)))


def make_xgboost(**override):
    return log_target(with_prep(xgb_regressor(**override)))


def make_lightgbm(**override):
    return log_target(with_prep(lgbm_regressor(**override)))


def make_catboost(**override):
    return log_target(with_prep(cat_regressor(**override)))


def stacking_base_names():
    names = ["random_forest", "xgboost"]
    if has_lightgbm():
        names.append("lightgbm")
    if has_catboost():
        names.append("catboost")
    return names


def make_stacking():
    """Base models (RF, XGBoost, LightGBM, CatBoost) -> Ridge meta-model on out-of-fold predictions (cv=5)."""
    builders = {"random_forest": rf_regressor, "xgboost": xgb_regressor,
                "lightgbm": lgbm_regressor, "catboost": cat_regressor}
    stack = StackingRegressor(
        estimators=[(n, with_prep(builders[n]())) for n in stacking_base_names()],
        final_estimator=RidgeCV(alphas=np.logspace(-3, 2, 12)),
        cv=5, n_jobs=1,
    )
    return log_target(stack)


class HybridRFANN(RegressorMixin, BaseEstimator):
    """P_hybrid = w * P_RF + (1 - w) * P_ANN  (w chosen on the validation set)."""

    def __init__(self, rf_weight=0.5):
        self.rf_weight = rf_weight

    def fit(self, X, y):
        self.rf_ = make_random_forest().fit(X, y)
        self.ann_ = log_target(with_prep(ann_regressor()), standardise=True).fit(X, y)
        return self

    def predict_components(self, X):
        return self.rf_.predict(X), self.ann_.predict(X)

    def predict(self, X):
        rf, ann = self.predict_components(X)
        return self.rf_weight * rf + (1 - self.rf_weight) * ann


def make_hybrid(rf_weight=0.5):
    return HybridRFANN(rf_weight=rf_weight)


def make_ann():
    """ANN on its own (used only to explain the hybrid)."""
    return log_target(with_prep(ann_regressor()), standardise=True)


_ALL = {
    "ridge": make_ridge,
    "random_forest": make_random_forest,
    "extra_trees": make_extra_trees,
    "xgboost": make_xgboost,
    "lightgbm": make_lightgbm,
    "catboost": make_catboost,
    "stacking": make_stacking,
    "hybrid": make_hybrid,
}


_WARNED = set()


def available_factories():
    """Ordered {key: factory} for the models whose libraries are installed."""
    out = {}
    for k, f in _ALL.items():
        if k == "lightgbm" and not has_lightgbm():
            if k not in _WARNED:
                _WARNED.add(k)
                warnings.warn("lightgbm is not installed - LightGBM skipped (pip install lightgbm)")
            continue
        if k == "catboost" and not has_catboost():
            if k not in _WARNED:
                _WARNED.add(k)
                warnings.warn("catboost is not installed - CatBoost skipped (pip install catboost)")
            continue
        out[k] = f
    return out


# kept for older notebooks / scripts
FACTORIES = _ALL


# ----------------------------------------------------------------------------
# Hyper-parameter tuning (train data only, 3-fold CV, MAE)
# ----------------------------------------------------------------------------
SEARCH_SPACES = {
    "random_forest": {
        "n_estimators": [100, 200, 300], "max_depth": [10, 15, 20, 30, None],
        "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4], "max_features": [0.5, 0.8, 1.0]},
    "extra_trees": {
        "n_estimators": [200, 300, 500], "max_depth": [15, 25, None],
        "min_samples_leaf": [1, 2, 4], "max_features": [0.5, 0.8, 1.0]},
    "xgboost": {
        "n_estimators": [200, 300, 500], "learning_rate": [0.02, 0.05, 0.1], "max_depth": [3, 4, 5, 6, 8],
        "subsample": [0.7, 0.8, 1.0], "colsample_bytree": [0.6, 0.8, 1.0], "min_child_weight": [1, 3, 5]},
    "lightgbm": {
        "n_estimators": [200, 400, 600], "learning_rate": [0.02, 0.05, 0.1], "num_leaves": [15, 31, 63],
        "min_child_samples": [5, 10, 20], "subsample": [0.7, 0.8, 1.0], "colsample_bytree": [0.6, 0.8, 1.0]},
    "catboost": {
        "iterations": [300, 500, 800], "learning_rate": [0.03, 0.05, 0.1], "depth": [4, 6, 8],
        "l2_leaf_reg": [1.0, 3.0, 5.0, 9.0]},
}


def tune(name, X_train, y_train, n_iter=12, cv=3):
    """RandomizedSearchCV on the training set only. Saves best params to outputs/best_params.json."""
    est = _ALL[name]()
    grid = {f"regressor__model__{k}": v for k, v in SEARCH_SPACES[name].items()}
    search = RandomizedSearchCV(est, grid, n_iter=n_iter, cv=cv, scoring="neg_mean_absolute_error",
                                random_state=config.RANDOM_STATE, n_jobs=1, refit=False)
    search.fit(X_train, y_train)
    best = {k.replace("regressor__model__", ""): v for k, v in search.best_params_.items()}
    best = {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v)
            for k, v in best.items()}
    save_params(name, best)
    return best, -search.best_score_, search
