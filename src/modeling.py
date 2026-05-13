from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from .config import PipelineConfig
from .data import logical_type
from .runtime import detect_gpu_availability, log


@dataclass
class TrainedModel:
    name: str
    estimator: Any
    best_params: dict[str, Any]
    cv_score: float
    threshold: float
    validation_metrics: dict[str, float]
    training_rationale: str
    param_grid: list[dict[str, Any]] | dict[str, Any]
    feature_cols: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    validation_threshold_sweep: pd.DataFrame | None = None


def _is_xgboost_estimator(estimator: Any) -> bool:
    name = estimator.__class__.__name__.lower()
    module = estimator.__class__.__module__.lower()
    return "xgb" in name or "xgboost" in module


def _xgboost_candidates(estimator: Any) -> list[Any]:
    candidates: list[Any] = []
    if hasattr(estimator, "named_steps") and "model" in getattr(estimator, "named_steps", {}):
        candidates.append(estimator.named_steps["model"])
    candidates.append(estimator)
    return [obj for obj in candidates if _is_xgboost_estimator(obj)]


def set_xgboost_prediction_device(estimator: Any, device: str = "cpu") -> None:
    """Align fitted XGBoost boosters with CPU-resident sklearn inputs before prediction.

    The pipeline preprocesses tabular CSV data with scikit-learn, which produces CPU
    NumPy arrays. If an XGBoost booster remains on CUDA, recent XGBoost versions warn
    that prediction is falling back to a DMatrix because booster and input devices differ.
    Moving the fitted booster to CPU before CPU prediction avoids that warning and avoids
    the extra fallback path.
    """
    for obj in _xgboost_candidates(estimator):
        try:
            obj.set_params(device=device)
        except Exception:
            pass
        try:
            booster = obj.get_booster()
            booster.set_param({"device": device})
        except Exception:
            pass


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def infer_model_feature_types(df: pd.DataFrame, feature_cols: list[str], config: PipelineConfig) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        ltype = logical_type(df[col], config)
        if ltype in {"categorical", "binary", "numeric_discrete"}:
            categorical.append(col)
        else:
            numeric.append(col)
    return numeric, categorical


def build_preprocessor(train_df: pd.DataFrame, feature_cols: list[str], config: PipelineConfig) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric, categorical = infer_model_feature_types(train_df, feature_cols, config)
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", _one_hot_encoder())]), categorical))
    if not transformers:
        raise RuntimeError("No model predictors are available after feature typing.")
    return ColumnTransformer(transformers=transformers, remainder="drop"), numeric, categorical


def _grid_size(grid: dict[str, list[Any]]) -> int:
    n = 1
    for values in grid.values():
        n *= max(len(values), 1)
    return int(n)


def _truncate_grid(grid: dict[str, list[Any]], max_grid: int) -> dict[str, list[Any]]:
    if max_grid <= 0:
        return grid
    out = {k: list(v) for k, v in grid.items()}
    while _grid_size(out) > max_grid:
        longest = max(out, key=lambda k: len(out[k]))
        if len(out[longest]) <= 1:
            break
        # Keep first, middle and last values where possible rather than blindly removing the last value.
        values = out[longest]
        if len(values) > 2:
            out[longest] = values[: len(values) - 1]
        else:
            out[longest] = values[:1]
    return out


def model_specifications(config: PipelineConfig) -> dict[str, dict[str, Any]]:
    max_grid = int(config.max_grid_per_model or 4)
    specs: dict[str, dict[str, Any]] = {
        "LogisticRegression": {
            "estimator": LogisticRegression(max_iter=3000, solver="liblinear", l1_ratio=0.0, random_state=config.random_state),
            "grid": _truncate_grid({"model__C": [0.05, 0.1, 0.5, 1.0, 2.0, 5.0], "model__class_weight": [None, "balanced"]}, max_grid),
            "rationale": "Interpretable linear baseline. C controls regularisation strength; class_weight='balanced' is tested because biomedical endpoints are often imbalanced.",
        },
        "ElasticNetLogistic": {
            "estimator": LogisticRegression(max_iter=5000, solver="saga", l1_ratio=0.5, random_state=config.random_state),
            "grid": _truncate_grid({"model__C": [0.05, 0.1, 0.5, 1.0, 2.0], "model__l1_ratio": [0.15, 0.5, 0.85], "model__class_weight": [None, "balanced"]}, max_grid),
            "rationale": "Sparse/regularised linear model. l1_ratio controls sparsity versus ridge shrinkage; C controls regularisation strength.",
        },
        "RandomForest": {
            "estimator": RandomForestClassifier(random_state=config.random_state, n_jobs=config.n_jobs),
            "grid": _truncate_grid({"model__n_estimators": [80, 120], "model__max_depth": [3, 5, None], "model__min_samples_leaf": [5, 10], "model__class_weight": [None, "balanced"]}, max_grid),
            "rationale": "Non-linear ensemble robust to interactions. max_depth and min_samples_leaf control overfitting; class_weight is tested for event imbalance.",
        },
        "ExtraTrees": {
            "estimator": ExtraTreesClassifier(random_state=config.random_state, n_jobs=config.n_jobs),
            "grid": _truncate_grid({"model__n_estimators": [80, 120], "model__max_depth": [3, 5, None], "model__min_samples_leaf": [5, 10], "model__class_weight": [None, "balanced"]}, max_grid),
            "rationale": "High-randomness tree ensemble used as a robustness comparator. Tree depth and leaf size constrain variance.",
        },
        "HistGradientBoosting": {
            "estimator": HistGradientBoostingClassifier(random_state=config.random_state),
            "grid": _truncate_grid({"model__max_iter": [80, 140], "model__learning_rate": [0.03, 0.08], "model__max_leaf_nodes": [7, 15], "model__l2_regularization": [0.0, 0.1]}, max_grid),
            "rationale": "Gradient-boosted trees for non-linear risk patterns. learning_rate and max_iter trade bias and variance; l2_regularization controls complexity.",
        },
        "SVC_RBF": {
            "estimator": SVC(kernel="rbf", probability=True, random_state=config.random_state),
            "grid": _truncate_grid({"model__C": [0.1, 0.5, 1.0, 2.0], "model__gamma": ["scale", 0.05], "model__class_weight": [None, "balanced"]}, max_grid),
            "rationale": "Kernel classifier for non-linear boundaries. C controls margin softness; gamma controls local smoothness.",
        },
    }
    try:
        from xgboost import XGBClassifier
        gpu_info = detect_gpu_availability(getattr(config, "use_gpu", "auto"))
        xgb_kwargs = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": config.random_state,
            "n_jobs": config.n_jobs,
            "tree_method": "hist",
        }
        xgb_rationale_suffix = ""
        policy = str(getattr(config, "xgboost_device_policy", "safe_cpu") or "safe_cpu").lower()
        gpu_ready = bool(gpu_info.get("gpu_available")) and bool(gpu_info.get("gpu_requested", True))
        if policy in {"cuda", "gpu", "force_cuda", "cuda_if_available", "auto_cuda"} and gpu_ready:
            xgb_kwargs["device"] = "cuda"
            xgb_rationale_suffix = (
                f" GPU detected ({gpu_info.get('gpu_name')}) and xgboost_device_policy='{policy}', "
                "so CUDA is requested for fitting; the fitted booster is moved to CPU before sklearn CPU-array prediction to avoid device-mismatch warnings."
            )
        else:
            xgb_kwargs["device"] = "cpu"
            if gpu_ready:
                xgb_rationale_suffix = (
                    f" GPU detected ({gpu_info.get('gpu_name')}), but xgboost_device_policy='{policy}' keeps XGBoost on CPU because the sklearn preprocessing pipeline outputs CPU arrays. "
                    "This avoids the XGBoost CUDA/CPU DMatrix fallback warning during prediction."
                )
            else:
                xgb_rationale_suffix = " XGBoost is configured on CPU because no compatible GPU was requested/detected."
        specs["XGBoost"] = {
            "estimator": XGBClassifier(**xgb_kwargs),
            "grid": _truncate_grid({"model__n_estimators": [80, 140], "model__max_depth": [2, 3], "model__learning_rate": [0.03, 0.08], "model__subsample": [0.8, 1.0]}, max_grid),
            "rationale": "Boosted trees with regularised additive modelling. Tree depth, learning rate, and subsampling manage overfitting." + xgb_rationale_suffix,
        }
    except Exception:
        pass
    return specs


def describe_grid(grid: dict[str, list[Any]]) -> str:
    clean = {k.replace("model__", ""): v for k, v in grid.items()}
    return json.dumps(clean, default=str)


def _predict_proba_binary(estimator: Any, X: pd.DataFrame) -> np.ndarray:
    set_xgboost_prediction_device(estimator, "cpu")
    if hasattr(estimator, "predict_proba"):
        proba = np.asarray(estimator.predict_proba(X))
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1].astype(float)
        return proba.reshape(-1).astype(float)
    if hasattr(estimator, "decision_function"):
        raw = np.asarray(estimator.decision_function(X), dtype=float).reshape(-1)
        return 1.0 / (1.0 + np.exp(-raw))
    raise TypeError("Estimator does not expose predict_proba or decision_function.")


def select_threshold(y_true: np.ndarray, prob: np.ndarray, threshold_grid: tuple[float, ...]) -> tuple[float, dict[str, float]]:
    from .evaluation import binary_classification_metrics

    best_threshold = 0.5
    best_metrics: dict[str, float] = {}
    best_score = -np.inf
    for threshold in threshold_grid:
        metrics = binary_classification_metrics(y_true, prob, float(threshold))
        score = metrics["balanced_accuracy"] + 1e-4 * metrics["f1"] + 1e-6 * metrics["selection_rate"]
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def _safe_cv_folds(y: np.ndarray, requested: int) -> int:
    counts = pd.Series(y).value_counts()
    if counts.empty or counts.min() < 2:
        return 2
    return int(max(2, min(int(requested), int(counts.min()))))


def train_models(train_df: pd.DataFrame, validation_df: pd.DataFrame, config: PipelineConfig, feature_cols: list[str]) -> tuple[dict[str, TrainedModel], pd.DataFrame, pd.DataFrame]:
    y_train = train_df["target"].astype(int).to_numpy()
    y_val = validation_df["target"].astype(int).to_numpy()
    if len(np.unique(y_train)) < 2:
        raise RuntimeError("Training data contains one target class only. Cannot train binary classifiers.")
    X_train = train_df[feature_cols]
    X_val = validation_df[feature_cols]
    preprocessor, numeric_features, categorical_features = build_preprocessor(train_df, feature_cols, config)
    specs = model_specifications(config)
    cv_folds = _safe_cv_folds(y_train, int(config.cv_folds or 2))
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.random_state)

    trained: dict[str, TrainedModel] = {}
    config_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for model_name in config.enabled_models or ():
        if model_name not in specs:
            log(f"[Model training] {model_name}: skipped because the estimator is unavailable in this environment.")
            continue
        spec = specs[model_name]
        pipeline = Pipeline([("preprocess", preprocessor), ("model", spec["estimator"])])
        grid = spec["grid"]
        if config.show_model_training_explanations:
            log("\n" + "=" * 80)
            log(f"[Model training] {model_name}")
            log(f"Reason for inclusion: {spec['rationale']}")
            log(f"Training data: n={len(train_df)}, events={int(y_train.sum())}, event_rate={y_train.mean():.4f}")
            log(f"Feature typing: {len(numeric_features)} numeric, {len(categorical_features)} categorical/discrete predictors")
            log(f"Cross-validation: {cv_folds}-fold StratifiedKFold; scoring={config.scoring_metric}")
            log(f"Hyperparameter grid used: {describe_grid(grid)}")

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            scoring=config.scoring_metric,
            cv=cv,
            n_jobs=int(config.n_jobs or 1),
            refit=True,
            error_score="raise",
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*\'penalty\' was deprecated.*", category=FutureWarning)
            warnings.filterwarnings("ignore", message=r".*'multi_class' was deprecated.*", category=FutureWarning)
            search.fit(X_train, y_train)
        val_prob = _predict_proba_binary(search.best_estimator_, X_val)
        threshold, val_metrics = select_threshold(y_val, val_prob, config.threshold_grid)
        from .evaluation import threshold_sweep
        validation_sweep = threshold_sweep(y_val, val_prob, config.threshold_grid, model_name)
        validation_sweep["selection_score"] = (
            validation_sweep["balanced_accuracy"].astype(float)
            + 1e-4 * validation_sweep["f1"].astype(float)
            + 1e-6 * validation_sweep["selection_rate"].astype(float)
        )
        validation_sweep["selected_threshold"] = float(threshold)
        validation_sweep["selected"] = np.isclose(validation_sweep["threshold"].astype(float), float(threshold))
        validation_sweep["selection_rule"] = "max balanced accuracy; ties broken by F1 and selection rate"
        best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        if config.show_model_training_explanations:
            log(f"Selected parameters: {json.dumps(best_params, default=str)}")
            log("Selection explanation: the best mean cross-validated ROC-AUC configuration was retained; the validation threshold was then chosen to maximise balanced accuracy.")
            log(f"Validation threshold={threshold:.4f}; balanced_accuracy={val_metrics['balanced_accuracy']:.4f}; roc_auc={val_metrics['roc_auc']:.4f}")

        trained[model_name] = TrainedModel(
            name=model_name,
            estimator=search.best_estimator_,
            best_params=best_params,
            cv_score=float(search.best_score_),
            threshold=float(threshold),
            validation_metrics=val_metrics,
            training_rationale=spec["rationale"],
            param_grid=grid,
            feature_cols=list(feature_cols),
            numeric_features=list(numeric_features),
            categorical_features=list(categorical_features),
            validation_threshold_sweep=validation_sweep,
        )
        config_rows.append({
            "model": model_name,
            "included": True,
            "preprocessing": "median imputation + standardisation for numeric predictors; most-frequent imputation + one-hot encoding for categorical/discrete predictors",
            "training_set_n": int(len(train_df)),
            "training_events": int(y_train.sum()),
            "feature_count": int(len(feature_cols)),
            "numeric_feature_count": int(len(numeric_features)),
            "categorical_feature_count": int(len(categorical_features)),
            "cv_strategy": f"{cv_folds}-fold StratifiedKFold",
            "selection_metric": config.scoring_metric,
            "hyperparameter_grid": describe_grid(grid),
            "selected_params": json.dumps(best_params, default=str),
            "why_this_grid": spec["rationale"],
            "cv_score": float(search.best_score_),
            "validation_threshold": float(threshold),
        })
        validation_rows.append({"model": model_name, "cv_score": float(search.best_score_), "threshold": float(threshold), **val_metrics})

    if not trained:
        raise RuntimeError("No model was trained. Check USER_CONFIG['models']['enabled_models'].")
    return trained, pd.DataFrame(config_rows), pd.DataFrame(validation_rows).sort_values("balanced_accuracy", ascending=False).reset_index(drop=True)




def generate_model_training_candidates(train_df: pd.DataFrame, feature_cols: list[str], config: PipelineConfig, include_unavailable: bool = False) -> pd.DataFrame:
    """Create a tabular plan of trainable model+hyperparameter candidates.

    The output is notebook-friendly and can be iterated to train specific
    configurations or used to inspect the search space before GridSearchCV.
    """
    from sklearn.model_selection import ParameterGrid

    preprocessor, numeric_features, categorical_features = build_preprocessor(train_df, feature_cols, config)
    specs = model_specifications(config)
    rows: list[dict[str, Any]] = []
    for model_name in config.enabled_models or ():
        spec = specs.get(model_name)
        if spec is None:
            if include_unavailable:
                rows.append({
                    "model": model_name,
                    "candidate_id": None,
                    "available": False,
                    "params": None,
                    "pipeline": None,
                    "rationale": "Estimator unavailable in this environment.",
                })
            continue
        pipeline = Pipeline([("preprocess", preprocessor), ("model", spec["estimator"])])
        grid = spec["grid"]
        for i, params in enumerate(ParameterGrid(grid), start=1):
            clean_params = {k.replace("model__", ""): v for k, v in params.items()}
            rows.append({
                "model": model_name,
                "candidate_id": f"{model_name}__{i}",
                "available": True,
                "params": clean_params,
                "pipeline": pipeline.set_params(**params),
                "rationale": spec["rationale"],
                "numeric_features": list(numeric_features),
                "categorical_features": list(categorical_features),
            })
    return pd.DataFrame(rows)

def predict_model(trained_model: TrainedModel, df: pd.DataFrame, config: PipelineConfig | None = None) -> np.ndarray:
    return _predict_proba_binary(trained_model.estimator, df[trained_model.feature_cols])
