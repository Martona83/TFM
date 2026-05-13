from __future__ import annotations

from typing import Any
from contextlib import contextmanager
import warnings
import time

from .config import DEFAULT_MITIGATION_METHODS
from .modeling import set_xgboost_prediction_device

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

@contextmanager
def _suppress_known_fairlearn_warnings():
    """Suppress known non-actionable Fairlearn/Pandas dtype FutureWarnings.

    Recent pandas versions can warn when Fairlearn's post-processing internals
    assign interpolated float probabilities into arrays created with a narrower
    dtype. The values are valid, but the warning can flood notebook output.
    This context suppresses only that Fairlearn post-processing FutureWarning.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*Setting an item of incompatible dtype.*",
            category=FutureWarning,
            module=r"fairlearn\.postprocessing.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*has dtype incompatible with.*",
            category=FutureWarning,
            module=r"fairlearn\.postprocessing.*",
        )
        yield


class _Float64ProbabilityEstimator:
    """Estimator wrapper that forces probability outputs to float64.

    Fairlearn ThresholdOptimizer may emit pandas dtype warnings when the base
    estimator returns float32 probabilities. This wrapper keeps the original
    fitted estimator behavior but returns float64 probability arrays.
    """
    def __init__(self, estimator: Any):
        self.estimator = estimator

    def predict_proba(self, X):
        return np.asarray(self.estimator.predict_proba(X), dtype=np.float64)

    def predict(self, X):
        return np.asarray(self.estimator.predict(X))

    def decision_function(self, X):
        if hasattr(self.estimator, "decision_function"):
            return np.asarray(self.estimator.decision_function(X), dtype=np.float64)
        proba = self.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            p = np.clip(proba[:, 1], 1e-12, 1 - 1e-12)
        else:
            p = np.clip(proba.reshape(-1), 1e-12, 1 - 1e-12)
        return np.log(p / (1.0 - p))

    def __getattr__(self, name: str):
        return getattr(self.estimator, name)



def _mitigation_progress(config: Any, message: str, level: str = "standard") -> None:
    """Print visible progress during mitigation/retraining loops.

    Long pre-processing and intra-processing mitigations can refit models and may
    otherwise look like a stalled notebook cell. This helper keeps progress visible
    without depending on notebook widgets, so it works in local Jupyter, Colab,
    Kaggle and terminal execution.
    """
    if not bool(getattr(config, "show_mitigation_progress", True)):
        return
    detail = str(getattr(config, "mitigation_progress_detail", "standard") or "standard").lower()
    if detail in {"none", "off", "false", "0"}:
        return
    if level == "verbose" and detail not in {"verbose", "debug"}:
        return
    stamp = time.strftime("%H:%M:%S")
    print(f"[Mitigation {stamp}] {message}", flush=True)


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else np.nan


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (prob >= lo) & (prob < hi if hi < 1 else prob <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(float(y_true[mask].mean()) - float(prob[mask].mean()))
    return float(ece)


def confusion_rates(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "tpr": safe_div(tp, tp + fn),
        "fnr": safe_div(fn, tp + fn),
        "tnr": safe_div(tn, tn + fp),
        "fpr": safe_div(fp, tn + fp),
        "selection_rate": safe_div(tp + fp, len(y_true)),
    }


def binary_classification_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float, y_pred: np.ndarray | None = None) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).astype(float)
    if y_pred is None:
        y_pred = (prob >= threshold).astype(int)
    else:
        y_pred = np.asarray(y_pred).astype(int)
    rates = confusion_rates(y_true, y_pred)
    try:
        roc_auc = float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) == 2 else np.nan
    except Exception:
        roc_auc = np.nan
    try:
        pr_auc = float(average_precision_score(y_true, prob)) if len(np.unique(y_true)) == 2 else np.nan
    except Exception:
        pr_auc = np.nan
    try:
        brier = float(brier_score_loss(y_true, prob))
    except Exception:
        brier = np.nan
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "average_precision": pr_auc,
        "brier": brier,
        "ece": expected_calibration_error(y_true, prob),
        **rates,
    }


def evaluate_all_models(models: dict[str, Any], test_df: pd.DataFrame, config) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    from .modeling import predict_model

    rows: list[dict[str, Any]] = []
    probs: dict[str, np.ndarray] = {}
    y_true = test_df["target"].astype(int).to_numpy()
    for name, model in models.items():
        prob = predict_model(model, test_df, config)
        probs[name] = prob
        rows.append({"model": name, **binary_classification_metrics(y_true, prob, model.threshold)})
    return pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False).reset_index(drop=True), probs


def threshold_sweep(y_true: np.ndarray, prob: np.ndarray, thresholds: tuple[float, ...], model_name: str, selected_threshold: float | None = None) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        metrics = binary_classification_metrics(y_true, prob, float(threshold))
        objective = metrics["balanced_accuracy"] + 1e-4 * metrics["f1"] + 1e-6 * metrics["selection_rate"]
        row = {"model": model_name, "threshold_selection_score": objective, **metrics}
        if selected_threshold is not None:
            row["selected_threshold"] = float(selected_threshold)
            row["is_selected_threshold"] = bool(abs(float(threshold) - float(selected_threshold)) < 1e-12)
        rows.append(row)
    return pd.DataFrame(rows)


def all_threshold_sweeps(models: dict[str, Any], df: pd.DataFrame, probs: dict[str, np.ndarray], config) -> pd.DataFrame:
    y = df["target"].astype(int).to_numpy()
    frames = []
    for name, model in models.items():
        frames.append(threshold_sweep(y, probs[name], config.threshold_grid, name, selected_threshold=model.threshold))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fairness_by_group(df: pd.DataFrame, y_prob: np.ndarray, threshold: float, sensitive_attrs: tuple[str, ...], model_name: str, prediction_label: str = "baseline") -> pd.DataFrame:
    y_true = df["target"].astype(int).to_numpy()
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    rows: list[dict[str, Any]] = []
    for attr in sensitive_attrs:
        if attr not in df.columns:
            continue
        for group, idx in df.groupby(attr, dropna=False).groups.items():
            idx = np.asarray(list(idx), dtype=int)
            rates = confusion_rates(y_true[idx], y_pred[idx])
            rows.append({
                "model": model_name,
                "prediction_type": prediction_label,
                "attribute": attr,
                "group": group,
                "n": int(len(idx)),
                "events": int(y_true[idx].sum()),
                "non_events": int(len(idx) - y_true[idx].sum()),
                "accuracy": float(accuracy_score(y_true[idx], y_pred[idx])) if len(idx) else np.nan,
                **rates,
            })
    return pd.DataFrame(rows)


def fairness_gap_summary(group_df: pd.DataFrame) -> pd.DataFrame:
    if group_df.empty:
        return pd.DataFrame()
    metric_cols = [m for m in ["selection_rate", "tpr", "fnr", "fpr", "accuracy"] if m in group_df.columns]
    rows: list[dict[str, Any]] = []
    for (model, prediction_type, attr), sub in group_df.groupby(["model", "prediction_type", "attribute"]):
        row = {"model": model, "prediction_type": prediction_type, "attribute": attr}
        for metric in metric_cols:
            values = sub[metric].astype(float)
            row[f"{metric}_max"] = float(values.max())
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_gap"] = float(values.max() - values.min())
        row["combined_fpr_fnr_gap"] = float(row.get("fpr_gap", 0.0) + row.get("fnr_gap", 0.0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model", "combined_fpr_fnr_gap"], ascending=[True, False]).reset_index(drop=True)


def select_primary_attribute(gap_df: pd.DataFrame, champion_model: str) -> str:
    sub = gap_df.loc[(gap_df["model"] == champion_model) & (gap_df["prediction_type"] == "baseline")].copy()
    if sub.empty:
        sub = gap_df.loc[gap_df["prediction_type"] == "baseline"].copy()
    if sub.empty:
        raise ValueError("Cannot select a primary fairness attribute from an empty gap table.")
    return str(sub.sort_values("combined_fpr_fnr_gap", ascending=False).iloc[0]["attribute"])


def _score_group_threshold(rates: dict[str, float], base_rates: dict[str, float], objective: str) -> float:
    objective = str(objective or "equalized_odds").lower()
    if objective in {"equalized_odds", "eq_odds"}:
        return abs(rates["fpr"] - base_rates["fpr"]) + abs(rates["fnr"] - base_rates["fnr"]) + 0.02 * abs(rates["selection_rate"] - base_rates["selection_rate"])
    if objective in {"equal_opportunity", "tpr_parity", "fnr_parity"}:
        return abs(rates["fnr"] - base_rates["fnr"]) + 0.02 * abs(rates["fpr"] - base_rates["fpr"])
    if objective in {"demographic_parity", "selection_rate_parity"}:
        return abs(rates["selection_rate"] - base_rates["selection_rate"]) + 0.02 * (abs(rates["fpr"] - base_rates["fpr"]) + abs(rates["fnr"] - base_rates["fnr"]))
    if objective in {"balanced_accuracy_constrained", "balanced_accuracy"}:
        return -float((rates["tpr"] + rates["tnr"]) / 2.0) + 0.05 * (abs(rates["fpr"] - base_rates["fpr"]) + abs(rates["fnr"] - base_rates["fnr"]))
    return abs(rates["fpr"] - base_rates["fpr"]) + abs(rates["fnr"] - base_rates["fnr"])


def group_thresholds_from_validation(validation_df: pd.DataFrame, y_prob: np.ndarray, overall_threshold: float, attribute: str, threshold_grid: tuple[float, ...], objective: str = "equalized_odds", min_group_size: int = 20) -> dict[Any, float]:
    y_true = validation_df["target"].astype(int).to_numpy()
    base_pred = (np.asarray(y_prob) >= overall_threshold).astype(int)
    base_rates = confusion_rates(y_true, base_pred)
    thresholds: dict[Any, float] = {}
    for group, idx in validation_df.groupby(attribute, dropna=False).groups.items():
        idx = np.asarray(list(idx), dtype=int)
        best_threshold = float(overall_threshold)
        best_score = np.inf
        if len(idx) < int(min_group_size or 0) or len(np.unique(y_true[idx])) < 2:
            thresholds[group] = best_threshold
            continue
        for threshold in threshold_grid:
            pred = (y_prob[idx] >= threshold).astype(int)
            rates = confusion_rates(y_true[idx], pred)
            score = _score_group_threshold(rates, base_rates, objective)
            if score < best_score:
                best_score = score
                best_threshold = float(threshold)
        thresholds[group] = best_threshold
    return thresholds


def apply_group_thresholds(df: pd.DataFrame, y_prob: np.ndarray, attribute: str, thresholds: dict[Any, float], fallback: float) -> np.ndarray:
    pred = np.zeros(len(df), dtype=int)
    values = df[attribute].to_numpy()
    for i, group in enumerate(values):
        threshold = thresholds.get(group, fallback)
        pred[i] = int(y_prob[i] >= threshold)
    return pred


def _fairness_by_group_from_predictions(df: pd.DataFrame, y_pred: np.ndarray, attribute: str, model_name: str, prediction_label: str) -> pd.DataFrame:
    y_true = df["target"].astype(int).to_numpy()
    rows: list[dict[str, Any]] = []
    for group, idx in df.groupby(attribute, dropna=False).groups.items():
        idx = np.asarray(list(idx), dtype=int)
        rates = confusion_rates(y_true[idx], y_pred[idx])
        rows.append({
            "model": model_name,
            "prediction_type": prediction_label,
            "attribute": attribute,
            "group": group,
            "n": int(len(idx)),
            "events": int(y_true[idx].sum()),
            "non_events": int(len(idx) - y_true[idx].sum()),
            "accuracy": float(accuracy_score(y_true[idx], y_pred[idx])) if len(idx) else np.nan,
            **rates,
        })
    return pd.DataFrame(rows)


def exact_mcnemar_p(before_error: np.ndarray, after_error: np.ndarray) -> tuple[float, int, int]:
    before_error = np.asarray(before_error).astype(bool)
    after_error = np.asarray(after_error).astype(bool)
    b = int(np.sum(before_error & ~after_error))
    c = int(np.sum(~before_error & after_error))
    discordant = b + c
    if discordant == 0:
        return 1.0, b, c
    try:
        p = float(binomtest(min(b, c), discordant, p=0.5, alternative="two-sided").pvalue)
    except Exception:
        p = 1.0
    return p, b, c


def _gap_value_for_indices(y_true: np.ndarray, pred: np.ndarray, attr_values: np.ndarray, indices: np.ndarray, metric: str) -> float:
    values: list[float] = []
    metric_l = str(metric).lower()
    for group in pd.unique(attr_values[indices]):
        mask = indices[attr_values[indices] == group]
        if len(mask) == 0:
            continue
        rates = confusion_rates(y_true[mask], pred[mask])
        if metric_l in {"combined", "combined_fpr_fnr", "combined_gap"}:
            val = rates.get("fpr", np.nan) + rates.get("fnr", np.nan)
        else:
            val = rates.get(metric_l, np.nan)
        if not np.isnan(val):
            values.append(float(val))
    if not values:
        return np.nan
    return float(np.nanmax(values) - np.nanmin(values))


def bootstrap_gap_change(df: pd.DataFrame, base_pred: np.ndarray, mitigated_pred: np.ndarray, attribute: str, metric: str, reps: int, seed: int) -> dict[str, float]:
    """Paired bootstrap for subgroup-gap change after mitigation.

    The returned delta is after-minus-before. For gap metrics, negative values mean
    the mitigation reduced disparity. The one-sided improvement probability is the
    bootstrap probability that the delta is <= 0.
    """
    rng = np.random.default_rng(seed)
    y_true = df["target"].astype(int).to_numpy()
    attr_values = df[attribute].to_numpy()
    n = len(df)
    metric_l = str(metric).lower()
    prefix = "combined_fpr_fnr" if metric_l in {"combined", "combined_gap", "combined_fpr_fnr"} else metric_l

    deltas: list[float] = []
    for _ in range(int(reps)):
        idx = rng.integers(0, n, size=n)
        before = _gap_value_for_indices(y_true, base_pred, attr_values, idx, metric_l)
        after = _gap_value_for_indices(y_true, mitigated_pred, attr_values, idx, metric_l)
        if not np.isnan(before) and not np.isnan(after):
            deltas.append(after - before)
    if not deltas:
        return {f"delta_{prefix}_gap_ci_low": np.nan, f"delta_{prefix}_gap_ci_high": np.nan, f"delta_{prefix}_gap_bootstrap_p_two_sided": np.nan, f"delta_{prefix}_gap_p_improvement": np.nan}
    arr = np.asarray(deltas)
    n_eff = len(arr)
    lower_tail = (float(np.sum(arr <= 0)) + 1.0) / (n_eff + 2.0)
    upper_tail = (float(np.sum(arr >= 0)) + 1.0) / (n_eff + 2.0)
    p_two = min(2.0 * min(lower_tail, upper_tail), 1.0)
    p_improve = lower_tail
    return {
        f"delta_{prefix}_gap_ci_low": float(np.percentile(arr, 2.5)),
        f"delta_{prefix}_gap_ci_high": float(np.percentile(arr, 97.5)),
        f"delta_{prefix}_gap_bootstrap_p_two_sided": p_two,
        f"delta_{prefix}_gap_p_improvement": p_improve,
        f"delta_{prefix}_gap_n_boot_effective": int(n_eff),
    }


def bootstrap_metric_change(y_true: np.ndarray, base_pred: np.ndarray, mitigated_pred: np.ndarray, metric: str, reps: int, seed: int) -> dict[str, float]:
    """Paired bootstrap for global performance change after mitigation.

    The returned delta is after-minus-before. For balanced accuracy and accuracy,
    positive values mean the mitigation improved predictive performance.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    base_pred = np.asarray(base_pred).astype(int)
    mitigated_pred = np.asarray(mitigated_pred).astype(int)
    n = len(y_true)
    metric_l = str(metric).lower()

    def metric_value(indices: np.ndarray, pred: np.ndarray) -> float:
        yt = y_true[indices]
        yp = pred[indices]
        if metric_l in {"balanced_accuracy", "balanced_accuracy_score"}:
            return float(balanced_accuracy_score(yt, yp))
        if metric_l == "accuracy":
            return float(accuracy_score(yt, yp))
        if metric_l == "recall":
            return float(recall_score(yt, yp, zero_division=0))
        if metric_l == "precision":
            return float(precision_score(yt, yp, zero_division=0))
        if metric_l == "f1":
            return float(f1_score(yt, yp, zero_division=0))
        return float(balanced_accuracy_score(yt, yp))

    deltas: list[float] = []
    for _ in range(int(reps)):
        idx = rng.integers(0, n, size=n)
        try:
            deltas.append(metric_value(idx, mitigated_pred) - metric_value(idx, base_pred))
        except Exception:
            continue
    prefix = metric_l.replace("_score", "")
    if not deltas:
        return {f"delta_{prefix}_ci_low": np.nan, f"delta_{prefix}_ci_high": np.nan, f"delta_{prefix}_bootstrap_p_two_sided": np.nan, f"delta_{prefix}_p_improvement": np.nan}
    arr = np.asarray(deltas)
    n_eff = len(arr)
    lower_tail = (float(np.sum(arr <= 0)) + 1.0) / (n_eff + 2.0)
    upper_tail = (float(np.sum(arr >= 0)) + 1.0) / (n_eff + 2.0)
    p_two = min(2.0 * min(lower_tail, upper_tail), 1.0)
    p_improve = upper_tail
    return {
        f"delta_{prefix}_ci_low": float(np.percentile(arr, 2.5)),
        f"delta_{prefix}_ci_high": float(np.percentile(arr, 97.5)),
        f"delta_{prefix}_bootstrap_p_two_sided": p_two,
        f"delta_{prefix}_p_improvement": p_improve,
        f"delta_{prefix}_n_boot_effective": int(n_eff),
    }


def mitigation_method_family(method: str) -> str:
    """Return the mitigation family used for reporting and filtering.

    The reporting taxonomy uses the three-stage fairness terminology requested
    in this notebook: pre-processing, intra-processing and post-processing.
    Common in-process/in-processing aliases are accepted for backwards compatibility.
    """
    m = str(method or "").lower()
    if m.startswith("preprocess") or "smote" in m or "oversampling" in m or "reweigh" in m:
        return "pre-processing"
    if m.startswith("inprocess") or m.startswith("intraprocess") or "expgrad" in m or "exponentiated" in m or "adversarial" in m:
        return "intra-processing"
    return "post-processing"


def _family_key(family: str) -> str:
    f = str(family or "").lower().replace("_", "-").strip()
    aliases = {
        "pre": "pre-processing",
        "preprocessing": "pre-processing",
        "pre-processing": "pre-processing",
        "pre-process": "pre-processing",
        "in": "intra-processing",
        "inprocessing": "intra-processing",
        "in-process": "intra-processing",
        "in-processing": "intra-processing",
        "intra": "intra-processing",
        "intraprocessing": "intra-processing",
        "intra-process": "intra-processing",
        "intra-processing": "intra-processing",
        "post": "post-processing",
        "postprocessing": "post-processing",
        "post-processing": "post-processing",
        "post-process": "post-processing",
    }
    return aliases.get(f, f)


def selected_mitigation_methods(config) -> tuple[str, ...]:
    methods = tuple(getattr(config, "mitigation_methods", None) or (getattr(config, "mitigation_method", "postprocess_group_threshold_equalized_odds"),))
    allowed = {_family_key(x) for x in (getattr(config, "mitigation_families", None) or ("pre-processing", "intra-processing", "post-processing"))}
    if not allowed:
        allowed = {"pre-processing", "intra-processing", "post-processing"}
    return tuple(str(m) for m in methods if _family_key(mitigation_method_family(str(m))) in allowed)


def mitigation_method_catalogue(config) -> pd.DataFrame:
    """Describe all recognised mitigation methods and mark those selected for the run."""
    descriptions = {
        "preprocess_reweighing": ("pre-processing", "support", "Learns sample weights for target-by-sensitive-group cells and refits the model."),
        "preprocess_random_oversampling": ("pre-processing", "imbalanced-learn optional", "Balances target-by-sensitive-group cells by random oversampling before refitting."),
        "preprocess_smote": ("pre-processing", "imbalanced-learn", "Attempts SMOTE synthetic oversampling on the preprocessed feature matrix."),
        "preprocess_smoten": ("pre-processing", "imbalanced-learn", "Attempts SMOTEN-style nominal oversampling; falls back safely if incompatible."),
        "preprocess_smotenc": ("pre-processing", "imbalanced-learn", "Attempts SMOTENC-style mixed-data oversampling; falls back safely if incompatible."),
        "preprocess_smoteenn": ("pre-processing", "imbalanced-learn", "Attempts SMOTEENN oversampling plus edited-nearest-neighbour cleaning."),
        "inprocess_fairlearn_expgrad_demographic_parity": ("intra-processing", "fairlearn", "Refits a classifier with Fairlearn ExponentiatedGradient under demographic-parity constraints."),
        "inprocess_fairlearn_expgrad_equalized_odds": ("intra-processing", "fairlearn", "Refits a classifier with Fairlearn ExponentiatedGradient under equalized-odds constraints."),
        "inprocess_support_fairness_aware_threshold_search": ("intra-processing", "support fallback", "Uses a fairness-aware validation search when external intra-processing is unavailable."),
        "postprocess_group_threshold_equalized_odds": ("post-processing", "support", "Learns group-specific thresholds to reduce FPR and FNR disparity."),
        "postprocess_group_threshold_equal_opportunity": ("post-processing", "support", "Learns group-specific thresholds to reduce FNR/TPR disparity."),
        "postprocess_group_threshold_demographic_parity": ("post-processing", "support", "Learns group-specific thresholds to reduce selection-rate disparity."),
        "postprocess_group_threshold_balanced_accuracy": ("post-processing", "support", "Learns group thresholds while prioritising balanced accuracy."),
        "postprocess_fairlearn_threshold_demographic_parity": ("post-processing", "fairlearn", "Uses Fairlearn ThresholdOptimizer with demographic-parity constraints when available."),
        "postprocess_fairlearn_threshold_equalized_odds": ("post-processing", "fairlearn", "Uses Fairlearn ThresholdOptimizer with equalized-odds constraints when available."),
    }
    selected = set(selected_mitigation_methods(config))
    configured = [str(m) for m in (getattr(config, "mitigation_methods", None) or ())]
    catalogue_methods = []
    for method in list(DEFAULT_MITIGATION_METHODS) + configured:
        if method not in catalogue_methods:
            catalogue_methods.append(method)
    rows = []
    for method in catalogue_methods:
        family, dependency, description = descriptions.get(method, (mitigation_method_family(method), "support", "Configured mitigation method."))
        family = _family_key(family)
        rows.append({
            "mitigation_method": method,
            "mitigation_family": family,
            "selected_in_current_run": bool(method in selected),
            "external_dependency": dependency,
            "description": description,
            "fallback_behaviour": "If the external dependency is unavailable or the method fails, the workflow records the failure and keeps the analysis running with built-in support operations when possible.",
        })
    return pd.DataFrame(rows).sort_values(["mitigation_family", "selected_in_current_run", "mitigation_method"], ascending=[True, False, True]).reset_index(drop=True)

def _objective_from_method(method: str) -> tuple[str, str]:
    m = str(method or "postprocess_group_threshold_equalized_odds").lower()
    if "fairlearn" in m and ("threshold" in m or "postprocess" in m) and "demographic" in m:
        return "fairlearn_threshold", "demographic_parity"
    if "fairlearn" in m and ("threshold" in m or "postprocess" in m) and "equal" in m:
        return "fairlearn_threshold", "equalized_odds"
    if "fairlearn" in m and ("expgrad" in m or "exponentiated" in m) and "demographic" in m:
        return "fairlearn_expgrad", "demographic_parity"
    if "fairlearn" in m and ("expgrad" in m or "exponentiated" in m) and "equal" in m:
        return "fairlearn_expgrad", "equalized_odds"
    if "opportunity" in m:
        return "support", "equal_opportunity"
    if "demographic" in m:
        return "support", "demographic_parity"
    if "balanced" in m or "global" in m:
        return "support", "balanced_accuracy_constrained"
    return "support", "equalized_odds"


def _try_fairlearn_threshold_optimizer(model: Any, validation_df: pd.DataFrame, test_df: pd.DataFrame, attribute: str, constraint: str) -> tuple[np.ndarray, dict[str, Any]] | None:
    try:
        from fairlearn.postprocessing import ThresholdOptimizer
    except Exception as exc:
        return None
    try:
        X_val = validation_df[model.feature_cols]
        y_val = validation_df["target"].astype(int).to_numpy()
        X_test = test_df[model.feature_cols]
        sensitive_val = validation_df[attribute].fillna("Missing").astype(str).to_numpy()
        sensitive_test = test_df[attribute].fillna("Missing").astype(str).to_numpy()

        # Keep the fitted sklearn pipeline on CPU and force probability outputs
        # to float64 to avoid Fairlearn/Pandas dtype FutureWarnings.
        set_xgboost_prediction_device(model.estimator, "cpu")
        estimator = _Float64ProbabilityEstimator(model.estimator)
        post = ThresholdOptimizer(
            estimator=estimator,
            constraints=constraint,
            objective="balanced_accuracy_score",
            prefit=True,
            predict_method="predict_proba",
        )
        with _suppress_known_fairlearn_warnings():
            post.fit(X_val, y_val, sensitive_features=sensitive_val)
            pred = np.asarray(post.predict(X_test, sensitive_features=sensitive_test)).astype(int).reshape(-1)
        return pred, {
            "external_library": "fairlearn",
            "constraint": constraint,
            "fairlearn_status": "used",
            "fairlearn_warning_policy": "float64_probability_wrapper_and_specific_futurewarning_suppression",
        }
    except Exception as exc:
        return None


def _pipeline_model_components(model: Any):
    """Return cloned preprocessor and final estimator from a fitted sklearn pipeline."""
    from sklearn.base import clone
    if not hasattr(model.estimator, "named_steps") or "preprocess" not in model.estimator.named_steps or "model" not in model.estimator.named_steps:
        raise TypeError("Mitigation refit expects a fitted pipeline with 'preprocess' and 'model' steps.")
    return clone(model.estimator.named_steps["preprocess"]), clone(model.estimator.named_steps["model"])


def _predict_processed(preprocessor: Any, estimator: Any, df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = preprocessor.transform(df[feature_cols])
    set_xgboost_prediction_device(estimator, "cpu")
    if hasattr(estimator, "predict_proba"):
        proba = np.asarray(estimator.predict_proba(X))
        prob = proba[:, 1].astype(float) if proba.ndim == 2 and proba.shape[1] >= 2 else proba.reshape(-1).astype(float)
    elif hasattr(estimator, "decision_function"):
        raw = np.asarray(estimator.decision_function(X), dtype=float).reshape(-1)
        prob = 1.0 / (1.0 + np.exp(-raw))
    else:
        pred = np.asarray(estimator.predict(X)).astype(int).reshape(-1)
        prob = pred.astype(float)
    pred = np.asarray(estimator.predict(X)).astype(int).reshape(-1) if hasattr(estimator, "predict") else (prob >= 0.5).astype(int)
    return prob, pred


def _combined_resampling_label(train_df: pd.DataFrame, attribute: str) -> np.ndarray:
    return train_df["target"].astype(int).astype(str).to_numpy() + "__" + train_df[attribute].fillna("Missing").astype(str).to_numpy()


def _target_from_combined_labels(labels: np.ndarray) -> np.ndarray:
    return np.asarray([int(str(x).split("__", 1)[0]) for x in labels], dtype=int)


def _try_reweighing_refit(model: Any, train_df: pd.DataFrame, test_df: pd.DataFrame, attribute: str) -> tuple[np.ndarray, dict[str, Any]] | None:
    try:
        from sklearn.base import clone
        X_train = train_df[model.feature_cols]
        y_train = train_df["target"].astype(int).to_numpy()
        cells = pd.Series(_combined_resampling_label(train_df, attribute))
        cell_counts = cells.value_counts().to_dict()
        n = len(cells)
        n_cells = max(len(cell_counts), 1)
        weights = cells.map(lambda x: n / (n_cells * max(cell_counts.get(x, 1), 1))).astype(float).to_numpy()
        estimator = clone(model.estimator)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*'penalty' was deprecated.*", category=FutureWarning)
            try:
                estimator.fit(X_train, y_train, model__sample_weight=weights)
                status = "used_model_sample_weight"
            except Exception:
                estimator.fit(X_train, y_train, sample_weight=weights)
                status = "used_pipeline_sample_weight"
        set_xgboost_prediction_device(estimator, "cpu")
        if hasattr(estimator, "predict_proba"):
            proba = np.asarray(estimator.predict_proba(test_df[model.feature_cols]))
            prob = proba[:, 1].astype(float) if proba.ndim == 2 and proba.shape[1] >= 2 else proba.reshape(-1).astype(float)
        else:
            pred0 = np.asarray(estimator.predict(test_df[model.feature_cols])).astype(int).reshape(-1)
            prob = pred0.astype(float)
        pred = (prob >= float(model.threshold)).astype(int)
        return pred, {"mitigation_backend": "support", "resampling_status": status, "resampling_unit": "target_by_sensitive_attribute_cell"}
    except Exception as exc:
        return None


def _try_resampling_refit(model: Any, train_df: pd.DataFrame, test_df: pd.DataFrame, attribute: str, method: str, config) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Refit the model after target-by-sensitive-group balancing.

    SMOTE-family methods are attempted on the preprocessed feature matrix so that the
    function can support mixed numeric/categorical CSVs after imputation and encoding.
    If an imbalanced-learn method is unavailable or incompatible with sparse subgroup
    cells, the caller can fall back to another support operation.
    """
    try:
        from sklearn.base import clone
        from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTEN, SMOTENC
        from imblearn.combine import SMOTEENN
    except Exception:
        return None
    try:
        preprocessor, estimator = _pipeline_model_components(model)
        X_train_raw = train_df[model.feature_cols]
        y_train = train_df["target"].astype(int).to_numpy()
        y_combo = _combined_resampling_label(train_df, attribute)
        X_train_proc = preprocessor.fit_transform(X_train_raw, y_train)
        method_l = str(method).lower()
        min_cell = int(pd.Series(y_combo).value_counts().min())
        k = max(1, min(5, min_cell - 1))
        if "random" in method_l or min_cell < 2:
            sampler = RandomOverSampler(random_state=int(config.random_state))
            sampler_name = "RandomOverSampler"
        elif "smoteenn" in method_l:
            sampler = SMOTEENN(random_state=int(config.random_state), smote=SMOTE(random_state=int(config.random_state), k_neighbors=k))
            sampler_name = "SMOTEENN"
        elif "smoten" in method_l:
            sampler = SMOTEN(random_state=int(config.random_state), k_neighbors=k)
            sampler_name = "SMOTEN"
        elif "smotenc" in method_l:
            # After one-hot encoding every column is numeric. SMOTENC has no reliable
            # categorical index in that representation, so this is a conservative SMOTE approximation.
            sampler = SMOTE(random_state=int(config.random_state), k_neighbors=k)
            sampler_name = "SMOTENC_approx_via_SMOTE_after_one_hot"
        else:
            sampler = SMOTE(random_state=int(config.random_state), k_neighbors=k)
            sampler_name = "SMOTE"
        X_res, y_combo_res = sampler.fit_resample(X_train_proc, y_combo)
        y_res = _target_from_combined_labels(y_combo_res)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*'penalty' was deprecated.*", category=FutureWarning)
            estimator.fit(X_res, y_res)
        _, pred = _predict_processed(preprocessor, estimator, test_df, model.feature_cols)
        return pred, {
            "mitigation_backend": "imbalanced_learn",
            "resampling_status": "used",
            "sampler": sampler_name,
            "resampling_unit": "target_by_sensitive_attribute_cell",
            "resampled_training_rows": int(X_res.shape[0]),
        }
    except Exception as exc:
        return None


def _try_fairlearn_expgrad(model: Any, train_df: pd.DataFrame, test_df: pd.DataFrame, attribute: str, constraint: str, config) -> tuple[np.ndarray, dict[str, Any]] | None:
    try:
        from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
    except Exception:
        return None
    try:
        preprocessor, estimator = _pipeline_model_components(model)
        X_train = preprocessor.fit_transform(train_df[model.feature_cols], train_df["target"].astype(int).to_numpy())
        X_test = preprocessor.transform(test_df[model.feature_cols])
        y_train = train_df["target"].astype(int).to_numpy()
        if str(constraint).lower() == "demographic_parity":
            constraints = DemographicParity()
        else:
            constraints = EqualizedOdds()
        mitigator = ExponentiatedGradient(estimator=estimator, constraints=constraints, eps=0.02)
        sensitive_train = train_df[attribute].fillna("Missing").astype(str).to_numpy()
        with _suppress_known_fairlearn_warnings():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r".*'penalty' was deprecated.*", category=FutureWarning)
                warnings.filterwarnings("ignore", message=r".*incompatible dtype.*", category=FutureWarning, module=r"fairlearn\..*")
                mitigator.fit(X_train, y_train, sensitive_features=sensitive_train)
            set_xgboost_prediction_device(estimator, "cpu")
            pred = np.asarray(mitigator.predict(X_test)).astype(int).reshape(-1)
        return pred, {"mitigation_backend": "fairlearn", "external_library": "fairlearn", "constraint": constraint, "fairlearn_status": "used", "reduction": "ExponentiatedGradient", "fairlearn_warning_policy": "specific_futurewarning_suppression"}
    except Exception as exc:
        return None


def _append_mcnemar_columns(merged: pd.DataFrame, test_df: pd.DataFrame, y_test: np.ndarray, base_pred: np.ndarray, mitigated_pred: np.ndarray, attr: str) -> pd.DataFrame:
    mcnemar_rows: list[dict[str, Any]] = []
    for group, idx in test_df.groupby(attr, dropna=False).groups.items():
        idx = np.asarray(list(idx), dtype=int)
        true_g = y_test[idx]
        before_g = base_pred[idx]
        after_g = mitigated_pred[idx]
        neg = true_g == 0
        if neg.any():
            p, improved, worsened = exact_mcnemar_p(before_g[neg] == 1, after_g[neg] == 1)
            mcnemar_rows.append({"group": group, "metric": "fpr", "mcnemar_p": p, "discordant_improved": improved, "discordant_worsened": worsened})
        pos = true_g == 1
        if pos.any():
            p, improved, worsened = exact_mcnemar_p(before_g[pos] == 0, after_g[pos] == 0)
            mcnemar_rows.append({"group": group, "metric": "fnr", "mcnemar_p": p, "discordant_improved": improved, "discordant_worsened": worsened})
    if mcnemar_rows:
        mcnemar = pd.DataFrame(mcnemar_rows)
        for metric in ["fpr", "fnr"]:
            sub = mcnemar[mcnemar["metric"] == metric].set_index("group")
            merged[f"{metric}_mcnemar_p"] = merged["group"].map(sub["mcnemar_p"])
            merged[f"{metric}_discordant_improved"] = merged["group"].map(sub["discordant_improved"])
            merged[f"{metric}_discordant_worsened"] = merged["group"].map(sub["discordant_worsened"])
    return merged


def _summarise_one_mitigation(model_name: str, method: str, attr: str, test_df: pd.DataFrame, prob_test: np.ndarray, base_pred: np.ndarray, mitigated_pred: np.ndarray, model_threshold: float, config, extra: dict[str, Any] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    y_test = test_df["target"].astype(int).to_numpy()
    base_metrics = binary_classification_metrics(y_test, prob_test, model_threshold, y_pred=base_pred)
    mitigated_metrics = binary_classification_metrics(y_test, prob_test, model_threshold, y_pred=mitigated_pred)
    extra = extra or {}
    method_family = extra.get("mitigation_family", mitigation_method_family(method))
    base_group = _fairness_by_group_from_predictions(test_df, base_pred, attr, model_name, "original")
    mitigated_group = _fairness_by_group_from_predictions(test_df, mitigated_pred, attr, model_name, "mitigated")
    merged = base_group.merge(mitigated_group, on=["model", "attribute", "group", "n", "events", "non_events"], suffixes=("_original", "_mitigated"))
    merged.insert(1, "mitigation_method", method)
    merged.insert(2, "mitigation_family", method_family)
    for metric in ["fpr", "fnr", "tpr", "selection_rate", "accuracy"]:
        merged[f"delta_{metric}"] = merged[f"{metric}_mitigated"] - merged[f"{metric}_original"]
    merged = _append_mcnemar_columns(merged, test_df, y_test, base_pred, mitigated_pred, attr)
    base_gap = fairness_gap_summary(base_group.assign(prediction_type="original"))
    mitigated_gap = fairness_gap_summary(mitigated_group.assign(prediction_type="mitigated"))
    fpr_gap_before = float(base_gap.iloc[0].get("fpr_gap", np.nan)) if not base_gap.empty else np.nan
    fnr_gap_before = float(base_gap.iloc[0].get("fnr_gap", np.nan)) if not base_gap.empty else np.nan
    fpr_gap_after = float(mitigated_gap.iloc[0].get("fpr_gap", np.nan)) if not mitigated_gap.empty else np.nan
    fnr_gap_after = float(mitigated_gap.iloc[0].get("fnr_gap", np.nan)) if not mitigated_gap.empty else np.nan
    boot_fpr = bootstrap_gap_change(test_df, base_pred, mitigated_pred, attr, "fpr", int(config.bootstrap_reps), int(config.random_state) + 11)
    boot_fnr = bootstrap_gap_change(test_df, base_pred, mitigated_pred, attr, "fnr", int(config.bootstrap_reps), int(config.random_state) + 13)
    boot_combined = bootstrap_gap_change(test_df, base_pred, mitigated_pred, attr, "combined_fpr_fnr", int(config.bootstrap_reps), int(config.random_state) + 17)
    boot_ba = bootstrap_metric_change(y_test, base_pred, mitigated_pred, "balanced_accuracy", int(config.bootstrap_reps), int(config.random_state) + 19)
    delta_ba = mitigated_metrics["balanced_accuracy"] - base_metrics["balanced_accuracy"]
    max_drop = getattr(config, "mitigation_max_accuracy_drop", None)
    if max_drop is None:
        max_drop = getattr(config, "max_balanced_accuracy_drop", None)
    if max_drop is None:
        max_drop = 1.0
    delta_combined = (fpr_gap_after + fnr_gap_after) - (fpr_gap_before + fnr_gap_before)
    alpha = float(getattr(config, "significance_alpha", 0.05))
    p_candidates = [
        boot_fpr.get("delta_fpr_gap_bootstrap_p_two_sided", np.nan),
        boot_fnr.get("delta_fnr_gap_bootstrap_p_two_sided", np.nan),
        boot_combined.get("delta_combined_fpr_fnr_gap_bootstrap_p_two_sided", np.nan),
    ]
    p_candidates = [float(p) for p in p_candidates if not pd.isna(p)]
    minimum_gap_p = min(p_candidates) if p_candidates else np.nan
    statistically_supported_gap_reduction = bool((delta_combined < 0) and (not pd.isna(minimum_gap_p)) and (minimum_gap_p < alpha))

    summary = {
        "model": model_name,
        "mitigation_method": method,
        "mitigation_family": method_family,
        "attribute": attr,
        "balanced_accuracy_original": base_metrics["balanced_accuracy"],
        "balanced_accuracy_mitigated": mitigated_metrics["balanced_accuracy"],
        "delta_balanced_accuracy": delta_ba,
        "max_balanced_accuracy_drop_allowed": max_drop,
        "passes_performance_guard": delta_ba >= -float(max_drop),
        "fpr_gap_original": fpr_gap_before,
        "fpr_gap_mitigated": fpr_gap_after,
        "delta_fpr_gap": fpr_gap_after - fpr_gap_before,
        "fnr_gap_original": fnr_gap_before,
        "fnr_gap_mitigated": fnr_gap_after,
        "delta_fnr_gap": fnr_gap_after - fnr_gap_before,
        "combined_gap_original": fpr_gap_before + fnr_gap_before,
        "combined_gap_mitigated": fpr_gap_after + fnr_gap_after,
        "delta_combined_gap": delta_combined,
        "gap_reduction_statistically_supported_at_alpha": statistically_supported_gap_reduction,
        "minimum_gap_bootstrap_p_two_sided": minimum_gap_p,
        "recommended_by_gap_and_guard": ((fpr_gap_after + fnr_gap_after) < (fpr_gap_before + fnr_gap_before)) and (delta_ba >= -float(max_drop)),
        **extra,
        **boot_fpr,
        **boot_fnr,
        **boot_combined,
        **boot_ba,
    }
    return merged, summary


def run_threshold_mitigation(models: dict[str, Any], train_df: pd.DataFrame, validation_df: pd.DataFrame, test_df: pd.DataFrame, validation_probs: dict[str, np.ndarray], test_probs: dict[str, np.ndarray], config, attributes: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    progress_rows: list[dict[str, Any]] = []
    methods = selected_mitigation_methods(config)
    total_jobs = len(models) * len(tuple(attributes or ())) * len(methods)
    _mitigation_progress(config, f"Mitigation/retraining stage started: {len(models)} model(s) × {len(tuple(attributes or ())) } attribute(s) × {len(methods)} method(s) = {total_jobs} job(s).")
    job_idx = 0

    for model_name, model in models.items():
        prob_val = validation_probs[model_name]
        prob_test = test_probs[model_name]
        base_pred = (prob_test >= model.threshold).astype(int)
        _mitigation_progress(config, f"Base model '{model_name}' ready; selected validation threshold={float(model.threshold):.4f}.", level="verbose")
        for attr in attributes:
            if attr not in test_df.columns or attr not in validation_df.columns:
                _mitigation_progress(config, f"Skipping attribute '{attr}' because it is not present in validation/test data.")
                continue
            for method in methods:
                job_idx += 1
                method = str(method)
                family = mitigation_method_family(method)
                backend, objective = _objective_from_method(method)
                used_method = method
                retraining_required = family in {"pre-processing", "intra-processing"}
                action = "model refit/retraining" if retraining_required else "threshold post-processing"
                _mitigation_progress(config, f"Job {job_idx}/{total_jobs}: {model_name} | {attr} | {method} ({family}; {action}) started.")
                start_time = time.time()
                extra: dict[str, Any] = {
                    "mitigation_objective": objective,
                    "mitigation_backend": backend,
                    "mitigation_family": family,
                    "retraining_required": retraining_required,
                }
                progress_row: dict[str, Any] = {
                    "mitigation_job": job_idx,
                    "total_jobs": total_jobs,
                    "model": model_name,
                    "attribute": attr,
                    "mitigation_method": method,
                    "mitigation_family": family,
                    "action": action,
                    "retraining_required": retraining_required,
                    "status": "started",
                    "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                thresholds: dict[Any, float] = {}
                mitigated_pred: np.ndarray | None = None

                if family == "pre-processing":
                    _mitigation_progress(config, f"Job {job_idx}/{total_jobs}: refitting '{model_name}' after {method} on target×{attr} cells.", level="verbose")
                    if "reweigh" in method.lower():
                        result = _try_reweighing_refit(model, train_df, test_df, attr)
                    else:
                        result = _try_resampling_refit(model, train_df, test_df, attr, method, config)
                    if result is not None:
                        mitigated_pred, ext = result
                        extra.update(ext)
                    else:
                        _mitigation_progress(config, f"Job {job_idx}/{total_jobs}: external/support pre-processing failed; falling back to validation-learned group thresholds.")
                        thresholds = group_thresholds_from_validation(validation_df, prob_val, model.threshold, attr, config.threshold_grid, objective="equalized_odds", min_group_size=getattr(config, "min_group_size_for_mitigation", 20))
                        mitigated_pred = apply_group_thresholds(test_df, prob_test, attr, thresholds, model.threshold)
                        extra.update({"mitigation_backend": "support_fallback", "external_status": "preprocessing_method_unavailable_or_failed"})

                elif backend == "fairlearn_expgrad":
                    _mitigation_progress(config, f"Job {job_idx}/{total_jobs}: attempting Fairlearn ExponentiatedGradient refit for '{model_name}'.", level="verbose")
                    result = _try_fairlearn_expgrad(model, train_df, test_df, attr, objective, config)
                    if result is not None:
                        mitigated_pred, ext = result
                        extra.update(ext)
                    else:
                        _mitigation_progress(config, f"Job {job_idx}/{total_jobs}: Fairlearn intra-processing unavailable/failed; falling back to validation-learned group thresholds.")
                        thresholds = group_thresholds_from_validation(validation_df, prob_val, model.threshold, attr, config.threshold_grid, objective=objective, min_group_size=getattr(config, "min_group_size_for_mitigation", 20))
                        mitigated_pred = apply_group_thresholds(test_df, prob_test, attr, thresholds, model.threshold)
                        extra.update({"mitigation_backend": "support_fallback", "external_library": "fairlearn", "external_status": "intraprocessing_unavailable_or_failed"})

                elif backend == "fairlearn_threshold":
                    _mitigation_progress(config, f"Job {job_idx}/{total_jobs}: attempting Fairlearn ThresholdOptimizer; no base-model retraining required.", level="verbose")
                    result = _try_fairlearn_threshold_optimizer(model, validation_df, test_df, attr, objective)
                    if result is None:
                        _mitigation_progress(config, f"Job {job_idx}/{total_jobs}: Fairlearn post-processing unavailable/failed; using support group-threshold search.", level="verbose")
                        thresholds = group_thresholds_from_validation(validation_df, prob_val, model.threshold, attr, config.threshold_grid, objective=objective, min_group_size=getattr(config, "min_group_size_for_mitigation", 20))
                        mitigated_pred = apply_group_thresholds(test_df, prob_test, attr, thresholds, model.threshold)
                        extra.update({"mitigation_backend": "support_fallback", "external_library": "fairlearn", "external_status": "postprocessing_unavailable_or_failed"})
                    else:
                        mitigated_pred, ext = result
                        extra.update(ext)

                else:
                    _mitigation_progress(config, f"Job {job_idx}/{total_jobs}: learning group-specific thresholds from validation data; no base-model retraining required.", level="verbose")
                    thresholds = group_thresholds_from_validation(validation_df, prob_val, model.threshold, attr, config.threshold_grid, objective=objective, min_group_size=getattr(config, "min_group_size_for_mitigation", 20))
                    mitigated_pred = apply_group_thresholds(test_df, prob_test, attr, thresholds, model.threshold)
                    extra.update({"mitigation_backend": "support"})

                if mitigated_pred is None:
                    elapsed = time.time() - start_time
                    progress_row.update({
                        "status": "skipped_no_predictions",
                        "elapsed_seconds": float(elapsed),
                        "backend_used": extra.get("mitigation_backend"),
                        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    })
                    progress_rows.append(progress_row)
                    _mitigation_progress(config, f"Job {job_idx}/{total_jobs}: skipped because no mitigated predictions were produced.")
                    continue
                merged, summary = _summarise_one_mitigation(model_name, used_method, attr, test_df, prob_test, base_pred, mitigated_pred, model.threshold, config, extra)
                elapsed = time.time() - start_time
                summary["mitigation_job"] = job_idx
                summary["elapsed_seconds"] = float(elapsed)
                progress_row.update({
                    "status": "completed",
                    "elapsed_seconds": float(elapsed),
                    "backend_used": summary.get("mitigation_backend", extra.get("mitigation_backend")),
                    "external_library": summary.get("external_library", extra.get("external_library")),
                    "delta_balanced_accuracy": summary.get("delta_balanced_accuracy"),
                    "delta_fpr_gap": summary.get("delta_fpr_gap"),
                    "delta_fnr_gap": summary.get("delta_fnr_gap"),
                    "delta_combined_gap": summary.get("delta_combined_gap"),
                    "minimum_gap_bootstrap_p_two_sided": summary.get("minimum_gap_bootstrap_p_two_sided"),
                    "recommended_by_gap_and_guard": summary.get("recommended_by_gap_and_guard"),
                    "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
                progress_rows.append(progress_row)
                _mitigation_progress(
                    config,
                    f"Job {job_idx}/{total_jobs}: completed in {elapsed:.1f}s | Δbalanced_accuracy={summary.get('delta_balanced_accuracy', np.nan):+.4f} | Δcombined_gap={summary.get('delta_combined_gap', np.nan):+.4f} | min_p={summary.get('minimum_gap_bootstrap_p_two_sided', np.nan)}."
                )
                group_rows.append(merged)
                summary_rows.append(summary)
                if thresholds:
                    for group, threshold in thresholds.items():
                        threshold_rows.append({
                            "model": model_name,
                            "mitigation_method": used_method,
                            "mitigation_family": family,
                            "attribute": attr,
                            "group": group,
                            "threshold": float(threshold),
                            "fallback_threshold": float(model.threshold),
                            "used_fallback_threshold": bool(abs(float(threshold) - float(model.threshold)) < 1e-12),
                            "mitigation_objective": objective,
                        })
                else:
                    threshold_rows.append({
                        "model": model_name,
                        "mitigation_method": used_method,
                        "mitigation_family": family,
                        "attribute": attr,
                        "group": "not_threshold_based",
                        "threshold": np.nan,
                        "fallback_threshold": float(model.threshold),
                        "used_fallback_threshold": False,
                        "mitigation_objective": objective,
                    })
    _mitigation_progress(config, "Mitigation/retraining stage finished.")
    run_threshold_mitigation.last_progress_log = pd.DataFrame(progress_rows)
    return (
        pd.concat(group_rows, ignore_index=True) if group_rows else pd.DataFrame(),
        pd.DataFrame(summary_rows).sort_values(["recommended_by_gap_and_guard", "delta_combined_gap", "delta_balanced_accuracy"], ascending=[False, True, False]).reset_index(drop=True) if summary_rows else pd.DataFrame(),
        pd.DataFrame(threshold_rows),
    )


def summarise_mitigation_combinations(mitigation_summary: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Aggregate model × mitigation-method results for strategy selection.

    A positive combined_gap_reduction_sum means that mitigation reduced the combined
    FPR+FNR gap across the evaluated attributes. The ranking score rewards fairness
    improvement and penalises balanced-accuracy loss.
    """
    if mitigation_summary is None or mitigation_summary.empty:
        return pd.DataFrame()
    df = mitigation_summary.copy()
    required = {"model", "mitigation_method", "mitigation_family", "attribute", "combined_gap_original", "combined_gap_mitigated", "delta_combined_gap", "delta_balanced_accuracy"}
    if required - set(df.columns):
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (model, family, method), sub in df.groupby(["model", "mitigation_family", "mitigation_method"], dropna=False):
        sub = sub.copy()
        original_sum = float(sub["combined_gap_original"].astype(float).sum(skipna=True))
        mitigated_sum = float(sub["combined_gap_mitigated"].astype(float).sum(skipna=True))
        gap_reduction = original_sum - mitigated_sum
        pct_reduction = gap_reduction / original_sum if original_sum > 0 else np.nan
        ba_delta_mean = float(sub["delta_balanced_accuracy"].astype(float).mean())
        ba_delta_min = float(sub["delta_balanced_accuracy"].astype(float).min())
        ranking_score = gap_reduction + 0.50 * min(ba_delta_mean, 0.0) + 0.10 * max(ba_delta_mean, 0.0)
        best_attr_row = sub.sort_values("delta_combined_gap", ascending=True).iloc[0]
        passes_all = bool(sub.get("passes_performance_guard", pd.Series([True] * len(sub))).fillna(False).all())
        recommended_count = int(sub.get("recommended_by_gap_and_guard", pd.Series([False] * len(sub))).fillna(False).sum())
        p_cols = [c for c in sub.columns if c.endswith("_bootstrap_p_two_sided")]
        min_p = float(sub[p_cols].min(numeric_only=True).min()) if p_cols else np.nan
        combined_p = float(sub["delta_combined_fpr_fnr_gap_bootstrap_p_two_sided"].min(skipna=True)) if "delta_combined_fpr_fnr_gap_bootstrap_p_two_sided" in sub.columns else np.nan
        rows.append({
            "model": model,
            "mitigation_family": family,
            "mitigation_method": method,
            "strategy_type": "model × mitigation method aggregated over evaluated attributes",
            "retraining_required": family in {"pre-processing", "intra-processing"},
            "n_attributes_evaluated": int(sub["attribute"].nunique()),
            "n_attributes_recommended": recommended_count,
            "combined_gap_original_sum": original_sum,
            "combined_gap_mitigated_sum": mitigated_sum,
            "combined_gap_reduction_sum": gap_reduction,
            "combined_gap_reduction_percent": pct_reduction,
            "mean_delta_combined_gap": float(sub["delta_combined_gap"].astype(float).mean()),
            "median_delta_combined_gap": float(sub["delta_combined_gap"].astype(float).median()),
            "mean_delta_balanced_accuracy": ba_delta_mean,
            "minimum_delta_balanced_accuracy": ba_delta_min,
            "passes_performance_guard_all_attributes": passes_all,
            "best_improved_attribute": best_attr_row.get("attribute"),
            "best_attribute_delta_combined_gap": float(best_attr_row.get("delta_combined_gap", np.nan)),
            "minimum_bootstrap_p_two_sided": min_p,
            "combined_gap_bootstrap_p_two_sided": combined_p,
            "statistically_supported_at_alpha": bool((not np.isnan(min_p)) and (min_p < float(alpha))),
            "combined_gap_reduction_supported_at_alpha": bool((gap_reduction > 0) and (not np.isnan(combined_p)) and (combined_p < float(alpha))),
            "selection_score": ranking_score,
            "recommended_combination": bool((gap_reduction > 0) and passes_all and recommended_count > 0),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(
        ["recommended_combination", "selection_score", "combined_gap_reduction_percent", "mean_delta_balanced_accuracy"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out

def mitigation_statistical_evidence_table(mitigation_summary: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Convert mitigation summary p-values and confidence intervals into an audit table.

    This table is intended for the notebook results section: it separates the statistical
    evidence for FPR-gap, FNR-gap, combined FPR+FNR-gap and balanced-accuracy changes.
    """
    if mitigation_summary is None or mitigation_summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    metric_specs = [
        ("FPR gap", "delta_fpr_gap", "delta_fpr_gap_ci_low", "delta_fpr_gap_ci_high", "delta_fpr_gap_bootstrap_p_two_sided", "delta_fpr_gap_p_improvement", "delta_fpr_gap_n_boot_effective", "lower_is_better"),
        ("FNR gap", "delta_fnr_gap", "delta_fnr_gap_ci_low", "delta_fnr_gap_ci_high", "delta_fnr_gap_bootstrap_p_two_sided", "delta_fnr_gap_p_improvement", "delta_fnr_gap_n_boot_effective", "lower_is_better"),
        ("Combined FPR+FNR gap", "delta_combined_gap", "delta_combined_fpr_fnr_gap_ci_low", "delta_combined_fpr_fnr_gap_ci_high", "delta_combined_fpr_fnr_gap_bootstrap_p_two_sided", "delta_combined_fpr_fnr_gap_p_improvement", "delta_combined_fpr_fnr_gap_n_boot_effective", "lower_is_better"),
        ("Balanced accuracy", "delta_balanced_accuracy", "delta_balanced_accuracy_ci_low", "delta_balanced_accuracy_ci_high", "delta_balanced_accuracy_bootstrap_p_two_sided", "delta_balanced_accuracy_p_improvement", "delta_balanced_accuracy_n_boot_effective", "higher_is_better"),
    ]
    for _, row in mitigation_summary.iterrows():
        for metric_name, delta_col, ci_low_col, ci_high_col, p_col, p_imp_col, n_boot_col, direction in metric_specs:
            delta = row.get(delta_col, np.nan)
            if pd.isna(delta):
                continue
            p_value = row.get(p_col, np.nan)
            ci_low = row.get(ci_low_col, np.nan)
            ci_high = row.get(ci_high_col, np.nan)
            p_improvement = row.get(p_imp_col, np.nan)
            n_boot_effective = row.get(n_boot_col, np.nan)
            improves = bool(delta < 0) if direction == "lower_is_better" else bool(delta > 0)
            significant = bool((not pd.isna(p_value)) and float(p_value) < float(alpha))
            if significant and improves:
                interpretation = "statistically supported improvement"
            elif significant and not improves:
                interpretation = "statistically supported worsening"
            elif improves:
                interpretation = "numerical improvement without statistical support"
            else:
                interpretation = "no improvement detected"
            rows.append({
                "model": row.get("model"),
                "mitigation_family": row.get("mitigation_family"),
                "mitigation_method": row.get("mitigation_method"),
                "attribute": row.get("attribute"),
                "metric": metric_name,
                "direction": "lower is better" if direction == "lower_is_better" else "higher is better",
                "delta_after_minus_before": float(delta),
                "ci_low_95": ci_low,
                "ci_high_95": ci_high,
                "bootstrap_p_two_sided": p_value,
                "bootstrap_probability_of_improvement": p_improvement,
                "n_boot_effective": n_boot_effective,
                "alpha": float(alpha),
                "is_statistically_significant": significant,
                "is_improvement": improves,
                "interpretation": interpretation,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order = ["is_statistically_significant", "is_improvement", "model", "mitigation_method", "attribute", "metric"]
    return out.sort_values(order, ascending=[False, False, True, True, True, True]).reset_index(drop=True)


# Backwards-compatible aliases used by older notebook/workflow cells.
def mitigation_combination_summary(mitigation_summary: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    return summarise_mitigation_combinations(mitigation_summary, alpha=alpha)


def mitigation_combination_leaderboard(mitigation_summary: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    return summarise_mitigation_combinations(mitigation_summary, alpha=alpha)


def merge_model_performance_fairness_summary(validation_results: pd.DataFrame, test_performance: pd.DataFrame, gap_df: pd.DataFrame) -> pd.DataFrame:
    """Merge validation, held-out test and baseline fairness gaps into one non-duplicated model-level table.

    The detailed subgroup table is kept separately. This table keeps one row per trained model and
    reports the worst fairness attribute plus aggregate gap summaries, so validation/test metrics are
    not repeated for every sensitive attribute.
    """
    val = validation_results.copy() if validation_results is not None else pd.DataFrame()
    test = test_performance.copy() if test_performance is not None else pd.DataFrame()
    gaps = gap_df.copy() if gap_df is not None else pd.DataFrame()
    if not val.empty:
        val = val.add_prefix("validation_").rename(columns={"validation_model": "model"})
    if not test.empty:
        test = test.add_prefix("test_").rename(columns={"test_model": "model"})
    if val.empty and test.empty:
        merged = pd.DataFrame()
    elif val.empty:
        merged = test
    elif test.empty:
        merged = val
    else:
        merged = val.merge(test, on="model", how="outer")
    if not gaps.empty:
        gaps = gaps[gaps.get("prediction_type", "baseline").astype(str).eq("baseline")] if "prediction_type" in gaps.columns else gaps
        rows = []
        for model, sub in gaps.groupby("model"):
            sub = sub.copy().sort_values("combined_fpr_fnr_gap", ascending=False)
            worst = sub.iloc[0]
            rows.append({
                "model": model,
                "worst_fairness_attribute": worst.get("attribute"),
                "worst_attribute_fpr_gap": float(worst.get("fpr_gap", np.nan)),
                "worst_attribute_fnr_gap": float(worst.get("fnr_gap", np.nan)),
                "worst_attribute_tpr_gap": float(worst.get("tpr_gap", np.nan)),
                "worst_attribute_selection_rate_gap": float(worst.get("selection_rate_gap", np.nan)),
                "worst_attribute_accuracy_gap": float(worst.get("accuracy_gap", np.nan)),
                "worst_attribute_combined_fpr_fnr_gap": float(worst.get("combined_fpr_fnr_gap", np.nan)),
                "mean_combined_fpr_fnr_gap": float(sub["combined_fpr_fnr_gap"].mean(skipna=True)),
                "max_combined_fpr_fnr_gap": float(sub["combined_fpr_fnr_gap"].max(skipna=True)),
                "audited_sensitive_attributes": ", ".join(dict.fromkeys(sub["attribute"].astype(str).tolist())),
            })
        fairness = pd.DataFrame(rows)
        merged = merged.merge(fairness, on="model", how="outer") if not merged.empty else fairness
    else:
        merged["worst_fairness_attribute"] = "no_sensitive_attribute"
    preferred = [
        "model",
        "validation_cv_score", "validation_threshold", "validation_balanced_accuracy", "validation_roc_auc", "validation_recall", "validation_precision", "validation_f1",
        "test_threshold", "test_balanced_accuracy", "test_roc_auc", "test_average_precision", "test_recall", "test_precision", "test_f1", "test_fpr", "test_fnr", "test_ece", "test_brier",
        "worst_fairness_attribute", "worst_attribute_fpr_gap", "worst_attribute_fnr_gap", "worst_attribute_tpr_gap", "worst_attribute_selection_rate_gap", "worst_attribute_accuracy_gap", "worst_attribute_combined_fpr_fnr_gap",
        "mean_combined_fpr_fnr_gap", "max_combined_fpr_fnr_gap", "audited_sensitive_attributes",
    ]
    if merged.empty:
        return merged
    cols = [c for c in preferred if c in merged.columns] + [c for c in merged.columns if c not in preferred]
    sort_cols = [c for c in ["test_balanced_accuracy", "worst_attribute_combined_fpr_fnr_gap"] if c in merged.columns]
    if sort_cols:
        ascending = [False if c == "test_balanced_accuracy" else True for c in sort_cols]
        merged = merged.sort_values(sort_cols, ascending=ascending)
    return merged[cols].reset_index(drop=True)


def rank_mitigation_candidates(mitigation_summary: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Rank each model × mitigation × attribute result for detailed audit selection."""
    if mitigation_summary is None or mitigation_summary.empty:
        return pd.DataFrame()
    df = mitigation_summary.copy()
    df["combined_gap_improvement"] = -df["delta_combined_gap"].astype(float)
    df["fpr_gap_improvement"] = -df["delta_fpr_gap"].astype(float)
    df["fnr_gap_improvement"] = -df["delta_fnr_gap"].astype(float)
    df["requires_model_refit"] = df["mitigation_family"].astype(str).str.contains("pre|in|in-", case=False, regex=True)
    p_cols = [c for c in df.columns if c.endswith("_bootstrap_p_two_sided") or c.endswith("_p_improvement")]
    if p_cols:
        df["minimum_reported_p_value"] = df[p_cols].min(axis=1, numeric_only=True)
        df["statistically_supported_at_alpha"] = df["minimum_reported_p_value"].astype(float) < float(alpha)
    else:
        df["minimum_reported_p_value"] = np.nan
        df["statistically_supported_at_alpha"] = False
    perf_guard = df.get("passes_performance_guard", pd.Series(True, index=df.index)).astype(bool)
    df["candidate_score"] = (
        df["combined_gap_improvement"].fillna(0)
        + 0.10 * df["delta_balanced_accuracy"].fillna(0)
        + 0.03 * df["statistically_supported_at_alpha"].astype(float)
        + 0.02 * perf_guard.astype(float)
    )
    df = df.sort_values(["passes_performance_guard", "candidate_score", "combined_gap_improvement", "delta_balanced_accuracy"], ascending=[False, False, False, False]).reset_index(drop=True)
    df.insert(0, "recommendation_rank", np.arange(1, len(df) + 1))
    df["recommendation"] = np.where(
        perf_guard & (df["combined_gap_improvement"] > 0),
        "candidate for final model comparison",
        "audit only; fairness or performance guard not improved",
    )
    preferred = [
        "recommendation_rank", "recommendation", "model", "mitigation_method", "mitigation_family", "attribute", "requires_model_refit",
        "balanced_accuracy_original", "balanced_accuracy_mitigated", "delta_balanced_accuracy", "passes_performance_guard",
        "combined_gap_original", "combined_gap_mitigated", "combined_gap_improvement",
        "fpr_gap_original", "fpr_gap_mitigated", "fpr_gap_improvement", "fnr_gap_original", "fnr_gap_mitigated", "fnr_gap_improvement",
        "minimum_reported_p_value", "statistically_supported_at_alpha", "candidate_score",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols]
