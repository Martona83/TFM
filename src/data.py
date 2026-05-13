from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from .config import (
    CLINICAL_EXAMPLE_BASELINE_FEATURES,
    CLINICAL_EXAMPLE_DEFAULT_SENSITIVE_ATTRS,
    CLINICAL_EXAMPLE_LANDMARK_FEATURES,
    PipelineConfig,
    is_id_like,
)
from .runtime import find_csv_path


CLINICAL_EXAMPLE_REQUIRED_COLUMNS = {
    "pidnum", "cid", "time", "trt", "age", "wtkg", "hemo", "homo", "drugs",
    "karnof", "oprior", "z30", "zprior", "preanti", "race", "gender", "str2",
    "strat", "symptom", "treat", "offtrt", "cd40", "cd420", "cd80", "cd820",
}
CLINICAL_EXAMPLE_LEAKAGE_OR_NON_PREDICTOR = {
    "pidnum", "cid", "time", "target", "target_original", "treat", "offtrt",
    "treatment_label", "strat_label",
}


def load_source_csv(config: PipelineConfig) -> tuple[pd.DataFrame, Path]:
    csv_path = find_csv_path(config.csv_path, config.csv_candidates)
    source = pd.read_csv(csv_path, **(config.csv_read_kwargs or {}))
    source.attrs["source_raw_columns"] = int(source.shape[1])
    source.attrs["source_missing_cells"] = int(source.isna().sum().sum())
    return source, csv_path


def detect_dataset_mode(source: pd.DataFrame, config: PipelineConfig) -> str:
    requested = str(config.dataset_mode or "auto").lower()
    if requested in {"clinical_survival", "clinical_example_survival", "actg175", "actg175_survival"}:
        return "clinical_survival"
    if requested in {"generic", "generic_tabular", "custom", "custom_tabular"}:
        return "generic_tabular"
    if CLINICAL_EXAMPLE_REQUIRED_COLUMNS.issubset(set(source.columns)):
        return "clinical_survival"
    return "generic_tabular"


def add_clinical_example_derived_labels(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["age_group"] = np.where(df["age"] < 40, "<40", "40+")
    df["gender_label"] = df["gender"].map({0: "Female", 1: "Male"}).fillna("Unknown")
    df["race_label"] = df["race"].map({0: "White", 1: "Non-white"}).fillna("Unknown")
    df["homo_label"] = df["homo"].map({0: "No", 1: "Yes"}).fillna("Unknown")
    df["drugs_label"] = df["drugs"].map({0: "No", 1: "Yes"}).fillna("Unknown")
    df["treatment_label"] = df["trt"].map({0: "ZDV only", 1: "ZDV + ddI", 2: "ZDV + Zal", 3: "ddI only"}).fillna("Unknown")
    df["strat_label"] = df["strat"].map({1: "Naive", 2: "<=52 weeks", 3: ">52 weeks"}).fillna("Unknown")
    df.attrs.update(raw.attrs)
    df.attrs["derived_columns_added"] = int(df.shape[1] - int(raw.attrs.get("source_raw_columns", raw.shape[1])))
    return df


def clinical_example_feature_list(config: PipelineConfig) -> list[str]:
    if config.scenario == "landmark20":
        return [c for c in CLINICAL_EXAMPLE_LANDMARK_FEATURES]
    return [c for c in CLINICAL_EXAMPLE_BASELINE_FEATURES]


def build_clinical_example_analytic(raw_source: pd.DataFrame, config: PipelineConfig, csv_path: Path) -> dict[str, Any]:
    missing = sorted(CLINICAL_EXAMPLE_REQUIRED_COLUMNS.difference(raw_source.columns))
    if missing:
        raise ValueError("The CSV does not match the clinical-survival example schema. Missing columns: " + ", ".join(missing))
    raw = add_clinical_example_derived_labels(raw_source)
    if config.scenario not in {"baseline", "landmark20"}:
        raise ValueError("The clinical-survival example scenario must be 'baseline' or 'landmark20'.")

    if config.scenario == "baseline":
        censored_before_horizon = (raw["cid"] == 0) & (raw["time"] < config.horizon_days)
        analytic = raw.loc[~censored_before_horizon].copy()
        analytic["target"] = ((analytic["cid"] == 1) & (analytic["time"] <= config.horizon_days)).astype(int)
        endpoint = f"Observed AIDS/death event by day {config.horizon_days}; participants censored before that horizon are excluded."
        eligibility = pd.DataFrame([
            {"rule": "Raw source rows", "n": int(len(raw))},
            {"rule": f"Excluded: censored before day {config.horizon_days}", "n": int(censored_before_horizon.sum())},
            {"rule": "Final analytic cohort", "n": int(len(analytic))},
            {"rule": f"Observed events by day {config.horizon_days}", "n": int(analytic["target"].sum())},
            {"rule": f"Non-events by day {config.horizon_days}", "n": int((1 - analytic["target"]).sum())},
        ])
    else:
        not_observed_to_landmark = raw["time"] <= config.landmark_day
        censored_between = (raw["time"] > config.landmark_day) & (raw["cid"] == 0) & (raw["time"] < config.horizon_days)
        analytic = raw.loc[(~not_observed_to_landmark) & (~censored_between)].copy()
        analytic["target"] = ((analytic["cid"] == 1) & (analytic["time"] <= config.horizon_days)).astype(int)
        endpoint = f"Observed AIDS/death event after landmark day {config.landmark_day} and by day {config.horizon_days}."
        eligibility = pd.DataFrame([
            {"rule": "Raw source rows", "n": int(len(raw))},
            {"rule": f"Excluded: event/censoring at or before landmark day {config.landmark_day}", "n": int(not_observed_to_landmark.sum())},
            {"rule": f"Excluded: censored between landmark and day {config.horizon_days}", "n": int(censored_between.sum())},
            {"rule": "Final analytic cohort", "n": int(len(analytic))},
            {"rule": f"Observed events between landmark and day {config.horizon_days}", "n": int(analytic["target"].sum())},
            {"rule": f"Non-events by day {config.horizon_days}", "n": int((1 - analytic["target"]).sum())},
        ])
    sensitive_attrs = tuple(config.sensitive_attrs) if isinstance(config.sensitive_attrs, tuple) else CLINICAL_EXAMPLE_DEFAULT_SENSITIVE_ATTRS
    feature_cols = tuple(config.feature_cols) if config.feature_cols else tuple(clinical_example_feature_list(config))
    return {
        "source": raw_source,
        "raw": raw,
        "analytic": analytic.reset_index(drop=True),
        "csv_path": csv_path,
        "dataset_mode": "clinical_survival",
        "dataset_name": "clinical_example" if str(config.dataset_name).lower() == "auto" else str(config.dataset_name),
        "target_col": "target",
        "target_original_col": "cid",
        "positive_target_value": 1,
        "sensitive_attrs": sensitive_attrs,
        "feature_cols_seed": feature_cols,
        "eligibility": eligibility,
        "endpoint_description": endpoint,
    }


def infer_target_col(df: pd.DataFrame, config: PipelineConfig) -> str:
    if config.target_col and config.target_col in df.columns:
        return str(config.target_col)
    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in config.target_candidates:
        if str(cand).lower() in lower_map:
            return str(lower_map[str(cand).lower()])
    binary_cols: list[str] = []
    for col in df.columns:
        s = df[col].dropna()
        if s.nunique() == 2 and not is_id_like(col, config.id_like_patterns):
            binary_cols.append(str(col))
    if len(binary_cols) == 1:
        return binary_cols[0]
    if binary_cols:
        raise ValueError("Could not infer a unique target column. Candidate binary columns: " + ", ".join(binary_cols[:20]) + ". Set USER_CONFIG['data']['target_col'] explicitly.")
    raise ValueError("Could not infer target column. Set USER_CONFIG['data']['target_col'] explicitly.")


def encode_binary_target(series: pd.Series, positive_value: Any) -> tuple[pd.Series, Any]:
    s = series.copy()
    if positive_value is None:
        values = list(pd.Series(s.dropna().unique()).sort_values())
        if len(values) != 2:
            raise ValueError("positive_target_value=None requires exactly two non-missing target levels.")
        positive_value = values[-1]
    encoded = (s == positive_value).astype(int)
    if encoded.nunique() < 2:
        values = list(s.dropna().unique())
        if len(values) == 2:
            positive_value = values[-1]
            encoded = (s == positive_value).astype(int)
    if encoded.nunique() < 2:
        raise ValueError("Encoded target has one class only. Check positive_target_value and target_col.")
    return encoded, positive_value


def _make_age_group_if_needed(df: pd.DataFrame, sensitive: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    fixed: list[str] = []
    for attr in sensitive:
        if attr == "age" and attr in out.columns and pd.api.types.is_numeric_dtype(out[attr]):
            new = "age_group"
            if new not in out.columns:
                median = float(out[attr].median())
                out[new] = np.where(out[attr] < median, f"<{median:g}", f">={median:g}")
            fixed.append(new)
        else:
            fixed.append(attr)
    # de-duplicate preserving order
    seen = set()
    dedup = []
    for attr in fixed:
        if attr in out.columns and attr not in seen:
            dedup.append(attr); seen.add(attr)
    return out, dedup


def infer_sensitive_attrs(df: pd.DataFrame, config: PipelineConfig, target_col: str) -> tuple[pd.DataFrame, tuple[str, ...]]:
    lower_map = {str(c).lower(): c for c in df.columns if c != target_col}

    def resolve_existing_column(name: str) -> str | None:
        if name in df.columns and name != target_col:
            return str(name)
        key = str(name).lower()
        if key in lower_map:
            return str(lower_map[key])
        return None

    if isinstance(config.sensitive_attrs, tuple):
        candidate = [col for x in config.sensitive_attrs if (col := resolve_existing_column(str(x)))]
        return _make_age_group_if_needed(df, candidate)
    if isinstance(config.sensitive_attrs, str) and config.sensitive_attrs.lower() not in {"auto", "none", ""}:
        resolved = resolve_existing_column(config.sensitive_attrs)
        candidate = [resolved] if resolved else []
        return _make_age_group_if_needed(df, candidate)
    if config.sensitive_attrs is None or str(config.sensitive_attrs).lower() == "none":
        return df, tuple()

    found: list[str] = []
    for cand in config.sensitive_candidates:
        key = str(cand).lower()
        if key in lower_map:
            found.append(str(lower_map[key]))
    # Also add columns with demographic-like names.
    demographic_tokens = [
        "gender", "sex", "race", "ethnic", "age_group", "skin", "insurance", "language",
        "marriage_status", "ever_married", "residence_type", "work_type",
    ]
    for col in df.columns:
        lc = str(col).lower()
        if col == target_col or col in found:
            continue
        if any(token in lc for token in demographic_tokens):
            found.append(str(col))
    if "age" in lower_map and "age_group" not in {str(x).lower() for x in found}:
        found.append(str(lower_map["age"]))
    out, found2 = _make_age_group_if_needed(df, found)
    return out, tuple(found2[:12])


def build_generic_analytic(raw_source: pd.DataFrame, config: PipelineConfig, csv_path: Path) -> dict[str, Any]:
    target_col = infer_target_col(raw_source, config)
    df = raw_source.copy()
    if config.target_col and config.target_col not in df.columns:
        raise ValueError(f"Configured target_col={config.target_col!r} was not found in the CSV.")
    encoded, positive = encode_binary_target(df[target_col], config.positive_target_value)
    df["target_original"] = df[target_col]
    df["target"] = encoded
    df, sensitive_attrs = infer_sensitive_attrs(df, config, target_col)
    # `target_original` preserves the source target; feature selection excludes both
    # `target` and the original target column later in `build_feature_matrix`.
    before = len(df)
    analytic = df.loc[df["target"].notna()].copy().reset_index(drop=True)
    endpoint = f"Generic binary classification target `{target_col}`; positive class = `{positive}`."
    eligibility = pd.DataFrame([
        {"rule": "Raw CSV rows", "n": int(before)},
        {"rule": "Excluded: missing target", "n": int(before - len(analytic))},
        {"rule": "Final analytic cohort", "n": int(len(analytic))},
        {"rule": "Positive target rows", "n": int(analytic["target"].sum())},
        {"rule": "Negative target rows", "n": int((1 - analytic["target"]).sum())},
    ])
    name = Path(csv_path).stem.lower() if str(config.dataset_name).lower() == "auto" else str(config.dataset_name)
    return {
        "source": raw_source,
        "raw": df,
        "analytic": analytic,
        "csv_path": csv_path,
        "dataset_mode": "generic_tabular",
        "dataset_name": name,
        "target_col": "target",
        "target_original_col": target_col,
        "positive_target_value": positive,
        "sensitive_attrs": tuple(sensitive_attrs),
        "feature_cols_seed": tuple(config.feature_cols or ()),
        "eligibility": eligibility,
        "endpoint_description": endpoint,
    }


def prepare_dataset(config: PipelineConfig) -> dict[str, Any]:
    source, csv_path = load_source_csv(config)
    mode = detect_dataset_mode(source, config)
    if mode == "clinical_survival":
        return build_clinical_example_analytic(source, config, csv_path)
    return build_generic_analytic(source, config, csv_path)


def logical_type(series: pd.Series, config: PipelineConfig) -> str:
    s = series.dropna()
    if s.empty:
        return "all_missing"
    if pd.api.types.is_bool_dtype(series) or s.nunique() == 2:
        return "binary"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        if s.nunique() <= config.numeric_as_categorical_max_unique:
            return "numeric_discrete"
        return "numeric_continuous"
    if s.nunique() <= config.max_categorical_levels:
        return "categorical"
    return "high_cardinality_categorical"


def variable_schema(df: pd.DataFrame, config: PipelineConfig, target_col: str | None = None, sensitive_attrs: tuple[str, ...] = (), feature_cols: tuple[str, ...] = ()) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    feature_set = set(feature_cols or ())
    sensitive_set = set(sensitive_attrs or ())
    for col in df.columns:
        s = df[col]
        role = "feature_candidate"
        reason = "eligible before automatic filtering"
        if col == target_col or col in {"target", "target_original"}:
            role = "target" if col in {target_col, "target"} else "target_original"
            reason = "outcome variable"
        elif col in sensitive_set:
            role = "sensitive_attribute"
            reason = "configured or automatically inferred sensitive attribute"
        elif col in feature_set:
            role = "selected_feature"
            reason = "selected by the automatic feature-selection step"
        elif col in CLINICAL_EXAMPLE_LEAKAGE_OR_NON_PREDICTOR:
            role = "excluded_leakage_or_non_predictor"
            reason = "dataset-specific outcome, follow-up, label, or non-predictor column"
        elif is_id_like(str(col), config.id_like_patterns):
            role = "excluded_id_like"
            reason = "identifier-like column name"
        missing = int(s.isna().sum())
        unique = int(s.nunique(dropna=True))
        sample_values = ", ".join(map(str, s.dropna().astype(str).unique()[:5]))
        rows.append({
            "column": col,
            "role": role,
            "reason": reason,
            "pandas_dtype": str(s.dtype),
            "logical_type": logical_type(s, config),
            "n_missing": missing,
            "missing_rate": missing / len(df) if len(df) else np.nan,
            "n_unique": unique,
            "example_values": sample_values,
        })
    return pd.DataFrame(rows)


def _target_association(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or "target" not in df.columns:
        return np.nan
    s = df[col]
    y = df["target"].astype(int)
    try:
        if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > 2:
            x = pd.to_numeric(s, errors="coerce")
            mask = x.notna() & y.notna()
            if mask.sum() < 5 or x[mask].nunique() < 2:
                return np.nan
            return float(abs(np.corrcoef(x[mask], y[mask])[0, 1]))
        tbl = pd.crosstab(s.fillna("Missing"), y)
        if min(tbl.shape) < 2:
            return np.nan
        chi2 = stats.chi2_contingency(tbl, correction=False)[0]
        n = tbl.to_numpy().sum()
        return float(np.sqrt(chi2 / (n * (min(tbl.shape) - 1)))) if n else np.nan
    except Exception:
        return np.nan


def _sensitive_proxy_score(df: pd.DataFrame, col: str, sensitive_attrs: tuple[str, ...]) -> float:
    scores: list[float] = []
    for attr in sensitive_attrs:
        if attr not in df.columns or attr == col:
            continue
        try:
            if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(df[attr]):
                x = pd.to_numeric(df[col], errors="coerce")
                a = pd.to_numeric(df[attr], errors="coerce")
                mask = x.notna() & a.notna()
                if mask.sum() >= 5 and x[mask].nunique() > 1 and a[mask].nunique() > 1:
                    scores.append(abs(float(np.corrcoef(x[mask], a[mask])[0, 1])))
            else:
                tbl = pd.crosstab(df[col].fillna("Missing"), df[attr].fillna("Missing"))
                if min(tbl.shape) >= 2:
                    chi2 = stats.chi2_contingency(tbl, correction=False)[0]
                    n = tbl.to_numpy().sum()
                    scores.append(float(np.sqrt(chi2 / (n * (min(tbl.shape) - 1)))) if n else np.nan)
        except Exception:
            continue
    return float(np.nanmax(scores)) if scores else np.nan


def auto_select_features(analytic: pd.DataFrame, prepared: dict[str, Any], config: PipelineConfig) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    sensitive_attrs = tuple(prepared["sensitive_attrs"])
    target_original = str(prepared.get("target_original_col", ""))
    explicit_features = tuple(config.feature_cols or ())
    if explicit_features:
        candidates = [c for c in explicit_features if c in analytic.columns]
    elif prepared["dataset_mode"] == "clinical_survival":
        candidates = [c for c in prepared["feature_cols_seed"] if c in analytic.columns]
    else:
        excluded = {"target", "target_original", target_original, *config.exclude_cols}
        if not config.include_sensitive_as_features:
            excluded |= set(sensitive_attrs)
        candidates = [c for c in analytic.columns if c not in excluded]

    rows: list[dict[str, Any]] = []
    selected: list[str] = []
    selected_numeric: list[str] = []
    for col in candidates:
        s = analytic[col]
        missing_rate = float(s.isna().mean()) if len(s) else np.nan
        n_unique = int(s.nunique(dropna=True))
        ltype = logical_type(s, config)
        action = "keep"
        reason = "eligible predictor"
        if col in config.exclude_cols:
            action, reason = "drop", "listed in exclude_cols"
        elif col in {"target", "target_original", target_original}:
            action, reason = "drop", "target or original target column"
        elif col in CLINICAL_EXAMPLE_LEAKAGE_OR_NON_PREDICTOR:
            action, reason = "drop", "dataset-specific outcome, follow-up, or non-predictor column"
        elif is_id_like(col, config.id_like_patterns):
            action, reason = "drop", "identifier-like column"
        elif (col in sensitive_attrs) and not config.include_sensitive_as_features:
            action, reason = "drop", "sensitive attribute excluded from predictors"
        elif missing_rate > config.max_missing_feature_frac:
            action, reason = "drop", f"missing_rate>{config.max_missing_feature_frac:.2f}"
        elif n_unique <= 1:
            action, reason = "drop", "constant or all-missing column"
        elif ltype == "high_cardinality_categorical" and n_unique > config.high_cardinality_threshold:
            action, reason = "drop", "high-cardinality categorical column"
        if action == "keep" and pd.api.types.is_numeric_dtype(s):
            # drop near duplicates among numeric predictors to avoid unstable visual heatmaps/model matrices
            x = pd.to_numeric(s, errors="coerce")
            for kept in selected_numeric:
                y = pd.to_numeric(analytic[kept], errors="coerce")
                mask = x.notna() & y.notna()
                if mask.sum() >= 10 and x[mask].nunique() > 1 and y[mask].nunique() > 1:
                    corr = abs(float(np.corrcoef(x[mask], y[mask])[0, 1]))
                    if corr >= config.correlation_drop_threshold:
                        action, reason = "drop", f"near-duplicate of `{kept}` (|r|={corr:.3f})"
                        break
        assoc = _target_association(analytic, col)
        proxy = _sensitive_proxy_score(analytic, col, sensitive_attrs)
        rows.append({
            "feature": col,
            "selected": action == "keep",
            "action": action,
            "reason": reason,
            "logical_type": ltype,
            "n_missing": int(s.isna().sum()),
            "missing_rate": missing_rate,
            "n_unique": n_unique,
            "target_association_strength": assoc,
            "max_sensitive_proxy_strength": proxy,
        })
        if action == "keep":
            selected.append(col)
            if pd.api.types.is_numeric_dtype(s):
                selected_numeric.append(col)
    if not selected:
        raise RuntimeError("Automatic feature selection selected zero predictors. Check feature_cols/exclude_cols and dataset schema.")
    feature_policy = pd.DataFrame(rows).sort_values(["selected", "target_association_strength"], ascending=[False, False]).reset_index(drop=True)
    schema = variable_schema(analytic, config, target_col="target", sensitive_attrs=sensitive_attrs, feature_cols=tuple(selected))
    return selected, schema, feature_policy


def build_dataset_overview(prepared: dict[str, Any], feature_cols: list[str]) -> pd.DataFrame:
    source = prepared["source"]
    raw = prepared["raw"]
    analytic = prepared["analytic"]
    rows: list[dict[str, Any]] = [
        {"section": "source", "metric": "csv_path", "value": str(prepared["csv_path"])},
        {"section": "source", "metric": "dataset_mode", "value": prepared["dataset_mode"]},
        {"section": "source", "metric": "raw_rows", "value": int(len(source))},
        {"section": "source", "metric": "source_raw_columns", "value": int(source.shape[1])},
        {"section": "source", "metric": "columns_after_derivation", "value": int(raw.shape[1])},
        {"section": "source", "metric": "missing_cells_in_source_csv", "value": int(source.isna().sum().sum())},
        {"section": "endpoint", "metric": "endpoint_definition", "value": prepared["endpoint_description"]},
        {"section": "cohort", "metric": "analytic_rows", "value": int(len(analytic))},
        {"section": "cohort", "metric": "analytic_events", "value": int(analytic["target"].sum())},
        {"section": "cohort", "metric": "analytic_non_events", "value": int((1 - analytic["target"]).sum())},
        {"section": "cohort", "metric": "analytic_event_rate", "value": round(float(analytic["target"].mean()), 6)},
        {"section": "features", "metric": "selected_predictor_count", "value": int(len(feature_cols))},
        {"section": "features", "metric": "selected_predictors", "value": ", ".join(feature_cols)},
        {"section": "fairness", "metric": "sensitive_attributes", "value": ", ".join(prepared["sensitive_attrs"])},
    ]
    if "pidnum" in raw.columns:
        rows.insert(4, {"section": "source", "metric": "unique_participants", "value": int(raw["pidnum"].nunique())})
    if prepared["dataset_mode"] == "clinical_survival":
        rows.extend([
            {"section": "endpoint", "metric": "raw_observed_events_all_followup", "value": int(raw["cid"].sum())},
            {"section": "endpoint", "metric": "raw_censored_all_followup", "value": int((1 - raw["cid"]).sum())},
            {"section": "follow_up", "metric": "median_time_raw_days", "value": round(float(raw["time"].median()), 3)},
            {"section": "follow_up", "metric": "median_time_analytic_days", "value": round(float(analytic["time"].median()), 3)},
        ])
    return pd.DataFrame(rows)


def sensitive_attribute_summary(analytic: pd.DataFrame, sensitive_attrs: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for attr in sensitive_attrs:
        if attr not in analytic.columns:
            continue
        for group, sub in analytic.groupby(attr, dropna=False):
            n = len(sub)
            events = int(sub["target"].sum())
            rows.append({
                "attribute": attr,
                "group": group,
                "n": int(n),
                "share": n / len(analytic) if len(analytic) else np.nan,
                "events": events,
                "non_events": int(n - events),
                "event_rate": events / n if n else np.nan,
            })
    return pd.DataFrame(rows).sort_values(["attribute", "group"]).reset_index(drop=True)


def dataset_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    duplicate_rows = int(df.duplicated().sum())
    return pd.DataFrame([
        {"metric": "rows", "value": int(df.shape[0])},
        {"metric": "columns", "value": int(df.shape[1])},
        {"metric": "missing_cells", "value": int(df.isna().sum().sum())},
        {"metric": "duplicate_rows", "value": duplicate_rows},
        {"metric": "duplicate_row_rate", "value": duplicate_rows / len(df) if len(df) else np.nan},
        {"metric": "memory_mb", "value": round(float(df.memory_usage(deep=True).sum() / 1024**2), 4)},
    ])


def numeric_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    numeric_cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return pd.DataFrame()
    out = df[numeric_cols].describe().T.reset_index().rename(columns={"index": "feature"})
    out["missing_rate"] = [df[c].isna().mean() for c in numeric_cols]
    return out


def categorical_summary(df: pd.DataFrame, cols: list[str], max_values: int = 5) -> pd.DataFrame:
    rows = []
    for col in cols:
        if col not in df.columns or pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique(dropna=True) > 12:
            continue
        counts = df[col].fillna("Missing").astype(str).value_counts(dropna=False)
        rows.append({
            "feature": col,
            "n_unique": int(df[col].nunique(dropna=True)),
            "missing_rate": float(df[col].isna().mean()),
            "top_values": "; ".join([f"{k}: {v}" for k, v in counts.head(max_values).items()]),
        })
    return pd.DataFrame(rows)


def _stratify_label(df: pd.DataFrame, sensitive_attrs: tuple[str, ...], preferred_attr: str | None) -> pd.Series:
    attr = None
    if preferred_attr and preferred_attr != "auto" and preferred_attr in df.columns:
        attr = preferred_attr
    elif sensitive_attrs:
        for a in sensitive_attrs:
            if a in df.columns and df[a].nunique(dropna=True) <= 20:
                attr = a
                break
    if attr:
        label = df["target"].astype(str) + "__" + df[attr].astype(str)
        counts = label.value_counts()
        if counts.min() >= 2:
            return label
    return df["target"].astype(str)


def split_data(analytic: pd.DataFrame, sensitive_attrs: tuple[str, ...], config: PipelineConfig) -> dict[str, pd.DataFrame]:
    strategy = str(getattr(config, "split_strategy", "random_stratified") or "random_stratified").lower()
    time_cols = tuple(getattr(config, "event_time_columns", ()) or ())
    if getattr(config, "enforce_temporal_for_event_time", True) and time_cols and strategy != "temporal":
        existing = [c for c in time_cols if c in analytic.columns]
        if existing:
            raise ValueError(f"Temporal split is required when event-time columns are configured: {existing}. Set split_strategy='temporal'.")
    if strategy == "temporal":
        time_col = getattr(config, "split_time_col", None) or (time_cols[0] if time_cols else None)
        if not time_col or time_col not in analytic.columns:
            raise ValueError("split_strategy='temporal' requires split_time_col present in analytic data.")
        ordered = analytic.sort_values(time_col).reset_index(drop=True)
        n = len(ordered)
        n_test = max(1, int(round(n * float(config.test_size))))
        n_train_full = max(2, n - n_test)
        train_full = ordered.iloc[:n_train_full].copy()
        test = ordered.iloc[n_train_full:].copy()
        n_val = max(1, int(round(len(train_full) * float(config.validation_size))))
        train = train_full.iloc[:-n_val].copy()
        validation = train_full.iloc[-n_val:].copy()
        return {"train": train.reset_index(drop=True), "validation": validation.reset_index(drop=True), "train_full": train_full.reset_index(drop=True), "test": test.reset_index(drop=True)}

    if strategy == "group":
        gid = getattr(config, "split_group_id_col", None)
        if not gid or gid not in analytic.columns:
            raise ValueError("split_strategy='group' requires split_group_id_col present in analytic data.")
        gss = GroupShuffleSplit(n_splits=1, test_size=config.test_size, random_state=config.random_state)
        train_idx, test_idx = next(gss.split(analytic, groups=analytic[gid]))
        train_full, test = analytic.iloc[train_idx].copy(), analytic.iloc[test_idx].copy()
        gss2 = GroupShuffleSplit(n_splits=1, test_size=config.validation_size, random_state=config.random_state+1)
        tr_idx, va_idx = next(gss2.split(train_full, groups=train_full[gid]))
        train, validation = train_full.iloc[tr_idx].copy(), train_full.iloc[va_idx].copy()
        return {"train": train.reset_index(drop=True), "validation": validation.reset_index(drop=True), "train_full": train_full.reset_index(drop=True), "test": test.reset_index(drop=True)}
    stratify = _stratify_label(analytic, sensitive_attrs, config.stratify_by)
    train_full, test = train_test_split(
        analytic,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify,
    )
    stratify_train = _stratify_label(train_full, sensitive_attrs, config.stratify_by)
    train, validation = train_test_split(
        train_full,
        test_size=config.validation_size,
        random_state=config.random_state + 1,
        stratify=stratify_train,
    )
    return {"train": train.reset_index(drop=True), "validation": validation.reset_index(drop=True), "train_full": train_full.reset_index(drop=True), "test": test.reset_index(drop=True)}


def split_summary(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name in ["train", "validation", "train_full", "test"]:
        df = splits[name]
        events = int(df["target"].sum())
        rows.append({"split": name, "n": int(len(df)), "events": events, "non_events": int(len(df) - events), "event_rate": events / len(df) if len(df) else np.nan})
    return pd.DataFrame(rows)
