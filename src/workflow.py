from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any
import pickle

import numpy as np
import pandas as pd

from . import __version__
from .config import PipelineConfig, config_from_user_config, resolve_project_paths
from .data import (
    auto_select_features,
    build_dataset_overview,
    categorical_summary,
    dataset_quality_summary,
    detect_dataset_mode,
    load_source_csv,
    numeric_summary,
    prepare_dataset,
    sensitive_attribute_summary,
    split_data,
    split_summary,
    variable_schema,
)
from .evaluation import (
    all_threshold_sweeps,
    evaluate_all_models,
    fairness_by_group,
    fairness_gap_summary,
    merge_model_performance_fairness_summary,
    mitigation_method_catalogue,
    mitigation_combination_summary,
    mitigation_statistical_evidence_table,
    run_threshold_mitigation,
    select_primary_attribute,
)
from .modeling import predict_model, train_models
from .reporting import (
    build_manifest,
    configure_plot_style,
    export_results_archive,
    plot_categorical_distributions,
    plot_confusion_matrices,
    plot_correlation_heatmap,
    plot_dataset_flow,
    plot_event_rates_by_sensitive,
    plot_fairness_gap_heatmap,
    plot_fairness_gaps,
    plot_feature_target_associations,
    plot_missingness,
    plot_mitigation_group_rates,
    plot_mitigation_heatmap,
    plot_mitigation_combination_summary,
    plot_mitigation_summary,
    plot_model_performance_summary,
    plot_numeric_distributions,
    plot_probability_histograms,
    plot_proxy_heatmap,
    plot_roc_curves,
    plot_target_distribution,
    plot_test_performance,
    plot_threshold_selection_all_models,
    plot_validation_model_comparison,
    plot_variable_type_counts,
    save_table,
    show_markdown,
    show_table,
)
from .runtime import fairness_support_status, set_reproducibility


@dataclass
class WorkflowContext:
    config: PipelineConfig
    paths: dict[str, Any]
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    figures: dict[str, Path] = field(default_factory=dict)
    prepared: dict[str, Any] = field(default_factory=dict)
    source: pd.DataFrame | None = None
    raw: pd.DataFrame | None = None
    analytic: pd.DataFrame | None = None
    feature_cols: list[str] = field(default_factory=list)
    sensitive_attrs: tuple[str, ...] = field(default_factory=tuple)
    intersectional_attrs: tuple[str, ...] = field(default_factory=tuple)
    splits: dict[str, pd.DataFrame] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)
    validation_probs: dict[str, np.ndarray] = field(default_factory=dict)
    test_probs: dict[str, np.ndarray] = field(default_factory=dict)
    champion_model: str | None = None
    primary_attribute: str | None = None
    results_zip_path: Path | None = None


def _config_snapshot(config: PipelineConfig, support: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {"setting": key, "value": str(value)}
        for key, value in sorted(config.__dict__.items())
        if key not in {"threshold_grid"}
    ]
    rows.extend({"setting": key, "value": str(value)} for key, value in support.items())
    return pd.DataFrame(rows)


def initialise_eda(user_config: dict | None = None) -> WorkflowContext:
    config = config_from_user_config(user_config)
    set_reproducibility(config.random_state)
    configure_plot_style()
    paths = resolve_project_paths(config)
    support = fairness_support_status(config)
    show_markdown("# Generic tabular fairness pipeline — modular v18")
    show_markdown(
        f"**Package version:** `{__version__}`  \n"
        f"**Runtime environment:** `{paths['environment_label']}`  \n"
        f"**Execution preset:** `{config.execution_preset}`  \n"
        f"**Output directory:** `{paths['root']}`  \n"
        f"**Fairness-library status:** `{support['fairness_support_mode']}`  \n"
        f"**Fairlearn available:** `{support['fairlearn_available']}`  \n"
        f"**AIF360 available:** `{support['aif360_available']}`  \n"
        f"**Imbalanced-learn available:** `{support.get('imbalanced_learn_available', False)}`  \n"
        f"**GPU detected:** `{support.get('gpu_available', False)}` — `{support.get('gpu_name', 'not detected')}`"
    )
    config_snapshot = _config_snapshot(config, support)
    acceleration_df = pd.DataFrame([
        {"item": "gpu_available", "value": support.get("gpu_available", False)},
        {"item": "gpu_name", "value": support.get("gpu_name", "not detected")},
        {"item": "xgboost_device_policy", "value": support.get("xgboost_device_policy", "safe_cpu")},
        {"item": "xgboost_effective_device", "value": support.get("xgboost_effective_device", "cpu")},
        {"item": "gpu_training_note", "value": support.get("gpu_acceleration_note", "GPU availability is reported; XGBoost uses safe_cpu by default inside sklearn pipelines.")},
        {"item": "mitigation_refits", "value": "Refit-based mitigation reuses the estimator configuration; XGBoost prediction is aligned to CPU for sklearn CPU matrices."},
    ])
    save_table(config_snapshot, paths, "00_runtime_and_input_configuration.csv")
    save_table(acceleration_df, paths, "00b_acceleration_status.csv")
    show_table("Runtime and input configuration", config_snapshot, max_rows=config.max_table_rows_display)
    show_table("Acceleration status", acceleration_df, max_rows=config.max_table_rows_display)
    if not support["fairlearn_available"] or not support["aif360_available"] or not support.get("imbalanced_learn_available", False):
        show_markdown(
            "The first notebook cell has attempted to install Fairlearn, AIF360 and imbalanced-learn. Because at least one optional package is unavailable, "
            "the workflow keeps running with built-in support operations: subgroup metrics, support reweighing/oversampling where possible, validation-learned threshold post-processing, McNemar tests, and bootstrap confidence intervals."
        )
    return WorkflowContext(config=config, paths=paths, tables={"runtime_and_input_configuration": config_snapshot, "acceleration_status": acceleration_df})


def initialise_pipeline(user_config: dict | None = None) -> WorkflowContext:
    config = config_from_user_config(user_config)
    set_reproducibility(config.random_state)
    configure_plot_style()
    paths = resolve_project_paths(config)
    support = fairness_support_status(config)
    show_markdown("# Configured fairness-analysis workflow")
    show_markdown(
        f"**Package version:** `{__version__}`  \n"
        f"**Runtime environment:** `{paths['environment_label']}`  \n"
        f"**Execution preset:** `{config.execution_preset}`  \n"
        f"**Output directory:** `{paths['root']}`  \n"
        f"**Enabled models:** `{', '.join(config.enabled_models or ())}`  \n"
        f"**Mitigation methods:** `{', '.join(config.mitigation_methods or ())}`  \n"
        f"**Mitigation scope:** `{config.mitigation_scope}`  \n"
        f"**Bootstrap repetitions for mitigation significance:** `{config.bootstrap_reps}`  \n"
        f"**Fairness support mode:** `{support['fairness_support_mode']}`  \n"
        f"**GPU detected:** `{support.get('gpu_available', False)}` — `{support.get('gpu_name', 'not detected')}`"
    )
    config_snapshot = _config_snapshot(config, support)
    acceleration_df = pd.DataFrame([
        {"item": "gpu_available", "value": support.get("gpu_available", False)},
        {"item": "gpu_name", "value": support.get("gpu_name", "not detected")},
        {"item": "gpu_backend", "value": support.get("gpu_backend", "none_detected")},
        {"item": "xgboost_device_policy", "value": support.get("xgboost_device_policy", "safe_cpu")},
        {"item": "xgboost_effective_device", "value": support.get("xgboost_effective_device", "cpu")},
        {"item": "gpu_training_note", "value": support.get("gpu_acceleration_note", "GPU availability is reported; XGBoost uses safe_cpu by default inside sklearn pipelines.")},
        {"item": "mitigation_refits", "value": "Refit-based mitigation reuses the estimator configuration; XGBoost prediction is aligned to CPU for sklearn CPU matrices."},
    ])
    save_table(config_snapshot, paths, "06_fairness_pipeline_configuration.csv")
    save_table(acceleration_df, paths, "06b_acceleration_status.csv")
    show_table("Fairness pipeline configuration", config_snapshot, max_rows=config.max_table_rows_display)
    show_table("Acceleration status", acceleration_df, max_rows=config.max_table_rows_display)
    return WorkflowContext(config=config, paths=paths, tables={"fairness_pipeline_configuration": config_snapshot, "acceleration_status": acceleration_df})


def stage_1_raw_dataset_eda(ctx: WorkflowContext) -> WorkflowContext:
    show_markdown("## 2. Raw CSV extraction and exploratory data analysis")
    source, csv_path = load_source_csv(ctx.config)
    detected_mode = detect_dataset_mode(source, ctx.config)
    ctx.source = source
    raw_schema_df = variable_schema(source, ctx.config)
    quality_df = dataset_quality_summary(source)
    numeric_df = numeric_summary(source, list(source.columns))
    categorical_df = categorical_summary(source, list(source.columns), max_values=8)
    mode_df = pd.DataFrame([
        {"item": "csv_path", "value": str(csv_path)},
        {"item": "detected_dataset_mode", "value": detected_mode},
        {"item": "rows", "value": int(source.shape[0])},
        {"item": "columns", "value": int(source.shape[1])},
    ])
    ctx.tables.update({
        "raw_dataset_overview": mode_df,
        "raw_variable_schema": raw_schema_df,
        "raw_data_quality": quality_df,
        "raw_numeric_summary": numeric_df,
        "raw_categorical_summary": categorical_df,
    })
    save_table(mode_df, ctx.paths, "01_raw_dataset_overview.csv")
    save_table(raw_schema_df, ctx.paths, "02_raw_variable_schema.csv")
    save_table(quality_df, ctx.paths, "03_raw_data_quality_summary.csv")
    save_table(numeric_df, ctx.paths, "04_raw_numeric_summary.csv")
    save_table(categorical_df, ctx.paths, "05_raw_categorical_summary.csv")

    show_markdown("I have first examined the raw CSV without requiring the final target, sensitive attributes, or feature list. Use these tables and figures to decide whether the next configuration cell should override the automatic choices.")
    show_table("Raw dataset overview", mode_df)
    show_table("Raw variable schema", raw_schema_df, max_rows=ctx.config.max_table_rows_display)
    show_table("Raw data quality", quality_df)
    show_table("Raw numeric summary", numeric_df, max_rows=ctx.config.max_table_rows_display)
    show_table("Raw categorical/discrete summary", categorical_df, max_rows=ctx.config.max_table_rows_display)

    ctx.figures["raw_variable_types"] = plot_variable_type_counts(raw_schema_df, ctx.paths, ctx.config.display_figures)
    ctx.figures["raw_missingness"] = plot_missingness(raw_schema_df, ctx.paths, ctx.config.display_figures)
    ctx.figures["raw_numeric_distributions"] = plot_numeric_distributions(source, numeric_df, ctx.paths, ctx.config.display_figures)
    ctx.figures["raw_categorical_distributions"] = plot_categorical_distributions(source, categorical_df, ctx.paths, ctx.config.display_figures)
    ctx.figures["raw_correlation_heatmap"] = plot_correlation_heatmap(source, list(source.columns), ctx.paths, ctx.config.display_figures)
    show_markdown("### Configuration checkpoint\nReview the EDA above. Then edit the next `USER_CONFIG` cell to define or override the target, sensitive attributes, exclusions, feature policy, and runtime preset.")
    return ctx


def stage_2_configured_dataset_preparation(ctx: WorkflowContext) -> WorkflowContext:
    show_markdown("## 4. Configured dataset preparation and automatic feature selection")
    prepared = prepare_dataset(ctx.config)
    feature_cols, schema_df, feature_policy_df = auto_select_features(prepared["analytic"], prepared, ctx.config)
    overview_df = build_dataset_overview(prepared, feature_cols)
    sensitive_df = sensitive_attribute_summary(prepared["analytic"], tuple(prepared["sensitive_attrs"]))
    splits = split_data(prepared["analytic"], tuple(prepared["sensitive_attrs"]), ctx.config)
    split_df = split_summary(splits)

    ctx.prepared = prepared
    ctx.source = prepared["source"]
    ctx.raw = prepared["raw"]
    ctx.analytic = prepared["analytic"]
    ctx.feature_cols = list(feature_cols)
    ctx.sensitive_attrs = tuple(prepared["sensitive_attrs"])
    ctx.splits = splits
    ctx.tables.update({
        "automatic_feature_selection_policy": feature_policy_df,
        "dataset_overview": overview_df,
        "cohort_or_target_flow": prepared["eligibility"],
        "sensitive_attribute_summary": sensitive_df,
        "split_summary": split_df,
    })
    save_table(feature_policy_df, ctx.paths, "07_automatic_feature_selection_policy.csv")
    save_table(overview_df, ctx.paths, "08_configured_dataset_overview.csv")
    save_table(prepared["eligibility"], ctx.paths, "09_cohort_or_target_flow.csv")
    save_table(sensitive_df, ctx.paths, "10_sensitive_attribute_summary.csv")
    save_table(split_df, ctx.paths, "11_train_validation_test_split_summary.csv")

    show_markdown(f"I have detected dataset mode `{prepared['dataset_mode']}` and internal target column `target`. The original target source is `{prepared.get('target_original_col')}`.")
    show_markdown(f"**Automatically selected predictors:** `{', '.join(feature_cols)}`")
    show_markdown(f"**Sensitive attributes used for fairness auditing:** `{', '.join(ctx.sensitive_attrs) if ctx.sensitive_attrs else 'none detected'}`")
    show_table("Automatic feature-selection policy", feature_policy_df, max_rows=ctx.config.max_table_rows_display)
    show_table("Configured dataset overview", overview_df)
    show_table("Cohort / target eligibility flow", prepared["eligibility"])
    show_table("Sensitive attribute summary", sensitive_df, max_rows=ctx.config.max_table_rows_display)
    show_table("Train / validation / test split summary", split_df)

    ctx.figures["dataset_flow"] = plot_dataset_flow(prepared["eligibility"], ctx.paths, ctx.config.display_figures)
    ctx.figures["target_distribution"] = plot_target_distribution(prepared["analytic"], ctx.paths, ctx.config.display_figures)
    ctx.figures["event_rate_by_sensitive"] = plot_event_rates_by_sensitive(sensitive_df, ctx.paths, ctx.config.display_figures)
    ctx.figures["feature_target_associations"] = plot_feature_target_associations(feature_policy_df, ctx.paths, ctx.config.display_figures)
    ctx.figures["proxy_screen"] = plot_proxy_heatmap(feature_policy_df, ctx.paths, ctx.config.display_figures)
    return ctx


def stage_3_train_models(ctx: WorkflowContext) -> WorkflowContext:
    show_markdown("## 5. Model training, parameter selection and rationale")
    if not ctx.splits:
        raise RuntimeError("Run stage_2_configured_dataset_preparation before model training.")
    models, model_config_df, validation_results_df = train_models(ctx.splits["train"], ctx.splits["validation"], ctx.config, ctx.feature_cols)
    ctx.models = models
    ctx.tables.update({"model_training_configuration": model_config_df, "validation_model_results": validation_results_df})
    save_table(model_config_df, ctx.paths, "12_model_training_configuration_and_rationale.csv")
    model_artifact_rows: list[dict[str, Any]] = []
    for model_name, trained in models.items():
        artifact_path = Path(ctx.paths["model_artifacts"]) / f"{model_name}.pkl"
        with artifact_path.open("wb") as f:
            pickle.dump(trained, f)
        model_artifact_rows.append({"model": model_name, "artifact_path": str(artifact_path), "round": "baseline_training"})
    model_artifacts_df = pd.DataFrame(model_artifact_rows).sort_values("model").reset_index(drop=True)
    save_table(model_artifacts_df, ctx.paths, "12b_model_artifact_registry.csv")
    ctx.tables["model_artifact_registry"] = model_artifacts_df
    show_table("Model training configuration and selected parameters", model_config_df, max_rows=ctx.config.max_table_rows_display)
    show_table("Saved trained-model artifacts", model_artifacts_df, max_rows=ctx.config.max_table_rows_display)
    show_markdown(f"Mitigation-ready folder for second-round upgrades: `{ctx.paths['mitigation_artifacts']}`")
    show_markdown("Validation metrics are stored internally and are merged with held-out test performance and fairness gaps after the fairness-audit stage.")
    return ctx


def stage_4_baseline_visual_evaluation(ctx: WorkflowContext) -> WorkflowContext:
    show_markdown("## 6. Held-out baseline evaluation and visual diagnostics")
    if not ctx.models:
        raise RuntimeError("Run stage_3_train_models before baseline evaluation.")
    test_performance_df, test_probs = evaluate_all_models(ctx.models, ctx.splits["test"], ctx.config)
    validation_probs = {name: predict_model(model, ctx.splits["validation"], ctx.config) for name, model in ctx.models.items()}
    validation_threshold_sweep_df = all_threshold_sweeps(ctx.models, ctx.splits["validation"], validation_probs, ctx.config)
    selected_thresholds = {name: model.threshold for name, model in ctx.models.items()}
    validation_threshold_sweep_df["split"] = "validation"
    validation_threshold_sweep_df["selected_threshold"] = validation_threshold_sweep_df["model"].map(selected_thresholds).astype(float)
    validation_threshold_sweep_df["is_selected_threshold"] = np.isclose(validation_threshold_sweep_df["threshold"].astype(float), validation_threshold_sweep_df["selected_threshold"].astype(float))
    ctx.test_probs = test_probs
    ctx.validation_probs = validation_probs
    ctx.champion_model = str(test_performance_df.sort_values("balanced_accuracy", ascending=False).iloc[0]["model"])
    ctx.tables.update({"test_performance_all_models": test_performance_df, "validation_threshold_sweep": validation_threshold_sweep_df})
    save_table(validation_threshold_sweep_df, ctx.paths, "13_validation_threshold_selection_sweep_all_models.csv")
    show_markdown(f"**Selected model for primary interpretation:** `{ctx.champion_model}` based on held-out balanced accuracy.")
    show_markdown("The detailed validation, test and baseline fairness metrics are intentionally reported later as one merged table to avoid duplicated displays.")
    show_table("Validation decision-threshold selection sweep", validation_threshold_sweep_df, max_rows=min(ctx.config.max_table_rows_display, 120))
    ctx.figures["roc_curves"] = plot_roc_curves(ctx.splits["test"], test_probs, ctx.paths, ctx.config.display_figures)
    ctx.figures["confusion_matrices"] = plot_confusion_matrices(ctx.models, ctx.splits["test"], test_probs, ctx.paths, ctx.config.display_figures)
    ctx.figures["probability_histograms"] = plot_probability_histograms(ctx.splits["test"], test_probs, ctx.paths, ctx.config.display_figures)
    ctx.figures["threshold_selection_all_models"] = plot_threshold_selection_all_models(validation_threshold_sweep_df, ctx.paths, ctx.config.display_figures)
    return ctx


def _intersection_name(*attrs: str) -> str:
    return "__x__".join(str(a) for a in attrs)


def _normalise_custom_intersection_entry(entry: Any) -> tuple[str, ...] | None:
    """Return a tuple of source attribute names from a user-specified intersection entry."""
    if entry is None:
        return None
    if isinstance(entry, str):
        cleaned = entry.strip()
        if not cleaned:
            return None
        if "__x__" in cleaned:
            parts = tuple(part.strip() for part in cleaned.split("__x__") if part.strip())
            return parts if len(parts) >= 2 else None
        if "+" in cleaned:
            parts = tuple(part.strip() for part in cleaned.split("+") if part.strip())
            return parts if len(parts) >= 2 else None
        return None
    if isinstance(entry, (tuple, list)):
        parts = tuple(str(part).strip() for part in entry if str(part).strip())
        return parts if len(parts) >= 2 else None
    return None


def _create_intersectional_attribute(ctx: WorkflowContext, attrs: tuple[str, ...]) -> str | None:
    """Create one intersectional attribute across all splits if support is manageable."""
    attrs = tuple(dict.fromkeys(str(a) for a in attrs if a))
    if len(attrs) < 2:
        return None
    if not ctx.splits:
        return None
    if any(attr not in ctx.splits["test"].columns for attr in attrs):
        return None
    name = _intersection_name(*attrs)
    if name in ctx.splits["test"].columns:
        if name not in ctx.intersectional_attrs:
            ctx.intersectional_attrs = tuple(list(ctx.intersectional_attrs) + [name])
        return name
    max_groups = int(getattr(ctx.config, "mitigation_max_intersectional_groups", None) or getattr(ctx.config, "mitigation_max_pair_groups", 40) or 40)
    test_values = ctx.splits["test"][list(attrs)].fillna("Missing").astype(str).agg(" | ".join, axis=1)
    if int(test_values.nunique(dropna=False)) > max_groups:
        return None
    for split_name, split_df in list(ctx.splits.items()):
        if all(attr in split_df.columns for attr in attrs):
            ctx.splits[split_name] = split_df.copy()
            ctx.splits[split_name][name] = split_df[list(attrs)].fillna("Missing").astype(str).agg(" | ".join, axis=1)
    if name not in ctx.intersectional_attrs:
        ctx.intersectional_attrs = tuple(list(ctx.intersectional_attrs) + [name])
    return name


def _ensure_intersectional_attributes(ctx: WorkflowContext) -> tuple[str, ...]:
    """Generate configured intersectional attributes, usually pairwise combinations."""
    if ctx.intersectional_attrs:
        return ctx.intersectional_attrs
    if len(ctx.sensitive_attrs) < 2 or not ctx.splits:
        ctx.intersectional_attrs = tuple()
        return ctx.intersectional_attrs
    orders = getattr(ctx.config, "mitigation_intersectional_orders", (2,)) or (2,)
    try:
        orders = tuple(sorted({int(o) for o in orders if 2 <= int(o) <= 4})) or (2,)
    except TypeError:
        orders = (2,)
    created: list[str] = []
    for order in orders:
        if order < 2 or order > len(ctx.sensitive_attrs):
            continue
        for attrs in combinations(ctx.sensitive_attrs, order):
            name = _create_intersectional_attribute(ctx, tuple(attrs))
            if name and name not in created:
                created.append(name)
    ctx.intersectional_attrs = tuple(created)
    return ctx.intersectional_attrs


def _custom_mitigation_intersection_targets(ctx: WorkflowContext, custom_attrs: tuple[str, ...]) -> tuple[str, ...]:
    """Resolve explicit custom intersectional mitigation targets.

    This makes it possible to mitigate race/drugs/gender/homo jointly without enabling
    all possible sensitive-attribute pairs such as age×race unless the analyst asks for them.
    """
    setting = getattr(ctx.config, "mitigation_custom_intersections", None)
    if setting in {None, False, "none", "None", ""}:
        return tuple()
    source_combos: list[tuple[str, ...]] = []
    if isinstance(setting, str):
        setting_l = setting.lower()
        if setting_l in {"all_pairs_among_custom_attributes", "all_pairs", "auto_pairs"}:
            orders = (2,)
            for order in orders:
                if order <= len(custom_attrs):
                    for attrs in combinations(custom_attrs, order):
                        source_combos.append(tuple(attrs))
        elif setting_l in {
            "all_intersections_among_custom_attributes",
            "all_combinations_among_custom_attributes",
            "all_groups_among_custom_attributes",
            "all_custom_intersections",
            "auto",
        }:
            configured_orders = getattr(ctx.config, "mitigation_intersectional_orders", (2, 3, 4)) or (2, 3, 4)
            try:
                orders = tuple(sorted({int(o) for o in configured_orders if 2 <= int(o) <= 4})) or (2,)
            except Exception:
                orders = (2, 3, 4)
            for order in orders:
                if order <= len(custom_attrs):
                    for attrs in combinations(custom_attrs, order):
                        source_combos.append(tuple(attrs))
        else:
            attrs = _normalise_custom_intersection_entry(setting)
            if attrs:
                source_combos.append(attrs)
    else:
        entries = setting if isinstance(setting, (tuple, list)) else (setting,)
        for entry in entries:
            attrs = _normalise_custom_intersection_entry(entry)
            if attrs:
                source_combos.append(attrs)
    created: list[str] = []
    allowed_sources = set(ctx.sensitive_attrs) | set(ctx.splits.get("test", pd.DataFrame()).columns)
    for attrs in source_combos:
        if not all(attr in allowed_sources for attr in attrs):
            continue
        name = _create_intersectional_attribute(ctx, attrs)
        if name and name not in created:
            created.append(name)
    max_inter = getattr(ctx.config, "mitigation_max_intersectional_attributes", None)
    if max_inter is not None:
        try:
            max_inter_i = max(0, int(max_inter))
            if len(created) > max_inter_i:
                created = list(_rank_attributes_by_baseline_gap(ctx, tuple(created))[:max_inter_i])
        except Exception:
            pass
    return tuple(created)


def _fairness_audit_attributes(ctx: WorkflowContext) -> tuple[str, ...]:
    scope = str(getattr(ctx.config, "fairness_audit_scope", "single") or "single").lower()
    if scope in {"single_plus_pairs", "all", "single_plus_intersectional"}:
        return tuple(ctx.sensitive_attrs) + _ensure_intersectional_attributes(ctx)
    return tuple(ctx.sensitive_attrs)


def stage_5_fairness_audit(ctx: WorkflowContext) -> WorkflowContext:
    show_markdown("## 7. Fairness audit across configured sensitive attributes")
    if not ctx.test_probs:
        raise RuntimeError("Run stage_4_baseline_visual_evaluation before fairness audit.")
    audit_attrs = _fairness_audit_attributes(ctx)
    if not audit_attrs:
        empty = pd.DataFrame()
        ctx.tables.update({"baseline_fairness_by_group": empty, "baseline_fairness_gaps": empty, "model_performance_fairness_summary": empty})
        show_markdown("No sensitive attributes were configured or inferred, so fairness auditing has been skipped.")
        return ctx
    fairness_frames: list[pd.DataFrame] = []
    for model_name, model in ctx.models.items():
        fairness_frames.append(fairness_by_group(ctx.splits["test"], ctx.test_probs[model_name], model.threshold, audit_attrs, model_name, "baseline"))
    fairness_group_df = pd.concat(fairness_frames, ignore_index=True) if fairness_frames else pd.DataFrame()
    gap_df = fairness_gap_summary(fairness_group_df)
    primary_attr = select_primary_attribute(gap_df, ctx.champion_model or "") if not gap_df.empty else None
    ctx.primary_attribute = primary_attr
    merged_summary_df = merge_model_performance_fairness_summary(ctx.tables.get("validation_model_results", pd.DataFrame()), ctx.tables.get("test_performance_all_models", pd.DataFrame()), gap_df)
    ctx.tables.update({"baseline_fairness_by_group": fairness_group_df, "baseline_fairness_gaps": gap_df, "model_performance_fairness_summary": merged_summary_df})
    save_table(merged_summary_df, ctx.paths, "14_model_performance_and_fairness_summary.csv")
    save_table(fairness_group_df, ctx.paths, "15_baseline_fairness_by_group_all_models.csv")
    if primary_attr:
        show_markdown(f"**Primary fairness attribute:** `{primary_attr}` based on the largest combined FPR+FNR gap for the selected model.")
    show_markdown(f"**Fairness audit scope:** `{getattr(ctx.config, 'fairness_audit_scope', 'single')}`. Attributes audited: `{', '.join(audit_attrs)}`")
    show_table("Merged model performance and baseline fairness summary", merged_summary_df, max_rows=ctx.config.max_table_rows_display)
    show_markdown("Detailed subgroup fairness rates are exported as `15_baseline_fairness_by_group_all_models.csv`; the main visible result is the merged model-performance/fairness summary to avoid duplicate result blocks.")
    ctx.figures["model_performance_summary"] = plot_model_performance_summary(merged_summary_df, ctx.paths, ctx.config.display_figures)
    ctx.figures["fairness_gap_heatmap"] = plot_fairness_gap_heatmap(gap_df, ctx.paths, ctx.config.display_figures)
    return ctx


def _canonical_mitigation_scope(scope: str) -> str:
    scope = str(scope or "all_individual").lower()
    aliases = {
        "primary_only": "primary",
        "all": "all_single",
        "all_individual": "all_single",
        "all_sensitive": "all_single",
        "top_gap": "top_k_by_gap",
        "top_k_gap_attributes": "top_k_by_gap",
        "all_pairs": "intersectional_pairs",
        "primary_plus_intersections": "primary_plus_pairs",
        "all_individual_and_pairs": "single_plus_intersectional",
        "all_individual_and_combinations": "single_plus_intersectional",
        "all_sensitive_and_pairs": "single_plus_intersectional",
        "all_sensitive_and_combinations": "single_plus_intersectional",
        "all_individual_and_intersections": "single_plus_intersectional",
        "all_sensitive_and_intersections": "single_plus_intersectional",
        "all_individual_and_groups": "single_plus_intersectional",
        "single_plus_groups": "single_plus_intersectional",
        "custom": "configured",
        "custom_individual": "configured",
        "custom_plus_pairs": "configured_plus_intersectional",
        "custom_plus_combinations": "configured_plus_intersectional",
        "configured_plus_pairs": "configured_plus_intersectional",
        "configured_plus_combinations": "configured_plus_intersectional",
        "selected_individuals_and_pairs": "configured_plus_intersectional",
        "configured_and_pairs": "configured_plus_intersectional",
        "custom_plus_intersections": "configured_plus_intersectional",
        "configured_plus_intersections": "configured_plus_intersectional",
        "custom_plus_groups": "configured_plus_intersectional",
        "configured_plus_groups": "configured_plus_intersectional",
    }
    return aliases.get(scope, scope)



def _rank_attributes_by_baseline_gap(ctx: WorkflowContext, attrs: tuple[str, ...]) -> tuple[str, ...]:
    """Rank candidate attributes by observed baseline combined FPR+FNR gap.

    When an attribute was not part of the baseline audit, keep its original order after
    the audited/high-gap attributes. This is useful for runtime caps because it keeps
    the most informative intersections first without silently dropping all intersections.
    """
    if not attrs:
        return tuple()
    gap_df = ctx.tables.get("baseline_fairness_gaps", pd.DataFrame()).copy()
    gap_lookup: dict[str, float] = {}
    if not gap_df.empty and {"attribute", "combined_fpr_fnr_gap"}.issubset(gap_df.columns):
        if ctx.champion_model and "model" in gap_df.columns:
            gap_df = gap_df[gap_df["model"].astype(str) == str(ctx.champion_model)]
        gap_lookup = dict(zip(gap_df["attribute"].astype(str), gap_df["combined_fpr_fnr_gap"].astype(float)))
    indexed = list(enumerate(attrs))
    indexed.sort(key=lambda x: (gap_lookup.get(str(x[1]), -1.0), -x[0]), reverse=True)
    return tuple(attr for _, attr in indexed)


def _apply_mitigation_runtime_caps(ctx: WorkflowContext, attrs: tuple[str, ...]) -> tuple[str, ...]:
    """Apply safe runtime caps to mitigation attributes.

    The v18 configuration can request all pairs, triples and four-way intersections; runtime guards remain configurable and are set high by default in the corrected thesis notebook.
    With seven sensitive attributes this can create 90+ intersectional targets before
    multiplying by models and methods. This helper preserves single sensitive
    attributes first and caps intersectional targets when the estimated job count would
    be excessive.
    """
    attrs = tuple(dict.fromkeys(str(a) for a in attrs if a))
    if not attrs:
        return attrs
    single_set = set(ctx.sensitive_attrs or ())
    singles = tuple(a for a in attrs if a in single_set)
    intersections = tuple(a for a in attrs if a not in single_set)

    # First cap the raw number of intersectional attributes, if requested.
    max_inter = getattr(ctx.config, "mitigation_max_intersectional_attributes", None)
    if max_inter is not None:
        try:
            max_inter_i = max(0, int(max_inter))
        except Exception:
            max_inter_i = 0
        if max_inter_i >= 0 and len(intersections) > max_inter_i:
            intersections = _rank_attributes_by_baseline_gap(ctx, intersections)[:max_inter_i]

    # Then cap by estimated mitigation jobs: models × attributes × methods.
    max_jobs = getattr(ctx.config, "mitigation_max_jobs", None)
    if max_jobs is not None:
        try:
            max_jobs_i = int(max_jobs)
        except Exception:
            max_jobs_i = 0
        if max_jobs_i > 0:
            try:
                from .evaluation import selected_mitigation_methods
                n_methods = max(1, len(selected_mitigation_methods(ctx.config)))
            except Exception:
                n_methods = max(1, len(getattr(ctx.config, "mitigation_methods", ()) or (1,)))
            n_models = max(1, len(ctx.models or {}))
            max_attrs = max(1, max_jobs_i // max(1, n_models * n_methods))
            if len(singles) >= max_attrs:
                # Keep all configured single sensitive attributes; this may exceed the
                # job cap, but it avoids dropping requested clinically/socially relevant
                # attributes. Drop intersections first.
                intersections = tuple()
            else:
                inter_slots = max_attrs - len(singles)
                if len(intersections) > inter_slots:
                    intersections = _rank_attributes_by_baseline_gap(ctx, intersections)[:inter_slots]
    return tuple(dict.fromkeys(singles + intersections))

def _resolve_mitigation_attributes(ctx: WorkflowContext) -> tuple[str, ...]:
    scope = str(getattr(ctx.config, "mitigation_scope", "all_single") or "all_single").lower()
    aliases = {
        "primary_only": "primary",
        "all": "all_single",
        "all_individual": "all_single",
        "top_gap": "top_k_by_gap",
        "top_k_gap_attributes": "top_k_by_gap",
        "all_pairs": "intersectional_pairs",
        "primary_plus_intersections": "primary_plus_pairs",
        "all_individual_and_pairs": "single_plus_intersectional",
        "all_individual_and_combinations": "single_plus_intersectional",
        "all_individual_and_intersections": "single_plus_intersectional",
        "all_individual_and_groups": "single_plus_intersectional",
        "single_plus_pairs": "single_plus_intersectional",
        "single_plus_groups": "single_plus_intersectional",
        "custom": "configured",
        "custom_individual": "configured",
        "configured_individual": "configured",
        "custom_plus_pairs": "configured_plus_intersectional",
        "custom_plus_combinations": "configured_plus_intersectional",
        "configured_plus_pairs": "configured_plus_intersectional",
        "configured_plus_combinations": "configured_plus_intersectional",
        "selected_individuals_and_pairs": "configured_plus_intersectional",
        "selected_plus_pairs": "configured_plus_intersectional",
        "configured_and_pairs": "configured_plus_intersectional",
        "custom_plus_intersections": "configured_plus_intersectional",
        "configured_plus_intersections": "configured_plus_intersectional",
        "custom_plus_groups": "configured_plus_intersectional",
        "configured_plus_groups": "configured_plus_intersectional",
    }
    scope = aliases.get(scope, scope)
    single = tuple(ctx.sensitive_attrs)
    inter = tuple(ctx.intersectional_attrs or ())
    if scope in {"intersectional_pairs", "primary_plus_pairs", "single_plus_intersectional"}:
        inter = _ensure_intersectional_attributes(ctx)
    if scope == "primary":
        return (ctx.primary_attribute,) if ctx.primary_attribute else tuple(single[:1])
    if scope == "all_single":
        return single
    if scope == "intersectional_pairs":
        return inter
    if scope == "single_plus_intersectional":
        return single + inter
    if scope == "primary_plus_pairs":
        primary = ctx.primary_attribute or (single[0] if single else None)
        if not primary:
            return tuple()
        selected_pairs = tuple(x for x in inter if x.startswith(primary + "__x__") or x.endswith("__x__" + primary))
        return (primary,) + selected_pairs
    if scope == "top_k_by_gap":
        gap_df = ctx.tables.get("baseline_fairness_gaps", pd.DataFrame()).copy()
        if gap_df.empty:
            return single[: int(getattr(ctx.config, "mitigation_top_k_attributes", 2) or 2)]
        k = int(getattr(ctx.config, "mitigation_top_k_attributes", 2) or 2)
        if ctx.champion_model:
            gap_df = gap_df[gap_df["model"].astype(str) == str(ctx.champion_model)]
        ranked = gap_df.sort_values("combined_fpr_fnr_gap", ascending=False)["attribute"].astype(str).tolist()
        allowed = set(single) | set(inter)
        out: list[str] = []
        for attr in ranked:
            if attr in allowed and attr not in out:
                out.append(attr)
            if len(out) >= k:
                break
        return tuple(out)

    # Configured/custom mode: keep analyst-selected singles and explicitly requested intersections.
    # The analyst can write "gender_label", "gender_label__x__homo_label",
    # "gender_label+homo_label", or simply ("gender_label", "homo_label").
    allowed_columns = set(single) | set(ctx.splits.get("test", pd.DataFrame()).columns)

    def add_custom_item(item: Any, out: list[str]) -> None:
        if isinstance(item, str) and item in allowed_columns:
            if item not in out:
                out.append(item)
            return
        attrs = _normalise_custom_intersection_entry(item)
        if attrs:
            name = _create_intersectional_attribute(ctx, attrs)
            if name and name not in out:
                out.append(name)

    custom_list: list[str] = []
    for item in (ctx.config.mitigation_custom_attributes or ()):  # names or pairs
        add_custom_item(item, custom_list)

    if not custom_list:
        setting = ctx.config.mitigation_attributes
        if isinstance(setting, str):
            if setting.lower() == "all":
                custom_list.extend(single)
            elif setting.lower() == "primary":
                custom_list.extend((ctx.primary_attribute,) if ctx.primary_attribute else tuple(single[:1]))
            elif setting in allowed_columns:
                custom_list.append(setting)
            else:
                add_custom_item(setting, custom_list)
        else:
            for item in setting:
                add_custom_item(item, custom_list)

    custom = tuple(dict.fromkeys(x for x in custom_list if x))
    if scope == "configured_plus_intersectional":
        custom_singles = tuple(x for x in custom if x in set(single))
        custom_intersections = _custom_mitigation_intersection_targets(ctx, custom_singles)
        return tuple(dict.fromkeys(custom + custom_intersections))
    return custom



def _mitigation_attribute_selection_table(ctx: WorkflowContext, attrs: tuple[str, ...]) -> pd.DataFrame:
    """Explain exactly why each attribute is included in the mitigation stage."""
    requested_scope = str(getattr(ctx.config, "mitigation_scope", "all_individual") or "all_individual")
    canonical_scope = _canonical_mitigation_scope(requested_scope)
    single = tuple(ctx.sensitive_attrs)
    inter = tuple(ctx.intersectional_attrs or ())
    gap_df = ctx.tables.get("baseline_fairness_gaps", pd.DataFrame()).copy()
    if not gap_df.empty and ctx.champion_model and "model" in gap_df.columns:
        gap_df = gap_df[gap_df["model"].astype(str) == str(ctx.champion_model)]
    gap_lookup = {}
    if not gap_df.empty and {"attribute", "combined_fpr_fnr_gap"}.issubset(gap_df.columns):
        gap_lookup = dict(zip(gap_df["attribute"].astype(str), gap_df["combined_fpr_fnr_gap"].astype(float)))
    rows = []
    for attr in attrs:
        attr_str = str(attr)
        if attr_str in single:
            attr_family = "single sensitive attribute"
            intersection_order = 1
        elif attr_str in inter or "__x__" in attr_str:
            intersection_order = attr_str.count("__x__") + 1
            attr_family = f"{intersection_order}-way intersectional combination"
        else:
            attr_family = "configured/generated attribute"
            intersection_order = np.nan
        if canonical_scope == "top_k_by_gap":
            reason = f"Selected because it is among the top {getattr(ctx.config, 'mitigation_top_k_attributes', 2)} baseline FPR+FNR gaps for the selected model."
        elif canonical_scope == "all_single":
            reason = "Selected because all individual sensitive attributes are mitigated."
        elif canonical_scope == "single_plus_intersectional":
            reason = f"Selected because all individual sensitive attributes and generated intersectional combinations of orders {getattr(ctx.config, 'mitigation_intersectional_orders', (2,))} are mitigated."
        elif canonical_scope == "configured_plus_intersectional":
            reason = f"Selected by analyst configuration, including custom intersectional combinations of orders {getattr(ctx.config, 'mitigation_intersectional_orders', (2,))}."
        elif canonical_scope == "primary":
            reason = "Selected because it is the primary largest-gap attribute."
        elif canonical_scope == "primary_plus_pairs":
            reason = "Selected because it is the primary attribute or a combination involving it."
        elif canonical_scope == "intersectional_pairs":
            reason = "Selected because pairwise-intersection mitigation is enabled."
        else:
            reason = "Selected by explicit analyst configuration."
        rows.append({
            "attribute": attr_str,
            "attribute_family": attr_family,
            "requested_scope": requested_scope,
            "canonical_scope": canonical_scope,
            "intersection_order": intersection_order,
            "baseline_combined_fpr_fnr_gap_for_selected_model": gap_lookup.get(attr_str, np.nan),
            "reason_for_inclusion": reason,
        })
    return pd.DataFrame(rows)

def stage_6_mitigation_and_significance(ctx: WorkflowContext) -> WorkflowContext:
    show_markdown("## 8. Mitigation families, combinations and statistical significance")
    if not ctx.test_probs or not ctx.validation_probs:
        raise RuntimeError("Run stage_4_baseline_visual_evaluation before mitigation.")
    raw_attrs = _resolve_mitigation_attributes(ctx)
    attrs = _apply_mitigation_runtime_caps(ctx, raw_attrs)
    if len(attrs) < len(raw_attrs):
        try:
            from .evaluation import selected_mitigation_methods
            n_methods = len(selected_mitigation_methods(ctx.config))
        except Exception:
            n_methods = len(getattr(ctx.config, "mitigation_methods", ()) or ())
        estimated_before = len(ctx.models) * len(raw_attrs) * max(1, n_methods)
        estimated_after = len(ctx.models) * len(attrs) * max(1, n_methods)
        show_markdown(
            f"**Runtime guard applied:** mitigation attributes were reduced from `{len(raw_attrs)}` to `{len(attrs)}` "
            f"to avoid an excessive mitigation loop (`{estimated_before}` → `{estimated_after}` estimated jobs). "
            "Single sensitive attributes are prioritised; intersectional attributes are ranked by observed baseline gap when available. "
            "Increase `mitigation_max_jobs` or set it to `None` for a final exhaustive run."
        )
    if not attrs:
        show_markdown("No mitigation attributes are available. Mitigation has been skipped.")
        return ctx
    support = fairness_support_status(ctx.config)
    scope_options_df = pd.DataFrame([
        {"attribute_scope": "all_individual", "meaning": "all configured or inferred sensitive attributes separately", "best_use": "recommended default; includes gender, race, homo/drugs and age when available"},
        {"attribute_scope": "top_k_gap_attributes", "meaning": "only the k attributes with largest baseline FPR+FNR gaps", "best_use": "runtime-focused sensitivity analysis; may exclude gender/homo if their baseline gaps are not top-k"},
        {"attribute_scope": "primary_only", "meaning": "largest-gap single attribute only", "best_use": "very fast first pass and small datasets"},
        {"attribute_scope": "primary_plus_intersections", "meaning": "primary attribute plus pairwise intersections involving it", "best_use": "intersectionality check without excessive group explosion"},
        {"attribute_scope": "all_individual_and_combinations", "meaning": "all single attributes and generated intersections controlled by mitigation_intersectional_orders", "best_use": "final intersectional analysis when subgroup support is sufficient; can include pairs, triples and four-way groups"},
        {"attribute_scope": "custom_plus_combinations", "meaning": "explicit custom single attributes plus selected/custom intersections controlled by mitigation_intersectional_orders", "best_use": "focused intersectional analysis, e.g. selected race/drugs/gender/homo/ever_married/Residence_type/work_type combinations"},
        {"attribute_scope": "custom", "meaning": "explicit single attributes from mitigation_custom_attributes", "best_use": "clinically or regulatorily prioritised subgroup analysis"},
    ])
    method_catalogue_df = mitigation_method_catalogue(ctx.config)
    selected_attrs_df = _mitigation_attribute_selection_table(ctx, attrs)
    save_table(scope_options_df, ctx.paths, "16_mitigation_attribute_scope_options.csv")
    save_table(selected_attrs_df, ctx.paths, "16a_mitigation_selected_attributes.csv")
    save_table(method_catalogue_df, ctx.paths, "17_mitigation_method_catalogue.csv")
    show_table("Mitigation attribute-scope options", scope_options_df, max_rows=ctx.config.max_table_rows_display)
    show_table("Selected attributes for mitigation", selected_attrs_df, max_rows=ctx.config.max_table_rows_display)
    show_table("Mitigation method catalogue by family", method_catalogue_df, max_rows=ctx.config.max_table_rows_display)
    ctx.tables["mitigation_attribute_scope_options"] = scope_options_df
    ctx.tables["mitigation_selected_attributes"] = selected_attrs_df
    ctx.tables["mitigation_method_catalogue"] = method_catalogue_df
    show_markdown("The mitigation stage separates methods into pre-processing, intra-processing and post-processing families. External Fairlearn/AIF360 packages are attempted first by the installation cell; when an external method is unavailable or fails, the workflow records the failure and keeps the analysis running with support operations. Known non-actionable Fairlearn/Pandas dtype FutureWarnings are handled inside the package so the progress output remains readable.")
    show_markdown(f"**External fairness-library status:** `{support['fairness_support_mode']}`")
    show_markdown(f"**GPU detected:** `{support.get('gpu_available', False)}` — `{support.get('gpu_name', 'not detected')}`")
    show_markdown(f"**Mitigation methods evaluated:** `{', '.join(ctx.config.mitigation_methods or ())}`")
    show_markdown(f"**Mitigation attributes evaluated:** `{', '.join(attrs)}`")
    show_markdown(f"**Intersectional mitigation orders:** `{getattr(ctx.config, 'mitigation_intersectional_orders', (2,))}`; maximum groups per generated intersection: `{getattr(ctx.config, 'mitigation_max_intersectional_groups', getattr(ctx.config, 'mitigation_max_pair_groups', 40))}`")
    if _canonical_mitigation_scope(getattr(ctx.config, "mitigation_scope", "all_individual")) == "top_k_by_gap":
        show_markdown("**Interpretation note:** this top-k scope intentionally limits mitigation to the largest observed baseline gaps. Use `all_individual` to include gender/homo/drugs/race/age separately, or `custom_plus_combinations` / `all_individual_and_combinations` with `mitigation_intersectional_orders=(2,3,4)` to include pairs, triples and four-way groups.")
    show_markdown("In v18, the corrected analyst configuration includes gender, race, homo, drugs, ever_married, Residence_type and work_type when present. With `mitigation_scope='custom_plus_pairs'`, it generates the requested 2-, 3- and 4-way intersectional groups among the selected variables when subgroup support is sufficient. The previous top-k mode remains available for runtime-constrained exploratory runs.")
    mitigation_group_df, mitigation_summary_df, mitigation_thresholds_df = run_threshold_mitigation(
        models=ctx.models,
        train_df=ctx.splits["train"],
        validation_df=ctx.splits["validation"],
        test_df=ctx.splits["test"],
        validation_probs=ctx.validation_probs,
        test_probs=ctx.test_probs,
        config=ctx.config,
        attributes=attrs,
    )
    progress_log_df = getattr(run_threshold_mitigation, "last_progress_log", pd.DataFrame())
    combination_summary_df = mitigation_combination_summary(mitigation_summary_df, alpha=float(ctx.config.significance_alpha))
    statistical_evidence_df = mitigation_statistical_evidence_table(mitigation_summary_df, alpha=float(ctx.config.significance_alpha))
    ctx.tables.update({
        "mitigation_progress_log": progress_log_df,
        "mitigation_group_comparison": mitigation_group_df,
        "mitigation_summary_significance": mitigation_summary_df,
        "best_mitigation_model_combinations": combination_summary_df,
        "mitigation_statistical_evidence_by_metric": statistical_evidence_df,
        "mitigation_thresholds": mitigation_thresholds_df,
    })
    save_table(progress_log_df, ctx.paths, "18a_mitigation_retraining_progress_log.csv")
    save_table(mitigation_group_df, ctx.paths, "18_mitigation_group_fpr_fnr_original_vs_mitigated.csv")
    save_table(mitigation_summary_df, ctx.paths, "19_mitigation_model_summary_with_significance.csv")
    save_table(combination_summary_df, ctx.paths, "20_best_mitigation_model_combinations.csv")
    save_table(statistical_evidence_df, ctx.paths, "20b_mitigation_statistical_evidence_by_metric.csv")
    save_table(mitigation_thresholds_df, ctx.paths, "21_mitigation_group_thresholds.csv")
    show_table("Mitigation retraining/progress log", progress_log_df, max_rows=ctx.config.max_table_rows_display)
    show_table("Best mitigation/model combinations for final selection", combination_summary_df, max_rows=ctx.config.max_table_rows_display)
    show_table("Mitigation statistical evidence by metric", statistical_evidence_df, max_rows=ctx.config.max_table_rows_display)
    show_table("Mitigation model-level summary with bootstrap significance tests", mitigation_summary_df, max_rows=ctx.config.max_table_rows_display)
    show_table("Mitigation group comparison: original and mitigated FPR/FNR", mitigation_group_df, max_rows=ctx.config.max_table_rows_display)
    show_markdown("Group-specific mitigation thresholds are exported as `21_mitigation_group_thresholds.csv` and are not repeated on screen unless audit reproduction requires them.")
    ctx.figures["mitigation_summary"] = plot_mitigation_summary(mitigation_summary_df, ctx.paths, ctx.config.display_figures)
    ctx.figures["mitigation_group_rates"] = plot_mitigation_group_rates(mitigation_group_df, ctx.paths, ctx.config.display_figures)
    ctx.figures["mitigation_heatmap"] = plot_mitigation_heatmap(mitigation_summary_df, ctx.paths, ctx.config.display_figures)
    ctx.figures["mitigation_combination_summary"] = plot_mitigation_combination_summary(combination_summary_df, ctx.paths, ctx.config.display_figures)
    return ctx

def stage_7_finalise(ctx: WorkflowContext) -> WorkflowContext:
    show_markdown("## 9. Final artifact registry and performance recommendations")
    support = fairness_support_status(ctx.config)
    final_selection = pd.DataFrame([
        {"decision": "dataset_mode", "selected": ctx.prepared.get("dataset_mode", "unknown"), "reason": "Detected from CSV columns unless explicitly configured."},
        {"decision": "endpoint_or_target", "selected": ctx.prepared.get("endpoint_description", "unknown"), "reason": "Derived from the configured clinical-survival rules or generic target configuration."},
        {"decision": "feature_selection", "selected": f"{len(ctx.feature_cols)} predictors", "reason": "Automatic schema-driven feature selection excluded targets, sensitive attributes, IDs, leakage columns, high-missingness variables, and constants."},
        {"decision": "model", "selected": ctx.champion_model, "reason": "Highest held-out balanced accuracy among trained models."},
        {"decision": "primary_fairness_attribute", "selected": ctx.primary_attribute, "reason": "Largest combined FPR+FNR gap for the selected model, when sensitive attributes are available."},
        {"decision": "mitigation_scope", "selected": ctx.config.mitigation_scope, "reason": "Controls whether mitigation is applied to all single sensitive attributes, top-k gaps, the primary attribute, or intersectional combinations. The v18 corrected default prioritises gender, race, homo, drugs, ever_married, Residence_type and work_type when present, and adds 2-, 3- and 4-way combinations when subgroup support is sufficient. Runtime guards are explicit and set high in the corrected thesis configuration so requested groups are not silently dropped."},
        {"decision": "mitigation_methods", "selected": ", ".join(ctx.config.mitigation_methods or ()), "reason": "External methods are attempted when available; built-in support methods remain the robust fallback."},
        {"decision": "gpu_acceleration", "selected": f"gpu_available={support.get('gpu_available', False)}; xgboost_device_policy={support.get('xgboost_device_policy', 'safe_cpu')}; effective_xgboost_device={support.get('xgboost_effective_device', 'cpu')}", "reason": "GPU availability is checked in every runtime. XGBoost defaults to safe_cpu in sklearn pipelines to avoid CUDA/CPU prediction-device mismatch; CUDA can be forced explicitly."},
    ])
    recommendations = pd.DataFrame([
        {"area": "Model search", "recommendation": "After a smoke run, compare quick and standard presets; include at least one calibrated linear model, one non-linear ensemble, and optionally XGBoost. Keep xgboost_device_policy='safe_cpu' unless GPU-compatible inputs are intentionally managed.", "rationale": "Different model classes may occupy different fairness-performance trade-off regions; safe CPU prediction avoids XGBoost device-mismatch warnings in sklearn pipelines."},
        {"area": "Thresholding", "recommendation": "Use the validation threshold-selection figure rather than accepting a default 0.5 operating point.", "rationale": "For imbalanced clinical endpoints, 0.5 is often clinically suboptimal; selected thresholds should be justified by validation performance."},
        {"area": "Mitigation selection", "recommendation": "Use table 20 for the ranked model × mitigation choice and table 20b for bootstrap evidence on FPR-gap, FNR-gap, combined-gap and balanced-accuracy changes.", "rationale": "A mitigation is useful only if the fairness improvement is achieved without an unacceptable loss of balanced accuracy and is supported by uncertainty analysis."},
        {"area": "Mitigation scope", "recommendation": "Run mitigation_scope='custom_plus_pairs' for focused mitigation over gender/race/homo/drugs/ever_married/Residence_type/work_type; use 'all_individual' for all single sensitive attributes only, or 'custom_plus_combinations' with mitigation_custom_intersections='all_intersections_among_custom_attributes' for a final exhaustive intersectionality check when subgroup support and runtime permit.", "rationale": "Evaluating the selected individual attributes prevents clinically relevant groups from being silently skipped; selected intersections allow gender, homo, race, and drug-use combinations to be tested without uncontrolled group explosion."},
        {"area": "Feature policy", "recommendation": "Audit high proxy-risk predictors and rerun with selected manual exclusions if a variable is both low-signal and strongly associated with sensitive attributes.", "rationale": "Proxy variables can preserve discrimination even when sensitive attributes are excluded from training."},
        {"area": "Class imbalance", "recommendation": "Compare class_weight='balanced', threshold tuning, reweighing, RandomOverSampler, SMOTE/SMOTENC/SMOTEN-style methods and SMOTEENN only inside the validation-controlled pipeline.", "rationale": "Clinical endpoints are often rare, and pure accuracy can hide under-detection of positive cases."},
        {"area": "Data quality", "recommendation": "Improve labels, reduce missingness in clinically relevant predictors, and collect more observations in under-represented sensitive groups where possible.", "rationale": "Fairness mitigation cannot fully compensate for weak signal, noisy labels, or very small subgroup event counts."},
        {"area": "Robustness", "recommendation": "Repeat the standard preset with several random seeds and, if possible, evaluate on an external dataset or temporal hold-out.", "rationale": "Fairness conclusions can be sensitive to split composition and subgroup event counts."},
    ])
    save_table(final_selection, ctx.paths, "22_final_decision_registry.csv")
    save_table(recommendations, ctx.paths, "23_accuracy_and_mitigation_recommendations.csv")
    manifest = build_manifest(ctx.paths)
    save_table(manifest, ctx.paths, "24_artifact_manifest.csv")
    ctx.tables.update({"final_decision_registry": final_selection, "accuracy_and_mitigation_recommendations": recommendations, "artifact_manifest": manifest})
    show_table("Final decision registry", final_selection)
    show_table("Accuracy and mitigation recommendations", recommendations, max_rows=ctx.config.max_table_rows_display)
    show_table("Artifact manifest", manifest, max_rows=ctx.config.max_table_rows_display)
    show_markdown(f"**All outputs have been exported to:** `{ctx.paths['root']}`")
    return ctx


def stage_8_export_results_zip(ctx: WorkflowContext) -> WorkflowContext:
    show_markdown("## 10. Export all generated results as a ZIP")
    zip_path = export_results_archive(ctx.paths, display=True)
    ctx.results_zip_path = zip_path
    ctx.tables["results_zip"] = pd.DataFrame([{"zip_path": str(zip_path)}])
    show_markdown("Use the generated ZIP link to download all exported tables and figures from the current server/runtime to the local computer.")
    return ctx


def stage_8_export_results_archive(ctx: WorkflowContext) -> WorkflowContext:
    return stage_8_export_results_zip(ctx)


def run_full_pipeline(user_config: dict | None = None) -> WorkflowContext:
    ctx = initialise_pipeline(user_config)
    ctx = stage_2_configured_dataset_preparation(ctx)
    ctx = stage_3_train_models(ctx)
    ctx = stage_4_baseline_visual_evaluation(ctx)
    ctx = stage_5_fairness_audit(ctx)
    ctx = stage_6_mitigation_and_significance(ctx)
    ctx = stage_7_finalise(ctx)
    ctx = stage_8_export_results_zip(ctx)
    return ctx
