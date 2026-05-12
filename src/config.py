from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import re

import numpy as np


CLINICAL_EXAMPLE_BASELINE_FEATURES = (
    "age", "wtkg", "hemo", "karnof", "oprior", "z30", "zprior",
    "preanti", "symptom", "cd40", "cd80", "trt", "strat",
)
CLINICAL_EXAMPLE_LANDMARK_FEATURES = CLINICAL_EXAMPLE_BASELINE_FEATURES + ("cd420", "cd820")
CLINICAL_EXAMPLE_DEFAULT_SENSITIVE_ATTRS = (
    "age_group", "gender_label", "race_label", "homo_label", "drugs_label",
)

COMMON_TARGET_CANDIDATES = (
    "target", "outcome", "label", "class", "y", "event", "stroke", "mortality",
    "death", "disease", "diagnosis", "readmission", "default", "approved",
)
COMMON_SENSITIVE_CANDIDATES = (
    "age_group", "gender_label", "sex_label", "race_label", "ethnicity_label",
    "gender", "sex", "race", "ethnicity", "skin_tone", "skin_type",
    "homo_label", "drugs_label", "homo", "drugs", "insurance", "language",
    "ever_married", "marriage_status", "residence_type", "Residence_type", "work_type",
    "location", "site", "hospital", "income_group", "socioeconomic_status",
)
COMMON_EXCLUDE_PATTERNS = (
    r"(^|_)id($|_)", r"identifier", r"patient", r"subject", r"record", r"pidnum",
    r"timestamp", r"date", r"datetime", r"uuid",
)
# Methods are intentionally named by mitigation family.
#   pre-processing: modifies sample weights or the training distribution before the model is fitted.
#   intra-processing: changes the training objective or model-selection criterion.
#   post-processing: changes the decision rule after the base model is fitted.
PIPELINE_VERSION = "v18.0-1000bootstrap-custom-intersections"

DEFAULT_MITIGATION_METHODS = (
    "preprocess_reweighing",
    "preprocess_random_oversampling",
    "preprocess_smote",
    "preprocess_smoten",
    "preprocess_smotenc",
    "preprocess_smoteenn",
    "inprocess_fairlearn_expgrad_demographic_parity",
    "inprocess_fairlearn_expgrad_equalized_odds",
    "inprocess_support_fairness_aware_threshold_search",
    "postprocess_group_threshold_equalized_odds",
    "postprocess_group_threshold_equal_opportunity",
    "postprocess_group_threshold_demographic_parity",
    "postprocess_group_threshold_balanced_accuracy",
    "postprocess_fairlearn_threshold_demographic_parity",
    "postprocess_fairlearn_threshold_equalized_odds",
)


@dataclass
class PipelineConfig:
    """Runtime, dataset, modelling, fairness and reporting configuration."""

    # Runtime and IO
    environment: str = "auto"  # auto, local, colab, kaggle
    execution_preset: str = "quick"  # smoke, quick, standard, full
    output_base_dir: str | None = None
    random_state: int = 42
    n_jobs: int | None = None
    use_gpu: str | bool = "auto"  # auto/true/false; GPU availability is reported in every runtime.
    xgboost_device_policy: str = "safe_cpu"  # safe_cpu avoids CPU/GPU prediction mismatch in sklearn pipelines; use cuda_if_available to force CUDA.
    require_fairness_libraries_first: bool = True

    # Dataset handling
    dataset_mode: str = "auto"  # auto, clinical_survival, generic_tabular
    dataset_name: str = "clinical_example"
    csv_path: str | None = "example_clinical_dataset.csv"
    csv_candidates: tuple[str, ...] = ("example_clinical_dataset.csv", "dataset.csv", "data.csv", "clinical_dataset.csv", "source_data.csv")
    csv_read_kwargs: dict[str, Any] = field(default_factory=dict)

    # Generic target and sensitive-attribute inference
    target_col: str | None = None
    positive_target_value: Any = 1
    target_candidates: tuple[str, ...] = COMMON_TARGET_CANDIDATES
    sensitive_attrs: tuple[str, ...] | str | None = "auto"
    sensitive_candidates: tuple[str, ...] = COMMON_SENSITIVE_CANDIDATES

    # Clinical-survival example cohort and endpoint
    scenario: str = "baseline"  # baseline or landmark20 for the clinical-survival example; custom for generic
    horizon_days: int = 365
    landmark_day: int = 140
    endpoint_label: str = "AIDS/death event by day 365"

    # Automatic variable and feature selection
    feature_cols: tuple[str, ...] | list[str] | None = None
    exclude_cols: tuple[str, ...] | list[str] = ()
    include_sensitive_as_features: bool = False
    feature_selection_mode: str = "auto"  # auto, all_eligible, custom
    max_missing_feature_frac: float = 0.40
    max_categorical_levels: int = 30
    numeric_as_categorical_max_unique: int = 12
    high_cardinality_threshold: int = 60
    correlation_drop_threshold: float = 0.995
    id_like_patterns: tuple[str, ...] = COMMON_EXCLUDE_PATTERNS

    # Data split
    test_size: float = 0.20
    validation_size: float = 0.20
    stratify_by: str | None = "auto"

    # Models and search
    enabled_models: tuple[str, ...] | list[str] | None = None
    scoring_metric: str = "roc_auc"
    cv_folds: int | None = None
    max_grid_per_model: int | None = None
    threshold_grid: tuple[float, ...] = field(default_factory=lambda: tuple(float(x) for x in np.round(np.linspace(0.01, 0.80, 80), 4)))

    # Fairness and mitigation
    fairness_audit_scope: str = "single"  # single, single_plus_pairs, single_plus_intersectional
    mitigation_scope: str = "all_individual"  # primary_only, all_individual, top_k_gap_attributes, primary_plus_intersections, all_individual_and_pairs, all_individual_and_intersections, custom_plus_pairs
    mitigation_attributes: str | tuple[str, ...] = "all"  # backwards-compatible configured mode; "all" means all single sensitive attributes
    mitigation_top_k_attributes: int = 2  # used only when mitigation_scope=top_k_by_gap
    mitigation_max_pair_groups: int = 40  # legacy name; applies to any generated intersection, not only pairs
    mitigation_max_intersectional_groups: int | None = None  # clearer alias; defaults to mitigation_max_pair_groups
    mitigation_max_jobs: int | None = None  # runtime guard: cap model × attribute × method mitigation jobs; singles are prioritised
    mitigation_max_intersectional_attributes: int | None = None  # runtime guard: cap generated/custom intersectional attributes
    mitigation_custom_attributes: tuple[Any, ...] | list[Any] | None = None  # accepts names, "a__x__b" strings, or ("a", "b") pairs
    mitigation_custom_pair_base_attrs: tuple[str, ...] | list[str] | None = None  # optional legacy alias; custom_intersections is preferred
    # Optional explicit intersectional mitigation targets. Accepted forms:
    #   None -> no custom intersections unless the scope asks for all intersections
    #   "all_pairs_among_custom_attributes" -> all pairwise intersections among mitigation_custom_attributes
    #   "all_intersections_among_custom_attributes" -> all 2-, 3- and 4-way intersections requested through mitigation_intersectional_orders
    #   (("race_label", "gender_label"), ("drugs_label", "homo_label", "gender_label")) -> selected intersections
    #   ("race_label__x__gender_label",) -> selected generated-column name
    mitigation_custom_intersections: tuple[Any, ...] | list[Any] | str | None = "all_intersections_among_custom_attributes"
    mitigation_intersectional_orders: tuple[int, ...] | list[int] = (2, 3, 4)
    # Backwards-compatible alias for notebooks that used the longer guard name.
    mitigation_max_generated_intersectional_attributes: int | None = None
    mitigation_method: str = "postprocess_group_threshold_equalized_odds"  # retained for backwards compatibility
    mitigation_methods: tuple[str, ...] | list[str] | str | None = None
    mitigation_families: tuple[str, ...] | list[str] | str = ("preprocessing", "intraprocessing", "postprocessing")
    mitigation_backend: str = "auto"  # auto, support, fairlearn, aif360
    mitigation_objective: str | None = None  # optional shortcut: equalized_odds, equal_opportunity, demographic_parity, balanced_accuracy
    min_group_size_for_mitigation: int = 20
    mitigation_max_accuracy_drop: float | None = 0.05
    max_balanced_accuracy_drop: float | None = None  # backwards-compatible alias
    bootstrap_reps: int | None = None
    significance_alpha: float = 0.05

    # Reporting controls
    display_figures: bool = True
    show_tables: bool = True
    show_model_training_explanations: bool = True
    export_tables: bool = True
    export_figures: bool = True
    max_table_rows_display: int = 80
    auto_download_results_zip: bool = False
    show_mitigation_progress: bool = True
    mitigation_progress_detail: str = "standard"  # none, standard, verbose


def preset_defaults(preset: str) -> dict[str, Any]:
    preset = str(preset or "quick").lower()
    defaults = {
        "smoke": {
            "enabled_models": ("LogisticRegression",),
            "cv_folds": 2,
            "max_grid_per_model": 2,
            "bootstrap_reps": 20,
            "n_jobs": 1,
            "threshold_grid": tuple(float(x) for x in np.round(np.linspace(0.05, 0.75, 21), 4)),
            "mitigation_methods": ("postprocess_group_threshold_equalized_odds",),
            "mitigation_max_jobs": 40,
            "mitigation_max_intersectional_attributes": 25,
        },
        "quick": {
            "enabled_models": ("LogisticRegression", "RandomForest"),
            "cv_folds": 2,
            "max_grid_per_model": 3,
            "bootstrap_reps": 50,
            "n_jobs": 1,
            "threshold_grid": tuple(float(x) for x in np.round(np.linspace(0.03, 0.80, 31), 4)),
            "mitigation_methods": (
                "preprocess_reweighing",
                "postprocess_group_threshold_equalized_odds",
                "postprocess_group_threshold_equal_opportunity",
                "postprocess_group_threshold_demographic_parity",
            ),
            "mitigation_max_jobs": 120,
            "mitigation_max_intersectional_attributes": 40,
        },
        "standard": {
            "enabled_models": ("LogisticRegression", "ElasticNetLogistic", "RandomForest", "ExtraTrees", "HistGradientBoosting"),
            "cv_folds": 5,
            "max_grid_per_model": 8,
            "bootstrap_reps": 300,
            "n_jobs": 1,
            "threshold_grid": tuple(float(x) for x in np.round(np.linspace(0.01, 0.80, 51), 4)),
            "mitigation_methods": DEFAULT_MITIGATION_METHODS,
            "mitigation_max_jobs": 600,
            "mitigation_max_intersectional_attributes": 80,
        },
        "full": {
            "enabled_models": ("LogisticRegression", "ElasticNetLogistic", "RandomForest", "ExtraTrees", "HistGradientBoosting", "SVC_RBF", "XGBoost"),
            "cv_folds": 5,
            "max_grid_per_model": 16,
            "bootstrap_reps": 1000,
            "n_jobs": 1,
            "threshold_grid": tuple(float(x) for x in np.round(np.linspace(0.01, 0.80, 80), 4)),
            "mitigation_methods": DEFAULT_MITIGATION_METHODS,
            "mitigation_max_jobs": 1200,
            "mitigation_max_intersectional_attributes": 120,
        },
    }
    return defaults.get(preset, defaults["quick"])


def _flatten_user_config(user_config: dict | None) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in (user_config or {}).items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    return flat


def config_from_user_config(user_config: dict | None = None) -> PipelineConfig:
    cfg = PipelineConfig()
    for key, value in _flatten_user_config(user_config).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    cfg.execution_preset = str(cfg.execution_preset).lower()
    defaults = preset_defaults(cfg.execution_preset)
    cfg.xgboost_device_policy = str(getattr(cfg, "xgboost_device_policy", "safe_cpu") or "safe_cpu").lower()
    if cfg.enabled_models is None:
        cfg.enabled_models = defaults["enabled_models"]
    if cfg.cv_folds is None:
        cfg.cv_folds = int(defaults["cv_folds"])
    if cfg.max_grid_per_model is None:
        cfg.max_grid_per_model = int(defaults["max_grid_per_model"])
    if cfg.bootstrap_reps is None:
        cfg.bootstrap_reps = int(defaults["bootstrap_reps"])
    if cfg.n_jobs is None:
        cfg.n_jobs = int(defaults["n_jobs"])
    if getattr(cfg, "mitigation_max_jobs", None) is None:
        cfg.mitigation_max_jobs = defaults.get("mitigation_max_jobs")
    if getattr(cfg, "mitigation_max_intersectional_attributes", None) is None:
        cfg.mitigation_max_intersectional_attributes = defaults.get("mitigation_max_intersectional_attributes")
    if getattr(cfg, "mitigation_max_generated_intersectional_attributes", None) is not None:
        cfg.mitigation_max_intersectional_attributes = cfg.mitigation_max_generated_intersectional_attributes
    try:
        default_threshold_grid = PipelineConfig.__dataclass_fields__["threshold_grid"].default_factory()
        if "threshold_grid" in defaults and tuple(float(x) for x in cfg.threshold_grid) == tuple(float(x) for x in default_threshold_grid):
            cfg.threshold_grid = defaults["threshold_grid"]
    except Exception:
        pass
    if cfg.mitigation_methods is None:
        cfg.mitigation_methods = tuple(defaults.get("mitigation_methods", DEFAULT_MITIGATION_METHODS))
    elif isinstance(cfg.mitigation_methods, str):
        if cfg.mitigation_methods.lower() in {"auto", "default"}:
            cfg.mitigation_methods = tuple(defaults.get("mitigation_methods", DEFAULT_MITIGATION_METHODS))
        else:
            cfg.mitigation_methods = (cfg.mitigation_methods,)
    else:
        cfg.mitigation_methods = tuple(str(x) for x in cfg.mitigation_methods)
    if cfg.mitigation_objective is not None:
        objective = str(cfg.mitigation_objective).lower()
        objective_to_method = {
            "equalized_odds": "postprocess_group_threshold_equalized_odds",
            "equalized_ods": "postprocess_group_threshold_equalized_odds",
            "equal_opportunity": "postprocess_group_threshold_equal_opportunity",
            "demographic_parity": "postprocess_group_threshold_demographic_parity",
            "balanced_accuracy": "postprocess_group_threshold_balanced_accuracy",
        }
        cfg.mitigation_methods = (objective_to_method.get(objective, f"postprocess_group_threshold_{objective}"),)
    if cfg.max_balanced_accuracy_drop is not None:
        cfg.mitigation_max_accuracy_drop = cfg.max_balanced_accuracy_drop

    cfg.csv_candidates = tuple(str(x) for x in cfg.csv_candidates)
    cfg.target_candidates = tuple(str(x) for x in cfg.target_candidates)
    cfg.sensitive_candidates = tuple(str(x) for x in cfg.sensitive_candidates)
    if isinstance(cfg.sensitive_attrs, (list, tuple)):
        cfg.sensitive_attrs = tuple(str(x) for x in cfg.sensitive_attrs)
    elif cfg.sensitive_attrs is not None:
        cfg.sensitive_attrs = str(cfg.sensitive_attrs)
    cfg.enabled_models = tuple(str(x) for x in (cfg.enabled_models or ()))
    if isinstance(cfg.mitigation_families, str):
        cfg.mitigation_families = (cfg.mitigation_families,)
    else:
        cfg.mitigation_families = tuple(str(x) for x in (cfg.mitigation_families or ()))
    cfg.exclude_cols = tuple(str(x) for x in (cfg.exclude_cols or ()))
    if cfg.mitigation_custom_attributes is None:
        cfg.mitigation_custom_attributes = None
    else:
        custom_items = []
        for item in cfg.mitigation_custom_attributes:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                custom_items.append(tuple(str(x) for x in item))
            else:
                custom_items.append(str(item))
        cfg.mitigation_custom_attributes = tuple(custom_items)

    if isinstance(getattr(cfg, "mitigation_custom_intersections", None), str):
        cfg.mitigation_custom_intersections = str(cfg.mitigation_custom_intersections)
    elif cfg.mitigation_custom_intersections is not None:
        intersection_items = []
        for item in cfg.mitigation_custom_intersections:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                intersection_items.append(tuple(str(x) for x in item))
            else:
                intersection_items.append(str(item))
        cfg.mitigation_custom_intersections = tuple(intersection_items)
    try:
        cfg.mitigation_intersectional_orders = tuple(
            sorted({int(x) for x in cfg.mitigation_intersectional_orders if 2 <= int(x) <= 4})
        ) or (2,)
    except Exception:
        cfg.mitigation_intersectional_orders = (2, 3, 4)

    if getattr(cfg, "mitigation_max_intersectional_groups", None) is None:
        cfg.mitigation_max_intersectional_groups = int(getattr(cfg, "mitigation_max_pair_groups", 40) or 40)
    else:
        cfg.mitigation_max_intersectional_groups = int(cfg.mitigation_max_intersectional_groups or getattr(cfg, "mitigation_max_pair_groups", 40) or 40)
    for guard_name in ("mitigation_max_intersectional_attributes", "mitigation_max_generated_intersectional_attributes", "mitigation_max_jobs"):
        value = getattr(cfg, guard_name, None)
        if value in {None, False, "", "none", "None"}:
            setattr(cfg, guard_name, None)
        else:
            try:
                parsed = int(value)
                setattr(cfg, guard_name, parsed if parsed > 0 else None)
            except Exception:
                setattr(cfg, guard_name, None)
    if getattr(cfg, "mitigation_max_generated_intersectional_attributes", None) is not None:
        cfg.mitigation_max_intersectional_attributes = cfg.mitigation_max_generated_intersectional_attributes

    if cfg.feature_cols is not None:
        cfg.feature_cols = tuple(str(x) for x in cfg.feature_cols)
    cfg.threshold_grid = tuple(float(x) for x in cfg.threshold_grid)
    return cfg


def is_id_like(column: str, patterns: tuple[str, ...] = COMMON_EXCLUDE_PATTERNS) -> bool:
    c = str(column).strip().lower()
    return any(re.search(pattern, c) for pattern in patterns)


def dataset_slug(config: PipelineConfig) -> str:
    if config.dataset_name and str(config.dataset_name).lower() != "auto":
        return str(config.dataset_name).lower().replace(" ", "_")
    if config.csv_path:
        return Path(str(config.csv_path)).stem.lower().replace(" ", "_")
    return "dataset"


def resolve_project_paths(config: PipelineConfig) -> dict[str, Path | str | bool]:
    from .runtime import detect_environment

    env = detect_environment(config.environment)
    if config.output_base_dir:
        base = Path(config.output_base_dir).expanduser().resolve()
    elif env["is_kaggle"]:
        base = Path("/kaggle/working")
    elif env["is_colab"]:
        base = Path("/content")
    else:
        base = Path.cwd()

    name = dataset_slug(config)
    scenario = str(config.scenario or "analysis").lower().replace(" ", "_")
    root = base / f"outputs_{name}_fairness_analysis_{scenario}_{config.execution_preset}"
    figures = root / "figures"
    tables = root / "tables"
    for p in (root, figures, tables):
        p.mkdir(parents=True, exist_ok=True)
    return {"base": base, "root": root, "figures": figures, "tables": tables, **env}
