# Generic tabular fairness pipeline — modular v18.0

Run the notebook `TFM_fairness_pipeline_modular_v18_1000bootstrap_custom_intersections.ipynb` from this folder.

## Main changes in v18.0

- The package remains neutral and is named `fairness_pipeline`.
- The first executable notebook cell attempts package installation first and suppresses the non-actionable pip version notice with `--disable-pip-version-check`.
- Fairlearn, AIF360 and imbalanced-learn are attempted first. If unavailable, the workflow continues with support operations.
- GPU availability is checked in every runtime. Scikit-learn models remain CPU-bound; XGBoost defaults to `safe_cpu` to avoid CPU/GPU prediction mismatch in sklearn pipelines.
- Mitigation is no longer restricted by default to the two largest-gap attributes. The v18 corrected analyst configuration evaluates `gender_label`, `race_label`, `homo_label`, `drugs_label`, `ever_married`, `Residence_type`, and `work_type` when present, and adds 2-, 3-, and 4-way intersections among the selected attributes when subgroup support is sufficient.
- The old `top_k_gap_attributes` mode remains available for fast exploratory runs. It may select only race/drugs if those are the two largest observed baseline gaps.
- A new table explains exactly why each mitigation attribute was included: `16a_mitigation_selected_attributes.csv`.
- Mitigation strategies remain documented as pre-processing, intra-processing and post-processing.
- The consolidated mitigation leaderboard remains available as `20_best_mitigation_model_combinations.csv`.
- Statistical evidence remains available as `20b_mitigation_statistical_evidence_by_metric.csv`.

## Recommended run

This corrected notebook is configured for the final thesis run by default:

```python
USER_CONFIG["runtime"]["execution_preset"] = "full"
```

The v18 final run uses 1000 bootstrap repetitions, `mitigation_scope="custom_plus_pairs"`, and 2-, 3-, and 4-way intersectional groups among the configured custom attributes when subgroup support is sufficient. This is intentionally heavier than the v17 smoke preset.


## Runtime guard for the mitigation cell

The mitigation cell can become very long when many sensitive attributes are combined. Version 18 keeps the runtime guards explicit. The corrected thesis configuration sets them high enough to avoid silently dropping requested intersectional groups:

```python
USER_CONFIG["fairness"].update({
    "mitigation_max_jobs": 20000,
    "mitigation_max_intersectional_attributes": 200,
})
```

Single sensitive attributes are prioritised. Intersectional attributes are capped and ranked by the observed baseline fairness gap when available. For smaller exploratory runs, reduce `mitigation_max_jobs` and `mitigation_max_intersectional_attributes` manually.

## Mitigation scope options

Focused mitigation with ACTG175 variables plus the requested generic/stroke-style variables, using pairs, triples and four-way groups where feasible:

```python
USER_CONFIG["fairness"].update({
    "fairness_audit_scope": "single_plus_intersectional",
    "mitigation_scope": "custom_plus_pairs",
    "mitigation_custom_attributes": (
        "gender_label", "race_label", "homo_label", "drugs_label",
        "ever_married", "Residence_type", "work_type",
    ),
    "mitigation_custom_intersections": "all_intersections_among_custom_attributes",
    "mitigation_intersectional_orders": (2, 3, 4),
    "bootstrap_reps": 1000,
})
```

Individual-only mitigation across all configured/inferred sensitive attributes:

```python
USER_CONFIG["fairness"].update({
    "mitigation_scope": "all_individual",
    "mitigation_custom_attributes": None,
})
```

Fast top-k mitigation, which may evaluate only race/drugs if they are the largest observed gaps:

```python
USER_CONFIG["fairness"].update({
    "mitigation_scope": "top_k_gap_attributes",
    "mitigation_top_k_attributes": 2,
})
```

Broadest intersectionality check across all inferred sensitive attributes:

```python
USER_CONFIG["fairness"].update({
    "fairness_audit_scope": "single_plus_intersectional",
    "mitigation_scope": "all_individual_and_pairs",
    "mitigation_intersectional_orders": (2, 3, 4),
})
```

## Generic CSV use

After the raw EDA stage, update the analyst configuration cell:

```python
USER_CONFIG["data"].update({
    "dataset_mode": "generic_tabular",
    "dataset_name": "my_dataset",
    "scenario": "custom",
    "csv_path": "/path/to/my_dataset.csv",
    "target_col": "outcome",
    "positive_target_value": 1,
    "sensitive_attrs": ("sex", "race", "ever_married", "Residence_type", "work_type"),
    "exclude_cols": ("patient_id", "visit_date"),
    "feature_cols": None,
})
```

The final notebook cell exports all generated tables and figures as a ZIP archive.
