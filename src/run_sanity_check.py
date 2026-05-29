<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
from fairness_pipeline.workflow import run_full_pipeline
=======
=======
>>>>>>> parent of ee2e379 (Potential fix for pull request finding)
=======
>>>>>>> parent of ee2e379 (Potential fix for pull request finding)
try:
    from fairness_pipeline.workflow import run_full_pipeline
except ModuleNotFoundError:
    from src.workflow import run_full_pipeline

>>>>>>> parent of ee2e379 (Potential fix for pull request finding)
=======
from fairness_pipeline.workflow import run_full_pipeline
>>>>>>> parent of a3da1b2 (ci: create uv-based workflow and harden sanity check import)
=======
from fairness_pipeline.workflow import run_full_pipeline
>>>>>>> parent of a3da1b2 (ci: create uv-based workflow and harden sanity check import)

USER_CONFIG = {
    "runtime": {"execution_preset": "smoke", "output_base_dir": "./sanity_outputs", "n_jobs": 1},
    "data": {"dataset_name": "clinical_example", "csv_path": "example_clinical_dataset.csv", "scenario": "baseline"},
    "models": {"enabled_models": ("LogisticRegression",), "max_grid_per_model": 1, "cv_folds": 2},
    "fairness": {
        "fairness_audit_scope": "single_plus_pairs",
        "bootstrap_reps": 5,
        "mitigation_scope": "custom_plus_pairs",
        "mitigation_custom_attributes": (
            "gender_label", "race_label", "homo_label", "drugs_label",
            "ever_married", "Residence_type", "work_type",
        ),
        "mitigation_custom_intersections": "all_intersections_among_custom_attributes",
        "mitigation_intersectional_orders": (2, 3, 4),
        "mitigation_methods": ("postprocess_group_threshold_equalized_odds",),
    },
    "reporting": {"display_figures": False, "show_model_training_explanations": False, "max_table_rows_display": 5},
}

ctx = run_full_pipeline(USER_CONFIG)
assert ctx.analytic is not None and len(ctx.analytic) > 0
assert ctx.feature_cols
assert ctx.models
selected = ctx.tables.get("mitigation_selected_attributes")
assert selected is not None and not selected.empty
selected_names = set(selected["attribute"].astype(str))
assert "gender_label" in selected_names
assert "homo_label" in selected_names
assert any("gender_label__x__homo_label" in x or "homo_label__x__gender_label" in x for x in selected_names)
assert any(x.count("__x__") == 2 for x in selected_names)  # a 3-way intersection was generated
assert any(x.count("__x__") == 3 for x in selected_names)  # a 4-way intersection was generated
print("Sanity check passed.")
print("Selected model:", ctx.champion_model)
print("Primary fairness attribute:", ctx.primary_attribute)
print("Mitigation attributes:", ", ".join(sorted(selected_names)))
print("Output folder:", ctx.paths["root"])
