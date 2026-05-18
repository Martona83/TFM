import unittest

import pandas as pd

from src.config import PipelineConfig
from src.workflow import (
    WorkflowContext,
    _apply_mitigation_runtime_caps,
    _canonical_mitigation_scope,
    _create_intersectional_attribute,
    _ensure_intersectional_attributes,
    _normalise_custom_intersection_entry,
    _resolve_mitigation_attributes,
)


class WorkflowHelperTests(unittest.TestCase):
    def _ctx(self, config: PipelineConfig) -> WorkflowContext:
        frame = pd.DataFrame(
            {
                "target": [0, 1, 0, 1],
                "gender_label": ["F", "M", "F", "M"],
                "race_label": ["A", "A", "B", "B"],
                "homo_label": ["No", "Yes", "No", "Yes"],
            }
        )
        return WorkflowContext(
            config=config,
            paths={},
            sensitive_attrs=("gender_label", "race_label", "homo_label"),
            splits={"train": frame.copy(), "validation": frame.copy(), "test": frame.copy()},
            models={"LogisticRegression": object()},
        )

    def test_normalise_custom_intersection_entry(self):
        self.assertEqual(_normalise_custom_intersection_entry("a__x__b"), ("a", "b"))
        self.assertEqual(_normalise_custom_intersection_entry("a+b+c"), ("a", "b", "c"))
        self.assertEqual(_normalise_custom_intersection_entry(("x", "y")), ("x", "y"))
        self.assertIsNone(_normalise_custom_intersection_entry("single"))

    def test_create_and_ensure_intersectional_attributes(self):
        cfg = PipelineConfig(mitigation_intersectional_orders=(2,))
        ctx = self._ctx(cfg)
        created = _create_intersectional_attribute(ctx, ("gender_label", "race_label"))
        self.assertEqual(created, "gender_label__x__race_label")
        self.assertIn(created, ctx.splits["test"].columns)

        ensured = _ensure_intersectional_attributes(ctx)
        self.assertTrue(any("__x__" in name for name in ensured))

    def test_canonical_scope_aliases(self):
        self.assertEqual(_canonical_mitigation_scope("all_individual"), "all_single")
        self.assertEqual(_canonical_mitigation_scope("custom_plus_pairs"), "configured_plus_intersectional")

    def test_resolve_configured_plus_intersectional_scope(self):
        cfg = PipelineConfig(
            mitigation_scope="custom_plus_pairs",
            mitigation_custom_attributes=("gender_label", "race_label"),
            mitigation_custom_intersections="all_pairs_among_custom_attributes",
            mitigation_intersectional_orders=(2,),
        )
        ctx = self._ctx(cfg)
        attrs = _resolve_mitigation_attributes(ctx)
        self.assertIn("gender_label", attrs)
        self.assertIn("race_label", attrs)
        self.assertTrue(any("__x__" in x for x in attrs))

    def test_runtime_caps_keep_singles_and_limit_intersections(self):
        cfg = PipelineConfig(
            mitigation_max_intersectional_attributes=1,
            mitigation_max_jobs=2,
            mitigation_methods=("postprocess_group_threshold_equalized_odds",),
        )
        ctx = self._ctx(cfg)
        ctx.tables["baseline_fairness_gaps"] = pd.DataFrame(
            {
                "model": ["LogisticRegression", "LogisticRegression"],
                "prediction_type": ["baseline", "baseline"],
                "attribute": ["gender_label__x__race_label", "gender_label__x__homo_label"],
                "combined_fpr_fnr_gap": [0.2, 0.1],
            }
        )
        ctx.champion_model = "LogisticRegression"
        attrs = _apply_mitigation_runtime_caps(
            ctx,
            ("gender_label", "race_label", "homo_label", "gender_label__x__race_label", "gender_label__x__homo_label"),
        )
        self.assertIn("gender_label", attrs)
        self.assertIn("race_label", attrs)
        self.assertIn("homo_label", attrs)
        self.assertFalse(any("__x__" in x for x in attrs))


if __name__ == "__main__":
    unittest.main()
