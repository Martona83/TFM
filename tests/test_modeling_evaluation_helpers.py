import unittest

import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.evaluation import (
    apply_group_thresholds,
    binary_classification_metrics,
    exact_mcnemar_p,
    fairness_by_group,
    fairness_gap_summary,
    mitigation_method_family,
    selected_mitigation_methods,
)
from src.modeling import _predict_proba_binary, _safe_cv_folds, _truncate_grid


class _ProbaEstimator:
    def predict_proba(self, X):
        return np.column_stack([1 - np.asarray(X["p"], dtype=float), np.asarray(X["p"], dtype=float)])


class _DecisionEstimator:
    def decision_function(self, X):
        return np.asarray(X["score"], dtype=float)


class ModelingAndEvaluationTests(unittest.TestCase):
    def test_truncate_grid_caps_combinations(self):
        grid = {"a": [1, 2, 3], "b": [10, 20, 30]}
        truncated = _truncate_grid(grid, max_grid=4)
        self.assertLessEqual(len(truncated["a"]) * len(truncated["b"]), 4)

    def test_safe_cv_folds_handles_minority_counts(self):
        y = np.array([0, 0, 1, 1, 1])
        self.assertEqual(_safe_cv_folds(y, requested=10), 2)

    def test_predict_proba_binary_supports_predict_proba_and_decision_function(self):
        df_prob = pd.DataFrame({"p": [0.2, 0.8]})
        proba = _predict_proba_binary(_ProbaEstimator(), df_prob)
        self.assertTrue(np.allclose(proba, np.array([0.2, 0.8])))

        df_score = pd.DataFrame({"score": [0.0, 2.0]})
        proba2 = _predict_proba_binary(_DecisionEstimator(), df_score)
        self.assertEqual(proba2.shape, (2,))
        self.assertGreater(proba2[1], proba2[0])

    def test_binary_classification_and_gap_summary(self):
        y_true = np.array([0, 0, 1, 1])
        prob = np.array([0.1, 0.9, 0.8, 0.2])
        metrics = binary_classification_metrics(y_true, prob, threshold=0.5)
        self.assertIn("balanced_accuracy", metrics)
        self.assertIn("ece", metrics)

        df = pd.DataFrame({"target": y_true, "group": ["A", "A", "B", "B"]})
        by_group = fairness_by_group(df, prob, 0.5, ("group",), "M1")
        gap = fairness_gap_summary(by_group)
        self.assertIn("combined_fpr_fnr_gap", gap.columns)
        self.assertEqual(gap.iloc[0]["attribute"], "group")

    def test_group_threshold_application_and_mcnemar(self):
        df = pd.DataFrame({"group": ["A", "B", "A", "B"]})
        pred = apply_group_thresholds(df, np.array([0.2, 0.9, 0.6, 0.4]), "group", {"A": 0.5, "B": 0.8}, 0.5)
        self.assertEqual(pred.tolist(), [0, 1, 1, 0])

        p, b, c = exact_mcnemar_p(np.array([True, False]), np.array([True, False]))
        self.assertEqual((p, b, c), (1.0, 0, 0))

    def test_mitigation_method_family_and_filtering(self):
        self.assertEqual(mitigation_method_family("preprocess_reweighing"), "pre-processing")
        self.assertEqual(mitigation_method_family("inprocess_fairlearn_expgrad_equalized_odds"), "intra-processing")
        self.assertEqual(mitigation_method_family("postprocess_group_threshold_equalized_odds"), "post-processing")

        cfg = PipelineConfig(
            mitigation_methods=(
                "preprocess_reweighing",
                "inprocess_fairlearn_expgrad_equalized_odds",
                "postprocess_group_threshold_equalized_odds",
            ),
            mitigation_families=("post",),
        )
        self.assertEqual(
            selected_mitigation_methods(cfg),
            ("postprocess_group_threshold_equalized_odds",),
        )


if __name__ == "__main__":
    unittest.main()
