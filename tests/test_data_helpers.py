import unittest

import pandas as pd

from src.config import PipelineConfig
from src.data import (
    _stratify_label,
    encode_binary_target,
    infer_sensitive_attrs,
    infer_target_col,
    logical_type,
    preflight_validate_dataset,
)


class DataHelperTests(unittest.TestCase):
    def test_infer_target_col_prefers_known_candidate(self):
        df = pd.DataFrame({"Outcome": [0, 1, 0], "x": [1, 2, 3]})
        cfg = PipelineConfig(target_col=None)
        self.assertEqual(infer_target_col(df, cfg), "Outcome")

    def test_infer_target_col_raises_on_ambiguous_binary_columns(self):
        df = pd.DataFrame({"a": [0, 1, 0], "b": [1, 0, 1], "value": [3, 4, 5]})
        with self.assertRaises(ValueError):
            infer_target_col(df, PipelineConfig(target_col=None, target_candidates=()))

    def test_encode_binary_target_auto_positive(self):
        encoded, positive = encode_binary_target(pd.Series(["no", "yes", "no", "yes"]), positive_value=None)
        self.assertEqual(positive, "yes")
        self.assertEqual(encoded.tolist(), [0, 1, 0, 1])

    def test_infer_sensitive_attrs_auto_builds_age_group(self):
        df = pd.DataFrame({"target": [0, 1, 0, 1], "age": [20, 30, 80, 90], "sex": ["F", "M", "F", "M"]})
        cfg = PipelineConfig(sensitive_attrs="auto")
        out_df, attrs = infer_sensitive_attrs(df, cfg, "target")
        self.assertIn("age_group", out_df.columns)
        self.assertIn("sex", attrs)
        self.assertIn("age_group", attrs)

    def test_logical_type_variants(self):
        cfg = PipelineConfig(numeric_as_categorical_max_unique=3, max_categorical_levels=5)
        self.assertEqual(logical_type(pd.Series([0, 1, 0, 1]), cfg), "binary")
        self.assertEqual(logical_type(pd.Series([1, 2, 3]), cfg), "numeric_discrete")
        self.assertEqual(logical_type(pd.Series([1.1, 2.2, 3.3, 4.4]), cfg), "numeric_continuous")
        self.assertEqual(logical_type(pd.Series(["a", "b", "c"]), cfg), "categorical")

    def test_preflight_validate_dataset_rejects_temporal_leakage_policy(self):
        analytic = pd.DataFrame({"target": [0, 1, 0, 1], "event_date": [1, 2, 3, 4], "feat": [5, 6, 7, 8]})
        cfg = PipelineConfig(split_strategy="random_stratified")
        with self.assertRaises(ValueError):
            preflight_validate_dataset(analytic, ["feat"], cfg)

    def test_stratify_label_uses_preferred_sensitive_attribute_when_supported(self):
        df = pd.DataFrame({"target": [0, 0, 1, 1], "sex": ["F", "M", "F", "M"]})
        label = _stratify_label(df, ("sex",), preferred_attr="sex")
        self.assertTrue(label.str.contains("__").all())


if __name__ == "__main__":
    unittest.main()
