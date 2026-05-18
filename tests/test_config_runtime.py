import tempfile
import unittest
from pathlib import Path

from src.config import (
    PipelineConfig,
    config_from_user_config,
    dataset_slug,
    is_id_like,
    preset_defaults,
    resolve_project_paths,
)
from src.runtime import detect_environment, find_csv_path, gpu_status


class ConfigAndRuntimeTests(unittest.TestCase):
    def test_preset_defaults_falls_back_to_quick(self):
        self.assertEqual(preset_defaults("unknown"), preset_defaults("quick"))

    def test_config_from_user_config_normalises_aliases_and_collections(self):
        cfg = config_from_user_config(
            {
                "runtime": {"execution_preset": "SMOKE"},
                "data": {"sensitive_attrs": ["sex", "race"]},
                "fairness": {
                    "mitigation_objective": "equal_opportunity",
                    "mitigation_intersectional_orders": [1, 2, 4, 8],
                    "mitigation_max_generated_intersectional_attributes": 7,
                },
            }
        )
        self.assertEqual(cfg.execution_preset, "smoke")
        self.assertEqual(cfg.sensitive_attrs, ("sex", "race"))
        self.assertEqual(cfg.mitigation_methods, ("postprocess_group_threshold_equal_opportunity",))
        self.assertEqual(cfg.mitigation_intersectional_orders, (2, 4))
        self.assertEqual(cfg.mitigation_max_intersectional_attributes, 7)

    def test_slug_and_identifier_helpers(self):
        self.assertTrue(is_id_like("patient_id"))
        self.assertFalse(is_id_like("age"))
        self.assertEqual(dataset_slug(PipelineConfig(dataset_name="Stroke Risk")), "stroke_risk")
        self.assertEqual(
            dataset_slug(PipelineConfig(dataset_name="auto", csv_path="/tmp/My Data.csv")),
            "my_data",
        )

    def test_resolve_project_paths_creates_output_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = PipelineConfig(
                output_base_dir=tmp,
                dataset_name="demo",
                scenario="custom",
                execution_preset="quick",
            )
            paths = resolve_project_paths(cfg)
            self.assertTrue(Path(paths["root"]).exists())
            self.assertTrue(Path(paths["figures"]).exists())
            self.assertTrue(Path(paths["tables"]).exists())

    def test_detect_environment_local_override(self):
        env = detect_environment("local")
        self.assertFalse(env["is_kaggle"])
        self.assertFalse(env["is_colab"])
        self.assertEqual(env["environment_label"], "local")

    def test_find_csv_path_returns_explicit_file_and_raises_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            file_path = Path(tmp) / "data.csv"
            file_path.write_text("x\n1\n", encoding="utf-8")
            resolved = find_csv_path(str(file_path), ())
            self.assertEqual(resolved, file_path.resolve())
        with self.assertRaises(FileNotFoundError):
            find_csv_path("/definitely/missing/file.csv", ())

    def test_gpu_status_false_disables_gpu(self):
        status = gpu_status(False)
        self.assertFalse(status["gpu_requested"])
        self.assertFalse(status["gpu_available"])
        self.assertEqual(status["gpu_backend"], "disabled_by_configuration")


if __name__ == "__main__":
    unittest.main()
