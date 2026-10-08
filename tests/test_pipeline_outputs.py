"""Small CPU-only pipeline regression tests, with artifacts in temporary folders."""

import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

common = importlib.import_module("common")
forest_output = importlib.import_module("forest_output")
output_io = importlib.import_module("output_io")


def load_step(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_pipeline(output_dir):
    rng = np.random.default_rng(42)
    raw = pd.DataFrame(rng.choice([-1, 0, 1], size=(100, 3)),
                       columns=["URL_Length", "having_IP_Address", "Prefix_Suffix"])
    raw["Result"] = np.tile([-1, 1], 50)
    stdout = io.StringIO()
    with contextlib.ExitStack() as stack:
        stack.enter_context(contextlib.redirect_stdout(stdout))
        for name, value in {
            "OUT_DIR": output_dir,
            "CLEAN_CSV": output_dir / "clean_dataset.csv",
            "LEAKAGE_JSON": output_dir / "leakage_report.json",
            "SELECTION_JSON": output_dir / "selected_features.json",
        }.items():
            stack.enter_context(patch.object(common, name, value))
        stack.enter_context(patch.object(common, "load_raw", return_value=raw))
        leakage = load_step("01_remove_leaky_features")
        selection = load_step("02_select_features")
        forest = load_step("03_train_random_forest")
        # Output modules bind paths at import time, just like the entry points.
        for name in ("leakage_output", "selection_output", "forest_output"):
            module = sys.modules.get(name)
            if module is not None:
                for attr in ("OUT_DIR", "CLEAN_CSV", "LEAKAGE_JSON", "SELECTION_JSON"):
                    if hasattr(module, attr):
                        stack.enter_context(patch.object(module, attr, getattr(common, attr)))
        original_make_model = selection.make_model
        stack.enter_context(patch.object(selection, "make_model",
                                        side_effect=lambda device: original_make_model(device).set_params(
                                            n_estimators=2, max_depth=2, n_jobs=1)))
        stack.enter_context(patch.object(forest, "PARAM_DIST", {
            "n_estimators": randint(2, 3), "max_depth": [2],
        }))
        for module, args in ((leakage, []), (selection, ["--device", "cpu"]),
                             (forest, ["--n-iter", "1"])):
            with patch.object(sys, "argv", [module.__name__, *args]):
                module.main()
    return stdout.getvalue()


class PipelineOutputTests(unittest.TestCase):
    def test_prediction_formatting_at_threshold(self):
        predictions = forest_output.raw_predictions(
            [9, 2, 8, 1], [10, 20, 30, 40], [1, 0, 0, 1],
            np.array([0.5, 0.4999999, 0.5, 0.1]), "test",
        )
        self.assertEqual(predictions["row_index"].tolist(), [9, 2, 8, 1])
        self.assertEqual(predictions["predicted_label"].tolist(), [1, 0, 1, 0])
        self.assertEqual(predictions["outcome"].tolist(), ["TP", "TN", "FP", "FN"])
        self.assertEqual(predictions["correct"].tolist(), [True, True, False, False])
        self.assertEqual(predictions["true_class"].tolist(),
                         ["phishing", "legitimate", "legitimate", "phishing"])
        self.assertEqual(predictions["prob_phishing"].tolist(), [0.5, 0.5, 0.5, 0.1])

    def test_json_writing_creates_parents_and_propagates_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "report.json"
            output_io.write_json(path, {"score": np.float32(0.5)})
            self.assertEqual(path.read_text(encoding="utf-8"), '{\n  "score": 0.5\n}')
            self.assertEqual(common.read_json(path), {"score": 0.5})
            with patch.object(Path, "write_text", side_effect=PermissionError("read-only")):
                with self.assertRaises(PermissionError):
                    output_io.write_json(path, {})

    def test_artifact_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            stdout = run_pipeline(out)
            expected = {
                "clean_dataset.csv", "leakage_report.json", "selected_features.json",
                "feature_selection_curve.png", "random_forest.joblib", "rf_metrics.json",
                "rf_metrics_summary.csv", "rf_results_report.txt", "rf_test_predictions.csv",
                "rf_train_oof_predictions.csv", "rf_feature_importance.png", "rf_confusion_matrix.png",
            }
            self.assertEqual({p.name for p in out.iterdir()}, expected)
            clean = pd.read_csv(out / "clean_dataset.csv")
            self.assertEqual(len(clean), 100)
            self.assertFalse(set(clean.loc[clean["split"] == "train", "group_id"]) &
                             set(clean.loc[clean["split"] == "test", "group_id"]))
            selection = common.read_json(out / "selected_features.json")
            metrics = common.read_json(out / "rf_metrics.json")
            model = joblib.load(out / "random_forest.joblib")
            self.assertEqual(model["features"], selection["selected_features"])
            self.assertEqual(model["positive_class"], "phishing")
            self.assertEqual(metrics["features"], model["features"])
            for filename, split in (("rf_test_predictions.csv", "test"),
                                    ("rf_train_oof_predictions.csv", "train_oof")):
                predictions = pd.read_csv(out / filename)
                self.assertEqual(list(predictions.columns[:11]), [
                    "row_index", "group_id", "split", "true_label", "true_class",
                    "predicted_label", "predicted_class", "prob_phishing",
                    "prob_legitimate", "correct", "outcome",
                ])
                self.assertEqual(set(predictions["split"]), {split})
                np.testing.assert_allclose(predictions["prob_phishing"] +
                                           predictions["prob_legitimate"], 1)
                self.assertEqual(set(predictions["outcome"]), {"TP", "TN", "FP", "FN"})
                source = clean.iloc[predictions["row_index"]]
                np.testing.assert_array_equal(predictions["true_label"], source["is_phishing"])
                np.testing.assert_array_equal(predictions[model["features"]],
                                              source[model["features"]])
                if split == "train_oof":
                    self.assertEqual(set(predictions["cv_fold"]), {1, 2, 3, 4, 5})
                else:
                    np.testing.assert_allclose(predictions["prob_phishing"],
                                               model["model"].predict_proba(source[model["features"]])[:, 1],
                                               atol=0.0000005)
            summary = pd.read_csv(out / "rf_metrics_summary.csv")
            self.assertEqual(len(summary), 12)
            self.assertNotIn("confusion_matrix", summary.columns)
            self.assertEqual(summary.iloc[0]["evaluation"], "test_holdout")
            self.assertEqual(summary.iloc[0]["accuracy"], round(metrics["test"]["accuracy"], 6))
            report = (out / "rf_results_report.txt").read_text(encoding="utf-8")
            for heading in ("HELD-OUT TEST SET", "Per-class report", "Per-fold mean +/- std",
                            "LEAKAGE REFERENCE RUNS"):
                self.assertIn(heading, report)
            self.assertIn("Wrote random_forest.joblib", stdout)
            for filename in expected:
                if filename.endswith(".png"):
                    self.assertTrue((out / filename).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertEqual(forest_output.plt.get_fignums(), [])


if __name__ == "__main__":
    unittest.main()
