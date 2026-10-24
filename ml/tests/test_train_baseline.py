"""Validate the baseline adapter; a short synthetic smoke run does not assess ML quality."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_dataset import (CATEGORICAL, DEFAULT_HISTORY, FEATURES, NUMERIC, SCHEMA,
                             TARGET, build_dataset, timestamp)
from train_baseline import PARAMS, feature_matrix, make_pool, metrics, select_rows, train


class BaselineAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = DEFAULT_HISTORY.read_bytes()
        cls.validation_start = "2026-08-01T15:00:00Z"
        cls.test_start = "2026-09-01T15:00:00Z"
        cls.rows = build_dataset(json.loads(raw), timestamp(cls.validation_start), timestamp(cls.test_start))
        cls.manifest = {
            "schemaVersion": SCHEMA, "targetVersion": TARGET,
            "categoricalFeatures": list(CATEGORICAL), "numericFeatures": list(NUMERIC),
            "validationStart": cls.validation_start, "testStart": cls.test_start,
            "historySha256": hashlib.sha256(raw).hexdigest(),
        }
        cls.train_rows, cls.validation_rows = select_rows(cls.rows, cls.manifest)

    def selected_with_mutated_row(self, split, mutate):
        rows = self.train_rows + self.validation_rows
        index = next(i for i, row in enumerate(rows) if row["split"] == split)
        rows[index] = copy.deepcopy(rows[index])
        mutate(rows[index])
        return rows

    def assert_nan_aware_equal(self, first, second):
        self.assertEqual(len(first), len(second))
        for left_row, right_row in zip(first, second):
            self.assertEqual(len(left_row), len(right_row))
            for left, right in zip(left_row, right_row):
                if isinstance(left, float) and math.isnan(left):
                    self.assertTrue(isinstance(right, float) and math.isnan(right))
                else:
                    self.assertEqual(left, right)

    def test_select_uses_only_train_validation_and_ignores_poisoned_other_splits(self):
        poisons = [{"split": split, "features": "DO NOT READ", "label": "DO NOT READ"}
                   for split in ("test", "pending", "excluded", "purged")]
        selected = select_rows(self.train_rows + self.validation_rows + poisons, self.manifest)
        self.assertEqual(selected, (self.train_rows, self.validation_rows))
        self.assertEqual({row["label"]["value"] for row in selected[0]}, {0, 1})
        self.assertEqual({row["label"]["value"] for row in selected[1]}, {0, 1})

    def test_schema_target_and_feature_order_must_match(self):
        for key, value in (("schemaVersion", "old-schema"), ("targetVersion", "old-target"),
                           ("categoricalFeatures", list(reversed(CATEGORICAL))),
                           ("numericFeatures", list(reversed(NUMERIC)))):
            with self.subTest(field=key):
                manifest = {**self.manifest, key: value}
                with self.assertRaises(ValueError):
                    select_rows(self.rows, manifest)

    def test_excluded_label_mislabeled_as_train_is_rejected(self):
        def mutate(row):
            row["label"]["reason"] = "excluded_nonfinancial_block"
            # Even an incorrect 0/1 must not bypass the explicit exclusion reason.
            row["label"]["value"] = 1
        rows = self.selected_with_mutated_row("train", mutate)
        with self.assertRaises(ValueError):
            select_rows(rows, self.manifest)

    def test_label_at_next_temporal_boundary_is_rejected(self):
        for split, boundary in (("train", self.validation_start), ("validation", self.test_start)):
            with self.subTest(split=split):
                rows = self.selected_with_mutated_row(split, lambda row: row["label"].update(availableAt=boundary))
                with self.assertRaises(ValueError):
                    select_rows(rows, self.manifest)

    def test_future_feature_audit_is_rejected(self):
        for key in ("maxFeatureOccurredAt", "maxFeatureRecordedAt"):
            with self.subTest(key=key):
                rows = self.selected_with_mutated_row("train", lambda row: row["audit"].update(
                    {key: "2099-01-01T15:00:00Z"}))
                with self.assertRaises(ValueError):
                    select_rows(rows, self.manifest)

    def test_invalid_cycle_weight_is_rejected(self):
        for weight in (0, -1, 100, float("nan"), float("inf")):
            with self.subTest(weight=weight):
                rows = self.selected_with_mutated_row("train", lambda row: row.update(cycleWeight=weight))
                with self.assertRaises(ValueError):
                    select_rows(rows, self.manifest)

    def test_cycle_shared_by_train_validation_is_rejected(self):
        original_cycle = self.validation_rows[0]["cycleId"]
        reused_cycle = self.train_rows[0]["cycleId"]
        rows = self.train_rows + [
            {**row, "cycleId": reused_cycle} if row["cycleId"] == original_cycle else row
            for row in self.validation_rows
        ]
        with self.assertRaises(ValueError):
            select_rows(rows, self.manifest)

    def test_unknown_feature_inside_features_is_rejected(self):
        rows = self.selected_with_mutated_row("train", lambda row: row["features"].update(future_outcome=1))
        with self.assertRaises(ValueError):
            select_rows(rows, self.manifest)

    def test_ids_label_and_future_metadata_do_not_change_feature_matrix(self):
        originals = self.train_rows[:10]
        changed = copy.deepcopy(originals)
        for row in changed:
            row.update(serviceId=999999, accountId=-123, sampleId="poison", cycleId="future-cycle",
                       future_status="deleted", future_payment=1000000,
                       futureMetadata={"outcome": 1}, audit={"not": "model input"},
                       label={"value": 999, "reason": "poison"}, cycleWeight=999,
                       predictionAt="2099-01-01T00:00:00Z")
        self.assert_nan_aware_equal(feature_matrix(originals), feature_matrix(changed))

    def test_null_numeric_becomes_nan_and_categories_remain_strings_in_declared_order(self):
        row = copy.deepcopy(self.train_rows[0])
        for name in NUMERIC:
            row["features"][name] = None
        for name in CATEGORICAL:
            row["features"][name] = f"category-{name}"
        matrix = feature_matrix([row])
        self.assertEqual(len(matrix), 1)
        self.assertEqual(len(matrix[0]), len(FEATURES))
        for name, value in zip(FEATURES, matrix[0]):
            if name in CATEGORICAL:
                self.assertIsInstance(value, str)
                self.assertEqual(value, f"category-{name}")
            else:
                self.assertTrue(math.isnan(value))

    def test_invalid_numeric_and_categorical_values_are_rejected(self):
        for name, value in [(NUMERIC[0], value) for value in ("12", float("nan"), float("inf"))] + [
                (CATEGORICAL[0], value) for value in (None, "", 123)]:
            with self.subTest(name=name, value=value):
                row = copy.deepcopy(self.train_rows[0])
                row["features"][name] = value
                with self.assertRaises(ValueError):
                    feature_matrix([row])

    @unittest.skipUnless(importlib.util.find_spec("catboost") is not None, "CatBoost not installed")
    def test_five_iteration_training_and_save_load_without_test_access(self):
        from catboost import CatBoostClassifier
        params = {**PARAMS, "iterations": 5, "thread_count": 1}
        poisoned_heldout = [{"split": split, "features": "NOT A MATRIX", "label": "DO NOT EVALUATE"}
                            for split in ("test", "pending", "excluded")]
        with tempfile.TemporaryDirectory(prefix="risk-baseline-smoke-") as directory:
            output = Path(directory) / "run"
            report = train(self.train_rows + self.validation_rows + poisoned_heldout,
                           self.manifest, output, params=params)
            self.assertFalse(report["testEvaluated"])
            self.assertTrue(report["syntheticOnly"])
            self.assertTrue(report["saveLoadPredictionCheckPassed"])
            self.assertEqual(set(report["splits"]), {"train", "validation"})
            self.assertEqual(report["parameters"]["loss_function"], "Logloss")
            self.assertGreater(report["trees"], 0)
            self.assertLessEqual(report["trees"], 5)
            self.assertEqual(report["modelSha256"], hashlib.sha256((output / "model.cbm").read_bytes()).hexdigest())
            self.assertEqual(json.loads((output / "report.json").read_text()), report)
            self.assertTrue((output / "learning-curve.json").is_file())
            restored = CatBoostClassifier()
            restored.load_model(str(output / "model.cbm"))
            predictions = restored.predict_proba(make_pool(self.validation_rows[:5]))[:, 1]
            self.assertEqual(len(predictions), 5)
            self.assertTrue(all(math.isfinite(probability) and 0 <= probability <= 1 for probability in predictions))


class BaselineMetricTests(unittest.TestCase):
    def test_weighted_logloss_brier_and_rates(self):
        rows = [{"cycleWeight": 0.25, "label": {"value": 0}},
                {"cycleWeight": 0.75, "label": {"value": 1}}]
        result = metrics(rows, [0.2, 0.8])
        self.assertAlmostEqual(result["logloss"], -math.log(0.8))
        self.assertAlmostEqual(result["brier"], 0.04)
        self.assertAlmostEqual(result["meanProbability"], 0.65)
        self.assertAlmostEqual(result["lossRate"], 0.75)

    def test_cycle_weights_prevent_daily_row_duplication_from_changing_metrics(self):
        rows = [{"cycleWeight": 1, "label": {"value": 0}},
                {"cycleWeight": 1, "label": {"value": 1}}]
        repeated = [{"cycleWeight": 0.25, "label": {"value": 0}} for _ in range(4)] + [rows[1]]
        expected = metrics(rows, [0.3, 0.7])
        actual = metrics(repeated, [0.3, 0.3, 0.3, 0.3, 0.7])
        for key in expected:
            self.assertAlmostEqual(actual[key], expected[key])

    def test_exact_probability_boundaries_produce_finite_logloss(self):
        rows = [{"cycleWeight": 1, "label": {"value": 0}},
                {"cycleWeight": 1, "label": {"value": 1}}]
        result = metrics(rows, [1, 0])
        self.assertTrue(math.isfinite(result["logloss"]))
        self.assertGreater(result["logloss"], 30)
        self.assertEqual(result["brier"], 1)

    def test_prediction_count_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            metrics([{"cycleWeight": 1, "label": {"value": 0}}], [])


if __name__ == "__main__":
    unittest.main()
