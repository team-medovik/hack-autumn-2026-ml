import copy
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ML = Path(__file__).parents[1]
sys.path.insert(0, str(ML))
import train_event_driven as training


class MatrixTests(unittest.TestCase):
    def sample(self):
        return {"features": {name: "known" if name in training.CATEGORICAL else 0
                             for name in training.FEATURES}, "label": {"value": 1},
                "diagnostics": {"future": 999}, "accountId": 999}

    def test_exact_feature_order_and_no_metadata(self):
        sample = self.sample()
        sample["features"]["wallet_balance_minor"] = 123
        sample["features"]["auto_prolong"] = 1
        sample["features"]["topup_count_30d"] = 4
        vector = training.matrix([sample], training.FEATURES)[0]
        self.assertEqual(len(vector), 36)
        for name, value in (("wallet_balance_minor", 123), ("auto_prolong", 1), ("topup_count_30d", 4)):
            self.assertEqual(vector[training.FEATURES.index(name)], value)
        self.assertNotIn(999, vector)

    def test_unknown_numeric_remains_nan_not_zero(self):
        sample = self.sample()
        sample["features"]["days_since_topup"] = None
        vector = training.matrix([sample], training.FEATURES)[0]
        self.assertTrue(math.isnan(vector[training.FEATURES.index("days_since_topup")]))

    def test_reject_invalid_categories_and_numbers(self):
        for name, value in (("status", None), ("tariff_code", ""), ("wallet_balance_minor", True),
                            ("wallet_balance_minor", math.inf), ("topup_count_30d", "4")):
            sample = self.sample()
            sample["features"][name] = value
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                training.matrix([sample], training.FEATURES)


@unittest.skipUnless((ML / "prepared-event-driven/lifecycle-v3/rows.jsonl").exists(), "Prepared local fixture required")
class PartitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepared = ML / "prepared-event-driven/lifecycle-v3"
        cls.manifest = json.loads((cls.prepared / "manifest.json").read_text())
        cls.rows = [json.loads(line) for line in (cls.prepared / "rows.jsonl").read_text().splitlines()]

    def test_only_three_mature_partitions_are_selected(self):
        parts = training.select_partitions(self.rows, self.manifest)
        self.assertEqual(set(parts), {"train", "validation", "test"})
        for split, rows in parts.items():
            self.assertEqual(len(rows), self.manifest["splits"][split]["rows"])
            self.assertTrue(all(r["label"]["value"] in (0, 1) for r in rows))
        self.assertFalse({r["cycleId"] for r in parts["train"]} & {r["cycleId"] for r in parts["test"]})

    def test_reject_schema_feature_order_and_target_mismatch(self):
        for key, value in (("schemaVersion", "legacy"), ("targetVersion", "different"),
                           ("featureNames", list(reversed(training.FEATURES)))):
            manifest = {**self.manifest, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                training.select_partitions(self.rows, manifest)

    def test_reject_future_evidence_and_wrong_split(self):
        for change in ("future", "split"):
            rows = copy.deepcopy(self.rows)
            sample = next(r for r in rows if r["split"] == "train")
            if change == "future":
                sample["audit"]["maxFeatureRecordedAt"] = sample["labelDeadline"]
            else:
                sample["split"] = "test"
            with self.subTest(change=change), self.assertRaises(ValueError):
                training.select_partitions(rows, self.manifest)

    def test_reject_inconsistent_views_and_weights(self):
        for change in ("views", "weights"):
            rows = copy.deepcopy(self.rows)
            sample = next(r for r in rows if r["split"] == "train")
            if change == "views":
                sample["legacy_features"]["days_to_expiry"] = -1
            else:
                sample["cycleWeight"] = 999
            with self.subTest(change=change), self.assertRaises(ValueError):
                training.select_partitions(rows, self.manifest)

    @unittest.skipUnless(importlib.util.find_spec("catboost") and (ML / "runs/generator-v2-logloss/model.cbm").exists(),
                         "CatBoost and frozen baseline required")
    def test_real_training_no_test_in_fit_all_outputs_and_reproducibility(self):
        from catboost import CatBoostClassifier
        source = ML.parent / "examples/risk/event-driven/generated/lifecycle-v3-seed20261024/observations.json"
        previous = ML / "runs/generator-v2-logloss"
        frozen = [source, self.prepared / "rows.jsonl", previous / "model.cbm"]
        hashes = [training.digest(p) for p in frozen]
        parts = training.select_partitions(self.rows, self.manifest)
        original_fit = CatBoostClassifier.fit
        calls = []

        def fit_spy(model, X, *args, **kwargs):
            validation = kwargs["eval_set"]
            self.assertEqual(X.num_row(), len(parts["train"]))
            self.assertEqual(validation.num_row(), len(parts["validation"]))
            self.assertEqual(list(X.get_label()), [r["label"]["value"] for r in parts["train"]])
            self.assertEqual(list(validation.get_label()), [r["label"]["value"] for r in parts["validation"]])
            calls.append(X.num_col())
            return original_fit(model, X, *args, **kwargs)

        with tempfile.TemporaryDirectory(prefix="aeza-event-training-") as temp:
            output = Path(temp) / "result"
            with patch.object(CatBoostClassifier, "fit", new=fit_spy):
                report = training.run(self.prepared, source, previous, output)
            self.assertEqual(calls, [36, 24])
            predictions = [json.loads(line) for line in (output / "all-predictions.jsonl").read_text().splitlines()]
            self.assertEqual(len(predictions), len(self.rows))
            self.assertEqual({p["sampleId"] for p in predictions}, {r["sampleId"] for r in self.rows})
            for item, row in zip(predictions, self.rows):
                self.assertEqual(item["actualLabel"], row["label"]["value"])
                self.assertEqual(item["usedForTraining"], row["split"] == "train")
                for key in ("newEnhanced36", "refitLegacy24", "frozenLegacy24", "constantNewTrain", "constantOldTrain"):
                    self.assertTrue(0 <= item[key] <= 1)
            prior = json.loads((ML / "runs/event-driven-v3-logloss/report.json").read_text())
            self.assertEqual(report["splits"], prior["splits"])
            self.assertEqual(report["allPredictionsSha256"], prior["allPredictionsSha256"])
            with self.assertRaises(ValueError):
                training.run(self.prepared, source, previous, output)
        self.assertEqual(hashes, [training.digest(p) for p in frozen])


if __name__ == "__main__":
    unittest.main()
