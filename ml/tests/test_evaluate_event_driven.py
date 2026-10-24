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
import evaluate_event_driven as evaluation


def row(label, cycle, weight=1.0, days=3):
    features = {name: 0 for name in evaluation.OLD_FEATURES}
    features["days_to_expiry"] = days
    return {"cycleId": cycle, "label": {"value": label}, "cycleWeight": weight,
            "predictionAt": "2027-08-01T15:00:00Z", "legacy_features": features,
            "features": {**features, "extra_observed": 999}, "diagnostics": {"hidden": "never use"}}


class EvaluationTests(unittest.TestCase):
    def test_bridge_selects_exact_legacy_features_without_mutating_input(self):
        original = [row(1, "a")]
        before = copy.deepcopy(original)
        converted = evaluation.legacy_rows(original)
        self.assertEqual(original, before)
        self.assertEqual(list(converted[0]["features"]), list(evaluation.OLD_FEATURES))
        self.assertNotIn("extra_observed", converted[0]["features"])
        self.assertNotIn("diagnostics", converted[0]["features"])

    def test_bridge_rejects_wrong_order_or_disagreeing_views(self):
        sample = row(1, "a")
        sample["legacy_features"] = dict(reversed(list(sample["legacy_features"].items())))
        with self.assertRaises(ValueError):
            evaluation.legacy_rows([sample])
        sample = row(1, "a")
        sample["features"]["days_to_expiry"] = 999
        with self.assertRaises(ValueError):
            evaluation.legacy_rows([sample])

    def test_probabilities_must_be_finite_bounded_and_aligned(self):
        for probabilities in ([math.nan], [math.inf], [-.1], [1.01], [], [.1, .2]):
            with self.subTest(values=probabilities), self.assertRaises(ValueError):
                evaluation.checked_probabilities(probabilities, 1)

    def test_auc_perfect_reversed_tied_and_one_class(self):
        samples = [row(0, "a"), row(1, "b")]
        self.assertEqual(evaluation.auc(samples, [.1, .9]), 1)
        self.assertEqual(evaluation.auc(samples, [.9, .1]), 0)
        self.assertEqual(evaluation.auc(samples, [.1, .1]), .5)
        self.assertIsNone(evaluation.auc([samples[0]], [.1]))

    def test_metrics_do_not_overweight_repeated_snapshots_of_one_cycle(self):
        single = evaluation.summarize([row(0, "a"), row(1, "b")], [.1, .8])
        repeated = evaluation.summarize([row(0, "a", .5), row(0, "a", .5), row(1, "b")], [.1, .1, .8])
        for key in ("logloss", "brier", "meanProbability", "lossRate", "rocAuc"):
            self.assertAlmostEqual(single[key], repeated[key])
        with self.assertRaises(ValueError):
            evaluation.summarize([row(0, "a"), row(0, "a")], [.1, .1])

    def test_example_snapshot_selection_does_not_use_outcome(self):
        samples = [row(1, "a", days=6), row(0, "a", days=3), row(1, "b", days=2)]
        self.assertEqual(evaluation.representative_indices(samples), [1, 2])
        for sample in samples:
            sample["label"]["value"] = 1 - sample["label"]["value"]
        self.assertEqual(evaluation.representative_indices(samples), [1, 2])

    @unittest.skipUnless(importlib.util.find_spec("catboost") is not None and
                         (ML / "runs/behavior-v1/model.cbm").exists() and
                         (ML / "prepared-event-driven/lifecycle-v3/rows.jsonl").exists(),
                         "Local frozen models, prepared rows, and CatBoost required")
    def test_real_frozen_inference_never_fits_and_preserves_models_and_artifacts(self):
        prepared = ML / "prepared-event-driven/lifecycle-v3"
        source = ML.parent / "examples/risk/event-driven/generated/lifecycle-v3-seed20261024/observations.json"
        tracked = [source, prepared / "rows.jsonl", prepared / "manifest.json",
                   ML / "runs/generator-v2-logloss/model.cbm", ML / "runs/behavior-v1/model.cbm"]
        hashes = [evaluation.digest(p) for p in tracked]
        with tempfile.TemporaryDirectory(prefix="aeza-frozen-evaluation-") as directory:
            output = Path(directory) / "result"
            with patch("catboost.CatBoostClassifier.fit", side_effect=AssertionError("Must never fit")):
                report = evaluation.evaluate(prepared, source, ML / "runs", output)
            self.assertFalse(report["trainingStarted"])
            predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual(len(predictions), 2 * (805 + 561))
            self.assertEqual({p["split"] for p in predictions}, {"validation", "test"})
            self.assertTrue(all(0 <= p["lossProbability"] <= 1 for p in predictions))
            with self.assertRaises(ValueError):
                evaluation.evaluate(prepared, source, ML / "runs", output)
        self.assertEqual(hashes, [evaluation.digest(p) for p in tracked])


if __name__ == "__main__":
    unittest.main()
