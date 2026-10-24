"""Selection and interpretation checks, not assertions about improved model quality."""
import copy
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compare_generators as comparison
from prepare_dataset import FEATURES


def example(cycle, remaining, label):
    return {"cycleId": cycle, "serviceId": cycle, "predictionAt": "2026-08-02T15:00:00Z",
            "dueAt": "2026-08-05T15:00:00Z", "labelDeadline": "2026-08-10T15:00:00Z",
            "label": {"value": label}, "features": {**dict.fromkeys(FEATURES, 0),
                                                       "days_to_expiry": remaining}}


class ExplanationTests(unittest.TestCase):
    def test_subgroups_keep_whole_cycles_when_90_day_feature_expires(self):
        rows = [example(1, 6, 1), example(1, 3, 1), example(2, 3, 0)]
        rows[0]["features"]["recoveries_90d"] = 1
        rows[1]["features"]["recoveries_90d"] = 0
        rows[2]["features"]["recoveries_90d"] = 1
        for row, weight in zip(rows, (0.5, 0.5, 1)):
            row["cycleWeight"] = weight
        result = comparison.evaluate_groups(rows, {"model": [0.2, 0.2, 0.1]})
        self.assertEqual(result["all"]["cycles"], 2)
        self.assertEqual(result["all"]["lossCycles"], 1)
        self.assertEqual(result["all"]["metrics"]["model"]["lossRate"], 0.5)
        self.assertEqual(result["previous_recovery_90d"]["rows"], 1)
        self.assertEqual(result["previous_recovery_90d"]["cycles"], 1)
        self.assertEqual(result["previous_recovery_90d"]["lossCycles"], 0)

    def test_one_snapshot_per_cycle_chosen_without_outcome(self):
        rows = [example(1, 6, 1), example(1, 2.9, 1), example(1, 1, 1), example(2, 3.4, 0)]
        selected = comparison.representative_cycles(rows)
        self.assertEqual([r["features"]["days_to_expiry"] for r in selected], [2.9, 3.4])
        for row in rows:
            row["label"]["value"] = 1 - row["label"]["value"]
        self.assertEqual([r["features"]["days_to_expiry"] for r in comparison.representative_cycles(rows)], [2.9, 3.4])

    def test_explanation_reconstructs_probability_and_separates_future_evidence(self):
        class Model:
            def predict_proba(self, pool):
                # Minimal array-like adapter supports [:, 1].
                class Probabilities:
                    def __getitem__(self, key):
                        return [1 / (1 + math.exp(1))] * len(pool)
                return Probabilities()

            def get_feature_importance(self, pool, type):
                return [[0.0] * len(FEATURES) + [-1.0] for _ in pool]

            def predict(self, pool, prediction_type):
                return [-1.0] * len(pool)

        row = example(1, 3, 1)
        event = {"serviceId": 1, "type": "payment_succeeded", "dueAt": row["dueAt"],
                 "occurredAt": "2026-08-11T15:00:00Z"}
        history = {"events": [event]}
        with patch.object(comparison, "make_pool", side_effect=lambda rows: rows) as pool:
            result = comparison.explain_examples(Model(), [row], history)
        self.assertTrue(result["shapAdditivityChecked"])
        self.assertEqual(pool.call_args.args[0], [row])
        self.assertEqual(result["examples"][0]["paymentAfterOutcomeWindow"], [event])
        self.assertEqual(result["examples"][0]["eventsWithinOutcomeWindow"], [])
        wrong = Model()
        wrong.predict = lambda *args, **kwargs: [100]
        with patch.object(comparison, "make_pool", side_effect=lambda rows: rows):
            with self.assertRaises(AssertionError):
                comparison.explain_examples(wrong, [row], history)

    def test_cycle_selection_does_not_mutate_inputs(self):
        rows = [example(1, 3, 0)]
        before = copy.deepcopy(rows)
        comparison.representative_cycles(rows)
        self.assertEqual(rows, before)


if __name__ == "__main__":
    unittest.main()
