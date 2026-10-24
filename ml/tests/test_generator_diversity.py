"""Synthetic coverage and invariants, never a requirement to improve model metrics."""
import copy
import hashlib
from collections import defaultdict
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "examples/risk/vps-business"))
from generate import SEED, SOURCE, SOURCE_SHA256, build
from validate import validate
from prepare_dataset import GRACE, History, build_dataset, timestamp


class GeneratorDiversityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validation_start = timestamp("2026-08-01T15:00:00Z")
        cls.test_start = timestamp("2026-09-01T15:00:00Z")
        cls.data = build(seed=SEED)
        cls.history = History(cls.data)
        cls.rows = build_dataset(cls.data, cls.validation_start, cls.test_start)
        cls.by_service = defaultdict(list)
        for event in cls.data["events"]:
            cls.by_service[event["serviceId"]].append(event)
        cls.initial_due = {}
        for sid, events in cls.by_service.items():
            payments = [event for event in events if event["type"] == "payment_succeeded"]
            if payments and payments[0]["paymentTerm"] == "month":
                cls.initial_due[sid] = payments[0]["paidThrough"]

    def test_build_is_reproducible_and_seed_changes_behavior(self):
        self.assertEqual(self.data, build(seed=SEED))
        self.assertEqual(self.data, build())
        alternative = build(seed=SEED + 1)
        self.assertNotEqual(self.data["events"], alternative["events"])
        self.assertEqual(alternative["seed"], SEED + 1)
        # Seed variation must not sacrifice the existing event/snapshot contract.
        self.assertGreater(validate(alternative)["events"], 0)

    def test_original_case_fixture_is_unchanged_and_generated_contract_valid(self):
        self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).hexdigest(), SOURCE_SHA256)
        self.assertEqual(SOURCE_SHA256, "6e0e224671a4cd22059431b01fb6fed21e075100bb21b1545d94e746e666292e")
        self.assertEqual(self.data["sourceSnapshot"]["sha256"], SOURCE_SHA256)
        self.assertGreater(validate(self.data)["events"], 0)

    def test_first_monthly_renewal_has_both_outcomes_in_train_and_validation(self):
        for split in ("train", "validation"):
            with self.subTest(split=split):
                rows = [row for row in self.rows if row["split"] == split
                        and row["dueAt"] == self.initial_due.get(row["serviceId"])]
                self.assertEqual({row["label"]["value"] for row in rows}, {0, 1})
                self.assertTrue(all(row["features"]["has_renewal_history"] == 0 for row in rows))

    def test_stable_monthly_services_have_at_least_twelve_paid_renewals_before_validation(self):
        stable = []
        for sid, events in self.by_service.items():
            previous = [event for event in events if timestamp(event["occurredAt"]) < self.validation_start]
            payments = [event for event in previous if event["type"] == "payment_succeeded"
                        and event["paymentTerm"] == "month"]
            if len(payments) < 13:  # Initial purchase is not a renewal.
                continue
            if any(event["type"] == "payment_failed" for event in previous):
                continue
            if all(timestamp(event["occurredAt"]) == timestamp(event["dueAt"]) for event in payments):
                stable.append(sid)
        self.assertGreater(len(stable), 0)
        self.assertTrue(any(row["serviceId"] in stable and row["split"] == "validation" for row in self.rows))

    def test_previous_delay_and_recovery_do_not_determine_future_outcome(self):
        for split in ("train", "validation"):
            for feature in ("mean_paid_delay_hours_90d", "recoveries_90d"):
                with self.subTest(split=split, feature=feature):
                    rows = [row for row in self.rows if row["split"] == split
                            and row["features"]["payment_term"] == "month"
                            and (row["features"][feature] or 0) > 0]
                    self.assertEqual({row["label"]["value"] for row in rows}, {0, 1})

    def test_payment_delay_buckets_match_grace_labels(self):
        buckets = {"paid_within_grace": [], "paid_after_grace_with_recovery": [], "never_paid": []}
        # One row per cycle is enough: repetitions before the same due date add no coverage.
        cycles = {row["cycleId"]: row for row in self.rows
                  if row["split"] in {"train", "validation"}
                  and row["features"]["payment_term"] == "month"}
        for row in cycles.values():
            events = self.by_service[row["serviceId"]]
            attempts = [event for event in events if event["type"] in {"payment_succeeded", "payment_failed"}
                        and event["dueAt"] == row["dueAt"]]
            if not any(event["type"] == "payment_failed" for event in attempts):
                continue
            successes = [event for event in attempts if event["type"] == "payment_succeeded"]
            if not successes:
                buckets["never_paid"].append(row)
                self.assertEqual(row["label"]["value"], 1)
                continue
            first = min(successes, key=lambda event: timestamp(event["occurredAt"]))
            delay = timestamp(first["occurredAt"]) - timestamp(row["dueAt"])
            if delay.total_seconds() <= 0:
                continue
            if delay <= GRACE:
                buckets["paid_within_grace"].append(row)
                self.assertEqual(row["label"]["value"], 0)
            elif any(event["type"] == "service_recovered"
                     and event["occurredAt"] == first["occurredAt"] for event in events):
                buckets["paid_after_grace_with_recovery"].append(row)
                self.assertEqual(row["label"]["value"], 1)
            self.assertEqual(self.history.label(row), row["label"])
        for bucket, rows in buckets.items():
            with self.subTest(bucket=bucket):
                self.assertGreater(len(rows), 0)

    def test_generator_scenarios_cannot_change_features_or_labels(self):
        changed = copy.deepcopy(self.data)
        changed["generatorScenariosDoNotTrainOn"] = [{"serviceId": -1, "generatorScenario": "future-loss",
                                                      "useForTraining": True, "futureLabel": 1}]
        changed["provenance"]["behavior"] = "poisoned metadata, not a feature"
        modified = History(changed)
        at = timestamp("2026-08-15T15:00:00Z")
        original_rows = self.history.features_at(at)
        new_rows = modified.features_at(at)
        self.assertGreater(len(original_rows), 0)
        self.assertEqual(original_rows, new_rows)
        self.assertEqual([self.history.label(row) for row in original_rows],
                         [modified.label(row) for row in new_rows])


if __name__ == "__main__":
    unittest.main()
