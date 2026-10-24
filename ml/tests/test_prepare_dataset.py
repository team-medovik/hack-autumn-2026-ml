"""Narrow point-in-time checks; synthetic data prove mechanics, not ML quality."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_dataset import (DEFAULT_HISTORY, FEATURES, GRACE, HORIZON, ROOT,
                             History, assign_split, build_dataset, timestamp)


PREDICTION = "2026-01-27T15:00:00Z"
DUE = "2026-01-31T15:00:00Z"
DEADLINE = "2026-02-05T15:00:00Z"


class ExampleHistory:
    """Small explicit histories make timing and ownership boundaries inspectable."""

    def __init__(self, as_of="2026-03-31T15:00:00Z"):
        self.data = {
            "asOf": as_of,
            "accountBusiness": [{"accountId": owner, "createdAt": "2025-01-01T15:00:00Z"}
                                for owner in (1, 2)],
            "events": [],
        }
        self.service(1, 1, "2026-01-01T15:00:00Z", DUE)

    def event(self, kind, at, sid=1, owner=1, recorded=None, **details):
        event = {"id": f"event-{len(self.data['events']):04d}", "type": kind,
                 "serviceId": sid, "accountId": owner, "occurredAt": at,
                 "recordedAt": recorded or at, **details}
        self.data["events"].append(event)
        return event

    def service(self, sid, owner, created, due, price=1000):
        self.event("service_created", created, sid, owner)
        self.event("contract_changed", created, sid, owner, tariffCode="TEST-VPS",
                   priceMinor=price, paymentTerm="month", currency="EUR", contractVersion=1)
        self.event("status_changed", created, sid, owner, fromStatus=None, toStatus="active")
        self.payment(created, created, due, sid, owner)

    def payment(self, at, due=DUE, through="2026-03-01T15:00:00Z", sid=1, owner=1,
                recorded=None):
        return self.event("payment_succeeded", at, sid, owner, recorded,
                          dueAt=due, paidThrough=through)

    def rows(self, at=PREDICTION):
        return History(self.data).features_at(timestamp(at))

    def label(self):
        history = History(self.data)
        return history.label(history.features_at(timestamp(PREDICTION))[0])


class PointInTimeFeatureTests(unittest.TestCase):
    def test_future_append_change_and_delete_do_not_change_features(self):
        fixture = ExampleHistory()
        original = fixture.rows()
        self.assertEqual(fixture.label()["value"], 1)
        payment = fixture.payment("2026-02-01T15:00:00Z")
        fixture.event("status_changed", "2026-02-02T15:00:00Z", toStatus="deleted")
        self.assertEqual(fixture.rows(), original)
        self.assertEqual(fixture.label()["value"], 0)
        payment["occurredAt"] = payment["recordedAt"] = "2026-02-06T15:00:00Z"
        self.assertEqual(fixture.rows(), original)
        self.assertEqual(fixture.label()["value"], 1)
        fixture.data["events"] = fixture.data["events"][:-2]
        self.assertEqual(fixture.rows(), original)

    def test_past_event_recorded_after_prediction_is_hidden(self):
        fixture = ExampleHistory()
        original = fixture.rows()
        fixture.event("contract_changed", "2026-01-20T15:00:00Z",
                      recorded="2026-02-01T15:00:00Z", tariffCode="FUTURE-KNOWLEDGE",
                      priceMinor=999999, paymentTerm="month", currency="EUR")
        fixture.event("payment_failed", "2026-01-22T15:00:00Z",
                      recorded="2026-02-01T15:00:00Z", dueAt=DUE)
        self.assertEqual(fixture.rows(), original)

    def test_delayed_backdated_creation_cannot_change_observation_completeness(self):
        fixture = ExampleHistory()
        original = fixture.rows()
        fixture.event("service_created", "2025-01-01T15:00:00Z", sid=2,
                      recorded="2026-02-01T15:00:00Z")
        self.assertEqual(fixture.rows(), original)

    def test_no_prior_renewals_are_unknown_not_zero_failure(self):
        features = ExampleHistory().rows()[0]["features"]
        self.assertEqual(features["renewal_cycles_90d"], 0)
        self.assertEqual(features["has_renewal_history"], 0)
        for key in ("failed_cycle_share_90d", "late_cycle_share_90d",
                    "mean_paid_delay_hours_90d", "max_paid_delay_hours_90d",
                    "days_since_successful_renewal", "portfolio_change_30d_minor"):
            with self.subTest(feature=key):
                self.assertIsNone(features[key])

    def test_repeated_failed_attempts_are_one_renewal_cycle(self):
        fixture = ExampleHistory()
        old_due = "2026-01-10T15:00:00Z"
        fixture.data["events"][-1]["paidThrough"] = old_due
        for date in ("2026-01-10T15:00:00Z", "2026-01-11T15:00:00Z",
                     "2026-01-12T14:00:00Z"):
            fixture.event("payment_failed", date, dueAt=old_due)
        fixture.payment("2026-01-12T15:00:00Z", due=old_due, through=DUE)
        features = fixture.rows()[0]["features"]
        self.assertEqual(features["renewal_cycles_90d"], 1)
        self.assertEqual(features["successful_renewals_90d"], 1)
        self.assertEqual(features["failed_cycle_share_90d"], 1)
        self.assertEqual(features["late_cycle_share_90d"], 1)
        self.assertEqual(features["mean_paid_delay_hours_90d"], 48)
        self.assertEqual(features["has_renewal_history"], 1)

    def test_status_eligibility_only_active_and_suspended(self):
        for status in ("active", "suspended", "prolong_wait", "deleted", "blocked"):
            with self.subTest(status=status):
                fixture = ExampleHistory()
                fixture.event("status_changed", "2026-01-26T15:00:00Z", toStatus=status)
                self.assertEqual(len(fixture.rows()), int(status in {"active", "suspended"}))

    def test_expiry_and_horizon_boundaries(self):
        for due, expected in ((PREDICTION, 0), ("2026-01-27T14:59:59Z", 0),
                              ("2026-01-27T15:00:01Z", 1),
                              ("2026-02-03T15:00:00Z", 1),
                              ("2026-02-03T15:00:01Z", 0)):
            with self.subTest(due=due):
                fixture = ExampleHistory()
                fixture.data["events"][-1]["paidThrough"] = due
                self.assertEqual(len(fixture.rows()), expected)


class OutcomeLabelTests(unittest.TestCase):
    def test_payment_at_deadline_included_one_second_later_excluded(self):
        for at, expected in ((DEADLINE, 0), ("2026-02-05T15:00:01Z", 1)):
            with self.subTest(at=at):
                fixture = ExampleHistory()
                fixture.payment(at)
                self.assertEqual(fixture.label()["value"], expected)

    def test_incomplete_window_remains_pending_even_after_observed_success(self):
        fixture = ExampleHistory(as_of="2026-02-04T15:00:00Z")
        fixture.payment("2026-02-01T15:00:00Z")
        label = fixture.label()
        self.assertIsNone(label["value"])
        self.assertIsNone(label["availableAt"])
        self.assertEqual(label["reason"], "observation_incomplete")

    def test_explicit_same_owner_replacement_including_downgrade_is_not_loss(self):
        for price in (500, 2000):
            with self.subTest(price=price):
                fixture = ExampleHistory()
                fixture.service(2, 1, "2026-02-01T15:00:00Z", "2026-03-01T15:00:00Z", price)
                fixture.event("replacement_linked", "2026-02-01T15:00:00Z", sid=2,
                              replacesServiceId=1)
                label = fixture.label()
                self.assertEqual(label["value"], 0)
                self.assertEqual(label["reason"], "confirmed_replacement")

    def test_new_purchase_without_replacement_link_is_still_loss(self):
        fixture = ExampleHistory()
        fixture.service(2, 1, "2026-02-01T15:00:00Z", "2026-03-01T15:00:00Z")
        self.assertEqual(fixture.label()["value"], 1)

    def test_cross_owner_replacement_cannot_clear_loss(self):
        fixture = ExampleHistory()
        fixture.service(2, 2, "2026-02-01T15:00:00Z", "2026-03-01T15:00:00Z")
        fixture.event("replacement_linked", "2026-02-01T15:00:00Z", sid=2, owner=2,
                      replacesServiceId=1)
        self.assertEqual(fixture.label()["value"], 1)

    def test_replacement_requires_existing_target_and_not_self_link(self):
        for sid in (1, 99):
            with self.subTest(sid=sid):
                fixture = ExampleHistory()
                fixture.event("replacement_linked", "2026-02-01T15:00:00Z", sid=sid,
                              replacesServiceId=1)
                self.assertEqual(fixture.label()["value"], 1)

    def test_delayed_recording_delays_label_availability(self):
        fixture = ExampleHistory()
        fixture.payment("2026-02-01T15:00:00Z", recorded="2026-02-10T15:00:00Z")
        label = fixture.label()
        self.assertEqual(label["value"], 0)
        self.assertEqual(label["availableAt"], "2026-02-10T15:00:00Z")

    def test_delayed_replacement_creation_delays_label_availability(self):
        fixture = ExampleHistory()
        creation = fixture.event("service_created", "2026-02-01T15:00:00Z", sid=2,
                                 recorded="2026-02-10T15:00:00Z")
        link = fixture.event("replacement_linked", "2026-02-01T15:00:00Z", sid=2,
                             replacesServiceId=1)
        label = fixture.label()
        self.assertEqual(label["value"], 0)
        self.assertEqual(label["availableAt"], "2026-02-10T15:00:00Z")
        self.assertTrue({creation["id"], link["id"]} <= set(label["evidenceEventIds"]))

    def test_early_payment_recorded_later_preserves_features_and_purges_label(self):
        validation, test = map(timestamp, ("2026-02-08T15:00:00Z", "2026-03-01T15:00:00Z"))
        for occurred in ("2026-01-26T15:00:00Z", PREDICTION):
            with self.subTest(occurred=occurred):
                fixture = ExampleHistory()
                original = fixture.rows()
                row = original[0]
                row["label"] = fixture.label()
                self.assertEqual(assign_split(row, validation, test), "train")
                payment = fixture.payment(occurred, recorded="2026-02-10T15:00:00Z")
                # The payment happened already, but the scoring system cannot see it yet.
                without_label = {key: value for key, value in row.items() if key != "label"}
                self.assertEqual(fixture.rows(), [without_label])
                label = fixture.label()
                self.assertEqual(label["value"], 0)
                self.assertEqual(label["reason"], "renewed_in_grace")
                self.assertGreaterEqual(timestamp(label["availableAt"]), timestamp(payment["recordedAt"]))
                row["label"] = label
                self.assertEqual(assign_split(row, validation, test), "purged")

    def test_early_replacement_recorded_later_preserves_features_and_purges_label(self):
        validation, test = map(timestamp, ("2026-02-08T15:00:00Z", "2026-03-01T15:00:00Z"))
        for occurred in ("2026-01-26T15:00:00Z", PREDICTION):
            with self.subTest(occurred=occurred):
                fixture = ExampleHistory()
                fixture.service(2, 1, "2026-01-25T15:00:00Z", "2026-03-01T15:00:00Z", price=500)
                original = fixture.rows()
                self.assertEqual(len(original), 1)
                link = fixture.event("replacement_linked", occurred, sid=2,
                                     recorded="2026-02-10T15:00:00Z", replacesServiceId=1)
                self.assertEqual(fixture.rows(), original)
                row = original[0]
                label = fixture.label()
                self.assertEqual(label["value"], 0)
                self.assertEqual(label["reason"], "confirmed_replacement")
                self.assertGreaterEqual(timestamp(label["availableAt"]), timestamp(link["recordedAt"]))
                row["label"] = label
                self.assertEqual(assign_split(row, validation, test), "purged")


class SplitBoundaryTests(unittest.TestCase):
    def test_label_available_exact_boundary_is_purged(self):
        validation, test = map(timestamp, ("2026-02-01T15:00:00Z", "2026-03-01T15:00:00Z"))
        for at, available, expected in (
                (PREDICTION, "2026-02-01T14:59:59Z", "train"),
                (PREDICTION, "2026-02-01T15:00:00Z", "purged"),
                ("2026-02-01T15:00:00Z", "2026-03-01T14:59:59Z", "validation"),
                ("2026-02-01T15:00:00Z", "2026-03-01T15:00:00Z", "purged"),
                ("2026-03-01T15:00:00Z", "2026-03-05T15:00:00Z", "test")):
            with self.subTest(at=at, available=available):
                row = {"predictionAt": at, "label": {"value": 1, "availableAt": available,
                                                     "reason": "no_renewal_by_grace"}}
                self.assertEqual(assign_split(row, validation, test), expected)
        pending = {"predictionAt": PREDICTION, "label": {"value": None, "availableAt": None,
                                                         "reason": "observation_incomplete"}}
        self.assertEqual(assign_split(pending, validation, test), "pending")


class NonfinancialBlockExclusionTests(unittest.TestCase):
    REASON = "synthetic_policy_review_not_payment_failure"

    def block(self, fixture, occurred="2026-02-01T15:00:00Z", **overrides):
        details = {"toStatus": "blocked", "reason": self.REASON, **overrides}
        return fixture.event("status_changed", occurred, **details)

    def test_confirmed_nonfinancial_block_is_excluded_without_feature_leakage(self):
        fixture = ExampleHistory()
        original = fixture.rows()
        block = self.block(fixture)
        self.assertEqual(fixture.rows(), original)
        label = fixture.label()
        self.assertIsNone(label["value"])
        self.assertEqual(label["reason"], "excluded_nonfinancial_block")
        self.assertEqual(label["evidenceEventIds"], [block["id"]])
        row = {**original[0], "label": label}
        # Explicit exclusion wins regardless of where the time split falls.
        for validation, test in (("2026-02-10T15:00:00Z", "2026-03-01T15:00:00Z"),
                                 ("2026-01-20T15:00:00Z", "2026-02-10T15:00:00Z"),
                                 ("2026-01-10T15:00:00Z", "2026-01-20T15:00:00Z")):
            with self.subTest(validation=validation, test=test):
                self.assertEqual(assign_split(row, timestamp(validation), timestamp(test)), "excluded")

    def test_exclusion_waits_for_full_outcome_window(self):
        fixture = ExampleHistory(as_of="2026-02-04T15:00:00Z")
        self.block(fixture)
        fixture.payment("2026-02-02T15:00:00Z")
        label = fixture.label()
        self.assertIsNone(label["value"])
        self.assertEqual(label["reason"], "observation_incomplete")
        row = {**fixture.rows()[0], "label": label}
        self.assertEqual(assign_split(row, timestamp("2026-02-10T15:00:00Z"),
                                      timestamp("2026-03-01T15:00:00Z")), "pending")

    def test_nonfinancial_block_exclusion_takes_precedence_over_successful_renewal(self):
        fixture = ExampleHistory()
        fixture.payment(DUE)
        block = self.block(fixture)
        label = fixture.label()
        self.assertIsNone(label["value"])
        self.assertEqual(label["reason"], "excluded_nonfinancial_block")
        self.assertEqual(label["evidenceEventIds"], [block["id"]])

    def test_delayed_block_recording_affects_availability_not_past_features(self):
        fixture = ExampleHistory()
        original = fixture.rows()
        block = self.block(fixture, recorded="2026-02-10T15:00:00Z")
        self.assertEqual(fixture.rows(), original)
        label = fixture.label()
        self.assertEqual(label["reason"], "excluded_nonfinancial_block")
        self.assertIsNone(label["value"])
        self.assertGreaterEqual(timestamp(label["availableAt"]), timestamp(block["recordedAt"]))
        # The same event cannot exclude a row in an earlier extract that has not received it.
        fixture.data["asOf"] = "2026-02-06T15:00:00Z"
        self.assertEqual(fixture.label()["value"], 1)

    def test_exclusion_window_includes_active_block_before_prediction_and_deadline(self):
        for occurred, excluded in (("2026-01-27T14:59:59Z", True), (PREDICTION, True),
                                   (DEADLINE, True), ("2026-02-05T15:00:01Z", False)):
            with self.subTest(occurred=occurred):
                fixture = ExampleHistory()
                self.block(fixture, occurred, recorded="2026-02-10T15:00:00Z")
                label = fixture.label()
                if excluded:
                    self.assertIsNone(label["value"])
                    self.assertEqual(label["reason"], "excluded_nonfinancial_block")
                else:
                    self.assertEqual(label["value"], 1)

    def test_late_recorded_block_already_active_at_prediction_excludes_label(self):
        fixture = ExampleHistory()
        original = fixture.rows()
        block = self.block(fixture, "2026-01-26T15:00:00Z", recorded="2026-02-10T15:00:00Z")
        self.assertEqual(fixture.rows(), original)
        label = fixture.label()
        self.assertIsNone(label["value"])
        self.assertEqual(label["reason"], "excluded_nonfinancial_block")
        self.assertIn(block["id"], label["evidenceEventIds"])
        self.assertGreaterEqual(timestamp(label["availableAt"]), timestamp(block["recordedAt"]))
        self.assertEqual(assign_split({**original[0], "label": label},
                                      timestamp("2026-02-08T15:00:00Z"),
                                      timestamp("2026-03-01T15:00:00Z")), "excluded")

    def test_old_block_restored_before_prediction_does_not_exclude_label(self):
        fixture = ExampleHistory()
        fixture.event("status_changed", "2026-01-26T15:00:00Z", toStatus="active")
        original = fixture.rows()
        self.block(fixture, "2026-01-25T15:00:00Z", recorded="2026-02-10T15:00:00Z")
        self.assertEqual(fixture.rows(), original)
        label = fixture.label()
        self.assertEqual(label["value"], 1)
        self.assertEqual(label["reason"], "no_renewal_by_grace")

    def test_unconfirmed_financial_or_missing_reason_does_not_exclude(self):
        for reason in (None, "insufficient_funds", "payment_failed", "unknown_policy"):
            with self.subTest(reason=reason):
                fixture = ExampleHistory()
                block = self.block(fixture, reason=reason)
                if reason is None:
                    del block["reason"]
                self.assertEqual(fixture.label()["value"], 1)

    def test_other_owner_service_or_status_does_not_exclude(self):
        for overrides in ({"owner": 2}, {"sid": 2}, {"toStatus": "suspended"}):
            with self.subTest(overrides=overrides):
                fixture = ExampleHistory()
                self.block(fixture, **overrides)
                self.assertEqual(fixture.label()["value"], 1)


class EnrichedDatasetIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(DEFAULT_HISTORY.read_text())
        cls.validation = timestamp("2026-08-01T15:00:00Z")
        cls.test = timestamp("2026-09-01T15:00:00Z")
        cls.rows = build_dataset(cls.data, cls.validation, cls.test)

    def test_enriched_rows_schema_time_and_disjoint_cycles(self):
        self.assertGreater(len(self.rows), 0)
        self.assertEqual(len(self.rows), len({row["sampleId"] for row in self.rows}))
        cycles, weights = {}, {}
        for row in self.rows:
            at, due = timestamp(row["predictionAt"]), timestamp(row["dueAt"])
            self.assertEqual(set(row["features"]), set(FEATURES))
            self.assertLess(at, due)
            self.assertLessEqual(due, at + HORIZON)
            self.assertEqual(timestamp(row["labelDeadline"]), due + GRACE)
            for key in ("maxFeatureOccurredAt", "maxFeatureRecordedAt"):
                self.assertLessEqual(timestamp(row["audit"][key]), at)
            self.assertIn(row["features"]["status"], {"active", "suspended"})
            split = row["split"]
            if split in {"train", "validation", "test"}:
                self.assertIn(row["label"]["value"], (0, 1))
                cycles.setdefault(row["cycleId"], set()).add(split)
                if split in {"train", "validation"}:
                    boundary = self.validation if split == "train" else self.test
                    self.assertLess(timestamp(row["label"]["availableAt"]), boundary)
            if split == "pending":
                self.assertIsNone(row["label"]["value"])
            key = (split, row["cycleId"])
            weights[key] = weights.get(key, 0) + row["cycleWeight"]
        self.assertTrue(all(len(splits) == 1 for splits in cycles.values()))
        for weight in weights.values():
            self.assertAlmostEqual(weight, 1, places=8)
        self.assertTrue({"train", "validation", "test", "pending", "purged"}
                        <= {row["split"] for row in self.rows})
        self.assertTrue({0, 1, None} <= {row["label"]["value"] for row in self.rows})

    def test_confirmed_nonfinancial_blocks_never_enter_training_or_evaluation(self):
        excluded = [row for row in self.rows if row["label"]["reason"] == "excluded_nonfinancial_block"]
        self.assertGreater(len(excluded), 0)
        events = {event["id"]: event for event in self.data["events"]}
        for row in excluded:
            self.assertEqual(row["split"], "excluded")
            self.assertIsNone(row["label"]["value"])
            self.assertGreater(len(row["label"]["evidenceEventIds"]), 0)
            for event_id in row["label"]["evidenceEventIds"]:
                event = events[event_id]
                self.assertEqual(event["type"], "status_changed")
                self.assertEqual(event["toStatus"], "blocked")
                self.assertEqual(event["reason"], NonfinancialBlockExclusionTests.REASON)
                self.assertEqual(event["accountId"], row["accountId"])
                self.assertEqual(event["serviceId"], row["serviceId"])
                self.assertLessEqual(timestamp(row["predictionAt"]), timestamp(event["occurredAt"]))
                self.assertLessEqual(timestamp(event["occurredAt"]), timestamp(row["labelDeadline"]))
        for row in self.rows:
            if row["split"] in {"train", "validation", "test"}:
                self.assertNotEqual(row["label"]["reason"], "excluded_nonfinancial_block")

    def test_final_snapshot_contracts_scenarios_and_catalog_are_not_features(self):
        modified = copy.deepcopy(self.data)
        for key in ("syntheticSnapshot", "currentContracts", "generatorScenariosDoNotTrainOn",
                    "sourceServiceTariffHypotheses", "catalog"):
            modified[key] = {"poison": "must never be used as historical features"}
        at = timestamp("2026-09-15T15:00:00Z")
        original_history, modified_history = History(self.data), History(modified)
        original_rows = original_history.features_at(at)
        modified_rows = modified_history.features_at(at)
        self.assertGreater(len(original_rows), 0)
        self.assertEqual(original_rows, modified_rows)
        self.assertEqual([original_history.label(row) for row in original_rows],
                         [modified_history.label(row) for row in modified_rows])

    def test_original_case_fixture_hash_unchanged(self):
        original = ROOT / "src/test/resources/risk/data.json"
        expected = "6e0e224671a4cd22059431b01fb6fed21e075100bb21b1545d94e746e666292e"
        self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(), expected)
        self.assertEqual(self.data["sourceSnapshot"]["sha256"], expected)


if __name__ == "__main__":
    unittest.main()
