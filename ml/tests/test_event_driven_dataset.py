"""Independent event-adapter contracts; these checks do not prove ML quality."""
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_dataset import FEATURES, iso, timestamp
from prepare_event_driven import EventHistory, build_dataset, manifest


START = "2026-01-01T00:00:00Z"
DUE = "2026-02-01T00:00:00Z"
PREDICTION = "2026-01-28T00:00:00Z"
DEADLINE = "2026-02-06T00:00:00Z"


class EventFixture:
    def __init__(self, term="month", as_of="2026-04-01T00:00:00Z"):
        self.data = {"schemaVersion": "vps-billing-events-v2", "asOf": as_of,
                     "events": []}
        self.account(1)
        self.account(2)
        self.top_up(START, 5000)
        due = {"month": DUE, "day": "2026-01-02T00:00:00Z",
               "hour": "2026-01-01T01:00:00Z"}[term]
        self.service(START, due=due, term=term)

    def event(self, kind, at, sid=None, aid=1, recorded=None, **fields):
        event = {"id": f"test-{len(self.data['events']):06d}", "type": kind,
                 "occurredAt": at, "recordedAt": recorded or at,
                 "accountId": aid, "serviceId": sid, "currency": "EUR", **fields}
        self.data["events"].append(event)
        return event

    def account(self, aid):
        return self.event("account_created", START, aid=aid,
                          email=f"test-{aid}@example.test")

    def top_up(self, at, amount, aid=1, recorded=None):
        return self.event("wallet_topped_up", at, aid=aid, recorded=recorded,
                          amountMinor=amount)

    def service(self, at, sid=10, aid=1, due=DUE, term="month", price=1000):
        self.event("service_created", at, sid, aid, name=f"VPS {sid}",
                   paymentTerm=term, priceMinor=price, tariffCode="TEST-VPS",
                   autoProlong=True)
        return self.payment(at, sid=sid, aid=aid, due=at, through=due,
                            mode="purchase", price=price, term=term)

    def payment(self, at, sid=10, aid=1, due=DUE,
                through="2026-03-01T00:00:00Z", mode="manual", price=1000,
                term="month", recorded=None):
        return self.event("payment_succeeded", at, sid, aid, recorded,
                          dueAt=due, paidThrough=through, mode=mode,
                          amountMinor=price, paymentTerm=term)

    def restriction(self, at, sid=10, recorded=None, category="nonfinancial"):
        return self.event("service_restricted", at, sid, recorded=recorded,
                          category=category, reason="policy_review", toStatus="blocked")

    def release(self, at, sid=10):
        return self.event("service_released", at, sid, category="nonfinancial",
                          reason="policy_review", toStatus="active")

    def replacement(self, at="2026-02-02T00:00:00Z", due=DUE, target=11):
        return self.event("replacement_confirmed", at, 10,
                          replacementServiceId=target, dueAt=due)

    def rows(self, at=PREDICTION):
        return EventHistory(self.data).features_at(timestamp(at))

    def row(self, at=PREDICTION):
        return next(row for row in self.rows(at) if row["serviceId"] == 10)

    def label(self, at=PREDICTION):
        history = EventHistory(self.data)
        row = next(row for row in history.features_at(timestamp(at))
                   if row["serviceId"] == 10)
        return history.label(row)

    def dataset(self, validation="2026-02-15T00:00:00Z", test="2026-03-15T00:00:00Z"):
        return build_dataset(self.data, timestamp(validation), timestamp(test))


class EventFeatureTests(unittest.TestCase):
    def test_deleted_service_leaves_live_portfolio_and_funding_obligations(self):
        fixture = EventFixture()
        fixture.service(START, sid=11, due="2026-01-20T00:00:00Z", price=2000)
        fixture.event("status_changed", "2026-01-20T00:00:00Z", 11,
                      toStatus="prolong_wait")
        before = fixture.row("2026-01-25T00:00:00Z")
        fixture.event("status_changed", "2026-01-25T00:00:00Z", 11,
                      toStatus="deleted", reason="monthly_unpaid_retention_elapsed")
        after = fixture.row("2026-01-25T00:00:00Z")
        self.assertEqual(before["features"]["live_vps_count"], 2)
        self.assertEqual(after["features"]["live_vps_count"], 1)
        self.assertEqual(after["features"]["portfolio_monthly_revenue_minor"], 1000)
        self.assertEqual(before["features"]["next_renewals_7d_required_minor"], 3000)
        self.assertEqual(after["features"]["next_renewals_7d_required_minor"], 1000)
        self.assertEqual(before["features"]["service_projected_funding_ratio"], 0)
        self.assertEqual(after["features"]["service_projected_funding_ratio"], 2)

    def test_future_deletion_cannot_change_earlier_features(self):
        fixture = EventFixture()
        fixture.service(START, sid=11, due="2026-01-24T00:00:00Z", price=2000)
        fixture.event("status_changed", "2026-01-24T00:00:00Z", 11,
                      toStatus="prolong_wait")
        before = fixture.row()
        fixture.event("status_changed", "2026-01-29T00:00:00Z", 11,
                      toStatus="deleted", reason="monthly_unpaid_retention_elapsed")
        self.assertEqual(before, fixture.row())

    def test_account_age_uses_observed_registration_not_first_purchase(self):
        fixture = EventFixture()
        for event in fixture.data["events"]:
            if event["type"] == "account_created" and event["accountId"] == 1:
                event["occurredAt"] = event["recordedAt"] = "2025-12-15T00:00:00Z"
        features = fixture.row()["features"]
        self.assertEqual(features["service_age_days"], 27)
        self.assertEqual(features["account_age_days"], 44)

    def test_source_policy_and_registration_metadata_do_not_enter_features(self):
        fixture = EventFixture()
        before = fixture.row()
        fixture.data["assumptions"] = {"monthlyDeletionPolicy": {"days": 5},
                                       "registeredAt": "1900-01-01T00:00:00Z"}
        self.assertEqual(before, fixture.row())

    def test_future_events_latent_diagnostics_and_final_state_do_not_change_features(self):
        fixture = EventFixture()
        original = fixture.row()
        fixture.payment("2026-02-01T00:00:00Z")
        fixture.top_up("2026-02-01T00:00:00Z", 999999)
        fixture.event("auto_prolong_changed", "2026-02-02T00:00:00Z", 10, enabled=False)
        fixture.data.update(diagnostics={"archetype": "project", "needEndsAt": START},
                            snapshot={"accounts": [{"id": 1, "balance": 999999999}]},
                            plannedActions=[{"kind": "top_up", "amount": 9999999}])
        changed = fixture.row()
        for key in ("features", "legacy_features", "enhanced_features", "audit"):
            self.assertEqual(original[key], changed[key], key)
        self.assertEqual(fixture.label()["value"], 0)

    def test_backdated_event_is_not_available_before_recording(self):
        fixture = EventFixture()
        original = fixture.row()
        fixture.top_up("2026-01-20T00:00:00Z", 9999,
                       recorded="2026-01-29T00:00:00Z")
        fixture.event("auto_prolong_changed", "2026-01-20T00:00:00Z", 10,
                      recorded="2026-01-29T00:00:00Z", enabled=False)
        self.assertEqual(original["features"], fixture.row()["features"])
        self.assertNotEqual(original["enhanced_features"],
                            fixture.row("2026-01-29T00:00:00Z")["enhanced_features"])

    def test_other_owner_topup_cannot_change_service_finance(self):
        fixture = EventFixture()
        original = fixture.row()["enhanced_features"]
        fixture.top_up("2026-01-20T00:00:00Z", 999999, aid=2)
        self.assertEqual(original, fixture.row()["enhanced_features"])

    def test_wallet_features_use_observed_ledger_not_exported_balances(self):
        fixture = EventFixture()
        fixture.data["events"][2]["balanceAfterMinor"] = 999999
        fixture.top_up("2026-01-20T00:00:00Z", 300)
        fixture.top_up("2026-01-25T00:00:00Z", 700,
                       recorded="2026-01-29T00:00:00Z")
        values = fixture.row()["enhanced_features"]
        self.assertEqual(values["wallet_balance_minor"], 4300)
        self.assertEqual(values["topup_count_30d"], 2)
        self.assertEqual(values["topup_sum_30d_minor"], 5300)
        self.assertEqual(values["payment_sum_30d_minor"], 1000)
        self.assertEqual(values["current_service_funding_ratio"], 4.3)
        self.assertEqual(values["next_renewals_7d_required_minor"], 1000)
        self.assertEqual(values["portfolio_funding_ratio_7d"], 4.3)
        self.assertEqual(fixture.row("2026-01-29T00:00:00Z")["enhanced_features"]["wallet_balance_minor"], 5000)

    def test_auto_prolong_changes_only_after_known_event(self):
        fixture = EventFixture()
        self.assertEqual(fixture.row()["enhanced_features"]["auto_prolong"], 1)
        fixture.event("auto_prolong_changed", "2026-01-27T00:00:00Z", 10, enabled=False)
        values = fixture.row()["enhanced_features"]
        self.assertEqual(values["auto_prolong"], 0)
        self.assertEqual(values["days_since_auto_change"], 1)

    def test_incomplete_delayed_cash_ledger_is_unknown_not_zero(self):
        fixture = EventFixture()
        topup = next(event for event in fixture.data["events"]
                     if event["type"] == "wallet_topped_up")
        topup["recordedAt"] = "2026-01-29T00:00:00Z"
        values = fixture.row()["enhanced_features"]
        for name in ("wallet_balance_minor", "current_service_funding_ratio",
                     "portfolio_funding_ratio_7d", "service_projected_funding_ratio"):
            self.assertIsNone(values[name], name)
        self.assertEqual(values["topup_count_30d"], 0)

    def test_projected_wallet_does_not_reserve_unaffordable_prior_charge(self):
        fixture = EventFixture()
        fixture.top_up("2026-01-10T00:00:00Z", 10000)
        fixture.service("2026-01-10T00:00:00Z", sid=11, price=10000,
                        due="2026-01-31T00:00:00Z")
        values = fixture.row()["enhanced_features"]
        self.assertEqual(values["wallet_balance_minor"], 4000)
        self.assertEqual(values["next_renewals_7d_required_minor"], 11000)
        self.assertEqual(values["service_projected_funding_ratio"], 4)
        fixture.top_up("2026-01-30T00:00:00Z", 100000)
        self.assertEqual(fixture.row()["enhanced_features"], values)

    def test_legacy_24_features_exact_schema_and_scalar_types(self):
        row = EventFixture().row()
        self.assertEqual(tuple(row["legacy_features"]), FEATURES)
        self.assertEqual(len(row["legacy_features"]), 24)
        for name, value in row["legacy_features"].items():
            self.assertEqual(row["features"][name], value)
            if name in ("status", "payment_term", "tariff_code"):
                self.assertIsInstance(value, str)
            else:
                self.assertTrue(value is None or type(value) in (float, int), name)

    def test_purchase_is_not_renewal_and_unknown_history_is_not_success(self):
        features = EventFixture().row()["legacy_features"]
        self.assertEqual(features["renewal_cycles_90d"], 0)
        self.assertEqual(features["successful_renewals_90d"], 0)
        self.assertEqual(features["has_renewal_history"], 0)
        self.assertIsNone(features["failed_cycle_share_90d"])
        self.assertIsNone(features["days_since_successful_renewal"])

    def test_repeated_failures_are_counted_as_one_cycle(self):
        fixture = EventFixture()
        old_due = "2026-01-10T00:00:00Z"
        fixture.data["events"][-1]["paidThrough"] = old_due
        for day in (10, 11, 12):
            fixture.event("payment_failed", f"2026-01-{day}T00:00:00Z", 10,
                          dueAt=old_due, mode="manual", amountMinor=1000,
                          reason="insufficient_funds")
        fixture.payment("2026-01-12T00:00:00Z", due=old_due, through=DUE)
        features = fixture.row()["legacy_features"]
        self.assertEqual(features["renewal_cycles_90d"], 1)
        self.assertEqual(features["successful_renewals_90d"], 1)
        self.assertEqual(features["failed_cycle_share_90d"], 1)
        self.assertEqual(features["mean_paid_delay_hours_90d"], 48)

    def test_delayed_creation_does_not_extend_known_history(self):
        fixture = EventFixture()
        original = fixture.row()["features"]
        fixture.service("2025-01-01T00:00:00Z", sid=11,
                        due="2026-03-01T00:00:00Z")
        for event in fixture.data["events"][-2:]:
            event["recordedAt"] = "2026-02-20T00:00:00Z"
        self.assertEqual(original, fixture.row()["features"])

    def test_status_eligibility(self):
        for status in ("active", "suspended", "blocked", "prolong_wait", "deleted"):
            with self.subTest(status=status):
                fixture = EventFixture()
                fixture.event("status_changed", "2026-01-27T00:00:00Z", 10,
                              toStatus=status)
                self.assertEqual(len(fixture.rows()), int(status in {"active", "suspended"}))

    def test_prediction_window_is_positive_and_at_most_seven_days(self):
        for due, eligible in ((PREDICTION, False), ("2026-01-28T00:00:01Z", True),
                              ("2026-02-04T00:00:00Z", True),
                              ("2026-02-04T00:00:01Z", False)):
            with self.subTest(due=due):
                fixture = EventFixture()
                fixture.data["events"][-1]["paidThrough"] = due
                self.assertEqual(bool(fixture.rows()), eligible)

    def test_valid_confirmed_replacement_removes_old_prediction(self):
        fixture = EventFixture()
        fixture.service("2026-01-20T00:00:00Z", sid=11,
                        due="2026-02-20T00:00:00Z")
        fixture.replacement("2026-01-21T00:00:00Z")
        self.assertNotIn(10, {row["serviceId"] for row in fixture.rows()})


class EventLabelTests(unittest.TestCase):
    def test_120_hours_inclusive_121_hours_is_loss(self):
        for at, value in ((DEADLINE, 0), ("2026-02-06T01:00:00Z", 1)):
            with self.subTest(at=at):
                fixture = EventFixture()
                fixture.payment(at)
                self.assertEqual(fixture.label()["value"], value)

    def test_full_followup_required_even_for_early_success(self):
        fixture = EventFixture(as_of="2026-02-05T23:59:59Z")
        fixture.payment("2026-01-29T00:00:00Z")
        label = fixture.label()
        self.assertIsNone(label["value"])
        self.assertIsNone(label["availableAt"])
        self.assertEqual(label["reason"], "observation_incomplete")

    def test_late_recorded_success_delays_label_availability(self):
        fixture = EventFixture()
        fixture.payment("2026-02-02T00:00:00Z", recorded="2026-02-20T00:00:00Z")
        label = fixture.label()
        self.assertEqual(label["value"], 0)
        self.assertEqual(label["availableAt"], "2026-02-20T00:00:00Z")

    def test_recorded_after_asof_success_cannot_clear_loss(self):
        fixture = EventFixture(as_of="2026-02-10T00:00:00Z")
        fixture.payment("2026-02-02T00:00:00Z", recorded="2026-02-20T00:00:00Z")
        self.assertEqual(fixture.label()["value"], 1)

    def test_new_purchase_requires_paid_same_owner_confirmation(self):
        fixture = EventFixture()
        fixture.service("2026-02-01T00:00:00Z", sid=11,
                        due="2026-03-01T00:00:00Z")
        self.assertEqual(fixture.label()["value"], 1)
        fixture.replacement()
        self.assertEqual(fixture.label()["value"], 0)
        self.assertEqual(fixture.label()["reason"], "confirmed_replacement")

    def test_late_target_release_proof_delays_replacement_label(self):
        fixture = EventFixture()
        fixture.service("2026-02-01T00:00:00Z", sid=11,
                        due="2026-03-01T00:00:00Z")
        fixture.restriction("2026-02-01T01:00:00Z", sid=11)
        release = fixture.release("2026-02-01T02:00:00Z", sid=11)
        release["recordedAt"] = "2026-02-20T00:00:00Z"
        fixture.replacement()
        label = fixture.label()
        self.assertEqual(label["value"], 0)
        self.assertEqual(label["reason"], "confirmed_replacement")
        self.assertEqual(label["availableAt"], "2026-02-20T00:00:00Z")
        self.assertIn(release["id"], label["evidenceEventIds"])

    def test_unpaid_nonexistent_crossowner_self_or_late_replacement_cannot_clear_loss(self):
        for defect in ("missing", "unpaid", "other_owner", "self", "late", "expired", "blocked"):
            with self.subTest(defect=defect):
                fixture = EventFixture()
                if defect not in ("missing", "self"):
                    aid = 2 if defect == "other_owner" else 1
                    if aid == 2:
                        fixture.top_up("2026-01-30T00:00:00Z", 1000, aid=2)
                    fixture.service("2026-02-01T00:00:00Z", sid=11, aid=aid,
                                    due="2026-03-01T00:00:00Z")
                    if defect == "unpaid":
                        fixture.data["events"].pop()
                    elif defect == "expired":
                        fixture.data["events"][-1]["paidThrough"] = "2026-02-02T00:00:00Z"
                    elif defect == "blocked":
                        fixture.restriction("2026-02-01T12:00:00Z", sid=11)
                fixture.replacement(at="2026-02-06T00:00:01Z" if defect == "late"
                                    else "2026-02-02T00:00:00Z",
                                    target=10 if defect == "self" else 11)
                self.assertEqual(fixture.label()["value"], 1)

    def test_replacement_for_next_cycle_cannot_erase_prior_loss(self):
        fixture = EventFixture()
        fixture.payment("2026-02-08T00:00:00Z")
        fixture.service("2026-02-20T00:00:00Z", sid=11,
                        due="2026-03-20T00:00:00Z")
        fixture.replacement("2026-02-21T00:00:00Z", due="2026-03-01T00:00:00Z")
        self.assertEqual(fixture.label()["value"], 1)

    def test_blocked_or_deleted_source_cannot_confirm_replacement(self):
        for status in ("blocked", "deleted"):
            with self.subTest(status=status):
                fixture = EventFixture()
                fixture.service("2026-02-01T00:00:00Z", sid=11,
                                due="2026-03-01T00:00:00Z")
                fixture.event("status_changed", "2026-02-01T12:00:00Z", 10,
                              toStatus=status, reason="unknown")
                fixture.replacement()
                self.assertEqual(fixture.label()["value"], 1)

    def test_same_replacement_target_cannot_clear_two_old_services(self):
        fixture = EventFixture()
        fixture.service(START, sid=12)
        fixture.service("2026-02-01T00:00:00Z", sid=11,
                        due="2026-03-01T00:00:00Z")
        fixture.replacement()
        fixture.event("replacement_confirmed", "2026-02-02T01:00:00Z", 12,
                      replacementServiceId=11, dueAt=DUE)
        history = EventHistory(fixture.data)
        row = next(row for row in history.features_at(PREDICTION)
                   if row["serviceId"] == 12)
        self.assertEqual(history.label(row)["value"], 1)
        self.assertFalse(history.replay("2026-02-03T00:00:00Z")[12]["replaced"])

    def test_old_service_cannot_confirm_second_replacement(self):
        fixture = EventFixture()
        for sid in (11, 12):
            fixture.service("2026-02-01T00:00:00Z", sid=sid,
                            due="2026-03-01T00:00:00Z")
        fixture.replacement()
        second_link = fixture.replacement("2026-02-02T01:00:00Z", target=12)
        history = EventHistory(fixture.data)
        self.assertIsNone(history.replay("2026-02-03T00:00:00Z")[12]["replaces"])
        self.assertNotIn(second_link["id"], fixture.label()["evidenceEventIds"])

    def test_old_released_block_before_actual_prediction_does_not_exclude(self):
        fixture = EventFixture()
        fixture.restriction("2026-01-25T00:00:00Z")
        fixture.release("2026-01-27T00:00:00Z")
        # Audit uses due minus seven days, but this prediction occurs after release.
        self.assertEqual(fixture.label()["value"], 1)

    def test_release_exactly_at_prediction_is_no_longer_restricted(self):
        fixture = EventFixture()
        fixture.restriction("2026-01-25T00:00:00Z")
        fixture.release(PREDICTION)
        self.assertEqual(fixture.label()["value"], 1)

    def test_late_release_proof_delays_label_availability(self):
        fixture = EventFixture()
        fixture.restriction("2026-01-20T00:00:00Z", recorded="2026-02-10T00:00:00Z")
        release = fixture.release("2026-01-21T00:00:00Z")
        release["recordedAt"] = "2026-02-20T00:00:00Z"
        label = fixture.label()
        self.assertEqual(label["value"], 1)
        # At Feb 10 this row would have been excluded: only the late release
        # establishes that this old restriction had ended before prediction.
        self.assertEqual(label["availableAt"], "2026-02-20T00:00:00Z")
        self.assertIn(release["id"], label["evidenceEventIds"])

    def test_block_at_prediction_detected_when_recorded_later(self):
        fixture = EventFixture()
        fixture.restriction("2026-01-27T00:00:00Z", recorded="2026-02-20T00:00:00Z")
        label = fixture.label()
        self.assertIsNone(label["value"])
        self.assertEqual(label["reason"], "excluded_nonfinancial_block")
        self.assertEqual(label["availableAt"], "2026-02-20T00:00:00Z")

    def test_future_block_only_excludes_overlapping_window(self):
        for at, excluded in (("2026-02-03T00:00:00Z", True),
                             (DEADLINE, True), ("2026-02-06T00:00:01Z", False)):
            with self.subTest(at=at):
                fixture = EventFixture()
                fixture.restriction(at)
                self.assertEqual(fixture.label()["value"], None if excluded else 1)

    def test_unknown_or_financial_block_not_silently_excluded(self):
        for category in ("financial", "unknown", None):
            with self.subTest(category=category):
                fixture = EventFixture()
                fixture.restriction("2026-02-02T00:00:00Z", category=category)
                self.assertEqual(fixture.label()["value"], 1)


class EventDatasetTests(unittest.TestCase):
    def test_deleted_service_never_sampled_after_deletion(self):
        fixture = EventFixture()
        fixture.event("status_changed", "2026-01-28T00:00:00Z", 10,
                      toStatus="deleted")
        rows = fixture.dataset()
        self.assertTrue(rows)
        self.assertTrue(all(timestamp(row["predictionAt"]) < timestamp(PREDICTION)
                            for row in rows))
        self.assertEqual(EventHistory(fixture.data).features_at(PREDICTION), [])

    def test_manifest_legacy_source_warns_about_unconfirmed_deletion_and_single_cohort(self):
        fixture = EventFixture()
        info = manifest(fixture.data, fixture.dataset(), "fixture")
        self.assertIsNone(info["sourceMonthlyDeletionPolicy"])
        self.assertEqual(info["sourceRegistrationSummary"]["distinctDates"], 1)
        self.assertTrue(any("Monthly deletion policy absent" in text for text in info["limitations"]))
        self.assertTrue(any("share one registration date" in text for text in info["limitations"]))

    def test_manifest_current_source_carries_policy_and_observed_cohort_summary(self):
        fixture = EventFixture()
        fixture.data["events"][1]["occurredAt"] = "2026-01-05T00:00:00Z"
        fixture.data["events"][1]["recordedAt"] = "2026-01-05T00:00:00Z"
        policy = {"days": 5, "sourceUrl": "https://aeza.net/en/terms", "clause": "13.6.1"}
        fixture.data["assumptions"] = {
            "monthlyDeletionPolicy": policy,
            "monthlyDeletionBoundary": "simulation: same-timestamp manual payment precedes deletion",
            "monthlyRestrictionIntersection": "simulation: no deletion while nonfinancially blocked",
            "otherDeletionPolicies": "daily deletion not simulated",
        }
        info = manifest(fixture.data, fixture.dataset(), "fixture")
        self.assertEqual(info["sourceMonthlyDeletionPolicy"], policy)
        self.assertEqual(info["sourceRegistrationSummary"]["distinctDates"], 2)
        self.assertEqual(info["sourceRegistrationSummary"]["lastRegisteredAt"], "2026-01-05T00:00:00Z")
        self.assertFalse(any("Monthly deletion policy absent" in text for text in info["limitations"]))
        self.assertFalse(any("share one registration date" in text for text in info["limitations"]))
        for key in ("monthlyDeletionBoundary", "monthlyRestrictionIntersection", "otherDeletionPolicies"):
            self.assertIn(fixture.data["assumptions"][key], info["limitations"])
        self.assertIn("assumptions", info["featureExclusions"])

    def test_cycles_normalized_and_no_temporal_split_leakage(self):
        fixture = EventFixture()
        fixture.top_up(START, 5000)
        fixture.service(START, sid=20, due="2026-01-10T00:00:00Z")
        fixture.service(START, sid=21, due="2026-02-20T00:00:00Z")
        fixture.service(START, sid=22, due="2026-03-10T00:00:00Z")
        fixture.payment(DUE)
        fixture.payment("2026-03-01T00:00:00Z", due="2026-03-01T00:00:00Z",
                        through="2026-04-01T00:00:00Z")
        rows = fixture.dataset(validation="2026-02-04T00:00:00Z",
                               test="2026-03-04T00:00:00Z")
        self.assertTrue(rows)
        groups, cycle_splits = defaultdict(float), defaultdict(set)
        for row in rows:
            groups[row["split"], row["cycleId"]] += row["cycleWeight"]
            at = timestamp(row["predictionAt"])
            for key in ("maxFeatureOccurredAt", "maxFeatureRecordedAt"):
                self.assertLessEqual(timestamp(row["audit"][key]), at)
            if row["split"] in ("train", "validation", "test"):
                cycle_splits[row["cycleId"]].add(row["split"])
                self.assertIn(row["label"]["value"], (0, 1))
            if row["split"] == "train":
                self.assertLess(timestamp(row["label"]["availableAt"]),
                                timestamp("2026-02-04T00:00:00Z"))
            if row["split"] == "validation":
                self.assertLess(timestamp(row["label"]["availableAt"]),
                                timestamp("2026-03-04T00:00:00Z"))
        self.assertTrue(any(row["split"] == "purged" for row in rows))
        self.assertTrue({"train", "validation", "test"} <= {row["split"] for row in rows})
        self.assertTrue(all(len(splits) == 1 for splits in cycle_splits.values()))
        for total in groups.values():
            self.assertAlmostEqual(total, 1)

    def test_short_period_predictions_use_known_current_due(self):
        for term, lead in (("hour", timedelta(minutes=30)), ("day", timedelta(hours=12))):
            with self.subTest(term=term):
                fixture = EventFixture(term)
                first_due = fixture.data["events"][-1]["paidThrough"]
                rows = fixture.dataset()
                row = next(row for row in rows if row["dueAt"] == first_due)
                self.assertEqual(timestamp(row["dueAt"]) - timestamp(row["predictionAt"]), lead)
                self.assertEqual(row["features"]["has_renewal_history"], 0)
                self.assertEqual(row["cycleWeight"], 1)

    def test_short_future_payment_does_not_remove_prior_prediction_or_change_features(self):
        for term, period in (("hour", timedelta(hours=1)), ("day", timedelta(days=1))):
            with self.subTest(term=term):
                fixture = EventFixture(term)
                due = fixture.data["events"][-1]["paidThrough"]
                original = next(row for row in fixture.dataset() if row["dueAt"] == due)
                fixture.payment(due, due=due, through=iso(timestamp(due) + period), term=term)
                changed = next(row for row in fixture.dataset() if row["dueAt"] == due)
                self.assertEqual(original["features"], changed["features"])
                self.assertEqual(changed["label"]["value"], 0)

    def test_short_early_renewal_cancels_obsolete_prediction(self):
        fixture = EventFixture("hour")
        fixture.payment("2026-01-01T00:10:00Z", due="2026-01-01T01:00:00Z",
                        through="2026-01-01T02:00:00Z", term="hour")
        rows = fixture.dataset()
        self.assertNotIn("2026-01-01T01:00:00Z", {row["dueAt"] for row in rows})
        self.assertIn("2026-01-01T02:00:00Z", {row["dueAt"] for row in rows})

    def test_no_sampling_from_unrecorded_purchase(self):
        fixture = EventFixture("hour")
        fixture.data["events"][-1]["recordedAt"] = "2026-01-01T00:31:00Z"
        self.assertEqual(fixture.dataset(), [])

    def test_label_available_at_split_boundary_is_purged(self):
        fixture = EventFixture()
        rows = fixture.dataset(validation=DEADLINE)
        first = [row for row in rows if row["dueAt"] == DUE]
        self.assertTrue(first)
        self.assertTrue(all(row["split"] == "purged" for row in first))


if __name__ == "__main__":
    unittest.main()
