"""Small explicit histories for per-cycle hourly/daily sampling; no model evaluation."""
from datetime import timedelta
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_dataset import GRACE, History, iso, timestamp
from prepare_short_periods import build_short_dataset


class ShortHistory:
    def __init__(self, term="hour", as_of="2026-04-10T15:00:00Z", price=100):
        self.start = timestamp("2026-03-01T10:10:00Z")
        self.term = term
        self.period = {"hour": timedelta(hours=1), "day": timedelta(days=1),
                       "month": timedelta(days=30)}[term]
        self.first_due = self.start + self.period
        self.data = {"asOf": as_of,
                     "accountBusiness": [{"accountId": 1, "createdAt": "2026-01-01T00:00:00Z"}],
                     "events": []}
        self.event("service_created", self.start)
        self.event("contract_changed", self.start, paymentTerm=term, priceMinor=price,
                   currency="EUR", tariffCode="SYNTHETIC-SHORT", contractVersion=1)
        self.payment(self.start, self.start)
        self.event("status_changed", self.start, toStatus="active")

    def event(self, kind, at, recorded=None, **details):
        event = {"id": f"event-{len(self.data['events']):04d}", "type": kind,
                 "serviceId": 1, "accountId": 1, "occurredAt": iso(at),
                 "recordedAt": iso(recorded or at), **details}
        self.data["events"].append(event)
        return event

    def payment(self, at, due):
        return self.event("payment_succeeded", at, dueAt=iso(due), paidThrough=iso(at + self.period),
                          paymentTerm=self.term)

    def rows(self):
        return build_short_dataset(self.data, timestamp("2026-03-15T00:00:00Z"),
                                   timestamp("2026-04-01T00:00:00Z"))

    def first_row(self):
        return next(row for row in self.rows() if row["dueAt"] == iso(self.first_due))


class ShortPeriodSamplingTests(unittest.TestCase):
    def test_hourly_expiry_between_daily_snapshots_is_not_missed(self):
        fixture = ShortHistory()
        daily_snapshot = History(fixture.data).features_at(timestamp("2026-03-01T15:00:00Z"))
        self.assertEqual(daily_snapshot, [])
        rows = fixture.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["predictionAt"], iso(fixture.first_due - timedelta(minutes=30)))
        self.assertEqual(rows[0]["dueAt"], iso(fixture.first_due))

    def test_daily_sample_is_exactly_twelve_hours_before_due_and_monthly_price_is_thirty_days(self):
        fixture = ShortHistory(term="day", price=123)
        row = fixture.first_row()
        self.assertEqual(row["predictionAt"], iso(fixture.first_due - timedelta(hours=12)))
        self.assertEqual(row["features"]["payment_term"], "day")
        self.assertEqual(row["features"]["monthly_revenue_minor"], 3690)

    def test_initial_purchase_is_not_a_renewal_or_its_own_training_target(self):
        for term in ("hour", "day"):
            with self.subTest(term=term):
                fixture = ShortHistory(term)
                rows = fixture.rows()
                self.assertEqual(len(rows), 1)
                row = rows[0]
                self.assertNotEqual(row["dueAt"], iso(fixture.start))
                self.assertEqual(row["dueAt"], iso(fixture.first_due))
                self.assertEqual(row["features"]["has_renewal_history"], 0)
                self.assertEqual(row["features"]["renewal_cycles_90d"], 0)
                self.assertIsNone(row["features"]["failed_cycle_share_90d"])
                self.assertIsNone(row["features"]["days_since_successful_renewal"])

    def test_same_five_day_grace_includes_exact_deadline_for_hour_and_day(self):
        for term in ("hour", "day"):
            for seconds_after, expected in ((0, 0), (1, 1)):
                with self.subTest(term=term, seconds_after=seconds_after):
                    fixture = ShortHistory(term)
                    deadline = fixture.first_due + GRACE
                    fixture.payment(deadline + timedelta(seconds=seconds_after), fixture.first_due)
                    row = fixture.first_row()
                    self.assertEqual(row["labelDeadline"], iso(deadline))
                    self.assertEqual(row["label"]["value"], expected)

    def test_observation_before_grace_end_is_pending_even_after_payment(self):
        for term in ("hour", "day"):
            with self.subTest(term=term):
                fixture = ShortHistory(term)
                fixture.payment(fixture.first_due, fixture.first_due)
                fixture.data["asOf"] = iso(fixture.first_due + GRACE - timedelta(seconds=1))
                rows = build_short_dataset(fixture.data, timestamp("2026-03-02T00:00:00Z"),
                                           timestamp("2026-03-03T00:00:00Z"))
                row = next(row for row in rows if row["dueAt"] == iso(fixture.first_due))
                self.assertIsNone(row["label"]["value"])
                self.assertEqual(row["label"]["reason"], "observation_incomplete")
                self.assertEqual(row["split"], "pending")

    def test_policy_block_excludes_short_period_row_from_training(self):
        for term in ("hour", "day"):
            with self.subTest(term=term):
                fixture = ShortHistory(term)
                block = fixture.event("status_changed", fixture.first_due + timedelta(hours=1),
                                      toStatus="blocked", reason="synthetic_policy_review_not_payment_failure")
                row = fixture.first_row()
                self.assertIsNone(row["label"]["value"])
                self.assertEqual(row["label"]["reason"], "excluded_nonfinancial_block")
                self.assertEqual(row["split"], "excluded")
                self.assertIn(block["id"], row["label"]["evidenceEventIds"])

    def test_every_observed_hourly_cycle_is_sampled_once_even_on_same_day(self):
        fixture = ShortHistory()
        for index in range(1, 6):
            due = fixture.start + index * fixture.period
            fixture.payment(due, due)
        rows = fixture.rows()
        self.assertEqual(len(rows), 6)
        self.assertEqual(len({row["cycleId"] for row in rows}), 6)
        self.assertEqual(len({row["predictionAt"][:10] for row in rows}), 1)
        self.assertTrue(all(row["cycleWeight"] == 1 for row in rows))
        for row in rows:
            prediction = timestamp(row["predictionAt"])
            self.assertEqual(timestamp(row["dueAt"]) - prediction, timedelta(minutes=30))
            self.assertGreaterEqual(prediction, fixture.start)
            for key in ("maxFeatureOccurredAt", "maxFeatureRecordedAt"):
                self.assertLessEqual(timestamp(row["audit"][key]), prediction)

    def test_unobserved_creation_or_payment_is_not_used_to_schedule_a_prediction(self):
        for event_type in ("service_created", "payment_succeeded"):
            with self.subTest(event_type=event_type):
                fixture = ShortHistory()
                hidden = next(event for event in fixture.data["events"] if event["type"] == event_type)
                hidden["recordedAt"] = iso(fixture.first_due - timedelta(minutes=29))
                self.assertEqual(fixture.rows(), [])

    def test_future_events_do_not_change_past_features_but_can_change_label(self):
        fixture = ShortHistory()
        original = fixture.first_row()
        self.assertEqual(original["label"]["value"], 1)
        payment = fixture.payment(fixture.first_due, fixture.first_due)
        modified = fixture.first_row()
        self.assertEqual(original["features"], modified["features"])
        self.assertEqual(original["audit"], modified["audit"])
        self.assertEqual(modified["label"]["value"], 0)
        late = fixture.first_due + GRACE + timedelta(seconds=1)
        payment.update(occurredAt=iso(late), recordedAt=iso(late), paidThrough=iso(late + fixture.period))
        changed_again = fixture.first_row()
        self.assertEqual(original["features"], changed_again["features"])
        self.assertEqual(original["audit"], changed_again["audit"])
        self.assertEqual(changed_again["label"]["value"], 1)

    def test_changed_known_due_does_not_emit_obsolete_planned_cycle(self):
        fixture = ShortHistory()
        # Early renewal becomes known before the original scheduled prediction.
        paid_at = fixture.start + timedelta(minutes=10)
        fixture.payment(paid_at, fixture.first_due)
        rows = fixture.rows()
        self.assertNotIn(iso(fixture.first_due), {row["dueAt"] for row in rows})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["dueAt"], iso(paid_at + fixture.period))

    def test_non_short_payment_terms_are_not_sampled(self):
        self.assertEqual(ShortHistory(term="month").rows(), [])

    def test_outcome_available_exactly_at_split_boundary_is_purged(self):
        fixture = ShortHistory()
        boundary = fixture.first_due + GRACE
        rows = build_short_dataset(fixture.data, boundary, timestamp("2026-04-01T00:00:00Z"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["split"], "purged")


if __name__ == "__main__":
    unittest.main()
