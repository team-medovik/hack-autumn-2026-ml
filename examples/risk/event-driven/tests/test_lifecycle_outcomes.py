"""Outcome boundaries and visibility; these tests do not measure ML accuracy."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from outcomes import cycle_outcomes


class LifecycleOutcomeTests(unittest.TestCase):
    def event(self, kind, at, sid=10, aid=1, **fields):
        return {"type": kind, "occurredAt": at, "recordedAt": at,
                "serviceId": sid, "accountId": aid, **fields}

    def purchase(self, at="2026-01-01T00:00:00Z", sid=10, aid=1,
                 due="2026-02-01T00:00:00Z"):
        return [self.event("service_created", at, sid, aid),
                self.event("payment_succeeded", at, sid, aid, mode="purchase",
                           amountMinor=100, paidThrough=due)]

    def payment(self, at, due="2026-02-01T00:00:00Z", through="2026-03-01T00:00:00Z"):
        return self.event("payment_succeeded", at, mode="manual", amountMinor=100,
                          dueAt=due, paidThrough=through)

    def restriction(self, at, **fields):
        return self.event("service_restricted", at, category="nonfinancial",
                          toStatus="blocked", reason="policy_review", **fields)

    def release(self, at):
        return self.event("service_released", at, category="nonfinancial", toStatus="active")

    def replacement(self, at="2026-02-02T00:00:00Z", **fields):
        return self.event("replacement_confirmed", at, replacementServiceId=11,
                          dueAt="2026-02-01T00:00:00Z", **fields)

    def outcome(self, events, as_of="2026-04-01T00:00:00Z", due="2026-02-01T00:00:00Z"):
        return next(r for r in cycle_outcomes({"events": events, "asOf": as_of})
                    if r["serviceId"] == 10 and r["dueAt"] == due)

    def test_120_hours_inclusive_but_121_hours_lost(self):
        for at, expected in (("2026-02-06T00:00:00Z", 0), ("2026-02-06T01:00:00Z", 1)):
            with self.subTest(at=at):
                result = self.outcome(self.purchase() + [self.payment(at)])
                self.assertEqual(result["label"], expected)
                self.assertEqual(result["firstPaymentAt"], at)

    def test_early_success_and_restriction_still_pending_without_full_followup(self):
        events = self.purchase() + [self.payment("2026-01-28T00:00:00Z"),
                                    self.restriction("2026-01-29T00:00:00Z")]
        self.assertEqual(self.outcome(events, "2026-02-05T23:59:59Z")["reason"], "pending")
        self.assertEqual(self.outcome(events, "2026-02-06T00:00:00Z")["reason"],
                         "excluded_nonfinancial_block")

    def test_no_followup_is_pending(self):
        self.assertIsNone(self.outcome(self.purchase(), "2026-01-02T00:00:00Z")["label"])

    def test_only_visible_events_can_create_cycles(self):
        events = self.purchase("2026-05-01T00:00:00Z", due="2026-06-01T00:00:00Z")
        self.assertEqual(cycle_outcomes({"events": events, "asOf": "2026-04-01T00:00:00Z"}), [])

    def test_late_recorded_payment_invisible_until_known(self):
        event = self.payment("2026-02-02T00:00:00Z")
        event["recordedAt"] = "2026-02-20T00:00:00Z"
        events = self.purchase() + [event]
        self.assertEqual(self.outcome(events, "2026-02-10T00:00:00Z")["label"], 1)
        self.assertEqual(self.outcome(events, "2026-02-20T00:00:00Z")["label"], 0)

    def test_plain_purchase_not_replacement(self):
        events = self.purchase() + self.purchase("2026-02-02T00:00:00Z", sid=11,
                                                 due="2026-03-02T00:00:00Z")
        self.assertEqual(self.outcome(events)["label"], 1)
        self.assertEqual(self.outcome(events + [self.replacement()])["reason"], "confirmed_replacement")

    def test_replacement_must_exist_be_paid_and_have_same_owner(self):
        for replacement_events in ([],
                                   [self.event("service_created", "2026-02-01T00:00:00Z", sid=11)],
                                   self.purchase("2026-02-01T00:00:00Z", sid=11, aid=2,
                                                 due="2026-03-01T00:00:00Z")):
            with self.subTest(events=replacement_events):
                self.assertEqual(self.outcome(self.purchase() + replacement_events +
                                              [self.replacement()])["label"], 1)

    def test_replacement_payment_must_precede_confirmation(self):
        events = self.purchase() + self.purchase("2026-02-03T00:00:00Z", sid=11,
                                                 due="2026-03-03T00:00:00Z")
        self.assertEqual(self.outcome(events + [self.replacement()])["label"], 1)

    def test_unrecorded_confirmation_or_purchase_is_not_visible(self):
        events = self.purchase() + self.purchase("2026-02-01T00:00:00Z", sid=11,
                                                 due="2026-03-01T00:00:00Z")
        link = self.replacement()
        link["recordedAt"] = "2026-02-20T00:00:00Z"
        self.assertEqual(self.outcome(events + [link], "2026-02-10T00:00:00Z")["label"], 1)
        self.assertEqual(self.outcome(events + [link], "2026-02-20T00:00:00Z")["label"], 0)
        events[-1]["recordedAt"] = "2026-02-20T00:00:00Z"
        self.assertEqual(self.outcome(events + [self.replacement()], "2026-02-10T00:00:00Z")["label"], 1)

    def test_expired_replacement_needs_successful_renewal(self):
        events = self.purchase() + self.purchase("2026-01-02T00:00:00Z", sid=11,
                                                 due="2026-02-02T00:00:00Z")
        self.assertEqual(self.outcome(events + [self.replacement()])["label"], 1)
        events.append(self.event("payment_succeeded", "2026-02-02T00:00:00Z", sid=11,
                                 mode="auto", amountMinor=100, dueAt="2026-02-02T00:00:00Z",
                                 paidThrough="2026-03-02T00:00:00Z"))
        self.assertEqual(self.outcome(events + [self.replacement()])["label"], 0)

    def test_restricted_or_deleted_replacement_is_not_valid(self):
        base = self.purchase() + self.purchase("2026-02-01T00:00:00Z", sid=11,
                                               due="2026-03-01T00:00:00Z")
        for kind, status in (("service_restricted", "blocked"), ("status_changed", "deleted")):
            event = self.event(kind, "2026-02-01T12:00:00Z", sid=11, toStatus=status,
                               category="nonfinancial")
            self.assertEqual(self.outcome(base + [event, self.replacement()])["label"], 1)

    def test_same_age_replacement_or_confirmation_before_creation_is_invalid(self):
        base = self.purchase() + self.purchase(sid=11, due="2026-03-01T00:00:00Z")
        self.assertEqual(self.outcome(base + [self.replacement()])["label"], 1)
        # Same-timestamp file ordering is part of the event contract: a purchase
        # appearing after confirmation must not retroactively validate the link.
        events = self.purchase() + [self.replacement()]
        events += self.purchase("2026-02-02T00:00:00Z", sid=11, due="2026-03-02T00:00:00Z")
        self.assertEqual(self.outcome(events)["label"], 1)

    def test_replacement_after_deadline_does_not_rewrite_loss(self):
        events = self.purchase() + self.purchase("2026-02-01T00:00:00Z", sid=11,
                                                 due="2026-03-01T00:00:00Z")
        self.assertEqual(self.outcome(events + [self.replacement("2026-02-06T00:00:00Z")])["label"], 0)
        self.assertEqual(self.outcome(events + [self.replacement("2026-02-06T00:00:01Z")])["label"], 1)

    def test_replacement_is_scoped_to_current_cycle_not_prior_loss(self):
        events = self.purchase() + [self.payment("2026-02-08T00:00:00Z")]
        events += self.purchase("2026-02-20T00:00:00Z", sid=11, due="2026-03-20T00:00:00Z")
        link = self.replacement("2026-02-21T00:00:00Z")
        link["dueAt"] = "2026-03-01T00:00:00Z"
        events.append(link)
        self.assertEqual(self.outcome(events)["label"], 1)
        self.assertEqual(self.outcome(events, due="2026-03-01T00:00:00Z")["label"], 0)

    def test_prior_active_block_overlaps_but_lifted_old_block_does_not(self):
        events = self.purchase() + [self.restriction("2026-01-10T00:00:00Z")]
        self.assertEqual(self.outcome(events)["reason"], "excluded_nonfinancial_block")
        # Window starts Jan 25. Release exactly at start means no overlap.
        events.append(self.release("2026-01-25T00:00:00Z"))
        self.assertEqual(self.outcome(events)["label"], 1)
        events[-1] = self.release("2026-01-25T00:00:01Z")
        self.assertIsNone(self.outcome(events)["label"])

    def test_block_excludes_only_affected_cycles(self):
        events = self.purchase() + [self.payment("2026-01-28T00:00:00Z"),
                                    self.restriction("2026-02-02T00:00:00Z"),
                                    self.release("2026-02-03T00:00:00Z")]
        self.assertIsNone(self.outcome(events)["label"])
        self.assertEqual(self.outcome(events, due="2026-03-01T00:00:00Z")["label"], 1)

    def test_block_after_deadline_or_unknown_reason_not_exclusion(self):
        late = self.restriction("2026-02-06T00:00:01Z")
        self.assertEqual(self.outcome(self.purchase() + [late])["label"], 1)
        for category in ("financial", "unknown", None):
            block = self.restriction("2026-02-02T00:00:00Z")
            block["category"] = category
            self.assertEqual(self.outcome(self.purchase() + [block])["label"], 1)

    def test_late_recorded_block_and_release_visibility(self):
        block = self.restriction("2026-01-20T00:00:00Z")
        block["recordedAt"] = "2026-02-11T00:00:00Z"
        events = self.purchase() + [block]
        self.assertEqual(self.outcome(events, "2026-02-10T00:00:00Z")["label"], 1)
        self.assertIsNone(self.outcome(events, "2026-02-11T00:00:00Z")["label"])
        release = self.release("2026-01-21T00:00:00Z")
        release["recordedAt"] = "2026-02-12T00:00:00Z"
        events.append(release)
        self.assertIsNone(self.outcome(events, "2026-02-11T00:00:00Z")["label"])
        self.assertEqual(self.outcome(events, "2026-02-12T00:00:00Z")["label"], 1)

    def test_window_start_cannot_precede_purchase_and_duplicate_cycles_collapse(self):
        events = self.purchase("2026-01-31T00:00:00Z")
        rows = cycle_outcomes({"events": events + events, "asOf": "2026-03-01T00:00:00Z"})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["windowStart"], "2026-01-31T00:00:00Z")


if __name__ == "__main__":
    unittest.main()
