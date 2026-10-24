"""Monthly nonpayment retention; restriction interaction is a simulator convention.

Five days from financial suspension comes from Aeza's offer, clause 13.6.1.
Exact-timestamp action ordering and nonfinancial restriction interaction are our
explicit simulation choices, not independently verified Aeza behavior.
"""
import importlib.util
import unittest
from datetime import timedelta
from pathlib import Path


SPEC = importlib.util.spec_from_file_location("retention_billing", Path(__file__).parents[1] / "billing.py")
billing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(billing)
START = billing.instant("2026-01-01T00:00:00Z")
DUE = billing.instant("2026-02-01T00:00:00Z")
DEADLINE = DUE + timedelta(days=5)
SECOND = timedelta(seconds=1)


class MonthlyRetentionTests(unittest.TestCase):
    def make(self, auto=True, funds=100):
        sim = billing.BillingSimulator(START)
        sim.add_account(1)
        sim.schedule("fund", START, "top_up", account_id=1, amount=funds)
        sim.schedule("purchase", START, "purchase", account_id=1, service_id=1,
                     price=100, term="month", tariff_code="test-month", auto_prolong=auto)
        return sim

    def topup(self, sim, at, amount=100, command="refill"):
        sim.schedule(command, at, "top_up", account_id=1, amount=amount)

    def renew(self, sim, at, due=DUE, command="renew"):
        sim.schedule(command, at, "manual_renew", account_id=1, service_id=1,
                     expected_due=billing.iso(due))

    def restrict(self, sim, at):
        sim.schedule("restrict", at, "restrict_service", account_id=1, service_id=1,
                     reason="policy_review")

    def release(self, sim, at):
        sim.schedule("release", at, "release_service", account_id=1, service_id=1)

    def deleted_events(self, sim):
        return [e for e in sim.events if e["type"] == "status_changed" and e["toStatus"] == "deleted"]

    def test_unpaid_month_is_retained_until_exactly_five_days(self):
        for auto in (True, False):
            with self.subTest(auto=auto):
                sim = self.make(auto=auto)
                sim.run(DEADLINE - SECOND)
                self.assertEqual(sim.services[1]["status"], "prolong_wait")
                self.assertFalse(self.deleted_events(sim))
                sim.run(DEADLINE)
                self.assertEqual(sim.services[1]["status"], "deleted")
                self.assertEqual([e["occurredAt"] for e in self.deleted_events(sim)], [billing.iso(DEADLINE)])
                self.assertEqual(sim.accounts[1]["balance"], 0)

    def test_sufficient_wallet_does_not_save_auto_disabled_unpaid_service(self):
        sim = self.make(auto=False, funds=1000)
        sim.run(DEADLINE)
        self.assertEqual(sim.services[1]["status"], "deleted")
        self.assertEqual(sim.accounts[1]["balance"], 900)
        self.assertEqual(len([e for e in sim.events if e["type"] == "payment_succeeded"]), 1)

    def test_topup_alone_does_not_retry_or_cancel_deletion(self):
        sim = self.make()
        self.topup(sim, DUE + timedelta(days=1), amount=500)
        sim.run(DEADLINE)
        self.assertEqual(sim.services[1]["status"], "deleted")
        self.assertEqual(sim.accounts[1]["balance"], 500)
        self.assertEqual(len([e for e in sim.events if e["type"] == "payment_failed"]), 1)
        self.assertEqual(len([e for e in sim.events if e["type"] == "payment_succeeded"]), 1)

    def test_auto_toggle_does_not_reset_financial_deadline_or_retry(self):
        sim = self.make()
        for n, enabled in ((1, False), (2, True), (3, False)):
            sim.schedule(f"toggle-{n}", DUE + timedelta(days=n), "set_auto_prolong",
                         account_id=1, service_id=1, enabled=enabled)
        self.topup(sim, DUE + timedelta(days=4))
        sim.run(DEADLINE)
        self.assertEqual(sim.services[1]["status"], "deleted")
        self.assertEqual(sim.accounts[1]["balance"], 100)

    def test_successful_manual_payment_before_or_at_deadline_cancels_old_timer(self):
        # Inclusive boundary is an explicit simulator ordering convention.
        for at in (DEADLINE - SECOND, DEADLINE):
            with self.subTest(at=at):
                sim = self.make()
                self.topup(sim, at)
                self.renew(sim, at)
                sim.run(DEADLINE + SECOND)
                self.assertEqual(sim.services[1]["status"], "active")
                self.assertEqual(sim.services[1]["expiresAt"], billing.iso(billing.period_end(at, "month")))
                self.assertFalse(self.deleted_events(sim))
                self.assertEqual(sim.accounts[1]["balance"], 0)

    def test_failed_manual_payment_at_deadline_does_not_extend_retention(self):
        sim = self.make()
        self.topup(sim, DEADLINE, amount=99)
        self.renew(sim, DEADLINE)
        sim.run(DEADLINE)
        self.assertEqual(sim.services[1]["status"], "deleted")
        self.assertEqual(sim.accounts[1]["balance"], 99)
        self.assertEqual(len([e for e in sim.events if e["type"] == "payment_failed"]), 2)

    def test_after_deadline_payment_cannot_resurrect_service(self):
        sim = self.make()
        self.topup(sim, DEADLINE + SECOND)
        self.renew(sim, DEADLINE + SECOND)
        sim.run(DEADLINE + SECOND)
        self.assertEqual(sim.services[1]["status"], "deleted")
        self.assertEqual(sim.accounts[1]["balance"], 100)
        self.assertEqual(sim.events[-1]["type"], "action_rejected")
        self.assertEqual(sim.events[-1]["reason"], "service_unavailable")
        self.assertEqual(len(self.deleted_events(sim)), 1)

    def test_renewal_gets_a_new_deadline_only_after_its_new_paid_period(self):
        sim = self.make()
        at = DUE + timedelta(days=2)
        self.topup(sim, at)
        self.renew(sim, at)
        next_due = billing.period_end(at, "month")
        sim.run(next_due + timedelta(days=5) - SECOND)
        self.assertEqual(sim.services[1]["status"], "prolong_wait")
        self.assertFalse(self.deleted_events(sim))
        sim.run(next_due + timedelta(days=5))
        self.assertEqual(sim.services[1]["status"], "deleted")
        self.assertEqual(len(self.deleted_events(sim)), 1)

    def test_early_manual_renewal_protects_all_paid_time(self):
        sim = self.make(auto=False, funds=200)
        self.renew(sim, DUE - timedelta(days=10))
        sim.run(DEADLINE)
        self.assertEqual(sim.services[1]["expiresAt"], "2026-03-01T00:00:00Z")
        self.assertEqual(sim.services[1]["status"], "active")
        self.assertFalse(self.deleted_events(sim))

    def test_successful_auto_renewal_does_not_leave_an_active_deletion_timer(self):
        sim = self.make(funds=200)
        sim.run(DEADLINE)
        self.assertEqual(sim.services[1]["status"], "active")
        self.assertEqual(sim.services[1]["expiresAt"], "2026-03-01T00:00:00Z")
        self.assertFalse(self.deleted_events(sim))

    def test_deleted_tombstone_and_historical_states_replay_without_money_changes(self):
        sim = self.make(auto=False, funds=250)
        sim.run(DEADLINE + timedelta(days=50))
        self.assertEqual(sim.snapshot(DUE - SECOND)["services"][0]["status"], "active")
        self.assertEqual(sim.snapshot(DEADLINE - SECOND)["services"][0]["status"], "prolong_wait")
        current = sim.snapshot()
        self.assertEqual(current["services"][0]["status"], "deleted")
        self.assertEqual(current["accounts"][0]["balance"], 150)
        self.assertEqual(current["accounts"][0]["totalReplenished"], 250)
        self.assertEqual(len(self.deleted_events(sim)), 1)

    def test_existing_financial_deadline_is_preserved_across_short_restriction(self):
        # Restriction interaction is an agreed simulation choice, not offer text.
        sim = self.make()
        self.restrict(sim, DUE + timedelta(days=1))
        self.release(sim, DUE + timedelta(days=3))
        sim.run(DEADLINE - SECOND)
        self.assertEqual(sim.services[1]["status"], "prolong_wait")
        sim.run(DEADLINE)
        self.assertEqual(sim.services[1]["status"], "deleted")

    def test_no_deletion_while_restricted_but_elapsed_deadline_applies_on_release(self):
        sim = self.make()
        self.restrict(sim, DUE + timedelta(days=1))
        release_at = DUE + timedelta(days=10)
        self.release(sim, release_at)
        sim.run(release_at - SECOND)
        self.assertEqual(sim.services[1]["status"], "blocked")
        self.assertFalse(self.deleted_events(sim))
        sim.run(release_at)
        self.assertEqual(sim.services[1]["status"], "deleted")
        self.assertEqual(self.deleted_events(sim)[0]["occurredAt"], billing.iso(release_at))

    def test_manual_payment_on_late_release_cannot_evade_elapsed_deadline(self):
        sim = self.make()
        self.restrict(sim, DUE + timedelta(days=1))
        release_at = DUE + timedelta(days=10)
        self.release(sim, release_at)
        self.topup(sim, release_at)
        self.renew(sim, release_at)
        sim.run(release_at)
        self.assertEqual(sim.services[1]["status"], "deleted")
        self.assertEqual(sim.accounts[1]["balance"], 100)
        self.assertEqual(len([e for e in sim.events if e["type"] == "payment_succeeded"]), 1)

    def test_first_financial_suspension_starts_on_release_if_expiry_was_blocked(self):
        sim = self.make()
        self.restrict(sim, DUE - SECOND)
        release_at = DUE + timedelta(days=10)
        self.release(sim, release_at)
        sim.run(release_at + timedelta(days=5) - SECOND)
        self.assertEqual(sim.services[1]["status"], "prolong_wait")
        self.assertFalse(self.deleted_events(sim))
        sim.run(release_at + timedelta(days=5))
        self.assertEqual(sim.services[1]["status"], "deleted")

    def test_manual_payment_after_release_can_recover_previously_blocked_expiry(self):
        sim = self.make()
        self.restrict(sim, DUE - SECOND)
        release_at = DUE + timedelta(days=10)
        self.release(sim, release_at)
        self.topup(sim, release_at)
        self.renew(sim, release_at)
        sim.run(release_at + timedelta(days=5))
        self.assertEqual(sim.services[1]["status"], "active")
        self.assertEqual(sim.services[1]["expiresAt"], billing.iso(billing.period_end(release_at, "month")))
        self.assertFalse(self.deleted_events(sim))


if __name__ == "__main__":
    unittest.main()
