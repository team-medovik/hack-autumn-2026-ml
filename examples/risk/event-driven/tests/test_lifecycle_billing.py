import importlib.util
import unittest
from datetime import timedelta
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("lifecycle_billing", Path(__file__).parents[1] / "billing.py")
billing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(billing)
START = billing.instant("2026-01-01T00:00:00Z")
DUE = billing.instant("2026-02-01T00:00:00Z")


class LifecycleBillingTests(unittest.TestCase):
    def make(self, term="month", auto=True):
        sim = billing.BillingSimulator(START)
        for aid in (1, 2):
            sim.add_account(aid)
            sim.schedule(f"fund-{aid}", START, "top_up", account_id=aid, amount=10000)
        sim.schedule("old", START, "purchase", account_id=1, service_id=1,
                     price=100, term=term, tariff_code="test", auto_prolong=auto)
        return sim

    def buy_new(self, sim, sid=2, aid=1, at=None):
        sim.schedule(f"buy-{sid}", at or START + timedelta(days=10), "purchase",
                     account_id=aid, service_id=sid, price=150, term="month", tariff_code="new")

    def restrict(self, sim, at=None):
        sim.schedule("restrict", at or DUE - timedelta(days=1), "restrict_service",
                     account_id=1, service_id=1, reason="policy_review")

    def confirm(self, sim, at=None, sid=1, replacement=2, command="confirm"):
        sim.schedule(command, at or START + timedelta(days=11), "confirm_replacement",
                     account_id=1, service_id=sid, replacement_service_id=replacement)

    def test_replacement_requires_paid_new_purchase_and_changes_no_money_or_auto(self):
        sim = self.make()
        self.buy_new(sim)
        self.confirm(sim)
        sim.run(START + timedelta(days=11))
        self.assertEqual(sim.accounts[1]["balance"], 9750)
        self.assertTrue(sim.services[1]["autoProlong"])
        self.assertEqual(sim.services[1]["expiresAt"], billing.iso(DUE))
        self.assertEqual(sim.services[1]["replacementServiceId"], 2)
        self.assertEqual(sim.services[2]["replacesServiceId"], 1)
        self.assertEqual(sim.events[-1]["type"], "replacement_confirmed")
        self.assertEqual(sim.events[-1]["dueAt"], billing.iso(DUE))
        sim.schedule("off", START + timedelta(days=12), "set_auto_prolong",
                     account_id=1, service_id=1, enabled=False)
        sim.run(DUE)
        self.assertEqual(sim.services[1]["status"], "prolong_wait")
        self.assertEqual(sim.accounts[1]["balance"], 9750)

    def test_replacement_rejects_missing_self_foreign_unpaid_or_not_newer(self):
        for case in ("missing", "self", "foreign", "unpaid", "same_creation", "older", "blocked"):
            with self.subTest(case=case):
                sim = self.make()
                if case not in ("missing", "self"):
                    self.buy_new(sim, aid=2 if case == "foreign" else 1,
                                 at=START if case == "same_creation" else None)
                at = START + timedelta(days=11)
                if case == "unpaid":
                    at = START + timedelta(days=45)
                if case == "blocked":
                    sim.schedule("block-new", START + timedelta(days=10, hours=1), "restrict_service",
                                 account_id=1, service_id=2, reason="security_review")
                self.confirm(sim, at=at, sid=2 if case == "older" else 1,
                             replacement=1 if case in ("self", "older") else 2)
                sim.run(at)
                self.assertFalse(any(e["type"] == "replacement_confirmed" for e in sim.events))
                self.assertEqual(sim.events[-1]["reason"], "invalid_replacement")

    def test_replacement_duplicate_and_reuse_rejected_but_forward_chain_allowed(self):
        sim = self.make()
        self.buy_new(sim)
        self.buy_new(sim, sid=3, at=START + timedelta(days=12))
        self.confirm(sim)
        self.confirm(sim, command="duplicate", at=START + timedelta(days=11, minutes=1))
        self.confirm(sim, sid=1, replacement=3, command="two-targets", at=START + timedelta(days=13))
        self.confirm(sim, sid=3, replacement=2, command="reuse", at=START + timedelta(days=13, minutes=1))
        self.confirm(sim, sid=2, replacement=3, command="chain", at=START + timedelta(days=14))
        self.confirm(sim, sid=3, replacement=1, command="cycle", at=START + timedelta(days=15))
        sim.run(START + timedelta(days=15))
        links = [(e["serviceId"], e["replacementServiceId"]) for e in sim.events
                 if e["type"] == "replacement_confirmed"]
        self.assertEqual(links, [(1, 2), (2, 3)])

    def test_restriction_prevents_confirming_old_service_replacement(self):
        sim = self.make()
        self.buy_new(sim)
        self.restrict(sim, START + timedelta(days=10, hours=1))
        self.confirm(sim)
        sim.run(START + timedelta(days=11))
        self.assertEqual(sim.events[-1]["reason"], "invalid_replacement")
        self.assertNotIn("replacementServiceId", sim.services[1])

    def test_incoming_replacement_cannot_be_reused_for_another_old_service(self):
        sim = self.make()
        self.buy_new(sim, sid=3, at=START + timedelta(days=1))
        self.buy_new(sim)
        self.confirm(sim)
        self.confirm(sim, sid=3, replacement=2, command="second-old",
                     at=START + timedelta(days=12))
        sim.run(START + timedelta(days=12))
        self.assertEqual(sim.events[-1]["reason"], "invalid_replacement")
        self.assertNotIn("replacementServiceId", sim.services[3])

    def test_restriction_suppresses_auto_manual_and_topup_does_not_release(self):
        sim = self.make()
        self.restrict(sim)
        sim.schedule("manual", DUE - timedelta(hours=1), "manual_renew",
                     account_id=1, service_id=1, expected_due=billing.iso(DUE))
        sim.schedule("topup-blocked", DUE, "top_up", account_id=1, amount=300)
        sim.run(DUE + timedelta(days=20))
        self.assertEqual(sim.accounts[1]["balance"], 10200)
        self.assertEqual(sim.services[1]["status"], "blocked")
        self.assertEqual(sim.services[1]["expiresAt"], billing.iso(DUE))
        self.assertEqual(len([e for e in sim.events if e["type"] == "renewal_paused"]), 1)
        self.assertFalse(any(e["type"] == "payment_failed" for e in sim.events))
        self.assertEqual(len([e for e in sim.events if e["type"] == "payment_succeeded"]), 1)

    def test_release_before_due_preserves_single_auto_payment(self):
        sim = self.make()
        self.restrict(sim, START + timedelta(days=2))
        sim.schedule("release", START + timedelta(days=5), "release_service", account_id=1, service_id=1)
        sim.run(DUE)
        self.assertEqual(sim.accounts[1]["balance"], 9800)
        self.assertEqual(len([e for e in sim.events if e["type"] == "renewal_due"]), 1)
        self.assertEqual(sim.services[1]["expiresAt"], "2026-03-01T00:00:00Z")

    def test_release_overdue_has_no_charge_debt_or_extension_until_manual(self):
        sim = self.make()
        self.restrict(sim)
        release = DUE + timedelta(days=40)
        sim.schedule("release", release, "release_service", account_id=1, service_id=1)
        sim.run(release)
        self.assertEqual(sim.services[1]["status"], "prolong_wait")
        self.assertEqual(sim.services[1]["expiresAt"], billing.iso(DUE))
        self.assertEqual(sim.accounts[1]["balance"], 9900)
        sim.schedule("pay", release + timedelta(seconds=1), "manual_renew", account_id=1,
                     service_id=1, expected_due=billing.iso(DUE))
        sim.run(release + timedelta(seconds=1))
        self.assertEqual(sim.accounts[1]["balance"], 9800)
        self.assertEqual(sim.services[1]["expiresAt"], "2026-04-13T00:00:01Z")

    def test_hourly_deletion_paused_and_ordinary_rule_resumes_on_release(self):
        sim = self.make(term="hour", auto=False)
        self.restrict(sim, START + timedelta(minutes=30))
        release = START + timedelta(days=3)
        sim.schedule("release", release, "release_service", account_id=1, service_id=1)
        sim.run(release - timedelta(seconds=1))
        self.assertEqual(sim.services[1]["status"], "blocked")
        sim.run(release)
        self.assertEqual(sim.services[1]["status"], "deleted")
        self.assertEqual(sim.accounts[1]["balance"], 9900)

    def test_repeated_restriction_release_and_financial_reason_validation(self):
        sim = self.make()
        for i, reason in enumerate(("", "insufficient_funds", "nonpayment", [], None)):
            with self.assertRaises(ValueError):
                sim.schedule(f"bad-{i}", START + timedelta(days=1), "restrict_service",
                             account_id=1, service_id=1, reason=reason)
        self.restrict(sim, START + timedelta(days=2))
        sim.schedule("restrict-twice", START + timedelta(days=3), "restrict_service",
                     account_id=1, service_id=1, reason="abuse_review")
        for n in (4, 5):
            sim.schedule(f"release-{n}", START + timedelta(days=n), "release_service",
                         account_id=1, service_id=1)
        sim.run(START + timedelta(days=5))
        self.assertEqual([e["reason"] for e in sim.events if e["type"] == "action_rejected"],
                         ["already_restricted", "not_restricted"])

    def test_lifecycle_commands_are_idempotent_and_enforce_owner(self):
        sim = self.make()
        at = START + timedelta(days=1)
        command = dict(account_id=1, service_id=1, reason="policy_review")
        self.assertTrue(sim.schedule("restrict", at, "restrict_service", **command))
        self.assertFalse(sim.schedule("restrict", at, "restrict_service", **command))
        sim.schedule("foreign-release", at + timedelta(hours=1), "release_service",
                     account_id=2, service_id=1)
        sim.run(at + timedelta(hours=1))
        self.assertEqual(sim.services[1]["status"], "blocked")
        self.assertEqual(len([e for e in sim.events if e["type"] == "service_restricted"]), 1)
        self.assertEqual(sim.events[-1]["reason"], "service_unavailable")

    def test_restriction_at_due_precedes_auto_payment_and_release_does_not_duplicate_timer(self):
        sim = self.make()
        self.restrict(sim, DUE)
        sim.schedule("release", DUE + timedelta(seconds=1), "release_service", account_id=1, service_id=1)
        sim.run(DUE + timedelta(seconds=1))
        self.assertEqual(sim.accounts[1]["balance"], 9900)
        self.assertEqual(len([e for e in sim.events if e["type"] == "renewal_paused"]), 1)
        self.assertEqual(sim.services[1]["status"], "prolong_wait")

    def test_replay_matches_live_state_and_snapshot_hides_lifecycle_internals(self):
        sim = self.make()
        self.buy_new(sim)
        self.confirm(sim)
        self.restrict(sim, START + timedelta(days=12))
        sim.schedule("release", DUE + timedelta(days=2), "release_service", account_id=1, service_id=1)
        sim.run(START + timedelta(days=12))
        blocked = sim.snapshot()
        sim.run(DUE + timedelta(days=2))
        self.assertEqual(sim.snapshot(START + timedelta(days=12)), blocked)
        accounts, services = {}, {}
        for event in sim.events:
            billing.apply_event(accounts, services, event)
        self.assertEqual(accounts, sim.accounts)
        self.assertEqual(services, sim.services)
        for service in sim.snapshot()["services"]:
            self.assertFalse({"replacementServiceId", "replacesServiceId", "restrictionReason"} & service.keys())
        self.assertEqual(sim.export()["schemaVersion"], "vps-billing-events-v2")


if __name__ == "__main__":
    unittest.main()
