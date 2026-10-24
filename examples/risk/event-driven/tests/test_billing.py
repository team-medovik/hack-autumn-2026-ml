import importlib.util
import random
import unittest
from datetime import timedelta
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("event_driven_billing", Path(__file__).parents[1] / "billing.py")
billing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(billing)
BillingSimulator, instant, iso = billing.BillingSimulator, billing.instant, billing.iso
START = instant("2026-09-01T00:00:00Z")
DUE = instant("2026-10-01T00:00:00Z")


class BillingTests(unittest.TestCase):
    def make(self, price=593, balance=593, auto=True, term="month"):
        sim = BillingSimulator(START)
        sim.add_account(1)
        sim.schedule("fund", START, "top_up", account_id=1, amount=balance)
        sim.schedule("buy", START, "purchase", account_id=1, service_id=1, price=price,
                     term=term, tariff_code="WAWs-1", auto_prolong=auto)
        return sim

    def test_purchase_requires_cash_and_updates_ledger(self):
        sim = self.make(balance=1000)
        snap = sim.run(START)
        self.assertEqual(snap["accounts"][0]["balance"], 407)
        self.assertEqual(snap["accounts"][0]["totalReplenished"], 1000)
        self.assertEqual(snap["services"][0]["expiresAt"], iso(DUE))

    def test_unfunded_purchase_does_not_create_service_or_debt(self):
        sim = self.make(balance=500)
        snap = sim.run(START)
        self.assertEqual(snap["services"], [])
        self.assertEqual(snap["accounts"][0]["balance"], 500)
        self.assertEqual(sim.events[-1]["reason"], "insufficient_funds")

    def test_auto_renewal_at_expiry_only(self):
        sim = self.make(balance=1186)
        sim.run(DUE - timedelta(microseconds=1))
        self.assertEqual(sim.accounts[1]["balance"], 593)
        sim.run(DUE)
        self.assertEqual(sim.accounts[1]["balance"], 0)
        self.assertEqual(sim.services[1]["expiresAt"], "2026-11-01T00:00:00Z")

    def test_failed_auto_has_no_debit_reservation_or_repeated_attempt(self):
        sim = self.make(balance=600)
        sim.run(DUE + timedelta(days=10))
        failures = [e for e in sim.events if e["type"] == "payment_failed"]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["balanceBeforeMinor"], 7)
        self.assertEqual(failures[0]["balanceAfterMinor"], 7)
        self.assertEqual(sim.services[1]["status"], "deleted")
        self.assertEqual(sim.accounts[1]["balance"], 7)

    def test_late_top_up_is_not_a_renewal(self):
        sim = self.make()
        sim.schedule("late", DUE + timedelta(hours=1), "top_up", account_id=1, amount=1000)
        sim.run(DUE + timedelta(hours=2))
        self.assertEqual(sim.accounts[1]["balance"], 1000)
        self.assertEqual(sim.services[1]["expiresAt"], iso(DUE))
        self.assertEqual(sim.services[1]["status"], "prolong_wait")
        self.assertEqual(len([e for e in sim.events if e["type"] == "payment_succeeded"]), 1)

    def test_manual_after_late_topup_spends_and_starts_new_paid_period(self):
        sim = self.make()
        sim.schedule("late", DUE + timedelta(hours=1), "top_up", account_id=1, amount=1000)
        sim.schedule("manual", DUE + timedelta(hours=2), "manual_renew",
                     account_id=1, service_id=1, expected_due=iso(DUE))
        sim.run(DUE + timedelta(hours=2))
        self.assertEqual(sim.accounts[1]["balance"], 407)
        self.assertEqual(sim.services[1]["status"], "active")
        self.assertEqual(sim.services[1]["expiresAt"], "2026-11-01T02:00:00Z")
        self.assertEqual(sim.events[-1]["mode"], "manual")

    def test_disabled_auto_does_not_spend_available_balance(self):
        sim = self.make(balance=1186, auto=False)
        sim.run(DUE)
        self.assertEqual(sim.accounts[1]["balance"], 593)
        self.assertFalse(any(e["type"] == "payment_failed" for e in sim.events))
        self.assertEqual(sim.services[1]["status"], "prolong_wait")

    def test_enabling_auto_after_expiry_does_not_retry(self):
        sim = self.make(balance=1186, auto=False)
        sim.schedule("enable", DUE + timedelta(hours=1), "set_auto_prolong",
                     account_id=1, service_id=1, enabled=True)
        sim.run(DUE + timedelta(hours=2))
        self.assertEqual(sim.accounts[1]["balance"], 593)
        self.assertEqual(sim.services[1]["status"], "prolong_wait")

    def test_disable_before_expiry_prevents_auto(self):
        sim = self.make(balance=1186)
        sim.schedule("disable", DUE - timedelta(hours=1), "set_auto_prolong",
                     account_id=1, service_id=1, enabled=False)
        sim.run(DUE)
        self.assertEqual(sim.accounts[1]["balance"], 593)

    def test_failed_expensive_service_does_not_block_cheaper_service(self):
        sim = self.make(price=1187, balance=1780)
        sim.schedule("buy2", START, "purchase", account_id=1, service_id=2, price=593,
                     term="month", tariff_code="WAWs-1", auto_prolong=True)
        sim.schedule("next", DUE - timedelta(hours=1), "top_up", account_id=1, amount=600)
        sim.run(DUE)
        self.assertEqual(sim.accounts[1]["balance"], 7)
        self.assertEqual(sim.services[1]["status"], "prolong_wait")
        self.assertEqual(sim.services[2]["status"], "active")

    def test_simultaneous_expiries_use_id_not_purchase_order_or_price(self):
        # Buy ID 2 before ID 1 executes by scheduling ID 1 later in another simulation.
        other = BillingSimulator(START)
        other.add_account(1)
        other.schedule("fund", START, "top_up", account_id=1, amount=1779)
        for sid in (2, 1):
            other.schedule(f"buy{sid}", START, "purchase", account_id=1, service_id=sid,
                           price=593, term="month", tariff_code="WAWs-1", auto_prolong=True)
        other.run(DUE)
        self.assertEqual(other.services[1]["status"], "active")
        self.assertEqual(other.services[2]["status"], "prolong_wait")

    def test_priority_field_is_not_supported(self):
        sim = BillingSimulator(START)
        sim.add_account(1)
        with self.assertRaises(ValueError):
            sim.schedule("buy", START, "purchase", account_id=1, service_id=1, price=593,
                         term="month", tariff_code="WAWs-1", priority=100)

    def test_duplicate_command_is_idempotent_even_after_execution(self):
        sim = self.make(balance=1186)
        sim.run(START)
        self.assertFalse(sim.schedule("fund", START, "top_up", account_id=1, amount=1186))
        self.assertEqual(sim.accounts[1]["balance"], 593)
        with self.assertRaises(ValueError):
            sim.schedule("fund", START, "top_up", account_id=1, amount=1187)

    def test_manual_at_expiry_invalidates_auto_timer(self):
        sim = self.make(balance=1779)
        sim.schedule("manual", DUE, "manual_renew", account_id=1, service_id=1, expected_due=iso(DUE))
        sim.run(DUE)
        self.assertEqual(sim.accounts[1]["balance"], 593)
        payments = [e for e in sim.events if e["type"] == "payment_succeeded"]
        self.assertEqual([e["mode"] for e in payments], ["purchase", "manual"])

    def test_early_manual_extends_existing_term_and_stale_request_cannot_charge_again(self):
        sim = self.make(balance=1779)
        for i in (1, 2):
            sim.schedule(f"manual{i}", START + timedelta(days=15, hours=i), "manual_renew",
                         account_id=1, service_id=1, expected_due=iso(DUE))
        sim.run(DUE)
        self.assertEqual(sim.accounts[1]["balance"], 593)
        self.assertEqual(sim.services[1]["expiresAt"], "2026-11-01T00:00:00Z")
        self.assertTrue(any(e.get("reason") == "stale_expiry" for e in sim.events))

    def test_failed_early_manual_does_not_end_paid_service(self):
        sim = self.make()
        sim.schedule("manual", START + timedelta(days=1), "manual_renew",
                     account_id=1, service_id=1, expected_due=iso(DUE))
        sim.run(START + timedelta(days=2))
        self.assertEqual(sim.services[1]["status"], "active")
        self.assertEqual(sim.services[1]["expiresAt"], iso(DUE))

    def test_separate_account_cannot_renew_someone_elses_service(self):
        sim = self.make()
        sim.add_account(2)
        sim.schedule("fund2", START, "top_up", account_id=2, amount=2000)
        sim.schedule("wrong", DUE, "manual_renew", account_id=2, service_id=1, expected_due=iso(DUE))
        sim.run(DUE)
        self.assertEqual(sim.accounts[2]["balance"], 2000)
        self.assertEqual(sim.services[1]["status"], "prolong_wait")

    def test_hourly_payment_every_60_minutes_until_funds_end(self):
        sim = self.make(price=4, balance=12, term="hour")
        sim.run(START + timedelta(hours=5))
        successes = [e for e in sim.events if e["type"] == "payment_succeeded"]
        self.assertEqual([e["occurredAt"] for e in successes], [iso(START + timedelta(hours=i)) for i in range(3)])
        self.assertEqual(sim.accounts[1]["balance"], 0)
        self.assertEqual(len([e for e in sim.events if e["type"] == "payment_failed"]), 1)

    def test_hourly_auto_off_deleted_after_24h_without_manual_payment(self):
        sim = self.make(price=4, balance=12, term="hour", auto=False)
        sim.run(START + timedelta(hours=24) - timedelta(microseconds=1))
        self.assertEqual(sim.services[1]["status"], "prolong_wait")
        sim.run(START + timedelta(hours=24))
        self.assertEqual(sim.services[1]["status"], "deleted")
        sim.schedule("late-manual", START + timedelta(hours=25), "manual_renew",
                     account_id=1, service_id=1, expected_due=iso(START + timedelta(hours=1)))
        sim.run(START + timedelta(hours=25))
        self.assertEqual(sim.accounts[1]["balance"], 8)
        self.assertEqual(sim.services[1]["status"], "deleted")

    def test_hourly_manual_resets_deletion_window(self):
        sim = self.make(price=4, balance=12, term="hour", auto=False)
        sim.schedule("manual", START + timedelta(hours=20), "manual_renew",
                     account_id=1, service_id=1, expected_due=iso(START + timedelta(hours=1)))
        sim.run(START + timedelta(hours=24))
        self.assertEqual(sim.services[1]["status"], "prolong_wait")
        sim.run(START + timedelta(hours=44))
        self.assertEqual(sim.services[1]["status"], "deleted")

    def test_hourly_manual_at_deletion_boundary_does_not_resurrect(self):
        sim = self.make(price=4, balance=12, term="hour", auto=False)
        sim.schedule("manual", START + timedelta(hours=24), "manual_renew",
                     account_id=1, service_id=1, expected_due=iso(START + timedelta(hours=1)))
        sim.run(START + timedelta(hours=24))
        self.assertEqual(sim.services[1]["status"], "deleted")
        self.assertEqual(sim.accounts[1]["balance"], 8)

    def test_calendar_month_and_synthetic_day(self):
        self.assertEqual(iso(billing.period_end(instant("2024-01-31T10:00:00Z"), "month")), "2024-02-29T10:00:00Z")
        self.assertEqual(iso(billing.period_end(instant("2026-12-31T10:00:00Z"), "month")), "2027-01-31T10:00:00Z")
        sim = self.make(price=96, balance=192, term="day")
        sim.run(START + timedelta(days=1))
        self.assertEqual(sim.accounts[1]["balance"], 0)
        self.assertEqual(sim.services[1]["expiresAt"], iso(START + timedelta(days=2)))

    def test_snapshot_replay_matches_live_state_and_future_has_no_effect(self):
        sim = self.make()
        before = sim.run(START)
        sim.schedule("future", DUE + timedelta(days=1), "top_up", account_id=1, amount=1000)
        sim.run(DUE + timedelta(days=2))
        self.assertEqual(sim.snapshot(START), before)
        snap = sim.snapshot()
        self.assertEqual(snap["accounts"], list(sim.accounts.values()))
        self.assertEqual(snap["services"][0]["status"], sim.services[1]["status"])
        self.assertFalse(any(instant(e["occurredAt"]) > sim.now for e in sim.export()["events"]))

    def test_money_conservation_and_failed_payment_has_zero_effect(self):
        sim = self.make(balance=1200)
        sim.schedule("later", DUE + timedelta(days=40), "top_up", account_id=1, amount=300)
        sim.run(DUE + timedelta(days=50))
        funds = sum(e["amountMinor"] for e in sim.events if e["type"] == "wallet_topped_up")
        debits = sum(e["amountMinor"] for e in sim.events if e["type"] == "payment_succeeded")
        self.assertEqual(funds - debits, sim.accounts[1]["balance"])
        for e in sim.events:
            if e["type"] == "payment_failed":
                self.assertEqual(e["balanceBeforeMinor"], e["balanceAfterMinor"])

    def test_determinism_and_incremental_run(self):
        first, second = self.make(balance=2000), self.make(balance=2000)
        first.run(DUE + timedelta(days=40))
        second.run(START)
        second.run(DUE)
        second.run(DUE + timedelta(days=40))
        self.assertEqual(first.export(), second.export())

    def test_legacy_export_never_truncates_real_price(self):
        sim = self.make(price=13572, balance=13572)
        sim.run(START)
        self.assertEqual(sim.events[-1]["amountMinor"], 13572)
        self.assertEqual(sim.snapshot()["transactions"], [])
        with self.assertRaisesRegex(ValueError, "limit 1000"):
            sim.snapshot(include_transactions=True)
        small = self.make()
        small.run(DUE)
        tx = small.snapshot(include_transactions=True)["transactions"]
        self.assertEqual([t["status"] for t in tx], ["performed", "not_performed"])
        self.assertIsNone(tx[-1]["performedAt"])

    def test_reject_invalid_money_timezone_and_past_actions(self):
        sim = self.make()
        for value in (-1, 0, 1.5, True, 2**63):
            with self.subTest(value=value), self.assertRaises(ValueError):
                sim.schedule(f"bad-{value}", START, "top_up", account_id=1, amount=value)
        with self.assertRaises(ValueError):
            BillingSimulator("2026-01-01T00:00:00")
        sim.run(DUE)
        with self.assertRaises(ValueError):
            sim.schedule("past", START, "top_up", account_id=1, amount=1)
        with self.assertRaises(ValueError):
            sim.run(START)

    def test_auto_is_off_by_default(self):
        sim = BillingSimulator(START)
        sim.add_account(1)
        sim.schedule("fund", START, "top_up", account_id=1, amount=1200)
        sim.schedule("buy", START, "purchase", account_id=1, service_id=1,
                     price=593, term="month", tariff_code="WAWs-1")
        sim.run(DUE)
        self.assertFalse(sim.services[1]["autoProlong"])
        self.assertEqual(sim.accounts[1]["balance"], 607)

    def test_same_instant_topup_before_expiry_is_explicit_convention(self):
        sim = self.make()
        sim.schedule("at-deadline", DUE, "top_up", account_id=1, amount=593)
        sim.run(DUE)
        self.assertEqual(sim.accounts[1]["balance"], 0)
        self.assertEqual(sim.services[1]["status"], "active")
        self.assertEqual([e["type"] for e in sim.events if e["occurredAt"] == iso(DUE)],
                         ["wallet_topped_up", "renewal_due", "status_changed", "payment_succeeded"])

    def test_topup_overflow_is_rejected_without_partial_mutation(self):
        sim = self.make(balance=2**63 - 1)
        sim.run(START)
        before = sim.snapshot()
        sim.schedule("overflow", START + timedelta(hours=1), "top_up", account_id=1, amount=1)
        with self.assertRaises(ValueError):
            sim.run(START + timedelta(hours=1))
        self.assertEqual(sim.snapshot(), before)

    def test_multi_account_ledger_invariants_across_fixed_random_seeds(self):
        # Varied scripted actions test the engine, not customer realism or ML accuracy.
        for seed in (7, 19, 20261024):
            rng = random.Random(seed)
            sim = BillingSimulator(START)
            for aid in range(1, 11):
                sim.add_account(aid)
                sim.schedule(f"fund-{aid}", START, "top_up", account_id=aid, amount=10000)
                for index, term in enumerate(("month", "day", "hour")):
                    sid = aid * 10 + index
                    price = {"month": 1187, "day": 96, "hour": 4}[term]
                    sim.schedule(f"buy-{sid}", START, "purchase", account_id=aid, service_id=sid,
                                 price=price, term=term, tariff_code="SYNTHETIC-TEST",
                                 auto_prolong=bool(rng.randrange(2)))
                for day in (10, 20, 40):
                    sim.schedule(f"fund-{aid}-{day}", START + timedelta(days=day), "top_up",
                                 account_id=aid, amount=rng.randint(1, 3000))
            sim.run(START + timedelta(days=50))
            balances, totals = dict.fromkeys(range(1, 11), 0), dict.fromkeys(range(1, 11), 0)
            for event in sim.events:
                aid = event["accountId"]
                if event["type"] in ("wallet_topped_up", "payment_succeeded", "payment_failed"):
                    self.assertEqual(event["balanceBeforeMinor"], balances[aid])
                    if event["type"] == "wallet_topped_up":
                        balances[aid] += event["amountMinor"]
                        totals[aid] += event["amountMinor"]
                    elif event["type"] == "payment_succeeded":
                        balances[aid] -= event["amountMinor"]
                    self.assertGreaterEqual(balances[aid], 0)
                    self.assertEqual(event["balanceAfterMinor"], balances[aid])
            for aid in balances:
                self.assertEqual(sim.accounts[aid]["balance"], balances[aid])
                self.assertEqual(sim.accounts[aid]["totalReplenished"], totals[aid])


if __name__ == "__main__":
    unittest.main()
