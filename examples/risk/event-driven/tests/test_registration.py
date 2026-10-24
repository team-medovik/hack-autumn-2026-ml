"""Registration is a past event, not an account-age feature fabricated afterwards."""
import copy
import sys
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[2] / "ml"))
from billing import BillingSimulator, instant
import behavior
from prepare_event_driven import EventHistory


START = "2026-01-01T00:00:00Z"
REGISTERED = "2026-01-05T00:00:00Z"
PURCHASED = "2026-01-10T00:00:00Z"


class RegistrationTests(unittest.TestCase):
    def funded_service(self, registration=REGISTERED, purchase=PURCHASED):
        sim = BillingSimulator(START)
        sim.add_account(1, at=registration)
        sim.schedule("fund", purchase, "top_up", account_id=1, amount=1000)
        sim.schedule("buy", purchase, "purchase", account_id=1, service_id=10,
                     price=1000, term="month", tariff_code="TEST-VPS")
        return sim

    def test_future_registration_is_not_visible_in_live_or_historical_state(self):
        sim = self.funded_service()
        self.assertEqual(sim.accounts, {})
        sim.run("2026-01-20T00:00:00Z")
        self.assertEqual(sim.snapshot("2026-01-04T23:59:59Z")["accounts"], [])
        self.assertEqual(len(sim.snapshot(REGISTERED)["accounts"]), 1)
        self.assertEqual(sim.snapshot(REGISTERED)["services"], [])
        self.assertEqual(len(sim.snapshot(PURCHASED)["services"]), 1)
        times = [instant(e["occurredAt"]) for e in sim.events]
        self.assertEqual(times, sorted(times))

    def test_same_instant_registration_precedes_funding_and_purchase(self):
        sim = self.funded_service(registration=PURCHASED)
        sim.run(PURCHASED)
        self.assertEqual([e["type"] for e in sim.events],
                         ["account_created", "wallet_topped_up", "service_created", "payment_succeeded"])
        self.assertEqual(sim.accounts[1]["balance"], 0)

    def test_cannot_schedule_before_registration_or_register_duplicate(self):
        sim = BillingSimulator(START)
        sim.add_account(1, at=REGISTERED)
        with self.assertRaisesRegex(ValueError, "before account registration"):
            sim.schedule("early", START, "top_up", account_id=1, amount=1000)
        with self.assertRaisesRegex(ValueError, "unique"):
            sim.add_account(1, at=PURCHASED)
        with self.assertRaisesRegex(ValueError, "unique"):
            sim.add_account(1)
        self.assertTrue(sim.schedule("early", REGISTERED, "top_up", account_id=1, amount=1000))
        sim.run(PURCHASED)
        self.assertEqual(len([e for e in sim.events if e["type"] == "account_created"]), 1)

    def test_registration_rejects_past_but_can_be_planned_after_advancing(self):
        sim = BillingSimulator(START)
        with self.assertRaises(ValueError):
            sim.add_account(1, at="2025-12-31T00:00:00Z")
        sim.advance(REGISTERED)
        with self.assertRaises(ValueError):
            sim.add_account(1, at=REGISTERED)
        with self.assertRaises(ValueError):
            sim.add_account(1)
        sim.add_account(1, at=PURCHASED)
        sim.run(PURCHASED)
        self.assertEqual(list(sim.accounts), [1])

    def test_legacy_immediate_registration_remains_supported(self):
        sim = BillingSimulator(START)
        sim.add_account(1)
        self.assertEqual(list(sim.accounts), [1])
        self.assertEqual(sim.events[0]["occurredAt"], START)
        sim.schedule("fund", START, "top_up", account_id=1, amount=1000)
        sim.run(START)
        self.assertEqual(sim.accounts[1]["balance"], 1000)

    def test_lifecycle_dates_vary_are_reproducible_and_precede_purchases(self):
        profiles = behavior.make_population(20261024, 30, days=1095, lifecycle=True)
        self.assertEqual(profiles, behavior.make_population(20261024, 30, days=1095, lifecycle=True))
        self.assertEqual(len({p["registeredAt"] for p in profiles}), len(profiles))
        for profile in profiles:
            self.assertLessEqual(instant(START), instant(profile["registeredAt"]))
            self.assertLess(instant(profile["registeredAt"]), instant(profile["createdAt"]))
        overridden = behavior.make_population(20261024, 30, days=1095, lifecycle=True,
                                               archetype_overrides={300001: "planner"})
        self.assertEqual([p["registeredAt"] for p in profiles], [p["registeredAt"] for p in overridden])

    def test_registration_rng_cannot_change_budgets_contracts_needs_or_habits(self):
        original_stream = behavior.stream
        profiles = behavior.make_population(17, 8, days=1095, lifecycle=True)

        class AlwaysStart:
            def random(self):
                return 0

        def altered_stream(seed, aid, namespace):
            return AlwaysStart() if namespace == "registration" else original_stream(seed, aid, namespace)

        with patch.object(behavior, "stream", side_effect=altered_stream):
            changed = behavior.make_population(17, 8, days=1095, lifecycle=True)
        self.assertNotEqual(profiles, changed)
        for original, altered in zip(profiles, changed):
            original.pop("registeredAt")
            self.assertEqual(altered.pop("registeredAt"), START)
            self.assertEqual(original, altered)

    def test_profiles_reject_registration_after_purchase_or_before_start(self):
        profile = behavior.make_population(17, 1, days=120, lifecycle=True)[0]
        for bad in ("2025-12-31T00:00:00Z", "2027-01-01T00:00:00Z"):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "registration"):
                changed = copy.deepcopy(profile)
                changed["registeredAt"] = bad
                behavior.BehaviorSimulation([changed], 17)

    def test_ml_replay_uses_observed_registration_for_age(self):
        sim = self.funded_service()
        observed_at = instant("2026-02-05T00:00:00Z")
        sim.run(observed_at)
        history = EventHistory(sim.export())
        before, _, _ = history._replay(instant(REGISTERED) - timedelta(microseconds=1))
        self.assertEqual(before, {})
        row = history.features_at(observed_at)[0]
        self.assertEqual(row["features"]["account_age_days"], 31)
        self.assertEqual(row["features"]["service_age_days"], 26)


if __name__ == "__main__":
    unittest.main()
