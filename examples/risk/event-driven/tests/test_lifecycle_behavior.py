"""Lifecycle causality tests; synthetic histories do not validate real ML accuracy."""
import copy
import sys
import unittest
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
import behavior
from billing import instant, iso, period_end


class LifecycleBehaviorTests(unittest.TestCase):
    def profile(self, days=800, monthly_budget=2000):
        profile = behavior.make_population(4, 1, days=days)[0]
        profile["services"] = [{"serviceId": 3000010, "term": "month", "priceMinor": 593,
                                "tariffCode": "WAWs-1", "needUntil": "2030-01-01T00:00:00Z"}]
        profile["habits"].update(preChance=1, autoChance=1, attendChance=1,
                                  reserveMultiplier=1, leadFraction=.2, topupDelayHours=.01,
                                  manualDelayHours=.01, followThroughChance=1,
                                  cancelDelayHours=.01, checkIntervalHours=24)
        profile["budget"]["monthlyAllowances"] = {key: monthly_budget for key in profile["budget"]["monthlyAllowances"]}
        profile["budget"]["monthlyAllowances"]["2026-01"] += 593
        return profile

    def replacement(self, profile, succeeds=True, price=849):
        profile["lifecyclePlans"] = [{
            "kind": "replace", "serviceId": 3000010, "at": "2026-02-10T00:00:00Z",
            "transferSucceeds": succeeds, "transferDelayHours": 2,
            "replacement": {"serviceId": 300001001, "term": "month", "priceMinor": price,
                            "tariffCode": "WAWs-2", "needUntil": "2030-01-01T00:00:00Z"}}]

    def test_same_seed_reproducible_and_cohorts_staggered(self):
        a = behavior.build(seed=12, account_count=8, days=400, lifecycle=True)
        self.assertEqual(a, behavior.build(seed=12, account_count=8, days=400, lifecycle=True))
        created = {e["occurredAt"] for e in a["observations"]["events"] if e["type"] == "service_created"}
        self.assertGreater(len(created), 1)
        self.assertEqual(a["report"]["generatorVersion"], "lifecycle-v3-retention-registration")

    def test_archetype_does_not_change_lifecycle_money_or_need(self):
        original = behavior.make_population(8, 5, days=1095, lifecycle=True)
        for archetype in behavior.ARCHETYPES:
            other = behavior.make_population(8, 5, days=1095, lifecycle=True,
                                             archetype_overrides={300001: archetype})
            for a, b in zip(original, other):
                for key in ("budget", "services", "createdAt", "lifecyclePlans"):
                    self.assertEqual(a[key], b[key])

    def test_long_need_does_not_change_habits_budget_or_tariffs(self):
        settings = {"cohortSpreadDays": 0, "longNeedChance": 0}
        short = behavior.make_population(8, 10, days=1095, lifecycle=settings)
        long = behavior.make_population(8, 10, days=1095, lifecycle={**settings, "longNeedChance": 1})
        for a, b in zip(short, long):
            self.assertEqual(a["habits"], b["habits"])
            self.assertEqual(a["budget"], b["budget"])
            for sa, sb in zip(a["services"], b["services"]):
                self.assertEqual({k: v for k, v in sa.items() if k != "needUntil"},
                                 {k: v for k, v in sb.items() if k != "needUntil"})
                self.assertGreater((instant(sb["needUntil"]) - instant(sb["createdAt"])).days, 729)

    def test_long_stable_history_still_fails_when_budget_ends(self):
        profile = self.profile()
        for month in profile["budget"]["monthlyAllowances"]:
            if month >= "2027-06":
                profile["budget"]["monthlyAllowances"][month] = 0
        result = behavior.build(seed=1, days=650, population=[profile])
        payments = [e for e in result["observations"]["events"]
                    if e["type"] == "payment_succeeded" and e["mode"] != "purchase"]
        self.assertGreaterEqual(len(payments), 16)
        self.assertTrue(all(instant(e["occurredAt"]) <= instant(e["dueAt"]) for e in payments))
        self.assertIn(1, [c["label"] for c in result["report"]["cycles"]])
        self.assertTrue(any(e["type"] == "payment_failed" for e in result["observations"]["events"]))

    def test_long_stable_history_still_stops_when_need_ends(self):
        profile = self.profile()
        profile["services"][0]["needUntil"] = "2027-06-15T00:00:00Z"
        result = behavior.build(seed=1, days=620, population=[profile])
        payments = [e for e in result["observations"]["events"] if e["type"] == "payment_succeeded"]
        self.assertGreater(len(payments), 13)
        self.assertTrue(all(instant(e["occurredAt"]) < instant("2027-06-15T00:00:00Z") for e in payments))
        self.assertIn(1, [c["label"] for c in result["report"]["cycles"]])

    def test_confirmed_replacement_stops_old_but_not_new_payments(self):
        profile = self.profile(days=120)
        self.replacement(profile)
        result = behavior.build(seed=1, days=120, population=[profile])
        events = result["observations"]["events"]
        confirmation = next(e for e in events if e["type"] == "replacement_confirmed")
        self.assertFalse(any(e["type"] == "payment_succeeded" and e["serviceId"] == 3000010
                             and instant(e["occurredAt"]) > instant(confirmation["occurredAt"]) for e in events))
        self.assertTrue(any(e["type"] == "payment_succeeded" and e["serviceId"] == 300001001
                            and e["mode"] != "purchase" for e in events))
        self.assertTrue(any(c["reason"] == "confirmed_replacement" for c in result["report"]["cycles"]))

    def test_failed_transfer_keeps_source_and_does_not_refund_purchase(self):
        profile = self.profile(days=120)
        self.replacement(profile, succeeds=False)
        result = behavior.build(seed=1, days=120, population=[profile])
        events = result["observations"]["events"]
        self.assertFalse(any(e["type"] == "replacement_confirmed" for e in events))
        self.assertTrue(any(e["type"] == "payment_succeeded" and e["serviceId"] == 3000010
                            and instant(e["occurredAt"]) > instant("2026-03-01T00:00:00Z") for e in events))
        target_payments = [e for e in events if e["type"] == "payment_succeeded" and e["serviceId"] == 300001001]
        self.assertEqual([e["mode"] for e in target_payments], ["purchase"])

    def test_failed_purchase_keeps_old_need_and_normal_renewal(self):
        profile = self.profile(days=120, monthly_budget=1000)
        self.replacement(profile, price=100000)
        result = behavior.build(seed=1, days=120, population=[profile])
        events = result["observations"]["events"]
        self.assertFalse(any(e["type"] == "replacement_confirmed" for e in events))
        self.assertTrue(any(e["type"] == "action_rejected" and e.get("action") == "purchase" for e in events))
        self.assertTrue(any(e["type"] == "payment_succeeded" and e["serviceId"] == 3000010
                            and instant(e["occurredAt"]) >= instant("2026-03-01T00:00:00Z") for e in events))

    def test_restriction_pauses_payment_and_release_needs_separate_manual(self):
        profile = self.profile(days=70)
        profile["lifecyclePlans"] = [{"kind": "restrict", "serviceId": 3000010,
                                      "at": "2026-01-20T00:00:00Z", "releaseAt": "2026-02-08T00:00:00Z",
                                      "reason": "policy_review"}]
        result = behavior.build(seed=1, days=70, population=[profile])
        events = result["observations"]["events"]
        self.assertTrue(any(e["type"] == "renewal_paused" for e in events))
        release = next(e for e in events if e["type"] == "service_released")
        self.assertFalse(any(e["type"] == "payment_succeeded"
                             and instant("2026-01-20T00:00:00Z") <= instant(e["occurredAt"]) <= instant(release["occurredAt"])
                             for e in events))
        recovery = next(e for e in events if e["type"] == "payment_succeeded"
                        and instant(e["occurredAt"]) > instant(release["occurredAt"]))
        self.assertEqual(recovery["mode"], "manual")
        self.assertIn("excluded_nonfinancial_block", [c["reason"] for c in result["report"]["cycles"]])

    def test_multiyear_money_conservation_and_no_private_leaks(self):
        result = behavior.build(seed=20261024, account_count=15, days=1095, lifecycle=True)
        incoming, outgoing, monthly = defaultdict(int), defaultdict(int), defaultdict(int)
        for event in result["observations"]["events"]:
            aid = event["accountId"]
            if event["type"] == "wallet_topped_up":
                incoming[aid] += event["amountMinor"]
                monthly[aid, behavior.month_key(event["occurredAt"])] += event["amountMinor"]
            if event["type"] == "payment_succeeded":
                outgoing[aid] += event["amountMinor"]
        for profile in result["diagnostics"]["profiles"]:
            for month, allowance in profile["budget"]["monthlyAllowances"].items():
                self.assertLessEqual(monthly[profile["accountId"], month], allowance)
        for account in result["observations"]["snapshot"]["accounts"]:
            self.assertEqual(account["balance"], incoming[account["id"]] - outgoing[account["id"]])
        forbidden = {"archetype", "habits", "budget", "needUntil", "lifecyclePlans", "transferSucceeds", "lifecycleSettings"}
        def walk(obj):
            if isinstance(obj, dict):
                self.assertFalse(forbidden.intersection(obj))
                for child in obj.values():
                    walk(child)
            elif isinstance(obj, list):
                for child in obj:
                    walk(child)
        walk(result["observations"])

    def test_generated_multiyear_data_contains_observed_long_streaks(self):
        result = behavior.build(seed=20261024, account_count=50, days=1095, lifecycle=True)
        streaks, maximum = defaultdict(int), 0
        for event in result["observations"]["events"]:
            if event["type"] == "payment_succeeded" and event["mode"] != "purchase":
                sid = event["serviceId"]
                streaks[sid] = streaks[sid] + 1 if instant(event["occurredAt"]) <= instant(event["dueAt"]) else 0
                maximum = max(maximum, streaks[sid])
        self.assertGreaterEqual(maximum, 12)
        self.assertGreater(result["report"]["cycleReasons"]["not_renewed_within_5d"], 0)

    def test_already_dispatched_manual_finishes_before_confirmation(self):
        profile = self.profile(days=120, monthly_budget=5000)
        self.replacement(profile)
        world = behavior.BehaviorSimulation([profile], 1, days=120)
        transfer_at = instant("2026-02-10T02:00:00Z") + behavior.EPS
        world.command(transfer_at - behavior.EPS, "top_up", account_id=300001, amount=593)
        world.task(transfer_at, "manual", sid=3000010, due="2026-03-01T00:00:00Z")
        observations = world.run()
        events = observations["events"]
        confirmation = next(e for e in events if e["type"] == "replacement_confirmed")
        self.assertTrue(any(e["type"] == "payment_succeeded" and e["mode"] == "manual"
                            and e["serviceId"] == 3000010 and instant(e["occurredAt"]) == transfer_at + behavior.EPS
                            for e in events))
        self.assertFalse(any(e["type"] == "payment_succeeded" and e["serviceId"] == 3000010
                             and instant(e["occurredAt"]) >= instant(confirmation["occurredAt"]) for e in events))

    def test_rejected_confirmation_cleans_target_and_keeps_source(self):
        profile = self.profile(days=120)
        self.replacement(profile)
        world = behavior.BehaviorSimulation([profile], 1, days=120)
        world.command(instant("2026-02-10T02:00:00Z") + 2 * behavior.EPS,
                      "restrict_service", account_id=300001, service_id=300001001, reason="policy_review")
        events = world.run()["events"]
        self.assertTrue(any(e["type"] == "action_rejected" and e.get("action") == "confirm_replacement" for e in events))
        self.assertFalse(world.sim.services[300001001]["autoProlong"])
        self.assertNotIn(3000010, world.pending_switches)
        self.assertTrue(any(e["type"] == "payment_succeeded" and e["serviceId"] == 3000010
                            and instant(e["occurredAt"]) > instant("2026-03-01T00:00:00Z") for e in events))

    def test_invalid_plan_fails_before_simulation(self):
        profile = self.profile(days=120)
        self.replacement(profile)
        invalid = [dict(transferDelayHours=-1), dict(transferSucceeds="yes"), dict(kind="unknown")]
        for change in invalid:
            fixture = copy.deepcopy(profile)
            fixture["lifecyclePlans"][0].update(change)
            with self.assertRaises(ValueError):
                behavior.BehaviorSimulation([fixture], 1, days=120)
        profile["lifecyclePlans"] = [{"kind": "restrict", "serviceId": 3000010,
                                      "at": "2026-01-20T00:00:00Z", "releaseAt": "2026-01-19T00:00:00Z",
                                      "reason": "policy_review"}]
        with self.assertRaises(ValueError):
            behavior.BehaviorSimulation([profile], 1, days=120)


if __name__ == "__main__":
    unittest.main()
