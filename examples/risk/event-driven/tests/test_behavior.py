"""Behavioral contracts, not claims about real Aeza customer frequencies."""
import copy
import importlib.util
import json
import sys
import unittest
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("event_driven_behavior", ROOT / "behavior.py")
behavior = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = behavior
SPEC.loader.exec_module(behavior)


class BehaviorTests(unittest.TestCase):
    def controlled_profile(self, *, archetype="planner", automatic=True, proactive=True,
                           follow_through=True, monthly_budget=2000, need_until="2027-01-01T00:00:00Z"):
        """Explicit boundary fixture; these probabilities never enter generated examples."""
        profile = behavior.make_population(1, 1, days=70,
                                          archetype_overrides={300001: archetype})[0]
        service = profile["services"][0]
        service.update(priceMinor=593, term="month", tariffCode="WAWs-1", needUntil=need_until)
        profile["services"] = [service]
        profile["budget"].update(initialPurchaseMinor=593, monthlyBaseMinor=monthly_budget,
                                monthlyAllowances={"2026-01": 593 + monthly_budget,
                                                   "2026-02": monthly_budget,
                                                   "2026-03": monthly_budget})
        profile["habits"].update(preChance=int(proactive), autoChance=int(automatic),
                                 attendChance=1, reserveMultiplier=1, leadFraction=.2,
                                 topupDelayHours=.01, manualDelayHours=.01,
                                 followThroughChance=int(follow_through), cancelDelayHours=.01)
        return profile

    def test_five_archetypes_and_equal_probability_population(self):
        self.assertEqual(set(behavior.ARCHETYPES),
                         {"planner", "controller", "reactive", "irregular", "project"})
        population = behavior.make_population(20261024, 2500)
        counts = Counter(p["archetype"] for p in population)
        self.assertEqual(set(counts), set(behavior.ARCHETYPES))
        # Equal selection probability does NOT require exactly equal sample counts.
        # Wide, deterministic bounds detect a grossly skewed implementation only.
        for name, count in counts.items():
            with self.subTest(archetype=name):
                self.assertGreater(count, 375)
                self.assertLess(count, 625)

    def test_population_reproducible_and_other_seed_changes_it(self):
        first = behavior.make_population(19, 10)
        self.assertEqual(first, behavior.make_population(19, 10))
        self.assertNotEqual(first, behavior.make_population(20, 10))

    def test_archetype_override_does_not_change_money_or_need(self):
        original = behavior.make_population(19, 8)
        for name in behavior.ARCHETYPES:
            changed = behavior.make_population(19, 8, archetype_overrides={300001: name})
            self.assertEqual(changed[0]["archetype"], name)
            for before, after in zip(original, changed):
                with self.subTest(archetype=name, account=before["accountId"]):
                    self.assertEqual(before["accountId"], after["accountId"])
                    self.assertEqual(before["budget"], after["budget"])
                    self.assertEqual(before["services"], after["services"])
            self.assertEqual(original[1:], changed[1:])

    def test_generated_history_is_reproducible(self):
        self.assertEqual(behavior.build(seed=31, account_count=8, days=70),
                         behavior.build(seed=31, account_count=8, days=70))

    def test_shared_wallet_conserves_money_and_failures_do_not_spend(self):
        result = behavior.build(seed=31, account_count=16, days=90)
        incoming, outgoing = defaultdict(int), defaultdict(int)
        for event in result["observations"]["events"]:
            aid = event["accountId"]
            if event["type"] == "wallet_topped_up":
                self.assertGreater(event["amountMinor"], 0)
                incoming[aid] += event["amountMinor"]
            elif event["type"] == "payment_succeeded":
                outgoing[aid] += event["amountMinor"]
                self.assertEqual(event["balanceBeforeMinor"] - event["amountMinor"],
                                 event["balanceAfterMinor"])
            elif event["type"] == "payment_failed":
                self.assertEqual(event["balanceBeforeMinor"], event["balanceAfterMinor"])
            if "balanceAfterMinor" in event:
                self.assertGreaterEqual(event["balanceAfterMinor"], 0)
        for account in result["observations"]["snapshot"]["accounts"]:
            self.assertEqual(incoming[account["id"]] - outgoing[account["id"]], account["balance"])
            self.assertEqual(incoming[account["id"]], account["totalReplenished"])

    def test_observations_have_no_latent_behavior_or_future_intentions(self):
        result = behavior.build(seed=7, account_count=10, days=70)
        private_keys = {"archetype", "habits", "budget", "monthlyBaseMinor", "monthlyAllowances",
                        "needUntil", "profiles", "decisions"}

        def inspect(value):
            if isinstance(value, dict):
                self.assertFalse(private_keys.intersection(value))
                for child in value.values():
                    inspect(child)
            elif isinstance(value, list):
                for child in value:
                    inspect(child)

        inspect(result["observations"])
        self.assertEqual(len(result["diagnostics"]["profiles"]), 10)

    def test_events_are_chronological_unique_and_within_observation_horizon(self):
        observations = behavior.build(seed=7, account_count=10, days=70)["observations"]
        events = observations["events"]
        self.assertEqual(len({e["id"] for e in events}), len(events))
        # ISO UTC timestamps produced by the billing kernel are lexically ordered.
        from billing import instant
        times = [instant(e["occurredAt"]) for e in events]
        self.assertEqual(times, sorted(times))
        for event in events:
            self.assertLessEqual(instant(event["occurredAt"]), instant(observations["asOf"]))
            self.assertLessEqual(instant(event["recordedAt"]), instant(observations["asOf"]))

    def test_input_population_is_not_mutated(self):
        population = behavior.make_population(13, 5, days=70)
        before = copy.deepcopy(population)
        behavior.build(seed=13, days=70, population=population)
        self.assertEqual(population, before)

    def test_export_is_json_serializable(self):
        result = behavior.build(seed=42, account_count=5, days=40)
        self.assertEqual(json.loads(json.dumps(result)), result)

    def test_proactive_vs_reactive_habits_change_actions_with_same_budget_and_need(self):
        proactive = self.controlled_profile()
        reactive = copy.deepcopy(proactive)
        reactive["archetype"] = "reactive"
        reactive["habits"]["preChance"] = 0
        first = behavior.build(seed=1, days=35, population=[proactive])
        second = behavior.build(seed=1, days=35, population=[reactive])
        self.assertEqual(proactive["budget"], reactive["budget"])
        self.assertEqual(proactive["services"], reactive["services"])
        planned_events = first["observations"]["events"]
        reactive_events = second["observations"]["events"]
        self.assertFalse(any(e["type"] == "payment_failed" for e in planned_events))
        self.assertTrue(any(e["type"] == "payment_succeeded" and e["mode"] == "auto"
                            for e in planned_events))
        failed = next(i for i, e in enumerate(reactive_events) if e["type"] == "payment_failed")
        funded = next(i for i, e in enumerate(reactive_events)
                      if i > failed and e["type"] == "wallet_topped_up")
        manual = next(i for i, e in enumerate(reactive_events)
                      if i > funded and e["type"] == "payment_succeeded" and e["mode"] == "manual")
        self.assertLess(failed, funded)
        self.assertLess(funded, manual)

    def test_controller_can_renew_manually_without_automatic_payment(self):
        profile = self.controlled_profile(archetype="controller", automatic=False)
        result = behavior.build(seed=1, days=40, population=[profile])
        payments = [e for e in result["observations"]["events"] if e["type"] == "payment_succeeded"]
        self.assertIn("manual", [e["mode"] for e in payments])
        self.assertNotIn("auto", [e["mode"] for e in payments])
        self.assertEqual(result["report"]["cycles"][0]["label"], 0)

    def test_planner_can_lose_service_when_external_budget_is_exhausted(self):
        profile = self.controlled_profile(monthly_budget=0)
        result = behavior.build(seed=1, days=40, population=[profile])
        self.assertEqual(result["report"]["cycles"][0]["label"], 1)
        self.assertEqual(result["observations"]["snapshot"]["accounts"][0]["balance"], 0)
        self.assertTrue(any(d["reason"] == "external_budget_exhausted"
                            for d in result["diagnostics"]["decisions"]))

    def test_populated_wallet_does_not_replace_forgotten_manual_action(self):
        profile = self.controlled_profile(archetype="irregular", proactive=False, follow_through=False)
        result = behavior.build(seed=1, days=40, population=[profile])
        snapshot = result["observations"]["snapshot"]
        self.assertEqual(snapshot["services"][0]["status"], "deleted")
        self.assertGreaterEqual(snapshot["accounts"][0]["balance"], 593)
        self.assertEqual(result["report"]["cycles"][0]["label"], 1)
        self.assertFalse(any(e["type"] == "payment_succeeded" and e["mode"] != "purchase"
                             for e in result["observations"]["events"]))

    def test_need_ending_stops_new_intent_despite_available_money(self):
        profile = self.controlled_profile(need_until="2026-01-10T00:00:00Z")
        result = behavior.build(seed=1, days=40, population=[profile])
        events = result["observations"]["events"]
        self.assertTrue(any(e["type"] == "auto_prolong_changed" and not e["enabled"] for e in events))
        self.assertEqual(len([e for e in events if e["type"] == "payment_succeeded"]), 1)
        self.assertEqual(result["report"]["cycles"][0]["label"], 1)

    def test_generated_topups_stay_within_each_account_month_allowance(self):
        result = behavior.build(seed=20261024, account_count=20, days=120)
        totals = defaultdict(int)
        profiles = {p["accountId"]: p for p in result["diagnostics"]["profiles"]}
        for event in result["observations"]["events"]:
            if event["type"] == "wallet_topped_up":
                totals[event["accountId"], event["occurredAt"][:7]] += event["amountMinor"]
        for (aid, month), amount in totals.items():
            self.assertLessEqual(amount, profiles[aid]["budget"]["monthlyAllowances"][month])

    def test_hour_and_day_histories_never_renew_deleted_services(self):
        from billing import instant
        for term, days in (("hour", 30), ("day", 70)):
            result = behavior.build(seed=20261024, account_count=8, days=days, terms=(term,))
            deleted = set()
            for event in result["observations"]["events"]:
                if event["type"] == "status_changed" and event["toStatus"] == "deleted":
                    deleted.add(event["serviceId"])
                if event["type"] == "payment_succeeded":
                    self.assertNotIn(event["serviceId"], deleted)
                    self.assertGreater(instant(event["paidThrough"]), instant(event["occurredAt"]))
            if term == "hour":
                self.assertGreater(len(deleted), 0)

    def test_default_seed_contains_diverse_observed_monthly_outcomes(self):
        from billing import instant
        result = behavior.build(seed=20261024, account_count=50, days=120)
        by_service = defaultdict(list)
        recovery_delays = []
        for cycle in result["report"]["cycles"]:
            by_service[cycle["serviceId"]].append(cycle)
            if cycle["firstPaymentAt"]:
                recovery_delays.append((instant(cycle["firstPaymentAt"]) -
                                        instant(cycle["dueAt"])).total_seconds() / 3600)
        firsts = [min(cycles, key=lambda c: instant(c["dueAt"])) for cycles in by_service.values()]
        self.assertTrue(any(c["label"] == 0 for c in firsts))
        self.assertTrue(any(c["label"] == 1 for c in firsts))
        self.assertTrue(any(0 < delay <= 120 for delay in recovery_delays))
        # Ordinary unpaid monthly services cannot be revived after retention.
        self.assertFalse(any(delay > 120 for delay in recovery_delays))
        self.assertTrue(any(c["label"] == 1 and c["firstPaymentAt"] is None
                            for c in result["report"]["cycles"]))
        self.assertTrue(any(d["reason"] == "need_ended" for d in result["diagnostics"]["decisions"]))
        longest_run = 0
        for cycles in by_service.values():
            run = 0
            for cycle in sorted(cycles, key=lambda c: instant(c["dueAt"])):
                timely = (cycle["firstPaymentAt"] is not None and
                          instant(cycle["firstPaymentAt"]) <= instant(cycle["dueAt"]))
                run = run + 1 if timely else 0
                longest_run = max(longest_run, run)
        # Four simulated months cannot demonstrate a year of stability.
        self.assertGreaterEqual(longest_run, 3)

    def test_default_history_defers_known_unaffordable_manual_attempts(self):
        result = behavior.build(seed=20261024, account_count=50, days=120)
        # This fixed sample has no simultaneous competing payment between the
        # user's balance check and click; the billing kernel still handles races.
        manual_failures = [e for e in result["observations"]["events"]
                           if e["type"] == "payment_failed" and e["mode"] == "manual"]
        self.assertEqual(manual_failures, [])
        self.assertTrue(any(d["reason"] == "manual_deferred_insufficient_wallet"
                            for d in result["diagnostics"]["decisions"]))
        self.assertTrue(any(e["type"] == "payment_failed" and e["mode"] == "auto"
                            for e in result["observations"]["events"]))

    def test_late_recovery_within_retention_can_happen_while_service_is_still_needed(self):
        from billing import instant
        result = behavior.build(seed=20261024, account_count=50, days=120)
        needs = {s["serviceId"]: instant(s["needUntil"])
                 for p in result["diagnostics"]["profiles"] for s in p["services"]}
        late = [c for c in result["report"]["cycles"]
                if c["firstPaymentAt"] and
                instant(c["firstPaymentAt"]) > instant(c["dueAt"])]
        self.assertGreater(len(late), 0)
        for cycle in late:
            self.assertLess(instant(cycle["firstPaymentAt"]), needs[cycle["serviceId"]])
            self.assertLessEqual(instant(cycle["firstPaymentAt"]), instant(cycle["dueAt"]) + timedelta(days=5))
            self.assertEqual(cycle["label"], 0)

    def test_still_needed_service_is_not_revived_after_retention(self):
        from billing import instant
        profile = self.controlled_profile(proactive=False, follow_through=False)
        result = behavior.build(seed=1, days=70, population=[profile])
        deletion = next(e for e in result["observations"]["events"]
                        if e["type"] == "status_changed" and e["toStatus"] == "deleted")
        self.assertLess(instant(deletion["occurredAt"]), instant(profile["services"][0]["needUntil"]))
        self.assertFalse(any(e["type"] == "payment_succeeded" and
                             instant(e["occurredAt"]) >= instant(deletion["occurredAt"])
                             for e in result["observations"]["events"]))

    def test_invalid_review_interval_and_lead_fraction_rejected(self):
        for field, value in (("checkIntervalHours", 0), ("leadFraction", 1)):
            profile = self.controlled_profile()
            profile["habits"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                behavior.build(seed=1, days=40, population=[profile])

    def test_cycle_outcome_uses_payment_not_topup_and_inclusive_five_days(self):
        from billing import instant, iso
        due = instant("2026-02-01T00:00:00Z")
        purchase = {"type": "payment_succeeded", "mode": "purchase", "accountId": 1,
                    "serviceId": 1, "paidThrough": iso(due),
                    "occurredAt": "2026-01-01T00:00:00Z"}
        for delay, expected in ((120, 0), (121, 1)):
            payment = {"type": "payment_succeeded", "mode": "manual", "accountId": 1,
                       "serviceId": 1, "paidThrough": "2026-03-06T00:00:00Z",
                       "dueAt": iso(due), "occurredAt": iso(due + timedelta(hours=delay))}
            result = behavior.cycle_outcomes({"asOf": "2026-02-10T00:00:00Z", "events": [purchase, payment]})
            self.assertEqual(result[0]["label"], expected)
        topup = {"type": "wallet_topped_up", "accountId": 1,
                 "occurredAt": "2026-02-02T00:00:00Z"}
        self.assertEqual(behavior.cycle_outcomes({"asOf": "2026-02-10T00:00:00Z",
                                                "events": [purchase, topup]})[0]["label"], 1)

    def test_cycle_is_pending_until_full_followup_even_if_renewed(self):
        purchase = {"type": "payment_succeeded", "mode": "purchase", "accountId": 1,
                    "serviceId": 1, "paidThrough": "2026-02-01T00:00:00Z",
                    "occurredAt": "2026-01-01T00:00:00Z"}
        payment = {"type": "payment_succeeded", "mode": "manual", "accountId": 1,
                   "serviceId": 1, "paidThrough": "2026-03-01T00:00:00Z",
                   "dueAt": "2026-02-01T00:00:00Z", "occurredAt": "2026-02-01T00:00:00Z"}
        self.assertIsNone(behavior.cycle_outcomes({"asOf": "2026-02-05T23:59:59Z",
                                                 "events": [purchase, payment]})[0]["label"])


if __name__ == "__main__":
    unittest.main()
