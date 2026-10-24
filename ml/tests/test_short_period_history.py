"""Short-period event contracts and grace labels, not model-score acceptance tests."""
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
from prepare_dataset import GRACE, History, assign_split, iso, timestamp

spec = importlib.util.spec_from_file_location("short_period_generator", ROOT / "examples/risk/short-periods/generate.py")
shortgen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shortgen)


def cycle_row(payment):
    due = timestamp(payment["paidThrough"])
    before = timedelta(minutes=30) if payment["paymentTerm"] == "hour" else timedelta(hours=12)
    return {"serviceId": payment["serviceId"], "accountId": payment["accountId"],
            "predictionAt": iso(due - before), "dueAt": iso(due), "labelDeadline": iso(due + GRACE)}


class ShortPeriodHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = shortgen.build()
        cls.history = History(cls.data)
        cls.by_service = defaultdict(list)
        for event in cls.data["events"]:
            cls.by_service[event["serviceId"]].append(event)

    def test_recurrence_is_exact_hour_or_utc_24_hours_across_calendar_boundaries(self):
        cases = [("2024-02-28T15:00:00Z", "day", "2024-02-29T15:00:00Z"),
                 ("2024-02-29T15:00:00Z", "day", "2024-03-01T15:00:00Z"),
                 ("2026-12-31T23:00:00Z", "hour", "2027-01-01T00:00:00Z"),
                 ("2026-03-28T12:00:00+03:00", "day", "2026-03-29T09:00:00Z")]
        for before, term, after in cases:
            self.assertEqual(shortgen.iso(shortgen.period_end(datetime.fromisoformat(before.replace("Z", "+00:00")), term)), after)
        with self.assertRaises(ValueError):
            shortgen.period_end(datetime(2026, 1, 1), "day")

    def test_seed_reproducibility_counts_and_source_integrity(self):
        self.assertEqual(self.data, shortgen.build(seed=shortgen.SEED))
        self.assertEqual(self.data, json.loads((shortgen.HERE / "history.json").read_text()))
        self.assertNotEqual(self.data["events"], shortgen.build(seed=shortgen.SEED + 1)["events"])
        self.assertEqual(self.data["serviceCount"], 120)
        self.assertEqual(self.data["accountCount"], 120)
        self.assertEqual(self.data["eventCount"], len(self.data["events"]))
        self.assertEqual(hashlib.sha256(shortgen.SOURCE.read_bytes()).hexdigest(), shortgen.SOURCE_SHA256)
        self.assertTrue(all(s["id"] >= 70001 and s["ownerId"] >= 60001 for s in self.data["syntheticSnapshot"]["services"]))

    def test_money_references_chronology_and_replayed_snapshot(self):
        catalog = {c["code"]: c for c in json.loads(shortgen.CATALOG.read_text())["tariffs"]}
        accounts = {a["accountId"]: a for a in self.data["accountBusiness"]}
        services = {s["id"]: s for s in self.data["syntheticSnapshot"]["services"]}
        self.assertEqual(len({e["id"] for e in self.data["events"]}), len(self.data["events"]))
        event_dates = [e["occurredAt"] for e in self.data["events"]]
        self.assertEqual(event_dates, sorted(event_dates))
        statuses, contracts, expiries = {}, {}, {}
        paid = Counter()
        for event in self.data["events"]:
            sid, aid = event["serviceId"], event["accountId"]
            self.assertEqual(aid, services[sid]["ownerId"])
            self.assertLessEqual(timestamp(accounts[aid]["createdAt"]), timestamp(event["occurredAt"]))
            self.assertLessEqual(timestamp(event["occurredAt"]), shortgen.AS_OF)
            self.assertEqual(event["recordedAt"], event["occurredAt"])
            kind = event["type"]
            if kind == "contract_changed":
                contracts[sid] = event
                multiplier = 24 if event["paymentTerm"] == "day" else 1
                self.assertEqual(event["priceMinor"], catalog[event["tariffCode"]]["hourPriceMinor"] * multiplier)
                if multiplier == 24:
                    self.assertEqual(event["priceProvenance"], "synthetic_24_times_public_hour_price")
            elif kind == "status_changed":
                self.assertEqual(event["fromStatus"], statuses.get(sid))
                statuses[sid] = event["toStatus"]
            elif kind in {"payment_succeeded", "payment_failed"}:
                self.assertEqual(event["amountMinor"], contracts[sid]["priceMinor"])
                self.assertEqual(event["currency"], "EUR")
                self.assertEqual(event["paymentTerm"], contracts[sid]["paymentTerm"])
                self.assertNotEqual(statuses.get(sid), "deleted")
                self.assertLessEqual(timestamp(event["dueAt"]), timestamp(event["occurredAt"]))
                if kind == "payment_succeeded":
                    self.assertEqual(timestamp(event["paidThrough"]), shortgen.period_end(timestamp(event["occurredAt"]), event["paymentTerm"]))
                    expiries[sid] = event["paidThrough"]
                    paid[aid] += event["amountMinor"]
                else:
                    self.assertIsNone(event["paidThrough"])
        for sid, service in services.items():
            self.assertEqual(service["status"], statuses[sid])
            self.assertEqual(service["expiresAt"], expiries[sid])
        for account in self.data["syntheticSnapshot"]["accounts"]:
            self.assertEqual(account["totalReplenished"], paid[account["id"]] + account["balance"])

    def test_each_term_has_both_mature_labels_in_train_and_validation(self):
        coverage = defaultdict(set)
        validation, test = timestamp("2026-08-01T15:00:00Z"), timestamp("2026-09-01T15:00:00Z")
        for event in self.data["events"]:
            if event["type"] != "payment_succeeded":
                continue
            row = cycle_row(event)
            # Stop work for a covered group; no need to build a daily dataset.
            expected_group = "train" if timestamp(row["predictionAt"]) < validation else "validation"
            if coverage[(event["paymentTerm"], expected_group)] == {0, 1}:
                continue
            row["label"] = self.history.label(row)
            split = assign_split(row, validation, test)
            if split in {"train", "validation"}:
                coverage[(event["paymentTerm"], split)].add(row["label"]["value"])
        for term in ("hour", "day"):
            for split in ("train", "validation"):
                self.assertEqual(coverage[(term, split)], {0, 1}, (term, split))

    def test_real_generated_cohort_contains_recovery_delay_buckets_and_stable_runs(self):
        buckets = defaultdict(set)
        stable = Counter()
        for sid, events in self.by_service.items():
            initial = next(e for e in events if e["type"] == "payment_succeeded")
            term = initial["paymentTerm"]
            successes = {e["dueAt"]: e for e in events if e["type"] == "payment_succeeded"}
            uninterrupted = 0
            for event in events:
                if event["type"] == "payment_succeeded" and event["reason"] == "renewal":
                    uninterrupted += 1
                    if uninterrupted >= (24 if term == "hour" else 7):
                        stable[term] += 1
                elif event["type"] == "payment_failed":
                    uninterrupted = 0
                    if timestamp(event["dueAt"]) + GRACE > shortgen.AS_OF:
                        continue
                    paid = successes.get(event["dueAt"])
                    bucket = ("unpaid" if paid is None else "within_grace"
                              if timestamp(paid["occurredAt"]) <= timestamp(event["dueAt"]) + GRACE else "after_grace")
                    buckets[term].add(bucket)
        for term in ("hour", "day"):
            self.assertEqual(buckets[term], {"unpaid", "within_grace", "after_grace"})
            self.assertGreater(stable[term], 0)

    def test_first_cycle_and_exact_grace_boundaries_use_unit_only_scenarios(self):
        # Forced outcomes belong ONLY to these unit fixtures, never the saved cohort.
        for outcome, delay, label in [("paid", 120, 0), ("paid", 121, 1), ("unpaid", 120, 1)]:
            with self.subTest(outcome=outcome, delay=delay), patch.object(shortgen, "renewal_outcome", return_value=(outcome, delay)):
                fixture = shortgen.build(services_per_term_per_month=1)
                history = History(fixture)
                firsts = [e for e in fixture["events"] if e["type"] == "payment_succeeded" and e["reason"] == "initial_purchase"]
                for payment in firsts:
                    row = cycle_row(payment)
                    self.assertEqual(history.label(row)["value"], label)
                    self.assertLess(timestamp(row["predictionAt"]), timestamp(row["dueAt"]))

    def test_incomplete_cycle_is_not_silently_a_negative_label(self):
        future_cycles = [cycle_row(e) for e in self.data["events"] if e["type"] == "payment_succeeded"
                         and timestamp(e["paidThrough"]) + GRACE > shortgen.AS_OF]
        self.assertGreater(len(future_cycles), 0)
        for row in future_cycles[:30]:
            self.assertIsNone(self.history.label(row)["value"])
            self.assertEqual(self.history.label(row)["reason"], "observation_incomplete")


if __name__ == "__main__":
    unittest.main()
