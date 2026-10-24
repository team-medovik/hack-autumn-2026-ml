#!/usr/bin/env python3
"""Separate short-period demo; day prices and every customer event are synthetic."""
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CATALOG = HERE.parent / "vps-business/catalog.json"
SOURCE = ROOT / "src/test/resources/risk/data.json"
SOURCE_SHA256 = "6e0e224671a4cd22059431b01fb6fed21e075100bb21b1545d94e746e666292e"
SEED = 20261025
AS_OF = datetime(2026, 10, 19, 15, tzinfo=timezone.utc)
GENERATOR_VERSION = "short-periods-v1"


def iso(value):
    if value.tzinfo is None:
        raise ValueError("Timezone required")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def period_end(value, term):
    if value.tzinfo is None:
        raise ValueError("Timezone required")
    hours = {"hour": 1, "day": 24}[term]
    return value.astimezone(timezone.utc) + timedelta(hours=hours)


def renewal_outcome(rng, profile):
    """Return an independent cycle decision; no fixed renewal number determines it."""
    if rng.random() < profile["cancel"]:
        return "cancel", 0
    if rng.random() >= profile["failure"]:
        return "paid", 0
    draw = rng.random()
    if draw < 0.25:
        return "unpaid", rng.randint(120, 288)
    return "paid", rng.randint(121, 240) if draw < 0.45 else rng.randint(8, 120)


def build(seed=SEED, services_per_term_per_month=12):
    if type(services_per_term_per_month) is not int or services_per_term_per_month <= 0:
        raise ValueError("services_per_term_per_month must be a positive integer")
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA256
    catalog = json.loads(CATALOG.read_text())
    rng = random.Random(seed)
    accounts, services, contracts, events, snapshot_accounts = [], [], [], [], []

    def event(when, kind, aid, sid, **fields):
        assert when <= AS_OF
        events.append({"_sequence": len(events), "type": kind, "occurredAt": iso(when),
                       "recordedAt": iso(when), "accountId": aid, "serviceId": sid,
                       "source": "synthetic", **fields})

    for month in range(6, 11):
        for term in ("hour", "day"):
            for _ in range(services_per_term_per_month):
                aid, sid = 60001 + len(services), 70001 + len(services)
                start = datetime(2026, month, rng.randint(1, 10), rng.randint(0, 23), tzinfo=timezone.utc)
                tariff = rng.choice(catalog["tariffs"])
                price = tariff["hourPriceMinor"] * (24 if term == "day" else 1)
                price_source = "synthetic_24_times_public_hour_price" if term == "day" else "public_hour_price_observed"
                profile = {"cancel": rng.uniform(0.0008, 0.002) if term == "hour" else rng.uniform(0.01, 0.04),
                           "failure": rng.uniform(0.002, 0.006) if term == "hour" else rng.uniform(0.08, 0.18)}
                accounts.append({"accountId": aid, "createdAt": iso(start - timedelta(days=rng.randint(1, 365))),
                                 "currency": "EUR", "source": "synthetic"})
                service = {"id": sid, "ownerId": aid, "name": f"short-demo-{sid}", "paymentTerm": term,
                           "autoProlong": True, "createdAt": iso(start), "expiresAt": None,
                           "status": "active", "price": price}
                services.append(service)
                status, due, total_paid = None, start, 0

                def status_change(when, new_status, reason):
                    nonlocal status
                    event(when, "status_changed", aid, sid, fromStatus=status, toStatus=new_status, reason=reason)
                    status = new_status
                    service["status"] = new_status

                def pay(when, expected_due, reason):
                    through = period_end(when, term)
                    event(when, "payment_succeeded", aid, sid, amountMinor=price, currency="EUR",
                          paymentTerm=term, contractVersion=1, dueAt=iso(expected_due), paidThrough=iso(through), reason=reason)
                    service["expiresAt"] = iso(through)
                    return through

                event(start, "service_created", aid, sid, productFamily="virtual_server")
                event(start, "contract_changed", aid, sid, contractVersion=1, tariffCode=tariff["code"],
                      paymentTerm=term, priceMinor=price, currency="EUR", reason="initial_contract",
                      priceProvenance=price_source, catalogObservedAt=catalog["observedAt"])
                due = pay(start, start, "initial_purchase")
                total_paid += price
                status_change(start, "active", "initial_purchase")
                while due <= AS_OF:
                    outcome, delay_hours = renewal_outcome(rng, profile)
                    if outcome == "cancel":
                        event(due, "service_terminated", aid, sid, reason="customer_cancellation", intentionalShortTerm=True)
                        status_change(due, "deleted", "customer_cancellation")
                        break
                    if delay_hours:
                        event(due, "payment_failed", aid, sid, amountMinor=price, currency="EUR", paymentTerm=term,
                              contractVersion=1, dueAt=iso(due), paidThrough=None,
                              reason=rng.choice(["insufficient_funds", "payment_provider_decline"]))
                        status_change(due, "prolong_wait", "payment_failed")
                        if due + timedelta(hours=6) <= AS_OF:
                            status_change(due + timedelta(hours=6), "suspended", "synthetic_grace_elapsed")
                        action_at = due + timedelta(hours=delay_hours)
                        if action_at > AS_OF:
                            break
                        if outcome == "unpaid":
                            event(action_at, "service_terminated", aid, sid,
                                  reason="unpaid_after_synthetic_grace", intentionalShortTerm=False)
                            status_change(action_at, "deleted", "unpaid_after_synthetic_grace")
                            break
                        previous_due = due
                        due = pay(action_at, previous_due, "retry_after_failure")
                        total_paid += price
                        status_change(action_at, "active", "payment_recovered")
                        event(action_at, "service_recovered", aid, sid, outageStartedAt=iso(previous_due),
                              reason="successful_payment_retry")
                    else:
                        due = pay(due, due, "renewal")
                        total_paid += price
                balance = rng.choice([0, price, price * rng.randint(2, 20)])
                snapshot_accounts.append({"id": aid, "email": f"short-{aid}@example.test", "balance": balance,
                                          "withdrawBalance": 0, "totalReplenished": total_paid + balance,
                                          "bonusBalance": 0, "bonusState": "locked"})
                contracts.append({"serviceId": sid, "accountId": aid, "asOf": iso(AS_OF),
                                  "tariffCode": tariff["code"], "contractVersion": 1, "paymentTerm": term,
                                  "priceMinor": price, "currency": "EUR", "source": "synthetic",
                                  "priceProvenance": price_source, "catalogObservedAt": catalog["observedAt"]})
    events.sort(key=lambda e: (e["occurredAt"], e["_sequence"]))
    for number, entry in enumerate(events, 1):
        entry["id"] = f"short-event-{number:07d}"
        del entry["_sequence"]
    return {"schemaVersion": "aeza-vps-business-demo-v1", "generatorVersion": GENERATOR_VERSION,
            "seed": seed, "accountCount": len(accounts), "serviceCount": len(services),
            "eventCount": len(events), "servicesPerTermPerMonth": services_per_term_per_month, "asOf": iso(AS_OF),
            "provenance": {"behavior": "entirely_synthetic", "catalog": "../vps-business/catalog.json",
                           "catalogObservedAt": catalog["observedAt"], "hourPrices": "public_catalog_observation",
                           "dayPrices": "synthetic_24_times_hour_NOT_an_Aeza_day_tariff",
                           "historicalCatalogUse": "demonstration_only_not_point_in_time", "labels": "not_defined",
                           "grace": "abstract_case_5_days_NOT_Aeza_hourly_deletion_policy",
                           "accountBalances": "synthetic_final_observation_not_a_reconciled_ledger"},
            "sourceSnapshot": {"pathFromRepositoryRoot": "src/test/resources/risk/data.json", "sha256": SOURCE_SHA256},
            "accountBusiness": accounts, "currentContracts": contracts, "events": events,
            "syntheticSnapshot": {"accounts": snapshot_accounts, "services": services, "transactions": []}}


if __name__ == "__main__":
    data = build()
    (HERE / "history.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"accounts": data["accountCount"], "services": data["serviceCount"],
                      "events": data["eventCount"], "types": dict(Counter(e["type"] for e in data["events"]))}, indent=2))
