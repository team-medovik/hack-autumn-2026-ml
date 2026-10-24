#!/usr/bin/env python3
"""Reproducible demonstration data, not a model or a real Aeza billing export."""
import calendar
import hashlib
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "src/test/resources/risk/data.json"
SOURCE_SHA256 = "6e0e224671a4cd22059431b01fb6fed21e075100bb21b1545d94e746e666292e"
SEED = 20261024
AS_OF = datetime(2026, 10, 19, 15, tzinfo=timezone.utc)


def iso(value):
    return value.isoformat().replace("+00:00", "Z")


def period_end(value, term):
    if term == "hour":
        return value + timedelta(hours=1)
    month = value.month % 12 + 1
    year = value.year + (value.month == 12)
    return value.replace(year=year, month=month,
                         day=min(value.day, calendar.monthrange(year, month)[1]))


def build():
    source_bytes = SOURCE.read_bytes()
    assert hashlib.sha256(source_bytes).hexdigest() == SOURCE_SHA256
    original = json.loads(source_bytes)
    catalog = json.loads((HERE / "catalog.json").read_text())
    tariffs = catalog["tariffs"]
    rng = random.Random(SEED)
    accounts, services, business_accounts, contracts, scenarios, events = [], [], [], [], [], []

    def event(when, kind, account_id, service_id, **payload):
        assert when <= AS_OF, "Never turn a future scheduled payment into an event"
        events.append({"sequence": len(events) + 1, "occurredAt": iso(when),
                       "recordedAt": iso(when), "type": kind, "accountId": account_id,
                       "serviceId": service_id, "source": "synthetic", **payload})

    def make_service(account_id, start, scenario, tariff_index, term="month", replacement_of=None):
        service_id = 20001 + len(services)
        tariff = tariffs[tariff_index]
        price = tariff["hourPriceMinor" if term == "hour" else "monthPriceMinor"]
        contract_version = 0
        status = None
        expires = period_end(start, term)
        service = {"id": service_id, "ownerId": account_id, "name": f"synthetic-vps-{service_id}",
                   "paymentTerm": term, "autoProlong": scenario not in {"planned_end", "short_lived"},
                   "createdAt": iso(start), "expiresAt": iso(expires), "status": "active", "price": price}
        services.append(service)
        scenarios.append({"serviceId": service_id, "generatorScenario": scenario,
                          "useForTraining": False})

        def change_status(when, value, reason):
            nonlocal status
            if status != value:
                event(when, "status_changed", account_id, service_id,
                      fromStatus=status, toStatus=value, reason=reason)
                status = value

        def change_contract(when, reason):
            nonlocal contract_version
            contract_version += 1
            event(when, "contract_changed", account_id, service_id,
                  contractVersion=contract_version, tariffCode=tariff["code"],
                  paymentTerm=term, priceMinor=price, currency="EUR", reason=reason,
                  catalogObservedAt=catalog["observedAt"])

        def payment(when, success, reason, due):
            nonlocal expires
            next_end = period_end(when, term) if success else None
            event(when, "payment_succeeded" if success else "payment_failed", account_id, service_id,
                  amountMinor=price, currency="EUR", paymentTerm=term,
                  contractVersion=contract_version, dueAt=iso(due), reason=reason,
                  paidThrough=iso(next_end) if success else None)
            if success:
                expires = next_end

        event(start, "service_created", account_id, service_id, productFamily="virtual_server")
        change_contract(start, "initial_contract")
        payment(start, True, "initial_purchase", start)
        change_status(start, "active", "initial_purchase")
        if replacement_of is not None:
            event(start, "replacement_linked", account_id, service_id,
                  replacesServiceId=replacement_of, reason="synthetic_customer_migration")

        cycle = 0
        stop_after = rng.randint(2, 4) if term == "month" else rng.randint(6, 24)
        while expires <= AS_OF:
            due = expires
            cycle += 1
            if scenario in {"planned_end", "short_lived", "replacement", "full_exit", "partial_exit"} and cycle >= stop_after:
                event(due, "service_terminated", account_id, service_id,
                      reason="customer_migration" if scenario == "replacement" else "customer_cancellation",
                      intentionalShortTerm=(scenario == "short_lived"))
                change_status(due, "deleted", "customer_cancellation")
                if scenario == "replacement" and due + timedelta(hours=1) <= AS_OF:
                    make_service(account_id, due + timedelta(hours=1), "healthy", rng.randrange(len(tariffs)),
                                 replacement_of=service_id)
                break
            if scenario == "blocked" and cycle == 2:
                change_status(due, "blocked", "synthetic_policy_review_not_payment_failure")
                break
            if cycle == 2 and scenario in {"upgrade", "downgrade"}:
                group = (tariff_index // 5) * 5
                tier = tariff_index % 5
                tier = min(4, tier + 1) if scenario == "upgrade" else max(0, tier - 1)
                tariff = tariffs[group + tier]
                price = tariff["monthPriceMinor"]
                change_contract(due, scenario)
            if scenario == "promotion" and cycle in {2, 4}:
                price = (tariff["monthPriceMinor"] * 90 // 100 if cycle == 2 else tariff["monthPriceMinor"])
                change_contract(due, "synthetic_discount_started" if cycle == 2 else "synthetic_discount_expired")

            failure = (scenario in {"recovery", "unpaid"} and cycle == 2) or rng.random() < 0.025
            if failure:
                payment(due, False, rng.choice(["insufficient_funds", "payment_provider_decline"]), due)
                change_status(due, "prolong_wait", "payment_failed")
                suspended_at = due + timedelta(hours=6)
                if suspended_at <= AS_OF:
                    change_status(suspended_at, "suspended", "synthetic_grace_elapsed")
                if scenario == "unpaid":
                    deleted_at = due + timedelta(days=5)
                    if deleted_at <= AS_OF:
                        event(deleted_at, "service_terminated", account_id, service_id,
                              reason="unpaid_after_synthetic_grace", intentionalShortTerm=False)
                        change_status(deleted_at, "deleted", "unpaid_after_synthetic_grace")
                    break
                recovered_at = due + timedelta(hours=rng.randint(8, 96))
                if recovered_at > AS_OF:
                    break
                payment(recovered_at, True, "retry_after_failure", due)
                change_status(recovered_at, "active", "payment_recovered")
                event(recovered_at, "service_recovered", account_id, service_id,
                      outageStartedAt=iso(due), reason="successful_payment_retry")
            else:
                payment(due, True, "renewal", due)
        service.update(status=status, expiresAt=iso(expires), price=price)
        contracts.append({"serviceId": service_id, "accountId": account_id,
                          "asOf": iso(AS_OF), "tariffCode": tariff["code"],
                          "contractVersion": contract_version, "paymentTerm": term,
                          "priceMinor": price, "currency": "EUR", "source": "synthetic",
                          "catalogObservedAt": catalog["observedAt"]})

    scenario_names = ["healthy", "recovery", "unpaid", "upgrade", "downgrade", "promotion",
                      "planned_end", "replacement", "blocked", "healthy"]
    for number in range(160):
        account_id = 10001 + number
        created = datetime(2026, rng.randint(1, 3), rng.randint(1, 25), 12, tzinfo=timezone.utc)
        business_accounts.append({"accountId": account_id, "createdAt": iso(created),
                                  "source": "synthetic", "currency": "EUR"})
        for position in range(1 + number % 3):
            scenario = rng.choice(scenario_names)
            if number % 16 == 0:
                scenario = "full_exit"
            elif number % 16 == 1:
                scenario = "partial_exit" if position == 0 else "healthy"
            tariff_index = rng.randrange(len(tariffs))
            # Avoid no-op boundary upgrades/downgrades while retaining random tariff families.
            if scenario == "upgrade" and tariff_index % 5 == 4:
                tariff_index -= 1
            if scenario == "downgrade" and tariff_index % 5 == 0:
                tariff_index += 1
            start = datetime(2026, rng.randint(4, 6), rng.randint(1, 20), rng.randint(0, 23), tzinfo=timezone.utc)
            term = "month"
            if number % 10 == 2 and position == 0:
                term = "hour"
                scenario = "short_lived" if number % 20 == 2 else "healthy"
                start = AS_OF - timedelta(hours=rng.randint(36, 72))
            make_service(account_id, start, scenario, tariff_index, term)

        paid = sum(e.get("amountMinor", 0) for e in events
                   if e["accountId"] == account_id and e["type"] == "payment_succeeded")
        balance = rng.choice([0, rng.randint(1, 500), rng.randint(500, 20000)])
        accounts.append({"id": account_id, "email": f"synthetic-{account_id}@example.test",
                         "balance": balance, "withdrawBalance": 0, "totalReplenished": paid + balance,
                         "bonusBalance": 0, "bonusState": "locked"})

    events.sort(key=lambda e: (e["occurredAt"], e["sequence"]))
    for index, item in enumerate(events, 1):
        item["id"] = f"event-{index:06d}"
        del item["sequence"]
    source_assignments = []
    for index, service in enumerate(original["services"]):
        tariff = tariffs[index % len(tariffs)]
        quoted = {"hour": tariff["hourPriceMinor"], "month": tariff["monthPriceMinor"]}.get(service["paymentTerm"])
        source_assignments.append({"sourceServiceId": service["id"], "hypotheticalTariffCode": tariff["code"],
                                   "assignmentStatus": "synthetic_not_inferred", "sourcePriceMinor": service["price"],
                                   "sourcePaymentTerm": service["paymentTerm"], "publicPriceForSameTermMinor": quoted,
                                   "overwriteSourcePrice": False})
    return {
        "schemaVersion": "aeza-vps-business-demo-v1", "seed": SEED, "asOf": iso(AS_OF),
        "provenance": {"behavior": "entirely_synthetic", "catalog": "catalog.json",
                       "catalogObservedAt": catalog["observedAt"], "historicalCatalogUse": "demonstration_only_not_point_in_time",
                       "labels": "not_defined", "accountBalances": "synthetic_final_observation_not_a_reconciled_ledger",
                       "eventRecordingDelay": "assumed_zero"},
        "sourceSnapshot": {"pathFromRepositoryRoot": "src/test/resources/risk/data.json", "sha256": SOURCE_SHA256,
                           "accounts": len(original["accounts"]), "services": len(original["services"]),
                           "transactions": len(original["transactions"])},
        "sourceServiceTariffHypotheses": source_assignments,
        "syntheticSnapshot": {"accounts": accounts, "services": sorted(services, key=lambda s: s["id"]), "transactions": []},
        "accountBusiness": business_accounts, "currentContracts": sorted(contracts, key=lambda c: c["serviceId"]),
        "events": events, "generatorScenariosDoNotTrainOn": sorted(scenarios, key=lambda s: s["serviceId"]),
    }


if __name__ == "__main__":
    dataset = build()
    (HERE / "history.json").write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n")
    print(f"Generated {len(dataset['syntheticSnapshot']['accounts'])} accounts, "
          f"{len(dataset['syntheticSnapshot']['services'])} services, {len(dataset['events'])} events")
