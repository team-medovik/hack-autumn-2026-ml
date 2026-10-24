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
GENERATOR_VERSION = "v2"
DEFAULT_ACCOUNTS = 800
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


def build(seed=SEED, account_count=DEFAULT_ACCOUNTS):
    if type(account_count) is not int or account_count <= 0:
        raise ValueError("account_count must be a positive integer")
    source_bytes = SOURCE.read_bytes()
    assert hashlib.sha256(source_bytes).hexdigest() == SOURCE_SHA256
    original = json.loads(source_bytes)
    catalog = json.loads((HERE / "catalog.json").read_text())
    tariffs = catalog["tariffs"]
    rng = random.Random(seed)
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
                   "paymentTerm": term, "autoProlong": rng.random() < 0.85,
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

        # Hidden continuous propensities overlap for all services. They are neither
        # features nor target labels and do not depend on tariff, start date or age.
        # New independent draws each cycle allow a loss at any renewal, including
        # the first, and both retention and loss after an earlier recovery.
        failure_probability = rng.uniform(0.04, 0.22) if term == "month" else rng.uniform(0.002, 0.008)
        cancellation_probability = rng.uniform(0.005, 0.035) if term == "month" else rng.uniform(0.01, 0.06)
        never_pay_probability = rng.uniform(0.12, 0.32)
        late_recovery_probability = rng.uniform(0.12, 0.30)
        promotion_cycles_remaining = 0
        while expires <= AS_OF:
            due = expires
            if rng.random() < cancellation_probability:
                replace = term == "month" and rng.random() < 0.30
                event(due, "service_terminated", account_id, service_id,
                      reason="customer_migration" if replace else "customer_cancellation",
                      intentionalShortTerm=(term == "hour"))
                change_status(due, "deleted", "customer_cancellation")
                if replace and due + timedelta(hours=1) <= AS_OF:
                    make_service(account_id, due + timedelta(hours=1), "mixed_monthly", rng.randrange(len(tariffs)),
                                 replacement_of=service_id)
                break
            if term == "month" and rng.random() < 0.004:
                change_status(due, "blocked", "synthetic_policy_review_not_payment_failure")
                break
            if term == "month":
                if promotion_cycles_remaining:
                    promotion_cycles_remaining -= 1
                    if not promotion_cycles_remaining:
                        price = tariff["monthPriceMinor"]
                        change_contract(due, "synthetic_discount_expired")
                elif rng.random() < 0.05:
                    price = tariff["monthPriceMinor"] * 90 // 100
                    promotion_cycles_remaining = rng.randint(1, 4)
                    change_contract(due, "synthetic_discount_started")
                elif rng.random() < 0.06:
                    group, old_tier = (tariff_index // 5) * 5, tariff_index % 5
                    tier = rng.choice([value for value in range(5) if value != old_tier])
                    tariff_index = group + tier
                    tariff = tariffs[tariff_index]
                    price = tariff["monthPriceMinor"]
                    change_contract(due, "upgrade" if tier > old_tier else "downgrade")

            if rng.random() < failure_probability:
                payment(due, False, rng.choice(["insufficient_funds", "payment_provider_decline"]), due)
                change_status(due, "prolong_wait", "payment_failed")
                suspended_at = due + timedelta(hours=6)
                if suspended_at <= AS_OF:
                    change_status(suspended_at, "suspended", "synthetic_grace_elapsed")
                outcome = rng.random()
                if outcome < never_pay_probability:
                    deleted_at = due + timedelta(days=rng.randint(5, 12))
                    if deleted_at <= AS_OF:
                        event(deleted_at, "service_terminated", account_id, service_id,
                              reason="unpaid_after_synthetic_grace", intentionalShortTerm=False)
                        change_status(deleted_at, "deleted", "unpaid_after_synthetic_grace")
                    break
                # A successful payment after day 5 still restores service, but
                # cannot retroactively change that earlier renewal window's label.
                delay_hours = (rng.randint(121, 240)
                               if outcome < never_pay_probability + late_recovery_probability
                               else rng.randint(8, 120))
                recovered_at = due + timedelta(hours=delay_hours)
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

    earliest_start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    latest_start = AS_OF - timedelta(days=1)
    for number in range(account_count):
        account_id = 10001 + number
        created = datetime(2024, rng.randint(1, 12), rng.randint(1, 25), 12, tzinfo=timezone.utc)
        business_accounts.append({"accountId": account_id, "createdAt": iso(created),
                                  "source": "synthetic", "currency": "EUR"})
        for position in range(1 + number % 3):
            # The age distribution is selected before the independent behavioral
            # profile. New cohorts continue to arrive near validation/test dates.
            start = earliest_start + timedelta(hours=rng.randrange(int((latest_start - earliest_start).total_seconds() // 3600)))
            tariff_index = rng.randrange(len(tariffs))
            term = "month"
            scenario = "mixed_monthly"
            if number % 10 == 2 and position == 0:
                term = "hour"
                scenario = "mixed_hourly"
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
        "schemaVersion": "aeza-vps-business-demo-v1", "generatorVersion": GENERATOR_VERSION,
        "seed": seed, "accountCount": account_count, "asOf": iso(AS_OF),
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
