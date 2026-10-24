#!/usr/bin/env python3
"""Narrow data-contract validation; no model training or quality evaluation."""
import hashlib
import json
from collections import Counter
from datetime import datetime

from generate import HERE, ROOT, build, period_end


def date(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def unique(items, key):
    result = {item[key]: item for item in items}
    assert len(result) == len(items), f"Duplicate {key}"
    return result


def validate(data):
    source = data["sourceSnapshot"]
    assert hashlib.sha256((ROOT / source["pathFromRepositoryRoot"]).read_bytes()).hexdigest() == source["sha256"]
    original = json.loads((ROOT / source["pathFromRepositoryRoot"]).read_text())
    catalog = json.loads((HERE / "catalog.json").read_text())
    tariffs = unique(catalog["tariffs"], "code")
    assert len(tariffs) == 20
    assert {t["allocationClass"] for t in tariffs.values()} == {"shared", "dedicated"}
    for tariff in tariffs.values():
        assert all(type(tariff[k]) is int and tariff[k] > 0 for k in
                   ["hourPriceMinor", "monthPriceMinor", "vCpu", "ramGb", "diskGb"])
    snapshot = data["syntheticSnapshot"]
    accounts = unique(snapshot["accounts"], "id")
    services = unique(snapshot["services"], "id")
    business = unique(data["accountBusiness"], "accountId")
    contracts = unique(data["currentContracts"], "serviceId")
    assert set(business) == set(accounts) and set(contracts) == set(services)
    assert type(data["accountCount"]) is int and data["accountCount"] > 0
    assert len(accounts) == data["accountCount"] and snapshot["transactions"] == []
    assert not set(accounts).intersection(a["id"] for a in original["accounts"])
    assert not set(services).intersection(s["id"] for s in original["services"])
    source_services = {s["id"]: s for s in original["services"]}
    assert len(data["sourceServiceTariffHypotheses"]) == len(source_services)
    for assignment in data["sourceServiceTariffHypotheses"]:
        actual = source_services[assignment["sourceServiceId"]]
        assert assignment["sourcePriceMinor"] == actual["price"]
        assert assignment["sourcePaymentTerm"] == actual["paymentTerm"]
        assert assignment["hypotheticalTariffCode"] in tariffs and assignment["overwriteSourcePrice"] is False

    events = data["events"]
    unique(events, "id")
    assert [e["occurredAt"] for e in events] == sorted(e["occurredAt"] for e in events)
    current_status, current_contract, paid_through, created, paid_totals = {}, {}, {}, set(), Counter()
    kinds = Counter()
    for event in events:
        sid, aid = event["serviceId"], event["accountId"]
        assert sid in services and aid == services[sid]["ownerId"] and aid in accounts
        assert event["source"] == "synthetic"
        assert date(business[aid]["createdAt"]) <= date(event["occurredAt"]) <= date(data["asOf"])
        assert event["recordedAt"] == event["occurredAt"]
        kind = event["type"]
        kinds[kind] += 1
        if kind == "service_created":
            assert sid not in created and event["occurredAt"] == services[sid]["createdAt"]
            created.add(sid)
        else:
            assert sid in created
        if kind == "contract_changed":
            assert event["tariffCode"] in tariffs and event["currency"] == "EUR"
            assert event["contractVersion"] == current_contract.get(sid, {}).get("contractVersion", 0) + 1
            tariff = tariffs[event["tariffCode"]]
            list_price = tariff["hourPriceMinor" if event["paymentTerm"] == "hour" else "monthPriceMinor"]
            expected_price = list_price * 90 // 100 if event["reason"] == "synthetic_discount_started" else list_price
            assert event["priceMinor"] == expected_price
            current_contract[sid] = event
        elif kind == "status_changed":
            assert event["fromStatus"] == current_status.get(sid)
            assert event["toStatus"] in {"active", "suspended", "prolong_wait", "deleted", "blocked"}
            current_status[sid] = event["toStatus"]
        elif kind in {"payment_succeeded", "payment_failed"}:
            contract = current_contract[sid]
            assert event["contractVersion"] == contract["contractVersion"]
            assert type(event["amountMinor"]) is int and event["amountMinor"] == contract["priceMinor"]
            assert event["currency"] == contract["currency"] == "EUR"
            assert event["paymentTerm"] == contract["paymentTerm"]
            assert date(event["dueAt"]) <= date(event["occurredAt"])
            assert current_status.get(sid) not in {"deleted", "blocked"}
            if kind == "payment_succeeded":
                assert date(event["paidThrough"]) == period_end(date(event["occurredAt"]), event["paymentTerm"])
                paid_through[sid] = event["paidThrough"]
                paid_totals[aid] += event["amountMinor"]
            else:
                assert event["paidThrough"] is None
        elif kind == "replacement_linked":
            old = event["replacesServiceId"]
            assert old != sid and services[old]["ownerId"] == aid and current_status[old] == "deleted"
        elif kind == "service_recovered":
            assert current_status[sid] == "active" and date(event["outageStartedAt"]) < date(event["occurredAt"])

    assert created == set(services)
    for sid, service in services.items():
        contract = current_contract[sid]
        assert service["status"] == current_status[sid]
        assert service["expiresAt"] == paid_through[sid]
        assert service["price"] == contract["priceMinor"]
        assert date(service["expiresAt"]) >= date(service["createdAt"])
        for key in ["tariffCode", "contractVersion", "priceMinor", "paymentTerm", "currency"]:
            assert contracts[sid][key] == contract[key]
    for aid, account in accounts.items():
        assert account["email"].endswith("@example.test")
        assert account["totalReplenished"] == paid_totals[aid] + account["balance"]
        assert all(type(account[k]) is int and account[k] >= 0 for k in
                   ["balance", "withdrawBalance", "totalReplenished", "bonusBalance"])
    assert all(kinds[kind] > 0 for kind in ["payment_failed", "payment_succeeded", "service_recovered",
                                          "replacement_linked", "service_terminated", "contract_changed"])
    assert any(e.get("amountMinor", 0) > 1000 for e in events), "Separate schema must support actual VPS prices"
    assert data["provenance"]["labels"] == "not_defined"
    assert all(not s["useForTraining"] for s in data["generatorScenariosDoNotTrainOn"])
    return {"accounts": len(accounts), "services": len(services), "events": len(events),
            "eventTypes": dict(sorted(kinds.items())), "statuses": dict(Counter(current_status.values())),
            "firstEventAt": events[0]["occurredAt"], "lastEventAt": events[-1]["occurredAt"]}


if __name__ == "__main__":
    loaded = json.loads((HERE / "history.json").read_text())
    summary = validate(loaded)
    assert loaded == build() == build(), "Seed reproducibility check failed"
    print(json.dumps({"validation": "passed", "reproducibility": "passed", **summary}, ensure_ascii=False, indent=2))
