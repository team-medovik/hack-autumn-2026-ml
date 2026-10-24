"""Four explicit billing examples; does not generate training labels or fit ML."""
import argparse
import json
from datetime import timedelta
from pathlib import Path

from billing import BillingSimulator, instant, iso

HERE = Path(__file__).resolve().parent
START = instant("2026-09-01T00:00:00Z")
DUE = instant("2026-10-01T00:00:00Z")


def examples():
    catalog = json.loads((HERE.parent / "vps-business/catalog.json").read_text())
    tariffs = {t["code"]: t for t in catalog["tariffs"]}
    result = []

    def setup():
        sim = BillingSimulator(START)
        sim.add_account(1)
        return sim

    def buy(sim, sid, code="WAWs-1", auto=True, term="month"):
        price = tariffs[code]["hourPriceMinor" if term == "hour" else "monthPriceMinor"]
        sim.schedule(f"buy-{sid}", START, "purchase", account_id=1, service_id=sid,
                     price=price, term=term, tariff_code=code, auto_prolong=auto)

    def save(sim, name, checkpoints):
        result.append({"name": name, "checkpoints": checkpoints, **sim.export()})

    sim = setup()
    sim.schedule("initial", START, "top_up", account_id=1, amount=1780)
    buy(sim, 1, "WAWs-2")
    buy(sim, 2)
    sim.schedule("next-month", DUE - timedelta(hours=1), "top_up", account_id=1, amount=600)
    sim.run(DUE)
    save(sim, "shared_wallet_no_reservation", [])

    sim = setup()
    sim.schedule("initial", START, "top_up", account_id=1, amount=593)
    buy(sim, 1)
    sim.schedule("late-topup", DUE + timedelta(hours=1), "top_up", account_id=1, amount=1000)
    sim.schedule("manual", DUE + timedelta(hours=2), "manual_renew",
                 account_id=1, service_id=1, expected_due=iso(DUE))
    checkpoints = []
    for label, at in (("failed_auto", DUE), ("topup_only", DUE + timedelta(hours=1)),
                      ("manual_paid", DUE + timedelta(hours=2))):
        checkpoints.append({"name": label, "at": iso(at), "snapshot": sim.run(at)})
    save(sim, "late_topup_requires_manual", checkpoints)

    sim = setup()
    sim.schedule("initial", START, "top_up", account_id=1, amount=1186)
    buy(sim, 1, auto=False)
    sim.schedule("manual", DUE + timedelta(minutes=30), "manual_renew",
                 account_id=1, service_id=1, expected_due=iso(DUE))
    checkpoints = [{"name": "auto_off_money_untouched", "at": iso(DUE), "snapshot": sim.run(DUE)}]
    sim.run(DUE + timedelta(minutes=30))
    save(sim, "auto_off_then_manual", checkpoints)

    sim = setup()
    sim.schedule("initial", START, "top_up", account_id=1, amount=8)
    buy(sim, 1, term="hour")
    sim.schedule("late-topup", START + timedelta(hours=3), "top_up", account_id=1, amount=8)
    sim.schedule("manual", START + timedelta(hours=3, minutes=30), "manual_renew",
                 account_id=1, service_id=1, expected_due=iso(START + timedelta(hours=2)))
    checkpoints = [{"name": "hourly_topup_no_retry", "at": iso(START + timedelta(hours=3)),
                    "snapshot": sim.run(START + timedelta(hours=3))}]
    sim.run(START + timedelta(hours=3, minutes=30))
    save(sim, "hourly_exhaustion_then_manual", checkpoints)
    return {"schemaVersion": "vps-billing-demonstrations-v1", "catalogObservedAt": catalog["observedAt"],
            "historicalPrices": "demonstration using saved catalogue, not historical Aeza quotes",
            "trainingDataset": False, "examples": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional NEW file; existing files are never overwritten")
    args = parser.parse_args()
    data = examples()
    if args.output:
        with args.output.open("x") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    print(json.dumps([{ "name": e["name"], "snapshot": e["snapshot"],
                        "checkpoints": e["checkpoints"], "eventCount": len(e["events"])}
                      for e in data["examples"]], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
