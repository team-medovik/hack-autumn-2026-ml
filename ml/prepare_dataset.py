#!/usr/bin/env python3
"""Point-in-time demo features and renewal labels. No model fitting or dependencies."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY = ROOT / "examples/risk/vps-business/history.json"
DEFAULT_OUTPUT = ROOT / "ml/prepared"
DAY = timedelta(days=1)
HORIZON = timedelta(days=7)
GRACE = timedelta(days=5)
LIVE = {"active", "suspended", "prolong_wait"}
ELIGIBLE = {"active", "suspended"}
SCHEMA = "vps-preexpiry-features-v1"
TARGET = "unrenewed-within-grace-excluding-confirmed-replacement-v1"
CATEGORICAL = ("status", "payment_term", "tariff_code")
NUMERIC = (
    "days_to_expiry", "service_age_days", "account_age_days", "observed_history_days",
    "price_minor", "monthly_revenue_minor", "renewal_cycles_90d", "successful_renewals_90d",
    "failed_cycle_share_90d", "late_cycle_share_90d", "mean_paid_delay_hours_90d",
    "max_paid_delay_hours_90d", "days_since_successful_renewal", "recoveries_90d",
    "contract_changes_90d", "has_renewal_history", "history_30d_complete", "history_90d_complete",
    "live_vps_count", "portfolio_monthly_revenue_minor", "portfolio_change_30d_minor",
)
FEATURES = CATEGORICAL + NUMERIC


def timestamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timestamps must contain a timezone")
    return result.astimezone(timezone.utc)


def iso(value):
    return value.isoformat().replace("+00:00", "Z")


def days(value):
    return round(value.total_seconds() / 86400, 6)


def monthly_revenue(contract):
    if contract["currency"] != "EUR":
        raise ValueError("Only EUR contracts supported; do not silently mix currencies")
    numerator, denominator = {
        "hour": (720, 1), "week": (30, 7), "month": (1, 1),
        "quarter_year": (1, 3), "half_year": (1, 6), "year": (1, 12),
    }[contract["paymentTerm"]]
    price = contract["priceMinor"]
    if type(price) is not int or price < 0:
        raise ValueError("Contract price must be non-negative integer cents")
    return int((Decimal(price) * numerator / denominator).quantize(Decimal(1), rounding=ROUND_HALF_UP))


class History:
    def __init__(self, data):
        self.as_of = timestamp(data["asOf"])
        self.accounts = {a["accountId"]: a for a in data["accountBusiness"]}
        events = data["events"]
        if not events or len({e["id"] for e in events}) != len(events):
            raise ValueError("Events required and event IDs must be unique")
        self.events = sorted(events, key=lambda e: (timestamp(e["occurredAt"]), e["id"]))
        self.times = {e["id"]: (timestamp(e["occurredAt"]), timestamp(e["recordedAt"])) for e in events}
        self.by_service = defaultdict(list)
        self.replacements = defaultdict(list)
        for event in self.events:
            occurred, recorded = self.times[event["id"]]
            if recorded < occurred:
                raise ValueError("recordedAt cannot precede occurredAt")
            self.by_service[event["serviceId"]].append(event)
            if event["type"] == "replacement_linked":
                self.replacements[event["replacesServiceId"]].append(event)
        self.history_start = min(self.times[e["id"]][0] for e in events if e["type"] == "service_created")

    def known(self, event, at):
        occurred, recorded = self.times[event["id"]]
        return occurred <= at and recorded <= at

    def replay(self, at):
        """Reconstruct from events only. Never read final snapshot or currentContracts."""
        states = {}
        for event in self.events:
            if not self.known(event, at):
                continue
            sid, kind = event["serviceId"], event["type"]
            if kind == "service_created":
                states[sid] = {"serviceId": sid, "accountId": event["accountId"],
                               "createdAt": timestamp(event["occurredAt"]), "contract": None,
                               "status": None, "dueAt": None, "events": [], "replaced": False}
            if sid not in states:
                continue  # Delayed creation means the service is not yet observable.
            state = states[sid]
            if event["accountId"] != state["accountId"]:
                raise ValueError("Event owner changed without a supported ownership event")
            state["events"].append(event)
            if kind == "contract_changed":
                state["contract"] = event
            elif kind == "status_changed":
                state["status"] = event["toStatus"]
            elif kind == "payment_succeeded":
                state["dueAt"] = timestamp(event["paidThrough"])
            elif kind == "replacement_linked" and event["replacesServiceId"] in states:
                old = states[event["replacesServiceId"]]
                if old["accountId"] != state["accountId"]:
                    raise ValueError("Replacement belongs to a different account")
                old["replaced"] = True
        return states

    def features_at(self, at):
        states = self.replay(at)
        # Late-recorded older creations must not retrospectively extend observed coverage.
        known_history_start = min((s["createdAt"] for s in states.values()), default=at)
        past = self.replay(at - 30 * DAY)
        portfolios, past_portfolios = portfolio(states), portfolio(past)
        rows = []
        for sid, state in sorted(states.items()):
            due = state["dueAt"]
            if (state["status"] not in ELIGIBLE or state["replaced"] or not state["contract"]
                    or due is None or not at < due <= at + HORIZON):
                continue
            account = self.accounts[state["accountId"]]
            registered = timestamp(account["createdAt"])
            if registered > at:
                continue
            contract, events = state["contract"], state["events"]
            success = [e for e in events if e["type"] == "payment_succeeded"]
            # Due dates derive from previously PAID periods; retries do not create new cycles.
            due_dates = {timestamp(e["paidThrough"]) for e in success}
            completed_dues = sorted(d for d in due_dates if at - 90 * DAY <= d <= at)
            attempts = defaultdict(list)
            for event in events:
                if event["type"] in {"payment_succeeded", "payment_failed"}:
                    attempts[timestamp(event["dueAt"])].append(event)
            failed, late, delays, paid_cycles = 0, 0, [], 0
            for deadline in completed_dues:
                cycle = attempts[deadline]
                failed += any(e["type"] == "payment_failed" for e in cycle)
                paid = [e for e in cycle if e["type"] == "payment_succeeded"]
                if paid:
                    payment_time = min(timestamp(e["occurredAt"]) for e in paid)
                    delay = max(0.0, (payment_time - deadline).total_seconds() / 3600)
                    delays.append(delay)
                    paid_cycles += 1
                    late += payment_time > deadline
                else:
                    late += at > deadline
            all_renewals = [e for e in success if timestamp(e["dueAt"]) in due_dates]
            last_renewal = max((timestamp(e["occurredAt"]) for e in all_renewals), default=None)
            observed_days = max(0.0, days(at - max(known_history_start, state["createdAt"])))
            complete30 = at - 30 * DAY >= known_history_start and registered <= at - 30 * DAY
            current_portfolio = portfolios.get(state["accountId"], (0, 0))
            values = {
                "status": state["status"], "payment_term": contract["paymentTerm"],
                "tariff_code": contract["tariffCode"], "days_to_expiry": days(due - at),
                "service_age_days": days(at - state["createdAt"]), "account_age_days": days(at - registered),
                "observed_history_days": observed_days, "price_minor": contract["priceMinor"],
                "monthly_revenue_minor": monthly_revenue(contract), "renewal_cycles_90d": len(completed_dues),
                "successful_renewals_90d": paid_cycles,
                "failed_cycle_share_90d": round(failed / len(completed_dues), 6) if completed_dues else None,
                "late_cycle_share_90d": round(late / len(completed_dues), 6) if completed_dues else None,
                "mean_paid_delay_hours_90d": round(sum(delays) / len(delays), 6) if delays else None,
                "max_paid_delay_hours_90d": max(delays) if delays else None,
                "days_since_successful_renewal": days(at - last_renewal) if last_renewal else None,
                "recoveries_90d": sum(e["type"] == "service_recovered" and
                    timestamp(e["occurredAt"]) >= at - 90 * DAY for e in events),
                "contract_changes_90d": sum(e["type"] == "contract_changed" and
                    timestamp(e["occurredAt"]) > state["createdAt"] and
                    timestamp(e["occurredAt"]) >= at - 90 * DAY for e in events),
                "has_renewal_history": int(bool(all_renewals)),
                "history_30d_complete": int(complete30), "history_90d_complete": int(observed_days >= 90),
                "live_vps_count": current_portfolio[0], "portfolio_monthly_revenue_minor": current_portfolio[1],
                "portfolio_change_30d_minor": current_portfolio[1] - past_portfolios.get(state["accountId"], (0, 0))[1]
                    if complete30 else None,
            }
            known_events = [e for s in states.values() if s["accountId"] == state["accountId"] for e in s["events"]]
            rows.append({
                "sampleId": f"{sid}@{iso(at)}", "cycleId": f"{sid}@{iso(due)}",
                "serviceId": sid, "accountId": state["accountId"], "predictionAt": iso(at),
                "dueAt": iso(due), "labelDeadline": iso(due + GRACE), "currency": "EUR",
                "features": {name: values[name] for name in FEATURES},
                "audit": {"maxFeatureOccurredAt": iso(max(self.times[e["id"]][0] for e in known_events)),
                          "maxFeatureRecordedAt": iso(max(self.times[e["id"]][1] for e in known_events)),
                          "contractEventId": contract["id"]},
            })
        return rows

    def label(self, row):
        due, deadline = map(timestamp, (row["dueAt"], row["labelDeadline"]))
        if deadline > self.as_of:
            return {"value": None, "reason": "observation_incomplete", "availableAt": None, "evidenceEventIds": []}
        # An early renewal or replacement may have been recorded AFTER prediction.
        # It cannot enter features, but still proves that this cycle was not lost.
        renewals = [e for e in self.by_service[row["serviceId"]] if e["type"] == "payment_succeeded"
                    and self.known(e, self.as_of) and timestamp(e["dueAt"]) == due
                    and timestamp(e["occurredAt"]) <= deadline and timestamp(e["paidThrough"]) > due]
        replacements = [e for e in self.replacements[row["serviceId"]]
                        if self.known(e, self.as_of) and timestamp(e["occurredAt"]) <= deadline
                        and e["accountId"] == row["accountId"] and e["serviceId"] != row["serviceId"]]
        # A link alone is insufficient: the replacement VPS must exist for the same owner.
        replacement_evidence = []
        for link in replacements:
            creations = [c for c in self.by_service[link["serviceId"]]
                         if c["type"] == "service_created" and c["accountId"] == row["accountId"]
                         and self.known(c, self.as_of) and timestamp(c["occurredAt"]) <= timestamp(link["occurredAt"])]
            if creations:
                replacement_evidence.extend([link, *creations])
        evidence = renewals or replacement_evidence
        available = max([deadline] + [timestamp(e["recordedAt"]) for e in evidence])
        return {"value": 0 if evidence else 1,
                "reason": "renewed_in_grace" if renewals else "confirmed_replacement" if replacement_evidence else "no_renewal_by_grace",
                "availableAt": iso(available), "evidenceEventIds": [e["id"] for e in evidence]}


def portfolio(states):
    totals = defaultdict(lambda: [0, 0])
    for state in states.values():
        if state["status"] in LIVE and state["contract"] and not state["replaced"]:
            total = totals[state["accountId"]]
            total[0] += 1
            total[1] += monthly_revenue(state["contract"])
    return totals


def assign_split(row, validation_start, test_start):
    if row["label"]["value"] is None:
        return "pending"
    at, available = timestamp(row["predictionAt"]), timestamp(row["label"]["availableAt"])
    if at < validation_start:
        return "train" if available < validation_start else "purged"
    if at < test_start:
        return "validation" if available < test_start else "purged"
    return "test"


def build_dataset(data, validation_start, test_start):
    history = History(data)
    if not history.history_start < validation_start < test_start <= history.as_of:
        raise ValueError("Require history start < validation start < test start <= asOf")
    if any(recorded > history.as_of for _, recorded in history.times.values()):
        raise ValueError("History contains records beyond its completeness watermark asOf")
    at = history.history_start.replace(hour=15, minute=0, second=0, microsecond=0)
    if at < history.history_start:
        at += DAY
    rows = []
    while at <= history.as_of:
        for row in history.features_at(at):
            row["label"] = history.label(row)
            row["split"] = assign_split(row, validation_start, test_start)
            rows.append(row)
        at += DAY
    cycle_counts = Counter((r["split"], r["cycleId"]) for r in rows)
    for row in rows:
        row["cycleWeight"] = round(1 / cycle_counts[row["split"], row["cycleId"]], 12)
    validate_rows(rows, validation_start, test_start)
    return rows


def validate_rows(rows, validation_start, test_start):
    assert len({r["sampleId"] for r in rows}) == len(rows)
    cycle_splits = defaultdict(set)
    for row in rows:
        assert tuple(row["features"]) == FEATURES
        at, due, deadline = map(timestamp, (row["predictionAt"], row["dueAt"], row["labelDeadline"]))
        assert at < due <= at + HORIZON and deadline == due + GRACE
        assert row["features"]["status"] in ELIGIBLE
        assert timestamp(row["audit"]["maxFeatureOccurredAt"]) <= at
        assert timestamp(row["audit"]["maxFeatureRecordedAt"]) <= at
        assert row["split"] == assign_split(row, validation_start, test_start)
        if row["split"] in {"train", "validation", "test"}:
            assert row["label"]["value"] in (0, 1)
            cycle_splits[row["cycleId"]].add(row["split"])
        else:
            assert row["split"] in {"purged", "pending"}
    assert all(len(splits) == 1 for splits in cycle_splits.values()), "Same renewal outcome crossed data splits"


def review_examples(rows, data):
    events = {e["id"]: e for e in data["events"]}
    chosen = {}
    for row in rows:
        if row["split"] not in {"train", "pending"}:
            continue  # Do not browse held-out outcomes while designing the model.
        label, f = row["label"], row["features"]
        if label["reason"] == "confirmed_replacement":
            chosen.setdefault("replacement_not_loss", row)
        if label["value"] == 1 and f["has_renewal_history"]:
            chosen.setdefault("loss_despite_previous_renewals", row)
        if label["reason"] == "renewed_in_grace":
            paid = [events[eid] for eid in label["evidenceEventIds"]]
            if any(timestamp(e["occurredAt"]) > timestamp(row["dueAt"]) for e in paid):
                chosen.setdefault("late_payment_not_loss", row)
            elif f["has_renewal_history"] == 0:
                chosen.setdefault("new_service_paid_on_time", row)
        if label["value"] is None and f["payment_term"] == "hour":
            chosen.setdefault("hourly_outcome_not_observed_yet", row)
        if label["value"] == 1 and any(e["type"] == "status_changed" and e.get("toStatus") == "blocked"
                and e["serviceId"] == row["serviceId"] and row["predictionAt"] < e["occurredAt"] <= row["labelDeadline"]
                for e in data["events"]):
            chosen.setdefault("policy_block_needs_business_review", row)
    return [{"case": name, **row} for name, row in chosen.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-start", default="2026-08-01T15:00:00Z")
    parser.add_argument("--test-start", default="2026-09-01T15:00:00Z")
    args = parser.parse_args()
    raw = args.history.read_bytes()
    data = json.loads(raw)
    validation_start, test_start = timestamp(args.validation_start), timestamp(args.test_start)
    rows = build_dataset(data, validation_start, test_start)
    stats = {}
    for split in ("train", "validation", "test", "purged", "pending"):
        subset = [r for r in rows if r["split"] == split]
        cycles = {r["cycleId"]: r["label"]["value"] for r in subset}
        stats[split] = {"rows": len(subset), "cycles": len({r["cycleId"] for r in subset}),
                        "accounts": len({r["accountId"] for r in subset}),
                        "cycleLabels": dict(Counter(str(label) for label in cycles.values())),
                        "labels": dict(Counter(str(r["label"]["value"]) for r in subset)),
                        "paymentTerms": dict(Counter(r["features"]["payment_term"] for r in subset))}
    manifest = {
        "schemaVersion": SCHEMA, "targetVersion": TARGET, "historySha256": hashlib.sha256(raw).hexdigest(),
        "sourceCaseSha256": data["sourceSnapshot"]["sha256"],
        "historyAsOf": data["asOf"], "sampling": "daily 15:00 UTC; 0 < dueAt - predictionAt <= 7 days",
        "validationStart": iso(validation_start), "testStart": iso(test_start), "rows": len(rows), "splits": stats,
        "categoricalFeatures": list(CATEGORICAL), "numericFeatures": list(NUMERIC),
        "nullableNumericPolicy": "null = unknown/no prior cycles, never silently zero; convert to NaN at model adapter",
        "excluded": ["final balances", "final contracts/statuses", "autoProlong without event history",
                     "generator scenarios and reasons", "IDs and label evidence", "future observed catalog specifications"],
        "replacementPolicy": "explicit same-owner link occurring by labelDeadline, not proximity of purchases",
        "cycleWeight": "1 / number of daily rows of this cycle in this split; not a class or revenue weight",
        "limitations": ["synthetic behavior and prices; not proof of real predictive quality",
                        "catalog observed after history: no historical CPU/RAM/location join",
                        "no historical balances; not reconstructed from final snapshot",
                        "asOf assumed complete event watermark; delayed delivery beyond it needs real watermark handling",
                        "policy blocks/planned endings can be positives under current mechanical target: review before training",
                        "daily sampling can miss short hourly cycles; do not extrapolate monthly training to hourly services"],
        "trainingStarted": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "rows.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (args.output / "review-examples.json").write_text(json.dumps(review_examples(rows, data), ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "splits": stats, "trainingStarted": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
