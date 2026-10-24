#!/usr/bin/env python3
"""Prepare point-in-time renewal examples from the v2 event ledger; never fit a model.

This adapter deliberately does not import the generator or read its final snapshot,
diagnostics, profiles or cycle reports. EUR amounts are integer minor units.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from prepare_dataset import (CATEGORICAL, FEATURES as LEGACY_FEATURES, GRACE, HORIZON,
                             DAY, days, iso, monthly_revenue, timestamp, assign_split)

SCHEMA = "event-driven-preexpiry-features-v1"
TARGET = "event-renewal-5d-paid-replacement-nonfinancial-excluded-v1"
ENHANCED_FEATURES = (
    "auto_prolong", "wallet_balance_minor", "current_service_funding_ratio",
    "next_renewals_7d_required_minor", "portfolio_funding_ratio_7d",
    "topup_count_30d", "topup_sum_30d_minor", "days_since_topup",
    "payment_sum_30d_minor", "days_since_auto_change",
    "service_projected_funding_ratio",
    "nonfinancial_restrictions_90d",
)
FEATURES = LEGACY_FEATURES + ENHANCED_FEATURES
ELIGIBLE = {"active", "suspended"}
LIVE = ELIGIBLE | {"prolong_wait"}
DEFAULT_VALIDATION = "2027-07-01T00:00:00Z"
DEFAULT_TEST = "2028-04-01T00:00:00Z"
EVENT_TYPES = {"account_created", "wallet_topped_up", "service_created",
               "payment_succeeded", "payment_failed", "auto_prolong_changed",
               "status_changed", "replacement_confirmed", "service_restricted",
               "service_released", "renewal_paused", "renewal_due", "action_rejected"}


def at_time(value):
    return timestamp(value) if isinstance(value, str) else timestamp(value.isoformat())


class EventHistory:
    def __init__(self, data):
        if data.get("schemaVersion") != "vps-billing-events-v2":
            raise ValueError("Expected schemaVersion vps-billing-events-v2")
        self.as_of = timestamp(data["asOf"])
        events = data["events"]
        if not events or len({e["id"] for e in events}) != len(events):
            raise ValueError("Events required, with unique IDs")
        self.times = {}
        self.order = {e["id"]: i for i, e in enumerate(events)}
        for e in events:
            for field in ("id", "type", "occurredAt", "recordedAt", "accountId", "serviceId", "currency"):
                if field not in e:
                    raise ValueError(f"Missing event field {field}")
            occurred, recorded = timestamp(e["occurredAt"]), timestamp(e["recordedAt"])
            if recorded < occurred:
                raise ValueError("Require occurredAt <= recordedAt")
            if e["currency"] != "EUR" or e["type"] not in EVENT_TYPES:
                raise ValueError("Unsupported currency or event type")
            if e["type"] == "service_created":
                for field in ("priceMinor", "paymentTerm", "tariffCode", "autoProlong"):
                    if field not in e:
                        raise ValueError(f"Missing service field {field}")
                monthly_revenue(e)
                if e["paymentTerm"] not in {"hour", "day", "month"}:
                    raise ValueError("v2 sampling supports hour/day/month only")
            if e["type"] in {"wallet_topped_up", "payment_succeeded"}:
                if type(e.get("amountMinor")) is not int or e["amountMinor"] < 0:
                    raise ValueError("Amounts must be nonnegative integer cents")
            self.times[e["id"]] = occurred, recorded
        # Stable sort preserves the authoritative input order at equal timestamps.
        self.events = sorted(events, key=lambda e: self.times[e["id"]][0])
        self.by_service, self.by_account = defaultdict(list), defaultdict(list)
        for e in self.events:
            self.by_service[e["serviceId"]].append(e)
            self.by_account[e["accountId"]].append(e)
        starts = [self.times[e["id"]][0] for e in events if e["type"] == "account_created"]
        if not starts:
            raise ValueError("account_created events required")
        self.history_start = min(starts)
        self._replacement_cache = None

    def known(self, event, at):
        occurred, recorded = self.times[event["id"]]
        return occurred <= at and recorded <= at

    def _replay(self, at):
        accounts, states, replacements = {}, {}, []
        for e in self.events:
            if not self.known(e, at):
                continue
            kind, aid, sid = e["type"], e["accountId"], e["serviceId"]
            if kind == "account_created":
                accounts[aid] = {"createdAt": self.times[e["id"]][0], "balance": 0, "events": []}
            if aid not in accounts:
                continue
            account = accounts[aid]
            account["events"].append(e)
            if kind == "wallet_topped_up":
                account["balance"] += e["amountMinor"]
            elif kind == "service_created":
                states[sid] = {"serviceId": sid, "accountId": aid, "createdAt": self.times[e["id"]][0],
                               "contract": e, "status": "active", "dueAt": None, "events": [],
                               "autoProlong": e["autoProlong"], "replaced": False, "replaces": None, "purchase": None}
            elif kind == "payment_succeeded":
                account["balance"] -= e["amountMinor"]
            if sid not in states:
                continue
            state = states[sid]
            if state["accountId"] != aid:
                raise ValueError("Service ownership changed without a supported event")
            state["events"].append(e)
            if kind == "payment_succeeded":
                state["dueAt"], state["status"] = timestamp(e["paidThrough"]), "active"
                if e["mode"] == "purchase" and e["amountMinor"] > 0:
                    state["purchase"] = e
            elif kind in {"status_changed", "service_restricted", "service_released"}:
                state["status"] = e["toStatus"]
            elif kind == "auto_prolong_changed":
                state["autoProlong"] = e["enabled"]
            elif kind == "replacement_confirmed":
                new = states.get(e.get("replacementServiceId"))
                valid = (new and new["accountId"] == aid and new["serviceId"] != sid
                         and state["status"] not in {"blocked", "deleted"} and not state["replaced"]
                         and new["replaces"] is None
                         and new["createdAt"] > state["createdAt"] and new["purchase"]
                         and new["status"] == "active" and new["dueAt"] > self.times[e["id"]][0]
                         and state["dueAt"] == timestamp(e["dueAt"]))
                if valid:
                    state["replaced"] = True
                    new["replaces"] = sid
                    paid = [p for p in new["events"] if p["type"] == "payment_succeeded"]
                    old_paid = [p for p in state["events"] if p["type"] == "payment_succeeded"]
                    status_kinds = {"service_created", "payment_succeeded", "status_changed", "service_restricted", "service_released"}
                    status_proof = [p for s in (state, new) for p in s["events"] if p["type"] in status_kinds]
                    replacements.append((e, [e, state["contract"], old_paid[-1], new["contract"], new["purchase"], paid[-1], *status_proof]))
        return accounts, states, replacements

    def replay(self, at):
        """Return independently reconstructed services; account ledger is never a snapshot join."""
        return self._replay(at_time(at))[1]

    @staticmethod
    def _portfolios(states):
        portfolios = defaultdict(lambda: [0, 0])
        for s in states.values():
            if s["status"] in LIVE and not s["replaced"]:
                p = portfolios[s["accountId"]]
                p[0] += 1
                p[1] += monthly_revenue(s["contract"])
        return portfolios

    def features_at(self, at, service_ids=None):
        at = at_time(at)
        if at > self.as_of:
            raise ValueError("Prediction exceeds event completeness watermark")
        accounts, states, _ = self._replay(at)
        past = self._portfolios(self.replay(at - 30 * DAY))
        portfolios = self._portfolios(states)
        known_start = min((s["createdAt"] for s in states.values()), default=at)
        rows = []
        for sid, state in sorted(states.items()):
            if service_ids is not None and sid not in service_ids:
                continue
            due = state["dueAt"]
            if (state["status"] not in ELIGIBLE or state["replaced"] or due is None
                    or not at < due <= at + HORIZON):
                continue
            aid, contract, events = state["accountId"], state["contract"], state["events"]
            account = accounts[aid]
            success = [e for e in events if e["type"] == "payment_succeeded"]
            due_dates = {timestamp(e["paidThrough"]) for e in success}
            completed = sorted(d for d in due_dates if at - 90 * DAY <= d <= at)
            attempts = defaultdict(list)
            for e in events:
                if e["type"] in {"payment_succeeded", "payment_failed"} and e.get("mode") != "purchase":
                    attempts[timestamp(e["dueAt"])].append(e)
            failed, late, paid_cycles, delays = 0, 0, 0, []
            for deadline in completed:
                cycle = attempts[deadline]
                failed += any(e["type"] == "payment_failed" for e in cycle)
                paid = [e for e in cycle if e["type"] == "payment_succeeded"]
                if paid:
                    paid_at = min(self.times[e["id"]][0] for e in paid)
                    delay = max(0.0, (paid_at - deadline).total_seconds() / 3600)
                    delays.append(delay)
                    paid_cycles += 1
                    late += paid_at > deadline
                else:
                    late += at > deadline
            renewals = [e for e in success if e.get("mode") != "purchase" and timestamp(e["dueAt"]) in due_dates]
            last = max((self.times[e["id"]][0] for e in renewals), default=None)
            observed = max(0.0, days(at - max(known_start, state["createdAt"])))
            complete30 = at - 30 * DAY >= known_start and account["createdAt"] <= at - 30 * DAY
            p = portfolios[aid]
            # Recoveries are derived from paid late renewal, not a synthetic reason or missing event type.
            recoveries = sum(self.times[e["id"]][0] >= at - 90 * DAY
                             and self.times[e["id"]][0] > timestamp(e["dueAt"]) for e in renewals)
            legacy = {
                "status": state["status"], "payment_term": contract["paymentTerm"], "tariff_code": contract["tariffCode"],
                "days_to_expiry": days(due - at), "service_age_days": days(at - state["createdAt"]),
                "account_age_days": days(at - account["createdAt"]), "observed_history_days": observed,
                "price_minor": contract["priceMinor"], "monthly_revenue_minor": monthly_revenue(contract),
                "renewal_cycles_90d": len(completed), "successful_renewals_90d": paid_cycles,
                "failed_cycle_share_90d": round(failed / len(completed), 6) if completed else None,
                "late_cycle_share_90d": round(late / len(completed), 6) if completed else None,
                "mean_paid_delay_hours_90d": round(sum(delays) / len(delays), 6) if delays else None,
                "max_paid_delay_hours_90d": max(delays) if delays else None,
                "days_since_successful_renewal": days(at - last) if last else None,
                "recoveries_90d": recoveries, "contract_changes_90d": 0,
                "has_renewal_history": int(bool(renewals)), "history_30d_complete": int(complete30),
                "history_90d_complete": int(observed >= 90), "live_vps_count": p[0],
                "portfolio_monthly_revenue_minor": p[1],
                "portfolio_change_30d_minor": p[1] - past.get(aid, (0, 0))[1] if complete30 else None,
            }
            account_events = account["events"]
            topups = [e for e in account_events if e["type"] == "wallet_topped_up"]
            recent_topups = [e for e in topups if self.times[e["id"]][0] >= at - 30 * DAY]
            payments = [e for e in account_events if e["type"] == "payment_succeeded"
                        and self.times[e["id"]][0] >= at - 30 * DAY]
            changes = [e for e in events if e["type"] == "auto_prolong_changed"]
            # One next renewal per currently live service, not a promise of repeated hourly charges.
            required = sum(s["contract"]["priceMinor"] for s in states.values()
                           if s["accountId"] == aid and s["status"] in LIVE and not s["replaced"]
                           and s["dueAt"] is not None and s["dueAt"] <= at + HORIZON)
            balance = account["balance"] if account["balance"] >= 0 else None
            remaining, projected = balance, None
            obligations = sorted((s for s in states.values() if s["accountId"] == aid
                                  and s["status"] in LIVE and not s["replaced"]
                                  and s["dueAt"] is not None and s["dueAt"] <= at + HORIZON),
                                 key=lambda s: (s["dueAt"], s["serviceId"]))
            for obligation in obligations:
                price = obligation["contract"]["priceMinor"]
                if obligation["serviceId"] == sid:
                    projected = round(remaining / price, 6) if remaining is not None and price else None
                    break
                if remaining is not None and remaining >= price:
                    remaining -= price  # Insufficient earlier charge reserves nothing.
            enhanced = {
                "auto_prolong": int(state["autoProlong"]), "wallet_balance_minor": balance,
                "current_service_funding_ratio": round(balance / contract["priceMinor"], 6)
                    if balance is not None and contract["priceMinor"] else None,
                "next_renewals_7d_required_minor": required,
                "portfolio_funding_ratio_7d": round(balance / required, 6) if balance is not None and required else None,
                "topup_count_30d": len(recent_topups), "topup_sum_30d_minor": sum(e["amountMinor"] for e in recent_topups),
                "days_since_topup": days(at - self.times[topups[-1]["id"]][0]) if topups else None,
                "payment_sum_30d_minor": sum(e["amountMinor"] for e in payments),
                "days_since_auto_change": days(at - self.times[changes[-1]["id"]][0]) if changes else None,
                "service_projected_funding_ratio": projected,
                "nonfinancial_restrictions_90d": sum(e["type"] == "service_restricted"
                    and e.get("category") == "nonfinancial" and self.times[e["id"]][0] >= at - 90 * DAY for e in events),
            }
            legacy = {name: legacy[name] for name in LEGACY_FEATURES}
            rows.append({
                "sampleId": f"{sid}@{iso(at)}", "cycleId": f"{sid}@{iso(due)}", "serviceId": sid,
                "accountId": aid, "predictionAt": iso(at), "dueAt": iso(due), "labelDeadline": iso(due + GRACE),
                "currency": "EUR", "features": {**legacy, **enhanced}, "legacy_features": legacy,
                "enhanced_features": enhanced,
                "audit": {"maxFeatureOccurredAt": iso(max(self.times[e["id"]][0] for e in account_events)),
                          "maxFeatureRecordedAt": iso(max(self.times[e["id"]][1] for e in account_events)),
                          "contractEventId": contract["id"]},
            })
        return rows

    def label(self, row):
        prediction, due, deadline = map(timestamp, (row["predictionAt"], row["dueAt"], row["labelDeadline"]))
        if deadline != due + GRACE:
            raise ValueError("Label deadline must equal dueAt + five days")
        if deadline > self.as_of:
            return {"value": None, "reason": "observation_incomplete", "availableAt": None, "evidenceEventIds": []}
        sid, aid = row["serviceId"], row["accountId"]
        events = [e for e in self.by_service[sid] if e["accountId"] == aid and self.known(e, self.as_of)]
        opened, intervals = None, []
        for e in events:
            if e["type"] == "service_restricted" and e.get("category") == "nonfinancial" and e.get("toStatus") == "blocked":
                if opened is None:
                    opened = e
            elif e["type"] == "service_released" and e.get("category") == "nonfinancial" and e.get("toStatus") in {"active", "prolong_wait"}:
                if opened is not None:
                    intervals.append((opened, e))
                    opened = None
        if opened is not None:
            intervals.append((opened, None))
        blocked, resolution_proof = [], []
        for start, finish in intervals:
            begin = self.times[start["id"]][0]
            end = self.times[finish["id"]][0] if finish else None
            if begin <= deadline and (end is None or end > prediction) and (end is None or end > begin):
                blocked.extend([start] + ([finish] if finish else []))
            elif finish is not None and begin <= prediction and end <= prediction:
                # Late-recorded release can reverse an apparent active restriction.
                # The final eligible label is unavailable until that proof arrives.
                resolution_proof.extend([start, finish])
        if blocked:
            evidence, reason, value = blocked, "excluded_nonfinancial_block", None
        else:
            renewals = [e for e in events if e["type"] == "payment_succeeded" and e.get("mode") != "purchase"
                        and timestamp(e["dueAt"]) == due and self.times[e["id"]][0] <= deadline
                        and timestamp(e["paidThrough"]) > due and e["amountMinor"] > 0]
            replacement_evidence = []
            if self._replacement_cache is None:
                self._replacement_cache = self._replay(self.as_of)[2]
            for link, proof in self._replacement_cache:
                if (link["serviceId"] == sid and link["accountId"] == aid and timestamp(link["dueAt"]) == due
                        and self.times[link["id"]][0] <= deadline):
                    replacement_evidence.extend(proof)
            evidence = renewals or replacement_evidence
            reason = "renewed_in_grace" if renewals else "confirmed_replacement" if evidence else "no_renewal_by_grace"
            value = 0 if evidence else 1
            evidence = [*evidence, *resolution_proof]
        unique = {e["id"]: e for e in evidence}
        return {"value": value, "reason": reason,
                "availableAt": iso(max([deadline] + [self.times[eid][1] for eid in unique])),
                "evidenceEventIds": list(unique)}


def build_dataset(data, validation_start=DEFAULT_VALIDATION, test_start=DEFAULT_TEST):
    history = EventHistory(data)
    if any(recorded > history.as_of for _, recorded in history.times.values()):
        raise ValueError("Dataset export contains records beyond asOf completeness watermark")
    validation_start, test_start = at_time(validation_start), at_time(test_start)
    if not history.history_start < validation_start < test_start <= history.as_of:
        raise ValueError("Require history start < validation start < test start <= asOf")
    # Schedule from paid-through dates observable before the requested cutoff,
    # never from final service state. Recheck due/status at the cutoff afterward.
    scheduled = defaultdict(set)
    for e in history.events:
        if e["type"] != "payment_succeeded" or e.get("paymentTerm") not in {"hour", "day"}:
            continue
        offset = timedelta(minutes=30) if e["paymentTerm"] == "hour" else timedelta(hours=12)
        at = timestamp(e["paidThrough"]) - offset
        if at <= history.as_of and history.known(e, at):
            scheduled[at].add(e["serviceId"])
    at = history.history_start.replace(hour=15, minute=0, second=0, microsecond=0)
    if at < history.history_start:
        at += DAY
    while at <= history.as_of:
        scheduled[at]  # Monthly daily grid; no due inferred from final snapshots.
        at += DAY
    rows = []
    for at in sorted(scheduled):
        daily = at.hour == 15 and at.minute == at.second == at.microsecond == 0
        for row in history.features_at(at):
            term = row["features"]["payment_term"]
            if term == "month":
                if not daily:
                    continue
            else:
                offset = timedelta(minutes=30) if term == "hour" else timedelta(hours=12)
                if row["serviceId"] not in scheduled[at] or timestamp(row["dueAt"]) - at != offset:
                    continue
            row["label"] = history.label(row)
            row["split"] = assign_split(row, validation_start, test_start)
            rows.append(row)
    counts = Counter((r["split"], r["cycleId"]) for r in rows)
    for row in rows:
        row["cycleWeight"] = 1 / counts[row["split"], row["cycleId"]]
    validate_rows(rows, validation_start, test_start)
    return rows


def validate_rows(rows, validation_start=DEFAULT_VALIDATION, test_start=DEFAULT_TEST):
    validation_start, test_start = at_time(validation_start), at_time(test_start)
    if len({r["sampleId"] for r in rows}) != len(rows):
        raise ValueError("Duplicate prediction sample")
    cycle_splits = defaultdict(set)
    for row in rows:
        at, due, deadline = map(timestamp, (row["predictionAt"], row["dueAt"], row["labelDeadline"]))
        if not at < due <= at + HORIZON or deadline != due + GRACE:
            raise ValueError("Invalid prediction or outcome window")
        if tuple(row["legacy_features"]) != LEGACY_FEATURES or tuple(row["features"]) != FEATURES:
            raise ValueError("Feature allowlist mismatch")
        if any(timestamp(row["audit"][name]) > at for name in ("maxFeatureOccurredAt", "maxFeatureRecordedAt")):
            raise ValueError("Future feature evidence")
        if row["split"] != assign_split(row, validation_start, test_start):
            raise ValueError("Incorrect temporal split")
        if row["split"] in {"train", "validation", "test"}:
            cycle_splits[row["cycleId"]].add(row["split"])
    if any(len(splits) > 1 for splits in cycle_splits.values()):
        raise ValueError("One renewal outcome leaked across temporal partitions")


def manifest(data, rows, source_hash, validation_start=DEFAULT_VALIDATION, test_start=DEFAULT_TEST):
    # These describe the source, never enter point-in-time predictive features.
    assumptions = data.get("assumptions", {})
    monthly_policy = assumptions.get("monthlyDeletionPolicy")
    registrations = [timestamp(e["occurredAt"]) for e in data["events"]
                     if e["type"] == "account_created"
                     and timestamp(e["occurredAt"]) <= timestamp(data["asOf"])
                     and timestamp(e["recordedAt"]) <= timestamp(data["asOf"])]
    registration_dates = {value.date().isoformat() for value in registrations}
    source_limitations = []
    if not monthly_policy:
        source_limitations.append("Monthly deletion policy absent from source metadata: long-overdue unreplaced services may remain in portfolio/funding stress; resolve observable termination policy before training")
    else:
        source_limitations.append("Monthly deletion policy is declared by the source, not inferred from labels; source policy metadata is audit-only, not a predictive input")
        for key in ("monthlyDeletionBoundary", "monthlyRestrictionIntersection", "otherDeletionPolicies"):
            source_limitations.append(assumptions.get(key) or f"Source does not declare {key}; do not assume an Aeza billing rule")
    if len(registration_dates) <= 1:
        source_limitations.append("Observed accounts share one registration date (or none is observed); account age is not representative of varied real customer cohorts")
    else:
        source_limitations.append("Registration dates vary in the observed ledger; synthetic cohort timing does not establish a representative real customer age distribution")
    stats = {}
    for split in ("train", "validation", "test", "purged", "pending", "excluded"):
        selected = [r for r in rows if r["split"] == split]
        cycles = {r["cycleId"]: r for r in selected}
        stats[split] = {"rows": len(selected), "cycles": len(cycles),
                        "accounts": len({r["accountId"] for r in selected}),
                        "labels": dict(Counter(str(r["label"]["value"]) for r in selected)),
                        "cycleLabels": dict(Counter(str(r["label"]["value"]) for r in cycles.values())),
                        "paymentTerms": dict(Counter(r["features"]["payment_term"] for r in selected)),
                        "cyclePaymentTerms": dict(Counter(r["features"]["payment_term"] for r in cycles.values()))}
    return {
        "schemaVersion": SCHEMA, "targetVersion": TARGET, "sourceSchemaVersion": data["schemaVersion"],
        "historySha256": source_hash, "historyAsOf": data["asOf"], "trainingStarted": False,
        "sourceMonthlyDeletionPolicy": monthly_policy,
        "sourceRegistrationSummary": {
            "accounts": len(registrations), "distinctDates": len(registration_dates),
            "firstRegisteredAt": iso(min(registrations)) if registrations else None,
            "lastRegisteredAt": iso(max(registrations)) if registrations else None,
        },
        "rows": len(rows), "splits": stats, "validationStart": iso(at_time(validation_start)),
        "testStart": iso(at_time(test_start)), "legacyFeatureNames": list(LEGACY_FEATURES),
        "enhancedFeatureNames": list(ENHANCED_FEATURES), "featureNames": list(FEATURES),
        "categoricalFeatures": list(CATEGORICAL),
        "sampling": {"month": "daily 15:00 UTC, 0 < due - T <= 7d", "hour": "30 minutes before observed due",
                     "day": "12 hours before observed due"},
        "target": "No successful renewal by due+5d inclusive; paid same-owner current-cycle confirmed replacement is not loss",
        "restrictionPolicy": "Exclude nonfinancial restriction overlapping [predictionAt, due+5d]; restriction end exclusive",
        "followupPolicy": "Require full due+5d observation even when early renewal is already seen",
        "splitPolicy": "Chronological prediction cutoffs; purge labels whose availableAt reaches next cutoff; same clients may recur",
        "cycleWeight": "1 / number of samples for this cycle in this partition; no revenue/class weighting",
        "financialPolicy": "Cash ledger only. 7d required includes ONE next charge per live unreplaced service incl past due; no repeat hourly forecast. Projected service funding processes those charges by (due, serviceId), skips unaffordable without reserving, assumes no future topups/auto changes/manual actions, regardless auto flag. This is a static funding stress scenario, not actual future payment. Ratios uncapped; zero denominator or negative incomplete wallet = null.",
        "legacyCompatibility": "Exact 24 names/types; v2 late-payment recoveries derived from events, immutable contracts imply contract_changes_90d=0. New ledger semantics cause domain shift, not a claim of numerical equivalence.",
        "featureExclusions": ["snapshot", "diagnostics", "assumptions", "report", "archetype", "future needs", "external budget", "IDs", "reasons", "label evidence"],
        "limitations": ["Synthetic data cannot establish predictive quality on real customers",
                        "Temporal test is exploratory: generator audit already inspected this dataset, not pristine untouched test",
                        "No unseen-client generalization claim; clients recur across temporal partitions",
                        "Do not pool hour/day/month evaluation silently; report metrics per payment term",
                        "asOf is assumed a complete event watermark; production requires authoritative completeness guarantees",
                        "Daily tariffs and restrictions billing pause are explicit simulation assumptions, not verified Aeza policy",
                        "Stable age/history features have no arbitrary bonus; magnitude must later be tested in model ablation",
                        "Legacy late-payment features include past nonfinancial pauses; enhanced restriction count records this context, not a causal financial interpretation",
                        "Names/source/reasons/catalog specifications are not joined as predictive inputs",
                        *source_limitations],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-start", default=DEFAULT_VALIDATION)
    parser.add_argument("--test-start", default=DEFAULT_TEST)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be a NEW directory; refusing to overwrite existing artifacts")
    raw = args.history.read_bytes()
    data = json.loads(raw)
    rows = build_dataset(data, args.validation_start, args.test_start)
    info = manifest(data, rows, hashlib.sha256(raw).hexdigest(), args.validation_start, args.test_start)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "rows.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    info["rowsSha256"] = hashlib.sha256((args.output / "rows.jsonl").read_bytes()).hexdigest()
    with (args.output / "manifest.json").open("x") as stream:
        stream.write(json.dumps(info, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"rows": len(rows), "splits": info["splits"], "trainingStarted": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
