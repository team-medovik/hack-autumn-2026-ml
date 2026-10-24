#!/usr/bin/env python3
"""Cycle-aware pre-expiry snapshots for hour/day. Same five-day target, no training."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

from prepare_dataset import (CATEGORICAL, NUMERIC, ROOT, SCHEMA, TARGET, History,
                             assign_split, iso, timestamp, validate_rows)

LEAD = {"hour": timedelta(minutes=30), "day": timedelta(hours=12)}
SAMPLING = "short-period-each-renewal-v1"


def build_short_dataset(data, validation_start, test_start):
    history = History(data)
    if not history.history_start < validation_start < test_start <= history.as_of:
        raise ValueError("Require history start < validation start < test start <= asOf")
    if any(recorded > history.as_of for _, recorded in history.times.values()):
        raise ValueError("History contains records after its completeness watermark")
    scheduled = defaultdict(dict)
    for event in history.events:
        if event["type"] != "payment_succeeded" or event.get("paymentTerm") not in LEAD:
            continue
        # A payment makes the NEXT paid-through expiry observable; never sample the
        # initial purchase itself or learn a next expiry from a future payment.
        due = timestamp(event["paidThrough"])
        at = due - LEAD[event["paymentTerm"]]
        if at <= history.as_of and history.known(event, at):
            scheduled[at][event["serviceId"]] = due
    rows = []
    for at, targets in sorted(scheduled.items()):
        for row in history.features_at(at, service_ids=set(targets)):
            term = row["features"]["payment_term"]
            if (term not in LEAD or timestamp(row["dueAt"]) != targets[row["serviceId"]]
                    or timestamp(row["dueAt"]) - at != LEAD[term]):
                continue  # The contract/expiry visible at prediction may have changed.
            row["label"] = history.label(row)
            row["split"] = assign_split(row, validation_start, test_start)
            row["cycleWeight"] = 1.0
            rows.append(row)
    validate_rows(rows, validation_start, test_start)
    if len({r["cycleId"] for r in rows}) != len(rows):
        raise AssertionError("A renewal cycle was sampled more than once")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=ROOT / "examples/risk/short-periods/history.json")
    parser.add_argument("--output", type=Path, default=ROOT / "ml/prepared-short")
    parser.add_argument("--validation-start", default="2026-08-01T15:00:00Z")
    parser.add_argument("--test-start", default="2026-09-01T15:00:00Z")
    args = parser.parse_args()
    raw = args.history.read_bytes()
    data = json.loads(raw)
    rows = build_short_dataset(data, timestamp(args.validation_start), timestamp(args.test_start))
    stats = {}
    for split in ("train", "validation", "test", "purged", "pending", "excluded"):
        stats[split] = {}
        for term in LEAD:
            subset = [r for r in rows if r["split"] == split and r["features"]["payment_term"] == term]
            stats[split][term] = {"rows": len(subset), "services": len({r["serviceId"] for r in subset}),
                                  "labels": dict(Counter(str(r["label"]["value"]) for r in subset))}
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "rows.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    manifest = {
        "schemaVersion": SCHEMA, "targetVersion": TARGET, "samplingVersion": SAMPLING,
        "historySha256": hashlib.sha256(raw).hexdigest(),
        "rowsSha256": hashlib.sha256((args.output / "rows.jsonl").read_bytes()).hexdigest(),
        "generatorVersion": data.get("generatorVersion"), "historyAsOf": data["asOf"],
        "validationStart": args.validation_start, "testStart": args.test_start,
        "categoricalFeatures": list(CATEGORICAL), "numericFeatures": list(NUMERIC),
        "sampling": "every observed upcoming renewal, hour: 30 min before; day: 12 h before",
        "cycleWeight": "one per observed cycle; NOT period balancing; do not blindly mix hour/day/month",
        "splits": stats, "rows": len(rows), "trainingStarted": False,
        "limitations": ["Synthetic behavior; daily price synthetic, not a verified Aeza daily offer",
                        "Five-day target unchanged for both terms; not a reproduction of provider deletion policy",
                        "Renewal opportunities observed from known successful payments; censored futures remain pending",
                        "Hour cycles are far more numerous; metrics and baselines must be separate per payment term",
                        "Previous monthly models not validated for hour/day; no model trained by this preparation"],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "splits": stats, "trainingStarted": False}, indent=2))


if __name__ == "__main__":
    main()
