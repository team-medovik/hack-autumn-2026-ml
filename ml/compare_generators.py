#!/usr/bin/env python3
"""Compare fixed old/new models on the SAME new validation set; never evaluate test."""
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from prepare_dataset import FEATURES, timestamp
from train_baseline import make_pool, metrics, select_rows


def load_run(directory):
    from catboost import CatBoostClassifier
    report = json.loads((directory / "report.json").read_text())
    path = directory / "model.cbm"
    if hashlib.sha256(path.read_bytes()).hexdigest() != report["modelSha256"]:
        raise ValueError("Model does not match its training report")
    model = CatBoostClassifier()
    model.load_model(str(path))
    if model.feature_names_ != list(FEATURES):
        raise ValueError("Model features mismatch")
    return model, report


def representative_cycles(rows):
    """One snapshot per validation cycle, closest to 3 days before expiry; no label use."""
    cycles = {}
    for row in rows:
        previous = cycles.get(row["cycleId"])
        key = lambda r: (abs(r["features"]["days_to_expiry"] - 3), r["predictionAt"])
        if previous is None or key(row) < key(previous):
            cycles[row["cycleId"]] = row
    return list(cycles.values())


def explain_examples(model, rows, history):
    rows = representative_cycles(rows)
    pool = make_pool(rows)
    probabilities = model.predict_proba(pool)[:, 1]
    shap = model.get_feature_importance(pool, type="ShapValues")
    raw = model.predict(pool, prediction_type="RawFormulaVal")
    if any(abs(sum(v) - r) > 1e-9 for v, r in zip(shap, raw)):
        raise AssertionError("SHAP contributions do not reconstruct raw predictions")
    if any(abs(1 / (1 + math.exp(-float(r))) - p) > 1e-10 for r, p in zip(raw, probabilities)):
        raise AssertionError("Raw predictions do not reconstruct probabilities")
    late_cycles = {(e["serviceId"], e["dueAt"]) for e in history["events"]
                   if e["type"] == "payment_succeeded"
                   and (timestamp(e["occurredAt"]) - timestamp(e["dueAt"])).total_seconds() > 5 * 86400}
    service_events = defaultdict(list)
    for event in history["events"]:
        service_events[event["serviceId"]].append(event)

    def long_stable(row):
        past = [e for e in service_events[row["serviceId"]]
                if timestamp(e["occurredAt"]) <= timestamp(row["predictionAt"])]
        paid = [e for e in past if e["type"] == "payment_succeeded"
                and e.get("paymentTerm") == "month" and e.get("reason") != "initial_purchase"]
        return len(paid) >= 12 and not any(e["type"] == "payment_failed" for e in past)
    selectors = [
        ("higher_risk_actual_loss", lambda r: r["label"]["value"] == 1, True),
        ("lower_risk_actual_renewal", lambda r: r["label"]["value"] == 0, False),
        ("higher_risk_but_renewed", lambda r: r["label"]["value"] == 0, True),
        ("lower_risk_but_lost", lambda r: r["label"]["value"] == 1, False),
        ("first_renewal_loss", lambda r: r["label"]["value"] == 1 and not r["features"]["has_renewal_history"], True),
        ("loss_after_previous_recovery", lambda r: r["label"]["value"] == 1 and r["features"]["recoveries_90d"] > 0, False),
        ("paid_after_grace_not_permanent_loss", lambda r: r["label"]["value"] == 1 and (r["serviceId"], r["dueAt"]) in late_cycles, True),
        ("long_stable_renewed", lambda r: r["label"]["value"] == 0 and long_stable(r), False),
    ]
    examples = []
    for case, predicate, high in selectors:
        candidates = [i for i, row in enumerate(rows) if predicate(row)]
        if not candidates:
            continue
        i = (max if high else min)(candidates, key=lambda j: probabilities[j])
        row = rows[i]
        # Events are explanation-only evidence, NEVER inputs to the model or SHAP pool.
        outcome_events = [e for e in history["events"] if e["serviceId"] == row["serviceId"]
                          and timestamp(row["predictionAt"]) <= timestamp(e["occurredAt"]) <= timestamp(row["labelDeadline"])]
        later_payments = [e for e in history["events"] if e["serviceId"] == row["serviceId"]
                          and e["type"] == "payment_succeeded" and e.get("dueAt") == row["dueAt"]
                          and timestamp(e["occurredAt"]) > timestamp(row["labelDeadline"])]
        contributions = sorted([{"feature": name, "value": row["features"][name], "rawLogOdds": float(v)}
                                for name, v in zip(FEATURES, shap[i][:-1])],
                               key=lambda item: abs(item["rawLogOdds"]), reverse=True)
        examples.append({"case": case, "risk": float(probabilities[i]), "row": row,
                         "baseRawLogOdds": float(shap[i][-1]), "contributions": contributions,
                         "eventsWithinOutcomeWindow": outcome_events, "paymentAfterOutcomeWindow": later_payments})
    return {"sampling": "one snapshot per validation cycle closest to 3 days before expiry",
            "selection": "contrasting outcome-aware examples, NOT a representative quality estimate",
            "cycles": len(rows), "riskMin": float(min(probabilities)), "riskMax": float(max(probabilities)),
            "shapAdditivityChecked": True, "examples": examples}


def evaluate_groups(validation_rows, probabilities):
    groups = {"all": lambda r: True,
              "no_previous_renewals": lambda r: not r["features"]["has_renewal_history"],
              "previous_recovery_90d": lambda r: r["features"]["recoveries_90d"] > 0}
    # 90-day features can change during a seven-day forecast window. Assign the
    # WHOLE cycle using one representative snapshot, preserving unit cycle weights.
    representatives = {r["cycleId"]: r for r in representative_cycles(validation_rows)}
    results = {}
    for name, predicate in groups.items():
        indices = [i for i, row in enumerate(validation_rows) if predicate(representatives[row["cycleId"]])]
        subset = [validation_rows[i] for i in indices]
        if subset:
            results[name] = {"rows": len(subset), "cycles": len({r["cycleId"] for r in subset}),
                             "lossCycles": sum(r["label"]["value"] for r in {r["cycleId"]: r for r in subset}.values()),
                             "metrics": {key: metrics(subset, [values[i] for i in indices])
                                         for key, values in probabilities.items()}}
    return results


def compare(rows, manifest, history, old_dir, new_dir):
    train_rows, validation_rows = select_rows(rows, manifest)
    old_model, old_report = load_run(old_dir)
    new_model, new_report = load_run(new_dir)
    if new_report["historySha256"] != manifest["historySha256"]:
        raise ValueError("New model trained on a different history")
    for report in (old_report, new_report):
        if report["targetVersion"] != manifest["targetVersion"] or report["schemaVersion"] != manifest["schemaVersion"]:
            raise ValueError("Cannot compare different feature/target definitions")
    pool = make_pool(validation_rows)
    old_probabilities = old_model.predict_proba(pool)[:, 1]
    new_probabilities = new_model.predict_proba(pool)[:, 1]
    prior = sum(r["cycleWeight"] * r["label"]["value"] for r in train_rows) / sum(r["cycleWeight"] for r in train_rows)
    probabilities = {"oldModel": old_probabilities, "newModel": new_probabilities,
                     "newTrainConstant": [prior] * len(validation_rows)}
    results = evaluate_groups(validation_rows, probabilities)
    explanation = explain_examples(new_model, validation_rows, history)
    old_example_probabilities = old_model.predict_proba(make_pool([e["row"] for e in explanation["examples"]]))[:, 1]
    for example, old_probability in zip(explanation["examples"], old_example_probabilities):
        example["oldModelRiskOnSameSnapshot"] = float(old_probability)
    return {"testEvaluated": False, "oldOriginalValidation": old_report["validation"],
            "subgroupAssignment": "whole cycles grouped by snapshot closest to 3 days before expiry; all daily predictions retain their original cycle weights",
            "newValidation": results, "oldHistorySha256": old_report["historySha256"],
            "newHistorySha256": manifest["historySha256"], "newRowsSha256": manifest["rowsSha256"],
            "oldModelSha256": old_report["modelSha256"], "newModelSha256": new_report["modelSha256"],
            "sameExplicitTrainingParameters": old_report["parameters"] == new_report["parameters"],
            "caution": "Changed synthetic data, not an independent real-world test; new validation selects new model stopping point. Old/new IDs belong to separate regenerated worlds.",
            "explanation": explanation}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).parent
    parser.add_argument("--old-run", type=Path, default=base / "runs/first-logloss-v2")
    parser.add_argument("--new-run", type=Path, default=base / "runs/generator-v2-logloss")
    parser.add_argument("--prepared", type=Path, default=base / "prepared")
    parser.add_argument("--history", type=Path, default=base.parent / "examples/risk/vps-business/history.json")
    parser.add_argument("--output", type=Path, default=base / "reports/generator-v2-comparison.json")
    args = parser.parse_args()
    manifest = json.loads((args.prepared / "manifest.json").read_text())
    raw_rows, raw_history = (args.prepared / "rows.jsonl").read_bytes(), args.history.read_bytes()
    if (hashlib.sha256(raw_rows).hexdigest() != manifest["rowsSha256"]
            or hashlib.sha256(raw_history).hexdigest() != manifest["historySha256"]):
        raise ValueError("Source/rows hashes mismatch")
    rows = [json.loads(line) for line in raw_rows.splitlines()]
    report = compare(rows, manifest, json.loads(raw_history), args.old_run, args.new_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"sameExplicitTrainingParameters": report["sameExplicitTrainingParameters"],
                      "newValidation": report["newValidation"], "testEvaluated": False}, indent=2))


if __name__ == "__main__":
    main()
