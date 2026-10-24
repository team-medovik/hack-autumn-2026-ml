#!/usr/bin/env python3
"""Payment-behavior experiment: no tariff/price inputs, no final test evaluation."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

from prepare_dataset import FEATURES, SCHEMA, TARGET
from train_baseline import PARAMS, make_pool as full_pool, metrics, select_rows

PROFILE = "payment-behavior-monotone-v1"
DIRECT_FEATURES = (
    "days_to_expiry", "renewal_cycles_90d", "successful_renewals_90d",
    "failed_cycle_share_90d", "late_cycle_share_90d",
    "mean_paid_delay_hours_90d", "max_paid_delay_hours_90d",
    "days_since_successful_renewal", "recoveries_90d", "has_renewal_history",
    "history_90d_complete",
)
STABILITY = "verified_on_time_share_90d"
BEHAVIOR_FEATURES = DIRECT_FEATURES + (STABILITY,)
CONSTRAINTS = {"failed_cycle_share_90d": 1, "late_cycle_share_90d": 1,
               "mean_paid_delay_hours_90d": 1, "max_paid_delay_hours_90d": 1,
               STABILITY: -1}
EXAMPLE = "20111@2026-08-01T15:00:00Z"


def behavior_values(row):
    source = row["features"]
    values = {name: source[name] for name in DIRECT_FEATURES}
    cycles, successful, late = (source[name] for name in
                                ("renewal_cycles_90d", "successful_renewals_90d", "late_cycle_share_90d"))
    # late_share also counts unpaid cycles. Only when ALL observed cycles are paid
    # can 1-late_share unambiguously mean the share paid on time.
    values[STABILITY] = (max(0.0, min(1.0, 1 - late))
                         if cycles > 0 and successful == cycles and late is not None else None)
    return values


def matrix(rows):
    result = []
    for row in rows:
        values = behavior_values(row)
        vector = []
        for name in BEHAVIOR_FEATURES:
            value = values[name]
            if value is None:
                vector.append(float("nan"))
            elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Invalid behavior feature: {name}")
            else:
                vector.append(value)
        result.append(vector)
    return result


def pool(rows):
    from catboost import Pool
    return Pool(matrix(rows), label=[r["label"]["value"] for r in rows],
                weight=[r["cycleWeight"] for r in rows], feature_names=list(BEHAVIOR_FEATURES))


def parameters():
    return {**PARAMS, "monotone_constraints": dict(CONSTRAINTS)}


def counterfactual_checks(model, rows):
    """Check invariance and finite-value coordinate ordering, not causal effects."""
    base = matrix(rows)
    altered = copy.deepcopy(rows)
    for row in altered:
        f = row["features"]
        f.update(tariff_code="COUNTERFACTUAL-NOT-A-PRICE-SIGNAL", price_minor=99999999,
                 monthly_revenue_minor=88888888, portfolio_monthly_revenue_minor=77777777,
                 portfolio_change_30d_minor=-66666666, account_age_days=99999,
                 service_age_days=99999, live_vps_count=99999)
    original_predictions = model.predict_proba(base)[:, 1]
    altered_predictions = model.predict_proba(matrix(altered))[:, 1]
    invariance_error = max(abs(a-b) for a, b in zip(original_predictions, altered_predictions))
    if invariance_error != 0:
        raise AssertionError("Excluded business attributes changed behavior risk")
    # These are explicit synthetic coordinates, not imputations in training/scoring.
    controls = copy.deepcopy(base)
    for vector in controls:
        for name, direction in CONSTRAINTS.items():
            i = BEHAVIOR_FEATURES.index(name)
            if math.isnan(vector[i]):
                vector[i] = 0.5 if direction < 0 else 0.0
    reference = model.predict_proba(controls)[:, 1]
    deltas = {}
    for name, direction in CONSTRAINTS.items():
        variants = copy.deepcopy(controls)
        index = BEHAVIOR_FEATURES.index(name)
        for vector in variants:
            vector[index] = (max(0.0, vector[index] - 0.25) if direction < 0 else
                             min(1.0, vector[index] + 0.25) if "share" in name else vector[index] + 24)
        predictions = model.predict_proba(variants)[:, 1]
        delta = min(float(a-b) for a, b in zip(predictions, reference))
        if delta < -1e-12:
            raise AssertionError(f"Monotonicity violated for {name}: {delta}")
        deltas[name] = delta
    return {"rowsChecked": len(rows), "excludedAttributesMaximumProbabilityDifference": invariance_error,
            "minimumRiskChangeWhenWorseningOneKnownCoordinate": deltas,
            "unknownVersusKnownOrderingGuaranteed": False}


def run(rows, manifest, previous_run, output):
    import catboost
    train_rows, validation_rows = select_rows(rows, manifest)
    old_report = json.loads((previous_run / "report.json").read_text())
    old_path = previous_run / "model.cbm"
    if (old_report["historySha256"] != manifest["historySha256"]
            or old_report["schemaVersion"] != SCHEMA or old_report["targetVersion"] != TARGET
            or hashlib.sha256(old_path.read_bytes()).hexdigest() != old_report["modelSha256"]):
        raise ValueError("Previous model must match the same history, target and report")
    old_model = catboost.CatBoostClassifier()
    old_model.load_model(str(old_path))
    if old_model.feature_names_ != list(FEATURES):
        raise ValueError("Previous model feature schema mismatch")
    output.mkdir(parents=True, exist_ok=False)
    model = catboost.CatBoostClassifier(**parameters())
    model.fit(pool(train_rows), eval_set=pool(validation_rows), verbose=False)
    new_predictions = model.predict_proba(pool(validation_rows))[:, 1]
    old_predictions = old_model.predict_proba(full_pool(validation_rows))[:, 1]
    prior = sum(r["cycleWeight"] * r["label"]["value"] for r in train_rows) / sum(r["cycleWeight"] for r in train_rows)
    checks = counterfactual_checks(model, validation_rows)
    model_path = output / "model.cbm"
    model.save_model(str(model_path))
    restored = catboost.CatBoostClassifier()
    restored.load_model(str(model_path))
    if any(abs(a-b) > 1e-12 for a, b in zip(new_predictions, restored.predict_proba(pool(validation_rows))[:, 1])):
        raise AssertionError("Save/reload changed predictions")
    examples = []
    for i, row in enumerate(validation_rows):
        if row["sampleId"] == EXAMPLE:
            examples.append({"sampleId": row["sampleId"], "label": row["label"]["value"],
                             "behaviorFeatures": behavior_values(row),
                             "oldRisk": float(old_predictions[i]), "behaviorRisk": float(new_predictions[i]),
                             "monthlyRevenueMinorNotModelInput": row["features"]["monthly_revenue_minor"]})
    report = {"profile": PROFILE, "schemaVersion": SCHEMA, "targetVersion": TARGET,
              "historySha256": manifest["historySha256"], "rowsSha256": manifest["rowsSha256"],
              "previousModelSha256": old_report["modelSha256"],
              "modelSha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
              "catboostVersion": catboost.__version__, "featureNames": list(BEHAVIOR_FEATURES),
              "parameters": parameters(), "effectiveParameters": model.get_all_params(),
              "trees": model.tree_count_, "bestIterationZeroBased": model.get_best_iteration(),
              "validation": {"previousGeneratorV2Model": metrics(validation_rows, old_predictions),
                             "behaviorModel": metrics(validation_rows, new_predictions),
                             "constantTrainRate": metrics(validation_rows, [prior]*len(validation_rows))},
              "trainRows": len(train_rows), "validationRows": len(validation_rows),
              "trainRate": prior, "counterfactualChecks": checks, "requestedExamples": examples,
              "testEvaluated": False, "syntheticOnly": True, "saveLoadPredictionCheckPassed": True,
              "limitations": ["One fixed experiment; no tuning until a desired probability is reached",
                              "Validation chooses stopping; not an independent quality estimate",
                              "Known-coordinate monotonicity permits equality; no promised probability increase",
                              "Changing several unconstrained inputs or unknownness can change ordering",
                              "Synthetic monthly behavior, not validated real-client probabilities",
                              "Risk excludes money; future financial priority is probability times monthly revenue",
                              "Feature independence from tariff/price does not prove causal fairness"]}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (output / "learning-curve.json").write_text(json.dumps(model.get_evals_result(), indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).parent
    parser.add_argument("--prepared", type=Path, default=base / "prepared")
    parser.add_argument("--previous-run", type=Path, default=base / "runs/generator-v2-logloss")
    parser.add_argument("--output", type=Path, default=base / "runs/behavior-v1")
    args = parser.parse_args()
    manifest = json.loads((args.prepared / "manifest.json").read_text())
    raw = (args.prepared / "rows.jsonl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["rowsSha256"]:
        raise ValueError("Prepared rows hash mismatch")
    report = run([json.loads(line) for line in raw.splitlines()], manifest, args.previous_run, args.output)
    print(json.dumps({name: report[name] for name in
                      ("trees", "validation", "counterfactualChecks", "requestedExamples", "testEvaluated")}, indent=2))


if __name__ == "__main__":
    main()
