#!/usr/bin/env python3
"""First reproducible CatBoost baseline. Never fit or predict on the final test set."""
import argparse
import hashlib
import json
import math
import platform
from collections import Counter
from pathlib import Path

from prepare_dataset import (CATEGORICAL, FEATURES, NUMERIC, SCHEMA, TARGET,
                             assign_split, timestamp)

PARAMS = dict(loss_function="Logloss", eval_metric="Logloss", iterations=300,
              depth=4, learning_rate=0.05, random_seed=20261024, thread_count=2,
              early_stopping_rounds=30, use_best_model=True, allow_writing_files=False)


def select_rows(rows, manifest):
    if (manifest["schemaVersion"] != SCHEMA or manifest["targetVersion"] != TARGET
            or manifest["categoricalFeatures"] != list(CATEGORICAL)
            or manifest["numericFeatures"] != list(NUMERIC)):
        raise ValueError("Dataset schema/target/feature order mismatch; rebuild preparation")
    val, test = timestamp(manifest["validationStart"]), timestamp(manifest["testStart"])
    selected = {s: [r for r in rows if r["split"] == s] for s in ("train", "validation")}
    cycle_sets = []
    for split, subset in selected.items():
        if not subset or {r["label"]["value"] for r in subset} != {0, 1}:
            raise ValueError(f"{split} must contain both classes")
        cycles = Counter(r["cycleId"] for r in subset)
        cycle_sets.append(set(cycles))
        for row in subset:
            if assign_split(row, val, test) != split:
                raise ValueError("Invalid time split or excluded target")
            if set(row["features"]) != set(FEATURES):
                raise ValueError("Unexpected features")
            at = timestamp(row["predictionAt"])
            if any(timestamp(row["audit"][key]) > at
                   for key in ("maxFeatureOccurredAt", "maxFeatureRecordedAt")):
                raise ValueError("Future feature evidence")
            if not math.isclose(row["cycleWeight"], 1 / cycles[row["cycleId"]], abs_tol=1e-10):
                raise ValueError("Invalid cycle weight")
    if cycle_sets[0] & cycle_sets[1]:
        raise ValueError("Renewal cycle crosses train and validation")
    return selected["train"], selected["validation"]


def feature_matrix(rows):
    matrix = []
    for row in rows:
        values = []
        for name in FEATURES:
            value = row["features"][name]
            if name in CATEGORICAL:
                if not isinstance(value, str) or not value:
                    raise ValueError(f"Missing categorical feature: {name}")
            elif value is None:
                value = float("nan")
            elif not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Invalid numeric feature: {name}")
            values.append(value)
        matrix.append(values)
    return matrix


def metrics(rows, probabilities):
    if len(rows) != len(probabilities):
        raise ValueError("Prediction length mismatch")
    weights = [r["cycleWeight"] for r in rows]
    labels = [r["label"]["value"] for r in rows]
    total = sum(weights)
    clipped = [max(1e-15, min(1 - 1e-15, float(p))) for p in probabilities]
    return {
        "logloss": -sum(w * (y * math.log(p) + (1-y) * math.log(1-p))
                        for w, y, p in zip(weights, labels, clipped)) / total,
        "brier": sum(w * (y-p)**2 for w, y, p in zip(weights, labels, probabilities)) / total,
        "meanProbability": sum(w*p for w, p in zip(weights, probabilities)) / total,
        "lossRate": sum(w*y for w, y in zip(weights, labels)) / total,
    }


def make_pool(rows):
    from catboost import Pool
    return Pool(feature_matrix(rows), label=[r["label"]["value"] for r in rows],
                weight=[r["cycleWeight"] for r in rows],
                cat_features=list(CATEGORICAL), feature_names=list(FEATURES))


def train(rows, manifest, output, params=None):
    import catboost
    train_rows, validation_rows = select_rows(rows, manifest)
    # A fresh directory prevents silently overwriting an earlier experiment/model.
    output.mkdir(parents=True, exist_ok=False)
    train_pool, validation_pool = make_pool(train_rows), make_pool(validation_rows)
    config = dict(PARAMS if params is None else params)
    model = catboost.CatBoostClassifier(**config)
    model.fit(train_pool, eval_set=validation_pool, verbose=False)
    probabilities = model.predict_proba(validation_pool)[:, 1]
    prevalence = sum(r["cycleWeight"] * r["label"]["value"] for r in train_rows) / sum(
        r["cycleWeight"] for r in train_rows)
    model.save_model(str(output / "model.cbm"))
    restored = catboost.CatBoostClassifier()
    restored.load_model(str(output / "model.cbm"))
    restored_probabilities = restored.predict_proba(validation_pool)[:, 1]
    if not all(abs(a-b) <= 1e-12 for a, b in zip(probabilities, restored_probabilities)):
        raise AssertionError("Saved model predictions changed")
    report = {
        "schemaVersion": SCHEMA, "targetVersion": TARGET, "syntheticOnly": True,
        "pythonVersion": platform.python_version(), "catboostVersion": catboost.__version__,
        "historySha256": manifest["historySha256"], "featureNames": list(FEATURES),
        "parameters": config, "effectiveParameters": model.get_all_params(),
        "bestIterationZeroBased": model.get_best_iteration(), "trees": model.tree_count_,
        "splits": {s: {"rows": len(rs), "cycles": len({r["cycleId"] for r in rs}),
                       "accounts": len({r["accountId"] for r in rs}),
                       "paymentTerms": dict(Counter(r["features"]["payment_term"] for r in rs))}
                   for s, rs in (("train", train_rows), ("validation", validation_rows))},
        "baselineTrainProbability": prevalence,
        "validation": {"catboost": metrics(validation_rows, probabilities),
                       "constantTrainPrevalence": metrics(validation_rows, [prevalence]*len(validation_rows))},
        "evaluationWeight": "one total unit per renewal cycle; no class/revenue weights",
        "testEvaluated": False, "saveLoadPredictionCheckPassed": True,
        "modelSha256": hashlib.sha256((output / "model.cbm").read_bytes()).hexdigest(),
        "limitations": ["Synthetic demonstration, not evidence of real retention quality",
                        "Validation selects stopping point; it is not a final unbiased test",
                        "Monthly outcomes only; hourly services unsupported by this training",
                        "Probabilities not separately calibrated; not production-ready",
                        "No API integration, rules replacement or revenue ranking implemented"],
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (output / "learning-curve.json").write_text(json.dumps(model.get_evals_result(), indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path(__file__).parent / "prepared")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "runs/first-logloss-v2")
    args = parser.parse_args()
    manifest = json.loads((args.prepared / "manifest.json").read_text())
    raw = (args.prepared / "rows.jsonl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["rowsSha256"]:
        raise ValueError("Prepared rows changed since manifest creation")
    rows = [json.loads(line) for line in raw.splitlines()]
    report = train(rows, manifest, args.output)
    print(json.dumps({k: report[k] for k in ("trees", "splits", "baselineTrainProbability",
                                           "validation", "testEvaluated")}, indent=2))


if __name__ == "__main__":
    main()
