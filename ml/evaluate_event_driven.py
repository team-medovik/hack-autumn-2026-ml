#!/usr/bin/env python3
"""Explicit, read-only model bridge: frozen legacy models on event-driven rows.

No fit, tuning, calibration, API integration, or rewriting dataset schema/target.
The target's business meaning is retained; event semantics and distribution changed.
"""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from behavior_model import BEHAVIOR_FEATURES, matrix as behavior_matrix
from prepare_dataset import FEATURES as OLD_FEATURES, CATEGORICAL, timestamp
from prepare_event_driven import SCHEMA, TARGET, FEATURES, validate_rows
from train_baseline import feature_matrix, metrics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def legacy_rows(rows):
    """Only the explicit compatible feature view crosses the model boundary."""
    result = []
    for row in rows:
        if tuple(row["legacy_features"]) != tuple(OLD_FEATURES):
            raise ValueError("Legacy feature order mismatch")
        if any(row["features"][k] != row["legacy_features"][k] for k in OLD_FEATURES):
            raise ValueError("Legacy and enhanced view disagree")
        result.append({**row, "features": dict(row["legacy_features"])})
    return result


def checked_probabilities(values, size):
    result = [float(p) for p in values]
    if len(result) != size or any(not math.isfinite(p) or not 0 <= p <= 1 for p in result):
        raise ValueError("Invalid model probability or prediction count")
    return result


def auc(rows, probabilities):
    """Weighted pairwise ROC AUC; ties count half, one-class result unknown."""
    groups = {}
    for row, p in zip(rows, probabilities):
        weights = groups.setdefault(p, [0.0, 0.0])
        weights[row["label"]["value"]] += row["cycleWeight"]
    negative = sum(v[0] for v in groups.values())
    positive = sum(v[1] for v in groups.values())
    if not positive or not negative:
        return None
    lower_negative = concordant = 0.0
    for _, (neg, pos) in sorted(groups.items()):
        concordant += pos * (lower_negative + neg / 2)
        lower_negative += neg
    return concordant / (positive * negative)


def summarize(rows, probabilities):
    probabilities = checked_probabilities(probabilities, len(rows))
    counts = Counter(r["cycleId"] for r in rows)
    if not rows or any(not math.isclose(r["cycleWeight"], 1 / counts[r["cycleId"]], abs_tol=1e-10)
                       for r in rows):
        raise ValueError("Empty partition or invalid cycle weights")
    return {**metrics(rows, probabilities), "rocAuc": auc(rows, probabilities),
            "minimumProbability": min(probabilities), "maximumProbability": max(probabilities),
            "rows": len(rows), "cycles": len(counts),
            "lossCycles": len({r["cycleId"] for r in rows if r["label"]["value"] == 1})}


def representative_indices(rows):
    """One snapshot per cycle closest to due-3d, selected without the outcome."""
    indices = {}
    key = lambda i: (abs(rows[i]["legacy_features"]["days_to_expiry"] - 3), rows[i]["predictionAt"])
    for i, row in enumerate(rows):
        previous = indices.get(row["cycleId"])
        if previous is None or key(i) < key(previous):
            indices[row["cycleId"]] = i
    return list(indices.values())


def examples(model, pool, rows, probabilities, names, matrix):
    candidates = representative_indices(rows)
    cases = []
    for label, high, name in ((1, True, "highest_risk_loss"), (0, False, "lowest_risk_retained"),
                              (1, False, "lowest_risk_loss"), (0, True, "highest_risk_retained")):
        selected = [i for i in candidates if rows[i]["label"]["value"] == label]
        if selected:
            i = (max if high else min)(selected, key=lambda n: probabilities[n])
            cases.append((name, i))
    selected_pool = pool.slice([i for _, i in cases])
    contributions = model.get_feature_importance(selected_pool, type="ShapValues")
    raw = model.predict(selected_pool, prediction_type="RawFormulaVal")
    result = []
    for (case, i), shap, logit in zip(cases, contributions, raw):
        if not math.isclose(sum(shap), float(logit), abs_tol=1e-8):
            raise AssertionError("SHAP does not reconstruct log odds")
        p = 1 / (1 + math.exp(-float(logit)))
        if not math.isclose(p, probabilities[i], abs_tol=1e-10):
            raise AssertionError("Log odds do not reconstruct probability")
        row = rows[i]
        factors = [{"feature": name, "value": None if isinstance(v, float) and math.isnan(v) else v,
                    "contributionLogOdds": float(c)} for name, v, c in zip(names, matrix[i], shap[:-1])]
        result.append({"case": case, "sampleId": row["sampleId"], "serviceId": row["serviceId"],
                       "accountId": row["accountId"], "predictionAt": row["predictionAt"],
                       "dueAt": row["dueAt"], "labelDeadline": row["labelDeadline"],
                       "probability": probabilities[i], "actualLabel": row["label"],
                       "observedFeatures": row["legacy_features"],
                       "extraObservedContextNotModelInput": row["enhanced_features"],
                       "baseLogOdds": float(shap[-1]),
                       "contributions": sorted(factors, key=lambda f: abs(f["contributionLogOdds"]), reverse=True)})
    return result


def evaluate(prepared, history, model_root, output):
    from catboost import CatBoostClassifier, Pool
    if output.exists():
        raise ValueError("Use a new output directory; existing evaluations are preserved")
    manifest = json.loads((prepared / "manifest.json").read_text())
    if manifest["schemaVersion"] != SCHEMA or manifest["targetVersion"] != TARGET:
        raise ValueError("Unsupported event-driven schema/target")
    if manifest["featureNames"] != list(FEATURES) or manifest["legacyFeatureNames"] != list(OLD_FEATURES):
        raise ValueError("Manifest feature order mismatch")
    if digest(prepared / "rows.jsonl") != manifest["rowsSha256"] or digest(history) != manifest["historySha256"]:
        raise ValueError("Prepared/source hash mismatch")
    rows = [json.loads(line) for line in (prepared / "rows.jsonl").read_text().splitlines()]
    validate_rows(rows, manifest["validationStart"], manifest["testStart"])
    compatible = legacy_rows(rows)
    if any(r["features"]["payment_term"] != "month" for r in compatible):
        raise ValueError("Saved models are monthly only; do not pool other periods")
    train = [r for r in compatible if r["split"] == "train"]
    train_rate = sum(r["cycleWeight"] * r["label"]["value"] for r in train) / sum(r["cycleWeight"] for r in train)
    report = {"evaluationVersion": "frozen-model-event-driven-v1", "trainingStarted": False,
              "historySha256": manifest["historySha256"], "rowsSha256": manifest["rowsSha256"],
              "sourceSchemaVersion": SCHEMA, "sourceTargetVersion": TARGET,
              "newTrainLossRate": train_rate, "models": {},
              "limitations": ["Synthetic out-of-generator evaluation, not real-client quality",
                              "Validation/test are exploratory; no model parameters chosen using this run",
                              "Legacy feature names match, event semantics/distribution changed",
                              "Additional 12 observed features do not enter either frozen model",
                              "Cycle weighting does not make repeated customers statistically independent",
                              "Examples chosen by outcome and risk are illustrative, not prevalence estimates",
                              "SHAP explains log odds, not causal effects or additive percentage points",
                              "No API integration or combination with expert score"]}
    predictions = []
    for name, kind in (("generator-v2-logloss", "legacy"), ("behavior-v1", "behavior")):
        path = model_root / name / "model.cbm"
        info = json.loads((path.parent / "report.json").read_text())
        before = digest(path)
        if before != info["modelSha256"]:
            raise ValueError("Model hash mismatch")
        model = CatBoostClassifier()
        model.load_model(str(path))
        names = list(OLD_FEATURES if kind == "legacy" else BEHAVIOR_FEATURES)
        if model.feature_names_ != names or info["featureNames"] != names or list(model.classes_) != [0, 1]:
            raise ValueError("Model features/classes mismatch")
        original_prior = info.get("baselineTrainProbability", info.get("trainRate"))
        result = {"modelSha256": before, "featureNames": names,
                  "trainedSchemaVersion": info["schemaVersion"], "trainedTargetVersion": info["targetVersion"],
                  "originalTrainLossRate": original_prior, "splits": {}, "examples": {}}
        for split in ("validation", "test"):
            indices = [i for i, r in enumerate(rows) if r["split"] == split]
            subset = [compatible[i] for i in indices]
            matrix = feature_matrix(subset) if kind == "legacy" else behavior_matrix(subset)
            pool = Pool(matrix, feature_names=names,
                        cat_features=list(CATEGORICAL) if kind == "legacy" else [])
            probabilities = checked_probabilities(model.predict_proba(pool)[:, 1], len(subset))
            result["splits"][split] = {"model": summarize(subset, probabilities),
                                       "constantOriginalTrain": summarize(subset, [original_prior] * len(subset)),
                                       "constantNewTrain": summarize(subset, [train_rate] * len(subset))}
            result["examples"][split] = examples(model, pool, [rows[i] for i in indices], probabilities, names, matrix)
            predictions.extend({"model": name, "sampleId": r["sampleId"], "serviceId": r["serviceId"],
                                "predictionAt": r["predictionAt"], "split": split, "cycleWeight": r["cycleWeight"],
                                "lossProbability": p, "actualLabel": r["label"]["value"]}
                               for r, p in zip(subset, probabilities))
        if digest(path) != before:
            raise AssertionError("Frozen model changed")
        report["models"][name] = result
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    with (output / "predictions.jsonl").open("x") as stream:
        for row in predictions:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("ml/prepared-event-driven/lifecycle-v3"))
    parser.add_argument("--history", type=Path, default=Path("examples/risk/event-driven/generated/lifecycle-v3-seed20261024/observations.json"))
    parser.add_argument("--models", type=Path, default=Path("ml/runs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.prepared, args.history, args.models, args.output)
    print(json.dumps({name: value["splits"] for name, value in report["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
