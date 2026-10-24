#!/usr/bin/env python3
"""Fixed Logloss experiment on event-driven data; no generator changes or test tuning."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

from prepare_event_driven import SCHEMA, TARGET, FEATURES, LEGACY_FEATURES, CATEGORICAL, validate_rows
from train_baseline import PARAMS
from evaluate_event_driven import digest, checked_probabilities, summarize, examples


def matrix(rows, names):
    result = []
    for row in rows:
        values = []
        for name in names:
            value = row["features"][name]
            if name in CATEGORICAL:
                if not isinstance(value, str) or not value:
                    raise ValueError(f"Invalid category: {name}")
            elif value is None:
                value = float("nan")
            elif type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"Invalid number: {name}")
            values.append(value)
        result.append(values)
    return result


def pool(rows, names, labels=False):
    from catboost import Pool
    return Pool(matrix(rows, names), feature_names=list(names), cat_features=list(CATEGORICAL),
                label=[r["label"]["value"] for r in rows] if labels else None,
                weight=[r["cycleWeight"] for r in rows] if labels else None)


def select_partitions(rows, manifest):
    if (manifest["schemaVersion"] != SCHEMA or manifest["targetVersion"] != TARGET
            or manifest["featureNames"] != list(FEATURES)
            or manifest["legacyFeatureNames"] != list(LEGACY_FEATURES)
            or manifest["categoricalFeatures"] != list(CATEGORICAL)):
        raise ValueError("Event dataset schema, target, or feature mismatch")
    validate_rows(rows, manifest["validationStart"], manifest["testStart"])
    for row in rows:
        if any(row["features"][name] != row["legacy_features"][name] for name in LEGACY_FEATURES):
            raise ValueError("Inconsistent feature views")
        if row["features"]["payment_term"] != "month":
            raise ValueError("This experiment supports monthly VPS only")
    partitions = {s: [r for r in rows if r["split"] == s] for s in ("train", "validation", "test")}
    for split, selected in partitions.items():
        if not selected or any(type(r["label"]["value"]) is not int or r["label"]["value"] not in (0, 1)
                               for r in selected):
            raise ValueError("Nonempty labeled partitions required")
        if split != "test" and {r["label"]["value"] for r in selected} != {0, 1}:
            raise ValueError("Train and validation need both classes")
        counts = Counter(r["cycleId"] for r in selected)
        if any(not math.isclose(r["cycleWeight"], 1 / counts[r["cycleId"]], abs_tol=1e-10) for r in selected):
            raise ValueError("Incorrect cycle weights")
    return partitions


def fit_model(train, validation, names):
    from catboost import CatBoostClassifier
    model = CatBoostClassifier(**PARAMS)
    model.fit(pool(train, names, labels=True), eval_set=pool(validation, names, labels=True), verbose=False)
    return model


def run(prepared, history, previous, output):
    from catboost import CatBoostClassifier
    if output.exists():
        raise ValueError("Output must be a new directory")
    manifest = json.loads((prepared / "manifest.json").read_text())
    if digest(prepared / "rows.jsonl") != manifest["rowsSha256"] or digest(history) != manifest["historySha256"]:
        raise ValueError("Source/prepared hash mismatch")
    rows = [json.loads(line) for line in (prepared / "rows.jsonl").read_text().splitlines()]
    parts = select_partitions(rows, manifest)
    previous_info = json.loads((previous / "report.json").read_text())
    old_hash = digest(previous / "model.cbm")
    if old_hash != previous_info["modelSha256"] or previous_info["featureNames"] != list(LEGACY_FEATURES):
        raise ValueError("Frozen model hash/features mismatch")
    old = CatBoostClassifier()
    old.load_model(str(previous / "model.cbm"))
    if old.feature_names_ != list(LEGACY_FEATURES) or list(old.classes_) != [0, 1]:
        raise ValueError("Frozen model feature/class mismatch")
    # Both fixed configurations train before any test predictions are calculated.
    enhanced = fit_model(parts["train"], parts["validation"], FEATURES)
    refit = fit_model(parts["train"], parts["validation"], LEGACY_FEATURES)
    models = {"newEnhanced36": (enhanced, FEATURES), "refitLegacy24": (refit, LEGACY_FEATURES),
              "frozenLegacy24": (old, LEGACY_FEATURES)}
    train_rate = sum(r["cycleWeight"] * r["label"]["value"] for r in parts["train"]) / sum(
        r["cycleWeight"] for r in parts["train"])
    constants = {"constantNewTrain": train_rate, "constantOldTrain": previous_info["baselineTrainProbability"]}
    report = {"experimentVersion": "event-driven-logloss-v1", "schemaVersion": SCHEMA,
              "targetVersion": TARGET, "historySha256": manifest["historySha256"],
              "rowsSha256": manifest["rowsSha256"], "parameters": dict(PARAMS),
              "constants": constants, "models": {}, "splits": {}, "examples": {},
              "allPredictionRows": len(rows), "syntheticOnly": True,
              "limitations": ["Exploratory test previously inspected; not independent final real-world validation",
                              "Fixed seed/configuration, no parameter/threshold search",
                              "Validation selects tree count; test never passed to fit or eval_set",
                              "All-row predictions include in-sample train and non-evaluation partitions, explicitly labeled",
                              "One total weight per cycle in each partition; same clients can recur",
                              "No price-based priority, calibration, expert-score fusion or API integration",
                              "SHAP is model attribution in log odds, not a causal explanation"]}
    output.mkdir(parents=True, exist_ok=False)
    predictions = {}
    for key, (model, names) in models.items():
        all_pool = pool(rows, names)
        predictions[key] = checked_probabilities(model.predict_proba(all_pool)[:, 1], len(rows))
        info = {"featureNames": list(names), "trees": model.tree_count_,
                "bestIterationZeroBased": model.get_best_iteration(), "modelSha256": old_hash}
        if key != "frozenLegacy24":
            model_path = output / f"{key}.cbm"
            model.save_model(str(model_path))
            restored = CatBoostClassifier()
            restored.load_model(str(model_path))
            restored_p = checked_probabilities(restored.predict_proba(all_pool)[:, 1], len(rows))
            if restored.feature_names_ != list(names) or any(abs(a-b) > 1e-12 for a, b in zip(predictions[key], restored_p)):
                raise AssertionError("Saved model differs")
            info.update(modelSha256=digest(model_path), saveLoadVerified=True,
                        effectiveParameters=model.get_all_params(),
                        featureImportance=sorted([{"feature": name, "importance": float(v)} for name, v in
                                                  zip(names, model.get_feature_importance())],
                                                 key=lambda f: f["importance"], reverse=True))
            (output / f"{key}-learning-curve.json").write_text(json.dumps(model.get_evals_result(), indent=2) + "\n")
        report["models"][key] = info
    for name, p in constants.items():
        predictions[name] = [p] * len(rows)
    for split in parts:
        indices = [i for i, row in enumerate(rows) if row["split"] == split]
        selected = [rows[i] for i in indices]
        report["splits"][split] = {key: summarize(selected, [p[i] for i in indices]) for key, p in predictions.items()}
        if split == "train":
            continue  # Training metrics are diagnostic, never evidence of generalization.
        selected_pool = pool(selected, FEATURES)
        explained = examples(enhanced, selected_pool, selected,
                             [predictions["newEnhanced36"][i] for i in indices], list(FEATURES), matrix(selected, FEATURES))
        lookup = {r["sampleId"]: i for i, r in enumerate(rows)}
        for item in explained:
            i = lookup[item["sampleId"]]
            item["observedFeatures"] = rows[i]["features"]
            item.pop("extraObservedContextNotModelInput")
            item["comparisonProbabilities"] = {key: p[i] for key, p in predictions.items()}
        report["examples"][split] = explained
    with (output / "all-predictions.jsonl").open("x") as stream:
        for i, row in enumerate(rows):
            record = {key: row[key] for key in ("sampleId", "cycleId", "serviceId", "accountId", "predictionAt",
                                               "dueAt", "labelDeadline", "split", "cycleWeight")}
            record.update(actualLabel=row["label"]["value"], labelReason=row["label"]["reason"],
                          usedForTraining=row["split"] == "train", usedForModelSelection=row["split"] == "validation",
                          **{key: p[i] for key, p in predictions.items()})
            record["observedContext"] = {k: row["features"][k] for k in (
                "price_minor", "auto_prolong", "wallet_balance_minor", "current_service_funding_ratio",
                "topup_count_30d", "topup_sum_30d_minor", "days_since_topup")}
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    report["allPredictionsSha256"] = digest(output / "all-predictions.jsonl")
    if digest(previous / "model.cbm") != old_hash:
        raise AssertionError("Frozen model was modified")
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("ml/prepared-event-driven/lifecycle-v3"))
    parser.add_argument("--history", type=Path, default=Path("examples/risk/event-driven/generated/lifecycle-v3-seed20261024/observations.json"))
    parser.add_argument("--previous", type=Path, default=Path("ml/runs/generator-v2-logloss"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.prepared, args.history, args.previous, args.output)
    print(json.dumps({"trees": {k: v["trees"] for k, v in report["models"].items()},
                      "validation": report["splits"]["validation"], "test": report["splits"]["test"]}, indent=2))


if __name__ == "__main__":
    main()
