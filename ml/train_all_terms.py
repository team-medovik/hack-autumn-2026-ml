#!/usr/bin/env python3
"""Fixed experiment on independent synthetic worlds for hour/day/month VPS.

No billing/behavior rule is changed. Sampling precedes labels, world allocation
precedes generation, and only validation selects tree count/deployment terms.
"""
import argparse
from collections import Counter, defaultdict
from datetime import timedelta
import json
import math
from pathlib import Path
import sys
import platform

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'examples/risk/event-driven'))
from behavior import BehaviorSimulation, make_population
from prepare_event_driven import EventHistory, FEATURES, SCHEMA, TARGET, CATEGORICAL
from prepare_dataset import timestamp, iso, DAY
from train_event_driven import matrix
from train_baseline import PARAMS
from evaluate_event_driven import summarize, digest

TERMS = ('hour', 'day', 'month')
WORLD_PLAN = [('train', 2026102600 + i) for i in range(1, 7)] + [
    ('validation', 2026102607), ('validation', 2026102608),
    ('test', 2026102609), ('test', 2026102610)]
START = '2026-01-01T00:00:00Z'


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sample_world(data, world_id, split):
    history = EventHistory(data)
    scheduled = defaultdict(set)
    # Observe the first half-period before initial expiry as well as daily
    # observation. This includes first-hour losses that a daily grid would miss.
    for event in history.events:
        if event['type'] != 'payment_succeeded' or event.get('mode') != 'purchase':
            continue
        term = event['paymentTerm']
        if term not in ('hour', 'day'):
            continue
        offset = timedelta(minutes=30) if term == 'hour' else timedelta(hours=12)
        at = timestamp(event['paidThrough']) - offset
        if at <= history.as_of and history.known(event, at):
            scheduled[at].add(event['serviceId'])
    at = history.history_start.replace(hour=15, minute=0, second=0, microsecond=0)
    if at < history.history_start:
        at += DAY
    while at <= history.as_of:
        scheduled[at]
        at += DAY
    rows = []
    for at, first_services in sorted(scheduled.items()):
        daily = at.hour == 15 and at.minute == at.second == at.microsecond == 0
        for row in history.features_at(at, None if daily else first_services):
            row['label'] = history.label(row)
            row['worldId'] = world_id
            row['accountGroup'] = f"{world_id}:{row['accountId']}"
            row['originalSampleId'] = row['sampleId']
            row['sampleId'] = world_id + ':' + row['sampleId']
            row['cycleId'] = world_id + ':' + row['cycleId']
            row['partition'] = split
            row['split'] = split if row['label']['value'] is not None else (
                'excluded' if row['label']['reason'] == 'excluded_nonfinancial_block' else 'pending')
            row['sampling'] = 'daily_15utc' if daily else 'initial_half_period'
            rows.append(row)
    counts = Counter((r['split'], r['cycleId']) for r in rows)
    for row in rows:
        row['cycleWeight'] = 1 / counts[row['split'], row['cycleId']]
    return rows


def validate(rows):
    if len({r['sampleId'] for r in rows}) != len(rows):
        raise ValueError('Duplicate observation')
    groups, cycles = defaultdict(set), defaultdict(set)
    counts = Counter((r['split'], r['cycleId']) for r in rows)
    for row in rows:
        if list(row['features']) != list(FEATURES):
            raise ValueError('Feature names/order changed')
        if row['features']['payment_term'] not in TERMS:
            raise ValueError('Unexpected payment term')
        if row['labelDeadline'] != iso(timestamp(row['dueAt']) + timedelta(days=5)):
            raise ValueError('Target definition changed')
        at = timestamp(row['predictionAt'])
        if not at < timestamp(row['dueAt']) <= at + timedelta(days=7):
            raise ValueError('Prediction outside preexpiry window')
        if any(timestamp(row['audit'][key]) > at for key in ('maxFeatureOccurredAt', 'maxFeatureRecordedAt')):
            raise ValueError('Future evidence leaked into features')
        if not math.isclose(row['cycleWeight'], 1 / counts[row['split'], row['cycleId']], abs_tol=1e-10):
            raise ValueError('Wrong cycle weight')
        if row['split'] in ('train', 'validation', 'test'):
            if type(row['label']['value']) is not int or row['label']['value'] not in (0, 1):
                raise ValueError('Unresolved label in fitted/evaluated rows')
            groups[row['accountGroup']].add(row['split'])
            cycles[row['cycleId']].add(row['split'])
    if any(len(parts) != 1 for parts in [*groups.values(), *cycles.values()]):
        raise ValueError('Client/cycle crossed data partitions')


def make_pool(rows, training=False):
    from catboost import Pool
    # Each completed cycle totals one; each term totals one third in fit/eval_set.
    # No class weights and no outcome-based oversampling.
    totals = {term: sum(r['cycleWeight'] for r in rows if r['features']['payment_term'] == term)
              for term in TERMS}
    if training and any(v == 0 for v in totals.values()):
        raise ValueError('Each term must be observed in train/validation')
    total = sum(totals.values())
    weights = [r['cycleWeight'] * total / (3 * totals[r['features']['payment_term']]) for r in rows] if training else None
    return Pool(matrix(rows, FEATURES), feature_names=list(FEATURES), cat_features=list(CATEGORICAL),
                label=[r['label']['value'] for r in rows] if training else None, weight=weights)


def partition_metrics(rows, probabilities):
    return {term: summarize([r for r in rows if r['features']['payment_term'] == term],
                            [p for r, p in zip(rows, probabilities) if r['features']['payment_term'] == term])
            for term in TERMS}


def validation_gate(scores, baselines):
    result = {}
    for term in TERMS:
        model, baseline = scores[term], baselines[term]
        enough = model['lossCycles'] >= 10 and model['cycles'] - model['lossCycles'] >= 10
        result[term] = {'enabled': enough and model['logloss'] < baseline['logloss'] and model['brier'] < baseline['brier'],
                        'enoughValidationOutcomes': enough,
                        'rule': '>=10 positive and >=10 negative validation cycles; Logloss and Brier below own train constant'}
    return result


def examples(rows, new, old, constants):
    selected = []
    for term in TERMS:
        ids = [i for i, r in enumerate(rows) if r['features']['payment_term'] == term]
        # Contrasts are labelled illustrations, not estimates of prevalence.
        for label, highest, name in ((1, True, 'highest_risk_loss'), (0, False, 'lowest_risk_retained'),
                                     (1, False, 'lowest_risk_loss'), (0, True, 'highest_risk_retained')):
            candidates = [i for i in ids if rows[i]['label']['value'] == label]
            if candidates:
                i = (max if highest else min)(candidates, key=lambda k: new[k])
                selected.append({'case': name, 'paymentTerm': term, 'row': rows[i],
                                 'newProbability': new[i], 'oldProbability': old[i],
                                 'constantProbability': constants[term]})
    return selected


def run(output, accounts=40, days=180):
    from catboost import CatBoostClassifier
    output.mkdir(parents=True, exist_ok=False)
    histories = output / 'histories'
    histories.mkdir()
    lifecycle = {'cohortSpreadDays': 30}
    plan = {'version': 'all-terms-independent-worlds-v1', 'worlds': WORLD_PLAN,
            'accountsPerWorld': accounts, 'days': days, 'startAt': START,
            'paymentTerms': TERMS, 'lifecycleOverrides': lifecycle, 'parameters': PARAMS,
            'sampling': 'first hour/day cycle at half period; all eligible services daily15UTC; before labels',
            'weights': 'each cycle totals1; train and validation eval_set equal total weight per payment term',
            'validationGate': '>=10 positive and >=10 negative cycles, Logloss and Brier below term train constant',
            'featureSchemaVersion': SCHEMA, 'targetVersion': TARGET,
            'generatorSourceSha256': {name: digest(ROOT / 'examples/risk/event-driven' / name) for name in ('billing.py', 'behavior.py')},
            'featurePreparationSha256': digest(HERE / 'prepare_event_driven.py'),
            'featureNames': list(FEATURES), 'categoricalFeatures': list(CATEGORICAL),
            'testPolicy': 'test worlds not used in fit, eval_set, model/threshold selection or term enablement'}
    write_json(output / 'plan.json', plan)
    rows, sources = [], []
    for split, seed in WORLD_PLAN:
        world_id = f'world-{seed}'
        print(f'Generating {world_id} {split}', flush=True)
        population = make_population(seed, accounts, terms=TERMS, start_at=START, days=days, lifecycle=lifecycle)
        world = BehaviorSimulation(population, seed, START, days)
        data = world.run()
        # Public observations only; no private profile/decisions are given to features.
        data = {'schemaVersion': data['schemaVersion'], 'asOf': data['asOf'], 'events': data['events']}
        path = histories / f'{world_id}.json'
        write_json(path, data)
        subset = sample_world(data, world_id, split)
        rows.extend(subset)
        sources.append({'worldId': world_id, 'seed': seed, 'partition': split,
                        'historyPath': str(path.relative_to(output)), 'historySha256': digest(path),
                        'eventCount': len(data['events']), 'rows': len(subset),
                        'archetypes': dict(Counter(p['archetype'] for p in population))})
        print(f'  events={len(data["events"])} rows={len(subset)} labels={dict(Counter(r["label"]["value"] for r in subset))}', flush=True)
    validate(rows)
    with (output / 'rows.jsonl').open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    parts = {split: [r for r in rows if r['split'] == split] for split in ('train', 'validation', 'test')}
    for split in ('train', 'validation'):
        if {r['label']['value'] for r in parts[split]} != {0, 1}:
            raise ValueError('Train/validation need both classes')
    constants = {term: sum(r['cycleWeight'] * r['label']['value'] for r in parts['train'] if r['features']['payment_term'] == term)
                 / sum(r['cycleWeight'] for r in parts['train'] if r['features']['payment_term'] == term) for term in TERMS}
    old_path = HERE / 'runs/event-driven-v3-logloss/newEnhanced36.cbm'
    old_hash = digest(old_path)
    old = CatBoostClassifier()
    old.load_model(str(old_path))
    if old.feature_names_ != list(FEATURES):
        raise ValueError('Old model schema mismatch')
    model = CatBoostClassifier(**PARAMS)
    print('Fitting fixed all-term model; test excluded', flush=True)
    model.fit(make_pool(parts['train'], True), eval_set=make_pool(parts['validation'], True), verbose=False)
    model.save_model(str(output / 'model.cbm'))
    val_p = list(map(float, model.predict_proba(make_pool(parts['validation']))[:, 1]))
    val_scores = partition_metrics(parts['validation'], val_p)
    val_constant = partition_metrics(parts['validation'], [constants[r['features']['payment_term']] for r in parts['validation']])
    gates = validation_gate(val_scores, val_constant)
    # Freeze term enablement before any test predictions.
    write_json(output / 'validation-decision.json', gates)
    predictions, results = [], {}
    test_examples = []
    for split in ('train', 'validation', 'test'):
        subset = parts[split]
        pool = make_pool(subset)
        pnew = list(map(float, model.predict_proba(pool)[:, 1]))
        pold = list(map(float, old.predict_proba(pool)[:, 1]))
        baseline = [constants[r['features']['payment_term']] for r in subset]
        results[split] = {'new': partition_metrics(subset, pnew), 'frozenMonthly36': partition_metrics(subset, pold),
                          'trainConstant': partition_metrics(subset, baseline)}
        predictions.extend({'sampleId': r['sampleId'], 'worldId': r['worldId'], 'serviceId': r['serviceId'],
                            'accountId': r['accountId'], 'predictionAt': r['predictionAt'], 'dueAt': r['dueAt'],
                            'paymentTerm': r['features']['payment_term'], 'split': split, 'label': r['label'],
                            'cycleWeight': r['cycleWeight'], 'newProbability': n, 'oldProbability': o,
                            'constantProbability': c} for r, n, o, c in zip(subset, pnew, pold, baseline))
        if split == 'test':
            test_examples = examples(subset, pnew, pold, constants)
    restored = CatBoostClassifier()
    restored.load_model(str(output / 'model.cbm'))
    check = list(map(float, restored.predict_proba(make_pool(parts['validation']))[:, 1]))
    if check != val_p or old_hash != digest(old_path):
        raise AssertionError('Artifact verification failed')
    with (output / 'predictions.jsonl').open('w') as stream:
        for row in predictions:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    write_json(output / 'examples.json', test_examples)
    report = {'experimentVersion': plan['version'], 'planSha256': digest(output / 'plan.json'),
              'rowsSha256': digest(output / 'rows.jsonl'), 'sources': sources,
              'featureSchemaVersion': SCHEMA, 'targetVersion': TARGET, 'featureNames': list(FEATURES),
              'syntheticOnly': True, 'modelSha256': digest(output / 'model.cbm'),
              'frozenMonthlySha256': old_hash, 'trees': model.tree_count_, 'parameters': PARAMS, 'pythonVersion': platform.python_version(),
              'catboostVersion': __import__('catboost').__version__,
              'trainConstants': constants, 'validationEnablement': gates,
              'splits': results, 'rowsByPartition': dict(Counter(r['split'] for r in rows)),
              'totalEvents': sum(s['eventCount'] for s in sources),
              'limitations': ['Synthetic hypotheses, not real Aeza behavior or real-world accuracy',
                              'Hour/day frozen monthly predictions are out-of-domain diagnostics, not a valid old deployed baseline',
                              'Daily snapshots plus first-cycle observations do not evaluate every hourly renewal',
                              'Unseen worlds/accounts, but same generator hypotheses across partitions',
                              'Days tariff remains synthetic; five-day label unchanged across terms',
                              'Per-term rollout selected on validation only; test is reported even when worse',
                              'No probability calibration or fit on test, no outcome-dependent sampling'],
              'saveLoadVerified': True}
    write_json(output / 'report.json', report)
    manifest = {'modelVersion': 'event-driven-all-terms-v1-' + report['modelSha256'][:12],
                'modelSha256': report['modelSha256'], 'featureSchemaVersion': SCHEMA, 'targetVersion': TARGET,
                'featureNames': list(FEATURES), 'supportedPaymentTerms': [t for t in TERMS if gates[t]['enabled']],
                'syntheticOnly': True, 'preExpiryHorizonDays': 7, 'outcomeGraceDays': 5}
    write_json(output / 'model-manifest.json', manifest)
    print(json.dumps({'trees': model.tree_count_, 'enabledTerms': manifest['supportedPaymentTerms'],
                      'counts': report['rowsByPartition'], 'test': results['test']}, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--accounts', type=int, default=40)
    parser.add_argument('--days', type=int, default=180)
    args = parser.parse_args()
    run(args.output, args.accounts, args.days)
