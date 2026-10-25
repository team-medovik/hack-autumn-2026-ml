"""Validated immutable artifact and the same feature projection used by training."""
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ml'))
from prepare_event_driven import EventHistory, FEATURES, CATEGORICAL, SCHEMA, TARGET
from prepare_dataset import timestamp, iso, HORIZON, GRACE


class ModelUnavailable(ValueError):
    pass


class Predictor:
    def __init__(self, model_path, manifest_path):
        from catboost import CatBoostClassifier, CatBoostError
        self.info = json.loads(Path(manifest_path).read_text())
        digest = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
        required = {'modelVersion', 'modelSha256', 'featureSchemaVersion', 'targetVersion',
                    'syntheticOnly', 'supportedPaymentTerms', 'preExpiryHorizonDays',
                    'outcomeGraceDays', 'featureNames'}
        if (set(self.info) != required or self.info['modelSha256'] != digest
                or self.info['featureSchemaVersion'] != SCHEMA or self.info['targetVersion'] != TARGET
                or self.info['featureNames'] != list(FEATURES)
                or self.info['syntheticOnly'] is not True
                or not self.info['modelVersion']
                or self.info['preExpiryHorizonDays'] != 7 or self.info['outcomeGraceDays'] != 5
                or not self.info['supportedPaymentTerms']
                or not set(self.info['supportedPaymentTerms']) <= {'hour', 'day', 'month'}
                or len(set(self.info['supportedPaymentTerms'])) != len(self.info['supportedPaymentTerms'])):
            raise ModelUnavailable('Artifact manifest is incompatible')
        self.model = CatBoostClassifier()
        try:
            self.model.load_model(str(model_path))
        except CatBoostError as exc:
            raise ModelUnavailable('Model artifact could not be loaded') from exc
        if (self.model.feature_names_ != list(FEATURES) or list(self.model.classes_) != [0, 1]
                or self.model.get_cat_feature_indices() != [list(FEATURES).index(n) for n in CATEGORICAL]):
            raise ModelUnavailable('Model feature/class schema is incompatible')

    def predict_rows(self, rows):
        from catboost import Pool, CatBoostError
        if not rows:
            return []
        matrix = []
        for row in rows:
            values = []
            for name in FEATURES:
                value = row['features'][name]
                if name in CATEGORICAL:
                    if not isinstance(value, str) or not value:
                        raise ValueError('Invalid category')
                elif value is None:
                    value = float('nan')
                elif type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError('Invalid numeric feature')
                values.append(value)
            matrix.append(values)
        pool = Pool(matrix, feature_names=list(FEATURES), cat_features=list(CATEGORICAL))
        try:
            values = [float(p) for p in self.model.predict_proba(pool, thread_count=2)[:, 1]]
        except CatBoostError as exc:
            raise ModelUnavailable('Inference failed') from exc
        if len(values) != len(rows) or any(not math.isfinite(p) or not 0 <= p <= 1 for p in values):
            raise ModelUnavailable('Model produced invalid probabilities')
        return values


def result(service_id, state=None, status='insufficient_data', reason='INCOMPLETE_HISTORY'):
    due = state.get('dueAt') if state else None
    return {'serviceId': service_id, 'accountId': state['accountId'] if state else None,
            'dueAt': iso(due) if due else None, 'outcomeDeadline': iso(due + GRACE) if due else None,
            'status': status, 'reasonCode': reason, 'lossProbability': None, 'warnings': []}


def history_issues(events):
    """Check wallet/ownership consistency; never silently fill missing payment history."""
    accounts, services, issues = {}, {}, {}
    def issue(aid, reason):
        if issues.get(aid) != 'INCONSISTENT_DATA':
            issues[aid] = reason
    for event in events:
        aid, sid, kind = event['accountId'], event['serviceId'], event['type']
        if kind == 'account_created':
            if aid in accounts:
                issue(aid, 'INCONSISTENT_DATA')
            else:
                accounts[aid] = 0
            continue
        if aid not in accounts:
            issue(aid, 'INCOMPLETE_HISTORY')
            continue
        if kind == 'action_rejected':
            # A rejected purchase can legitimately refer to a VPS never created.
            continue
        if kind == 'service_created':
            if sid in services:
                issue(aid, 'INCONSISTENT_DATA')
                issue(services[sid]['owner'], 'INCONSISTENT_DATA')
            services[sid] = {'owner': aid, 'due': None, 'term': event['paymentTerm'],
                             'price': event['priceMinor'], 'status': 'active'}
        elif sid is not None and (sid not in services or services[sid]['owner'] != aid):
            issue(aid, 'INCOMPLETE_HISTORY' if sid not in services else 'INCONSISTENT_DATA')
            continue
        if kind in {'wallet_topped_up', 'payment_succeeded', 'payment_failed'}:
            before = accounts[aid]
            after = before + (event['amountMinor'] if kind == 'wallet_topped_up' else
                              -event['amountMinor'] if kind == 'payment_succeeded' else 0)
            if (event['balanceBeforeMinor'] != before or event['balanceAfterMinor'] != after or after < 0):
                issue(aid, 'INCONSISTENT_DATA')
            accounts[aid] = after
        if kind in {'payment_succeeded', 'payment_failed'}:
            service = services[sid]
            due = timestamp(event['dueAt'])
            if (event['paymentTerm'] != service['term'] or event['amountMinor'] != service['price']
                    or (event['mode'] != 'purchase' and service['due'] != due)
                    or (event['mode'] == 'purchase' and service['due'] is not None)
                    or service['status'] in {'blocked', 'deleted'}):
                issue(aid, 'INCONSISTENT_DATA')
            if kind == 'payment_succeeded':
                paid = timestamp(event['paidThrough'])
                if paid <= due or paid <= timestamp(event['occurredAt']):
                    issue(aid, 'INCONSISTENT_DATA')
                service.update(due=paid, status='active')
        elif kind in {'status_changed', 'service_restricted', 'service_released'}:
            services[sid]['status'] = event['toStatus']
            if kind == 'status_changed' and event['toStatus'] == 'prolong_wait':
                due = services[sid]['due']
                if due is not None and timestamp(event['occurredAt']) < due:
                    issue(aid, 'INCONSISTENT_DATA')
        elif kind == 'replacement_confirmed':
            other = services.get(event['replacementServiceId'])
            if not other or other['owner'] != aid:
                issue(aid, 'INCOMPLETE_HISTORY' if not other else 'INCONSISTENT_DATA')
    return issues


def predictions(predictor, request):
    source = request['input']
    ids = request['serviceIds']
    at = timestamp(request['predictionAt'])
    if source['kind'] == 'snapshot':
        by_id = {s['id']: s for s in source['snapshot']['snapshot']['services']}
        return [result(sid, {'accountId': by_id[sid]['ownerId'], 'dueAt': timestamp(by_id[sid]['expiresAt'])}
                       if sid in by_id else None, reason='SNAPSHOT_HAS_NO_HISTORY') for sid in ids]
    data = source['history']
    known = [e for e in data['events'] if timestamp(e['occurredAt']) <= at and timestamp(e['recordedAt']) <= at]
    if not known:
        return [result(sid, reason='INCOMPLETE_HISTORY') for sid in ids]
    issues = history_issues(known)
    # A broken account must not break inference for other accounts. Retain all
    # coherent accounts, not just selected service IDs, for portfolio features.
    usable = [e for e in known if e['accountId'] not in issues]
    state, rows = {}, {}
    if usable:
        history = EventHistory({**data, 'events': usable})
        state = history.replay(at)
        rows = {r['serviceId']: r for r in history.features_at(at, set(ids))}
    ownership = {e['serviceId']: e['accountId'] for e in known if e['type'] == 'service_created'}
    output, eligible, positions = [], [], []
    for sid in ids:
        service = state.get(sid)
        aid = ownership.get(sid)
        if aid in issues:
            reason = issues[aid]
            item = result(sid, {'accountId': aid, 'dueAt': None},
                          'invalid_data' if reason == 'INCONSISTENT_DATA' else 'insufficient_data', reason)
        elif service is None:
            item = result(sid, status='not_applicable', reason='SERVICE_NOT_FOUND')
        elif service['status'] == 'blocked':
            item = result(sid, service, 'not_applicable', 'NONFINANCIAL_RESTRICTION')
        elif service['status'] == 'deleted' or service['replaced']:
            item = result(sid, service, 'not_applicable', 'DELETED_OR_REPLACED')
        elif service['contract']['paymentTerm'] not in predictor.info['supportedPaymentTerms']:
            item = result(sid, service, 'not_applicable', 'UNSUPPORTED_PAYMENT_TERM')
        elif service['dueAt'] is None:
            item = result(sid, service)
        elif service['status'] == 'prolong_wait' or service['dueAt'] <= at:
            item = result(sid, service, 'not_applicable', 'OVERDUE_SERVICE')
        elif service['dueAt'] > at + HORIZON:
            item = result(sid, service, 'not_applicable', 'OUTSIDE_PREDICTION_WINDOW')
        elif sid not in rows:
            item = result(sid, service)
        else:
            item = result(sid, service, 'predicted', None)
            eligible.append(rows[sid])
            positions.append(len(output))
        output.append(item)
    for position, value in zip(positions, predictor.predict_rows(eligible)):
        output[position]['lossProbability'] = value
    return output
