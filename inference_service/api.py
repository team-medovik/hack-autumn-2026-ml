"""Read-only internal HTTP API for the trained CatBoost artifact."""
from dataclasses import dataclass
import copy
import hmac
import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from jsonschema import Draft202012Validator, FormatChecker, ValidationError
import yaml

from .model import ROOT, Predictor, ModelUnavailable, predictions, timestamp

LOG = logging.getLogger('inference.api')
PATHS = ('/health/live', '/health/ready', '/internal/v1/model', '/internal/v1/renewal-risk/predict')


@dataclass(frozen=True)
class Settings:
    token: str
    model_path: Path = ROOT / 'inference_service/artifacts/model.cbm'
    manifest_path: Path = ROOT / 'inference_service/artifacts/manifest.json'
    max_body_bytes: int = 32 * 1024 * 1024

    def __post_init__(self):
        if len(self.token.encode()) < 16 or '\n' in self.token or '\r' in self.token:
            raise ValueError('ML_SERVICE_TOKEN must contain at least 16 bytes')
        if not 1024 <= self.max_body_bytes <= 64 * 1024 * 1024:
            raise ValueError('Invalid maximum request size')

    @classmethod
    def from_env(cls):
        return cls(os.environ.get('ML_SERVICE_TOKEN', ''),
                   Path(os.environ.get('ML_MODEL_PATH', str(cls.model_path))),
                   Path(os.environ.get('ML_MODEL_MANIFEST_PATH', str(cls.manifest_path))),
                   int(os.environ.get('ML_MAX_BODY_BYTES', 32 * 1024 * 1024)))


class InputError(ValueError):
    pass


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InputError('Duplicate JSON key')
        result[key] = value
    return result


def error(code, message, status_code, request_id=None):
    return JSONResponse(status_code=status_code, content={
        'code': code, 'message': message, 'requestId': request_id}, headers={'Cache-Control': 'no-store'})


def validators(spec):
    # Event allOf contains one branch for each kind. Compile only the matching
    # branch per event; validating all branches for 100k events is unnecessary.
    components = copy.deepcopy(spec['components'])
    event = components['schemas']['Event']
    by_type = {}
    for kind in event['properties']['type']['enum']:
        schema = {k: copy.deepcopy(v) for k, v in event.items() if k != 'allOf'}
        for branch in event['allOf']:
            if branch['if']['properties']['type']['const'] == kind:
                schema['required'] = list(set(schema['required'] + branch['then'].get('required', [])))
                for key, value in branch['then'].get('properties', {}).items():
                    schema['properties'][key].update(value)
        by_type[kind] = Draft202012Validator(schema, format_checker=FormatChecker())
    components['schemas']['History']['properties']['events']['items'] = {}
    request = Draft202012Validator({'$ref': '#/components/schemas/PredictRequest', 'components': components},
                                   format_checker=FormatChecker())
    return request, by_type


def check_request(data, request_validator, event_validators):
    request_validator.validate(data)
    source = data['input']
    at = timestamp(data['predictionAt'])
    if source['kind'] == 'history':
        history = source['history']
        watermark = timestamp(history['asOf'])
        if at > watermark:
            raise InputError('Prediction after history watermark')
        seen, previous = set(), None
        for event in history['events']:
            if not isinstance(event, dict) or event.get('type') not in event_validators:
                raise InputError('Unknown event')
            event_validators[event['type']].validate(event)
            occurred, recorded = timestamp(event['occurredAt']), timestamp(event['recordedAt'])
            if (event['id'] in seen or recorded < occurred or recorded > watermark
                    or (previous is not None and occurred < previous)):
                raise InputError('Invalid event identity/order/cutoff')
            seen.add(event['id'])
            previous = occurred
    elif source['kind'] == 'snapshot':
        if at != timestamp(source['snapshot']['asOf']):
            raise InputError('Snapshot time differs from prediction time')
    elif source['kind'] == 'dataset' and source['revision'] != data['snapshotRevision']:
        raise InputError('Dataset revision differs')


def create_app(settings=None):
    settings = settings or Settings.from_env()
    spec = yaml.safe_load((ROOT / 'contracts/openapi.yaml').read_text())
    request_validator, event_validators = validators(spec)
    predictor = None
    try:
        predictor = Predictor(settings.model_path, settings.manifest_path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        LOG.error('ML artifact unavailable (%s)', type(exc).__name__)
    app = FastAPI(title='Aeza ML inference', docs_url='/docs', redoc_url=None)
    public_spec = copy.deepcopy(spec)
    public_spec['paths'] = {key: value for key, value in spec['paths'].items() if key in PATHS}
    public_spec['info']['title'] = 'Aeza ML inference — internal API'
    public_spec['servers'] = [{'url': 'http://ml-service:8000'}]
    app.openapi = lambda: copy.deepcopy(public_spec)

    @app.middleware('http')
    async def protect(request, call_next):
        if request.url.path not in ('/health/live', '/health/ready'):
            supplied = request.headers.get('authorization', '').encode()
            if not hmac.compare_digest(supplied, ('Bearer ' + settings.token).encode()):
                response = error('UNAUTHORIZED', 'Требуется service token', 401)
                response.headers['WWW-Authenticate'] = 'Bearer'
                return response
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.get('/health/live')
    def live():
        return {'status': 'alive'}

    @app.get('/health/ready')
    def ready():
        return JSONResponse({'status': 'ready' if predictor else 'not_ready'}, status_code=200 if predictor else 503)

    @app.get('/internal/v1/model')
    def model():
        if predictor is None:
            return error('MODEL_UNAVAILABLE', 'Модель недоступна; используйте алгоритм', 503)
        return copy.deepcopy(predictor.info)

    @app.post('/internal/v1/renewal-risk/predict')
    async def predict(request: Request):
        body = bytearray()
        request_id = None
        try:
            async for part in request.stream():
                if len(body) + len(part) > settings.max_body_bytes:
                    return error('INVALID_REQUEST', 'Превышен допустимый размер запроса', 422)
                body.extend(part)
            data = json.loads(body, object_pairs_hook=unique_object,
                              parse_constant=lambda _: (_ for _ in ()).throw(InputError('Non-finite JSON number')))
            if isinstance(data, dict) and isinstance(data.get('requestId'), str):
                request_id = data['requestId'][:256]
            await run_in_threadpool(check_request, data, request_validator, event_validators)
        except (ValueError, TypeError, KeyError, ValidationError, RecursionError):
            return error('INVALID_REQUEST', 'Проверьте контракт и временные границы данных', 422, request_id)
        if predictor is None:
            return error('MODEL_UNAVAILABLE', 'Модель недоступна; используйте алгоритм', 503, request_id)
        if data['input']['kind'] == 'dataset':
            return error('DATASET_UNAVAILABLE', 'Источник MinIO пока не подключён; передайте историю событий', 503, request_id)
        try:
            # Run CPU work in the shared bounded threadpool, not the asyncio loop.
            values = await run_in_threadpool(predictions, predictor, data)
        except (ValueError, KeyError, TypeError, OverflowError):
            LOG.warning('ML input could not be evaluated')
            return error('INFERENCE_UNAVAILABLE', 'Прогноз недоступен; используйте алгоритм', 503, request_id)
        return {key: data[key] for key in ('requestId', 'contractVersion', 'snapshotRevision', 'predictionAt')} | {
            'modelVersion': predictor.info['modelVersion'], 'featureSchemaVersion': predictor.info['featureSchemaVersion'],
            'targetVersion': predictor.info['targetVersion'], 'syntheticOnly': True, 'predictions': values}

    return app
