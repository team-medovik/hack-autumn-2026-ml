"""Read-only FastAPI facade for the independently running generator worker."""
from contextlib import asynccontextmanager
import copy
import hmac
import logging
from typing import Annotated

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException
import yaml

from .state import ROOT, Settings, CheckpointError, load_checkpoint, status, snapshot, event_page

LOG = logging.getLogger("generator.api")
GENERATOR_PATHS = ("/health/live", "/health/ready", "/internal/v1/generator/status",
                   "/internal/v1/generator/events", "/internal/v1/generator/snapshot")


class APIError(Exception):
    def __init__(self, status_code, code, message):
        self.status_code, self.code, self.message = status_code, code, message


def error_response(status_code, code, message):
    return JSONResponse(status_code=status_code,
                        content={"code": code, "message": message, "requestId": None},
                        headers={"Cache-Control": "no-store"})


def create_app(settings=None):
    settings = settings or Settings.from_env()
    if not settings.token or len(settings.token.encode()) < 16:
        raise ValueError("Set GENERATOR_SERVICE_TOKEN to a secret of at least 16 bytes")

    @asynccontextmanager
    async def lifespan(app):
        # The worker is intentionally NOT started here: multiple HTTP processes are safe.
        yield

    app = FastAPI(title="Aeza live generator", lifespan=lifespan,
                  docs_url="/docs", redoc_url=None)
    spec = yaml.safe_load((ROOT / "contracts/openapi.yaml").read_text())
    spec["paths"] = {key: value for key, value in spec["paths"].items() if key in GENERATOR_PATHS}
    spec["info"]["title"] = "Aeza live generator — internal read API"
    spec["servers"] = [{"url": "http://generator-api:8000"}]
    app.openapi = lambda: copy.deepcopy(spec)

    @app.middleware("http")
    async def protect(request: Request, call_next):
        if request.url.path not in ("/health/live", "/health/ready"):
            supplied = request.headers.get("authorization", "").encode()
            if not hmac.compare_digest(supplied, ("Bearer " + settings.token).encode()):
                response = error_response(401, "UNAUTHORIZED", "Требуется service token")
                response.headers["WWW-Authenticate"] = "Bearer"
                return response
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(APIError)
    async def expected_error(request, error):
        return error_response(error.status_code, error.code, error.message)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, error):
        return error_response(422, "INVALID_REQUEST", "Проверьте обязательные параметры и их диапазоны")

    @app.exception_handler(HTTPException)
    async def http_error(request, error):
        return error_response(error.status_code, "HTTP_ERROR", "Ресурс или метод недоступен")

    def read(run_id=None):
        try:
            saved = load_checkpoint(settings.checkpoint)
        except (OSError, CheckpointError):
            LOG.warning("Committed generator state unavailable")
            raise APIError(503, "STATE_UNAVAILABLE", "Сохранённое состояние генератора недоступно") from None
        if run_id is not None and run_id != saved.run_id:
            raise APIError(404, "RUN_NOT_FOUND", "Запуск не найден")
        return saved

    @app.get("/health/live")
    def live():
        return {"status": "alive"}

    @app.get("/health/ready")
    def ready():
        try:
            read()
        except APIError:
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        # Readiness means ability to serve committed data, including a paused run.
        return {"status": "ready"}

    @app.get("/internal/v1/generator/status")
    def generator_status():
        saved = read()
        try:
            return status(settings, saved)
        except OSError:
            raise APIError(503, "STATE_UNAVAILABLE", "Состояние генератора недоступно") from None

    @app.get("/internal/v1/generator/events")
    def events(runId: Annotated[str, Query(min_length=1)],
               afterSequence: Annotated[int, Query(ge=0)],
               limit: Annotated[int, Query(ge=1, le=1000)] = 100):
        saved = read(runId)
        if afterSequence > len(saved.world.sim.events):
            raise APIError(409, "CURSOR_AHEAD", "Курсор находится за концом журнала")
        return event_page(saved, afterSequence, limit)

    @app.get("/internal/v1/generator/snapshot")
    def current_snapshot(runId: Annotated[str, Query(min_length=1)],
                         expectedRevision: Annotated[int | None, Query(ge=0)] = None):
        saved = read(runId)
        if expectedRevision is not None and expectedRevision != saved.revision:
            raise APIError(409, "REVISION_CONFLICT", "Ревизия состояния изменилась")
        return snapshot(saved)

    return app
