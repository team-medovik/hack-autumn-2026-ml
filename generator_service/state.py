"""Read committed engine state; HTTP readers never advance the simulation."""
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
import re
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples/risk/event-driven"))
from checkpoint import load_checkpoint, CheckpointError  # noqa: E402
from billing import iso, instant  # noqa: E402


def utc_now():
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Settings:
    checkpoint: Path
    token: str = ""
    time_scale: float = 60.0
    tick_seconds: float = 1.0
    max_step_seconds: float = 3600.0
    seed: int = 20261024
    accounts: int = 50
    days: int = 365
    start: str = "2026-01-01T00:00:00Z"
    database_url: str = ""
    run_key: str = "default"

    def __post_init__(self):
        object.__setattr__(self, "checkpoint", Path(self.checkpoint).absolute())
        for value in (self.time_scale, self.tick_seconds, self.max_step_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Clock settings must be positive and finite")
        if not 1 <= self.accounts <= 10000 or not 1 <= self.days <= 3650:
            raise ValueError("Invalid population or simulation horizon")
        instant(self.start)
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", self.run_key):
            raise ValueError("Invalid generator run key")
        if self.database_url and not self.database_url.startswith(("postgresql://", "postgres://")):
            raise ValueError("GENERATOR_DATABASE_URL must be a PostgreSQL URL")

    @property
    def heartbeat(self):
        return Path(str(self.checkpoint) + ".worker.json")

    @classmethod
    def from_env(cls):
        return cls(
            checkpoint=Path(os.environ.get("GENERATOR_CHECKPOINT", "runtime/live.json")),
            token=os.environ.get("GENERATOR_SERVICE_TOKEN", ""),
            time_scale=float(os.environ.get("GENERATOR_TIME_SCALE", "60")),
            tick_seconds=float(os.environ.get("GENERATOR_TICK_SECONDS", "1")),
            max_step_seconds=float(os.environ.get("GENERATOR_MAX_STEP_SECONDS", "3600")),
            seed=int(os.environ.get("GENERATOR_SEED", "20261024")),
            accounts=int(os.environ.get("GENERATOR_ACCOUNTS", "50")),
            days=int(os.environ.get("GENERATOR_DAYS", "365")),
            start=os.environ.get("GENERATOR_START", "2026-01-01T00:00:00Z"),
            database_url=os.environ.get("GENERATOR_DATABASE_URL", ""),
            run_key=os.environ.get("GENERATOR_RUN_KEY", "default"),
        )


def write_json_atomic(path, data):
    """Only worker telemetry; authoritative billing state uses checkpoint.py."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".worker-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(data, allow_nan=False).encode())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def status(settings, checkpoint):
    core = checkpoint.public_status()
    heartbeat = {}
    try:
        with settings.heartbeat.open() as handle:
            heartbeat = json.load(handle)
        if heartbeat.get("runId") != checkpoint.run_id:
            heartbeat = {}
    except (OSError, ValueError):
        pass
    state = heartbeat.get("workerState", "starting")
    stamp = heartbeat.get("lastHeartbeatAt")
    if state == "running":
        try:
            age = (utc_now() - instant(stamp)).total_seconds()
            if age > max(10, float(heartbeat["tickSeconds"]) * 3) or age < -5:
                state = "error"
        except (ValueError, TypeError, KeyError):
            state = "error"
    if core["finished"]:
        state = "finished"
    # Checkpoint can be newer than telemetry after a crash between the two writes.
    # The public revision always comes from the authoritative checkpoint.
    return {"generator": core, "workerState": state, "mode": "live_simulation",
            "timeScale": heartbeat.get("timeScale", settings.time_scale),
            "lastHeartbeatAt": stamp,
            "lastCheckpointAt": iso(datetime.fromtimestamp(settings.checkpoint.stat().st_mtime, timezone.utc)),
            "syntheticOnly": True}


def snapshot(checkpoint):
    return {"schemaVersion": "risk-snapshot-envelope-v1", "runId": checkpoint.run_id,
            "revision": checkpoint.revision, "asOf": iso(checkpoint.world.sim.now),
            "syntheticOnly": True, "currency": "EUR", "transactionCoverage": "omitted",
            "snapshot": checkpoint.world.sim.snapshot()}


def event_page(checkpoint, after, limit):
    journal = checkpoint.world.sim.events
    selected = journal[after:after + limit]
    return {"runId": checkpoint.run_id, "revision": checkpoint.revision,
            "asOf": iso(checkpoint.world.sim.now),
            "events": [{"sequence": after + index + 1, "event": event}
                       for index, event in enumerate(selected)],
            "nextSequence": after + len(selected), "hasMore": after + len(selected) < len(journal)}
