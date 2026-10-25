"""Private, versioned JSON checkpoints for a single-host simulation worker.

No pickle, dynamic imports, or executable payloads. A checksum detects corruption,
not malicious modification: files belong on a trusted private persistent volume.
Only the committed event journal may be published; transport/outbox is a later layer.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import uuid

from behavior import BehaviorSimulation, make_population
from billing import BillingSimulator, apply_event, instant, iso

SCHEMA = "generator-checkpoint-v1"
MAX_BYTES = 128 * 1024 * 1024
BEHAVIOR_FIELDS = frozenset((
    "start", "end", "population", "spent", "pending_funding", "needs", "owner", "metadata",
    "queue", "decisions", "sequence", "cursor", "planned_cycles", "noticed_cycles", "replaced",
    "pending_switches", "manual_commands",
))
BILLING_FIELDS = frozenset((
    "start_at", "now", "accounts", "services", "events", "_queue", "_commands",
    "_planned_services", "_registrations", "_sequence", "_ran", "_cause",
))


class CheckpointError(ValueError):
    pass


class RevisionConflict(CheckpointError):
    pass


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def engine_fingerprint():
    # Refuse silent continuation under different billing/behavior/restore code.
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for name in ("billing.py", "behavior.py", "checkpoint.py"):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def encode(value):
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, datetime):
        return {"type": "datetime", "value": iso(value)}
    if isinstance(value, dict):
        return {"type": "dict", "value": [[encode(k), encode(v)] for k, v in value.items()]}
    if isinstance(value, (list, tuple, set)):
        values = [encode(v) for v in value]
        if isinstance(value, set):
            values.sort(key=canonical)
        return {"type": type(value).__name__, "value": values}
    raise CheckpointError(f"Unsupported state type: {type(value).__name__}")


def decode(value):
    if value is None or type(value) in (str, int, float, bool):
        return value
    if not isinstance(value, dict) or set(value) != {"type", "value"}:
        raise CheckpointError("Invalid tagged state")
    kind, data = value["type"], value["value"]
    if kind == "datetime":
        return instant(data)
    if not isinstance(data, list):
        raise CheckpointError("Invalid collection state")
    if kind == "dict":
        pairs = [(decode(k), decode(v)) for k, v in data]
        result = dict(pairs)
        if len(result) != len(pairs):
            raise CheckpointError("Duplicate state keys")
        return result
    values = [decode(v) for v in data]
    if kind == "list":
        return values
    if kind == "tuple":
        return tuple(values)
    if kind == "set":
        return set(values)
    raise CheckpointError("Unknown state tag")


def state_of(world):
    if set(vars(world)) != BEHAVIOR_FIELDS | {"sim", "random", "profiles"}:
        raise CheckpointError("Behavior state fields changed: update checkpoint schema")
    if set(vars(world.sim)) != BILLING_FIELDS:
        raise CheckpointError("Billing state fields changed: update checkpoint schema")
    return {
        "behavior": {k: getattr(world, k) for k in sorted(BEHAVIOR_FIELDS)},
        "billing": {k: getattr(world.sim, k) for k in sorted(BILLING_FIELDS)},
        "random": {aid: rng.getstate() for aid, rng in world.random.items()},
    }


def restore(state):
    if set(state) != {"behavior", "billing", "random"}:
        raise CheckpointError("Invalid checkpoint sections")
    if set(state["behavior"]) != BEHAVIOR_FIELDS or set(state["billing"]) != BILLING_FIELDS:
        raise CheckpointError("Invalid checkpoint fields")
    world = object.__new__(BehaviorSimulation)
    world.__dict__.update(state["behavior"])
    world.sim = object.__new__(BillingSimulator)
    world.sim.__dict__.update(state["billing"])
    world.profiles = {p["accountId"]: p for p in world.population}
    if set(state["random"]) != set(world.profiles):
        raise CheckpointError("Missing random stream")
    world.random = {}
    for aid, rng_state in state["random"].items():
        rng = random.Random(0)
        rng.setstate(rng_state)
        world.random[aid] = rng
    world.spent = defaultdict(int, world.spent)
    if not world.start == world.sim.start_at <= world.sim.now <= world.end:
        raise CheckpointError("Invalid simulation clock")
    if world.cursor != len(world.sim.events):
        raise CheckpointError("Checkpoint is not at a completed timestamp boundary")
    accounts, services = {}, {}
    previous = world.start
    for i, event in enumerate(world.sim.events, 1):
        occurred, recorded = instant(event["occurredAt"]), instant(event["recordedAt"])
        if (event["id"] != f"billing-{i:07d}" or
                not previous <= occurred <= recorded <= world.sim.now):
            raise CheckpointError("Invalid event order, ID, or cutoff")
        previous = occurred
        apply_event(accounts, services, event)
    if accounts != world.sim.accounts or services != world.sim.services:
        raise CheckpointError("Balances/services disagree with the committed journal")
    for queue in (world.queue, world.sim._queue):
        for i, task in enumerate(queue):
            if task[0] < world.sim.now or (world.sim._ran and task[0] == world.sim.now):
                raise CheckpointError("Unprocessed task before checkpoint boundary")
            if i and task < queue[(i - 1) // 2]:
                raise CheckpointError("Invalid task heap")
    return world


@dataclass
class Checkpoint:
    run_id: str
    revision: int
    world: BehaviorSimulation

    def public_status(self):
        return {"runId": self.run_id, "revision": self.revision,
                "virtualTime": iso(self.world.sim.now), "endAt": iso(self.world.end),
                "nextEventAt": iso(self.world.next_event_at) if self.world.next_event_at else None,
                "eventCount": len(self.world.sim.events), "finished": self.world.finished}


def document(checkpoint):
    result = {"schemaVersion": SCHEMA, "engineFingerprint": engine_fingerprint(),
              "pythonVersion": list(sys.version_info[:2]), "runId": checkpoint.run_id,
              "revision": checkpoint.revision, "state": encode(state_of(checkpoint.world))}
    result["checksumSha256"] = hashlib.sha256(canonical(result)).hexdigest()
    return result


def load_checkpoint(path):
    path = Path(path)
    if path.is_symlink():
        raise CheckpointError("Checkpoint path must not be a symlink")
    with path.open("rb") as handle:
        raw = handle.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise CheckpointError("Checkpoint exceeds size limit")
    try:
        data = json.loads(raw)
        if set(data) != {"schemaVersion", "engineFingerprint", "pythonVersion", "runId", "revision", "state", "checksumSha256"}:
            raise CheckpointError("Invalid checkpoint envelope")
        checksum = data.pop("checksumSha256")
        if checksum != hashlib.sha256(canonical(data)).hexdigest():
            raise CheckpointError("Checkpoint checksum mismatch")
        if data["schemaVersion"] != SCHEMA or data["engineFingerprint"] != engine_fingerprint():
            raise CheckpointError("Incompatible checkpoint schema/engine; explicit migration required")
        if data["pythonVersion"] != list(sys.version_info[:2]):
            raise CheckpointError("Checkpoint requires the same Python major/minor version")
        if (not isinstance(data["runId"], str) or not data["runId"] or
                type(data["revision"]) is not int or data["revision"] < 0):
            raise CheckpointError("Invalid run identity/revision")
        return Checkpoint(data["runId"], data["revision"], restore(decode(data["state"])))
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError) as error:
        raise CheckpointError(str(error)) from error


@contextmanager
def writer_lock(path):
    # All writers must use this API, on one host/local POSIX persistent volume.
    lock_path = str(path) + ".lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def atomic_write(path, checkpoint):
    path = Path(path)
    if path.is_symlink():
        raise CheckpointError("Checkpoint path must not be a symlink")
    payload = canonical(document(checkpoint)) + b"\n"
    if len(payload) > MAX_BYTES:
        raise CheckpointError("Checkpoint exceeds size limit")
    temp = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".checkpoint-", suffix=".tmp", dir=path.parent,
                                         delete=False) as handle:
            temp = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temp is not None and temp.exists():
            temp.unlink()


def create_checkpoint(world, path, run_id=None):
    path = Path(path)
    # Validate a complete, restorable boundary before writing private state.
    restore(decode(encode(state_of(world))))
    checkpoint = Checkpoint(run_id or str(uuid.uuid4()), 0, world)
    if not isinstance(checkpoint.run_id, str) or not checkpoint.run_id:
        raise CheckpointError("Nonempty run ID required")
    with writer_lock(path):
        if path.exists() or path.is_symlink():
            raise FileExistsError("Checkpoint already exists; use advance to resume")
        atomic_write(path, checkpoint)
    return checkpoint


def advance_checkpoint(path, until=None, expected_revision=None):
    """Read-modify-commit under one lock. Never publish uncommitted events.

    Repeating an absolute `until` at the committed boundary is idempotent.
    A stale expected_revision fails explicitly; retries of `step` need this guard.
    """
    with writer_lock(path):
        checkpoint = load_checkpoint(path)
        if expected_revision is not None and (type(expected_revision) is not int or
                                              expected_revision != checkpoint.revision):
            raise RevisionConflict("Checkpoint revision changed")
        before = canonical(encode(state_of(checkpoint.world)))
        if until is None:
            checkpoint.world.step()
        else:
            checkpoint.world.advance(until)
        after = canonical(encode(state_of(checkpoint.world)))
        if before != after:
            checkpoint.revision += 1
            atomic_write(path, checkpoint)
        return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    actions = parser.add_subparsers(dest="action", required=True)
    create = actions.add_parser("create")
    create.add_argument("--seed", type=int, default=20261024)
    create.add_argument("--accounts", type=int, default=50)
    create.add_argument("--days", type=int, default=1095)
    create.add_argument("--start", default="2026-01-01T00:00:00Z")
    create.add_argument("--run-id")
    advance = actions.add_parser("advance")
    advance.add_argument("--until", help="UTC timestamp; omitted means one next timestamp batch")
    advance.add_argument("--expected-revision", type=int)
    actions.add_parser("status")
    args = parser.parse_args()
    if args.action == "create":
        population = make_population(args.seed, args.accounts, start_at=args.start,
                                     days=args.days, lifecycle=True)
        result = create_checkpoint(BehaviorSimulation(population, args.seed, args.start, args.days),
                                   args.checkpoint, args.run_id)
    elif args.action == "advance":
        result = advance_checkpoint(args.checkpoint, args.until, args.expected_revision)
    else:
        result = load_checkpoint(args.checkpoint)
    print(json.dumps(result.public_status(), ensure_ascii=False))


if __name__ == "__main__":
    main()
