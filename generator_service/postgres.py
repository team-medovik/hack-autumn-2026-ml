"""Transactional persistence for the existing deterministic simulation kernel.

The run row is the serialization boundary. Pending tasks, RNG/state, newly emitted
events and the virtual clock commit together. A heap is only a transaction-local
execution structure; PostgreSQL owns the durable queue. Historical events are
never reloaded during an ordinary tick or public read.
"""
from collections import defaultdict
from datetime import timedelta
import heapq
import json
from pathlib import Path
import random
import sys
import uuid

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples/risk/event-driven"))
from behavior import BehaviorSimulation, make_population
from billing import BillingSimulator, instant, iso
from checkpoint import (BEHAVIOR_FIELDS, BILLING_FIELDS, CheckpointError,
                        RevisionConflict, canonical, decode, encode,
                        engine_fingerprint, load_checkpoint)

SCHEMA = "generator-postgres-v1"
BEHAVIOR_OMITTED = {"queue", "decisions", "planned_cycles", "noticed_cycles"}
BILLING_OMITTED = {"events", "_queue", "_commands"}


class StoreUnavailable(CheckpointError):
    pass


class RunNotFound(CheckpointError):
    pass


class CursorAhead(CheckpointError):
    pass


class SpeedRunConflict(CheckpointError):
    pass


class RecentJournal:
    """Absolute indexes, backed only by events emitted in this transaction."""
    def __init__(self, offset):
        self.offset, self.pending = offset, []

    def __len__(self):
        return self.offset + len(self.pending)

    def append(self, event):
        self.pending.append(event)

    def __getitem__(self, key):
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise CheckpointError("Only recent journal slices are available to the live engine")
        start, stop, _ = key.indices(len(self))
        if start < self.offset:
            raise CheckpointError("Engine attempted to read historical events during a tick")
        return self.pending[start - self.offset:stop - self.offset]


class CommandMap:
    """Persistent command idempotency without loading all previous commands."""
    def __init__(self, connection, run_id):
        self.connection, self.run_id, self.pending = connection, run_id, {}

    def __contains__(self, key):
        return key in self.pending or self.connection.execute(
            "SELECT 1 FROM generator_sim.commands WHERE run_id=%s AND command_id=%s",
            (self.run_id, key)).fetchone() is not None

    def __getitem__(self, key):
        if key in self.pending:
            return self.pending[key]
        row = self.connection.execute(
            "SELECT payload FROM generator_sim.commands WHERE run_id=%s AND command_id=%s",
            (self.run_id, key)).fetchone()
        if row is None:
            raise KeyError(key)
        return decode(row["payload"])

    def __setitem__(self, key, value):
        self.pending[key] = value


class CycleMarks:
    """Append-only membership set, kept outside the current-state document."""
    def __init__(self, connection, run_id, kind):
        self.connection, self.run_id, self.kind = connection, run_id, kind
        self.pending = set()

    @staticmethod
    def key(value):
        return canonical(encode(value)).decode()

    def __contains__(self, value):
        key = self.key(value)
        return key in self.pending or self.connection.execute(
            "SELECT 1 FROM generator_sim.cycle_marks WHERE run_id=%s AND kind=%s AND mark=%s",
            (self.run_id, self.kind, key)).fetchone() is not None

    def add(self, value):
        self.pending.add(self.key(value))


def private_state(world):
    if set(vars(world)) != BEHAVIOR_FIELDS | {"sim", "random", "profiles"}:
        raise CheckpointError("Behavior fields changed; migrate PostgreSQL state explicitly")
    if set(vars(world.sim)) != BILLING_FIELDS:
        raise CheckpointError("Billing fields changed; migrate PostgreSQL state explicitly")
    return encode({
        "behavior": {k: getattr(world, k) for k in sorted(BEHAVIOR_FIELDS - BEHAVIOR_OMITTED)},
        "billing": {k: getattr(world.sim, k) for k in sorted(BILLING_FIELDS - BILLING_OMITTED)},
        "random": {aid: rng.getstate() for aid, rng in world.random.items()},
    })


def task_map(world):
    result = {}
    for queue, tasks in (("behavior", world.queue), ("billing", world.sim._queue)):
        for task in tasks:
            seq = task[1] if queue == "behavior" else task[3]
            result[(queue, seq)] = task
    return result


def projected_snapshot(state):
    billing = decode(state)["billing"]
    excluded = {"tariffCode", "lastManualPaymentAt", "replacementServiceId",
                "replacesServiceId", "restrictionReason", "unpaidSince"}
    return {
        "accounts": [billing["accounts"][key] for key in sorted(billing["accounts"])],
        "services": [{k: v for k, v in billing["services"][sid].items() if k not in excluded}
                     for sid in sorted(billing["services"])],
        "transactions": [],
    }


class PostgresStore:
    def __init__(self, dsn, run_key="default"):
        if not isinstance(run_key, str) or not run_key or len(run_key) > 100:
            raise ValueError("run_key must be a non-empty string of at most 100 characters")
        self.dsn, self.run_key = dsn, run_key

    def connect(self):
        return psycopg.connect(self.dsn, row_factory=dict_row, connect_timeout=10,
                               options="-c statement_timeout=30000 -c lock_timeout=5000 "
                                       "-c idle_in_transaction_session_timeout=60000")

    def migrate(self):
        with self.connect() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(641026001)")
            conn.execute((Path(__file__).parent / "migrations/001_simulation.sql").read_text())

    @staticmethod
    def compatible(row):
        if (row["schema_version"] != SCHEMA or row["engine_fingerprint"] != engine_fingerprint()
                or row["python_version"] != f"{sys.version_info.major}.{sys.version_info.minor}"):
            raise CheckpointError("Incompatible simulation schema/engine/Python; explicit migration required")

    def row(self, conn, run_id=None, lock=False, include_state=False):
        columns = ("run_key,run_id,revision,virtual_time,end_at,event_count,decision_count,"
                   "schema_version,engine_fingerprint,python_version,next_tick_at,"
                   "clock_config,heartbeat,updated_at")
        if lock or include_state:
            columns += ",state"
        sql = f"SELECT {columns} FROM generator_sim.runs WHERE run_key=%s"
        if lock:
            sql += " FOR UPDATE SKIP LOCKED"
        row = conn.execute(sql, (self.run_key,)).fetchone()
        if row is None:
            if lock and conn.execute("SELECT 1 FROM generator_sim.runs WHERE run_key=%s",
                                     (self.run_key,)).fetchone():
                return None
            raise StoreUnavailable("Simulation has not been initialized")
        if run_id is not None and row["run_id"] != run_id:
            raise RunNotFound("Run not found")
        self.compatible(row)
        return row

    @staticmethod
    def read_snapshot(conn):
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")

    def ensure_run(self, settings):
        with self.connect() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 641026002))", (self.run_key,))
            row = conn.execute("SELECT * FROM generator_sim.runs WHERE run_key=%s FOR UPDATE", (self.run_key,)).fetchone()
            if row:
                self.compatible(row)
                self.bind_clock(conn, row, settings)
                return row["run_id"]
            population = make_population(settings.seed, settings.accounts, start_at=settings.start,
                                         days=settings.days, lifecycle=True, terms=settings.payment_terms)
            world = BehaviorSimulation(population, settings.seed, settings.start, settings.days)
            run_id = str(uuid.uuid4())
            self.insert_world(conn, world, run_id, 0)
            self.bind_clock(conn, self.row(conn), settings)
            return run_id

    def bind_clock(self, conn, row, settings):
        config = {"timeScale": settings.time_scale, "tickSeconds": settings.tick_seconds,
                  "maxStepSeconds": settings.max_step_seconds}
        if row["clock_config"] is None:
            conn.execute("UPDATE generator_sim.runs SET clock_config=%s WHERE run_key=%s",
                         (Jsonb(config), self.run_key))
            return config
        saved = row["clock_config"]
        if any(saved[key] != config[key] for key in ("tickSeconds", "maxStepSeconds")):
            raise CheckpointError("Worker tick settings differ from persisted run settings")
        # timeScale is operator-managed state. A restarting worker's environment
        # supplies only the initial value and must never reset the saved speed.
        return saved

    def set_speed(self, run_id, time_scale):
        """Change only the speed; serialize with ticks without advancing the run."""
        if type(time_scale) is not int or time_scale not in (1, 10, 60):
            raise ValueError("Supported speeds are 1, 10 and 60")
        with self.connect() as conn:
            # Unlike background workers, a control request waits for the current
            # step to finish (bounded by lock_timeout), never skips the run lock.
            locked = conn.execute("SELECT run_key FROM generator_sim.runs WHERE run_key=%s FOR UPDATE",
                                  (self.run_key,)).fetchone()
            if locked is None:
                raise StoreUnavailable("Simulation has not been initialized")
            row = self.row(conn)
            if row["run_id"] != run_id:
                raise SpeedRunConflict("The requested simulation is no longer current")
            if row["clock_config"] is None:
                raise StoreUnavailable("Worker clock has not been initialized")
            clock = dict(row["clock_config"], timeScale=time_scale)
            conn.execute("UPDATE generator_sim.runs SET clock_config=%s WHERE run_key=%s",
                         (Jsonb(clock), self.run_key))
            return self.status_in(conn, self.row(conn))

    def import_checkpoint(self, path):
        saved = load_checkpoint(path)
        with self.connect() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 641026002))", (self.run_key,))
            if conn.execute("SELECT 1 FROM generator_sim.runs WHERE run_key=%s OR run_id=%s",
                            (self.run_key, saved.run_id)).fetchone():
                raise FileExistsError("Run key or checkpoint runId already exists; import never overwrites a run")
            self.insert_world(conn, saved.world, saved.run_id, saved.revision)
        return saved.run_id

    def insert_world(self, conn, world, run_id, revision):
        conn.execute("""INSERT INTO generator_sim.runs
            (run_key,run_id,revision,virtual_time,end_at,event_count,decision_count,state,
             schema_version,engine_fingerprint,python_version)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (self.run_key, run_id, revision, world.sim.now, world.end, len(world.sim.events),
             len(world.decisions), Jsonb(private_state(world)), SCHEMA, engine_fingerprint(),
             f"{sys.version_info.major}.{sys.version_info.minor}"))
        self.save_tasks(conn, run_id, {}, task_map(world))
        self.insert_events(conn, run_id, 0, world.sim.events)
        self.insert_decisions(conn, run_id, 0, world.decisions)
        for key, command in world.sim._commands.items():
            conn.execute("INSERT INTO generator_sim.commands VALUES (%s,%s,%s)",
                         (run_id, key, Jsonb(encode(command))))
        for kind in ("planned_cycles", "noticed_cycles"):
            for mark in getattr(world, kind):
                conn.execute("INSERT INTO generator_sim.cycle_marks VALUES (%s,%s,%s)",
                             (run_id, kind, CycleMarks.key(mark)))

    def restore_world(self, conn, row):
        state = decode(row["state"])
        if (set(state) != {"behavior", "billing", "random"}
                or set(state["behavior"]) != BEHAVIOR_FIELDS - BEHAVIOR_OMITTED
                or set(state["billing"]) != BILLING_FIELDS - BILLING_OMITTED):
            raise CheckpointError("Invalid PostgreSQL private-state fields")
        world = object.__new__(BehaviorSimulation)
        world.__dict__.update(state["behavior"])
        world.sim = object.__new__(BillingSimulator)
        world.sim.__dict__.update(state["billing"])
        world.profiles = {p["accountId"]: p for p in world.population}
        if set(state["random"]) != set(world.profiles):
            raise CheckpointError("Missing client random stream")
        world.random = {}
        for aid, rng_state in state["random"].items():
            rng = random.Random(0)
            rng.setstate(rng_state)
            world.random[aid] = rng
        world.spent = defaultdict(int, world.spent)
        world.decisions, world.queue, world.sim._queue = [], [], []
        world.sim.events = RecentJournal(row["event_count"])
        world.sim._commands = CommandMap(conn, row["run_id"])
        world.planned_cycles = CycleMarks(conn, row["run_id"], "planned_cycles")
        world.noticed_cycles = CycleMarks(conn, row["run_id"], "noticed_cycles")
        if (world.cursor != row["event_count"] or world.sim.now != row["virtual_time"]
                or world.end != row["end_at"] or not world.start <= world.sim.now <= world.end):
            raise CheckpointError("State disagrees with committed run boundary")
        for item in conn.execute("SELECT * FROM generator_sim.tasks WHERE run_id=%s", (row["run_id"],)):
            task = decode(item["payload"])
            sequence = task[1] if item["queue"] == "behavior" else task[3]
            if (task[0] != item["due_at"] or sequence != item["sequence"]
                    or task[0] < world.sim.now or (world.sim._ran and task[0] == world.sim.now)):
                raise CheckpointError("Invalid persisted task boundary")
            (world.queue if item["queue"] == "behavior" else world.sim._queue).append(task)
        heapq.heapify(world.queue)
        heapq.heapify(world.sim._queue)
        return world

    @staticmethod
    def insert_events(conn, run_id, offset, events):
        for index, event in enumerate(events, offset + 1):
            if event["id"] != f"billing-{index:07d}":
                raise CheckpointError("Non-contiguous generated event identity")
            conn.execute("INSERT INTO generator_sim.events VALUES (%s,%s,%s,%s,%s)",
                         (run_id, index, event["id"], instant(event["occurredAt"]), Jsonb(event)))

    @staticmethod
    def insert_decisions(conn, run_id, offset, decisions):
        for index, decision in enumerate(decisions, offset + 1):
            conn.execute("INSERT INTO generator_sim.decisions VALUES (%s,%s,%s)",
                         (run_id, index, Jsonb(decision)))

    @staticmethod
    def save_tasks(conn, run_id, old, new):
        for queue, sequence in old.keys() - new.keys():
            conn.execute("DELETE FROM generator_sim.tasks WHERE run_id=%s AND queue=%s AND sequence=%s",
                         (run_id, queue, sequence))
        for key, task in new.items():
            if key in old:
                if old[key] != task:
                    raise CheckpointError("Pending task identity reused with changed payload")
                continue
            queue, sequence = key
            phase, tie = (0, 0) if queue == "behavior" else (task[1], task[2])
            conn.execute("INSERT INTO generator_sim.tasks VALUES (%s,%s,%s,%s,%s,%s,%s)",
                         (run_id, queue, sequence, task[0], phase, tie, Jsonb(encode(task))))

    def apply_step(self, conn, row, until):
        world = self.restore_world(conn, row)
        old_tasks = task_map(world)
        clock = dict(row["clock_config"] or {})
        from .stream import inject_due
        stream_until = until
        if stream_until is None and isinstance(clock.get("demoStreamV2"), dict):
            stream_until = instant(clock["demoStreamV2"]["nextServiceAt"])
        if stream_until is not None:
            inject_due(world, stream_until, clock)
        before = world.sim.now
        world.step() if until is None else world.advance(until)
        if world.sim.now == before and not world.sim.events.pending:
            return
        if world.cursor != len(world.sim.events):
            raise CheckpointError("Cannot commit incomplete timestamp")
        self.save_tasks(conn, row["run_id"], old_tasks, task_map(world))
        self.insert_events(conn, row["run_id"], row["event_count"], world.sim.events.pending)
        self.insert_decisions(conn, row["run_id"], row["decision_count"], world.decisions)
        for key, command in world.sim._commands.pending.items():
            conn.execute("INSERT INTO generator_sim.commands VALUES (%s,%s,%s)",
                         (row["run_id"], key, Jsonb(encode(command))))
        for kind in ("planned_cycles", "noticed_cycles"):
            for mark in getattr(world, kind).pending:
                conn.execute("INSERT INTO generator_sim.cycle_marks VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                             (row["run_id"], kind, mark))
        conn.execute("""UPDATE generator_sim.runs SET state=%s,virtual_time=%s,revision=revision+1,
            event_count=%s,decision_count=decision_count+%s,clock_config=%s,
            updated_at=CURRENT_TIMESTAMP WHERE run_key=%s""",
            (Jsonb(private_state(world)), world.sim.now, len(world.sim.events), len(world.decisions),
             Jsonb(clock), self.run_key))

    def advance(self, until=None, expected_revision=None):
        with self.connect() as conn:
            row = self.row(conn, lock=True)
            if row is None:
                return None
            if expected_revision is not None and (type(expected_revision) is not int or
                                                   expected_revision != row["revision"]):
                raise RevisionConflict("Simulation revision changed")
            self.apply_step(conn, row, until)
            return self.status_in(conn, self.row(conn))

    def tick(self, settings):
        """One fixed logical tick, paced by a DB timestamp shared by all workers.

        Downtime and slow processing never accumulate catch-up ticks. Contention
        skips a tick, never a task. Independent runs can advance concurrently.
        """
        with self.connect() as conn:
            row = self.row(conn, lock=True)
            if row is None:
                return None
            clock = self.bind_clock(conn, row, settings)
            now = conn.execute("SELECT CURRENT_TIMESTAMP AS now").fetchone()["now"]
            if row["next_tick_at"] > now:
                return None
            until = min(row["end_at"], row["virtual_time"] + timedelta(
                seconds=min(clock["tickSeconds"] * clock["timeScale"], clock["maxStepSeconds"])))
            if not self.generator_status(conn, row)["finished"]:
                self.apply_step(conn, row, until)
            fresh = self.row(conn)
            state = "finished" if self.generator_status(conn, fresh)["finished"] else "running"
            self.write_heartbeat(conn, state, settings)
            conn.execute("UPDATE generator_sim.runs SET next_tick_at=%s WHERE run_key=%s",
                         (now + timedelta(seconds=settings.tick_seconds), self.run_key))
            return self.status_in(conn, self.row(conn))

    def write_heartbeat(self, conn, state, settings):
        now = conn.execute("SELECT CURRENT_TIMESTAMP AS now").fetchone()["now"]
        clock = self.row(conn)["clock_config"]
        if clock is None:
            raise StoreUnavailable("Worker clock has not been initialized")
        heartbeat = {"workerState": state, "lastHeartbeatAt": iso(now),
                     "timeScale": clock["timeScale"], "tickSeconds": clock["tickSeconds"]}
        conn.execute("UPDATE generator_sim.runs SET heartbeat=%s WHERE run_key=%s",
                     (Jsonb(heartbeat), self.run_key))

    def heartbeat(self, state, settings):
        with self.connect() as conn:
            if self.row(conn, lock=True) is not None:
                self.write_heartbeat(conn, state, settings)

    @staticmethod
    def generator_status(conn, row):
        next_at = conn.execute("SELECT min(due_at) AS due FROM generator_sim.tasks WHERE run_id=%s AND due_at<=%s",
                               (row["run_id"], row["end_at"])).fetchone()["due"]
        return {"runId": row["run_id"], "revision": row["revision"],
                "virtualTime": iso(row["virtual_time"]), "endAt": iso(row["end_at"]),
                "nextEventAt": iso(next_at) if next_at else None, "eventCount": row["event_count"],
                "finished": row["virtual_time"] == row["end_at"] and next_at is None}

    def status_in(self, conn, row):
        core, heartbeat = self.generator_status(conn, row), row["heartbeat"]
        state = heartbeat.get("workerState", "starting")
        if state == "running":
            now = conn.execute("SELECT CURRENT_TIMESTAMP AS now").fetchone()["now"]
            age = (now - instant(heartbeat["lastHeartbeatAt"])).total_seconds()
            if age > max(10, heartbeat["tickSeconds"] * 3) or age < -5:
                state = "error"
        if core["finished"]:
            state = "finished"
        return {"generator": core, "workerState": state, "mode": "live_simulation",
                "timeScale": (row["clock_config"] or {}).get("timeScale", heartbeat.get("timeScale", 60)),
                "lastHeartbeatAt": heartbeat.get("lastHeartbeatAt"),
                "lastCheckpointAt": iso(row["updated_at"]), "syntheticOnly": True}

    def status(self):
        with self.connect() as conn:
            self.read_snapshot(conn)
            return self.status_in(conn, self.row(conn))

    def event_page(self, run_id, after, limit):
        if type(after) is not int or after < 0 or type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("Invalid event cursor or page size")
        with self.connect() as conn:
            self.read_snapshot(conn)
            row = self.row(conn, run_id)
            if after > row["event_count"]:
                raise CursorAhead("Cursor is beyond the committed event journal")
            items = conn.execute("""SELECT sequence,payload FROM generator_sim.events
                WHERE run_id=%s AND sequence>%s ORDER BY sequence LIMIT %s""", (run_id, after, limit)).fetchall()
            return {"runId": run_id, "revision": row["revision"], "asOf": iso(row["virtual_time"]),
                    "events": [{"sequence": item["sequence"], "event": item["payload"]} for item in items],
                    "nextSequence": after + len(items), "hasMore": after + len(items) < row["event_count"]}

    def snapshot(self, run_id, expected_revision=None):
        with self.connect() as conn:
            self.read_snapshot(conn)
            row = self.row(conn, run_id, include_state=True)
            if expected_revision is not None and expected_revision != row["revision"]:
                raise RevisionConflict("Simulation revision changed")
            return {"schemaVersion": "risk-snapshot-envelope-v1", "runId": run_id,
                    "revision": row["revision"], "asOf": iso(row["virtual_time"]), "syntheticOnly": True,
                    "currency": "EUR", "transactionCoverage": "omitted", "snapshot": projected_snapshot(row["state"])}


def main():
    import argparse
    import os
    parser = argparse.ArgumentParser(description="Import one private checkpoint into PostgreSQL, preserving its public cursor")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--run-key", default=os.environ.get("GENERATOR_RUN_KEY", "default"))
    args = parser.parse_args()
    dsn = os.environ.get("GENERATOR_DATABASE_URL")
    if not dsn:
        parser.error("GENERATOR_DATABASE_URL is required")
    store = PostgresStore(dsn, args.run_key)
    store.migrate()
    store.import_checkpoint(args.checkpoint)
    print(json.dumps(store.status(), ensure_ascii=False))


if __name__ == "__main__":
    main()
