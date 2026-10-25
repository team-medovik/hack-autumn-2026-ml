"""One-time, in-place pacing of pending initial cohorts in a PostgreSQL run.

The committed event prefix, run id and virtual clock never change. Only initial
registration/funding/purchase work that has not happened yet is moved. The
operation is explicit and idempotent through a marker stored in clock_config.
"""
import argparse
import copy
from datetime import timedelta
import heapq
import json
import math
import os

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples/risk/event-driven"))
from billing import instant, iso  # noqa: E402
from behavior import month_key  # noqa: E402


MARKER_KEY = "cohortRepaceMarker"


def shifted(value, delta):
    return iso(instant(value) + delta)


def shift_plan(plan, delta):
    result = copy.deepcopy(plan)
    for key in ("at", "releaseAt"):
        if key in result:
            result[key] = shifted(result[key], delta)
    replacement = result.get("replacement")
    if isinstance(replacement, dict) and "needUntil" in replacement:
        replacement["needUntil"] = shifted(replacement["needUntil"], delta)
    return result


def repace_world(world, spread_days=60, warmup_seconds=1):
    """Move every wholly unpurchased initial cohort into a steady future window."""
    if (not isinstance(spread_days, (int, float)) or isinstance(spread_days, bool)
            or not math.isfinite(spread_days) or spread_days <= 0):
        raise ValueError("spread_days must be positive and finite")
    if (not isinstance(warmup_seconds, (int, float)) or isinstance(warmup_seconds, bool)
            or not math.isfinite(warmup_seconds) or warmup_seconds <= 0):
        raise ValueError("warmup_seconds must be positive and finite")
    now = world.sim.now
    pending = []
    for profile in world.population:
        service_ids = {service["serviceId"] for service in profile["services"]}
        materialized = service_ids & set(world.sim.services)
        if materialized and materialized != service_ids:
            raise ValueError("Partially materialized initial cohort cannot be repaced")
        if not materialized and instant(profile.get("createdAt", world.start)) > now:
            pending.append(profile)
    pending.sort(key=lambda profile: (instant(profile["createdAt"]), profile["accountId"]))
    if not pending:
        raise ValueError("No pending initial cohorts to repace")

    first = now + timedelta(seconds=warmup_seconds)
    last = first + timedelta(days=spread_days)
    if last >= world.end:
        raise ValueError("Repace window must end before the simulation horizon")
    ordered_services = [(profile, service) for profile in pending for service in profile["services"]]
    service_spacing = (timedelta(0) if len(ordered_services) == 1
                       else (last - first) / (len(ordered_services) - 1))
    service_schedule = {service["serviceId"]: first + service_spacing * index
                        for index, (_, service) in enumerate(ordered_services)}
    schedule = {}
    affected_services = {}
    for profile in pending:
        aid = profile["accountId"]
        old_created = instant(profile["createdAt"])
        new_created = min(service_schedule[service["serviceId"]] for service in profile["services"])
        delta = new_created - old_created
        initial = sum(service["priceMinor"] for service in profile["services"])
        allowances = profile["budget"]["monthlyAllowances"]
        old_month, new_month = month_key(old_created), month_key(new_created)
        if old_month != new_month:
            if allowances.get(old_month, 0) < initial:
                raise ValueError("Initial allowance cannot be moved safely")
            allowances[old_month] -= initial
            allowances[new_month] = allowances.get(new_month, 0) + initial
        profile["createdAt"] = iso(new_created)
        already_registered = aid in world.sim.accounts
        if already_registered:
            new_registered = instant(profile["registeredAt"])
        else:
            new_registered = max(first, new_created - service_spacing / 2)
            profile["registeredAt"] = iso(new_registered)
            world.sim._registrations[aid] = new_registered
        profile["lifecyclePlans"] = [shift_plan(
            plan, service_schedule[plan["serviceId"]] - old_created)
            for plan in profile.get("lifecyclePlans", [])]
        for service in profile["services"]:
            sid = service["serviceId"]
            service_delta = service_schedule[sid] - old_created
            service["createdAt"] = iso(service_schedule[sid])
            service["needUntil"] = shifted(service["needUntil"], service_delta)
            world.metadata[sid] = copy.deepcopy(service)
            world.needs[sid] = instant(service["needUntil"])
            affected_services[sid] = aid
        schedule[aid] = {"old": old_created, "created": new_created,
                         "registered": new_registered, "delta": delta,
                         "already_registered": already_registered}

    command_updates = {}
    billing = []
    seen = {aid: {"register": 0, "top_up": 0, "purchase": 0} for aid in schedule}
    for task in world.sim._queue:
        at, phase, tie, sequence, kind, payload, cause = task
        aid = payload.get("account_id")
        if aid in schedule:
            cohort = schedule[aid]
            if kind == "register_account":
                if cohort["already_registered"]:
                    raise ValueError("Registered account still has a registration task")
                at = cohort["registered"]
                seen[aid]["register"] += 1
            elif kind == "top_up" and instant(at) == cohort["old"]:
                at = cohort["created"]
                seen[aid]["top_up"] += 1
            elif kind == "purchase" and payload.get("service_id") in affected_services:
                at = service_schedule[payload["service_id"]]
                seen[aid]["purchase"] += 1
            else:
                raise ValueError("Unexpected pending billing task for an unpurchased cohort")
            if cause.startswith("user-action-"):
                old_command = world.sim._commands[cause]
                if old_command[1] != kind or old_command[2] != payload:
                    raise ValueError("Command and queue payload differ")
                new_command = (at, kind, copy.deepcopy(payload))
                world.sim._commands[cause] = new_command
                command_updates[cause] = new_command
        billing.append((at, phase, tie, sequence, kind, payload, cause))
    for profile in pending:
        aid = profile["accountId"]
        expected_register = 0 if schedule[aid]["already_registered"] else 1
        if (seen[aid]["register"] != expected_register or seen[aid]["top_up"] != 1
                or seen[aid]["purchase"] != len(profile["services"])):
            raise ValueError("Incomplete initial cohort task set")
    heapq.heapify(billing)
    world.sim._queue = billing

    behavior = []
    for task in world.queue:
        at, sequence, kind, payload = task
        aid = payload.get("aid")
        sid = payload.get("sid")
        if aid not in schedule and sid in affected_services:
            aid = affected_services[sid]
        if aid in schedule:
            delta = (service_schedule[sid] - schedule[aid]["old"]
                     if sid in service_schedule else schedule[aid]["delta"])
            at += delta
            payload = copy.deepcopy(payload)
            if isinstance(payload.get("plan"), dict):
                payload["plan"] = shift_plan(payload["plan"], delta)
            if at <= now:
                raise ValueError("Repaced behavior task is not in the future")
        behavior.append((at, sequence, kind, payload))
    heapq.heapify(behavior)
    world.queue = behavior
    if any(task[0] <= now for task in world.sim._queue):
        raise ValueError("Repaced billing task is not in the future")
    return {
        "cohorts": len(pending),
        "services": len(affected_services),
        "firstAt": iso(first),
        "lastAt": iso(last),
        "commands": command_updates,
    }


def apply(settings, marker, spread_days=60, warmup_seconds=1):
    from psycopg.types.json import Jsonb
    from .postgres import PostgresStore, encode, private_state, task_map
    if not marker or len(marker) > 100 or not all(ch.isalnum() or ch in "_.-" for ch in marker):
        raise ValueError("marker must contain 1-100 safe characters")
    store = PostgresStore(settings.database_url, settings.run_key)
    with store.connect() as connection:
        row = connection.execute("SELECT * FROM generator_sim.runs WHERE run_key=%s FOR UPDATE",
                                 (settings.run_key,)).fetchone()
        if row is None:
            raise ValueError("Run is not initialized")
        store.compatible(row)
        clock = dict(row["clock_config"] or {})
        if clock.get("timeScale") != 60:
            raise ValueError("The 60-day demo window requires the persisted 60x speed")
        if clock.get(MARKER_KEY) == marker:
            return {"status": "already_applied", "revision": row["revision"]}
        world = store.restore_world(connection, row)
        event_count, event_cursor, virtual_time = row["event_count"], world.cursor, world.sim.now
        old_tasks = task_map(world)
        result = repace_world(world, spread_days, warmup_seconds)
        new_tasks = task_map(world)
        if set(old_tasks) != set(new_tasks) or len(world.sim.events) != event_count:
            raise ValueError("Repace must preserve task identities and event prefix")
        for command_id, command in result.pop("commands").items():
            changed = connection.execute("UPDATE generator_sim.commands SET payload=%s "
                "WHERE run_id=%s AND command_id=%s", (Jsonb(encode(command)), row["run_id"], command_id))
            if changed.rowcount != 1:
                raise ValueError("Pending command is missing")
        connection.execute("DELETE FROM generator_sim.tasks WHERE run_id=%s", (row["run_id"],))
        store.save_tasks(connection, row["run_id"], {}, new_tasks)
        clock[MARKER_KEY] = marker
        changed = connection.execute("""UPDATE generator_sim.runs
            SET state=%s, clock_config=%s, revision=revision+1,
                next_tick_at=CURRENT_TIMESTAMP, updated_at=CURRENT_TIMESTAMP
            WHERE run_key=%s AND run_id=%s AND event_count=%s AND virtual_time=%s""",
            (Jsonb(private_state(world)), Jsonb(clock), settings.run_key, row["run_id"],
             event_count, virtual_time))
        if changed.rowcount != 1 or world.cursor != event_cursor:
            raise ValueError("Run boundary changed during repace")
        verified = store.restore_world(connection, connection.execute(
            "SELECT * FROM generator_sim.runs WHERE run_key=%s", (settings.run_key,)).fetchone())
        if (verified.sim.now != virtual_time or verified.cursor != event_cursor
                or len(verified.sim.events) != event_count or task_map(verified) != new_tasks):
            raise ValueError("Repaced state failed transactional restore validation")
        return {"status": "applied", "revision": row["revision"] + 1, **result}


def main():
    from .state import Settings
    parser = argparse.ArgumentParser(description="Repace pending initial cohorts in-place")
    parser.add_argument("--apply", action="store_true", help="required explicit mutation flag")
    parser.add_argument("--marker", required=True)
    parser.add_argument("--spread-days", type=float, default=60)
    parser.add_argument("--warmup-seconds", type=float, default=1)
    args = parser.parse_args()
    if not args.apply:
        raise SystemExit("Refusing to mutate without --apply")
    settings = Settings.from_env()
    if not settings.database_url:
        raise SystemExit("GENERATOR_DATABASE_URL is required")
    try:
        result = apply(settings, args.marker, args.spread_days, args.warmup_seconds)
    except Exception:
        # Database and validation errors may contain private state or DSN details.
        raise SystemExit("Generator cohort repace failed; transaction rolled back") from None
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
