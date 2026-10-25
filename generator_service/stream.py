"""Persistent rolling demo stream for an existing PostgreSQL generator run."""
import argparse, copy, hashlib, json, sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples/risk/event-driven"))
from behavior import ARCHETYPES, RANGES, month_key, stream  # noqa: E402
from billing import instant, iso  # noqa: E402

CATALOG = json.loads((ROOT / "examples/risk/vps-business/catalog.json").read_text())["tariffs"]
CONFIG_KEY = "demoStreamV2"


def unit(seed, sequence):
    raw = hashlib.sha256(f"{seed}:{sequence}".encode()).digest()[:8]
    return int.from_bytes(raw, "big") / (2 ** 64 - 1)


def new_account(world, config, at):
    aid = config["accountBase"] + config["clientSequence"]
    if aid in world.profiles or aid in world.sim._registrations:
        raise ValueError("Demo account id collision")
    archetype = stream(config["seed"], aid, "archetype").choice(ARCHETYPES)
    rng = stream(config["seed"], aid, "habits")
    habits = {key: rng.uniform(*bounds) for key, bounds in RANGES[archetype].items()}
    profile = {"accountId": aid, "archetype": archetype, "habits": habits,
        "budget": {"monthlyBaseMinor": 0, "monthlyAllowances": {}, "initialPurchaseMinor": 0},
        "services": [], "createdAt": iso(at), "registeredAt": iso(at), "lifecyclePlans": []}
    world.population.append(profile); world.profiles[aid] = profile
    world.random[aid] = stream(config["seed"], aid, "actions")
    world.sim.add_account(aid, at=at)
    config["clientSequence"] += 1
    return aid


def new_service(world, config, aid, at):
    sequence = config["serviceSequence"]; sid = config["serviceBase"] + sequence
    if sid in world.owner or sid in world.sim._planned_services:
        raise ValueError("Demo service id collision")
    tariff = CATALOG[int(unit(config["seed"] + 1, sequence) * len(CATALOG)) % len(CATALOG)]
    price = tariff["monthPriceMinor"]
    service = {"serviceId": sid, "term": "month", "priceMinor": price,
        "tariffCode": tariff["code"], "createdAt": iso(at),
        "needUntil": iso(at + timedelta(days=35 + int(unit(config["seed"] + 2, sequence) * 331))),
        "priceProvenance": "saved_public_catalog"}
    profile = world.profiles[aid]; profile["services"].append(service)
    budget = profile["budget"]; budget["monthlyBaseMinor"] += price; budget["initialPurchaseMinor"] += price
    key = month_key(at); budget["monthlyAllowances"][key] = budget["monthlyAllowances"].get(key, 0) + price
    world.owner[sid] = aid; world.metadata[sid] = copy.deepcopy(service); world.needs[sid] = instant(service["needUntil"])
    world.command(at, "top_up", account_id=aid, amount=price)
    world.command(at, "purchase", account_id=aid, service_id=sid, price=price, term="month",
                  tariff_code=tariff["code"], auto_prolong=world.random[aid].random() < profile["habits"]["autoChance"])
    world.task(world.needs[sid], "need_ended", sid=sid)
    if len(profile["services"]) == 1:
        world.task(at + world.delay(aid, "checkIntervalHours"), "periodic_review", aid=aid)
    config["serviceSequence"] += 1


def inject_due(world, until, clock):
    config = clock.get(CONFIG_KEY)
    if not isinstance(config, dict) or not config.get("enabled"): return 0
    until = instant(until); count = 0
    while instant(config["nextServiceAt"]) <= until:
        at = max(instant(config["nextServiceAt"]), world.sim.now + timedelta(microseconds=1))
        if config["serviceSequence"] % config["servicesPerClient"] == 0:
            config["activeAccountId"] = new_account(world, config, at)
        new_service(world, config, config["activeAccountId"], at)
        seconds = config["minRealSeconds"] + (config["maxRealSeconds"] - config["minRealSeconds"]) * unit(config["seed"] + 3, config["serviceSequence"])
        config["nextServiceAt"] = iso(at + timedelta(seconds=seconds * float(clock.get("timeScale", 60))))
        count += 1
        if count > 1000: raise ValueError("Demo stream catch-up limit exceeded")
    return count


def enable(settings, marker, min_real_seconds=10, max_real_seconds=30, services_per_client=6, minimum_real_hours=72):
    from psycopg.types.json import Jsonb
    from .postgres import PostgresStore, private_state
    if not marker or len(marker) > 100 or not all(ch.isalnum() or ch in "_.-" for ch in marker): raise ValueError("Invalid marker")
    if not 1 <= min_real_seconds <= max_real_seconds <= 300 or not 1 <= services_per_client <= 100 or minimum_real_hours < 72: raise ValueError("Invalid stream settings")
    store = PostgresStore(settings.database_url, settings.run_key)
    with store.connect() as connection:
        row = connection.execute("SELECT * FROM generator_sim.runs WHERE run_key=%s FOR UPDATE", (settings.run_key,)).fetchone()
        if row is None: raise ValueError("Run is not initialized")
        store.compatible(row); clock = dict(row["clock_config"] or {}); existing = clock.get(CONFIG_KEY)
        if existing is not None and not isinstance(existing, dict):
            raise ValueError("Invalid persisted stream configuration")
        scale = float(clock.get("timeScale", 60)); end_at = max(row["end_at"], row["virtual_time"] + timedelta(hours=minimum_real_hours * scale))
        world = store.restore_world(connection, row); world.end = end_at
        if existing:
            required = {"seed", "accountBase", "serviceBase", "clientSequence",
                        "serviceSequence", "activeAccountId"}
            if not required <= existing.keys():
                raise ValueError("Persisted stream cannot be updated safely")
            config = dict(existing)
            status = "updated"
        else:
            config = {"seed": 20261025, "accountBase": 900000000,
                      "serviceBase": 900000000000, "clientSequence": 0,
                      "serviceSequence": 0, "activeAccountId": None}
            status = "enabled"
        config.update({"enabled": True, "marker": marker,
            "minRealSeconds": min_real_seconds, "maxRealSeconds": max_real_seconds,
            "servicesPerClient": services_per_client,
            "nextServiceAt": iso(row["virtual_time"] + timedelta(seconds=1))})
        clock[CONFIG_KEY] = config
        connection.execute("""UPDATE generator_sim.runs SET state=%s,end_at=%s,clock_config=%s,
            revision=revision+1,next_tick_at=CURRENT_TIMESTAMP,updated_at=CURRENT_TIMESTAMP
            WHERE run_key=%s AND run_id=%s""", (Jsonb(private_state(world)), end_at, Jsonb(clock), settings.run_key, row["run_id"]))
        return {"status": status, "minRealSeconds": min_real_seconds, "maxRealSeconds": max_real_seconds,
                "servicesPerClient": services_per_client, "minimumRealHours": minimum_real_hours}


def main():
    from .state import Settings
    parser = argparse.ArgumentParser(); parser.add_argument("--apply", action="store_true"); parser.add_argument("--marker", required=True)
    parser.add_argument("--min-real-seconds", type=float, default=10); parser.add_argument("--max-real-seconds", type=float, default=30)
    parser.add_argument("--services-per-client", type=int, default=6); parser.add_argument("--minimum-real-hours", type=float, default=72)
    args = parser.parse_args()
    if not args.apply: raise SystemExit("Refusing to mutate without --apply")
    settings = Settings.from_env()
    try: result = enable(settings, args.marker, args.min_real_seconds, args.max_real_seconds, args.services_per_client, args.minimum_real_hours)
    except Exception: raise SystemExit("Generator rolling stream enable failed; transaction rolled back") from None
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__": main()
