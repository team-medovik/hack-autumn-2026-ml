"""One autonomous writer; no HTTP requests or ML required to generate events."""
from contextlib import contextmanager
from datetime import timedelta
import fcntl
import logging
import os
import signal
import threading
import time

from .state import Settings, utc_now, iso, write_json_atomic
from checkpoint import create_checkpoint, advance_checkpoint, load_checkpoint
from behavior import BehaviorSimulation, make_population

LOG = logging.getLogger("generator.worker")


@contextmanager
def exclusive_worker(path):
    """Keep ownership for the whole process, not just individual checkpoint writes."""
    fd = os.open(str(path) + ".worker.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another worker owns this simulation") from error
        yield
    finally:
        os.close(fd)


def run(settings, stop=None):
    stop = stop if stop is not None else threading.Event()
    if settings.database_url:
        return run_postgres(settings, stop)
    settings.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_worker(settings.checkpoint):
        if settings.checkpoint.exists() or settings.checkpoint.is_symlink():
            saved = load_checkpoint(settings.checkpoint)
            LOG.info("Resuming run=%s revision=%s", saved.run_id, saved.revision)
        else:
            population = make_population(settings.seed, settings.accounts, start_at=settings.start,
                                         days=settings.days, lifecycle=True, terms=settings.payment_terms)
            saved = create_checkpoint(BehaviorSimulation(population, settings.seed, settings.start,
                                                         settings.days), settings.checkpoint)
            LOG.info("Created run=%s", saved.run_id)

        def heartbeat(state):
            write_json_atomic(settings.heartbeat, {
                "runId": saved.run_id, "revision": saved.revision, "workerState": state,
                "lastHeartbeatAt": iso(utc_now()), "timeScale": settings.time_scale,
                "tickSeconds": settings.tick_seconds,
            })

        # Restart establishes a new real-time anchor: downtime never advances virtual time.
        virtual_anchor, real_anchor = saved.world.sim.now, time.monotonic()
        heartbeat("finished" if saved.world.finished else "running")
        try:
            while not stop.is_set():
                if stop.wait(settings.tick_seconds):
                    break
                if saved.world.finished:
                    heartbeat("finished")
                    continue
                target = min(saved.world.end,
                             virtual_anchor + timedelta(seconds=(time.monotonic() - real_anchor) * settings.time_scale),
                             saved.world.sim.now + timedelta(seconds=settings.max_step_seconds))
                before_count = len(saved.world.sim.events)
                saved = advance_checkpoint(settings.checkpoint, until=target,
                                           expected_revision=saved.revision)
                heartbeat("finished" if saved.world.finished else "running")
                if len(saved.world.sim.events) != before_count:
                    LOG.info("Committed run=%s revision=%s events=%s virtualTime=%s",
                             saved.run_id, saved.revision, len(saved.world.sim.events), iso(saved.world.sim.now))
            heartbeat("finished" if saved.world.finished else "paused")
            return saved
        except Exception:
            heartbeat("error")
            raise


def run_postgres(settings, stop):
    """The database owns both the run lock and tick pacing across worker replicas."""
    from .postgres import PostgresStore
    from psycopg import OperationalError, InterfaceError

    store = PostgresStore(settings.database_url, settings.run_key)
    latest, initialized, retry_seconds = None, False, 1.0
    while not stop.is_set():
        try:
            if not initialized:
                store.migrate()
                run_id = store.ensure_run(settings)
                LOG.info("PostgreSQL simulation ready run=%s", run_id)
                initialized = True
            # Every attempt reloads committed state. If the previous connection
            # failed after COMMIT, the DB cursor and due gate prevent reapplying it.
            result = store.tick(settings)
        except (OperationalError, InterfaceError) as error:
            # Never log exception text: a connection error can contain DSN data.
            LOG.warning("PostgreSQL unavailable (%s); retry in %.0fs",
                        type(error).__name__, retry_seconds)
            if stop.wait(retry_seconds):
                break
            retry_seconds = min(retry_seconds * 2, 30.0)
            continue
        retry_seconds = 1.0
        if result is not None:
            core = result["generator"]
            if latest is None or core["eventCount"] != latest["generator"]["eventCount"]:
                LOG.info("Committed run=%s revision=%s events=%s virtualTime=%s",
                         core["runId"], core["revision"], core["eventCount"], core["virtualTime"])
            latest = result
        if stop.wait(settings.tick_seconds):
            break
    # Another replica may still be serving this run. Do not mark shared state
    # paused just because this particular process is shutting down.
    return latest


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop.set())
    run(Settings.from_env(), stop)


if __name__ == "__main__":
    main()
