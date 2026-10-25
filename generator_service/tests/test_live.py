"""Real worker/process/HTTP checks; no model or fabricated ML responses."""
from datetime import timedelta
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest

import httpx
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator, FormatChecker
import yaml

from generator_service.api import create_app, GENERATOR_PATHS
from generator_service.state import ROOT, Settings, status, write_json_atomic, iso, utc_now
from generator_service.worker import exclusive_worker
from checkpoint import create_checkpoint, advance_checkpoint, load_checkpoint, state_of
from behavior import make_population, BehaviorSimulation

TOKEN = "local-test-token-not-for-production"
AUTH = {"Authorization": "Bearer " + TOKEN}


def world():
    population = make_population(19, 4, days=80, terms=("month", "day", "hour"),
                                 lifecycle={"cohortSpreadDays": 3})
    return BehaviorSimulation(population, 19, days=80)


def wait_for(predicate, timeout=12):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        result = predicate()
        if result:
            return result
        time.sleep(.025)
    raise AssertionError("Timed out waiting for observable process progress")


class APITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "live.json"
        self.settings = Settings(self.path, token=TOKEN)
        self.saved = create_checkpoint(world(), self.path, "api-test")
        self.saved = advance_checkpoint(self.path, until="2026-01-08T00:00:00Z")
        self.client = TestClient(create_app(self.settings), headers=AUTH)
        self.addCleanup(self.client.close)
        spec = yaml.safe_load((ROOT / "contracts/openapi.yaml").read_text())
        self.components = spec["components"]

    def valid(self, name, response):
        self.assertEqual(200, response.status_code, response.text)
        Draft202012Validator({"$ref": "#/components/schemas/" + name,
                              "components": self.components}, format_checker=FormatChecker()).validate(response.json())
        return response.json()

    def events(self, after=0, limit=7, run="api-test"):
        return self.client.get("/internal/v1/generator/events", params={
            "runId": run, "afterSequence": after, "limit": limit})

    def test_every_read_matches_openapi_and_does_not_mutate_checkpoint(self):
        before = self.path.read_bytes()
        self.valid("GeneratorStatus", self.client.get("/internal/v1/generator/status"))
        self.valid("EventPage", self.events())
        snap = self.valid("SnapshotEnvelope", self.client.get("/internal/v1/generator/snapshot",
                          params={"runId": "api-test", "expectedRevision": self.saved.revision}))
        self.assertEqual("omitted", snap["transactionCoverage"])
        self.assertEqual(self.saved.world.sim.snapshot(), snap["snapshot"])
        self.assertEqual(before, self.path.read_bytes())
        rendered = json.dumps(snap)
        for secret in ("archetype", "monthlyAllowances", "needUntil", "random", "pending_funding"):
            self.assertNotIn(secret, rendered)

    def test_retry_and_paginate_entire_committed_ledger_without_gaps(self):
        cursor, received = 0, []
        while True:
            first = self.valid("EventPage", self.events(cursor))
            again = self.events(cursor).json()
            self.assertEqual(first, again)
            received.extend(first["events"])
            cursor = first["nextSequence"]
            if not first["hasMore"]:
                break
        self.assertEqual(list(range(1, len(received) + 1)), [e["sequence"] for e in received])
        self.assertEqual(self.saved.world.sim.events, [e["event"] for e in received])
        empty = self.events(cursor).json()
        self.assertEqual([], empty["events"])
        self.assertEqual(cursor, empty["nextSequence"])
        # Empty read does not consume a position: later appended events are visible.
        later = advance_checkpoint(self.path, until="2026-01-20T00:00:00Z")
        self.assertGreater(len(later.world.sim.events), cursor)
        self.assertEqual(cursor + 1, self.events(cursor).json()["events"][0]["sequence"])
        self.assertEqual(received[:7], self.events().json()["events"])

    def test_independent_consumers_restart_reads_from_their_own_cursor(self):
        a = self.events(0, 10).json()
        b = self.events(0, 3).json()
        restarted = TestClient(create_app(self.settings), headers=AUTH)
        self.addCleanup(restarted.close)
        page = restarted.get("/internal/v1/generator/events", params={
            "runId": "api-test", "afterSequence": b["nextSequence"], "limit": 7}).json()
        self.assertEqual(a["events"], b["events"] + page["events"])

    def test_invalid_cursor_revision_run_and_queries_are_explicit(self):
        self.assertEqual(404, self.events(run="different-run").status_code)
        self.assertEqual(409, self.events(after=100000).status_code)
        for after, limit in ((-1, 1), (0, 0), (0, 1001), ("bad", 1)):
            self.assertEqual(422, self.events(after, limit).status_code)
        self.assertEqual(422, self.client.get("/internal/v1/generator/events").status_code)
        self.assertEqual(409, self.client.get("/internal/v1/generator/snapshot",
                          params={"runId": "api-test", "expectedRevision": 0}).status_code)
        self.assertEqual(404, self.client.get("/internal/v1/generator/snapshot",
                          params={"runId": "wrong"}).status_code)

    def test_auth_errors_health_and_no_generator_control_routes(self):
        for path in GENERATOR_PATHS[2:]:
            response = self.client.get(path, headers={"Authorization": "Bearer wrong"})
            self.assertEqual(401, response.status_code)
            self.assertEqual({"code", "message", "requestId"}, set(response.json()))
            self.assertNotIn(TOKEN, response.text)
        self.assertEqual({"status": "alive"}, self.client.get("/health/live", headers={"Authorization": ""}).json())
        self.assertEqual({"status": "ready"}, self.client.get("/health/ready").json())
        self.assertEqual(405, self.client.post("/internal/v1/generator/status").status_code)
        self.assertEqual(404, self.client.post("/internal/v1/renewal-risk/predict", json={}).status_code)
        schema = self.client.get("/openapi.json").json()
        self.assertEqual(set(GENERATOR_PATHS), set(schema["paths"]))

    def test_missing_or_corrupt_state_fails_closed_without_path_or_payload(self):
        self.path.write_text('{"secret": "private-state"}')
        self.assertEqual(503, self.client.get("/health/ready").status_code)
        response = self.events()
        self.assertEqual(503, response.status_code)
        self.assertNotIn("private-state", response.text)
        self.assertNotIn(str(self.path), response.text)
        self.path.unlink()
        self.assertEqual(503, self.events().status_code)
        self.assertEqual(200, self.client.get("/health/live").status_code)

    def test_stale_heartbeat_is_error_while_journal_remains_readable(self):
        write_json_atomic(self.settings.heartbeat, {"runId": "api-test", "workerState": "running",
                          "lastHeartbeatAt": iso(utc_now() - timedelta(minutes=1)),
                          "timeScale": 7200, "tickSeconds": .1})
        self.assertEqual("error", status(self.settings, self.saved)["workerState"])
        self.assertEqual(200, self.events().status_code)
        self.assertEqual(200, self.client.get("/health/ready").status_code)

    def test_configuration_requires_secret_and_finite_clock(self):
        with self.assertRaises(ValueError):
            create_app(Settings(self.path))
        for value in (float("nan"), float("inf"), 0, -1):
            with self.assertRaises(ValueError):
                Settings(self.path, time_scale=value)


class ProcessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "state.json"
        self.processes = []
        self.addCleanup(self.cleanup_processes)
        self.environment = dict(os.environ, GENERATOR_CHECKPOINT=str(self.path),
                                GENERATOR_SERVICE_TOKEN=TOKEN, GENERATOR_TIME_SCALE="864000",
                                GENERATOR_TICK_SECONDS="0.04", GENERATOR_MAX_STEP_SECONDS="21600")

    def cleanup_processes(self):
        for process in self.processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)

    def spawn(self, *command, env=None):
        process = subprocess.Popen([sys.executable, "-B", *command], cwd=ROOT,
                                   env=env or self.environment, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
        self.processes.append(process)
        return process

    def worker(self, **overrides):
        return self.spawn("-m", "generator_service.worker", env=dict(self.environment, **overrides))

    def current(self):
        return load_checkpoint(self.path) if self.path.exists() else None

    def test_automatic_events_graceful_restart_preserves_rng_queue_and_no_downtime_catchup(self):
        create_checkpoint(world(), self.path, "restart-test")
        process = self.worker()
        wait_for(lambda: len(self.current().world.sim.events) > 20)
        process.terminate()
        self.assertEqual(0, process.wait(timeout=8))
        before = self.current()
        reference = world()
        reference.advance(before.world.sim.now)
        self.assertEqual(state_of(reference), state_of(before.world))
        time.sleep(.1)
        self.assertEqual(before.revision, self.current().revision)
        # Changing creation seed on restart must never regenerate existing clients.
        resumed = self.worker(GENERATOR_SEED="999", GENERATOR_TICK_SECONDS="2")
        wait_for(lambda: json.loads(Path(str(self.path) + ".worker.json").read_text())["workerState"] == "running")
        self.assertEqual(before.revision, self.current().revision)
        self.assertEqual(before.world.sim.now, self.current().world.sim.now)
        wait_for(lambda: self.current().revision > before.revision)
        resumed.terminate()
        self.assertEqual(0, resumed.wait(timeout=8))
        after = self.current()
        reference.advance(after.world.sim.now)
        self.assertEqual(state_of(reference), state_of(after.world))
        self.assertEqual(before.world.sim.events, after.world.sim.events[:len(before.world.sim.events)])
        self.assertEqual(before.run_id, after.run_id)

    def test_sigkill_resume_keeps_committed_prefix_and_excludes_second_worker(self):
        create_checkpoint(world(), self.path, "kill-test")
        process = self.worker()
        wait_for(lambda: self.current().revision >= 3)
        other = self.worker()
        self.assertNotEqual(0, other.wait(timeout=8))
        process.kill()
        process.wait(timeout=5)
        committed = self.current()
        resumed = self.worker()
        wait_for(lambda: self.current().revision > committed.revision + 2)
        resumed.terminate()
        self.assertEqual(0, resumed.wait(timeout=8))
        after = self.current()
        expected = world()
        expected.advance(after.world.sim.now)
        self.assertEqual(state_of(expected), state_of(after.world))
        self.assertEqual(committed.world.sim.events, after.world.sim.events[:len(committed.world.sim.events)])

    def test_fresh_worker_creates_once_and_finishes_without_resetting_run(self):
        process = self.worker(GENERATOR_DAYS="1", GENERATOR_ACCOUNTS="2",
                              GENERATOR_TIME_SCALE="86400000", GENERATOR_MAX_STEP_SECONDS="86400")
        wait_for(lambda: self.current() is not None and self.current().world.finished)
        process.terminate()
        self.assertEqual(0, process.wait(timeout=8))
        saved = self.current()
        self.assertTrue(saved.world.finished)
        self.assertGreater(len(saved.world.sim.events), 0)
        original = self.path.read_bytes()
        again = self.worker()
        time.sleep(.3)
        self.assertIsNone(again.poll())
        again.terminate()
        self.assertEqual(0, again.wait(timeout=8))
        self.assertEqual(original, self.path.read_bytes())

    def test_real_http_reads_during_generation_and_after_api_restart(self):
        create_checkpoint(world(), self.path, "http-live")
        # Select a loopback port for the isolated subprocess smoke test.
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        def api():
            return self.spawn("-m", "uvicorn", "generator_service.api:create_app", "--factory",
                              "--host", "127.0.0.1", "--port", str(port), "--no-access-log")
        server = api()
        client = httpx.Client(base_url=f"http://127.0.0.1:{port}", headers=AUTH, timeout=5)
        self.addCleanup(client.close)
        def alive():
            try:
                return client.get("/health/live").status_code == 200
            except httpx.TransportError:
                return False
        wait_for(alive)
        initial = self.path.read_bytes()
        for _ in range(3):
            self.assertEqual(200, client.get("/internal/v1/generator/status").status_code)
        self.assertEqual(initial, self.path.read_bytes())
        worker = self.worker()
        wait_for(lambda: len(self.current().world.sim.events) > 20)
        params = {"runId": "http-live", "afterSequence": 0, "limit": 5}
        page = client.get("/internal/v1/generator/events", params=params).json()
        self.assertEqual(5, len(page["events"]))
        server.terminate()
        self.assertIn(server.wait(timeout=8), (0, -signal.SIGTERM))
        server = api()
        wait_for(alive)
        self.assertEqual(page["events"], client.get("/internal/v1/generator/events", params=params).json()["events"])
        worker.terminate()
        self.assertEqual(0, worker.wait(timeout=8))
        expected = self.current().world.sim.events
        cursor, received = 0, []
        while True:
            params.update(afterSequence=cursor, limit=17)
            response = client.get("/internal/v1/generator/events", params=params)
            self.assertEqual(200, response.status_code, response.text)
            batch = response.json()
            received.extend(item["event"] for item in batch["events"])
            cursor = batch["nextSequence"]
            if not batch["hasMore"]:
                break
        self.assertEqual(expected, received)
        self.assertEqual(len(expected), len({e["id"] for e in received}))
        server.terminate()
        self.assertIn(server.wait(timeout=8), (0, -signal.SIGTERM))


if __name__ == "__main__":
    unittest.main()
