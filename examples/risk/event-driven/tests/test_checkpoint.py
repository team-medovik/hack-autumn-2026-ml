"""Restart/atomicity tests, not evidence of real-world behavioral accuracy."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import behavior
import checkpoint as cp
from billing import instant, iso


def world(seed=19, days=180, terms=("month",), accounts=4):
    population = behavior.make_population(seed, accounts, days=days, terms=terms,
                                         lifecycle={"cohortSpreadDays": min(30, days - 1)})
    return behavior.BehaviorSimulation(population, seed, days=days)


class SteppingTests(unittest.TestCase):
    def test_steps_match_batch_for_multiple_seeds_and_payment_terms(self):
        for seed, terms, days in ((1, ("month",), 180), (7, ("day",), 40),
                                  (23, ("hour",), 8), (12, ("month", "day", "hour"), 15)):
            with self.subTest(seed=seed):
                expected = world(seed, days, terms)
                expected.run()
                actual = world(seed, days, terms)
                events = []
                while not actual.finished:
                    events.extend(actual.step())
                self.assertEqual(actual.sim.export(), expected.sim.export())
                self.assertEqual(actual.decisions, expected.decisions)
                self.assertEqual(events, actual.sim.events)
                self.assertEqual(cp.state_of(actual), cp.state_of(expected))

    def test_arbitrary_time_chunks_match_batch(self):
        expected, actual = world(), world()
        expected.run()
        for at in ("2026-01-01T00:00:00Z", "2026-01-10T03:45:00Z",
                   "2026-02-01T00:00:00Z", "2026-04-03T12:00:00Z", iso(actual.end)):
            actual.advance(at)
        self.assertEqual(actual.sim.export(), expected.sim.export())
        self.assertEqual(actual.decisions, expected.decisions)

    def test_rejects_backward_naive_and_beyond_horizon_without_mutation(self):
        actual = world()
        actual.advance("2026-02-01T00:00:00Z")
        before = cp.canonical(cp.encode(cp.state_of(actual)))
        for at in ("2025-01-01T00:00:00Z", "2030-01-01T00:00:00Z", "2026-02-02T00:00:00"):
            with self.assertRaises(ValueError):
                actual.advance(at)
            self.assertEqual(before, cp.canonical(cp.encode(cp.state_of(actual))))

    def test_same_boundary_and_completed_run_are_idempotent(self):
        actual = world()
        actual.step()
        self.assertEqual([], actual.advance(actual.sim.now))
        actual.run()
        before = cp.state_of(actual)
        self.assertEqual([], actual.step())
        self.assertEqual(before, cp.state_of(actual))
        self.assertTrue(actual.finished)
        self.assertIsNone(actual.next_event_at)

    def test_observation_is_only_through_cutoff_and_detached(self):
        actual = world()
        events = actual.advance("2026-02-01T00:00:00Z")
        self.assertTrue(events)
        self.assertTrue(all(instant(e["occurredAt"]) <= actual.sim.now for e in events))
        events[0]["id"] = "changed-by-reader"
        self.assertEqual(actual.sim.events[0]["id"], "billing-0000001")
        self.assertEqual(actual.cursor, len(actual.sim.events))

    def test_historical_lifecycle_v3_events_are_unchanged(self):
        path = ROOT / "generated/lifecycle-v3-seed20261024/observations.json"
        previous = json.loads(path.read_text())
        current = behavior.build(seed=20261024, account_count=50, days=1095, lifecycle=True)
        self.assertEqual(previous, current["observations"])


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "state.json"

    def test_initial_restore_keeps_all_typed_state(self):
        original = world()
        cp.create_checkpoint(original, self.path, "test-run")
        result = cp.load_checkpoint(self.path)
        self.assertEqual(result.run_id, "test-run")
        self.assertEqual(result.revision, 0)
        self.assertEqual(cp.state_of(original), cp.state_of(result.world))
        self.assertIsInstance(result.world.queue[0], tuple)
        self.assertIsInstance(result.world.pending_funding, set)

    def test_repeated_restarts_match_uninterrupted_events_decisions_and_rng(self):
        original = world()
        original.run()
        cp.create_checkpoint(world(), self.path)
        for at in ("2026-01-12T00:00:00Z", "2026-02-05T12:45:00Z",
                   "2026-04-11T00:00:00Z", iso(original.end)):
            committed = cp.advance_checkpoint(self.path, at)
            restored = cp.load_checkpoint(self.path)
            self.assertEqual(cp.state_of(committed.world), cp.state_of(restored.world))
        self.assertEqual(cp.state_of(original), cp.state_of(restored.world))

    def test_restore_at_each_early_step_preserves_equal_time_order(self):
        expected = world(days=8, terms=("hour", "day"), accounts=2)
        expected.run()
        cp.create_checkpoint(world(days=8, terms=("hour", "day"), accounts=2), self.path)
        for _ in range(35):
            cp.advance_checkpoint(self.path)
        actual = cp.advance_checkpoint(self.path, iso(expected.end)).world
        self.assertEqual(cp.state_of(expected), cp.state_of(actual))

    def test_restart_while_replacement_and_restriction_are_pending(self):
        population = behavior.make_population(3, 1, days=180)
        profile = population[0]
        profile["services"] = [{"serviceId": 3000010, "term": "month", "priceMinor": 593,
                                "tariffCode": "WAWs-1", "needUntil": "2028-01-01T00:00:00Z"}]
        profile["habits"].update(preChance=0, autoChance=1, attendChance=1,
                                 followThroughChance=1, topupDelayHours=.01, manualDelayHours=.01)
        profile["budget"]["monthlyAllowances"] = {k: 100000 for k in profile["budget"]["monthlyAllowances"]}
        profile["lifecyclePlans"] = [
            {"kind": "restrict", "serviceId": 3000010, "at": "2026-01-20T00:00:00Z",
             "releaseAt": "2026-02-08T00:00:00Z", "reason": "policy_review"},
            {"kind": "replace", "serviceId": 3000010, "at": "2026-03-10T00:00:00Z",
             "transferDelayHours": 24, "transferSucceeds": True,
             "replacement": {"serviceId": 300001001, "term": "month", "priceMinor": 849,
                             "tariffCode": "WAWs-2"}},
        ]
        expected = behavior.BehaviorSimulation(population, 3, days=180)
        expected.run()
        cp.create_checkpoint(behavior.BehaviorSimulation(population, 3, days=180), self.path)
        for at in ("2026-02-01T00:00:00Z", "2026-03-10T12:00:00Z", iso(expected.end)):
            actual = cp.advance_checkpoint(self.path, at).world
        self.assertEqual(cp.state_of(expected), cp.state_of(actual))
        kinds = {e["type"] for e in actual.sim.events}
        self.assertTrue({"service_restricted", "service_released", "replacement_confirmed"} <= kinds)

    def test_resume_in_another_process(self):
        cp.create_checkpoint(world(), self.path)
        cp.advance_checkpoint(self.path, "2026-02-01T00:00:00Z")
        expected = world()
        expected.run()
        result = subprocess.run([sys.executable, "-B", str(ROOT / "checkpoint.py"),
                                 "--checkpoint", str(self.path), "advance", "--until", iso(expected.end)],
                                capture_output=True, text=True, check=True)
        self.assertTrue(json.loads(result.stdout)["finished"])
        self.assertEqual(cp.state_of(expected), cp.state_of(cp.load_checkpoint(self.path).world))

    def test_creation_does_not_overwrite_existing_state(self):
        cp.create_checkpoint(world(), self.path)
        before = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            cp.create_checkpoint(world(2), self.path)
        self.assertEqual(before, self.path.read_bytes())

    def test_stale_revision_and_invalid_time_do_not_write(self):
        cp.create_checkpoint(world(), self.path)
        cp.advance_checkpoint(self.path, expected_revision=0)
        before = self.path.read_bytes()
        with self.assertRaises(cp.RevisionConflict):
            cp.advance_checkpoint(self.path, expected_revision=0)
        with self.assertRaises(ValueError):
            cp.advance_checkpoint(self.path, "2020-01-01T00:00:00Z")
        self.assertEqual(before, self.path.read_bytes())

    def test_same_absolute_boundary_does_not_increment_revision(self):
        cp.create_checkpoint(world(), self.path)
        a = cp.advance_checkpoint(self.path, "2026-02-01T00:00:00Z")
        before = self.path.read_bytes()
        b = cp.advance_checkpoint(self.path, "2026-02-01T00:00:00Z")
        self.assertEqual(a.revision, b.revision)
        self.assertEqual(before, self.path.read_bytes())

    def test_failed_atomic_replace_preserves_old_commit_and_retry(self):
        cp.create_checkpoint(world(), self.path)
        before = self.path.read_bytes()
        with patch.object(cp.os, "replace", side_effect=OSError("simulated disk failure")):
            with self.assertRaises(OSError):
                cp.advance_checkpoint(self.path, "2026-02-01T00:00:00Z")
        self.assertEqual(before, self.path.read_bytes())
        self.assertEqual([], list(self.path.parent.glob(".checkpoint-*.tmp")))
        actual = cp.advance_checkpoint(self.path, "2026-02-01T00:00:00Z")
        expected = world()
        expected.advance("2026-02-01T00:00:00Z")
        self.assertEqual(cp.state_of(expected), cp.state_of(actual.world))

    def test_checksum_truncation_and_wrong_engine_are_rejected(self):
        cp.create_checkpoint(world(), self.path)
        original = self.path.read_bytes()
        for raw in (original[:100], original.replace(b'"revision":0', b'"revision":9')):
            self.path.write_bytes(raw)
            with self.assertRaises(cp.CheckpointError):
                cp.load_checkpoint(self.path)
        self.path.write_bytes(original)
        with patch.object(cp, "engine_fingerprint", return_value="incompatible"):
            with self.assertRaisesRegex(cp.CheckpointError, "Incompatible"):
                cp.load_checkpoint(self.path)

    def test_schema_runtime_and_state_validation(self):
        cp.create_checkpoint(world(), self.path)
        base = json.loads(self.path.read_text())
        for key, value in (("schemaVersion", "v99"), ("pythonVersion", [0, 0]), ("revision", True)):
            data = copy.deepcopy(base)
            data[key] = value
            data.pop("checksumSha256")
            data["checksumSha256"] = hashlib.sha256(cp.canonical(data)).hexdigest()
            self.path.write_bytes(cp.canonical(data))
            with self.assertRaises(cp.CheckpointError):
                cp.load_checkpoint(self.path)
        bad = world()
        bad.step()
        bad.sim.accounts[next(iter(bad.sim.accounts))]["balance"] += 1
        with self.assertRaisesRegex(cp.CheckpointError, "journal"):
            cp.restore(cp.decode(cp.encode(cp.state_of(bad))))

    def test_private_state_not_in_status_or_snapshot_and_file_is_private(self):
        cp.create_checkpoint(world(), self.path)
        committed = cp.advance_checkpoint(self.path, "2026-02-01T00:00:00Z")
        public = json.dumps({"status": committed.public_status(), "snapshot": committed.world.sim.snapshot()})
        for forbidden in ("archetype", "monthlyAllowances", "needUntil", "lifecyclePlans", "checksumSha256"):
            self.assertNotIn(forbidden, public)
        self.assertEqual(0o600, self.path.stat().st_mode & 0o777)

    def test_two_processes_with_same_revision_only_one_commits(self):
        cp.create_checkpoint(world(), self.path)
        command = [sys.executable, "-B", str(ROOT / "checkpoint.py"), "--checkpoint", str(self.path),
                   "advance", "--expected-revision", "0"]
        processes = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(2)]
        results = [(p.communicate(timeout=30), p.returncode) for p in processes]
        self.assertEqual([0, 1], sorted(code for _, code in results))
        self.assertEqual(1, cp.load_checkpoint(self.path).revision)

    def test_process_killed_before_replace_leaves_previous_checkpoint_usable(self):
        cp.create_checkpoint(world(), self.path)
        original = self.path.read_bytes()
        script = (
            "import sys, os, signal; sys.path.insert(0, sys.argv[1]); import checkpoint as cp; "
            "cp.os.replace = lambda *args: os.kill(os.getpid(), signal.SIGKILL); "
            "cp.advance_checkpoint(sys.argv[2], '2026-02-01T00:00:00Z')"
        )
        result = subprocess.run([sys.executable, "-B", "-c", script, str(ROOT), str(self.path)],
                                capture_output=True, timeout=30)
        self.assertEqual(-9, result.returncode)
        self.assertEqual(original, self.path.read_bytes())
        # A killed writer can leave a private temp file; readers never mistake it for a commit.
        self.assertEqual(1, len(list(self.path.parent.glob(".checkpoint-*.tmp"))))
        resumed = cp.advance_checkpoint(self.path, "2026-02-01T00:00:00Z")
        self.assertEqual(1, resumed.revision)

    def test_process_killed_after_replace_resumes_committed_revision(self):
        cp.create_checkpoint(world(), self.path)
        script = """
import sys, os, signal
sys.path.insert(0, sys.argv[1])
import checkpoint as cp
replace = cp.os.replace
def committed_then_killed(*args):
    replace(*args)
    os.kill(os.getpid(), signal.SIGKILL)
cp.os.replace = committed_then_killed
cp.advance_checkpoint(sys.argv[2], '2026-02-01T00:00:00Z')
"""
        result = subprocess.run([sys.executable, "-B", "-c", script, str(ROOT), str(self.path)],
                                capture_output=True, timeout=30)
        self.assertEqual(-9, result.returncode)
        self.assertEqual(1, cp.load_checkpoint(self.path).revision)
        retried = cp.advance_checkpoint(self.path, "2026-02-01T00:00:00Z")
        self.assertEqual(1, retried.revision)

    def test_command_idempotency_registry_survives_restart(self):
        cp.create_checkpoint(world(), self.path)
        cp.advance_checkpoint(self.path, "2026-02-01T00:00:00Z")
        restored = cp.load_checkpoint(self.path).world
        command_id, (at, kind, payload) = next(iter(restored.sim._commands.items()))
        before = cp.state_of(restored)
        self.assertFalse(restored.sim.schedule(command_id, at, kind, **payload))
        with self.assertRaisesRegex(ValueError, "reused"):
            restored.sim.schedule(command_id, at, "different-kind", **payload)
        self.assertEqual(before, cp.state_of(restored))


if __name__ == "__main__":
    unittest.main()
