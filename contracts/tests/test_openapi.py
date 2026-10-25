"""Contract-only checks; these do not claim the HTTP handlers exist yet."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from jsonschema import Draft202012Validator, FormatChecker, ValidationError
from openapi_spec_validator import validate
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/risk/event-driven"))
import behavior
from checkpoint import create_checkpoint, advance_checkpoint


class OpenAPIContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = yaml.safe_load((ROOT / "contracts/openapi.yaml").read_text())
        cls.history = json.loads((ROOT / "examples/risk/event-driven/generated/lifecycle-v3-seed20261024/observations.json").read_text())

    def validator(self, name):
        return Draft202012Validator({"$ref": "#/components/schemas/" + name,
                                     "components": self.spec["components"]}, format_checker=FormatChecker())

    def test_valid_standalone_openapi_and_unique_operations(self):
        validate(self.spec)
        ids = [operation["operationId"] for path in self.spec["paths"].values()
               for method, operation in path.items() if method in ("get", "post", "patch")]
        self.assertEqual(8, len(ids))
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("НЕ реализованы", self.spec["info"]["description"])

    def test_every_declared_field_explained_and_references_are_local(self):
        def walk(node):
            if isinstance(node, list):
                for value in node:
                    walk(value)
            if not isinstance(node, dict):
                return
            if "$ref" in node:
                self.assertTrue(node["$ref"].startswith("#/"))
            if node.get("type") == "object" and "properties" in node:
                for key, value in node["properties"].items():
                    self.assertTrue(value.get("description"), key)
            for value in node.values():
                walk(value)
        walk(self.spec)

    def test_read_endpoints_are_protected_and_do_not_expose_checkpoint(self):
        self.assertEqual([{"ServiceToken": []}], self.spec["security"])
        for path, methods in self.spec["paths"].items():
            if "generator" in path:
                expected = {"patch"} if path == "/internal/v1/generator/speed" else {"get"}
                self.assertEqual(expected, set(methods))
            if not path.startswith("/health"):
                for operation in methods.values():
                    self.assertNotEqual([], operation.get("security"))
        self.assertFalse(any("checkpoint" in p for p in self.spec["paths"]))

    def test_all_saved_v3_events_match_documented_schema(self):
        validator = self.validator("Event")
        for event in self.history["events"]:
            with self.subTest(event=event["id"]):
                validator.validate(event)

    def test_snapshot_export_and_explicit_omitted_transactions(self):
        envelope = {"schemaVersion": "risk-snapshot-envelope-v1", "runId": "demo-v3",
                    "revision": 42, "asOf": self.history["asOf"], "syntheticOnly": True,
                    "currency": "EUR", "transactionCoverage": "omitted",
                    "snapshot": self.history["snapshot"]}
        self.validator("SnapshotEnvelope").validate(envelope)
        del envelope["transactionCoverage"]
        with self.assertRaises(ValidationError):
            self.validator("SnapshotEnvelope").validate(envelope)

    def test_all_three_inference_inputs_are_explicit(self):
        base = {"requestId": "request-1", "contractVersion": "renewal-risk-v1",
                "snapshotRevision": "42", "predictionAt": "2028-12-31T00:00:00Z", "serviceIds": [3000180]}
        inputs = [
            {"kind": "history", "history": {"schemaVersion": "vps-billing-events-v2", "runId": "demo-v3",
                                             "asOf": self.history["asOf"], "events": [], "syntheticOnly": True}},
            {"kind": "snapshot", "snapshot": {"schemaVersion": "risk-snapshot-envelope-v1", "runId": "demo-v3",
                                               "revision": 42, "asOf": self.history["asOf"], "syntheticOnly": True,
                                               "currency": "EUR", "transactionCoverage": "omitted",
                                               "snapshot": self.history["snapshot"]}},
            {"kind": "dataset", "datasetId": "known-dataset", "revision": "42"},
        ]
        for source in inputs:
            request = {**base, "input": source}
            self.validator("PredictRequest").validate(request)
        for extra in ("url", "accessKey", "secretKey", "label", "archetype"):
            request = {**base, "input": {**inputs[2], extra: "forbidden"}}
            with self.assertRaises(ValidationError):
                self.validator("PredictRequest").validate(request)
        request = {**base, "input": inputs[2], "serviceIds": [1, 1]}
        with self.assertRaises(ValidationError):
            self.validator("PredictRequest").validate(request)

    def test_predictions_require_probability_or_explicit_reason_not_fake_zero(self):
        prediction = {"serviceId": 1, "accountId": 1, "dueAt": "2026-02-01T00:00:00Z",
                      "outcomeDeadline": "2026-02-06T00:00:00Z", "status": "predicted",
                      "reasonCode": None, "lossProbability": .7, "warnings": []}
        validator = self.validator("Prediction")
        validator.validate(prediction)
        for value in (-.1, 1.1, None, True):
            with self.assertRaises(ValidationError):
                validator.validate({**prediction, "lossProbability": value})
        skipped = {**prediction, "status": "insufficient_data", "reasonCode": "SNAPSHOT_HAS_NO_HISTORY",
                   "lossProbability": None}
        validator.validate(skipped)
        with self.assertRaises(ValidationError):
            validator.validate({**skipped, "lossProbability": 0})

    def test_actual_checkpoint_public_status_matches_schema(self):
        population = behavior.make_population(1, 2, days=10, lifecycle=True)
        world = behavior.BehaviorSimulation(population, 1, days=10)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "state.json"
            saved = create_checkpoint(world, path)
            self.validator("GeneratorCoreStatus").validate(saved.public_status())
            saved = advance_checkpoint(path)
            self.validator("GeneratorCoreStatus").validate(saved.public_status())

    def test_event_required_payload_and_private_future_rejected(self):
        event = next(e for e in self.history["events"] if e["type"] == "payment_succeeded")
        validator = self.validator("Event")
        for key in ("amountMinor", "paidThrough", "mode"):
            broken = copy.deepcopy(event)
            del broken[key]
            with self.assertRaises(ValidationError):
                validator.validate(broken)
        with self.assertRaises(ValidationError):
            validator.validate({**event, "needUntil": "2030-01-01T00:00:00Z"})


if __name__ == "__main__":
    unittest.main()
