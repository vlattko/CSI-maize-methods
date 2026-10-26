import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


class ExperimentLogTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(
            importlib.util.find_spec("experiment_lab"), "Experiment log implementation is missing"
        )
        import experiment_lab

        return experiment_lab

    def test_pending_capture_cannot_be_recorded_as_completed(self):
        m = self.module()
        result = {k: [] for k in ["findings", "limitations", "artifacts"]}
        result.update(
            id="E05",
            title="Mechanics",
            status="requires_new_data",
            question="Settling?",
            methods={},
            metrics={},
            next_requirements=[],
            input_sha256={},
        )
        with self.assertRaises(ValueError):
            m.validate_result(result)
        result["next_requirements"] = ["Acquire timestamped settling videos"]
        m.validate_result(result)

    def test_event_history_keeps_previous_bytes_and_increments_sequence(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            m.append_event(path, "E01", "started", {"seed": 20261024})
            first = path.read_bytes()
            m.append_event(path, "E01", "finished", {"status": "completed_exploratory"})
            self.assertTrue(path.read_bytes().startswith(first))
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([r["sequence"] for r in records], [1, 2])
            self.assertEqual(
                records[1]["previous_event_sha256"], m.sha256_bytes(first.rstrip(b"\n"))
            )

    def test_source_hash_verifier_detects_modified_input(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sample.csv"
            source.write_text("a,b\n1,2\n")
            mapping = {"sample.csv": m.sha256_file(source)}
            m.verify_hashes(mapping, root)
            source.write_text("a,b\n1,3\n")
            with self.assertRaises(ValueError):
                m.verify_hashes(mapping, root)

    def test_utf8_result_text_survives_windows_locale(self):
        m = self.module()
        self.assertTrue(hasattr(m, "read_json"), "Explicit UTF-8 result reader is missing")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(
                json.dumps({"title": "RGB\u2013NoIR \u03bc"}, ensure_ascii=False), encoding="utf-8"
            )
            self.assertEqual(m.read_json(path)["title"], "RGB\u2013NoIR \u03bc")


if __name__ == "__main__":
    unittest.main()
