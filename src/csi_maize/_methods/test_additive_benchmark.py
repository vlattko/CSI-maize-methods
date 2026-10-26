import importlib.util
from pathlib import Path
import tempfile
import unittest


class BenchmarkEvidenceTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(
            importlib.util.find_spec("additive_benchmark"), "Benchmark assembler is not implemented"
        )
        import additive_benchmark

        return additive_benchmark

    def test_automatic_mask_cannot_be_called_ground_truth(self):
        m = self.module()
        case = {"annotation_status": "automatic_prediction", "is_ground_truth": True}
        with self.assertRaises(ValueError):
            m.validate_case(case)

    def test_challenge_cannot_claim_an_untouched_split(self):
        m = self.module()
        case = {
            "annotation_status": "automatic_prediction",
            "is_ground_truth": False,
            "split": "independent_test",
        }
        with self.assertRaises(ValueError):
            m.validate_case(case)

    def test_case_keeps_content_hash_not_only_filename(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "source.jpg"
            path.write_bytes(b"original")
            case = m.source_case("case1", "segmentation", path, root)
            self.assertEqual(case["source_sha256"], m.sha256_file(path))
            self.assertFalse(case["is_ground_truth"])
            path.write_bytes(b"changed")
            self.assertNotEqual(case["source_sha256"], m.sha256_file(path))


if __name__ == "__main__":
    unittest.main()
