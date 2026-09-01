import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CONVERTER_PATH = REPO_ROOT / "training" / "v2" / "nnue-pytorch_to_ember.py"


def load_converter():
    spec = importlib.util.spec_from_file_location("nnue_converter", CONVERTER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NnueConverterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.converter = load_converter()

    def test_same_file_detection_covers_hard_links(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "last.ckpt"
            hard_link = Path(directory) / "output.nnue"
            checkpoint.write_bytes(b"checkpoint")
            os.link(checkpoint, hard_link)

            self.assertTrue(
                self.converter.paths_refer_to_same_file(checkpoint, hard_link)
            )
            self.assertFalse(
                self.converter.paths_refer_to_same_file(
                    checkpoint, Path(directory) / "different.nnue"
                )
            )

    def test_cli_rejects_overwriting_checkpoint_before_importing_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "last.ckpt"
            checkpoint.write_bytes(b"checkpoint")

            result = subprocess.run(
                [
                    sys.executable,
                    str(CONVERTER_PATH),
                    "--repo",
                    str(Path(directory) / "missing-repository"),
                    "--ckpt",
                    str(checkpoint),
                    "--out",
                    str(checkpoint),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("must not refer to the input checkpoint", result.stderr)
            self.assertNotIn("Repository not found", result.stderr)

    def test_failed_verification_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "network.nnue"
            output.write_bytes(b"known-good")

            def writer(path):
                Path(path).write_bytes(b"incomplete")

            def reject(_path):
                raise RuntimeError("invalid container")

            with self.assertRaisesRegex(RuntimeError, "invalid container"):
                self.converter.write_verified_output(output, writer, reject)

            self.assertEqual(output.read_bytes(), b"known-good")
            self.assertEqual(list(Path(directory).glob(".network.nnue.*.tmp")), [])

    def test_verified_output_atomically_replaces_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "network.nnue"
            output.write_bytes(b"old")

            def writer(path):
                Path(path).write_bytes(b"verified")

            def verify(path):
                self.assertEqual(Path(path).read_bytes(), b"verified")
                return {"bytes": 8}

            info = self.converter.write_verified_output(output, writer, verify)

            self.assertEqual(info, {"bytes": 8})
            self.assertEqual(output.read_bytes(), b"verified")
            self.assertEqual(list(Path(directory).glob(".network.nnue.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
