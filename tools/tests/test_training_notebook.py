import ast
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = REPO_ROOT / "training" / "v2" / "train_nnue_colab_v2.ipynb"


class TrainingNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.cells = {
            cell["id"]: "".join(cell.get("source", []))
            for cell in notebook["cells"]
        }

    def test_dependency_setup_stops_after_unexpected_failures(self):
        source = self.cells["718c9888"]

        self.assertTrue(source.startswith("%%bash\nset -euo pipefail\n"))
        self.assertFalse(any(line.startswith("!") for line in source.splitlines()))
        self.assertLess(source.index("pip uninstall"), source.index("python -c"))

    def test_training_interrupt_terminates_the_process_group(self):
        source = self.cells["b35ec220"]
        ast.parse(source)

        self.assertIn("start_new_session=True", source)
        self.assertIn("except KeyboardInterrupt:", source)
        self.assertIn("os.killpg(process.pid, signal.SIGTERM)", source)
        self.assertIn("os.killpg(process.pid, signal.SIGKILL)", source)
        self.assertNotIn("sys.exit(", source)

    def test_training_failure_is_reported_without_zero_exit(self):
        source = self.cells["b35ec220"]

        self.assertIn("return_code = process.wait()", source)
        self.assertIn(
            "raise subprocess.CalledProcessError(return_code, cmd)",
            source,
        )


if __name__ == "__main__":
    unittest.main()
