import ast
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATHS = [
    REPO_ROOT / "training" / "ru" / "v2" / "train_nnue_colab_v2.ipynb",
    REPO_ROOT / "training" / "en" / "v2" / "train_nnue_colab_v2.ipynb",
]


class TrainingNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebooks = {
            path.relative_to(REPO_ROOT).as_posix(): {
                cell["id"]: "".join(cell.get("source", []))
                for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]
            }
            for path in NOTEBOOK_PATHS
        }

    def test_dependency_setup_stops_after_unexpected_failures(self):
        for notebook, cells in self.notebooks.items():
            with self.subTest(notebook=notebook):
                source = cells["718c9888"]

                self.assertTrue(source.startswith("%%bash\nset -euo pipefail\n"))
                self.assertFalse(any(line.startswith("!") for line in source.splitlines()))
                self.assertLess(source.index("pip uninstall"), source.index("python -c"))

    def test_training_interrupt_terminates_the_process_group(self):
        for notebook, cells in self.notebooks.items():
            with self.subTest(notebook=notebook):
                source = cells["b35ec220"]
                ast.parse(source)

                self.assertIn("start_new_session=True", source)
                self.assertIn("except KeyboardInterrupt:", source)
                self.assertIn("os.killpg(process.pid, signal.SIGTERM)", source)
                self.assertIn("os.killpg(process.pid, signal.SIGKILL)", source)
                self.assertNotIn("sys.exit(", source)

    def test_training_failure_is_reported_without_zero_exit(self):
        for notebook, cells in self.notebooks.items():
            with self.subTest(notebook=notebook):
                source = cells["b35ec220"]

                self.assertIn("return_code = process.wait()", source)
                self.assertIn(
                    "raise subprocess.CalledProcessError(return_code, cmd)",
                    source,
                )


if __name__ == "__main__":
    unittest.main()
