import ast
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATHS = [
    REPO_ROOT / "training" / "ru" / "v2" / "train_nnue_colab_v2.ipynb",
    REPO_ROOT / "training" / "en" / "v2" / "train_nnue_colab_v2.ipynb",
]
FINE_TUNE_NOTEBOOK_PATHS = [
    REPO_ROOT
    / "training"
    / language
    / "v2"
    / "train_nnue_colab_v2_fine-tuning.ipynb"
    for language in ("ru", "en")
]


def notebook_cells(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return {
        cell["id"]: "".join(cell.get("source", []))
        for cell in notebook["cells"]
    }


def notebook_functions(source, *names):
    wanted = set(names)
    tree = ast.parse(source)
    body = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in wanted
    ]
    if {node.name for node in body} != wanted:
        raise AssertionError("requested notebook function was not found")
    namespace = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), "<notebook>", "exec"), namespace)
    return namespace


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
        cls.fine_tune_notebooks = {
            path.relative_to(REPO_ROOT).as_posix(): notebook_cells(path)
            for path in FINE_TUNE_NOTEBOOK_PATHS
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

    def test_fine_tuning_download_is_resumable_and_verified(self):
        for notebook, cells in self.fine_tune_notebooks.items():
            with self.subTest(notebook=notebook):
                source = cells["cadfc528"]

                self.assertTrue(source.startswith("%%bash\nset -euo pipefail\n"))
                self.assertIn("--fail --location --continue-at -", source)
                self.assertIn("EXPECTED_SIZE=20144023865", source)
                self.assertIn(
                    "cebf6e5aa62a0df447f3748c90a542ded13105c873a1a819d7f71bdf041ca8cb",
                    source,
                )
                self.assertIn("sha256sum --check", source)
                self.assertIn('PART="${DATASET}.part"', source)
                self.assertIn(
                    'if [ "$PART_SIZE" -lt "$EXPECTED_SIZE" ]; then', source
                )
                self.assertLess(
                    source.index("sha256sum --check"), source.rindex('mv "$PART"')
                )

    def test_fine_tuning_binds_the_base_and_additional_epochs(self):
        for notebook, cells in self.fine_tune_notebooks.items():
            with self.subTest(notebook=notebook):
                config = cells["418fa23c"]
                source = cells["b35ec220"]
                ast.parse(source)

                self.assertIn("BASE_CHECKPOINT", config)
                self.assertIn("BASE_CHECKPOINT_SHA256", config)
                self.assertIn("RESUME_CHECKPOINT", config)
                self.assertIn("ADDITIONAL_EPOCHS", config)
                self.assertNotIn("MAX_EPOCHS", config)
                self.assertIn(
                    "derive_target_max_epochs(base_epoch, additional_epochs)",
                    source,
                )
                self.assertIn("checkpoint_epoch(BASE_CHECKPOINT)", source)
                self.assertIn('"--resume-from-checkpoint"', source)
                self.assertIn("BASE_CHECKPOINT must be outside", source)
                self.assertIn("nnue-pytorch revision mismatch", source)
                self.assertNotIn("glob.glob", source)
                self.assertNotIn("getmtime", source)

                functions = notebook_functions(
                    source,
                    "derive_target_max_epochs",
                    "validate_resume_epoch",
                )
                self.assertEqual(functions["derive_target_max_epochs"](299, 30), 330)
                functions["validate_resume_epoch"](299, 300, 330)
                with self.assertRaisesRegex(ValueError, "positive"):
                    functions["derive_target_max_epochs"](299, 0)
                with self.assertRaisesRegex(ValueError, "predates"):
                    functions["validate_resume_epoch"](299, 298, 330)
                with self.assertRaisesRegex(ValueError, "already reached"):
                    functions["validate_resume_epoch"](299, 329, 330)

    def test_fine_tuning_separates_smoke_and_persists_provenance(self):
        for notebook, cells in self.fine_tune_notebooks.items():
            with self.subTest(notebook=notebook):
                source = cells["b35ec220"]

                self.assertIn("f'{RUN_NAME}-smoke' if smoke else RUN_NAME", source)
                self.assertIn("fine-tune-manifest.json", source)
                self.assertIn("active-launch.json", source)
                self.assertIn("resume_checkpoint_sha256", source)
                self.assertIn("atomic_write_json", source)
                self.assertIn("existing run checkpoints require", source)

    def test_fine_tuning_refuses_concurrent_checkpoint_writers(self):
        sources = []
        for notebook, cells in self.fine_tune_notebooks.items():
            with self.subTest(notebook=notebook):
                source = cells["b35ec220"]
                ast.parse(source)
                sources.append(source)

                self.assertIn("def process_identity(pid):", source)
                self.assertIn("'process_start_ticks'", source)
                self.assertIn("'hostname': socket.gethostname()", source)
                self.assertIn("trainer is still running with pid", source)
                self.assertIn("active launch identity is incomplete", source)
                self.assertIn("active launch belongs to another host", source)
                self.assertIn("active launch belongs to another output", source)
                self.assertIn("set RECOVER_DEAD_LAUNCH=True", source)
                self.assertIn("'output_root': str(OUT_ROOT)", source)

        self.assertEqual(sources[0], sources[1])

    def test_fine_tuning_archives_every_terminal_launch(self):
        for notebook, cells in self.fine_tune_notebooks.items():
            with self.subTest(notebook=notebook):
                source = cells["b35ec220"]

                self.assertIn("def archive_launch(output_root, value):", source)
                self.assertIn("'launch_id': uuid.uuid4().hex", source)
                finish = source[source.index("def finish_launch") :]
                self.assertLess(
                    finish.index("atomic_write_json(launch_path, launch)"),
                    finish.index("archive_launch(OUT_ROOT, launch)"),
                )
                self.assertLess(
                    finish.index("archive_launch(OUT_ROOT, launch)"),
                    finish.index("launch_path.unlink()"),
                )


if __name__ == "__main__":
    unittest.main()
