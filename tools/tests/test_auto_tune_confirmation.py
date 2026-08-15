import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


TOOLS = Path(__file__).resolve().parents[1]
AUTO_TUNE = TOOLS / "auto_tune"
sys.path.insert(0, str(AUTO_TUNE))
sys.path.insert(0, str(TOOLS))

from confirm import (  # noqa: E402
    confirmation_run_id,
    generate_confirmation_config,
    prepare_confirmation_config,
    run_confirmation,
)
from seek import read_toml  # noqa: E402


class AutoTuneConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "results_dir": "results/tune",
            "common": {
                "timemargin_ms": 50,
                "seed": 1,
                "opening_source": "polyglot",
                "polyglot_book": "src/book.bin",
                "book_min_plies": 8,
                "book_max_plies": 20,
                "max_moves": 220,
                "hash_mb": 64,
                "threads": 1,
            },
            "confirmation": {
                "enabled": True,
                "time_control": "1+0.01",
                "max_pairs": 1000,
                "min_pairs": 20,
                "batch_pairs": 20,
                "seed": 2,
                "elo0": 0,
                "elo1": 3,
                "alpha": 0.025,
                "beta": 0.05,
            },
        }
        self.params = [
            {
                "name": "PROBCUT_MIN_DEPTH",
                "base": 8,
                "min": 4,
                "max": 16,
                "step": 1,
            },
            {
                "name": "PROBCUT_MARGIN_CP",
                "base": 350,
                "min": 200,
                "max": 600,
                "step": 25,
            },
        ]
        self.best = {
            "values": {
                "PROBCUT_MIN_DEPTH": 9,
                "PROBCUT_MARGIN_CP": 375,
            }
        }
        self.binary_sha256 = "a" * 64

    def make_config(self):
        return generate_confirmation_config(
            self.cfg,
            self.best,
            self.params,
            "target/release/ember",
            self.binary_sha256,
        )

    def test_confirmation_compares_full_tuned_vector_with_full_defaults(self):
        config = self.make_config()

        self.assertEqual(config["engine_a"]["name"], "Tuned")
        self.assertEqual(
            config["engine_a"]["options"]["Tune"],
            "PROBCUT_MIN_DEPTH=9,PROBCUT_MARGIN_CP=375",
        )
        self.assertEqual(config["engine_b"]["name"], "Defaults")
        self.assertEqual(
            config["engine_b"]["options"]["Tune"],
            "PROBCUT_MIN_DEPTH=8,PROBCUT_MARGIN_CP=350",
        )
        self.assertEqual(config["run"]["seed"], 2)
        self.assertEqual(config["run"]["timemargin_ms"], 50)
        self.assertEqual(config["sprt"]["alpha"], 0.025)
        self.assertEqual(
            config["run"]["results_dir"],
            str(Path("results/tune") / "confirmations"),
        )

    def test_confirmation_rejects_an_unchanged_vector(self):
        with self.assertRaisesRegex(ValueError, "no changes to confirm"):
            generate_confirmation_config(
                self.cfg,
                {"values": {}},
                self.params,
                "target/release/ember",
                self.binary_sha256,
            )

    def test_confirmation_run_identity_covers_config_and_binary(self):
        config = self.make_config()
        run_id = confirmation_run_id(config)

        self.assertEqual(confirmation_run_id(copy.deepcopy(config)), run_id)
        changed_binary = copy.deepcopy(config)
        changed_binary["confirmation_meta"]["binary_sha256"] = "b" * 64
        self.assertNotEqual(confirmation_run_id(changed_binary), run_id)
        changed_candidate = copy.deepcopy(config)
        changed_candidate["engine_a"]["options"]["Tune"] = (
            "PROBCUT_MIN_DEPTH=10,PROBCUT_MARGIN_CP=375"
        )
        self.assertNotEqual(confirmation_run_id(changed_candidate), run_id)

    def test_existing_run_must_have_the_exact_same_config(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.make_config()
            config["run"]["results_dir"] = directory
            run_id = confirmation_run_id(config)

            path, resumed = prepare_confirmation_config(config, run_id)
            self.assertFalse(resumed)
            self.assertEqual(read_toml(path), config)

            resumed_path, resumed = prepare_confirmation_config(config, run_id)
            self.assertTrue(resumed)
            self.assertEqual(resumed_path, path)

            changed = copy.deepcopy(config)
            changed["run"]["time_control"] = "2+0.02"
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                prepare_confirmation_config(changed, run_id)

    def test_confirmation_uses_full_head_to_head_workflow(self):
        with patch("confirm.subprocess.run") as subprocess_run:
            subprocess_run.return_value.returncode = 0
            run_confirmation(Path("match.toml"), "confirm-test")

        command = subprocess_run.call_args.args[0]
        self.assertIn("all", command)
        self.assertEqual(command[-2:], ["--run-id", "confirm-test"])


if __name__ == "__main__":
    unittest.main()
