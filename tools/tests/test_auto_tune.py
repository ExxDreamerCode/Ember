import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


TOOLS = Path(__file__).resolve().parents[1]
AUTO_TUNE = TOOLS / "auto_tune"
sys.path.insert(0, str(AUTO_TUNE))
sys.path.insert(0, str(TOOLS))

from head_to_head import decision  # noqa: E402
from seek import generate_match_config, try_candidate  # noqa: E402
from sprt import pentanomial_sprt  # noqa: E402


class AutoTuneTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "results_dir": "results/tune",
            "common": {
                "max_pairs": 1000,
                "min_pairs": 20,
                "batch_pairs": 20,
                "seed": 1,
                "opening_source": "polyglot",
                "polyglot_book": "src/book.bin",
                "book_min_plies": 8,
                "book_max_plies": 20,
                "max_moves": 220,
                "hash_mb": 64,
                "threads": 1,
            },
            "sprt": {"elo0": 0, "elo1": 3, "alpha": 0.05, "beta": 0.05},
        }
        self.params = [
            {"name": "PROBCUT_MIN_DEPTH", "base": 8, "min": 4, "max": 16},
        ]
        self.best = {"values": {}}

    def test_candidate_is_engine_a_for_positive_sprt_alternative(self):
        config = generate_match_config(
            self.cfg,
            self.cfg["common"],
            self.cfg["sprt"],
            "PROBCUT_MIN_DEPTH",
            9,
            self.best,
            self.params,
            "target/release/ember",
            "1+0.01",
            None,
            None,
        )

        self.assertEqual(config["engine_a"]["name"], "Candidate")
        self.assertEqual(
            config["engine_a"]["options"]["Tune"], "PROBCUT_MIN_DEPTH=9"
        )
        self.assertEqual(config["engine_b"]["name"], "Incumbent")
        self.assertEqual(
            config["engine_b"]["options"]["Tune"], "PROBCUT_MIN_DEPTH=8"
        )

    def test_selected_candidate_keeps_other_best_values_on_both_sides(self):
        params = [
            {
                "name": "PROBCUT_MIN_DEPTH",
                "base": 8,
                "min": 4,
                "max": 16,
            },
            {
                "name": "PROBCUT_MARGIN_CP",
                "base": 350,
                "min": 200,
                "max": 600,
            },
        ]
        best = {"values": {"PROBCUT_MARGIN_CP": 375}}

        config = generate_match_config(
            self.cfg,
            self.cfg["common"],
            self.cfg["sprt"],
            "PROBCUT_MIN_DEPTH",
            9,
            best,
            params,
            "target/release/ember",
            "1+0.01",
            None,
            None,
        )

        self.assertEqual(
            config["engine_a"]["options"]["Tune"],
            "PROBCUT_MIN_DEPTH=9,PROBCUT_MARGIN_CP=375",
        )
        self.assertEqual(
            config["engine_b"]["options"]["Tune"],
            "PROBCUT_MIN_DEPTH=8,PROBCUT_MARGIN_CP=375",
        )

    def test_real_sprt_maps_candidate_advantage_to_engine_a_better(self):
        head_to_head_cfg = {
            "run": {"min_pairs": 20},
            "sprt": {"enabled": True},
        }
        candidate_advantage = pentanomial_sprt(
            [100, 200, 400, 300, 200], 0, 3, alpha=0.10, beta=0.05
        )
        incumbent_advantage = pentanomial_sprt(
            [200, 300, 400, 200, 100], 0, 3, alpha=0.10, beta=0.05
        )

        self.assertEqual(candidate_advantage["state"], "accept_h1")
        self.assertEqual(
            decision(
                {"pairs": candidate_advantage["pairs"], "sprt": candidate_advantage},
                head_to_head_cfg,
            ),
            "engine_a_better",
        )
        self.assertEqual(incumbent_advantage["state"], "accept_h0")
        self.assertEqual(
            decision(
                {"pairs": incumbent_advantage["pairs"], "sprt": incumbent_advantage},
                head_to_head_cfg,
            ),
            "engine_b_better",
        )

        equality = pentanomial_sprt(
            [0, 0, 1000, 0, 0], 0, 3, alpha=0.10, beta=0.05
        )
        self.assertEqual(equality["state"], "accept_h0")
        self.assertNotEqual(equality["state"], "accept_h1")

    def test_only_candidate_positive_sprt_verdict_updates_best(self):
        with tempfile.TemporaryDirectory() as directory:
            best_path = Path(directory) / "best.json"
            with patch(
                "seek.run_single_match",
                return_value=({"verdict": "engine_a_better"}, "run", {}),
            ):
                accepted = try_candidate(
                    self.cfg,
                    self.best,
                    self.params,
                    "PROBCUT_MIN_DEPTH",
                    9,
                    "target/release/ember",
                    Path(directory) / "journal.jsonl",
                    best_path,
                    False,
                    None,
                    None,
                )

            self.assertTrue(accepted)
            self.assertEqual(self.best["values"]["PROBCUT_MIN_DEPTH"], 9)

            self.best["values"].clear()
            with patch(
                "seek.run_single_match",
                return_value=({"verdict": "engine_b_better"}, "run", {}),
            ):
                accepted = try_candidate(
                    self.cfg,
                    self.best,
                    self.params,
                    "PROBCUT_MIN_DEPTH",
                    9,
                    "target/release/ember",
                    Path(directory) / "journal.jsonl",
                    best_path,
                    False,
                    None,
                    None,
                )

            self.assertFalse(accepted)
            self.assertEqual(self.best["values"], {})
