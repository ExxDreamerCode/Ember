import copy
import json
import subprocess
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
from seek import (  # noqa: E402
    TuningState,
    generate_match_config,
    read_json,
    read_toml,
    run_single_match,
    select_param_specs,
    split_command,
    try_candidate,
    tune_parameter,
    validate_best,
    validate_engine_params,
    validate_runtime_options,
    validate_tune_config,
    write_json,
)
from sprt import pentanomial_sprt  # noqa: E402


class AutoTuneTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "results_dir": "results/tune",
            "common": {
                "max_pairs": 1000,
                "min_pairs": 20,
                "batch_pairs": 20,
                "timemargin_ms": 50,
                "time_controls": ["1+0.01"],
                "seed": 1,
                "opening_source": "polyglot",
                "polyglot_book": "src/book.bin",
                "book_min_plies": 8,
                "book_max_plies": 20,
                "max_moves": 220,
                "hash_mb": 64,
                "threads": 1,
            },
            "sprt": {
                "enabled": True,
                "elo0": 0,
                "elo1": 3,
                "alpha": 0.05,
                "beta": 0.05,
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
            "recheck": {
                "enabled": True,
                "time_control": "1+0.01",
                "max_pairs": 5000,
                "min_pairs": 20,
                "batch_pairs": 20,
                "seed": 3,
                "min_elo": 5,
                "elo0": 0,
                "elo1": 3,
                "alpha": 0.05,
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
        ]
        self.cfg["params"] = self.params
        self.best = {"values": {}}

    def test_lmp_range_stops_at_the_implemented_move_count_table(self):
        cfg = read_toml(AUTO_TUNE / "tune.toml")
        lmp = next(
            spec for spec in cfg["params"] if spec["name"] == "LMP_MAX_DEPTH"
        )

        self.assertEqual(lmp["base"], 8)
        self.assertEqual(lmp["max"], 8)

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
        self.assertEqual(config["run"]["timemargin_ms"], 50)

    def test_match_time_margin_is_configurable(self):
        common = {**self.cfg["common"], "timemargin_ms": 75}

        config = generate_match_config(
            self.cfg,
            common,
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

        self.assertEqual(config["run"]["timemargin_ms"], 75)

    def test_preflight_validates_config_best_and_selection(self):
        params = validate_tune_config(self.cfg)
        validate_best(self.best, params)
        self.assertEqual(
            select_param_specs(params, "PROBCUT_MIN_DEPTH"),
            self.params,
        )

        duplicate = copy.deepcopy(self.cfg)
        duplicate["params"].append(dict(duplicate["params"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate tune parameter"):
            validate_tune_config(duplicate)

        disabled = copy.deepcopy(self.cfg)
        disabled["sprt"]["enabled"] = False
        with self.assertRaisesRegex(ValueError, "sprt.enabled must be true"):
            validate_tune_config(disabled)

        reused_seed = copy.deepcopy(self.cfg)
        reused_seed["confirmation"]["seed"] = reused_seed["common"]["seed"]
        with self.assertRaisesRegex(ValueError, "seed must differ"):
            validate_tune_config(reused_seed)

        no_recheck = copy.deepcopy(self.cfg)
        del no_recheck["recheck"]
        with self.assertRaisesRegex(ValueError, r"must contain \[recheck\]"):
            validate_tune_config(no_recheck)

        reused_recheck_seed = copy.deepcopy(self.cfg)
        reused_recheck_seed["recheck"]["seed"] = reused_recheck_seed["common"]["seed"]
        with self.assertRaisesRegex(ValueError, "recheck.seed must differ"):
            validate_tune_config(reused_recheck_seed)

        bad_min_elo = copy.deepcopy(self.cfg)
        bad_min_elo["recheck"]["min_elo"] = 0
        with self.assertRaisesRegex(ValueError, "recheck.min_elo must be positive"):
            validate_tune_config(bad_min_elo)

        loose_confirmation = copy.deepcopy(self.cfg)
        loose_confirmation["confirmation"]["alpha"] = self.cfg["sprt"]["alpha"]
        with self.assertRaisesRegex(ValueError, "below discovery alpha"):
            validate_tune_config(loose_confirmation)

        nonpositive = copy.deepcopy(self.cfg)
        nonpositive["params"][0]["min"] = 0
        with self.assertRaisesRegex(ValueError, "min must be positive"):
            validate_tune_config(nonpositive)
        nonpositive["params"][0]["allow_nonpositive"] = True
        nonpositive_params = validate_tune_config(nonpositive)
        validate_best({"values": {"PROBCUT_MIN_DEPTH": 0}}, nonpositive_params)

        invalid_opt_in = copy.deepcopy(self.cfg)
        invalid_opt_in["params"][0]["allow_nonpositive"] = "yes"
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            validate_tune_config(invalid_opt_in)

        with self.assertRaisesRegex(ValueError, "unknown --params"):
            select_param_specs(params, "NOT_A_PARAMETER")
        with self.assertRaisesRegex(ValueError, "unknown parameters"):
            validate_best({"values": {"NOT_A_PARAMETER": 1}}, params)
        with self.assertRaisesRegex(ValueError, "outside its range"):
            validate_best({"values": {"PROBCUT_MIN_DEPTH": 100}}, params)
        stepped = [{**params[0], "step": 2}]
        with self.assertRaisesRegex(ValueError, "off its step grid"):
            validate_best({"values": {"PROBCUT_MIN_DEPTH": 9}}, stepped)

    def test_split_command_keeps_windows_backslashes(self):
        self.assertEqual(
            split_command(r"C:\Tools\SomeDir\engine.exe"),
            [r"C:\Tools\SomeDir\engine.exe"],
        )
        self.assertEqual(
            split_command(r'"C:\Program Files\Engine\engine.exe" -arg'),
            [r"C:\Program Files\Engine\engine.exe", "-arg"],
        )

    def test_preflight_validates_runtime_overrides(self):
        validate_runtime_options("1+0.01", 1, 0.5)
        with self.assertRaisesRegex(ValueError, r"BASE\+INCREMENT"):
            validate_runtime_options("fast", None, None)
        with self.assertRaisesRegex(ValueError, "--workers must be positive"):
            validate_runtime_options(None, 0, None)
        with self.assertRaisesRegex(ValueError, "--worker-multiplier"):
            validate_runtime_options(None, None, float("nan"))

    def test_preflight_checks_parameter_names_against_engine(self):
        output = "info string tune PROBCUT_MIN_DEPTH = 8\n"
        completed = subprocess.CompletedProcess(
            ["ember"],
            0,
            stdout=output,
        )
        with (
            patch("seek.engine_binary", return_value="/tmp/ember"),
            patch("seek.subprocess.run", return_value=completed),
        ):
            validate_engine_params("ember", self.best, self.params)

        completed.stdout = "info string tune: no active overrides\n"
        with (
            patch("seek.engine_binary", return_value="/tmp/ember"),
            patch("seek.subprocess.run", return_value=completed),
            self.assertRaisesRegex(ValueError, "did not accept"),
        ):
            validate_engine_params("ember", self.best, self.params)

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
            with (
                patch(
                    "seek.run_single_match",
                    return_value=(
                        {"verdict": "engine_a_better"},
                        "run-a",
                        {
                            "run_id": "run-a",
                            "verdict": "engine_a_better",
                            "accepted": True,
                        },
                    ),
                ),
                patch("seek.write_match_report"),
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
            with (
                patch(
                    "seek.run_single_match",
                    return_value=(
                        {"verdict": "engine_b_better"},
                        "run-b",
                        {
                            "run_id": "run-b",
                            "verdict": "engine_b_better",
                            "accepted": False,
                        },
                    ),
                ),
                patch("seek.write_match_report"),
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

    def test_parameter_search_continues_only_in_accepted_direction(self):
        spec = {
            "name": "PROBCUT_MIN_DEPTH",
            "base": 8,
            "min": 4,
            "max": 16,
            "step": 1,
        }
        attempted = []

        def accept_downward(_cfg, _best, _params, _name, candidate, *_args):
            attempted.append(candidate)
            return candidate in {7, 6}

        with patch("seek.try_candidate", side_effect=accept_downward):
            settled = tune_parameter(
                self.cfg,
                self.best,
                [spec],
                spec,
                "target/release/ember",
                "journal.jsonl",
                "best.json",
                False,
                None,
                None,
            )

        self.assertEqual(settled, 6)
        self.assertEqual(attempted, [9, 7, 6, 5])
        self.assertEqual(len(attempted), len(set(attempted)))

    def test_parameter_search_checks_both_wider_directions(self):
        spec = {
            "name": "PROBCUT_MIN_DEPTH",
            "base": 8,
            "min": 4,
            "max": 16,
            "step": 1,
        }
        attempted = []

        def accept_negative_wider(_cfg, _best, _params, _name, candidate, *_args):
            attempted.append(candidate)
            return candidate == 6

        with patch("seek.try_candidate", side_effect=accept_negative_wider):
            settled = tune_parameter(
                self.cfg,
                self.best,
                [spec],
                spec,
                "target/release/ember",
                "journal.jsonl",
                "best.json",
                False,
                None,
                None,
            )

        self.assertEqual(settled, 6)
        self.assertEqual(attempted, [9, 7, 10, 6, 5])
        self.assertEqual(len(attempted), len(set(attempted)))

    def test_json_replacement_keeps_previous_file_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.json"
            write_json(path, {"values": {"A": 1}})

            with patch("seek.os.replace", side_effect=OSError("interrupted")):
                with self.assertRaisesRegex(OSError, "interrupted"):
                    write_json(path, {"values": {"A": 2}})

            self.assertEqual(read_json(path), {"values": {"A": 1}})
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_parameter_resume_does_not_repeat_completed_probes(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            session = {"test": "parameter-resume"}
            state = TuningState(state_path, session)
            spec = {
                "name": "PROBCUT_MIN_DEPTH",
                "base": 8,
                "min": 4,
                "max": 16,
                "step": 1,
            }
            state.start_parameter(spec["name"], spec["base"])
            for candidate in (9, 7, 10, 6):
                state.record_attempt(spec["name"], candidate, False)

            resumed_state = TuningState(state_path, session)
            with patch("seek.try_candidate") as repeated_match:
                settled = tune_parameter(
                    self.cfg,
                    self.best,
                    [spec],
                    spec,
                    "target/release/ember",
                    "journal.jsonl",
                    "best.json",
                    False,
                    None,
                    None,
                    resumed_state,
                )

            self.assertEqual(settled, 8)
            repeated_match.assert_not_called()
            self.assertIn(spec["name"], resumed_state.data["completed_params"])
            self.assertIsNone(resumed_state.data["parameter"])

    def test_different_invocation_discards_the_stale_state(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            TuningState(state_path, {"binary_sha256": "first"})

            state = TuningState(state_path, {"binary_sha256": "second"})
            self.assertEqual(state.data["session"], {"binary_sha256": "second"})
            self.assertEqual(state.data["completed_params"], [])
            self.assertIsNone(state.data["parameter"])
            self.assertIsNone(state.data["active_match"])

    def test_interrupted_match_resumes_and_commits_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = {
                **self.cfg,
                "results_dir": str(root / "reports"),
                "common": {
                    **self.cfg["common"],
                    "time_control": "1+0.01",
                },
            }
            spec = {
                "name": "PROBCUT_MIN_DEPTH",
                "base": 8,
                "min": 4,
                "max": 16,
                "step": 1,
            }
            best = {"values": {}}
            best_path = root / "best.json"
            journal_path = root / "journal.jsonl"
            state_path = root / "state.json"
            session = {"test": "resume"}
            state = TuningState(state_path, session)
            state.start_parameter(spec["name"], spec["base"])

            with patch("seek.run_match", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    run_single_match(
                        cfg,
                        best,
                        [spec],
                        spec["name"],
                        9,
                        sys.executable,
                        journal_path,
                        None,
                        None,
                        state,
                    )

            active = dict(state.data["active_match"])
            config_path = Path(active["config_path"])
            self.assertTrue(config_path.is_file())

            summary = {
                "verdict": "engine_a_better",
                "elo": 4.0,
                "score_rate": 0.51,
                "pairs": 20,
                "games": 40,
                "sprt": {"llr": 3.0},
            }
            with (
                patch("seek.run_match") as resumed_match,
                patch("seek.read_summary", return_value=summary),
            ):
                _summary, run_id, _record = run_single_match(
                    cfg,
                    best,
                    [spec],
                    spec["name"],
                    9,
                    sys.executable,
                    journal_path,
                    None,
                    None,
                    state,
                )

            self.assertEqual(run_id, active["run_id"])
            resumed_match.assert_called_once_with(config_path, active["run_id"])
            self.assertEqual(state.data["active_match"]["status"], "completed")
            write_json(config_path.parent / "estimates" / "summary.json", summary)

            with (
                patch.object(state, "record_attempt", side_effect=KeyboardInterrupt),
                self.assertRaises(KeyboardInterrupt),
            ):
                try_candidate(
                    cfg,
                    best,
                    [spec],
                    spec["name"],
                    9,
                    sys.executable,
                    journal_path,
                    best_path,
                    False,
                    None,
                    None,
                    state,
                )

            resumed_state = TuningState(state_path, session)
            resumed_best = read_json(best_path)
            with patch("seek.run_match") as duplicate_match:
                accepted = try_candidate(
                    cfg,
                    resumed_best,
                    [spec],
                    spec["name"],
                    9,
                    sys.executable,
                    journal_path,
                    best_path,
                    False,
                    None,
                    None,
                    resumed_state,
                )

            self.assertTrue(accepted)
            duplicate_match.assert_not_called()
            self.assertEqual(read_json(best_path)["values"][spec["name"]], 9)
            records = [json.loads(line) for line in journal_path.read_text().splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["run_id"], active["run_id"])
            self.assertIsNone(resumed_state.data["active_match"])
            self.assertEqual(resumed_state.data["parameter"]["current"], 9)
