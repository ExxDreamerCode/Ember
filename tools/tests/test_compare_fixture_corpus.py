import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from compare_fixture_corpus import (  # noqa: E402
    DEFAULT_HASH_MB,
    FixtureCheck,
    direction,
    disabled_status,
    evaluate_gate,
    format_gate_report,
    move_matches,
    parse_uci_option,
    parse_fixture,
    run_check,
    search_observations,
    summarize,
    uci_setup_commands,
)


class CompareFixtureCorpusTests(unittest.TestCase):
    def fake_engine(
        self,
        directory,
        info_line,
        exit_code=0,
        diagnostic=None,
        setup_info=None,
        bestmove="a1a2",
    ):
        script = Path(directory) / "fake-engine"
        info_code = (
            f"print({info_line!r}, flush=True)" if info_line is not None else "pass"
        )
        diagnostic_code = (
            f"print({diagnostic!r}, flush=True)" if diagnostic is not None else "pass"
        )
        setup_info_code = (
            f"print({setup_info!r}, flush=True)" if setup_info is not None else "pass"
        )
        script.write_text(
            f"""#!/usr/bin/env python3
import sys

for command in sys.stdin:
    command = command.strip()
    if command == "uci":
        {setup_info_code}
        print("uciok", flush=True)
    elif command == "isready":
        print("readyok", flush=True)
    elif command.startswith("position "):
        {diagnostic_code}
    elif command.startswith("go "):
        {info_code}
        print("bestmove {bestmove}", flush=True)
    elif command == "quit":
        raise SystemExit({exit_code})
""",
            encoding="utf-8",
        )
        script.chmod(0o755)
        return script

    def fixture_check(self, depth):
        return FixtureCheck(
            fixture="cases.tsv",
            line_number=2,
            activation="active",
            fixture_format="standard",
            variant="standard",
            case_id="case",
            depth=depth,
            fen="8/8/8/8/8/8/8/K6k w - - 0 1",
            setup_move="-",
            expected_move="a1a2",
        )

    def test_parses_active_and_both_disabled_formats(self):
        fen = "8/8/8/8/8/8/8/K6k w - - 0 1"
        contents = """\
# source comment
id\tdepth\tfen_before_blunder\tsetup_move\texpected_move\tthemes\trating\tpopularity\tplays
active\t4\t{fen}\t-\ta1a2\ttheme\t0\t0\t0
book\t0\t{fen}\t-\ta2a3\tbook\t0\t0\t0
# disabled\t7\t{fen}\t-\tb1b2|b1c3\ttheme\t0\t0\t0
# failed_id\tfen_before_blunder\tsetup_move\texpected_move\tgot_depth2\tgot_depth3\tgot_depth4\tthemes\trating\tpopularity\tplays
# mined\t{fen}\tc1c2\td1d2\te1e2\te1e3\te1e4\ttheme\t0\t0\t0
""".format(fen=fen)
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "cases.tsv"
            fixture.write_text(contents, encoding="utf-8")
            checks = parse_fixture(fixture)

        self.assertEqual(len(checks), 6)
        self.assertEqual(
            [(check.case_id, check.depth, check.activation) for check in checks],
            [
                ("active", 4, "active"),
                ("book", 0, "active"),
                ("disabled", 7, "disabled"),
                ("mined", 2, "disabled"),
                ("mined", 3, "disabled"),
                ("mined", 4, "disabled"),
            ],
        )

    def test_move_expectations(self):
        self.assertTrue(move_matches("a1a2", "a1a2"))
        self.assertTrue(move_matches("a1a3", "a1a2|a1a3"))
        self.assertFalse(move_matches("a1a4", "a1a2|a1a3"))
        self.assertTrue(move_matches("a1a4", "!a1a2|a1a3"))
        self.assertFalse(move_matches("a1a2", "!a1a2|a1a3"))

    def test_rejects_malformed_fen_rank_widths(self):
        contents = """\
id\tdepth\tfen_before_blunder\tsetup_move\texpected_move\tthemes\trating\tpopularity\tplays
bad-rank\t4\t8/8/8/8/8/7/8/K6k w - - 0 1\t-\ta1a2\ttheme\t0\t0\t0
"""
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "cases.tsv"
            fixture.write_text(contents, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expands to 7 squares"):
                parse_fixture(fixture)

    def test_directions(self):
        self.assertEqual(direction(True, True), "both-pass")
        self.assertEqual(direction(True, False), "baseline-only")
        self.assertEqual(direction(False, True), "candidate-only")
        self.assertEqual(direction(False, False), "neither-pass")

    def test_disabled_status_marks_stale_and_newly_fixed_rows(self):
        def row(activation, baseline, candidate):
            return {
                "check": {"activation": activation},
                "baseline": {"passed": baseline},
                "candidate": {"passed": candidate},
            }

        self.assertEqual(disabled_status(row("active", True, True)), "-")
        self.assertEqual(
            disabled_status(row("disabled", False, False)),
            "still-red",
        )
        self.assertEqual(
            disabled_status(row("disabled", False, True)),
            "fixed-by-candidate",
        )
        self.assertEqual(
            disabled_status(row("disabled", True, True)),
            "stale-passes-in-both",
        )
        self.assertEqual(
            disabled_status(row("disabled", True, False)),
            "lost-while-disabled",
        )

    def test_uci_setup_matches_the_engine_fixture_defaults(self):
        self.assertEqual(DEFAULT_HASH_MB, 256)
        self.assertIn(
            "setoption name Hash value 256",
            uci_setup_commands(DEFAULT_HASH_MB),
        )
        self.assertIn(
            "setoption name Book value",
            uci_setup_commands(DEFAULT_HASH_MB),
        )
        self.assertIn(
            "setoption name Book value <embedded>",
            uci_setup_commands(DEFAULT_HASH_MB, use_embedded_book=True),
        )
        self.assertEqual(parse_uci_option("NNUE=/tmp/net.nnue"), ("NNUE", "/tmp/net.nnue"))
        self.assertIn(
            "setoption name NNUE value /tmp/net.nnue",
            uci_setup_commands(
                DEFAULT_HASH_MB,
                options=[("NNUE", "/tmp/net.nnue")],
            ),
        )

    def test_extracts_depth_and_nodes_from_search_output(self):
        self.assertEqual(
            search_observations(
                [
                    "info depth 1 score cp 3 nodes 5 pv a1a2",
                    "info depth 4 score cp 7 nodes 42 pv a1a2",
                ]
            ),
            (4, 42),
        )
        self.assertEqual(
            search_observations(
                [
                    "info depth 4 score cp 7 nodes 40 pv a1a2",
                    "info depth 4 score cp 8 nodes 42 pv a1a2",
                ]
            ),
            (4, 42),
        )
        self.assertEqual(search_observations(["info nodes 7"]), (None, 7))

    def test_run_check_ignores_setup_telemetry(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = self.fake_engine(
                directory,
                None,
                setup_info="info depth 40 score cp 0 nodes 999 pv a1a2",
            )
            result = run_check(
                engine,
                self.fixture_check(4),
                5.0,
                DEFAULT_HASH_MB,
            )
        self.assertFalse(result["passed"])
        self.assertIn("did not report search depth and nodes", result["error"])

    def test_run_check_rejects_search_below_requested_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = self.fake_engine(
                directory,
                "info depth 3 score cp 0 nodes 10 pv a1a2",
            )
            result = run_check(engine, self.fixture_check(4), 5.0, DEFAULT_HASH_MB)
        self.assertFalse(result["passed"])
        self.assertIn("below requested depth", result["error"])

    def test_run_check_requires_zero_nodes_for_book_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = self.fake_engine(
                directory,
                "info depth 1 score cp 0 nodes 1 pv a1a2",
            )
            result = run_check(engine, self.fixture_check(0), 5.0, DEFAULT_HASH_MB)
        self.assertFalse(result["passed"])
        self.assertIn("book check searched 1 node", result["error"])

    def test_unreported_book_stats_require_explicit_baseline_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = self.fake_engine(directory, None)
            strict = run_check(engine, self.fixture_check(0), 5.0, DEFAULT_HASH_MB)
            compatible = run_check(
                engine,
                self.fixture_check(0),
                5.0,
                DEFAULT_HASH_MB,
                allow_unreported_book_stats=True,
            )
        self.assertFalse(strict["passed"])
        self.assertIn("did not report search depth and nodes", strict["error"])
        self.assertTrue(compatible["passed"])
        self.assertIsNone(compatible["reported_depth"])
        self.assertIsNone(compatible["reported_nodes"])

    def test_run_check_rejects_position_setup_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = self.fake_engine(
                directory,
                "info depth 4 score cp 0 nodes 10 pv a1a2",
                diagnostic="info string Stopping position move list at illegal move: a1a2",
            )
            result = run_check(engine, self.fixture_check(4), 5.0, DEFAULT_HASH_MB)
        self.assertFalse(result["passed"])
        self.assertIn("position setup was rejected", result["error"])

    def test_forbidden_fixture_rejects_null_or_illegal_bestmove(self):
        check = self.fixture_check(4)
        check = FixtureCheck(
            **{
                **check.__dict__,
                "expected_move": "!a1b1",
            }
        )
        for bestmove in ("0000", "a1a8", "invalid"):
            with self.subTest(bestmove=bestmove), tempfile.TemporaryDirectory() as directory:
                engine = self.fake_engine(
                    directory,
                    "info depth 4 score cp 0 nodes 10 pv a1a2",
                    bestmove=bestmove,
                )
                result = run_check(
                    engine,
                    check,
                    5.0,
                    DEFAULT_HASH_MB,
                )
            self.assertFalse(result["passed"])
            self.assertIn("bestmove", result["error"])

    def test_run_check_rejects_nonzero_engine_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = self.fake_engine(
                directory,
                "info depth 4 score cp 0 nodes 10 pv a1a2",
                exit_code=7,
            )
            result = run_check(engine, self.fixture_check(4), 5.0, DEFAULT_HASH_MB)
        self.assertFalse(result["passed"])
        self.assertEqual(result["error"], "engine exited with status 7")

    def test_position_summary_compares_pass_counts_across_depths(self):
        def row(line, baseline, candidate):
            return {
                "check": {
                    "fixture": "cases.tsv",
                    "line_number": line,
                    "activation": "disabled",
                },
                "baseline": {"passed": baseline, "error": None},
                "candidate": {"passed": candidate, "error": None},
                "direction": direction(baseline, candidate),
            }

        summary = summarize(
            [
                row(10, True, False),
                row(10, False, True),
                row(11, False, True),
            ]
        )["all/all"]

        self.assertEqual(summary["positions"], 2)
        self.assertEqual(summary["baseline-better-positions"], 0)
        self.assertEqual(summary["candidate-better-positions"], 1)
        self.assertEqual(summary["equal-positions"], 1)
        self.assertEqual(
            summary["disabled-status"],
            {"lost-while-disabled": 1, "fixed-by-candidate": 2},
        )


def gate_row(
    activation,
    fixture,
    line,
    case_id,
    baseline_passed,
    candidate_passed,
    baseline_move="-",
    candidate_move="-",
    expected="a1a2",
    depth=4,
):
    return {
        "check": {
            "activation": activation,
            "fixture": fixture,
            "line_number": line,
            "fixture_format": "standard",
            "variant": "standard",
            "case_id": case_id,
            "depth": depth,
            "expected_move": expected,
        },
        "baseline": {"passed": baseline_passed, "error": None, "bestmove": baseline_move},
        "candidate": {"passed": candidate_passed, "error": None, "bestmove": candidate_move},
        "direction": direction(baseline_passed, candidate_passed),
    }


class FixtureGateTests(unittest.TestCase):
    def test_counts_only_active_cases(self):
        rows = [
            gate_row("active", "a.tsv", 1, "a1", True, True),
            gate_row("active", "a.tsv", 2, "a2", True, False),
            gate_row("disabled", "a.tsv", 3, "a3", True, False),
            gate_row("disabled", "a.tsv", 4, "a4", False, True),
        ]
        gate = evaluate_gate(
            rows,
            net_tolerance_permille=1000,
            floor_ratio_permille=0,
        )
        self.assertEqual(gate["active_checks"], 2)
        self.assertEqual(gate["baseline_passes"], 2)
        self.assertEqual(gate["candidate_passes"], 1)
        self.assertEqual(len(gate["regressed"]), 1)
        self.assertEqual(len(gate["fixed"]), 0)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["reasons"], [])

    def test_net_loss_within_tolerance_passes(self):
        rows = [
            gate_row("active", "a.tsv", i, f"a{i}", True, i != 99)
            for i in range(100)
        ]
        gate = evaluate_gate(rows, net_tolerance_permille=15)
        self.assertEqual(gate["net_loss"], 1)
        self.assertEqual(gate["net_tolerance"], 1)
        self.assertTrue(gate["passed"])

    def test_net_loss_uses_exact_ratio_instead_of_rounded_allowance(self):
        rows = [
            gate_row("active", "a.tsv", i, f"a{i}", True, i != 33)
            for i in range(34)
        ]
        gate = evaluate_gate(rows, net_tolerance_permille=15)
        self.assertEqual(gate["net_tolerance"], 0)
        self.assertFalse(gate["passed"])

    def test_net_loss_beyond_tolerance_fails(self):
        rows = [
            gate_row("active", "a.tsv", i, f"a{i}", True, i < 96)
            for i in range(100)
        ]
        gate = evaluate_gate(rows, net_tolerance_permille=10)
        self.assertEqual(gate["net_loss"], 4)
        self.assertEqual(gate["net_tolerance"], 1)
        self.assertFalse(gate["passed"])
        self.assertTrue(any("net loss" in reason for reason in gate["reasons"]))

    def test_candidate_only_fixes_never_fail_gate(self):
        rows = [
            gate_row("active", "a.tsv", 1, "a1", False, True),
            gate_row("active", "a.tsv", 2, "a2", False, True),
        ]
        gate = evaluate_gate(
            rows,
            net_tolerance_permille=1000,
            floor_ratio_permille=0,
        )
        self.assertEqual(gate["baseline_passes"], 0)
        self.assertEqual(gate["candidate_passes"], 2)
        self.assertEqual(gate["net_loss"], -2)
        self.assertTrue(gate["passed"])
        self.assertEqual(len(gate["fixed"]), 2)

    def test_absolute_floor_blocks_collapse(self):
        rows = [
            gate_row("active", "a.tsv", i, f"a{i}", True, i < 79)
            for i in range(100)
        ]
        gate = evaluate_gate(
            rows,
            profile="rearchitecture",
            floor_ratio_permille=800,
        )
        self.assertEqual(gate["floor"], 80)
        self.assertFalse(gate["passed"])
        self.assertTrue(any("below proportional floor" in r for r in gate["reasons"]))

    def test_absolute_floor_is_relative_to_baseline(self):
        rows = [
            gate_row("active", "a.tsv", i, f"a{i}", True, i < 50)
            for i in range(100)
        ]
        gate = evaluate_gate(
            rows,
            profile="rearchitecture",
            net_tolerance_permille=1000,
            floor_ratio_permille=500,
        )
        self.assertEqual(gate["floor"], 50)
        self.assertEqual(gate["candidate_passes"], 50)
        self.assertTrue(gate["passed"])

    def test_hard_layer_regression_fails_regardless_of_net(self):
        rows = [
            gate_row("active", "engine_regressions.tsv", 1, "hard1", True, False),
            gate_row("active", "a.tsv", 2, "a1", True, True),
            gate_row("active", "a.tsv", 3, "a2", True, True),
        ]
        gate = evaluate_gate(rows, hard_fixtures=["engine_regressions.tsv"])
        self.assertFalse(gate["passed"])
        self.assertEqual(len(gate["hard_regressed"]), 1)
        self.assertTrue(any("hard-layer" in r for r in gate["reasons"]))

    def test_hard_layer_fix_is_allowed(self):
        rows = [
            gate_row("active", "engine_regressions.tsv", 1, "hard1", False, True),
            gate_row("active", "a.tsv", 2, "a1", True, True),
        ]
        gate = evaluate_gate(rows, hard_fixtures=["engine_regressions.tsv"])
        self.assertEqual(len(gate["hard_regressed"]), 0)
        self.assertTrue(gate["passed"])

    def test_hard_layer_must_pass_even_when_baseline_also_fails(self):
        rows = [
            gate_row(
                "active", "engine_regressions.tsv", 1, "hard1", False, False
            ),
        ]
        gate = evaluate_gate(
            rows,
            hard_fixtures=["engine_regressions.tsv"],
        )
        self.assertFalse(gate["passed"])
        self.assertEqual(len(gate["hard_failed"]), 1)

    def test_missing_hard_fixture_is_a_configuration_failure(self):
        rows = [gate_row("active", "a.tsv", 1, "a1", True, True)]
        gate = evaluate_gate(
            rows,
            hard_fixtures=["engine_regressions.tsv"],
        )
        self.assertFalse(gate["passed"])
        self.assertEqual(
            gate["missing_hard_fixtures"],
            ["engine_regressions.tsv"],
        )

    def test_zero_baseline_is_not_garbage(self):
        rows = [
            gate_row("active", "a.tsv", 1, "a1", False, True),
            gate_row("active", "a.tsv", 2, "a2", False, False),
        ]
        gate = evaluate_gate(rows)
        self.assertEqual(gate["baseline_passes"], 0)
        self.assertEqual(gate["net_tolerance"], 0)
        self.assertEqual(gate["floor"], 0)
        self.assertTrue(gate["passed"])

    def test_errors_count_as_failures_and_are_reported(self):
        row = gate_row("disabled", "a.tsv", 1, "a1", False, False)
        row["candidate"]["error"] = "engine crashed"
        row["candidate"]["bestmove"] = None
        gate = evaluate_gate([row])
        self.assertEqual(gate["candidate_errors"], 1)
        self.assertFalse(gate["passed"])
        self.assertTrue(any("invalidate" in reason for reason in gate["reasons"]))
        self.assertIn("engine errors", format_gate_report(gate))

    def test_gate_profiles_do_not_combine_unrelated_thresholds(self):
        rows = [
            gate_row("active", "a.tsv", i, f"a{i}", True, i < 79)
            for i in range(100)
        ]
        strict = evaluate_gate(
            rows,
            profile="strict",
            net_tolerance_permille=1000,
            floor_ratio_permille=800,
        )
        rearchitecture = evaluate_gate(
            rows,
            profile="rearchitecture",
            net_tolerance_permille=0,
            floor_ratio_permille=800,
        )
        self.assertTrue(strict["passed"])
        self.assertFalse(rearchitecture["passed"])

    def test_report_lists_regressed_and_fixed(self):
        rows = [
            gate_row(
                "active", "a.tsv", 1, "a1", True, False,
                baseline_move="e2e4", candidate_move="g1f3", expected="e2e4",
            ),
            gate_row(
                "active", "b.tsv", 2, "b1", False, True,
                baseline_move="-", candidate_move="d2d4", expected="d2d4",
            ),
        ]
        gate = evaluate_gate(rows)
        report = format_gate_report(gate)
        self.assertIn("a.tsv:1 a1", report)
        self.assertIn("b.tsv:2 b1", report)
        self.assertIn("gate verdict: PASS", report)


if __name__ == "__main__":
    unittest.main()
