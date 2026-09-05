import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from compare_fixture_corpus import (  # noqa: E402
    DEFAULT_HASH_MB,
    direction,
    disabled_status,
    evaluate_gate,
    format_gate_report,
    move_matches,
    parse_uci_option,
    parse_fixture,
    summarize,
    uci_setup_commands,
)


class CompareFixtureCorpusTests(unittest.TestCase):
    def test_parses_active_and_both_disabled_formats(self):
        contents = """\
# source comment
id\tdepth\tfen_before_blunder\tsetup_move\texpected_move\tthemes\trating\tpopularity\tplays
active\t4\tactive fen\t-\ta1a2\ttheme\t0\t0\t0
book\t0\tbook fen\t-\ta2a3\tbook\t0\t0\t0
# disabled\t7\tdisabled fen\t-\tb1b2|b1c3\ttheme\t0\t0\t0
# failed_id\tfen_before_blunder\tsetup_move\texpected_move\tgot_depth2\tgot_depth3\tgot_depth4\tthemes\trating\tpopularity\tplays
# mined\tmined fen\tc1c2\td1d2\te1e2\te1e3\te1e4\ttheme\t0\t0\t0
"""
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
        self.assertEqual(gate["net_tolerance"], 2)
        self.assertTrue(gate["passed"])

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
        gate = evaluate_gate(rows, floor_ratio_permille=800)
        self.assertEqual(gate["floor"], 80)
        self.assertFalse(gate["passed"])
        self.assertTrue(any("below absolute floor" in r for r in gate["reasons"]))

    def test_absolute_floor_is_relative_to_baseline(self):
        rows = [
            gate_row("active", "a.tsv", i, f"a{i}", True, i < 50)
            for i in range(100)
        ]
        gate = evaluate_gate(
            rows,
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
        row = gate_row("active", "a.tsv", 1, "a1", True, False)
        row["candidate"]["error"] = "engine crashed"
        row["candidate"]["bestmove"] = None
        gate = evaluate_gate([row])
        self.assertEqual(gate["candidate_errors"], 1)
        self.assertIn("engine errors", format_gate_report(gate))

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
