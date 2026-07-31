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
    move_matches,
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


if __name__ == "__main__":
    unittest.main()
