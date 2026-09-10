import sys
import tempfile
import types
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from advantage_fixture import write_fixture  # noqa: E402
from hunt_lost_advantage import (  # noqa: E402
    fixture_rejection,
    parse_args,
    resolve_fixture_output,
)


def valid_case(**overrides):
    values = dict(
        case_id="advpres-test",
        seed=1,
        source_game_index=1,
        stockfish_color="white",
        anchor_ply=10,
        anchor_move="e2e4",
        anchor_advantage_cp=350,
        previous_advantage_cp=100,
        replay_start_fen="startpos",
        bad_ply=20,
        bad_fen="test fen",
        bad_move="a1a2",
        bad_depth=8,
        bad_score_cp_before=300,
        bad_score_cp_after=100,
        stockfish_best_move="a1b1",
        stockfish_best_score_cp=300,
        eval_loss_cp=200,
        replay_result="0-1",
        termination="normal",
        bucket="quiet-advantage-loss",
        history_before_bad=["e2e4", "e7e5"],
        replay_moves=["e2e4", "e7e5", "a1a2"],
        source_pgn="source.pgn",
        replay_pgn="replay.pgn",
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


class FixtureOutputTests(unittest.TestCase):
    def test_new_cases_are_report_only(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "cases.tsv"
            write_fixture(fixture, [valid_case()])
            contents = fixture.read_text(encoding="utf-8")

        self.assertIn("# DISABLED: collected for bucket triage", contents)
        self.assertIn("# advpres-test\t8\t", contents)
        self.assertNotIn("\nadvpres-test\t8\t", contents)

    def test_rejects_move_that_matches_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "matches the Stockfish best move"):
                write_fixture(
                    Path(directory) / "cases.tsv",
                    [valid_case(stockfish_best_move="a1a2")],
                )

    def test_rejects_missing_reference_move(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "missing Stockfish best move"):
                write_fixture(
                    Path(directory) / "cases.tsv",
                    [valid_case(stockfish_best_move=None)],
                )

    def test_rejects_missing_or_nonpositive_loss_evidence(self):
        for loss in (None, 0, -1):
            with self.subTest(loss=loss), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, "positive evaluation-loss"):
                    write_fixture(
                        Path(directory) / "cases.tsv",
                        [valid_case(eval_loss_cp=loss)],
                    )

    def test_hunt_rejects_invalid_case_before_counting_it(self):
        self.assertIn(
            "matches the Stockfish best move",
            fixture_rejection(valid_case(stockfish_best_move="a1a2")),
        )
        self.assertIsNone(fixture_rejection(valid_case()))

    def test_default_fixture_stays_inside_the_result_directory(self):
        args = parse_args(["--ember", "ember"])
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "run"
            self.assertEqual(
                resolve_fixture_output(args, run_root),
                run_root / "advantage_preservation.tsv",
            )

    def test_existing_fixture_requires_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "cases.tsv"
            fixture.write_text("preserve me\n", encoding="utf-8")
            args = parse_args(
                ["--ember", "ember", "--fixture-output", str(fixture)]
            )
            with self.assertRaisesRegex(FileExistsError, "--overwrite-fixture"):
                resolve_fixture_output(args, Path(directory) / "run")

            args.overwrite_fixture = True
            self.assertEqual(
                resolve_fixture_output(args, Path(directory) / "run"),
                fixture,
            )


if __name__ == "__main__":
    unittest.main()
