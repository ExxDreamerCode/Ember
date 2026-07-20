import sys
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from compare_versions import (  # noqa: E402
    compare_outcome_records,
    engine_args,
    make_scenario_specs,
    summarize,
)


def opponent(name, band, ponder):
    return {
        "name": name,
        "cmd": name.lower(),
        "proto": "uci",
        "band": band,
        "supports_ponder": ponder,
        "options": {"Threads": "1"},
    }


class ScenarioScheduleTests(unittest.TestCase):
    def setUp(self):
        self.opponents = {
            "weaker": [opponent("Weak-A", "weaker", True), opponent("Weak-B", "weaker", False)],
            "stronger": [opponent("Strong-A", "stronger", True)],
        }

    def test_seed_reproduces_balanced_schedule(self):
        first = make_scenario_specs(
            self.opponents, ["1+0.01", "8+0.08"], [False, True], 17, 1234, True
        )
        second = make_scenario_specs(
            self.opponents, ["1+0.01", "8+0.08"], [False, True], 17, 1234, True
        )
        different = make_scenario_specs(
            self.opponents, ["1+0.01", "8+0.08"], [False, True], 17, 1235, True
        )

        self.assertEqual(first, second)
        self.assertNotEqual(first, different)
        weak = sum(spec["band"] == "weaker" for spec in first)
        strong = sum(spec["band"] == "stronger" for spec in first)
        self.assertLessEqual(abs(weak - strong), 1)
        self.assertTrue(any(spec["ponder"] for spec in first))
        self.assertTrue(
            all(not spec["ponder"] or spec["opponent"]["supports_ponder"] for spec in first)
        )

    def test_ponder_is_removed_when_either_ember_cannot_ponder(self):
        specs = make_scenario_specs(
            self.opponents, ["3+0.03"], [False, True], 10, 9, False
        )
        self.assertTrue(all(not spec["ponder"] for spec in specs))


class OutcomeComparisonTests(unittest.TestCase):
    def test_games_are_aligned_by_scenario_and_ember_color(self):
        manifest = {
            "scenarios": [
                {
                    "id": "scenario-0001",
                    "band": "stronger",
                    "opponent": {"name": "Strong-A"},
                    "time_control": "8+0.08",
                    "ponder": False,
                    "opening_epd": "start",
                }
            ]
        }
        baseline = {
            ("scenario-0001", "white"): self.record("white", 1.0, "win", "base.pgn", 1),
            ("scenario-0001", "black"): self.record("black", 0.5, "draw", "base.pgn", 2),
        }
        candidate = {
            ("scenario-0001", "black"): self.record("black", 1.0, "win", "new.pgn", 1),
            ("scenario-0001", "white"): self.record("white", 0.0, "loss", "new.pgn", 2),
        }

        rows = compare_outcome_records(manifest, baseline, candidate)

        self.assertEqual([row["ember_color"] for row in rows], ["white", "black"])
        self.assertEqual([row["change"] for row in rows], ["regression", "improvement"])
        self.assertEqual([row["score_delta"] for row in rows], [-1.0, 0.5])

    def test_clustered_statistics_are_seeded_and_count_time_losses(self):
        rows = []
        for index, (before, after) in enumerate(((1.0, 0.0), (0.5, 1.0), (0.0, 0.5))):
            rows.append(self.row(f"s{index}", "white", before, after, "White loses on time"))
            rows.append(self.row(f"s{index}", "black", before, after, "normal"))

        first = summarize(rows, 42, 200, 500)
        second = summarize(rows, 42, 200, 500)

        self.assertEqual(first, second)
        self.assertEqual(first["candidate_time_forfeit_losses"], 1)
        self.assertEqual(first["games_per_version"], 6)
        self.assertTrue(any(item["dimension"] == "time_control" for item in first["breakdowns"]))
        self.assertTrue(
            all("paired_randomization_p_two_sided" in item for item in first["breakdowns"])
        )

    @staticmethod
    def record(color, score, outcome, pgn, game):
        return {
            "ember_color": color,
            "score": score,
            "outcome": outcome,
            "result": "1-0",
            "termination": "normal",
            "pgn": pgn,
            "pgn_game": game,
        }

    @staticmethod
    def row(scenario, color, before, after, candidate_termination):
        names = {0.0: "loss", 0.5: "draw", 1.0: "win"}
        return {
            "scenario": scenario,
            "band": "stronger",
            "opponent": "Strong-A",
            "time_control": "8+0.08",
            "ponder": False,
            "ember_color": color,
            "baseline_score": before,
            "candidate_score": after,
            "baseline_outcome": names[before],
            "candidate_outcome": names[after],
            "baseline_termination": "normal",
            "candidate_termination": candidate_termination,
            "change": "regression" if after < before else "improvement" if after > before else "unchanged",
            "transition": f"{names[before]} -> {names[after]}",
        }


class CuteChessArgumentTests(unittest.TestCase):
    def test_ponder_is_a_bare_engine_flag(self):
        args = engine_args(opponent("Engine", "weaker", True), ponder=True)
        self.assertIn("ponder", args)
        self.assertNotIn("option.Ponder=true", args)


if __name__ == "__main__":
    unittest.main()
