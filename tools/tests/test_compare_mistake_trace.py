import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from compare_mistake_trace import (  # noqa: E402
    RootDag,
    TraceNode,
    compare_witness_positions,
    parse_fixture_rows,
    position_key,
)


class CompareMistakeTraceTests(unittest.TestCase):
    def test_parse_fixture_rows_includes_commented_disabled_rows(self):
        content = "\n".join(
            [
                "# comment",
                "id\tdepth\tfen_before_blunder\tsetup_move\texpected_move\tthemes\trating\tpopularity\tplays",
                "",
                "# DISABLED: known issue",
                "# advpres-test\t8\tfen text\te2e4 e7e5\t!g1f3\tbucket,theme\t0\t0\t0",
                "active-test\t1\tfen text\t-\te2e4\tbook\t1\t2\t3",
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.tsv"
            path.write_text(content + "\n", encoding="utf-8")

            rows = parse_fixture_rows(path)

        self.assertEqual([row.case_id for row in rows], ["advpres-test", "active-test"])
        self.assertTrue(rows[0].disabled)
        self.assertEqual(rows[0].forbidden_moves, ["g1f3"])
        self.assertFalse(rows[1].disabled)
        self.assertEqual(rows[1].setup_moves, [])

    def test_position_key_ignores_fifty_move_and_fullmove_counters(self):
        self.assertEqual(
            position_key("8/8/8/8/8/8/8/K6k w - - 17 91"),
            position_key("8/8/8/8/8/8/8/K6k w - - 99 122"),
        )

    def test_compare_witness_positions_reports_first_missing_ply(self):
        dag = RootDag(
            root={"move": "a1a2", "score": 10},
            nodes_by_key={
                "fen-a": TraceNode("fen-a", "fen-a 0 1", {"main_visits": 2}),
                "fen-b": TraceNode("fen-b", "fen-b 0 1", {"main_visits": 1}),
            },
        )
        witness = [
            {"ply": 1, "move": "a1a2", "fen": "fen-a 0 1", "key": "fen-a"},
            {"ply": 2, "move": "a8a7", "fen": "fen-b 0 1", "key": "fen-b"},
            {"ply": 3, "move": "a2a3", "fen": "fen-c 0 1", "key": "fen-c"},
        ]

        result = compare_witness_positions(dag, witness)

        self.assertEqual(result["visited_prefix_plies"], 2)
        self.assertEqual(result["first_missing"]["ply"], 3)
        self.assertEqual([row["visited"] for row in result["positions"]], [True, True, False])


if __name__ == "__main__":
    unittest.main()
