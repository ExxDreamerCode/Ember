import json
import tempfile
import unittest
from pathlib import Path

import sys

TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from compare_search_dags import compare_search_dags, load_search_dag  # noqa: E402


def write_trace(path, records):
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def root(sequence, positions, edges, move="a1a2"):
    return {
        "type": "root",
        "sequence": sequence,
        "depth": 4,
        "move": move,
        "score": 0,
        "searched_nodes": 10,
        "positions": positions,
        "edges": edges,
        "truncated": False,
    }


def node(sequence, node_hash, evaluation):
    return {
        "type": "node",
        "sequence": sequence,
        "hash": node_hash,
        "fen": f"fen-{node_hash}",
        "eval_visits": 1,
        "min_eval": evaluation,
        "max_eval": evaluation,
    }


def edge(sequence, parent, child):
    return {
        "type": "edge",
        "sequence": sequence,
        "parent": parent,
        "child": child,
        "visits": 1,
    }


class CompareSearchDagTests(unittest.TestCase):
    def test_compare_search_dags_finds_frontier_and_evaluation_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline_path = Path(directory) / "baseline.jsonl"
            candidate_path = Path(directory) / "candidate.jsonl"
            write_trace(
                baseline_path,
                [
                    root(1, 3, 2),
                    node(1, "common-parent", 10),
                    node(1, "common-child", 20),
                    node(1, "baseline-only", 30),
                    edge(1, "common-parent", "common-child"),
                    edge(1, "common-child", "baseline-only"),
                ],
            )
            write_trace(
                candidate_path,
                [
                    root(1, 3, 2),
                    node(1, "common-parent", 10),
                    node(1, "common-child", 25),
                    node(1, "candidate-only", 40),
                    edge(1, "common-parent", "common-child"),
                    edge(1, "common-child", "candidate-only"),
                ],
            )

            result = compare_search_dags(
                load_search_dag(baseline_path), load_search_dag(candidate_path)
            )

        self.assertEqual(
            result["summary"],
            {
                "common_positions": 2,
                "baseline_only_positions": 1,
                "candidate_only_positions": 1,
                "common_positions_with_different_evaluation": 1,
                "baseline_truncated": False,
                "candidate_truncated": False,
            },
        )
        self.assertEqual(
            result["baseline_only_frontier"],
            {
                "baseline-only": [
                    {
                        "fen": "fen-baseline-only",
                        "parent": "common-child",
                        "parent_fen": "fen-common-child",
                        "visits": 1,
                    }
                ]
            },
        )
        self.assertEqual(
            result["candidate_only_frontier"],
            {
                "candidate-only": [
                    {
                        "fen": "fen-candidate-only",
                        "parent": "common-child",
                        "parent_fen": "fen-common-child",
                        "visits": 1,
                    }
                ]
            },
        )
        self.assertEqual(result["evaluation_deltas"][0]["hash"], "common-child")

    def test_load_search_dag_selects_last_matching_root(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            write_trace(
                path,
                [
                    root(1, 1, 0),
                    node(1, "old", 10),
                    root(2, 1, 0),
                    node(2, "new", 20),
                ],
            )

            trace = load_search_dag(path, depth=4, move="a1a2")

        self.assertEqual(trace.root["sequence"], 2)
        self.assertEqual(set(trace.nodes), {"new"})

    def test_load_search_dag_rejects_incomplete_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            write_trace(path, [root(1, 2, 0), node(1, "only-one", 10)])

            with self.assertRaisesRegex(ValueError, "declares 2 positions"):
                load_search_dag(path)
