import importlib.util
import pathlib
import unittest


MODULE_PATH = pathlib.Path(__file__).parents[1] / "analyze_singular_trace.py"
SPEC = importlib.util.spec_from_file_location("analyze_singular_trace", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def event(**overrides):
    result = {
        "fen": "8/8/8/8/8/8/5K2/7k w - - 0 1",
        "move": "f2f3",
        "depth": 12,
        "pv": False,
        "tt_flag": 0,
        "verification_nodes": 100,
        "verification_score": 20,
        "threshold": 30,
        "capture": False,
        "promotion": False,
        "outcome": "rejected",
    }
    result.update(overrides)
    return result


class TraceParsingTests(unittest.TestCase):
    def test_parse_trace_line_ignores_unrelated_output(self):
        self.assertIsNone(MODULE.parse_trace_line("info depth 12 nodes 123"))

    def test_parse_trace_line_accepts_prefixed_worker_output(self):
        payload = event()
        line = "worker 2: " + MODULE.TRACE_PREFIX + MODULE.json.dumps(payload)
        self.assertEqual(MODULE.parse_trace_line(line), payload)

    def test_parse_trace_line_rejects_incomplete_events(self):
        with self.assertRaisesRegex(ValueError, "missing fields"):
            MODULE.parse_trace_line(MODULE.TRACE_PREFIX + '{"fen":"x"}')


class TraceSummaryTests(unittest.TestCase):
    def test_summary_reports_verification_economics(self):
        events = [
            event(outcome="extended", verification_nodes=100),
            event(outcome="rejected", verification_nodes=200),
            event(outcome="rejected", verification_nodes=300),
        ]
        overall = MODULE.summarize(events)["overall"]
        self.assertEqual(overall["candidates"], 3)
        self.assertEqual(overall["outcomes"], {"extended": 1, "rejected": 2})
        self.assertAlmostEqual(overall["extension_rate"], 1 / 3)
        self.assertEqual(overall["verification_nodes"]["total"], 600)
        self.assertEqual(overall["verification_nodes"]["median"], 200)
        self.assertEqual(overall["verification_nodes"]["p95"], 300)
        self.assertEqual(overall["nodes_per_extension"], 600)


class UciParsingTests(unittest.TestCase):
    def test_parse_centipawn_line(self):
        parsed = MODULE.parse_uci_info(
            "info depth 20 multipv 2 score cp -37 nodes 1234 pv e2e4 e7e5"
        )
        self.assertEqual(parsed["multipv"], 2)
        self.assertEqual(parsed["score_cp"], -37)
        self.assertEqual(parsed["pv"], ["e2e4", "e7e5"])

    def test_parse_mate_line(self):
        winning = MODULE.parse_uci_info("info score mate 3 pv e2e4")
        losing = MODULE.parse_uci_info("info score mate -4 pv e2e4")
        self.assertEqual(winning["score_cp"], MODULE.MATE_SCORE - 3)
        self.assertEqual(losing["score_cp"], -MODULE.MATE_SCORE + 4)


class FakeOracle:
    def __init__(self):
        self.calls = []

    def search(self, fen, nodes, multipv=1, searchmoves=None):
        self.calls.append((fen, nodes, multipv, searchmoves))
        if searchmoves:
            return [{"score_cp": 120, "pv": [searchmoves[0], "a7a6"]}]
        return [
            {"score_cp": 120, "pv": ["e2e4", "a7a6"]},
            {"score_cp": 20, "pv": ["d2d4", "a7a6"]},
        ]


class OracleLabelTests(unittest.TestCase):
    def test_labels_are_deduplicated_by_position_and_move(self):
        oracle = FakeOracle()
        events = [
            event(move="e2e4", outcome="extended"),
            event(move="e2e4", outcome="rejected"),
        ]
        labeled = MODULE.label_events(events, oracle, nodes=1000, gap_cp=80)
        self.assertEqual(len(oracle.calls), 2)
        self.assertEqual(len(labeled), 2)
        self.assertTrue(labeled[0]["oracle"]["oracle_singular"])
        self.assertEqual(labeled[0]["oracle"]["candidate_gap_cp"], 100)


if __name__ == "__main__":
    unittest.main()
