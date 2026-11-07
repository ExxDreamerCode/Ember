import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

TOOLS = pathlib.Path(__file__).parents[1]
sys.path.insert(0, str(TOOLS))
import analyze_ldse_trace as analysis


def event(**changes):
    data = {
        "hash": "1234", "fen": "8/8/8/8/8/8/5K2/7k w - - 0 1", "move": "f2f3",
        "depth": 3, "alpha": 49, "beta": 50, "eval": -100, "tt_score": 80,
        "line_uses": 0, "probe_depth": 2, "probe_score": 51, "probe_nodes": 7,
        "outcome": "cutoff",
    }
    data.update(changes)
    return data


class TraceTests(unittest.TestCase):
    def test_summary_counts_rejected_probe_costs(self):
        summary = analysis.summarize([event(), event(outcome="rejected", probe_nodes=13)])
        self.assertEqual(summary["probes"], 2)
        self.assertEqual(summary["probe_nodes"], 20)
        self.assertEqual(summary["nodes_per_cutoff"], 20)
        self.assertEqual(summary["cutoff_rate"], 0.5)

    def test_reader_rejects_incomplete_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "trace"
            path.write_text(analysis.PREFIX + json.dumps({"fen": "x"}) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                analysis.read_events([path])
            path.write_text(analysis.PREFIX + "[]\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                analysis.read_events([path])
            path.write_text(analysis.PREFIX + json.dumps(event(probe_nodes=-1)) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                analysis.read_events([path])

    def test_oracle_scores_candidate_against_each_node_window(self):
        class Oracle:
            def __init__(self):
                self.calls = []

            def search(self, fen, nodes, multipv, searchmoves=None):
                self.calls.append((fen, nodes, multipv, searchmoves))
                return [{"score_cp": 40 if searchmoves else 80, "pv": ["f2f3"]}]

        oracle = Oracle()
        labels = analysis.label_events([event(), event(beta=30)], oracle, 1000, 10)
        self.assertEqual(len(oracle.calls), 2)
        self.assertFalse(labels[0]["oracle_move_reaches_beta"])
        self.assertTrue(labels[1]["oracle_move_reaches_beta"])
        mate_window = analysis.label_events([event(beta=90_000)], oracle, 1000, 10)
        self.assertIsNone(mate_window[0]["oracle_move_reaches_beta"])

        other = event(fen="8/8/8/8/8/8/4K3/7k w - - 0 1")
        first = analysis.label_events([event(), other], Oracle(), 1000, 1)
        second = analysis.label_events([other, event()], Oracle(), 1000, 1)
        self.assertEqual(first[0]["fen"], second[0]["fen"])

        rejected = event(fen="8/8/8/8/8/8/3K4/7k w - - 0 1", outcome="rejected")
        stratified = analysis.label_events([event(), other, rejected], Oracle(), 1000, 2)
        self.assertEqual({row["outcome"] for row in stratified}, {"cutoff", "rejected"})

    @unittest.skipIf(os.name == "nt", "fixture uses an executable Python shebang")
    def test_oracle_does_not_deadlock_on_stderr(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = pathlib.Path(tmp) / "oracle"
            fake.write_text(
                f"#!{sys.executable}\n"
                "import sys\n"
                "for line in sys.stdin:\n"
                " if line.strip() == 'uci':\n"
                "  sys.stderr.write('x' * 131072); sys.stderr.flush()\n"
                "  print('uciok', flush=True)\n"
                " elif line.strip() == 'isready': print('readyok', flush=True)\n"
                " elif line.strip() == 'quit': break\n",
                encoding="utf-8",
            )
            fake.chmod(0o755)
            driver = (
                "import sys; from pathlib import Path; "
                "sys.path.insert(0, sys.argv[1]); "
                "from analyze_ldse_trace import TraceOracle; "
                "oracle=TraceOracle(Path(sys.argv[2]), 1); oracle.close()"
            )
            result = subprocess.run(
                [sys.executable, "-c", driver, str(TOOLS), str(fake)],
                capture_output=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr[-1000:])
            self.assertEqual(len(result.stderr), 131072)

    def test_oracle_nonzero_exit_is_rejected(self):
        class FailedProcess:
            stdin = None
            stdout = None

            def poll(self):
                return 7

            def wait(self, timeout):
                return 7

        oracle = object.__new__(analysis.TraceOracle)
        oracle.process = FailedProcess()
        oracle.logging_error = None
        with self.assertRaisesRegex(RuntimeError, "status 7"):
            oracle.close()

    def test_transcript_failure_still_drains_the_response(self):
        class BrokenTranscript:
            def write(self, text):
                raise OSError("disk full")

        class Process:
            stdout = iter(["info depth 1\n", "bestmove f2f3\n"])

        oracle = object.__new__(analysis.TraceOracle)
        oracle.process = Process()
        oracle.transcript = BrokenTranscript()
        oracle.logging_error = None
        with self.assertRaisesRegex(RuntimeError, "transcript write failed"):
            oracle._read_until("bestmove")
        self.assertEqual(list(oracle.process.stdout), [])

    @unittest.skipIf(os.name == "nt", "fixture uses an executable Python shebang")
    def test_chess960_requires_oracle_option(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = pathlib.Path(tmp) / "oracle"
            fake.write_text(
                f"#!{sys.executable}\n"
                "import sys\n"
                "for line in sys.stdin:\n"
                " if line.strip() == 'uci': print('uciok', flush=True)\n"
                " elif line.strip() == 'quit': break\n",
                encoding="utf-8",
            )
            fake.chmod(0o755)
            with self.assertRaisesRegex(RuntimeError, "UCI_Chess960"):
                analysis.TraceOracle(fake, 1, chess960=True)

    @unittest.skipIf(os.name == "nt", "fixture uses an executable Python shebang")
    def test_cli_records_oracle_transcript_and_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            fake = root / "oracle"
            fake.write_text(
                f"#!{sys.executable}\n"
                "import sys\n"
                "for line in sys.stdin:\n"
                " command=line.strip()\n"
                " if command == 'uci':\n"
                "  print('option name UCI_Chess960 type check default false', flush=True)\n"
                "  print('uciok', flush=True)\n"
                " elif command == 'isready': print('readyok', flush=True)\n"
                " elif command.startswith('go '):\n"
                "  print('info depth 1 multipv 1 score cp 40 nodes 10 pv f2f3', flush=True)\n"
                "  print('bestmove f2f3', flush=True)\n"
                " elif command == 'quit': break\n",
                encoding="utf-8",
            )
            fake.chmod(0o755)
            trace = root / "events.log"
            trace.write_text(analysis.PREFIX + json.dumps(event()) + "\n", encoding="utf-8")
            output = root / "report.json"
            transcript = root / "oracle.log"
            result = subprocess.run(
                [sys.executable, str(TOOLS / "analyze_ldse_trace.py"), str(trace),
                 "--oracle", str(fake), "--chess960", "--nodes", "10", "--transcript", str(transcript),
                 "--output", str(output)],
                capture_output=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(report["oracle"]["events"][0]["oracle_move_reaches_beta"])
            self.assertEqual(report["oracle"]["variant"], "Chess960")
            self.assertIn("> setoption name UCI_Chess960 value true", transcript.read_text(encoding="utf-8"))
            self.assertIn("> go nodes 10 searchmoves f2f3", transcript.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
