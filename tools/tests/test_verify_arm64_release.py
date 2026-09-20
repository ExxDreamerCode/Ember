import io
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verify_arm64_release import _run_smp_search, bench_behavior, run_smp_search
from smoke_test_uci import run_smoke
from stress_test_uci import (
    run_eof_during_search,
    run_malformed_input_recovery,
    run_queued_quit_during_search,
    run_stress,
)


class FailingTranscript(io.StringIO):
    """Fail once at a chosen line; closing the stream still succeeds."""

    def __init__(self, operation, marker):
        super().__init__()
        self.operation = operation
        self.marker = marker
        self.last_line = ""
        self.failure = OSError("injected transcript failure")
        self.failed = False

    def fail_once(self, operation):
        if not self.failed and self.operation == operation and self.marker in self.last_line:
            self.failed = True
            raise self.failure

    def write(self, line):
        self.last_line = line
        self.fail_once("write")
        return super().write(line)

    def flush(self):
        self.fail_once("flush")
        return super().flush()


def transcript():
    rows = []
    for i in range(1, 9):
        rows += [
            "info depth 8 score cp 12 nodes 100 nps 1000 time 100 pv e2e4 e7e5",
            f"info string bench {i}/8 position{i} depth 8 nodes 100 time 100ms nps 1000",
        ]
    rows.append("info string bench total: 8 positions, depth 8, nodes 800, "
                "time 800ms, nps 1000, signature 123456789abcdef0")
    return "\n".join(rows)


class Arm64ReleaseParityTests(unittest.TestCase):
    def fake_command(self, root, mode):
        script = root / "engine.py"
        script.write_text(
            "import sys, time\n"
            "mode = sys.argv[1]\n"
            "long_search = False\n"
            "for line in sys.stdin:\n"
            "    line = line.strip()\n"
            "    if line == 'uci':\n"
            "        print('id name Ember 1.2.3\\nuciok', flush=True)\n"
            "    elif line == 'isready':\n"
            "        print('readyok', flush=True)\n"
            "    elif line.startswith('setoption name NNUEBackend'):\n"
            "        print('info string NNUE backend set to auto (aarch64-simd256)', flush=True)\n"
            "    elif line.startswith('go '):\n"
            "        long_search = line == 'go depth 16'\n"
            "        print('info string search started', flush=True)\n"
            "        if mode == 'search-timeout': time.sleep(60)\n"
            "        print('info depth 4 score cp 0 nodes 10 nps 100 pv e2e4', flush=True)\n"
            "        print('bestmove e2e4', flush=True)\n"
            "    elif line == 'quit':\n"
            "        print('info string shutdown diagnostic', flush=True)\n"
            "        if mode == 'shutdown-timeout': time.sleep(60)\n"
            "        if mode == 'shutdown-error' or (mode == 'queued-error' and long_search): sys.exit(3)\n"
            "        break\n"
            "else:\n"
            "    print('info string EOF diagnostic', flush=True)\n"
            "    if mode == 'eof-error': sys.exit(3)\n",
            encoding="utf-8",
        )
        return [sys.executable, str(script), mode]

    def test_transcript_failures_reach_caller_after_clean_shutdown(self):
        scenarios = [
            ("queued-quit", run_queued_quit_during_search, "shutdown diagnostic"),
            ("search-eof", run_eof_during_search, "EOF diagnostic"),
            ("malformed", run_malformed_input_recovery, "search started"),
            ("smp", _run_smp_search, "search started"),
        ]
        for operation in ("write", "flush"):
            for name, run, marker in scenarios:
                with self.subTest(operation=operation, scenario=name):
                    self.check_transcript_failure(operation, marker, run)
            # Failure after bestmove used to pass smoke validation as well.
            with self.subTest(operation=operation, scenario="smoke"):
                self.check_transcript_failure(operation, "shutdown diagnostic", None)

    def check_transcript_failure(self, operation, marker, run):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cargo = root / "Cargo.toml"
            cargo.write_text('[package]\nversion = "1.2.3"\n')
            command = self.fake_command(root, "normal")
            processes = []
            popen = subprocess.Popen

            def record_process(*args, **kwargs):
                process = popen(*args, **kwargs)
                processes.append(process)
                return process

            with FailingTranscript(operation, marker) as transcript:
                with patch("subprocess.Popen", side_effect=record_process):
                    with self.assertRaisesRegex(RuntimeError, "UCI output capture failed") as error:
                        if run is None:
                            run_smoke(command, cargo, 5.0, transcript)
                        else:
                            run(command, 5.0, transcript)
                self.assertTrue(transcript.failed)
                self.assertIs(error.exception.__cause__, transcript.failure)
                self.assertEqual(len(processes), 1)
                self.assertEqual(processes[0].poll(), 0)
                self.assertTrue(processes[0].stdin.closed)
                self.assertTrue(processes[0].stdout.closed)

    def test_smoke_failure_streams_transcript(self):
        for mode, expected in [("search-timeout", "search started"),
                               ("shutdown-timeout", "shutdown diagnostic")]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                cargo = root / "Cargo.toml"
                cargo.write_text('[package]\nversion = "1.2.3"\n')
                log_path = root / "smoke.log"
                with log_path.open("x") as log, self.assertRaises(TimeoutError):
                    run_smoke(self.fake_command(root, mode), cargo, 1.0, log)
                self.assertIn(expected, log_path.read_text())

    def test_stress_preserves_successes_before_later_failure(self):
        for mode, failed, marker in [("queued-error", "queued-quit", "shutdown diagnostic"),
                                     ("eof-error", "search-eof", "EOF diagnostic")]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.assertRaisesRegex(RuntimeError, "exited with 3"):
                    run_stress(self.fake_command(root, mode), 5.0, root, "case-")
                self.assertIn("bestmove", (root / "case-malformed-input.log").read_text())
                self.assertIn(marker, (root / f"case-{failed}.log").read_text())

    def test_smp_failure_preserves_shutdown_output(self):
        for mode, exception in [("shutdown-error", RuntimeError),
                                ("shutdown-timeout", TimeoutError)]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = root / "smp.log"
                with self.assertRaises(exception):
                    run_smp_search(self.fake_command(root, mode), 1.0, path)
                output = path.read_text()
                self.assertEqual(output.count("bestmove e2e4"), 2)
                self.assertIn("shutdown diagnostic", output)

    def test_ignores_timing_but_preserves_behavior(self):
        original = transcript()
        expected = bench_behavior(original, 8)
        self.assertEqual(expected, bench_behavior(original.replace("nps 1000", "nps 2000")
                                                  .replace("time 100", "time 50"), 8))
        for before, after in [("cp 12", "cp 13"),
                              ("e2e4", "d2d4"), ("123456789abcdef0", "023456789abcdef0")]:
            with self.subTest(field=before):
                self.assertNotEqual(expected, bench_behavior(original.replace(before, after), 8))
        self.assertNotEqual(expected, bench_behavior(original.replace("nodes 100", "nodes 101")
                                                    .replace("nodes 800", "nodes 808"), 8))

    def test_rejects_missing_and_partial_workloads(self):
        for output in ["", transcript().replace("bench 4/8", "bench 3/8"),
                       transcript().replace("8 positions", "7 positions"),
                       transcript().replace("score cp", "score bogus"),
                       transcript().replace("info depth 8", "info depth 7"),
                       transcript().replace("nodes 800", "nodes 801"),
                       transcript().replace("info depth 8 score cp 12 nodes 100 nps 1000 time 100 pv e2e4 e7e5\n", "", 1)]:
            with self.subTest(output=output), self.assertRaises(ValueError):
                bench_behavior(output, 8)
        with self.assertRaises(ValueError):
            bench_behavior(transcript(), 9)
