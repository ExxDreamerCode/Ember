import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch


TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from benchmark_search import (  # noqa: E402
    bench_once,
    main,
    parse_option_arg,
    uci_input,
    validate_option_transcript,
)


class SearchBenchmarkProtocolTests(unittest.TestCase):
    def test_rejects_inherited_backend_before_creating_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir) / "run"
            argv = ["benchmark_search", "--binary", f"engine={sys.executable}",
                    "--out-dir", temp_dir, "--run-id", "run"]
            with patch.object(sys, "argv", argv), patch.dict(os.environ, {"EMBER_SEARCH_BACKEND": "scalar"}):
                with self.assertRaisesRegex(SystemExit, "unset EMBER_SEARCH_BACKEND"):
                    main()
            self.assertFalse(out.exists())

    def test_failure_preserves_configuration_and_completed_samples(self):
        for interleave in [False, True]:
            with self.subTest(interleave=interleave), tempfile.TemporaryDirectory() as temp_dir:
                out = Path(temp_dir) / "run"
                argv = ["benchmark_search", "--binary", f"engine={sys.executable}",
                        "--out-dir", temp_dir, "--run-id", "run", "--repeats", "1",
                        "--depth", "9", "--hash-mb", "32", "--timeout", "10",
                        "--option", "engine:NNUEBackend=scalar"] + (["--interleave"] if interleave else [])
                calls = 0

                def sample(*args):
                    nonlocal calls
                    invocation = json.loads((out / "invocation.json").read_text())
                    self.assertEqual(invocation["depth"], 9)
                    self.assertEqual(invocation["hash_mb"], 32)
                    self.assertEqual(invocation["timeout"], 10)
                    self.assertEqual(invocation["interleave"], interleave)
                    self.assertEqual(invocation["options"]["engine"], [{"name": "NNUEBackend", "value": "scalar"}])
                    self.assertEqual(len(invocation["binaries"][0]["sha256"]), 64)
                    calls += 1
                    args[-1].write_text("partial transcript")
                    if calls == 2:
                        raise TimeoutError("second search stalled")
                    return {"position": args[1][0], "reported_depth": 9,
                            "nodes": 10, "nps": 10, "wall_seconds": 1.0}

                with patch.object(sys, "argv", argv), patch.dict(os.environ, {"EMBER_SEARCH_BACKEND": ""}), \
                        patch("benchmark_search.bench_once", side_effect=sample):
                    with self.assertRaisesRegex(TimeoutError, "second search stalled"):
                        main()
                result = json.loads((out / "result.json").read_text())
                self.assertEqual(result["status"], "running")
                self.assertEqual(len(result["samples"]), 1)
                self.assertEqual(len(list((out / "raw").glob("*.log"))), 2)
                self.assertFalse((out / "summary.md").exists())

    def test_rejects_mismatched_backend_even_after_an_earlier_matching_ack(self):
        with self.assertRaisesRegex(RuntimeError, "backend mismatch"):
            validate_option_transcript(
                "info string NNUE backend set to aarch64-simd256\n"
                "info string NNUE backend set to scalar\n",
                [("NNUEBackend", "aarch64-simd256")],
            )

    def test_backend_aliases_and_auto_acknowledgements(self):
        for requested, acknowledged in [
            (" ARM64_SIMD256 ", "aarch64-simd256"),
            ("portable", "scalar"),
            ("avx2", "x86-v3"),
            ("v4", "x86-avx512"),
            ("default", "auto (aarch64-simd256)"),
            ("auto", "auto (scalar)"),
            ("", "auto (x86-v3)"),
            ("simd", "x86-avx512"),
            ("simd256", "aarch64-simd256"),
        ]:
            with self.subTest(requested=requested):
                self.assertEqual(
                    validate_option_transcript(
                        f"info string NNUE backend set to {acknowledged}\n",
                        [("NNUEBackend", requested)],
                    ),
                    {"NNUEBackend": acknowledged},
                )
        for requested, acknowledged in [
            ("auto", "scalar"),
            ("scalar", "auto (scalar)"),
            ("simd256", "x86-avx512"),
            ("auto", "auto (unknown)"),
        ]:
            with self.subTest(requested=requested, acknowledged=acknowledged):
                with self.assertRaisesRegex(RuntimeError, "backend mismatch"):
                    validate_option_transcript(
                        f"info string NNUE backend set to {acknowledged}\n",
                        [("NNUEBackend", requested)],
                    )

    def test_transcripts_remain_distinct_for_colliding_labels(self):
        for interleave in [False, True]:
            with self.subTest(interleave=interleave), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                positions = root / "positions.json"
                positions.write_text(json.dumps([
                    {"label": label, "position": "startpos"}
                    for label in ["a+b", "a-b"]
                ]))
                argv = [
                    "benchmark_search", "--binary", f"a+b={sys.executable}",
                    "--binary", f"a-b={sys.executable}", "--positions", str(positions),
                    "--out-dir", str(root), "--run-id", "run", "--repeats", "2",
                ] + (["--interleave"] if interleave else [])

                def sample(*args):
                    transcript = args[-1]
                    transcript.write_text(str(transcript))
                    return {"position": args[1][0], "reported_depth": 6,
                            "nodes": 10, "nps": 10, "wall_seconds": 1.0}

                with patch.object(sys, "argv", argv), patch("benchmark_search.bench_once", side_effect=sample):
                    main()
                result = json.loads((root / "run/result.json").read_text())
                paths = [row["raw_output"] for row in result["samples"]]
                self.assertEqual(len(paths), 8)
                self.assertEqual(len(set(paths)), 8)
                self.assertEqual(len(list((root / "run/raw").glob("*.log"))), 8)
                with patch.object(sys, "argv", argv), patch("benchmark_search.bench_once") as bench:
                    with self.assertRaises(FileExistsError):
                        main()
                    bench.assert_not_called()

    @unittest.skipIf(sys.platform == "win32", "uses a POSIX shebang script")
    def test_timeout_preserves_transcript_during_search_and_shutdown(self):
        for shutdown in [False, True]:
            with self.subTest(shutdown=shutdown), tempfile.TemporaryDirectory() as temp_dir:
                engine = Path(temp_dir) / "hanging_engine"
                raw = Path(temp_dir) / "raw.log"
                engine.write_text(
                    f"#!{sys.executable}\nimport sys, time\n"
                    "for line in sys.stdin:\n"
                    "    if line.startswith('go '):\n"
                    "        print('info depth 1 nodes 10 nps 20', flush=True)\n"
                    + ("        print('bestmove e2e4', flush=True)\n" if shutdown else "")
                    + "        time.sleep(60)\n"
                )
                engine.chmod(0o755)
                with self.assertRaises(subprocess.TimeoutExpired):
                    bench_once(engine, ("startpos", "startpos"), 1, 16, 1, 1.0, True,
                               raw_output_path=raw)
                self.assertIn("info depth 1 nodes 10 nps 20", raw.read_text())
                if shutdown:
                    self.assertIn("bestmove e2e4", raw.read_text())

    def test_validates_requested_backend_acknowledgement(self):
        self.assertEqual(
            validate_option_transcript(
                "info string NNUE backend set to aarch64-simd256\n",
                [("NNUEBackend", "aarch64-simd256")],
            ),
            {"NNUEBackend": "aarch64-simd256"},
        )
        with self.assertRaisesRegex(RuntimeError, "did not acknowledge"):
            validate_option_transcript("readyok\n", [("NNUEBackend", "scalar")])
        with self.assertRaisesRegex(RuntimeError, "rejected benchmark option"):
            validate_option_transcript(
                "info string Unknown NNUE backend: missing\n",
                [("NNUEBackend", "missing")],
            )

    def test_parses_and_sends_labelled_uci_options(self):
        self.assertEqual(
            parse_option_arg("stockfish-net:NNUE=/tmp/example.nnue"),
            ("stockfish-net", "NNUE", "/tmp/example.nnue"),
        )
        commands = uci_input(
            "startpos",
            depth=7,
            hash_mb=64,
            threads=1,
            disable_book=True,
            options=[("NNUE", "/tmp/example.nnue")],
        )
        self.assertIn("setoption name NNUE value /tmp/example.nnue\n", commands)
        self.assertLess(
            commands.index("setoption name NNUE value /tmp/example.nnue"),
            commands.index("isready"),
        )

    @unittest.skipIf(
        sys.platform == "win32",
        "creates a POSIX shebang script; only runs in the Nix/Unix CI shell",
    )
    def test_waits_for_bestmove_before_sending_quit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = Path(temp_dir) / "async_engine"
            raw_output = Path(temp_dir) / "raw.log"
            engine.write_text(
                textwrap.dedent(
                    f"""\
                    #!{sys.executable}
                    import sys
                    import threading
                    import time

                    search_finished = threading.Event()

                    def search():
                        time.sleep(0.1)
                        print("info depth 7 score cp 12 nodes 1234 nps 12000", flush=True)
                        search_finished.set()
                        print("bestmove e2e4", flush=True)

                    for command in sys.stdin:
                        command = command.strip()
                        if command.startswith("go "):
                            threading.Thread(target=search, daemon=True).start()
                        elif command == "quit":
                            if not search_finished.is_set():
                                raise SystemExit(2)
                            break
                    """
                ),
                encoding="utf-8",
            )
            engine.chmod(0o755)

            sample = bench_once(
                engine,
                ("fake", "startpos"),
                depth=7,
                hash_mb=64,
                threads=1,
                timeout=5.0,
                disable_book=True,
                raw_output_path=raw_output,
            )
            raw_text = raw_output.read_text(encoding="utf-8")

        self.assertEqual(sample["reported_depth"], 7)
        self.assertEqual(sample["nodes"], 1234)
        self.assertEqual(sample["nps"], 12000)
        self.assertGreaterEqual(sample["wall_seconds"], 0.1)
        self.assertIn("bestmove e2e4", raw_text)


if __name__ == "__main__":
    unittest.main()
