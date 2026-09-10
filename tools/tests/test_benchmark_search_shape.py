import argparse
import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import unittest


MODULE_PATH = pathlib.Path(__file__).parents[1] / "benchmark_search_shape.py"
SPEC = importlib.util.spec_from_file_location("benchmark_search_shape", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SearchShapeArtifactTests(unittest.TestCase):
    def test_safe_component_is_portable_and_nonempty(self):
        self.assertEqual(MODULE.safe_component("candidate / startpos"), "candidate-startpos")
        self.assertEqual(MODULE.safe_component("***"), "sample")

    def test_sha256_file_records_the_exact_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "engine"
            path.write_bytes(b"ember")
            self.assertEqual(
                MODULE.sha256_file(path),
                "7cadc15d609c4ae9b4be6265b8e1cace16e6fa78a81ab0c7db82e687a7c867a5",
            )

    def test_uci_options_cannot_override_core_resource_options(self):
        options = [MODULE.parse_option_arg("NNUEBackend=x86-v3")]

        self.assertEqual(
            MODULE.uci_option_commands(64, 4, True, options),
            [
                "setoption name Hash value 64",
                "setoption name Threads value 4",
                "setoption name Book value",
                "setoption name NNUEBackend value x86-v3",
            ],
        )
        for option in ("Hash=1", "threads=2", "BOOK=/tmp/book.bin"):
            with self.subTest(option=option), self.assertRaisesRegex(
                argparse.ArgumentTypeError,
                "dedicated benchmark flag",
            ):
                MODULE.parse_option_arg(option)

    def make_launchable(self, script: pathlib.Path) -> pathlib.Path:
        if os.name != "nt":
            script.chmod(0o755)
            return script
        wrapper = script.with_suffix(".bat")
        wrapper.write_text(
            "@echo off\r\n"
            f'"{sys.executable}" "%~dp0{script.name}" %*\r\n',
            encoding="ascii",
        )
        return wrapper

    def fake_engine(self, directory, setup_lines):
        script = pathlib.Path(directory) / "fake-engine"
        script.write_text(
            """#!/usr/bin/env python3
import sys

setup_lines = {setup_lines}
for command in sys.stdin:
    command = command.strip()
    if command == "uci":
        print("uciok", flush=True)
    elif command == "isready":
        for line in setup_lines:
            print(line, flush=True)
        print("readyok", flush=True)
    elif command.startswith("go"):
        print("info depth 4 nodes 10 nps 1000 time 10", flush=True)
        print("bestmove e2e4", flush=True)
    elif command == "quit":
        break
""".format(setup_lines=json.dumps(setup_lines)),
            encoding="utf-8",
        )
        return self.make_launchable(script)

    def test_backend_must_be_acknowledged_and_raw_log_keeps_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = self.fake_engine(
                directory,
                ["info string NNUE backend set to x86-v3"],
            )
            raw = pathlib.Path(directory) / "raw.log"
            row = MODULE.run_one(
                engine,
                ("start", "startpos"),
                "go depth 4",
                64,
                1,
                True,
                5.0,
                [("NNUEBackend", "x86-v3")],
                raw,
            )

            self.assertEqual(row["effective_options"]["NNUEBackend"], "x86-v3")
            self.assertIn("NNUE backend set to x86-v3", raw.read_text())

    def test_rejected_or_unacknowledged_backend_aborts_sample(self):
        cases = (
            ["info string Unknown NNUE backend: missing"],
            ["info string NNUE backend x86-v3 is not available on this CPU"],
            [],
        )
        for setup_lines in cases:
            with self.subTest(lines=setup_lines), tempfile.TemporaryDirectory() as directory:
                engine = self.fake_engine(directory, setup_lines)
                with self.assertRaisesRegex(
                    RuntimeError,
                    "rejected benchmark option|did not acknowledge",
                ):
                    MODULE.run_one(
                        engine,
                        ("start", "startpos"),
                        "go depth 4",
                        64,
                        1,
                        True,
                        5.0,
                        [("NNUEBackend", "x86-v3")],
                    )


if __name__ == "__main__":
    unittest.main()
