#!/usr/bin/env python3
"""Exercise malformed and early-termination UCI exchanges against Ember."""

from __future__ import annotations

import argparse
import queue
import subprocess
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import TextIO


CRASH_MARKERS = (
    "panicked at",
    "overflowed its stack",
    "fatal runtime error",
    "stack backtrace:",
)


class UciOutputReader(threading.Thread):
    """Drain output even after logging fails, then surface the failure to the caller."""

    def __init__(self, stdout: TextIO, lines: queue.Queue[str], transcript: TextIO | None):
        super().__init__(daemon=True)
        self.stdout = stdout
        self.lines = lines
        self.transcript = transcript
        self.error: Exception | None = None

    def run(self) -> None:
        try:
            for line in self.stdout:
                self.lines.put(line)
                if self.transcript is not None and self.error is None:
                    try:
                        self.transcript.write(line)
                        self.transcript.flush()
                    except Exception as error:
                        # Keep reading so the engine can finish and be reaped.
                        self.error = error
        except Exception as error:
            if self.error is None:
                self.error = error

    def check_capture(self) -> None:
        """Call after joining and closing the process streams, before accepting output."""
        if self.error is not None:
            raise RuntimeError("UCI output capture failed") from self.error
        if self.is_alive():
            raise RuntimeError("UCI output capture did not finish")


def start_process(
    command: list[str],
    transcript: TextIO | None = None,
) -> tuple[subprocess.Popen[str], queue.Queue[str], UciOutputReader]:
    if not command:
        raise ValueError("no Ember command was provided")

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None

    lines: queue.Queue[str] = queue.Queue()

    reader = UciOutputReader(process.stdout, lines, transcript)
    reader.start()
    return process, lines, reader


def drain_lines(lines: queue.Queue[str]) -> str:
    captured: list[str] = []
    while True:
        try:
            captured.append(lines.get_nowait())
        except queue.Empty:
            break
    return "".join(captured)


def assert_no_crash_output(output: str) -> None:
    lowered = output.lower()
    for marker in CRASH_MARKERS:
        if marker in lowered:
            raise RuntimeError(f"UCI stress output contains crash marker {marker!r}:\n{output}")


def wait_for_exit(
    process: subprocess.Popen[str],
    lines: queue.Queue[str],
    reader: UciOutputReader,
    timeout: float,
) -> str:
    deadline = time.monotonic() + timeout
    captured: list[str] = []
    while time.monotonic() < deadline:
        captured.append(drain_lines(lines))
        if process.poll() is not None:
            break
        time.sleep(0.02)
    else:
        process.kill()
        process.wait()
        reader.join(timeout=1.0)
        captured.append(drain_lines(lines))
        if process.stdout is not None:
            process.stdout.close()
        raise TimeoutError(f"UCI stress process did not exit:\n{''.join(captured)}")

    reader.join(timeout=1.0)
    captured.append(drain_lines(lines))
    if process.stdout is not None:
        process.stdout.close()
    reader.check_capture()
    output = "".join(captured)
    assert_no_crash_output(output)
    if process.returncode != 0:
        raise RuntimeError(f"UCI stress command exited with {process.returncode}:\n{output}")
    return output


def wait_for_line(
    process: subprocess.Popen[str],
    lines: queue.Queue[str],
    captured: list[str],
    predicate,
    timeout: float,
) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            line = lines.get(timeout=min(0.25, max(0.0, deadline - time.monotonic())))
        except queue.Empty:
            if process.poll() is not None:
                break
            continue
        captured.append(line)
        if predicate(line):
            return line
    raise TimeoutError(f"UCI stress did not observe the expected line:\n{''.join(captured)}")


def run_malformed_input_recovery(command: list[str], timeout: float, transcript: TextIO | None = None) -> str:
    process, lines, reader = start_process(command, transcript)
    assert process.stdin is not None
    captured: list[str] = []

    try:
        process.stdin.write(
            "\n".join(
                [
                    "uci",
                    "setoption name Hash value 16",
                    "setoption name Threads value 1",
                    "setoption name Book value",
                    "isready",
                    "",
                ]
            )
        )
        process.stdin.flush()
        wait_for_line(
            process,
            lines,
            captured,
            lambda line: line.startswith("readyok"),
            timeout,
        )

        process.stdin.write(
            "\n".join(
                [
                    "position startpos moves 0000 g1f3",
                    "position startpos moves zzzz",
                    "position startpos moves e2e4x",
                    "position fen 8/8/8/8/8/8/8/8 w - - 0 1",
                    "position startpos",
                    "go movetime nope depth 1",
                    "",
                ]
            )
        )
        process.stdin.flush()
        wait_for_line(
            process,
            lines,
            captured,
            lambda line: line.startswith("bestmove ") and line.rstrip() != "bestmove 0000",
            timeout,
        )

        process.stdin.write("quit\n")
        process.stdin.flush()
    except Exception:
        process.kill()
        process.wait()
        reader.join(timeout=1.0)
        if process.stdout is not None:
            process.stdout.close()
        raise
    finally:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()

    captured.append(wait_for_exit(process, lines, reader, timeout))
    output = "".join(captured)
    assert_no_crash_output(output)
    return output


def run_batch(command: list[str], commands: list[str], timeout: float, transcript: TextIO | None = None) -> str:
    process, lines, reader = start_process(command, transcript)
    assert process.stdin is not None
    process.stdin.write("\n".join(commands))
    process.stdin.write("\n")
    process.stdin.flush()
    process.stdin.close()
    return wait_for_exit(process, lines, reader, timeout)


def run_queued_quit_during_search(command: list[str], timeout: float, transcript: TextIO | None = None) -> str:
    return run_batch(
        command,
        [
            "uci",
            "setoption name Hash value 16",
            "setoption name Threads value 2",
            "setoption name Book value",
            "isready",
            "position startpos",
            "go depth 16",
            "quit",
        ],
        timeout,
        transcript,
    )


def run_eof_during_search(command: list[str], timeout: float, transcript: TextIO | None = None) -> str:
    return run_batch(
        command,
        [
            "uci",
            "setoption name Hash value 16",
            "setoption name Threads value 2",
            "setoption name Book value",
            "isready",
            "position startpos",
            "go depth 16",
        ],
        timeout,
        transcript,
    )


def run_stress(command: list[str], timeout: float, out_dir: Path | None = None,
               log_prefix: str = "") -> dict[str, str]:
    outputs = {}
    for name, run in [("malformed-input", run_malformed_input_recovery),
                      ("queued-quit", run_queued_quit_during_search),
                      ("search-eof", run_eof_during_search)]:
        # Stream before validating or returning: later failures must not discard
        # earlier cases, and the failing process's output must survive too.
        with ((out_dir / f"{log_prefix}{name}.log").open("x", encoding="utf-8")
              if out_dir is not None else nullcontext()) as transcript:
            outputs[name] = run(command, timeout, transcript)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="optional directory for per-scenario UCI logs",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="executable command, optionally preceded by an emulator",
    )
    args = parser.parse_args()

    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = run_stress(command, args.timeout, args.out_dir, "uci-stress-")
    for name, output in outputs.items():
        if args.out_dir is None:
            print(f"{name}: ok")
        else:
            log_path = args.out_dir / f"uci-stress-{name}.log"
            print(f"{name}: ok ({log_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
