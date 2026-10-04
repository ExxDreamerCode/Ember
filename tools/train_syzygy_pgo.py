"""Collect release PGO counts from compact, real Syzygy root probes."""

import argparse
from pathlib import Path
import queue
import subprocess
import threading
import time

POSITIONS = (
    "7k/8/8/8/8/8/8/1Q2K3 w - - 0 1",
    "6k1/8/8/3P4/4K3/8/8/8 w - - 0 1",
    "5k2/R7/8/8/5K2/p7/8/8 w - - 0 62",
    "6rk/8/8/8/8/8/8/KNN5 w - - 0 1",
)

parser = argparse.ArgumentParser()
parser.add_argument("binary")
parser.add_argument("tables")
parser.add_argument("--repeats", type=int, default=250)
parser.add_argument("--qemu")
parser.add_argument("--cpu")
args = parser.parse_args()
if bool(args.qemu) != bool(args.cpu):
    parser.error("--qemu and --cpu must be supplied together")
if args.repeats < 1:
    parser.error("--repeats must be positive")
tables = Path(args.tables)
for material in ("KQvK", "KPvK", "KRvK", "KRvKP", "KNNvKR"):
    for extension in ("rtbw", "rtbz"):
        file = tables / f"{material}.{extension}"
        if not file.is_file():
            parser.error(f"missing training table {file}")
command = ([args.qemu, "-cpu", args.cpu] if args.qemu else []) + [args.binary]

process = subprocess.Popen(
    command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT, text=True, bufsize=1,
)
lines = queue.Queue()
reader_error = []

def read_output():
    try:
        for line in process.stdout:
            lines.put(line.rstrip("\n"))
    except Exception as exc:
        reader_error.append(exc)
    finally:
        lines.put(None)

reader = threading.Thread(target=read_output)
reader.start()
capture = []

def send(command):
    process.stdin.write(command + "\n")
    process.stdin.flush()

def until(prefix):
    deadline = time.monotonic() + 30
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"engine did not produce {prefix}")
        try:
            line = lines.get(timeout=remaining)
        except queue.Empty as error:
            raise TimeoutError(f"engine did not produce {prefix}") from error
        if line is None:
            raise RuntimeError("engine exited before " + prefix)
        capture.append(line)
        if line.startswith(prefix):
            return line

try:
    send("uci")
    until("uciok")
    if "option name OwnBook type check default false" not in capture:
        raise RuntimeError("PGO training requires OwnBook disabled by default")
    send("setoption name SyzygyPath value " + args.tables)
    send("isready")
    until("readyok")
    if f"info string Syzygy tables loaded: {args.tables}" not in capture:
        raise RuntimeError("tablebase loading was not acknowledged")
    for _ in range(args.repeats):
        for fen in POSITIONS:
            start = len(capture)
            send("position fen " + fen)
            send("go depth 1")
            move = until("bestmove ")
            if move == "bestmove 0000":
                raise RuntimeError("unexpected null move")
            if not any("nodes 0" in line and line.startswith("info depth 1 ") for line in capture[start:]):
                raise RuntimeError("search did not use the tablebase: " + repr(capture[start:]))
    send("quit")
    process.stdin.close()
    if process.wait(timeout=30) != 0:
        raise RuntimeError("engine exited unsuccessfully")
    reader.join(timeout=5)
    if reader.is_alive() or reader_error:
        raise RuntimeError("output reader failed")
    print(f"trained {args.repeats * len(POSITIONS)} searches")
    print("\n".join(capture))
finally:
    if process.poll() is None:
        process.kill()
        process.wait()
    reader.join(timeout=5)
