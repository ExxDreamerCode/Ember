#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import time

from pyrrhic_tools import load_fork_tool

REPO = pathlib.Path(__file__).resolve().parent.parent
PGO_DATA = REPO / "pgo-data"
MERGED_PROFILE = PGO_DATA / "merged.profdata"
INSTRUMENTED_DIR = REPO / "target-pgo"
OPTIMIZED_DIR = REPO / "target-pgo-use"
DEFAULT_RUSTFLAGS = "-C target-cpu=x86-64-v3"


def binary_name() -> str:
    return "ember.exe" if os.name == "nt" else "ember"


def binary_path(target_dir: pathlib.Path, target: str | None) -> pathlib.Path:
    return target_dir / (target or "") / "release" / binary_name()


def find_llvm_profdata() -> pathlib.Path:
    override = os.environ.get("EMBER_LLVM_PROFDATA")
    if override:
        path = pathlib.Path(override)
        if path.is_file():
            return path
    rustup_home = pathlib.Path(os.environ.get("RUSTUP_HOME", pathlib.Path.home() / ".rustup"))
    toolchains = rustup_home / "toolchains"
    candidates = sorted(toolchains.glob("*/lib/rustlib/*/bin/llvm-profdata.exe"))
    candidates += sorted(toolchains.glob("*/lib/rustlib/*/bin/llvm-profdata"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    found = shutil.which("llvm-profdata")
    if found:
        return pathlib.Path(found)
    raise SystemExit(
        "llvm-profdata not found. Run: rustup component add llvm-tools "
        "(or set EMBER_LLVM_PROFDATA to the binary path)."
    )


def run_cargo(
    rustflags: str,
    target_dir: pathlib.Path,
    extra_args: list[str],
    target: str | None,
) -> None:
    env = dict(os.environ)
    env["RUSTFLAGS"] = rustflags
    command = ["cargo", "build", "--release", "--locked", f"--target-dir={target_dir.name}"]
    if target:
        command.append(f"--target={target}")
    command.extend(extra_args)
    started = time.monotonic()
    subprocess.run(command, cwd=REPO, env=env, check=True)
    print(f"cargo build finished in {time.monotonic() - started:.0f}s (target-dir={target_dir.name})")


def run_bench(binary: pathlib.Path, depths: list[int],
              profile_name: str | None = None) -> list[tuple[int, str, int]]:
    input_text = "".join(f"bench depth {d}\n" for d in depths) + "quit\n"
    env = dict(os.environ)
    if profile_name is not None:
        env["LLVM_PROFILE_FILE"] = str(PGO_DATA / f"{profile_name}-%p.profraw")
    proc = subprocess.run(
        [str(binary)],
        input=input_text,
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    results: list[tuple[int, str, int]] = []
    for line in (proc.stdout + proc.stderr).splitlines():
        if "bench total:" not in line:
            continue
        fields = [token.rstrip(",") for token in line.split()]
        depth = int(fields[fields.index("depth") + 1])
        nodes = int(fields[fields.index("nodes") + 1])
        signature = fields[fields.index("signature") + 1]
        results.append((depth, signature, nodes))
        print(f"  depth {depth}: nodes {nodes} signature {signature}")
    if len(results) != len(depths):
        raise SystemExit(f"bench produced {len(results)} totals, expected {len(depths)}")
    if [depth for depth, _, _ in results] != depths:
        raise SystemExit(f"bench depths {results} do not match requested {depths}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depths", type=int, nargs="+", default=[12, 14], help="profile workload bench depths")
    parser.add_argument(
        "--verify-depths", type=int, nargs="+", default=[8, 12, 14],
        help="bench depths used for the plain-vs-PGO signature comparison",
    )
    parser.add_argument("--rustflags", default=DEFAULT_RUSTFLAGS, help="base RUSTFLAGS, e.g. target-cpu")
    parser.add_argument("--target", default=None, help="optional cargo --target triple")
    parser.add_argument(
        "--cargo-args", default="",
        help='extra cargo build args as one quoted string, e.g. "--bin ember"',
    )
    parser.add_argument(
        "--syzygy-path", type=pathlib.Path,
        help="hash-pinned compact Syzygy tables for release PGO training",
    )
    args = parser.parse_args()
    cargo_args = shlex.split(args.cargo_args)
    if args.syzygy_path is not None:
        load_fork_tool("fetch_syzygy_ci.py").verify_dataset(args.syzygy_path)

    profdata_tool = find_llvm_profdata()
    PGO_DATA.mkdir(exist_ok=True)

    print("== step 1/5: instrumented build ==")
    run_cargo(f"{args.rustflags} -Cprofile-generate={PGO_DATA.as_posix()}", INSTRUMENTED_DIR, cargo_args, args.target)
    instrumented = binary_path(INSTRUMENTED_DIR, args.target)

    print("== step 2/5: profile workload ==")
    for stale in PGO_DATA.glob("*.profraw"):
        stale.unlink()
    run_bench(instrumented, args.depths, profile_name="bench")
    if args.syzygy_path is not None:
        training_log = PGO_DATA / "syzygy-training.log"
        training_env = dict(os.environ)
        training_env["LLVM_PROFILE_FILE"] = str(PGO_DATA / "syzygy-%p.profraw")
        with training_log.open("w", encoding="utf-8") as output:
            subprocess.run(
                [sys.executable, str(REPO / "tools" / "train_syzygy_pgo.py"),
                 str(instrumented), str(args.syzygy_path)],
                cwd=REPO, env=training_env, stdout=output,
                stderr=subprocess.STDOUT, check=True,
            )
        print(f"  trained compact Syzygy roots; transcript: {training_log}")
    raw_profiles = sorted(PGO_DATA.glob("*.profraw"))
    if not raw_profiles:
        raise SystemExit("no .profraw written; the process must exit normally (no kill/panic)")
    print(f"  collected {len(raw_profiles)} profile(s)")

    print("== step 3/5: merge profile ==")
    subprocess.run(
        [str(profdata_tool), "merge", "-o", str(MERGED_PROFILE)] + [str(p) for p in raw_profiles],
        check=True,
    )
    print(f"  {MERGED_PROFILE} ({MERGED_PROFILE.stat().st_size / 1e6:.1f} MB)")
    if args.syzygy_path is not None:
        covered = subprocess.run(
            [str(profdata_tool), "show", "--covered", str(MERGED_PROFILE)],
            capture_output=True, text=True, check=True,
        ).stdout
        (PGO_DATA / "covered-functions.txt").write_text(covered, encoding="utf-8")
        if "table_decoder11decode_pair" not in covered:
            raise SystemExit("PGO workload did not execute the Syzygy decoder")

    print("== step 4/5: PGO build ==")
    run_cargo(f"{args.rustflags} -Cprofile-use={MERGED_PROFILE.as_posix()}", OPTIMIZED_DIR, cargo_args, args.target)
    optimized = binary_path(OPTIMIZED_DIR, args.target)

    print("== step 5/5: verify identical behavior ==")
    plain = binary_path(REPO / "target", args.target)
    if not plain.is_file():
        raise SystemExit(f"plain release binary not found at {plain}; build it first for comparison")
    plain_results = run_bench(plain, args.verify_depths)
    pgo_results = run_bench(optimized, args.verify_depths)
    for (depth, plain_sig, plain_nodes), (_, pgo_sig, pgo_nodes) in zip(plain_results, pgo_results):
        if plain_sig != pgo_sig or plain_nodes != pgo_nodes:
            raise SystemExit(
                f"behavior mismatch at depth {depth}: plain {plain_sig}/{plain_nodes} "
                f"vs pgo {pgo_sig}/{pgo_nodes}"
            )
    print("PGO binary matches the plain binary on every verified bench signature.")


if __name__ == "__main__":
    main()
