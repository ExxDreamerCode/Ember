#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path

from compare_fixture_corpus import load_checks, run_check


def select_check(fixture_dir, case_id, depth):
    matches = [
        check
        for check in load_checks(fixture_dir)
        if check.case_id == case_id and check.depth == depth
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one {case_id!r} depth {depth} check, found {len(matches)}"
        )
    return matches[0]


def build_engine(cargo, target_dir):
    command = [cargo, "build", "--locked", "--release", "--bin", "ember"]
    env = None
    if target_dir:
        import os

        env = dict(os.environ)
        env["CARGO_TARGET_DIR"] = str(Path(target_dir).resolve())
    try:
        subprocess.run(command, check=True, env=env)
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"build skipped: {error}", file=sys.stderr)
        return None
    target = Path(target_dir) if target_dir else Path("target")
    return target / "release" / "ember"


def main():
    parser = argparse.ArgumentParser(
        description="Build the checked-out revision and classify one TSV fixture for git bisect."
    )
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--depth", required=True, type=int)
    parser.add_argument("--cargo", default="cargo")
    parser.add_argument("--target-dir")
    parser.add_argument("--hash-mb", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    check = select_check(args.fixtures, args.case_id, args.depth)
    binary = build_engine(args.cargo, args.target_dir)
    if binary is None:
        return 125
    result = run_check(binary, check, args.timeout, args.hash_mb)
    print(
        f"{check.case_id} depth={check.depth} bestmove={result['bestmove']} "
        f"passed={result['passed']} error={result['error']}",
        flush=True,
    )
    if result["error"] is not None:
        return 125
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
