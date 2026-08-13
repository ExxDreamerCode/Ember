#!/usr/bin/env python3
import argparse
import json
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_TUNE_TOML = HERE / "tune.toml"
DEFAULT_BEST_JSON = HERE / "best.json"


def main():
    parser = argparse.ArgumentParser(description="Show tuned values for manual apply")
    parser.add_argument("--config", default=str(DEFAULT_TUNE_TOML))
    parser.add_argument("--best", default=str(DEFAULT_BEST_JSON))
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    with open(args.best, "r", encoding="utf-8") as f:
        best = json.load(f)

    changes = []
    for spec in cfg["params"]:
        value = best.get("values", {}).get(spec["name"], spec["base"])
        if value != spec["base"]:
            changes.append((spec["name"], spec["base"], value))

    if not changes:
        print("No differences from the compile-time defaults.")
        return

    print("Tuned values to apply:")
    for name, base, value in changes:
        print(f"  {name}: {base} -> {value}")


if __name__ == "__main__":
    main()