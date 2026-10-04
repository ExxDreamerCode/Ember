"""Locate test tools in the Pyrrhic source used by this Ember checkout."""

import importlib.util
import json
import os
import subprocess
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def pyrrhic_root():
    manifest = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    dependency = manifest["dependencies"]["pyrrhic-rs"]
    revision = dependency["rev"]
    expected_source = f'git+{dependency["git"]}?rev={revision}#{revision}'
    lock = tomllib.loads((ROOT / "Cargo.lock").read_text(encoding="utf-8"))
    locked = [
        package for package in lock["package"]
        if package["name"] == "pyrrhic-rs"
    ]
    if len(locked) != 1 or locked[0].get("source") != expected_source:
        raise ValueError("Cargo.lock does not pin the Pyrrhic fork revision")

    env = dict(os.environ)
    env["CARGO_NET_GIT_FETCH_WITH_CLI"] = "true"
    try:
        result = subprocess.run(
            ["cargo", "metadata", "--locked", "--format-version", "1"],
            cwd=ROOT, env=env, capture_output=True, text=True,
            timeout=180, check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ValueError("could not locate the pinned Pyrrhic checkout") from exc
    packages = [
        package for package in json.loads(result.stdout)["packages"]
        if package["name"] == "pyrrhic-rs"
        and package.get("source") == expected_source
    ]
    if len(packages) != 1:
        raise ValueError("Cargo metadata does not contain the pinned Pyrrhic fork")
    return Path(packages[0]["manifest_path"]).parent.resolve(strict=True)


def load_fork_tool(name):
    path = pyrrhic_root() / "tools" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
