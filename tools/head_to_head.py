#!/usr/bin/env python3
import argparse
import copy
import concurrent.futures
import contextlib
import csv
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from sprt import pentanomial_counts, pentanomial_sprt

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib


RESULT_SCORE = {
    "1-0": (1.0, 0.0),
    "0-1": (0.0, 1.0),
    "1/2-1/2": (0.5, 0.5),
}


def now_utc():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def exe_suffix():
    return ".exe" if os.name == "nt" else ""


def platform_binary(name):
    p = Path(name)
    if p.suffix.lower() == ".exe":
        p = p.with_suffix("")
    return p.with_name(p.name + exe_suffix())


def read_toml(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def write_text_atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_json(path, data):
    write_text_atomic(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def shell_quote(s):
    if re.match(r"^[A-Za-z0-9_./:=+@,%^-]+$", s):
        return s
    return "'" + s.replace("'", "'\"'\"'") + "'"


def run_cmd(args, log_path=None, check=True, env=None, cwd=None):
    start = time.time()
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("$ " + " ".join(shell_quote(a) for a in args) + "\n")
            f.write(f"[started={now_utc()}]\n")
            f.flush()
            os.fsync(f.fileno())
    proc = subprocess.run(
        args,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = time.time() - start
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(proc.stdout)
            if proc.stdout and not proc.stdout.endswith("\n"):
                f.write("\n")
            f.write(f"[exit={proc.returncode} elapsed={elapsed:.3f}s]\n\n")
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(args)}")
    return proc


def command_executable(command):
    if isinstance(command, str):
        path = Path(command).expanduser()
        if path.is_file():
            return command
        parts = shlex.split(command)
        if parts:
            return parts[0]
        raise ValueError("command must not be empty")
    if command:
        return str(command[0])
    raise ValueError("command must not be empty")


def command_label(command):
    if isinstance(command, str):
        return command
    if command:
        return " ".join(str(argument) for argument in command)
    raise ValueError("command must not be empty")


def resolve_path(value, base=None):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(base or Path.cwd()) / path
    return path.resolve()


def engine_directory(engine):
    return resolve_path(engine.get("dir", "."))


def resolve_executable(command, cwd=None):
    executable = command_executable(command)
    candidate = Path(executable).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        local = resolve_path(candidate, cwd)
        if local.is_file():
            resolved = local
        else:
            found = shutil.which(executable)
            if found is None:
                raise RuntimeError(f"executable is unavailable: {executable}")
            resolved = Path(found).resolve()
    if not resolved.is_file():
        raise RuntimeError(f"executable is unavailable: {resolved}")
    if os.name != "nt" and not os.access(resolved, os.X_OK):
        raise RuntimeError(f"configured executable is not executable: {resolved}")
    return resolved


def replace_command_executable(command, executable):
    executable = str(executable)
    if isinstance(command, str):
        original = Path(command).expanduser()
        if original.is_file() or len(shlex.split(command)) == 1:
            return executable
        parts = shlex.split(command)
        return [executable, *parts[1:]]
    return [executable, *[str(argument) for argument in command[1:]]]


def directory_identity(path):
    resolved = resolve_path(path)
    if not resolved.is_dir():
        raise RuntimeError(f"configured input directory is missing: {resolved}")
    entries = []
    for item in sorted(resolved.rglob("*")):
        if item.is_symlink():
            raise RuntimeError(
                f"configured input directory contains a symlink: {item}"
            )
        if not item.is_file():
            continue
        stat = item.stat()
        entries.append(
            {
                "path": item.relative_to(resolved).as_posix(),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return {
        "path": str(resolved),
        "entries": entries,
        "fingerprint": stable_hash(entries),
    }


def validate_directory_identity(expected):
    if directory_identity(expected["path"]) != expected:
        raise RuntimeError(
            f"configured input directory changed during run: {expected['path']}"
        )


@contextlib.contextmanager
def exclusive_run_lock(rd, operation):
    rd = Path(rd)
    rd.mkdir(parents=True, exist_ok=True)
    lock_dir = rd / ".run.lock"
    owner = {
        "hostname": socket.gethostname(),
        "operation": operation,
        "pid": os.getpid(),
        "started_at": now_utc(),
        "token": uuid.uuid4().hex,
    }
    try:
        lock_dir.mkdir()
    except FileExistsError as error:
        owner_path = lock_dir / "owner.json"
        try:
            current = owner_path.read_text(encoding="utf-8").strip()
        except OSError:
            current = "owner metadata is unavailable"
        raise RuntimeError(f"run is locked by {current}") from error
    try:
        write_json(lock_dir / "owner.json", owner)
        yield
    finally:
        owner_path = lock_dir / "owner.json"
        try:
            current = json.loads(owner_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = None
        if current == owner:
            owner_path.unlink()
            lock_dir.rmdir()


def load_config(path):
    cfg = read_toml(path)
    if "engine_a" not in cfg or "engine_b" not in cfg:
        raise ValueError("config must contain [engine_a] and [engine_b]")
    return cfg


def make_run_id(cfg):
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    name = safe_name(cfg["run"].get("name", "head-to-head"))
    return f"{stamp}-{name}-{stable_hash(cfg)[:8]}"


def run_dir_for(cfg, run_id):
    return Path(cfg["run"].get("results_dir", "results/head-to-head")) / run_id


def git_output(repo, *args):
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"git {' '.join(args)} failed")
    return proc.stdout.strip()


def file_identity(path, base=None):
    resolved = resolve_path(path, base)
    if not resolved.is_file():
        raise RuntimeError(f"configured input file is missing: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _configured_path(value):
    if value is None:
        return None
    value = str(value).strip()
    if not value or (value.startswith("<") and value.endswith(">")):
        return None
    return value


def configured_input_identities(cfg):
    inputs = {}
    for engine_id in ("engine_a", "engine_b"):
        engine = cfg[engine_id]
        for name, value in engine.get("options", {}).items():
            configured = _configured_path(value)
            if configured is None:
                continue
            label = f"{engine_id}.option.{name}"
            if name.casefold() in {"nnue", "book"}:
                inputs[label] = file_identity(
                    configured,
                    engine_directory(engine),
                )
            elif name.casefold() == "syzygypath":
                inputs[label] = directory_identity(
                    resolve_path(configured, engine_directory(engine))
                )

    run_cfg = cfg["run"]
    source = run_cfg.get("opening_source", "file")
    key = "polyglot_book" if source == "polyglot" else "opening_file"
    configured = _configured_path(run_cfg.get(key))
    if configured is not None:
        inputs[f"run.{key}"] = file_identity(configured)
    return inputs


def resolved_revision_identities(cfg):
    repo = Path(cfg.get("build", {}).get("repo", ".")).resolve()
    revisions = {}
    for engine_id in ("engine_a", "engine_b"):
        revision = cfg[engine_id].get("revision")
        if revision is not None:
            revisions[engine_id] = {
                "configured": revision,
                "resolved": git_output(repo, "rev-parse", f"{revision}^{{commit}}"),
            }
    return revisions


def validate_run_configuration(cfg, explicit_workers=None, explicit_max_pairs=None):
    run_cfg = cfg.get("run")
    if not isinstance(run_cfg, dict):
        raise ValueError("config must contain [run]")
    names = []
    for engine_id in ("engine_a", "engine_b"):
        engine = cfg.get(engine_id)
        if not isinstance(engine, dict):
            raise ValueError(f"config must contain [{engine_id}]")
        name = str(engine.get("name", "")).strip()
        if not name:
            raise ValueError(f"{engine_id}.name must not be empty")
        names.append(name)
        if "cmd" not in engine and "revision" not in engine:
            raise ValueError(f"{engine_id} must configure cmd or revision")
        directory = engine_directory(engine)
        if not directory.is_dir():
            raise ValueError(f"{engine_id}.dir is not a directory: {directory}")
        engine_thread_count(engine)
    if names[0] == names[1]:
        raise ValueError("head-to-head engine names must be distinct")

    detect_workers(cfg, explicit_workers)
    max_pairs = int(
        explicit_max_pairs
        if explicit_max_pairs is not None
        else run_cfg.get("max_pairs", 200)
    )
    if max_pairs < 1:
        raise ValueError("max_pairs must be positive")
    worker_saturating_batch_pairs(run_cfg, 1, max_pairs)
    engine_limit_args(run_cfg)
    alternative = run_cfg.get("alternative", "greater")
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError(f"unknown alternative: {alternative}")

    source = run_cfg.get("opening_source", "file")
    if source not in {"file", "polyglot"}:
        raise ValueError(f"unknown opening_source: {source}")
    key = "polyglot_book" if source == "polyglot" else "opening_file"
    if _configured_path(run_cfg.get(key)) is None:
        raise ValueError(f"run.{key} must name an input file")


def validate_external_commands(cfg, require_engines=False):
    resolve_executable(cfg["run"].get("cutechess_cmd", "cutechess-cli"))
    has_build_command = bool(cfg.get("build", {}).get("command"))
    for engine_id in ("engine_a", "engine_b"):
        engine = cfg[engine_id]
        if "revision" in engine:
            continue
        if not require_engines and has_build_command:
            continue
        resolve_executable(engine["cmd"], engine_directory(engine))


def run_request_payload(
    config_path,
    cfg,
    explicit_workers=None,
    explicit_max_pairs=None,
    explicit_alpha=None,
):
    validate_run_configuration(cfg, explicit_workers, explicit_max_pairs)
    return {
        "schema_version": 1,
        "config": cfg,
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": sha256_file(config_path),
        "resolved_revisions": resolved_revision_identities(cfg),
        "configured_inputs": configured_input_identities(cfg),
        "cli_overrides": {
            "workers": None if explicit_workers is None else int(explicit_workers),
            "max_pairs": (
                None if explicit_max_pairs is None else int(explicit_max_pairs)
            ),
            "alpha": None if explicit_alpha is None else float(explicit_alpha),
        },
    }


def ensure_run_request(
    config_path,
    cfg,
    run_id,
    explicit_workers=None,
    explicit_max_pairs=None,
    explicit_alpha=None,
):
    rd = run_dir_for(cfg, run_id)
    request_path = rd / "run-request.json"
    payload = run_request_payload(
        config_path,
        cfg,
        explicit_workers,
        explicit_max_pairs,
        explicit_alpha,
    )
    request = {
        "fingerprint": stable_hash(payload),
        "payload": payload,
    }
    if request_path.exists():
        existing = json.loads(request_path.read_text(encoding="utf-8"))
        if existing != request:
            raise RuntimeError(
                f"run {run_id} has different immutable inputs; use a new run ID"
            )
        return request

    evidence_paths = [
        rd / "games",
        rd / "openings",
        rd / "metadata.json",
        rd / "state.json",
        rd / "run-manifest.json",
    ]
    if any(path.exists() for path in evidence_paths):
        raise RuntimeError(
            f"run {run_id} contains legacy evidence without an immutable request; "
            "use a new run ID"
        )
    write_json(request_path, request)
    return request


def _copy_content_addressed(source, destination):
    source = Path(source)
    destination = Path(destination)
    if destination.exists():
        if sha256_file(destination) != sha256_file(source):
            raise RuntimeError(f"staged input hash mismatch: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    try:
        shutil.copy2(source, temporary)
        if sha256_file(temporary) != sha256_file(source):
            raise RuntimeError(f"failed to stage {source} without modification")
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def stage_run_inputs(cfg, rd):
    effective = copy.deepcopy(cfg)
    staged = {}
    bound_directories = {}

    def stage(label, value, base=None):
        identity = file_identity(value, base)
        source = Path(identity["path"])
        suffix = "".join(source.suffixes)
        destination = (rd / "inputs" / f"{identity['sha256']}-{safe_name(label)}{suffix}").resolve()
        _copy_content_addressed(source, destination)
        staged[label] = {
            **identity,
            "staged_path": str(destination),
        }
        return str(destination)

    for engine_id in ("engine_a", "engine_b"):
        engine = effective[engine_id]
        directory = engine_directory(engine)
        engine["dir"] = str(directory)
        options = engine.get("options", {})
        for name, value in list(options.items()):
            configured = _configured_path(value)
            if configured is None:
                continue
            label = f"{engine_id}-{name}"
            if name.casefold() in {"nnue", "book"}:
                options[name] = stage(label, configured, directory)
            elif name.casefold() == "syzygypath":
                identity = directory_identity(resolve_path(configured, directory))
                bound_directories[label] = identity
                options[name] = identity["path"]

    run_cfg = effective["run"]
    source = run_cfg.get("opening_source", "file")
    key = "polyglot_book" if source == "polyglot" else "opening_file"
    configured = _configured_path(run_cfg.get(key))
    if configured is not None:
        run_cfg[key] = stage(f"run-{key}", configured)
    return effective, staged, bound_directories


def executable_identity(command, cwd=None):
    return file_identity(resolve_executable(command, cwd))


def prepare_run_manifest(
    config_path,
    cfg,
    run_id,
    explicit_workers=None,
    explicit_max_pairs=None,
    explicit_alpha=None,
):
    rd = run_dir_for(cfg, run_id)
    validate_run_configuration(cfg, explicit_workers, explicit_max_pairs)
    preflight = copy.deepcopy(cfg)
    materialize_revision_commands(preflight, rd)
    validate_external_commands(preflight, require_engines=True)
    request = ensure_run_request(
        config_path,
        cfg,
        run_id,
        explicit_workers,
        explicit_max_pairs,
        explicit_alpha,
    )
    effective = copy.deepcopy(cfg)
    materialize_revision_commands(effective, rd)
    if explicit_alpha is not None:
        effective["run"]["alpha"] = float(explicit_alpha)
    workers, cores, worker_source = detect_workers(effective, explicit_workers)
    max_pairs = int(
        explicit_max_pairs
        if explicit_max_pairs is not None
        else effective["run"].get("max_pairs", 200)
    )
    effective, staged_inputs, bound_directories = stage_run_inputs(effective, rd)

    binaries = {}
    for engine_id in ("engine_a", "engine_b"):
        engine = effective[engine_id]
        command = engine["cmd"]
        if (
            isinstance(command, str)
            and not Path(command).expanduser().is_file()
            and len(shlex.split(command)) != 1
        ):
            raise RuntimeError(
                f"{engine_id}.cmd must contain only the executable; "
                "put arguments in the engine args list"
            )
        binary = executable_identity(engine["cmd"], engine["dir"])
        binaries[engine_id] = binary
        engine["cmd"] = binary["path"]
    cutechess = effective["run"].get("cutechess_cmd", "cutechess-cli")
    tool = executable_identity(cutechess)
    effective["run"]["cutechess_cmd"] = replace_command_executable(
        cutechess,
        tool["path"],
    )
    identity = {
        "request_fingerprint": request["fingerprint"],
        "effective_config": effective,
        "workers": workers,
        "worker_source": worker_source,
        "cpu_count": cores,
        "max_pairs": max_pairs,
        "binaries": binaries,
        "staged_inputs": staged_inputs,
        "bound_directories": bound_directories,
        "cutechess": tool,
        "head_to_head_sha256": sha256_file(__file__),
    }
    manifest = {
        "schema_version": 1,
        "fingerprint": stable_hash(identity),
        "identity": identity,
    }
    manifest_path = rd / "run-manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError(
                f"run {run_id} no longer matches its immutable manifest; "
                "use a new run ID"
            )
    else:
        write_json(manifest_path, manifest)
    validate_run_manifest(manifest, request)
    return manifest


def validate_run_manifest(manifest, request=None, allow_tool_change=False):
    if manifest.get("schema_version") != 1:
        raise RuntimeError("unsupported head-to-head manifest schema")
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise RuntimeError("head-to-head manifest has no identity")
    if manifest.get("fingerprint") != stable_hash(identity):
        raise RuntimeError("head-to-head manifest fingerprint is invalid")
    required = {
        "binaries",
        "cutechess",
        "effective_config",
        "head_to_head_sha256",
        "request_fingerprint",
        "staged_inputs",
    }
    missing = sorted(required - identity.keys())
    if missing:
        raise RuntimeError(
            f"head-to-head manifest identity is incomplete: {', '.join(missing)}"
        )
    if request is not None:
        if request.get("schema_version") is not None:
            raise RuntimeError("head-to-head request has an invalid envelope")
        payload = request.get("payload")
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise RuntimeError("unsupported head-to-head request schema")
        if request.get("fingerprint") != stable_hash(payload):
            raise RuntimeError("head-to-head request fingerprint is invalid")
        if identity.get("request_fingerprint") != request["fingerprint"]:
            raise RuntimeError("manifest does not match its run request")
    for label, item in identity["binaries"].items():
        actual = file_identity(item["path"])
        if actual != item:
            raise RuntimeError(f"engine binary changed during run: {label}")
        configured = identity["effective_config"][label]["cmd"]
        if configured != item["path"]:
            raise RuntimeError(f"effective engine command changed: {label}")
    for label, item in identity["staged_inputs"].items():
        actual = file_identity(item["staged_path"])
        if actual["bytes"] != item["bytes"] or actual["sha256"] != item["sha256"]:
            raise RuntimeError(f"staged input changed during run: {label}")
    for item in identity.get("bound_directories", {}).values():
        validate_directory_identity(item)
    actual_tool = file_identity(identity["cutechess"]["path"])
    if actual_tool != identity["cutechess"]:
        raise RuntimeError("Cute Chess executable changed during run")
    effective_tool = identity["effective_config"]["run"]["cutechess_cmd"]
    if command_executable(effective_tool) != identity["cutechess"]["path"]:
        raise RuntimeError("effective Cute Chess command changed")
    if not allow_tool_change:
        current_tool = sha256_file(__file__)
        if identity.get("head_to_head_sha256") != current_tool:
            raise RuntimeError(
                "head-to-head runner changed; pass the explicit historical-analysis "
                "override to regenerate derived reports"
            )


def load_run_manifest(
    config_path,
    cfg,
    run_id,
    allow_tool_change=False,
):
    rd = run_dir_for(cfg, run_id)
    request_path = rd / "run-request.json"
    manifest_path = rd / "run-manifest.json"
    if not request_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(
            f"run {run_id} has no immutable manifest; use a new run ID"
        )
    request = json.loads(request_path.read_text(encoding="utf-8"))
    payload = request.get("payload")
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RuntimeError("unsupported head-to-head request schema")
    overrides = payload["cli_overrides"]
    ensure_run_request(
        config_path,
        cfg,
        run_id,
        overrides["workers"],
        overrides["max_pairs"],
        overrides["alpha"],
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_run_manifest(
        manifest,
        request,
        allow_tool_change=allow_tool_change,
    )
    return manifest


def materialize_revision_commands(cfg, rd):
    for engine_id in ["engine_a", "engine_b"]:
        if "revision" in cfg[engine_id]:
            cfg[engine_id]["cmd"] = str(
                (rd / "builds" / engine_id / "bin" / platform_binary("ember")).resolve()
            )


def build_revision(cfg, rd, engine_id):
    build_cfg = cfg.get("build", {})
    engine = cfg[engine_id]
    repo = Path(build_cfg.get("repo", ".")).resolve()
    revision = git_output(repo, "rev-parse", f"{engine['revision']}^{{commit}}")
    root = rd / "builds" / engine_id
    installed = root / "bin" / platform_binary("ember")
    metadata_path = root / "metadata.json"
    if installed.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("revision") == revision and metadata.get(
            "sha256"
        ) == sha256_file(installed):
            return metadata

    source = root / "source"
    archive = root / "source.tar"
    if source.exists():
        shutil.rmtree(source)
    source.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            "git",
            "-C",
            str(repo),
            "archive",
            "--format=tar",
            f"--output={archive}",
            revision,
        ],
        log_path=root / "build.log",
    )
    run_cmd(
        ["tar", "-xf", str(archive), "-C", str(source)],
        log_path=root / "build.log",
    )
    command = list(
        build_cfg.get(
            "command",
            ["cargo", "build", "--locked", "--release", "--bin", "ember"],
        )
    )
    run_cmd(command, log_path=root / "build.log", cwd=source)
    built = source / platform_binary(build_cfg.get("binary", "target/release/ember"))
    if not built.is_file():
        raise RuntimeError(f"build did not produce {built}")
    installed.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built, installed)
    if os.name != "nt":
        installed.chmod(0o755)
    metadata = {
        "engine": engine_id,
        "configured_revision": engine["revision"],
        "revision": revision,
        "binary": str(installed.resolve()),
        "sha256": sha256_file(installed),
        "command": command,
    }
    write_json(metadata_path, metadata)
    return metadata


def record_revision_metadata(metadata, cfg, revision_metadata):
    metadata.setdefault("engine_binaries", {})
    metadata.setdefault("tools", {})
    for engine_id, binary in revision_metadata.items():
        metadata["engine_binaries"][cfg[engine_id]["name"]] = binary
        path = binary["binary"]
        metadata["tools"][path] = {"path": path, "available": True}


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")


def engine_thread_count(engine):
    raw = next(
        (
            value
            for name, value in engine.get("options", {}).items()
            if name.casefold() == "threads"
        ),
        1,
    )
    try:
        threads = int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{engine.get('name', 'engine')} has an invalid Threads option: {raw!r}"
        ) from error
    if threads < 1:
        raise ValueError(
            f"{engine.get('name', 'engine')} has a non-positive Threads option: {raw!r}"
        )
    return threads


def threads_per_game(cfg):
    return max(
        engine_thread_count(cfg["engine_a"]),
        engine_thread_count(cfg["engine_b"]),
    )


def batch_games_per_worker(run_cfg):
    value = int(run_cfg.get("batch_games_per_worker", 4))
    if value < 1:
        raise ValueError("run.batch_games_per_worker must be positive")
    return value


def worker_saturating_batch_pairs(run_cfg, workers, remaining_pairs):
    configured = max(1, int(run_cfg.get("batch_pairs", 16)))
    games_per_worker = batch_games_per_worker(run_cfg)
    worker_floor = math.ceil(workers * games_per_worker / 2)
    return min(remaining_pairs, max(configured, worker_floor))


def detect_workers(cfg, explicit_workers):
    cores = os.cpu_count() or 1
    if explicit_workers is not None:
        workers = int(explicit_workers)
        if workers < 1:
            raise ValueError("workers must be positive")
        return workers, cores, "cli"
    configured = str(cfg["run"].get("workers", "auto"))
    if configured != "auto":
        workers = int(configured)
        if workers < 1:
            raise ValueError("run.workers must be positive")
        return workers, cores, "config"
    multiplier = float(cfg["run"].get("worker_multiplier", 1.0))
    if multiplier <= 0.0:
        raise ValueError("run.worker_multiplier must be positive")
    search_threads = threads_per_game(cfg)
    workers = max(1, int(math.floor(cores * multiplier / search_threads)))
    return workers, cores, "auto"


def engine_args(engine):
    args = [
        "-engine",
        f"name={engine['name']}",
        f"cmd={engine['cmd']}",
        f"proto={engine.get('proto', 'uci')}",
    ]
    if "dir" in engine:
        args.append(f"dir={engine['dir']}")
    for arg in engine.get("args", []):
        args.append(f"arg={arg}")
    for key, value in engine.get("options", {}).items():
        if key.lower() == "syzygypath" and value and value.lower() != "<empty>":
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = path.resolve()
            value = str(path)
        args.append(f"option.{key}={value}")
    return args


def engine_limit_args(run_cfg):
    limits = []
    if "time_control" in run_cfg:
        limits.append(("tc", run_cfg["time_control"]))
    if "nodes" in run_cfg:
        limits.append(("nodes", int(run_cfg["nodes"])))
    if "depth" in run_cfg:
        limits.append(("depth", int(run_cfg["depth"])))
    if "move_time" in run_cfg:
        limits.append(("st", run_cfg["move_time"]))
    if len(limits) != 1:
        raise RuntimeError(
            "configure exactly one of time_control, nodes, depth, or move_time"
        )

    name, value = limits[0]
    args = [f"{name}={value}"]
    if name in {"nodes", "depth"}:
        args.insert(0, "tc=inf")
    if name in {"tc", "st"}:
        args.append(f"timemargin={int(run_cfg.get('timemargin_ms', 2000))}")
    return args


def read_opening_lines(path):
    lines = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    if not lines:
        raise RuntimeError(f"no openings in {path}")
    return lines


def sample_polyglot_openings(cfg, count):
    try:
        import chess
        import chess.polyglot
    except ImportError as e:
        raise RuntimeError(
            "polyglot opening sampling requires python-chess"
        ) from e

    run_cfg = cfg["run"]
    book = Path(run_cfg.get("polyglot_book", "src/book.bin"))
    if not book.exists():
        raise RuntimeError(f"polyglot book not found: {book}")

    seed = int(run_cfg.get("seed", 20260714))
    min_plies = int(run_cfg.get("book_min_plies", 8))
    max_plies = int(run_cfg.get("book_max_plies", 20))
    if min_plies > max_plies:
        raise RuntimeError("book_min_plies must be <= book_max_plies")

    rng = random.Random(seed)
    openings = []
    seen = set()
    attempts = max(count * 80, 2000)
    with chess.polyglot.open_reader(str(book)) as reader:
        for _ in range(attempts):
            board = chess.Board()
            target_plies = rng.randint(min_plies, max_plies)
            for _ply in range(target_plies):
                entries = list(reader.find_all(board))
                if not entries:
                    break
                weights = [max(1, entry.weight) for entry in entries]
                entry = rng.choices(entries, weights=weights, k=1)[0]
                board.push(entry.move)
                if board.is_game_over(claim_draw=True):
                    break
            if board.ply() < min_plies or board.is_game_over(claim_draw=True):
                continue
            epd = board.epd()
            if epd not in seen:
                seen.add(epd)
                openings.append(epd)
                if len(openings) >= count:
                    break
    if len(openings) < count:
        raise RuntimeError(f"sampled only {len(openings)} unique book openings, wanted {count}")
    return openings


def prepare_openings(cfg, rd, max_pairs):
    openings_path = rd / "openings" / "sampled.epd"
    identity_path = rd / "openings" / "manifest.json"
    if openings_path.exists():
        if not identity_path.exists():
            raise RuntimeError(
                f"existing openings lack an immutable manifest: {openings_path}"
            )
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        lines = read_opening_lines(openings_path)
        actual = {
            "sha256": sha256_file(openings_path),
            "count": len(lines),
            "max_pairs": max_pairs,
        }
        if identity != actual:
            raise RuntimeError("sampled openings changed or use different run limits")
        return openings_path, len(lines)

    source = cfg["run"].get("opening_source", "file")
    if source == "polyglot":
        lines = sample_polyglot_openings(cfg, max_pairs)
    elif source == "file":
        opening_file = Path(cfg["run"]["opening_file"])
        lines = read_opening_lines(opening_file)
        rng = random.Random(int(cfg["run"].get("seed", 20260714)))
        rng.shuffle(lines)
        lines = lines[:max_pairs]
    else:
        raise RuntimeError(f"unknown opening_source: {source}")

    write_text_atomic(openings_path, "\n".join(lines) + "\n")
    write_json(
        identity_path,
        {
            "sha256": sha256_file(openings_path),
            "count": len(lines),
            "max_pairs": max_pairs,
        },
    )
    return openings_path, len(lines)


def parse_pgn_file(path):
    games = []
    tags = {}
    order = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("[Event ") and tags:
                order += 1
                maybe_add_game(games, tags, path, order)
                tags = {}
            if line.startswith("[") and line.endswith("]"):
                m = re.match(r'^\[([A-Za-z0-9_]+)\s+"(.*)"\]$', line)
                if m:
                    tags[m.group(1)] = m.group(2)
    if tags:
        order += 1
        maybe_add_game(games, tags, path, order)
    return games


def maybe_add_game(games, tags, path, order):
    result = tags.get("Result")
    if result not in RESULT_SCORE:
        return
    white = tags.get("White")
    black = tags.get("Black")
    if not white or not black:
        return
    ws, bs = RESULT_SCORE[result]
    games.append(
        {
            "white": white,
            "black": black,
            "result": result,
            "white_score": ws,
            "black_score": bs,
            "termination": tags.get("Termination", ""),
            "fen": tags.get("FEN", ""),
            "round": tags.get("Round", ""),
            "event": tags.get("Event", ""),
            "pgn": str(path),
            "order": order,
        }
    )


def canonical_fen(value):
    fields = str(value).split()
    if len(fields) < 4:
        return None
    return " ".join(fields[:4])


def pair_opening_matches(entry, games):
    opening = Path(entry.get("opening", ""))
    if not opening.is_file():
        return False
    if sha256_file(opening) != entry.get("opening_sha256"):
        return False
    lines = read_opening_lines(opening)
    if len(lines) != 1:
        return False
    expected = canonical_fen(lines[0])
    return expected is not None and all(
        canonical_fen(game.get("fen")) == expected for game in games
    )


def parse_all_games(rd, engine_a=None, engine_b=None):
    games = []
    state_path = rd / "batch-state.json"
    if not state_path.exists():
        return games
    state = json.loads(state_path.read_text(encoding="utf-8"))
    completed = sorted(
        (
            entry
            for entry in state.get("pairs", {}).values()
            if entry.get("status") == "completed"
        ),
        key=lambda entry: int(entry["pair"]),
    )
    for entry in completed:
        pgn = Path(entry["pgn"])
        if not pgn.is_file():
            raise RuntimeError(f"completed pair PGN is missing: {pgn}")
        if sha256_file(pgn) != entry.get("pgn_sha256"):
            raise RuntimeError(f"completed pair PGN changed: {pgn}")
        parsed = parse_pgn_file(pgn)
        if not pair_opening_matches(entry, parsed):
            raise RuntimeError(f"completed pair opening does not match PGN: {pgn}")
        if engine_a is not None and not pair_evidence_is_complete(
            entry,
            engine_a,
            engine_b,
        ):
            raise RuntimeError(f"completed pair evidence is invalid: {pgn}")
        games.extend(parsed)
    return games


def score_for(game, player):
    if game["white"] == player:
        return game["white_score"]
    if game["black"] == player:
        return game["black_score"]
    raise ValueError(f"{player} not in game {game}")


def pair_games(games, engine_a, engine_b):
    pairs = []
    buckets = {}
    fallback = []
    ordered = sorted(games, key=lambda g: (g["pgn"], g["order"]))
    for game in ordered:
        if {game["white"], game["black"]} != {engine_a, engine_b}:
            continue
        if game.get("fen"):
            buckets.setdefault((game["pgn"], game["fen"]), []).append(game)
        else:
            fallback.append(game)

    chunks = []
    for chunk in buckets.values():
        chunks.extend(chunk[i : i + 2] for i in range(0, len(chunk) - 1, 2))
    chunks.extend(fallback[i : i + 2] for i in range(0, len(fallback) - 1, 2))

    for chunk in chunks:
        if len(chunk) != 2:
            continue
        colors_ok = (
            chunk[0]["white"] != chunk[1]["white"]
            and chunk[0]["black"] != chunk[1]["black"]
        )
        if colors_ok:
            a_score = sum(score_for(game, engine_a) for game in chunk)
            pairs.append(
                {
                    "pair": len(pairs) + 1,
                    "a_score": a_score,
                    "delta": a_score - 1.0,
                    "results": " ".join(game["result"] for game in chunk),
                    "pgn": chunk[0]["pgn"],
                    "rounds": " ".join(game["round"] for game in chunk),
                }
            )
    return pairs


def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def score_to_elo(score):
    eps = 1e-6
    score = min(1.0 - eps, max(eps, score))
    return 400.0 * math.log10(score / (1.0 - score))


def analyze_pairs(pairs, alpha):
    n = len(pairs)
    if n == 0:
        return {
            "pairs": 0,
            "games": 0,
            "score": 0.0,
            "score_rate": 0.0,
            "elo": None,
            "p_greater": None,
            "p_less": None,
            "p_two_sided": None,
            "los": None,
            "significant": False,
        }

    deltas = [pair["delta"] for pair in pairs]
    total_score = sum(pair["a_score"] for pair in pairs)
    mean_delta = sum(deltas) / n
    if n > 1:
        variance = sum((d - mean_delta) ** 2 for d in deltas) / (n - 1)
    else:
        variance = 0.0
    sd = math.sqrt(variance)
    se = sd / math.sqrt(n) if n > 0 else float("inf")
    if se == 0.0:
        if mean_delta > 0:
            z = float("inf")
        elif mean_delta < 0:
            z = float("-inf")
        else:
            z = 0.0
    else:
        z = mean_delta / se

    if math.isinf(z):
        cdf = 1.0 if z > 0 else 0.0
    else:
        cdf = normal_cdf(z)
    p_less = cdf
    p_greater = 1.0 - cdf
    p_two = min(1.0, 2.0 * min(p_less, p_greater))
    score_rate = total_score / (2.0 * n)
    ci_delta = 1.959963984540054 * se
    score_ci_low = max(0.0, (mean_delta - ci_delta + 1.0) / 2.0)
    score_ci_high = min(1.0, (mean_delta + ci_delta + 1.0) / 2.0)

    a_wins = sum(1 for p in pairs if p["a_score"] > 1.0)
    a_losses = sum(1 for p in pairs if p["a_score"] < 1.0)
    tied_pairs = n - a_wins - a_losses
    return {
        "pairs": n,
        "games": 2 * n,
        "score": total_score,
        "score_rate": score_rate,
        "pair_mean_delta": mean_delta,
        "pair_sd": sd,
        "pair_se": se,
        "z": z,
        "p_greater": p_greater,
        "p_less": p_less,
        "p_two_sided": p_two,
        "los": cdf,
        "elo": score_to_elo(score_rate),
        "elo_ci95_low": score_to_elo(score_ci_low),
        "elo_ci95_high": score_to_elo(score_ci_high),
        "a_won_pairs": a_wins,
        "a_lost_pairs": a_losses,
        "tied_pairs": tied_pairs,
        "significant": min(p_greater, p_less) <= alpha,
    }


def selected_p_value(stats, alternative):
    if alternative == "greater":
        return stats["p_greater"]
    if alternative == "less":
        return stats["p_less"]
    if alternative == "two-sided":
        return stats["p_two_sided"]
    raise RuntimeError(f"unknown alternative: {alternative}")


def decision(stats, cfg, fixed_sample_complete=True):
    run_cfg = cfg["run"]
    sprt_cfg = cfg.get("sprt", {})
    if sprt_cfg.get("enabled", False):
        min_pairs = int(sprt_cfg.get("min_pairs", run_cfg.get("min_pairs", 1)))
        if stats["pairs"] < min_pairs:
            return "continue"
        state = stats["sprt"]["state"]
        if state == "accept_h1":
            return "engine_a_better"
        if state == "accept_h0":
            return "engine_a_improvement_not_established"
        return "continue"

    if not fixed_sample_complete:
        return "continue"
    alpha = float(run_cfg.get("alpha", 0.05))
    min_pairs = int(run_cfg.get("min_pairs", 30))
    if stats["pairs"] < min_pairs:
        return "continue"
    alternative = run_cfg.get("alternative", "greater")
    p_value = selected_p_value(stats, alternative)
    if p_value is None or p_value > alpha:
        return "continue"
    if alternative == "greater":
        return "engine_a_better"
    if alternative == "less":
        return "engine_b_better"
    if stats.get("pair_mean_delta", 0.0) > 0.0:
        return "engine_a_better"
    if stats.get("pair_mean_delta", 0.0) < 0.0:
        return "engine_b_better"
    return "continue"


def capped_verdict(stats, verdict, max_pairs):
    if verdict == "continue" and stats["pairs"] >= max_pairs:
        return "inconclusive"
    return verdict


def write_pair_csv(rd, pairs):
    out = rd / "estimates" / "paired-openings.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["pair", "a_score", "delta", "results", "pgn", "rounds"]
        )
        writer.writeheader()
        writer.writerows(pairs)


def fmt_float(value, digits=1, suffix=""):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}{suffix}"


def fmt_sig(value):
    if value is None:
        return "n/a"
    return f"{value:.6g}"


def write_report(rd, cfg, stats, verdict):
    a = cfg["engine_a"]["name"]
    b = cfg["engine_b"]["name"]
    p = selected_p_value(stats, cfg["run"].get("alternative", "greater"))
    los = None if stats["los"] is None else 100.0 * stats["los"]
    lines = [
        "# Head-to-head report",
        "",
        f"Engine A: **{a}**",
        f"Engine B: **{b}**",
        f"Opening design: unique random openings, two games per opening, colors swapped.",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Pairs | {stats['pairs']} |",
        f"| Games | {stats['games']} |",
        f"| A score | {stats['score']:.1f} / {stats['games']} |",
        f"| A score rate | {100.0 * stats['score_rate']:.2f}% |",
        f"| Elo A-B | {fmt_float(stats['elo'])} |",
        f"| 95% Elo CI | {fmt_float(stats.get('elo_ci95_low'))} to {fmt_float(stats.get('elo_ci95_high'))} |",
        f"| Paired p(A>B) one-sided | {fmt_sig(stats['p_greater'])} |",
        f"| Paired p(A<B) one-sided | {fmt_sig(stats['p_less'])} |",
        f"| Paired p two-sided | {fmt_sig(stats['p_two_sided'])} |",
        f"| LOS-like P(A>B) | {fmt_float(los, 2, '%')} |",
        f"| A won/lost/tied pairs | {stats.get('a_won_pairs', 0)} / {stats.get('a_lost_pairs', 0)} / {stats.get('tied_pairs', 0)} |",
        "",
        f"Configured alpha: `{float(cfg['run'].get('alpha', 0.05))}`",
        f"Selected alternative: `{cfg['run'].get('alternative', 'greater')}`",
        f"Selected p-value: `{fmt_sig(p)}`",
        f"Verdict: **{verdict}**",
        "",
    ]
    analysis = stats.get("analysis")
    if analysis is not None:
        lines.extend(
            [
                "## Analysis provenance",
                "",
                f"Analyzed at: `{analysis['analyzed_at']}`",
                f"Analyzer SHA-256: `{analysis['tool_sha256']}`",
                "Historical analyzer override: "
                f"`{str(analysis['historical_tool_override']).lower()}`",
                "",
            ]
        )
    if "sprt" in stats:
        sprt = stats["sprt"]
        lines.extend(
            [
                "## Sequential test",
                "",
                f"Pentanomial: `{sprt['pentanomial']}`",
                f"Hypotheses: `{sprt['elo0']:.1f}` versus `{sprt['elo1']:.1f}` Elo",
                f"Alpha / beta: `{sprt['alpha']:.3f}` / `{sprt['beta']:.3f}`",
                f"LLR: `{sprt['llr']:.6f}`",
                f"Bounds: `{sprt['lower_bound']:.6f}` to `{sprt['upper_bound']:.6f}`",
                f"State: **{sprt['state']}**",
                "",
            ]
        )
    (rd / "report.md").write_text("\n".join(lines), encoding="utf-8")


def fixed_sample_target(rd, manifest):
    identity_path = rd / "openings" / "manifest.json"
    if identity_path.is_file():
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        return int(identity["count"])
    return int(manifest["identity"]["max_pairs"])


def _analyze(
    config_path,
    run_id,
    cfg_override=None,
    allow_tool_change=False,
):
    configured = load_config(config_path)
    manifest = load_run_manifest(
        config_path,
        configured,
        run_id,
        allow_tool_change=allow_tool_change,
    )
    cfg = cfg_override or manifest["identity"]["effective_config"]
    rd = run_dir_for(cfg, run_id)
    games = parse_all_games(
        rd,
        cfg["engine_a"]["name"],
        cfg["engine_b"]["name"],
    )
    pairs = pair_games(games, cfg["engine_a"]["name"], cfg["engine_b"]["name"])
    alpha = float(cfg["run"].get("alpha", 0.05))
    stats = analyze_pairs(pairs, alpha)
    sprt_cfg = cfg.get("sprt", {})
    if sprt_cfg.get("enabled", False) and pairs:
        stats["sprt"] = pentanomial_sprt(
            pentanomial_counts(pairs),
            float(sprt_cfg["elo0"]),
            float(sprt_cfg["elo1"]),
            float(sprt_cfg.get("alpha", 0.05)),
            float(sprt_cfg.get("beta", 0.05)),
        )
    target_pairs = fixed_sample_target(rd, manifest)
    fixed_sample_complete = stats["pairs"] >= target_pairs
    verdict = decision(
        stats,
        cfg,
        fixed_sample_complete=fixed_sample_complete,
    )
    stats["analysis"] = {
        "analyzed_at": now_utc(),
        "historical_tool_override": allow_tool_change,
        "tool_sha256": sha256_file(__file__),
    }
    write_pair_csv(rd, pairs)
    summary = dict(stats)
    summary["verdict"] = verdict
    write_json(rd / "estimates" / "summary.json", summary)
    write_report(rd, cfg, stats, verdict)
    return stats, verdict


def analyze(
    config_path,
    run_id,
    cfg_override=None,
    allow_tool_change=False,
):
    cfg = load_config(config_path)
    rd = run_dir_for(cfg, run_id)
    with exclusive_run_lock(rd, "analyze"):
        return _analyze(
            config_path,
            run_id,
            cfg_override,
            allow_tool_change,
        )


def _probe(
    config_path,
    run_id,
    explicit_workers=None,
    explicit_max_pairs=None,
    explicit_alpha=None,
):
    cfg = load_config(config_path)
    rd = run_dir_for(cfg, run_id)
    request = ensure_run_request(
        config_path,
        cfg,
        run_id,
        explicit_workers,
        explicit_max_pairs,
        explicit_alpha,
    )
    materialize_revision_commands(cfg, rd)
    workers, cores, worker_source = detect_workers(cfg, explicit_workers)
    search_threads = threads_per_game(cfg)
    configured_max_pairs = int(
        explicit_max_pairs
        if explicit_max_pairs is not None
        else cfg["run"].get("max_pairs", 200)
    )
    planned_batch_pairs = worker_saturating_batch_pairs(
        cfg["run"],
        workers,
        configured_max_pairs,
    )
    meta = {
        "run_id": run_id,
        "started_at": now_utc(),
        "hostname": socket.gethostname(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "request_fingerprint": request["fingerprint"],
        "cpu_count": cores,
        "workers": workers,
        "worker_source": worker_source,
        "search_threads_per_game": search_threads,
        "planned_search_threads": workers * search_threads,
        "configured_batch_pairs": max(
            1,
            int(cfg["run"].get("batch_pairs", 16)),
        ),
        "planned_batch_pairs": planned_batch_pairs,
        "engine_a": cfg["engine_a"],
        "engine_b": cfg["engine_b"],
        "tools": {},
    }
    commands = [
        "python3",
        cfg["run"].get("cutechess_cmd", "cutechess-cli"),
    ]
    for command in commands:
        executable = command_executable(command)
        path = shutil.which(executable)
        meta["tools"][command_label(command)] = {
            "path": path,
            "available": path is not None,
        }
    for engine in [cfg["engine_a"], cfg["engine_b"]]:
        exe = str(engine["cmd"]).split()[0]
        meta["tools"][engine["cmd"]] = {"path": shutil.which(exe), "available": shutil.which(exe) is not None}
    rd.mkdir(parents=True, exist_ok=True)
    write_json(rd / "metadata.json", meta)
    write_json(rd / "state.json", {"phase": "probe", "metadata": meta})
    print(
        json.dumps(
            {
                "run_id": run_id,
                "workers": workers,
                "cpu_count": cores,
                "search_threads_per_game": search_threads,
                "planned_search_threads": workers * search_threads,
                "configured_batch_pairs": max(
                    1,
                    int(cfg["run"].get("batch_pairs", 16)),
                ),
                "planned_batch_pairs": planned_batch_pairs,
            },
            indent=2,
        )
    )


def probe(
    config_path,
    run_id,
    explicit_workers=None,
    explicit_max_pairs=None,
    explicit_alpha=None,
):
    cfg = load_config(config_path)
    validate_run_configuration(cfg, explicit_workers, explicit_max_pairs)
    validate_external_commands(cfg)
    rd = run_dir_for(cfg, run_id)
    with exclusive_run_lock(rd, "probe"):
        return _probe(
            config_path,
            run_id,
            explicit_workers,
            explicit_max_pairs,
            explicit_alpha,
        )


def _build(
    config_path,
    run_id,
    explicit_workers=None,
    explicit_max_pairs=None,
    explicit_alpha=None,
):
    cfg = load_config(config_path)
    rd = run_dir_for(cfg, run_id)
    ensure_run_request(
        config_path,
        cfg,
        run_id,
        explicit_workers,
        explicit_max_pairs,
        explicit_alpha,
    )
    revision_engines = [
        engine_id
        for engine_id in ["engine_a", "engine_b"]
        if "revision" in cfg[engine_id]
    ]
    if revision_engines:
        revision_metadata = {}
        for engine_id in revision_engines:
            revision_metadata[engine_id] = build_revision(cfg, rd, engine_id)
        materialize_revision_commands(cfg, rd)
        metadata_path = rd / "metadata.json"
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {}
        )
        record_revision_metadata(metadata, cfg, revision_metadata)
        write_json(metadata_path, metadata)
        if len(revision_engines) == 2:
            prepare_run_manifest(
                config_path,
                cfg,
                run_id,
                explicit_workers,
                explicit_max_pairs,
                explicit_alpha,
            )
            return

    command = cfg.get("build", {}).get("command")
    if not command:
        return
    run_cmd(command, log_path=rd / "build.log")
    for engine in [cfg["engine_a"], cfg["engine_b"]]:
        cmd = Path(engine["cmd"])
        if cmd.exists() and cmd.is_file():
            meta_path = rd / "metadata.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            meta.setdefault("engine_binaries", {})[engine["name"]] = {
                "path": str(cmd),
                "sha256": sha256_file(cmd),
            }
            write_json(meta_path, meta)
    prepare_run_manifest(
        config_path,
        cfg,
        run_id,
        explicit_workers,
        explicit_max_pairs,
        explicit_alpha,
    )


def build(
    config_path,
    run_id,
    explicit_workers=None,
    explicit_max_pairs=None,
    explicit_alpha=None,
):
    cfg = load_config(config_path)
    validate_run_configuration(cfg, explicit_workers, explicit_max_pairs)
    validate_external_commands(cfg)
    rd = run_dir_for(cfg, run_id)
    with exclusive_run_lock(rd, "build"):
        return _build(
            config_path,
            run_id,
            explicit_workers,
            explicit_max_pairs,
            explicit_alpha,
        )


def pair_opening_path(rd, all_openings, pair_index):
    path = rd / "openings" / "pairs" / f"pair-{pair_index + 1:06d}.epd"
    expected = all_openings[pair_index] + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != expected:
            raise RuntimeError(f"pair opening changed: {path}")
    else:
        write_text_atomic(path, expected)
    return path


def pair_command(cfg, opening, pgn):
    cutechess = cfg["run"].get("cutechess_cmd", "cutechess-cli")
    if isinstance(cutechess, str):
        args = [cutechess]
    else:
        args = list(cutechess)
    args.extend(engine_args(cfg["engine_a"]))
    args.extend(engine_args(cfg["engine_b"]))
    args.extend(
        [
            "-each",
            *engine_limit_args(cfg["run"]),
            "-openings",
            f"file={opening}",
            f"format={cfg['run'].get('opening_format', 'epd')}",
            "order=sequential",
            "policy=round",
            "-games",
            "2",
            "-rounds",
            "1",
            "-repeat",
            "-concurrency",
            "1",
            "-pgnout",
            str(pgn),
            "-recover",
            "-ratinginterval",
            str(int(cfg["run"].get("rating_interval", 20))),
        ]
    )
    max_moves = int(cfg["run"].get("max_moves", 0))
    if max_moves > 0:
        args.extend(["-maxmoves", str(max_moves)])
    return args


def load_batch_state(rd):
    path = rd / "batch-state.json"
    if not path.exists():
        return {"schema_version": 1, "boundaries": [], "pairs": {}}
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("schema_version") != 1:
        raise RuntimeError("unsupported head-to-head batch-state schema")
    return state


def write_batch_state(rd, state):
    write_json(rd / "batch-state.json", state)


def pair_evidence_is_complete(entry, engine_a, engine_b):
    pgn = Path(entry["pgn"])
    if not pgn.is_file():
        return False
    games = parse_pgn_file(pgn)
    if len(games) != 2:
        return False
    if not pair_opening_matches(entry, games):
        return False
    pairs = pair_games(games, engine_a, engine_b)
    return len(pairs) == 1


def archive_incomplete_pair(rd, entry):
    pgn = Path(entry["pgn"])
    if not pgn.exists():
        return None
    interrupted = rd / "games" / "interrupted"
    interrupted.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    destination = interrupted / f"pair-{int(entry['pair']):06d}-{stamp}.pgn"
    os.replace(pgn, destination)
    return destination


def reconcile_pair_entry(rd, state, entry, engine_a, engine_b):
    pgn = Path(entry["pgn"])
    if entry["status"] == "completed":
        if not pair_evidence_is_complete(entry, engine_a, engine_b):
            raise RuntimeError(f"completed pair evidence is invalid: {pgn}")
        if sha256_file(pgn) != entry.get("pgn_sha256"):
            raise RuntimeError(f"completed pair evidence changed: {pgn}")
        return True
    if pair_evidence_is_complete(entry, engine_a, engine_b):
        entry["status"] = "completed"
        entry["pgn_sha256"] = sha256_file(pgn)
        entry["completed_at"] = now_utc()
        entry.pop("error", None)
        write_batch_state(rd, state)
        return True
    archived = archive_incomplete_pair(rd, entry)
    if archived is not None:
        entry.setdefault("interrupted_evidence", []).append(str(archived))
    entry["status"] = "pending"
    entry.pop("error", None)
    write_batch_state(rd, state)
    return False


def prepare_pair_batch(rd, cfg, all_openings, start, count):
    state = load_batch_state(rd)
    boundary = {"start_pair": start + 1, "pair_count": count}
    if boundary not in state["boundaries"]:
        state["boundaries"].append(boundary)
    entries = []
    for pair_index in range(start, start + count):
        opening = pair_opening_path(rd, all_openings, pair_index).resolve()
        pgn = (rd / "games" / "pairs" / f"pair-{pair_index + 1:06d}.pgn").resolve()
        args = pair_command(cfg, opening, pgn)
        expected = {
            "pair": pair_index + 1,
            "status": "pending",
            "opening": str(opening),
            "opening_sha256": sha256_file(opening),
            "pgn": str(pgn),
            "command": args,
            "command_sha256": stable_hash(args),
        }
        key = str(pair_index + 1)
        if key in state["pairs"]:
            entry = state["pairs"][key]
            for field in (
                "pair",
                "opening",
                "opening_sha256",
                "pgn",
                "command",
                "command_sha256",
            ):
                if entry.get(field) != expected[field]:
                    raise RuntimeError(f"pair {pair_index + 1} manifest changed")
        else:
            state["pairs"][key] = expected
        entries.append(state["pairs"][key])
    write_batch_state(rd, state)
    return state, entries


def run_pair_entry(rd, entry):
    pgn = Path(entry["pgn"])
    pgn.parent.mkdir(parents=True, exist_ok=True)
    log = rd / "games" / "logs" / f"pair-{int(entry['pair']):06d}.log"
    run_cmd(entry["command"], log_path=log)
    return int(entry["pair"])


def run_pair_batch(rd, cfg, state, entries, workers):
    engine_a = cfg["engine_a"]["name"]
    engine_b = cfg["engine_b"]["name"]
    pending = []
    for entry in entries:
        if not reconcile_pair_entry(rd, state, entry, engine_a, engine_b):
            entry["status"] = "active"
            entry["started_at"] = now_utc()
            pending.append(entry)
    write_batch_state(rd, state)
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_pair_entry, rd, entry): entry for entry in pending
        }
        for future in concurrent.futures.as_completed(futures):
            entry = futures[future]
            try:
                future.result()
                if not pair_evidence_is_complete(entry, engine_a, engine_b):
                    raise RuntimeError(
                        f"pair {entry['pair']} did not produce two color-swapped games"
                    )
                entry["status"] = "completed"
                entry["pgn_sha256"] = sha256_file(entry["pgn"])
                entry["completed_at"] = now_utc()
                entry.pop("error", None)
            except Exception as error:
                entry["status"] = "failed"
                entry["error"] = str(error)
                failures.append((entry["pair"], str(error)))
            write_batch_state(rd, state)
    if failures:
        details = "; ".join(f"pair {pair}: {error}" for pair, error in failures)
        raise RuntimeError(f"head-to-head batch failed: {details}")


def resume_declared_batches(rd, cfg, all_openings, workers):
    state = load_batch_state(rd)
    for boundary in state["boundaries"]:
        start = int(boundary["start_pair"]) - 1
        count = int(boundary["pair_count"])
        state, entries = prepare_pair_batch(rd, cfg, all_openings, start, count)
        if any(entry["status"] != "completed" for entry in entries):
            run_pair_batch(rd, cfg, state, entries, workers)


def _run_batches(
    config_path,
    run_id,
    explicit_workers=None,
    explicit_max_pairs=None,
    explicit_alpha=None,
):
    configured = load_config(config_path)
    manifest = prepare_run_manifest(
        config_path,
        configured,
        run_id,
        explicit_workers,
        explicit_max_pairs,
        explicit_alpha,
    )
    cfg = manifest["identity"]["effective_config"]
    rd = run_dir_for(cfg, run_id)
    rd.mkdir(parents=True, exist_ok=True)
    workers = int(manifest["identity"]["workers"])
    max_pairs = int(manifest["identity"]["max_pairs"])
    openings_path, opening_count = prepare_openings(cfg, rd, max_pairs)
    max_pairs = min(max_pairs, opening_count)
    all_openings = read_opening_lines(openings_path)
    resume_declared_batches(rd, cfg, all_openings, workers)
    start_pair = 0

    while start_pair < max_pairs:
        validate_run_manifest(manifest)
        stats, verdict = _analyze(config_path, run_id, cfg)
        if verdict != "continue":
            break
        start_pair = stats["pairs"]
        if start_pair >= max_pairs:
            break
        pairs_this_batch = worker_saturating_batch_pairs(
            cfg["run"],
            workers,
            max_pairs - start_pair,
        )
        state, entries = prepare_pair_batch(
            rd,
            cfg,
            all_openings,
            start_pair,
            pairs_this_batch,
        )
        run_pair_batch(rd, cfg, state, entries, workers)

    stats, verdict = _analyze(config_path, run_id, cfg)
    verdict = capped_verdict(stats, verdict, max_pairs)
    if verdict == "inconclusive":
        summary = dict(stats)
        summary["verdict"] = verdict
        write_json(rd / "estimates" / "summary.json", summary)
        write_report(rd, cfg, stats, verdict)
    state = {
        "phase": "finished",
        "finished_at": now_utc(),
        "stats": stats,
        "verdict": verdict,
        "max_pairs": max_pairs,
    }
    write_json(rd / "state.json", state)
    print(json.dumps(state, indent=2, sort_keys=True))


def run_batches(
    config_path,
    run_id,
    explicit_workers=None,
    explicit_max_pairs=None,
    explicit_alpha=None,
):
    cfg = load_config(config_path)
    validate_run_configuration(cfg, explicit_workers, explicit_max_pairs)
    rd = run_dir_for(cfg, run_id)
    materialized = copy.deepcopy(cfg)
    materialize_revision_commands(materialized, rd)
    validate_external_commands(materialized, require_engines=True)
    with exclusive_run_lock(rd, "run"):
        return _run_batches(
            config_path,
            run_id,
            explicit_workers,
            explicit_max_pairs,
            explicit_alpha,
        )


def main():
    parser = argparse.ArgumentParser(description="Paired-opening head-to-head runner")
    parser.add_argument("command", choices=["probe", "build", "run", "analyze", "all"])
    parser.add_argument("--config", default="configs/head-to-head/ember-syzygy.toml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--workers", default=None)
    parser.add_argument("--max-pairs", default=None)
    parser.add_argument("--alpha", default=None)
    parser.add_argument(
        "--allow-analysis-tool-change",
        action="store_true",
        help="regenerate derived reports with a different runner version",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = load_config(config_path)
    run_id = args.run_id or make_run_id(cfg)

    if args.command in {"probe", "all"}:
        probe(
            config_path,
            run_id,
            args.workers,
            args.max_pairs,
            args.alpha,
        )
    if args.command in {"build", "all"}:
        build(
            config_path,
            run_id,
            args.workers,
            args.max_pairs,
            args.alpha,
        )
    if args.command in {"run", "all"}:
        run_batches(config_path, run_id, args.workers, args.max_pairs, args.alpha)
    if args.command in {"analyze"}:
        stats, verdict = analyze(
            config_path,
            run_id,
            allow_tool_change=args.allow_analysis_tool_change,
        )
        print(json.dumps({"stats": stats, "verdict": verdict}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
