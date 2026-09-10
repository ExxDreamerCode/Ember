import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

from head_to_head import (  # noqa: E402
    _build,
    capped_verdict,
    command_executable,
    command_label,
    decision,
    detect_workers,
    engine_thread_count,
    ensure_run_request,
    exclusive_run_lock,
    load_config,
    materialize_revision_commands,
    platform_binary,
    parse_all_games,
    pair_evidence_is_complete,
    prepare_run_manifest,
    probe,
    reconcile_pair_entry,
    record_revision_metadata,
    run_dir_for,
    threads_per_game,
    validate_run_manifest,
    worker_saturating_batch_pairs,
)


class HeadToHeadSprtTests(unittest.TestCase):
    @staticmethod
    def write_pair_pgn(path, complete=True):
        games = [
            """[Event "pair"]
[White "candidate"]
[Black "baseline"]
[Result "1-0"]
[FEN "8/8/8/8/8/8/8/K6k w - - 0 1"]

1-0
""",
        ]
        if complete:
            games.append(
                """[Event "pair"]
[White "baseline"]
[Black "candidate"]
[Result "0-1"]
[FEN "8/8/8/8/8/8/8/K6k w - - 0 1"]

0-1
"""
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(games), encoding="utf-8")

    @staticmethod
    def pair_entry(root, pgn, status="active"):
        opening = root / "openings/pairs/pair-000001.epd"
        opening.parent.mkdir(parents=True, exist_ok=True)
        opening.write_text(
            "8/8/8/8/8/8/8/K6k w - -\n",
            encoding="utf-8",
        )
        import hashlib

        return {
            "pair": 1,
            "status": status,
            "opening": str(opening),
            "opening_sha256": hashlib.sha256(opening.read_bytes()).hexdigest(),
            "pgn": str(pgn),
        }

    @staticmethod
    def write_identity_config(root, network=None, engine_a_cmd=sys.executable):
        opening = root / "openings.epd"
        opening.write_text("8/8/8/8/8/8/8/K6k w - -\n", encoding="utf-8")
        config = root / "match.toml"
        nnue_option = ""
        if network is not None:
            nnue_option = f'\n[engine_a.options]\nNNUE = {json.dumps(str(network))}\n'
        config.write_text(
            f"""
[run]
name = "identity"
results_dir = {json.dumps(str(root / "results"))}
opening_source = "file"
opening_file = {json.dumps(str(opening))}
cutechess_cmd = {json.dumps(sys.executable)}
workers = 1
max_pairs = 4
depth = 1

[engine_a]
name = "candidate"
cmd = {json.dumps(str(engine_a_cmd))}
{nnue_option}
[engine_b]
name = "baseline"
cmd = {json.dumps(sys.executable)}
""",
            encoding="utf-8",
        )
        return config

    def test_command_executable_accepts_string_and_list_commands(self):
        self.assertEqual(
            command_executable("cutechess-cli -recover"),
            "cutechess-cli",
        )
        self.assertEqual(
            command_executable(
                ["/nix/store/cutechess/bin/cutechess-cli", "-recover"]
            ),
            "/nix/store/cutechess/bin/cutechess-cli",
        )
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            command_executable([])

    def test_command_label_preserves_strings_and_joins_lists(self):
        self.assertEqual(
            command_label("cutechess-cli -recover"),
            "cutechess-cli -recover",
        )
        self.assertEqual(
            command_label(
                ["/nix/store/cutechess/bin/cutechess-cli", "-recover"]
            ),
            "/nix/store/cutechess/bin/cutechess-cli -recover",
        )
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            command_label([])

    def test_probe_rejects_unavailable_list_form_cutechess_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "match.toml"
            config.write_text(
                f"""
[run]
cutechess_cmd = ["/not-installed/cutechess-cli", "-recover"]
results_dir = {json.dumps(str(root / "results"))}
workers = 1
depth = 1
opening_source = "file"
opening_file = {json.dumps(str(root / "openings.epd"))}

[engine_a]
name = "a"
cmd = {json.dumps(sys.executable)}

[engine_b]
name = "b"
cmd = {json.dumps(sys.executable)}
""",
                encoding="utf-8",
            )
            (root / "openings.epd").write_text(
                "8/8/8/8/8/8/8/K6k w - -\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "executable is unavailable"):
                probe(config, "list-command", explicit_workers=1)
            self.assertFalse((root / "results/list-command").exists())

    @staticmethod
    def worker_config(workers="auto", engine_a_threads="1", engine_b_threads="1"):
        return {
            "run": {"workers": workers, "worker_multiplier": 1.0},
            "engine_a": {
                "name": "engine-a",
                "options": {"Threads": engine_a_threads},
            },
            "engine_b": {
                "name": "engine-b",
                "options": {"Threads": engine_b_threads},
            },
        }

    @patch("head_to_head.os.cpu_count", return_value=16)
    def test_auto_workers_fill_cores_without_oversubscribing_engine_threads(
        self, _cpu_count
    ):
        single_threaded = self.worker_config()
        multi_threaded = self.worker_config(
            engine_a_threads="2", engine_b_threads="4"
        )

        self.assertEqual(threads_per_game(single_threaded), 1)
        self.assertEqual(detect_workers(single_threaded, None), (16, 16, "auto"))
        self.assertEqual(threads_per_game(multi_threaded), 4)
        self.assertEqual(detect_workers(multi_threaded, None), (4, 16, "auto"))

    @patch("head_to_head.os.cpu_count", return_value=16)
    def test_explicit_worker_counts_override_automatic_sizing(self, _cpu_count):
        cfg = self.worker_config(workers="3", engine_a_threads="4")

        self.assertEqual(detect_workers(cfg, None), (3, 16, "config"))
        self.assertEqual(detect_workers(cfg, 7), (7, 16, "cli"))

    def test_engine_threads_must_be_positive_integers(self):
        with self.assertRaisesRegex(ValueError, "invalid Threads"):
            engine_thread_count({"name": "bad", "options": {"Threads": "many"}})
        with self.assertRaisesRegex(ValueError, "non-positive Threads"):
            engine_thread_count({"name": "bad", "options": {"Threads": "0"}})

    def test_batch_size_keeps_multiple_games_queued_per_worker(self):
        run_cfg = {"batch_pairs": 20, "batch_games_per_worker": 4}

        self.assertEqual(worker_saturating_batch_pairs(run_cfg, 4, 100), 20)
        self.assertEqual(worker_saturating_batch_pairs(run_cfg, 16, 100), 32)
        self.assertEqual(worker_saturating_batch_pairs(run_cfg, 16, 12), 12)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            worker_saturating_batch_pairs(
                {"batch_games_per_worker": 0},
                16,
                100,
            )

    def test_sprt_decision_waits_for_minimum_pairs_and_maps_hypotheses(self):
        cfg = {
            "run": {"min_pairs": 2},
            "sprt": {"enabled": True, "min_pairs": 3},
        }
        stats = {"pairs": 2, "sprt": {"state": "accept_h1"}}
        self.assertEqual(decision(stats, cfg), "continue")

        stats["pairs"] = 3
        self.assertEqual(decision(stats, cfg), "engine_a_better")
        stats["sprt"]["state"] = "accept_h0"
        self.assertEqual(
            decision(stats, cfg),
            "engine_a_improvement_not_established",
        )

    def test_unresolved_test_becomes_inconclusive_at_the_cap(self):
        stats = {"pairs": 40}

        self.assertEqual(capped_verdict(stats, "continue", 41), "continue")
        self.assertEqual(capped_verdict(stats, "continue", 40), "inconclusive")
        self.assertEqual(
            capped_verdict(stats, "engine_a_better", 40), "engine_a_better"
        )
        self.assertEqual(
            capped_verdict(
                stats,
                "engine_a_improvement_not_established",
                40,
            ),
            "engine_a_improvement_not_established",
        )

    def test_fixed_sample_waits_for_cap_and_obeys_selected_alternative(self):
        cfg = {
            "run": {
                "min_pairs": 3,
                "alpha": 0.05,
                "alternative": "less",
            }
        }
        stats = {
            "pairs": 3,
            "p_greater": 0.99,
            "p_less": 0.01,
            "p_two_sided": 0.02,
            "pair_mean_delta": -0.5,
        }

        self.assertEqual(
            decision(stats, cfg, fixed_sample_complete=False),
            "continue",
        )
        self.assertEqual(decision(stats, cfg), "engine_b_better")
        cfg["run"]["alternative"] = "greater"
        self.assertEqual(decision(stats, cfg), "continue")
        cfg["run"]["alternative"] = "two-sided"
        self.assertEqual(decision(stats, cfg), "engine_b_better")

    def test_revision_commands_are_isolated_inside_the_run(self):
        cfg = {
            "engine_a": {"revision": "HEAD"},
            "engine_b": {"revision": "V1.1.2"},
        }
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            materialize_revision_commands(cfg, run_dir)

            for engine_id in ("engine_a", "engine_b"):
                self.assertEqual(
                    Path(cfg[engine_id]["cmd"]),
                    (
                        run_dir
                        / "builds"
                        / engine_id
                        / "bin"
                        / platform_binary("ember")
                    ).resolve(),
                )

    def test_built_revisions_replace_stale_probe_availability(self):
        cfg = {"engine_a": {"name": "candidate"}}
        binary = "/tmp/run/builds/engine_a/bin/ember"
        metadata = {
            "tools": {binary: {"path": None, "available": False}},
        }
        revision_metadata = {
            "engine_a": {
                "binary": binary,
                "revision": "0123456789abcdef",
                "sha256": "fedcba9876543210",
            }
        }

        record_revision_metadata(metadata, cfg, revision_metadata)

        self.assertEqual(
            metadata["engine_binaries"]["candidate"],
            revision_metadata["engine_a"],
        )
        self.assertEqual(
            metadata["tools"][binary],
            {"path": binary, "available": True},
        )

    def test_build_phase_keeps_the_run_request_fingerprint_stable(self):
        import subprocess

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / "README.md").write_text("identity fixture\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "-c",
                    "user.email=fixture@example.invalid",
                    "-c",
                    "user.name=fixture",
                    "commit",
                    "-qm",
                    "identity fixture",
                ],
                check=True,
            )
            head = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

            opening = root / "openings.epd"
            opening.write_text("8/8/8/8/8/8/8/K6k w - -\n", encoding="utf-8")
            config_path = root / "match.toml"
            config_path.write_text(
                f"""
[run]
name = "identity"
results_dir = {json.dumps(str(root / "results"))}
opening_source = "file"
opening_file = {json.dumps(str(opening))}
cutechess_cmd = {json.dumps(sys.executable)}
workers = 1
max_pairs = 4
depth = 1

[build]
repo = {json.dumps(str(root))}
command = ["true"]

[engine_a]
name = "candidate"
revision = {json.dumps(head)}

[engine_b]
name = "baseline"
revision = {json.dumps(head)}
""",
                encoding="utf-8",
            )

            def fake_build_revision(cfg, run_dir, engine_id):
                binary = (
                    run_dir
                    / "builds"
                    / engine_id
                    / "bin"
                    / platform_binary("ember")
                )
                binary.parent.mkdir(parents=True, exist_ok=True)
                binary.write_bytes(b"fake engine")
                # resolve_executable requires the executable bit on
                # POSIX, so the fake binary must be launchable there.
                binary.chmod(0o755)
                return {
                    "engine": engine_id,
                    "revision": "deadbeef",
                    "binary": str(binary),
                    "sha256": "0" * 64,
                    "command": ["true"],
                }

            # The probe phase records the run request from the pristine
            # configuration, exactly like the `probe`/`run` phases do.
            ensure_run_request(config_path, load_config(config_path), "run")

            with patch(
                "head_to_head.build_revision", side_effect=fake_build_revision
            ):
                _build(config_path, "run")

            # The run phase validates the same pristine configuration again;
            # a build phase that materialized commands into the live config
            # would have changed the fingerprint and rejected the run.
            ensure_run_request(config_path, load_config(config_path), "run")

    def test_run_request_allows_only_exact_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self.write_identity_config(root)
            cfg = load_config(config_path)

            first = ensure_run_request(config_path, cfg, "same", 1, 4, 0.025)
            second = ensure_run_request(config_path, cfg, "same", 1, 4, 0.025)
            self.assertEqual(first, second)

            with self.assertRaisesRegex(RuntimeError, "different immutable inputs"):
                ensure_run_request(config_path, cfg, "same", 2, 4, 0.025)

    def test_run_request_rejects_changed_external_nnue_at_same_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            network = root / "candidate.nnue"
            network.write_bytes(b"first network")
            config_path = self.write_identity_config(root, network)
            cfg = load_config(config_path)
            ensure_run_request(config_path, cfg, "network")

            network.write_bytes(b"different network")
            with self.assertRaisesRegex(RuntimeError, "different immutable inputs"):
                ensure_run_request(config_path, load_config(config_path), "network")

    def test_run_request_rejects_changed_config_and_swapped_orientation(self):
        mutations = {
            "changed config": lambda text: text.replace(
                "max_pairs = 4", "max_pairs = 5"
            ),
            "swapped orientation": lambda text: text.replace(
                'name = "candidate"', 'name = "temporary"'
            )
            .replace('name = "baseline"', 'name = "candidate"')
            .replace('name = "temporary"', 'name = "baseline"'),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config_path = self.write_identity_config(root)
                ensure_run_request(config_path, load_config(config_path), "same")

                config_path.write_text(
                    mutate(config_path.read_text(encoding="utf-8")),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    RuntimeError, "different immutable inputs"
                ):
                    ensure_run_request(
                        config_path,
                        load_config(config_path),
                        "same",
                    )

    def test_manifest_rejects_changed_engine_executable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = root / "engine"
            engine.write_bytes(b"first executable")
            engine.chmod(0o755)
            config_path = self.write_identity_config(root, engine_a_cmd=engine)
            manifest = prepare_run_manifest(
                config_path,
                load_config(config_path),
                "binary",
            )

            engine.write_bytes(b"changed executable")
            with self.assertRaisesRegex(RuntimeError, "engine binary changed"):
                validate_run_manifest(manifest)

    def test_manifest_executes_resolved_target_after_symlink_retarget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            probe_target = root / "symlink-probe-target"
            probe_target.write_bytes(b"")
            probe = root / "symlink-probe"
            try:
                probe.symlink_to(probe_target)
            except OSError:
                self.skipTest("creating symlinks requires privileges on this platform")
            probe.unlink()

            first = root / "engine-first"
            second = root / "engine-second"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            first.chmod(0o755)
            second.chmod(0o755)
            link = root / "engine"
            link.symlink_to(first)
            config_path = self.write_identity_config(root, engine_a_cmd=link)

            manifest = prepare_run_manifest(
                config_path,
                load_config(config_path),
                "symlink",
            )
            link.unlink()
            link.symlink_to(second)

            self.assertEqual(
                manifest["identity"]["effective_config"]["engine_a"]["cmd"],
                str(first.resolve()),
            )
            validate_run_manifest(manifest)

    @unittest.skipIf(
        os.name == "nt",
        "Windows has no executable permission bit; head_to_head skips the"
        " X_OK check there",
    )
    def test_manifest_rejects_non_executable_engine_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = root / "engine"
            engine.write_bytes(b"not executable")
            config_path = self.write_identity_config(root, engine_a_cmd=engine)

            with self.assertRaisesRegex(RuntimeError, "not executable"):
                prepare_run_manifest(
                    config_path,
                    load_config(config_path),
                    "permissions",
                )
            self.assertFalse(
                (root / "results/permissions/run-request.json").exists()
            )

    def test_manifest_preflight_rejects_duplicate_names_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self.write_identity_config(root)
            text = config_path.read_text(encoding="utf-8").replace(
                'name = "baseline"',
                'name = "candidate"',
            )
            config_path.write_text(text, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "names must be distinct"):
                prepare_run_manifest(
                    config_path,
                    load_config(config_path),
                    "duplicate",
                )
            self.assertFalse((root / "results/duplicate").exists())

    def test_engine_relative_inputs_resolve_from_engine_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine_dir = root / "engine-dir"
            engine_dir.mkdir()
            engine = engine_dir / "engine"
            network = engine_dir / "network.nnue"
            engine.write_bytes(b"engine")
            engine.chmod(0o755)
            network.write_bytes(b"network")
            config_path = self.write_identity_config(root)
            text = config_path.read_text(encoding="utf-8")
            text = text.replace(
                f'[engine_a]\nname = "candidate"\ncmd = {json.dumps(sys.executable)}',
                '[engine_a]\nname = "candidate"\ncmd = "engine"\ndir = '
                + json.dumps(str(engine_dir))
                + '\n\n[engine_a.options]\nNNUE = "network.nnue"',
            )
            config_path.write_text(text, encoding="utf-8")

            manifest = prepare_run_manifest(
                config_path,
                load_config(config_path),
                "relative",
            )

            effective = manifest["identity"]["effective_config"]["engine_a"]
            self.assertEqual(effective["cmd"], str(engine.resolve()))
            self.assertTrue(Path(effective["options"]["NNUE"]).is_file())

    def test_manifest_rejects_changed_syzygy_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tables = root / "tables"
            tables.mkdir()
            table = tables / "KvK.rtbw"
            table.write_bytes(b"table")
            config_path = self.write_identity_config(root)
            text = config_path.read_text(encoding="utf-8").replace(
                f'[engine_a]\nname = "candidate"\ncmd = {json.dumps(sys.executable)}',
                f'[engine_a]\nname = "candidate"\ncmd = {json.dumps(sys.executable)}'
                '\n\n[engine_a.options]\nSyzygyPath = '
                + json.dumps(str(tables)),
            )
            config_path.write_text(text, encoding="utf-8")
            manifest = prepare_run_manifest(
                config_path,
                load_config(config_path),
                "syzygy",
            )

            table.write_bytes(b"changed table")
            with self.assertRaisesRegex(RuntimeError, "directory changed"):
                validate_run_manifest(manifest)

    def test_manifest_validates_its_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self.write_identity_config(root)
            manifest = prepare_run_manifest(
                config_path,
                load_config(config_path),
                "fingerprint",
            )
            manifest["identity"]["workers"] = 99

            with self.assertRaisesRegex(RuntimeError, "fingerprint"):
                validate_run_manifest(manifest)

    def test_run_lock_reports_the_current_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            rd = Path(directory) / "run"
            with exclusive_run_lock(rd, "first"):
                with self.assertRaisesRegex(RuntimeError, '"operation": "first"'):
                    with exclusive_run_lock(rd, "second"):
                        self.fail("a second owner acquired the run lock")
            self.assertFalse((rd / ".run.lock").exists())

    def test_legacy_games_without_request_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self.write_identity_config(root)
            cfg = load_config(config_path)
            games = run_dir_for(cfg, "legacy") / "games"
            games.mkdir(parents=True)
            (games / "old.pgn").write_text("[Event \"old\"]\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "legacy evidence"):
                ensure_run_request(config_path, cfg, "legacy")

    def test_manifest_stages_and_revalidates_file_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            network = root / "candidate.nnue"
            network.write_bytes(b"immutable network")
            config_path = self.write_identity_config(root, network)
            cfg = load_config(config_path)

            manifest = prepare_run_manifest(config_path, cfg, "staged")
            staged = Path(
                manifest["identity"]["staged_inputs"]["engine_a-NNUE"][
                    "staged_path"
                ]
            )
            self.assertEqual(staged.read_bytes(), b"immutable network")
            validate_run_manifest(manifest)

            staged.write_bytes(b"mutated")
            with self.assertRaisesRegex(RuntimeError, "staged input changed"):
                validate_run_manifest(manifest)

    def test_interrupted_pair_evidence_is_preserved_before_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            rd = Path(directory)
            pgn = rd / "games/pairs/pair-000001.pgn"
            self.write_pair_pgn(pgn, complete=False)
            entry = self.pair_entry(rd, pgn)
            state = {"schema_version": 1, "boundaries": [], "pairs": {"1": entry}}

            self.assertFalse(
                reconcile_pair_entry(
                    rd,
                    state,
                    entry,
                    "candidate",
                    "baseline",
                )
            )
            self.assertEqual(entry["status"], "pending")
            self.assertFalse(pgn.exists())
            archived = [Path(path) for path in entry["interrupted_evidence"]]
            self.assertEqual(len(archived), 1)
            self.assertTrue(archived[0].is_file())

    def test_completed_pair_is_reconciled_and_only_manifest_games_are_read(self):
        with tempfile.TemporaryDirectory() as directory:
            rd = Path(directory)
            pgn = rd / "games/pairs/pair-000001.pgn"
            stale = rd / "games/pairs/stale.pgn"
            self.write_pair_pgn(pgn)
            self.write_pair_pgn(stale)
            entry = self.pair_entry(rd, pgn)
            state = {"schema_version": 1, "boundaries": [], "pairs": {"1": entry}}

            self.assertTrue(
                reconcile_pair_entry(
                    rd,
                    state,
                    entry,
                    "candidate",
                    "baseline",
                )
            )
            self.assertEqual(entry["status"], "completed")
            games = parse_all_games(rd)
            self.assertEqual(len(games), 2)
            self.assertTrue(all(Path(game["pgn"]) == pgn for game in games))

    def test_pair_evidence_must_match_the_declared_opening(self):
        with tempfile.TemporaryDirectory() as directory:
            rd = Path(directory)
            pgn = rd / "games/pairs/pair-000001.pgn"
            self.write_pair_pgn(pgn)
            entry = self.pair_entry(rd, pgn)
            opening = Path(entry["opening"])
            opening.write_text(
                "8/8/8/8/8/8/K7/7k w - -\n",
                encoding="utf-8",
            )
            import hashlib

            entry["opening_sha256"] = hashlib.sha256(
                opening.read_bytes()
            ).hexdigest()

            self.assertFalse(
                pair_evidence_is_complete(entry, "candidate", "baseline")
            )


if __name__ == "__main__":
    unittest.main()
