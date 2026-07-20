from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import textwrap
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock


WINDOWS_DIR = Path(__file__).resolve().parents[1]


def load_module(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, WINDOWS_DIR / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


battle_runner = load_module("battle_runner.py", "battle_runner")
verify_bundle = load_module("verify_bundle.py", "verify_bundle")


class BattleRunnerTests(unittest.TestCase):
    def test_default_config(self) -> None:
        config = battle_runner.load_config(WINDOWS_DIR / "battle.toml")
        self.assertEqual(config.hash_mb, 1024)
        self.assertEqual(config.games[0].opponent, "Lynx_BOT")
        self.assertEqual(
            config.games[0].opponents,
            ["Lynx_BOT", "pawn_git", "simbelmyne-bot", "CubixChess", "bot_adario"],
        )
        self.assertEqual(config.challenge_timeout_seconds, 15)
        self.assertEqual(config.opponent_wait_timeout_seconds, 0)
        self.assertEqual(config.games[0].mode, "casual")
        self.assertFalse(config.games[0].scoring)

    def test_logical_cpu_count_is_capped(self) -> None:
        with mock.patch.object(battle_runner.os, "name", "posix"), mock.patch.object(
            battle_runner.os, "cpu_count", return_value=999
        ):
            self.assertEqual(battle_runner.logical_cpu_count(), 256)

    def test_benchmark_waits_for_async_bestmove(self) -> None:
        config = replace(
            battle_runner.load_config(WINDOWS_DIR / "battle.toml"),
            benchmark_depth=1,
            benchmark_repeats=1,
            benchmark_timeout_seconds=5,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = root / "async-engine"
            engine.write_text(
                textwrap.dedent(
                    f"""\
                    #!{sys.executable}
                    import sys
                    import threading
                    import time

                    search_finished = threading.Event()

                    def search():
                        time.sleep(0.02)
                        print("info depth 1 score cp 12 nodes 1234 nps 12000", flush=True)
                        search_finished.set()
                        print("bestmove e2e4", flush=True)

                    for command in sys.stdin:
                        command = command.strip()
                        if command.startswith("go "):
                            threading.Thread(target=search, daemon=True).start()
                        elif command == "quit":
                            if not search_finished.is_set():
                                raise SystemExit(9)
                            break
                    """
                ),
                encoding="utf-8",
            )
            engine.chmod(0o755)

            result = battle_runner.run_benchmark(engine, config, threads=2, output_dir=root)
            raw_log = (root / "benchmark-raw.log").read_text(encoding="utf-8")

        expected_searches = len(battle_runner.BENCHMARK_POSITIONS) + 1
        self.assertEqual(len(result["samples"]), len(battle_runner.BENCHMARK_POSITIONS))
        self.assertEqual(result["median_nps"], 12000)
        self.assertEqual(raw_log.count("returncode=0"), expected_searches)
        self.assertEqual(raw_log.count("bestmove e2e4"), expected_searches)

    def test_failed_benchmark_preserves_raw_engine_output(self) -> None:
        config = replace(
            battle_runner.load_config(WINDOWS_DIR / "battle.toml"),
            benchmark_depth=1,
            benchmark_repeats=1,
            benchmark_timeout_seconds=5,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = root / "failing-engine"
            engine.write_text(
                textwrap.dedent(
                    f"""\
                    #!{sys.executable}
                    import sys

                    for command in sys.stdin:
                        if command.startswith("go "):
                            print("info string deliberate benchmark failure", flush=True)
                            raise SystemExit(7)
                    """
                ),
                encoding="utf-8",
            )
            engine.chmod(0o755)

            with self.assertRaisesRegex(battle_runner.RunnerError, "failed on warmup"):
                battle_runner.run_benchmark(engine, config, threads=2, output_dir=root)
            raw_log = (root / "benchmark-raw.log").read_text(encoding="utf-8")

        self.assertIn("warmup depth=1 returncode=7", raw_log)
        self.assertIn("deliberate benchmark failure", raw_log)

    def test_parse_search_nps_uses_final_info_before_bestmove(self) -> None:
        value = battle_runner.parse_search_nps(
            [
                "info depth 1 nodes 10 nps 100",
                "info depth 2 nodes 40 nps 200",
                "bestmove e2e4",
                "info depth 1 nodes 20 nps 100",
                "bestmove e7e5",
            ]
        )
        self.assertEqual(value["searches"], 2)
        self.assertEqual(value["total_nodes"], 60)
        self.assertAlmostEqual(value["weighted_nps"], 150.0)

    def test_retry_after_is_safe_and_never_less_than_one_minute(self) -> None:
        self.assertEqual(battle_runner.retry_after_seconds("5"), 60)
        self.assertEqual(battle_runner.retry_after_seconds("120"), 120)
        self.assertEqual(battle_runner.retry_after_seconds("not-a-number"), 60)

    def test_retry_at_is_extracted_from_lichess_daily_limit(self) -> None:
        value = battle_runner.retry_at_from_lichess_error(
            "Bot played 100 games today, please wait until 2026-07-17T06:18:21.762Z to challenge them."
        )
        self.assertIsNotNone(value)
        assert value is not None
        self.assertEqual(value.isoformat(), "2026-07-17T06:18:21.762000+00:00")

    def test_challenge_stream_records_id_and_acceptance(self) -> None:
        response = mock.Mock(status_code=200, headers={})
        response.iter_lines.return_value = iter(
            [
                '{"challenge":{"id":"abc123"}}',
                '{"done":"accepted"}',
            ]
        )
        client = battle_runner.LichessClient("https://lichess.example", "secret")
        client.session.request = mock.Mock(return_value=response)
        game = battle_runner.load_config(WINDOWS_DIR / "battle.toml").games[0]

        challenge_id, outcome, initial = client.create_challenge(game, timeout_seconds=60)

        self.assertEqual(challenge_id, "abc123")
        self.assertEqual(outcome, "accepted")
        self.assertEqual(initial["challenge"]["id"], "abc123")
        request = client.session.request.call_args
        self.assertEqual(request.args[:2], ("POST", "https://lichess.example/api/challenge/Lynx_BOT"))
        self.assertEqual(request.kwargs["data"]["keepAliveStream"], "true")
        self.assertEqual(request.kwargs["timeout"], (30, 60))
        response.close.assert_called_once()

    def test_challenge_stream_enforces_acceptance_timeout(self) -> None:
        response = mock.Mock(status_code=200, headers={})
        response.iter_lines.return_value = iter(['{"challenge":{"id":"abc123"}}'])
        client = battle_runner.LichessClient("https://lichess.example", "secret")
        client.session.request = mock.Mock(return_value=response)
        game = battle_runner.load_config(WINDOWS_DIR / "battle.toml").games[0]

        with mock.patch.object(battle_runner.time, "monotonic", side_effect=[100.0, 100.0, 116.0]):
            challenge_id, outcome, initial = client.create_challenge(game, timeout_seconds=15)

        self.assertEqual(challenge_id, "abc123")
        self.assertEqual(outcome, "timeout")
        self.assertEqual(initial["challenge"]["id"], "abc123")
        request = client.session.request.call_args
        self.assertEqual(request.kwargs["timeout"], (30, 15))
        response.close.assert_called_once()

    def test_challenge_rejection_includes_lichess_error_and_closes_stream(self) -> None:
        response = mock.Mock(status_code=400, headers={})
        response.json.return_value = {"error": "The player does not accept challenges."}
        client = battle_runner.LichessClient("https://lichess.example", "secret")
        client.session.request = mock.Mock(return_value=response)
        game = battle_runner.load_config(WINDOWS_DIR / "battle.toml").games[0]

        with self.assertRaises(battle_runner.ChallengeRejected) as caught:
            client.create_challenge(game, timeout_seconds=60)

        self.assertEqual(caught.exception.status_code, 400)
        self.assertEqual(caught.exception.detail, "The player does not accept challenges.")
        self.assertIn("Lynx_BOT", str(caught.exception))
        response.close.assert_called_once()

    def test_opponent_pool_skips_busy_bot_and_uses_ready_bot(self) -> None:
        config = battle_runner.load_config(WINDOWS_DIR / "battle.toml")
        game = config.games[0]
        client = mock.Mock()
        client.opponents_status.return_value = {
            "lynx_bot": {"id": "lynx_bot", "online": True, "playing": True},
            "pawn_git": {"id": "pawn_git", "online": True, "playing": False},
        }
        client.create_challenge.return_value = (
            "game1234",
            "accepted",
            {"challenge": {"id": "game1234"}},
        )
        process = mock.Mock()
        process.poll.return_value = None

        result = battle_runner.wait_for_opponent_and_challenge(
            client, game, config, process, "[1/1]"
        )

        self.assertTrue(result["accepted"])
        self.assertEqual(result["opponent"], "pawn_git")
        client.create_challenge.assert_called_once_with(
            game, config.challenge_timeout_seconds, opponent="pawn_git"
        )

    def test_rejected_pool_candidate_falls_back_to_next_ready_bot(self) -> None:
        config = battle_runner.load_config(WINDOWS_DIR / "battle.toml")
        game = config.games[0]
        client = mock.Mock()
        client.opponents_status.return_value = {
            "lynx_bot": {"id": "lynx_bot", "online": True, "playing": False},
            "pawn_git": {"id": "pawn_git", "online": True, "playing": False},
        }
        client.create_challenge.side_effect = [
            battle_runner.ChallengeRejected(
                "Lynx_BOT",
                400,
                "Lynx_BOT played 100 games today, please wait until 2099-07-17T06:18:21Z to challenge them.",
            ),
            ("game1234", "accepted", {"challenge": {"id": "game1234"}}),
        ]
        process = mock.Mock()
        process.poll.return_value = None

        result = battle_runner.wait_for_opponent_and_challenge(
            client, game, config, process, "[1/1]"
        )

        self.assertEqual(result["opponent"], "pawn_git")
        self.assertEqual(result["attempts"][0]["status"], "REJECTED")
        self.assertEqual(result["attempts"][1]["status"], "ACCEPTED")

    def test_timeout_pool_candidate_is_cancelled_before_next_ready_bot(self) -> None:
        config = battle_runner.load_config(WINDOWS_DIR / "battle.toml")
        game = config.games[0]
        client = mock.Mock()
        client.opponents_status.return_value = {
            "lynx_bot": {"id": "lynx_bot", "online": True, "playing": False},
            "pawn_git": {"id": "pawn_git", "online": True, "playing": False},
        }
        client.create_challenge.side_effect = [
            ("challenge-timeout", "timeout", {"challenge": {"id": "challenge-timeout"}}),
            ("game1234", "accepted", {"challenge": {"id": "game1234"}}),
        ]
        process = mock.Mock()
        process.poll.return_value = None

        result = battle_runner.wait_for_opponent_and_challenge(
            client, game, config, process, "[1/1]"
        )

        self.assertEqual(result["opponent"], "pawn_git")
        self.assertEqual(result["attempts"][0]["status"], "TIMEOUT")
        self.assertEqual(result["attempts"][1]["status"], "ACCEPTED")
        client.cancel_challenge.assert_called_once_with("challenge-timeout")

    def test_game_monitor_retries_transient_disconnect(self) -> None:
        client = mock.Mock()
        client.game_json.side_effect = [
            battle_runner.TransientLichessError("connection dropped"),
            {"id": "game1234", "status": "mate"},
        ]
        process = mock.Mock()
        process.poll.return_value = None

        with mock.patch.object(battle_runner.time, "sleep"):
            result = battle_runner.wait_for_game(
                client, "game1234", process, poll_seconds=1, timeout_seconds=30
            )

        self.assertEqual(result["status"], "mate")
        self.assertEqual(client.game_json.call_count, 2)

    def test_single_opponent_config_remains_supported(self) -> None:
        original = (WINDOWS_DIR / "battle.toml").read_text(encoding="utf-8")
        start = original.index("opponents = [")
        end = original.index("]\n", start) + 2
        old_style = original[:start] + 'opponent = "Lynx_BOT"\n' + original[end:]
        old_style = "\n".join(
            line
            for line in old_style.splitlines()
            if not line.startswith(
                (
                    "availability_poll_seconds",
                    "opponent_retry_seconds",
                    "opponent_wait_timeout_seconds",
                )
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "battle.toml"
            path.write_text(old_style + "\n", encoding="utf-8")
            config = battle_runner.load_config(path)
        self.assertEqual(config.games[0].opponents, ["Lynx_BOT"])
        self.assertEqual(config.challenge_timeout_seconds, 15)
        self.assertEqual(config.availability_poll_seconds, 15)
        self.assertEqual(config.opponent_retry_seconds, 300)
        self.assertEqual(config.opponent_wait_timeout_seconds, 0)

    def test_generic_lichess_error_reports_json_detail(self) -> None:
        response = mock.Mock(status_code=403, reason="Forbidden")
        response.json.return_value = {"error": "Missing scope: challenge:write"}

        with self.assertRaisesRegex(
            battle_runner.RunnerError, "Missing scope: challenge:write"
        ):
            battle_runner.raise_for_lichess_error(response, "authentication check failed")

    def test_generated_config_has_no_token_or_tablebases(self) -> None:
        template = {
            "token": "placeholder",
            "url": "https://lichess.org/",
            "engine": {
                "dir": "",
                "name": "",
                "working_dir": "",
                "protocol": "uci",
                "debug": False,
                "ponder": False,
                "polyglot": {"enabled": True},
                "online_moves": {
                    "chessdb_book": {"enabled": True},
                    "lichess_cloud_analysis": {"enabled": True},
                    "lichess_opening_explorer": {"enabled": True},
                    "online_egtb": {"enabled": True},
                },
                "lichess_bot_tbs": {
                    "syzygy": {"enabled": True},
                    "gaviota": {"enabled": True},
                },
            },
            "challenge": {"concurrency": 5},
            "matchmaking": {"allow_matchmaking": True},
        }
        config = battle_runner.load_config(WINDOWS_DIR / "battle.toml")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template_path = root / "default.yml"
            template_path.write_text(battle_runner.yaml.safe_dump(template), encoding="utf-8")
            destination = root / "generated.yml"
            engine = root / "engine" / "ember.exe"
            engine.parent.mkdir()
            battle_runner.generate_lichess_config(
                template_path, destination, engine, root, config, threads=8
            )
            generated = battle_runner.yaml.safe_load(destination.read_text(encoding="utf-8"))
        self.assertEqual(generated["token"], "")
        self.assertEqual(generated["engine"]["uci_options"]["Threads"], 8)
        self.assertEqual(generated["engine"]["uci_options"]["SyzygyPath"], "")
        self.assertFalse(generated["engine"]["lichess_bot_tbs"]["syzygy"]["enabled"])
        self.assertFalse(generated["matchmaking"]["allow_matchmaking"])

    def test_manifest_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "payload.txt"
            payload.write_text("safe\n", encoding="utf-8")
            digest = verify_bundle.sha256_file(payload)
            manifest = root / "SHA256SUMS.txt"
            manifest.write_text(f"{digest}  payload.txt\n", encoding="utf-8")
            self.assertEqual(verify_bundle.verify(manifest), [])
            payload.write_text("changed\n", encoding="utf-8")
            self.assertEqual(verify_bundle.verify(manifest), ["CHANGED payload.txt"])

    def test_source_does_not_contain_token_value(self) -> None:
        marker = "this-must-never-be-in-the-source-or-zip"
        self.assertNotIn(marker, (WINDOWS_DIR / "battle_runner.py").read_text(encoding="utf-8"))
        self.assertNotIn(marker, (WINDOWS_DIR / "battle.toml").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
