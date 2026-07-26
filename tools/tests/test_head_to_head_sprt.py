import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

from head_to_head import (  # noqa: E402
    capped_verdict,
    decision,
    detect_workers,
    engine_thread_count,
    materialize_revision_commands,
    record_revision_metadata,
    threads_per_game,
)


class HeadToHeadSprtTests(unittest.TestCase):
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
        self.assertEqual(decision(stats, cfg), "engine_b_better")

    def test_unresolved_test_becomes_inconclusive_at_the_cap(self):
        stats = {"pairs": 40}

        self.assertEqual(capped_verdict(stats, "continue", 41), "continue")
        self.assertEqual(capped_verdict(stats, "continue", 40), "inconclusive")
        self.assertEqual(
            capped_verdict(stats, "engine_a_better", 40), "engine_a_better"
        )

    def test_revision_commands_are_isolated_inside_the_run(self):
        cfg = {
            "engine_a": {"revision": "HEAD"},
            "engine_b": {"revision": "V1.1.2"},
        }
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            materialize_revision_commands(cfg, run_dir)

            self.assertEqual(
                Path(cfg["engine_a"]["cmd"]),
                (run_dir / "builds/engine_a/bin/ember").resolve(),
            )
            self.assertEqual(
                Path(cfg["engine_b"]["cmd"]),
                (run_dir / "builds/engine_b/bin/ember").resolve(),
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


if __name__ == "__main__":
    unittest.main()
