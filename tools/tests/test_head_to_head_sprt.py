import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

from head_to_head import (  # noqa: E402
    capped_verdict,
    decision,
    materialize_revision_commands,
    record_revision_metadata,
)


class HeadToHeadSprtTests(unittest.TestCase):
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
