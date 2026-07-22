import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

from head_to_head import (  # noqa: E402
    materialize_revision_commands,
    record_revision_metadata,
)


class HeadToHeadSprtTests(unittest.TestCase):
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
