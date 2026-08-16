import copy
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
AUTO_TUNE = TOOLS / "auto_tune"
sys.path.insert(0, str(AUTO_TUNE))
sys.path.insert(0, str(TOOLS))

from seek import (  # noqa: E402
    load_pending,
    pending_add,
    pending_entries,
    should_pend,
    read_toml,
)
from recheck import (  # noqa: E402
    generate_recheck_config,
    prepare_config,
    recheck_run_id,
    update_pending_after_recheck,
)


def make_cfg():
    return {
        "results_dir": "results/tune",
        "common": {
            "timemargin_ms": 50,
            "seed": 1,
            "opening_source": "polyglot",
            "polyglot_book": "src/book.bin",
            "book_min_plies": 8,
            "book_max_plies": 20,
            "max_moves": 220,
            "hash_mb": 64,
            "threads": 1,
        },
        "confirmation": {
            "enabled": True,
            "time_control": "1+0.01",
            "max_pairs": 1000,
            "min_pairs": 20,
            "batch_pairs": 20,
            "seed": 2,
            "elo0": 0,
            "elo1": 3,
            "alpha": 0.05,
            "beta": 0.05,
        },
        "sprt": {
            "enabled": True,
            "elo0": 0,
            "elo1": 3,
            "alpha": 0.1,
            "beta": 0.05,
        },
        "recheck": {
            "enabled": True,
            "time_control": "1+0.01",
            "max_pairs": 4000,
            "min_pairs": 20,
            "batch_pairs": 20,
            "seed": 3,
            "min_elo": 5,
            "accept_elo_ge": 0.0,
        },
    }


def make_params():
    return [
        {"name": "PROBCUT_MIN_DEPTH", "base": 8, "min": 4, "max": 16, "step": 1},
        {"name": "PROBCUT_MARGIN_CP", "base": 350, "min": 200, "max": 600, "step": 25},
    ]


def make_record(verdict="inconclusive", elo=6.0):
    return {
        "run_id": "tune-probcut_min_depth-9-20260816-000000-000000",
        "timestamp": "2026-08-16T00:00:00+00:00",
        "param": "PROBCUT_MIN_DEPTH",
        "old_value": 8,
        "new_value": 9,
        "verdict": verdict,
        "accepted": verdict == "engine_a_better",
        "elo": elo,
        "score_rate": 0.5075,
        "pairs": 1000,
        "games": 2000,
        "llr": 0.24,
        "binary_sha256": "a" * 64,
        "time_control": "1+0.01",
        "sprt_elo0": 0,
        "sprt_elo1": 3,
        "sprt_alpha": 0.1,
        "sprt_beta": 0.05,
    }


class ShouldPendTests(unittest.TestCase):
    def test_pends_inconclusive_with_enough_elo(self):
        self.assertTrue(should_pend(make_record(), 5))

    def test_does_not_pend_low_elo(self):
        self.assertFalse(should_pend(make_record(elo=4.9), 5))

    def test_does_not_pend_rejected_verdict(self):
        self.assertFalse(should_pend(make_record(verdict="engine_b_better", elo=50), 5))

    def test_does_not_pend_accepted_verdict(self):
        self.assertFalse(should_pend(make_record(verdict="engine_a_better", elo=50), 5))

    def test_does_not_pend_missing_elo(self):
        record = make_record()
        record["elo"] = None
        self.assertFalse(should_pend(record, 5))


class PendingAddTests(unittest.TestCase):
    def test_add_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pending.json"
            record = make_record()
            first = pending_add(str(path), "PROBCUT_MIN_DEPTH", 9, record)
            second = pending_add(str(path), "PROBCUT_MIN_DEPTH", 9, record)
            self.assertEqual(first, second)
            self.assertEqual(second["status"], "pending")
            self.assertEqual(second["discovery_elo"], 6.0)
            self.assertEqual(second["discovery_run_id"], record["run_id"])

    def test_pending_entries_only_returns_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pending.json"
            pending_add(str(path), "PROBCUT_MIN_DEPTH", 9, make_record())
            pending = load_pending(str(path))
            pending["candidates"]["PROBCUT_MIN_DEPTH=9"]["status"] = "accepted"
            self.assertEqual(pending_entries(pending), [])


class RecheckConfigTests(unittest.TestCase):
    def setUp(self):
        self.cfg = make_cfg()
        self.params = make_params()
        self.best = {"values": {}}
        self.binary_sha256 = "a" * 64
        self.config = generate_recheck_config(
            self.cfg,
            self.best,
            self.params,
            "target/release/ember",
            "PROBCUT_MIN_DEPTH",
            9,
            self.binary_sha256,
        )

    def test_recheck_config_shape(self):
        self.assertEqual(self.config["engine_a"]["name"], "RecheckCandidate")
        self.assertEqual(
            self.config["engine_a"]["options"]["Tune"],
            "PROBCUT_MIN_DEPTH=9,PROBCUT_MARGIN_CP=350",
        )
        self.assertEqual(
            self.config["engine_b"]["options"]["Tune"],
            "PROBCUT_MIN_DEPTH=8,PROBCUT_MARGIN_CP=350",
        )
        self.assertEqual(self.config["run"]["seed"], 3)
        self.assertEqual(self.config["run"]["max_pairs"], 4000)
        self.assertEqual(
            self.config["run"]["results_dir"],
            str(Path("results/tune") / "rechecks"),
        )

    def test_recheck_run_id_is_deterministic(self):
        run_id = recheck_run_id(self.config)
        self.assertEqual(recheck_run_id(copy.deepcopy(self.config)), run_id)
        changed = copy.deepcopy(self.config)
        changed["engine_a"]["options"]["Tune"] = (
            "PROBCUT_MIN_DEPTH=10,PROBCUT_MARGIN_CP=350"
        )
        self.assertNotEqual(recheck_run_id(changed), run_id)

    def test_existing_run_must_have_the_exact_same_config(self):
        with tempfile.TemporaryDirectory() as directory:
            self.config["run"]["results_dir"] = directory
            run_id = recheck_run_id(self.config)
            path, resumed = prepare_config(self.config, run_id)
            self.assertFalse(resumed)
            self.assertEqual(read_toml(path), self.config)
            resumed_path, resumed = prepare_config(self.config, run_id)
            self.assertTrue(resumed)
            self.assertEqual(resumed_path, path)
            changed = copy.deepcopy(self.config)
            changed["run"]["time_control"] = "2+0.02"
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                prepare_config(changed, run_id)


class UpdatePendingAfterRecheckTests(unittest.TestCase):
    def test_accepted_status_written(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pending.json"
            pending_add(str(path), "PROBCUT_MIN_DEPTH", 9, make_record())
            summary = {"elo": 12.5, "pairs": 4000, "games": 8000, "verdict": "inconclusive"}
            entry = update_pending_after_recheck(
                str(path), "PROBCUT_MIN_DEPTH=9", "accepted", summary, "recheck-abc"
            )
            self.assertEqual(entry["status"], "accepted")
            self.assertEqual(entry["recheck_run_id"], "recheck-abc")
            self.assertEqual(entry["recheck_elo"], 12.5)
            self.assertEqual(entry["recheck_pairs"], 4000)


if __name__ == "__main__":
    unittest.main()