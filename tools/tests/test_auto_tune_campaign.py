import sys
import unittest
from pathlib import Path
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1]
AUTO_TUNE = TOOLS / "auto_tune"
sys.path.insert(0, str(AUTO_TUNE))

from campaign import (  # noqa: E402
    advance_campaign,
    pass_paths,
    pass_seeds,
    validate_seed_schedule,
)


def make_cfg():
    return {
        "common": {"seed": 100},
        "recheck": {"seed": 200},
        "confirmation": {"seed": 300},
    }


def make_state(max_passes=3):
    return {
        "session": {"max_passes": max_passes},
        "pass_index": 0,
        "phase": "recheck",
        "before_values": {},
        "active_seeds": [100, 200],
        "completed_passes": [],
    }


class CampaignPathTests(unittest.TestCase):
    def test_passes_use_isolated_seek_and_pending_state(self):
        state, pending = pass_paths("campaign.json", 1)
        self.assertEqual(state, Path("campaign.pass-2.seek.json"))
        self.assertEqual(pending, Path("campaign.pass-2.pending.json"))


class CampaignSeedTests(unittest.TestCase):
    def test_each_pass_uses_fresh_independent_seeds(self):
        cfg = make_cfg()
        self.assertEqual(pass_seeds(cfg, 0), (100, 200))
        self.assertEqual(pass_seeds(cfg, 2), (102, 202))
        validate_seed_schedule(cfg, 3)

    def test_rejects_seed_reuse_across_phases(self):
        cfg = make_cfg()
        cfg["recheck"]["seed"] = 101
        with self.assertRaisesRegex(ValueError, "must be unique"):
            validate_seed_schedule(cfg, 2)

    def test_rejects_confirmation_seed_reuse(self):
        cfg = make_cfg()
        cfg["confirmation"]["seed"] = 101
        with self.assertRaisesRegex(ValueError, "confirmation seed"):
            validate_seed_schedule(cfg, 2)


class CampaignAdvanceTests(unittest.TestCase):
    @patch("campaign.now_utc", return_value="2026-08-16T00:00:00+00:00")
    def test_stops_after_a_pass_without_changes(self, _now):
        state = make_state()
        self.assertEqual(advance_campaign(state, {}), "converged")
        self.assertFalse(state["completed_passes"][0]["changed"])

    @patch("campaign.now_utc", return_value="2026-08-16T00:00:00+00:00")
    def test_advances_after_a_changed_pass(self, _now):
        state = make_state()
        self.assertEqual(
            advance_campaign(state, {"PROBCUT_MIN_DEPTH": 9}),
            "continue",
        )
        self.assertEqual(state["pass_index"], 1)
        self.assertEqual(state["phase"], "discovery")
        self.assertEqual(state["before_values"], {"PROBCUT_MIN_DEPTH": 9})
        self.assertIsNone(state["active_seeds"])

    @patch("campaign.now_utc", return_value="2026-08-16T00:00:00+00:00")
    def test_stops_at_the_pass_limit(self, _now):
        state = make_state(max_passes=1)
        self.assertEqual(
            advance_campaign(state, {"PROBCUT_MIN_DEPTH": 9}),
            "pass_limit",
        )


if __name__ == "__main__":
    unittest.main()
