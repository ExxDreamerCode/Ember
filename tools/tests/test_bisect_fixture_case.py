import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from bisect_fixture_case import select_check  # noqa: E402


class BisectFixtureCaseTests(unittest.TestCase):
    def test_selects_one_depth_from_a_mined_fixture(self):
        fixture = """\
# failed_id\tfen_before_blunder\tsetup_move\texpected_move\tgot_depth2\tgot_depth3\tgot_depth4\tthemes\trating\tpopularity\tplays
# puzzle\tfen\tsetup\texpected\tgot2\tgot3\tgot4\ttheme\t0\t0\t0
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.tsv"
            path.write_text(fixture, encoding="utf-8")

            check = select_check(directory, "puzzle", 3)

        self.assertEqual(check.case_id, "puzzle")
        self.assertEqual(check.depth, 3)
        self.assertEqual(check.expected_move, "expected")

    def test_rejects_missing_or_ambiguous_checks(self):
        fixture = """\
id\tdepth\tfen_before_blunder\tsetup_move\texpected_move\tthemes\trating\tpopularity\tplays
duplicate\t4\tfen\t-\tmove\ttheme\t0\t0\t0
duplicate\t4\tfen\t-\tmove\ttheme\t0\t0\t0
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.tsv"
            path.write_text(fixture, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "found 2"):
                select_check(directory, "duplicate", 4)
            with self.assertRaisesRegex(ValueError, "found 0"):
                select_check(directory, "missing", 4)
