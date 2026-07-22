import math
import sys
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

from sprt import logistic_llr, pentanomial_counts, pentanomial_sprt  # noqa: E402


class SprtTests(unittest.TestCase):
    def test_pentanomial_counts_use_paired_scores(self):
        pairs = [
            {"a_score": 0.0},
            {"a_score": 0.5},
            {"a_score": 1.0},
            {"a_score": 1.5},
            {"a_score": 2.0},
            {"a_score": 1.0},
        ]

        self.assertEqual(pentanomial_counts(pairs), [1, 1, 2, 1, 1])

    def test_symmetric_results_have_zero_symmetric_llr(self):
        llr = logistic_llr([10, 20, 40, 20, 10], -5.0, 5.0)

        self.assertAlmostEqual(llr, 0.0, places=10)

    def test_reversing_results_reverses_symmetric_llr(self):
        counts = [3, 7, 21, 31, 38]

        forward = logistic_llr(counts, -5.0, 5.0)
        reverse = logistic_llr(list(reversed(counts)), -5.0, 5.0)

        self.assertAlmostEqual(forward, -reverse, places=10)

    def test_decisive_samples_cross_the_expected_bounds(self):
        winning = pentanomial_sprt([0, 0, 0, 20, 100], -5.0, 5.0)
        losing = pentanomial_sprt([100, 20, 0, 0, 0], -5.0, 5.0)

        self.assertEqual(winning["state"], "accept_h1")
        self.assertGreaterEqual(winning["llr"], winning["upper_bound"])
        self.assertEqual(losing["state"], "accept_h0")
        self.assertLessEqual(losing["llr"], losing["lower_bound"])
        self.assertTrue(math.isclose(winning["llr"], -losing["llr"], rel_tol=1e-12))


if __name__ == "__main__":
    unittest.main()
