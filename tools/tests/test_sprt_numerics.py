"""Small, self-contained likelihood regressions; no engine or saved results."""

import math
import sys
import unittest
from pathlib import Path


TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
sys.path.insert(0, str(TESTS.parent))

from sprt import logistic_llr, pentanomial_sprt  # noqa: E402
from sprt_decimal_reference import decimal_llr  # noqa: E402


HYPOTHESES = (("0", "5"), ("-5", "5"), ("0", "3"))
CASES = (
    (0, 0, 0, 0, 4000),  # Sparse near-pole normalization counterexample.
    (4000, 0, 0, 0, 0),
    (0, 0, 0, 1, 3998),
    (3998, 0, 0, 1, 0),
    (12, 34, 56, 34, 12),
    (40, 160, 600, 155, 45),
    (11, 103, 276, 161, 25),
    (0, 0, 0, 0, 0),
)


class SprtNumericsTests(unittest.TestCase):
    def assert_reference(self, counts, elo0, elo1):
        expected = float(decimal_llr(counts, elo0, elo1))
        actual = logistic_llr(counts, float(elo0), float(elo1))
        self.assertTrue(math.isfinite(actual))
        self.assertLessEqual(abs(actual - expected), 1e-9)
        return expected

    def test_original_all_win_normalization_counterexample(self):
        self.assert_reference((0, 0, 0, 0, 4000), "0", "5")

    def test_sparse_dense_and_zero_counts_match_primal_decimal_fits(self):
        for elo0, elo1 in HYPOTHESES:
            for counts in CASES:
                with self.subTest(counts=counts, hypotheses=(elo0, elo1)):
                    self.assert_reference(counts, elo0, elo1)

    def test_metadata_bounds_and_state_keep_unregularized_pair_counts(self):
        for elo0, elo1 in HYPOTHESES:
            for counts in (CASES[0], CASES[1], CASES[4], CASES[-1]):
                with self.subTest(counts=counts, hypotheses=(elo0, elo1)):
                    expected = float(decimal_llr(counts, elo0, elo1))
                    result = pentanomial_sprt(counts, float(elo0), float(elo1), .01, .05)
                    lower = math.log(.05 / .99)
                    upper = math.log(.95 / .01)
                    state = "accept_h0" if expected <= lower else "accept_h1" if expected >= upper else "continue"
                    self.assertEqual(set(result), {"alpha", "beta", "elo0", "elo1", "llr", "lower_bound", "upper_bound", "state", "pentanomial", "pairs"})
                    self.assertEqual(result["pentanomial"], list(counts))
                    self.assertEqual(result["pairs"], sum(counts))
                    self.assertEqual((result["alpha"], result["beta"]), (.01, .05))
                    self.assertEqual((result["elo0"], result["elo1"]), (float(elo0), float(elo1)))
                    self.assertLessEqual(abs(result["llr"] - expected), 1e-9)
                    self.assertLessEqual(abs(result["lower_bound"] - lower), 1e-14)
                    self.assertLessEqual(abs(result["upper_bound"] - upper), 1e-14)
                    self.assertEqual(result["state"], state)

    def test_symmetric_hypotheses_reverse_and_favor_positive_candidate(self):
        for counts in (CASES[0], CASES[4], CASES[6]):
            with self.subTest(counts=counts):
                forward = logistic_llr(counts, -5, 5)
                reverse = logistic_llr(counts[::-1], -5, 5)
                self.assertLessEqual(abs(forward + reverse), 1e-9)
                self.assert_reference(counts[::-1], "-5", "5")
        self.assertGreater(logistic_llr(CASES[0], -5, 5), 0)
        self.assertLess(logistic_llr(CASES[1], -5, 5), 0)
        self.assertLessEqual(abs(logistic_llr(CASES[4], -5, 5)), 1e-9)

    def test_asymmetric_hypotheses_use_separate_reversed_reference(self):
        counts = CASES[6]
        for elo1 in ("5", "3"):
            self.assert_reference(counts, "0", elo1)
            self.assert_reference(counts[::-1], "0", elo1)
            # Reversing outcomes also changes the relation to the nonzero midpoint.
            self.assertGreater(abs(logistic_llr(counts, 0, float(elo1)) + logistic_llr(counts[::-1], 0, float(elo1))), 1e-6)

    def test_zero_count_regularization_is_preserved(self):
        counts = (0, 0, 0, 0, 0)
        self.assert_reference(counts, "0", "5")
        # Existing semantics add 0.001 per empty bucket, but report zero real pairs.
        self.assertLessEqual(abs(logistic_llr(counts, 0, 5) - logistic_llr((.001,) * 5, 0, 5)), 1e-14)
        self.assertEqual(pentanomial_sprt(counts, 0, 5)["pairs"], 0)


if __name__ == "__main__":
    unittest.main()
