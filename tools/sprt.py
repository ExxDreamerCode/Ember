#!/usr/bin/env python3
"""Pentanomial generalized SPRT helpers for paired chess games.

The likelihood calculation follows the logistic-Elo GSPRT used by the
official Stockfish fishtest project:
https://github.com/official-stockfish/fishtest/blob/master/server/fishtest/stats/LLRcalc.py
"""

import math


def logistic_score(elo):
    return 1.0 / (1.0 + 10.0 ** (-float(elo) / 400.0))


def pentanomial_counts(pairs):
    counts = [0, 0, 0, 0, 0]
    for pair in pairs:
        score = float(pair["a_score"])
        index = round(score * 2.0)
        if index not in range(5) or not math.isclose(score, index / 2.0):
            raise ValueError(f"invalid paired score for SPRT: {score}")
        counts[index] += 1
    return counts


def _regularized_counts(counts):
    if len(counts) != 5 or any(count < 0 for count in counts):
        raise ValueError("pentanomial counts must contain five nonnegative values")
    return [float(count) if count else 1e-3 for count in counts]


def _mle_with_expected_score(empirical_pdf, expected_score):
    values = [index / 4.0 for index in range(5)]
    shifts = [value - expected_score for value in values]
    negative = min(shifts)
    positive = max(shifts)
    if not negative < 0.0 < positive:
        raise ValueError("expected score must lie strictly inside the outcome support")

    epsilon = 1e-12
    lower = -1.0 / positive + epsilon
    upper = -1.0 / negative - epsilon

    def secular(value):
        return sum(
            probability * shift / (1.0 + value * shift)
            for probability, shift in zip(empirical_pdf, shifts)
        )

    for _ in range(120):
        midpoint = (lower + upper) / 2.0
        if secular(midpoint) > 0.0:
            lower = midpoint
        else:
            upper = midpoint

    root = (lower + upper) / 2.0
    probabilities = [
        probability / (1.0 + root * shift)
        for probability, shift in zip(empirical_pdf, shifts)
    ]
    total = sum(probabilities)
    return [probability / total for probability in probabilities]


def logistic_llr(counts, elo0, elo1):
    regularized = _regularized_counts(counts)
    total = sum(regularized)
    empirical_pdf = [count / total for count in regularized]
    hypothesis0 = _mle_with_expected_score(empirical_pdf, logistic_score(elo0))
    hypothesis1 = _mle_with_expected_score(empirical_pdf, logistic_score(elo1))
    return sum(
        count * math.log(probability1 / probability0)
        for count, probability0, probability1 in zip(
            regularized, hypothesis0, hypothesis1
        )
    )


def pentanomial_sprt(counts, elo0, elo1, alpha=0.05, beta=0.05):
    if not 0.0 < alpha < 1.0 or not 0.0 < beta < 1.0:
        raise ValueError("SPRT alpha and beta must lie strictly between zero and one")
    if elo0 >= elo1:
        raise ValueError("SPRT requires elo0 < elo1")

    lower_bound = math.log(beta / (1.0 - alpha))
    upper_bound = math.log((1.0 - beta) / alpha)
    llr = logistic_llr(counts, elo0, elo1)
    if llr <= lower_bound:
        state = "accept_h0"
    elif llr >= upper_bound:
        state = "accept_h1"
    else:
        state = "continue"
    return {
        "alpha": alpha,
        "beta": beta,
        "elo0": elo0,
        "elo1": elo1,
        "llr": llr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "state": state,
        "pentanomial": list(counts),
        "pairs": sum(counts),
    }
