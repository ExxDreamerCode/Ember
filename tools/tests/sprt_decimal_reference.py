"""Test-only primal likelihood oracle, independent of the float dual expression.

Fit and normalize each constrained probability distribution with Decimal,
then sum weighted log probability ratios. No production fitting helper or
log1p likelihood expression is used. Elo inputs are exact decimal strings.
"""

from decimal import Decimal, localcontext


def decimal_llr(counts, elo0, elo1, precision=80):
    if precision < 60:
        raise ValueError("the reference requires at least 60 digits")
    with localcontext() as context:
        context.prec = precision
        weights = [Decimal(str(count)) if count else Decimal("0.001") for count in counts]
        total = sum(weights)
        frequencies = [weight / total for weight in weights]
        outcomes = [Decimal(index) / 4 for index in range(5)]
        fits = []
        for elo in (elo0, elo1):
            expectation = 1 / (1 + Decimal(10) ** (-Decimal(elo) / 400))
            offsets = [outcome - expectation for outcome in outcomes]
            margin = Decimal(10) ** (10 - precision)
            low = -1 / max(offsets) + margin
            high = -1 / min(offsets) - margin
            for _ in range(4 * precision):
                midpoint = (low + high) / 2
                if midpoint in (low, high):
                    break
                residual = sum(
                    frequency * offset / (1 + midpoint * offset)
                    for frequency, offset in zip(frequencies, offsets)
                )
                if residual > 0:
                    low = midpoint
                else:
                    high = midpoint
            multiplier = (low + high) / 2
            fit = [
                frequency / (1 + multiplier * offset)
                for frequency, offset in zip(frequencies, offsets)
            ]
            normalization = sum(fit)
            fits.append([probability / normalization for probability in fit])
        return sum(
            weight * (probability1 / probability0).ln()
            for weight, probability0, probability1 in zip(weights, *fits)
        )
