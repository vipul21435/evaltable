"""Interval estimates for a proportion."""

from __future__ import annotations

import math
from statistics import NormalDist


def wilson_interval(
    successes: float, n: float, confidence: float = 0.95
) -> tuple[float, float] | None:
    """Wilson score interval for a binomial proportion.

    ``successes`` may be fractional (a sum of scores in [0, 1]); the interval is then the
    same formula applied to the mean, which is an approximation. Returns ``None`` for ``n == 0``.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n < 0 or successes < 0 or successes > n:
        raise ValueError(f"need 0 <= successes <= n, got successes={successes}, n={n}")
    if n == 0:
        return None
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    p = successes / n
    z2n = z * z / n
    centre = (p + z2n / 2.0) / (1.0 + z2n)
    half = z / (1.0 + z2n) * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))
