"""## Executive summary (read this first)

Rebuild the organizer's documented outcome-free M0 normalizers for research.
This implements only M0-BASELINE section 5 divisors, not any scoring formula.
The organizer's scorer computes all actual forecast losses.
"""

import numpy as np
from scipy.special import gamma, hyp1f1, ndtr, ndtri


def normal_scales(mean, covariance, levels=(0.01, 0.05, 0.95, 0.99)):
    sd = np.sqrt(np.diag(covariance))
    delta = mean[:, None] - mean[None, :]
    variance = np.maximum(
        np.diag(covariance)[:, None] + np.diag(covariance)[None, :] - 2 * covariance, 0.0
    )
    mask = ~np.eye(len(mean), dtype=bool)
    d, s = delta[mask], np.sqrt(variance[mask])
    positive = s > 0
    d, s = d[positive], s[positive]
    z = d / s
    absolute = s * np.sqrt(2 / np.pi) * np.exp(-0.5 * z * z) + d * (2 * ndtr(z) - 1)
    root_abs = (
        np.sqrt(s) * 2**0.25 * gamma(0.75) / np.sqrt(np.pi) * hyp1f1(-0.25, 0.5, -0.5 * z * z)
    )
    joint = float(np.maximum(absolute - root_abs**2, 0).sum())
    out = dict(
        marginal=float(sd.mean() / np.sqrt(np.pi)),
        joint=joint,
        tail=float(sd.mean() * np.exp(-0.5 * ndtri(levels) ** 2).mean() / np.sqrt(2 * np.pi)),
    )
    return {k: v if v > 0 else 1.0 for k, v in out.items()}


def distribution(changes, anchor, horizons):
    x = np.asarray(changes[-299:], float)
    a, h = np.indices((len(anchor), len(horizons)))
    a, h = a.ravel(), np.asarray(horizons)[h.ravel()]
    mean = anchor[a] + x.mean(0)[a] * h
    cov = np.atleast_2d(np.cov(x, rowvar=False))
    c = cov[np.ix_(a, a)] * np.minimum.outer(h, h) + np.eye(len(h)) * 1.1e-9
    try:
        np.linalg.cholesky(c)
    except np.linalg.LinAlgError:
        c = np.diag(np.diag(c))
    return mean, c
