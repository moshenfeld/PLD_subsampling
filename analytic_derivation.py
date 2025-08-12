import numpy as np
from typing import Tuple
from scipy.stats import norm


def subsampled_gaussian_probabilities_from_losses(
    sigma: float,
    sampling_prob: float,
    losses: np.ndarray,
) -> np.ndarray:
    """
    Compute discrete probabilities over the provided loss grid for the
    subsampled Gaussian mechanism via an analytic CDF formula.

    For each loss value l, define the CDF:
        P(l) = [ q * exp(l) / (q + exp(l) - 1) ] * Phi( ( ln(1 + (exp((l - 0.5)/sigma^2) - 1)/q) - 1 ) / sigma )

    We then return the PMF on this grid using the forward differences:
        p[i] = P(losses[i]) - P(losses[i-1]), with P(losses[-1]) := 0.

    Args:
        sigma: Standard deviation of the Gaussian noise.
        sampling_prob: Subsampling probability q in (0, 1].
        losses: Sorted array of loss values (ascending) defining the grid.

    Returns:
        probs: Array of probabilities aligned with `losses`.

    Raises:
        ValueError: If inputs are outside expected domains or losses are not sorted.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if not (0 < sampling_prob <= 1):
        raise ValueError("sampling_prob (q) must be in (0, 1]")
    if losses.ndim != 1:
        raise ValueError("losses must be a 1D array")
    if not np.all(np.diff(losses) >= 0):
        raise ValueError("losses must be sorted in ascending order")

    q = float(sampling_prob)
    l = losses.astype(np.float64)

    # Prefactor: q * e^l / (q + e^l - 1)
    exp_l = np.exp(l)
    denom = q + exp_l - 1.0
    # Guard against division by zero; if denom == 0, set prefactor to 0.
    with np.errstate(divide='ignore', invalid='ignore'):
        prefactor = np.where(denom != 0.0, q * exp_l / denom, 0.0)

    # Inner Phi argument: ( ln(1 + (exp((l - 0.5)/sigma^2) - 1)/q) - 1 ) / sigma
    exp_term = np.exp((l - 0.5) / (sigma * sigma))
    # For very small exp_term, (exp_term - 1)/q < -1 and log1p is undefined.
    # In that region we set the CDF contribution to 0.
    arg = (exp_term - 1.0) / q
    valid = arg > -1.0
    inner_log = np.zeros_like(exp_term)
    inner_log[valid] = np.log1p(arg[valid])
    z = (inner_log - 1.0) / sigma
    # For invalid region, set z to -inf so Phi(z)=0
    z[~valid] = -np.inf
    cdf_vals = norm.cdf(z)

    # CDF on the grid
    P = prefactor * cdf_vals

    # Ensure CDF is non-decreasing and bounded in [0, 1]
    P = np.clip(P, 0.0, 1.0)
    P = np.maximum.accumulate(P)

    # PMF by forward differences with P[-1] = 0
    P_prev = np.concatenate(([0.0], P[:-1]))
    probs = P - P_prev

    # Clip small negatives from numerical issues and renormalize to sum <= 1
    probs = np.maximum(probs, 0.0)
    total = float(np.sum(probs, dtype=np.float64))
    if total > 1.0:
        probs = probs / total
    return probs


