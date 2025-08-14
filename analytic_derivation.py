import numpy as np
from scipy import stats
from typing import Tuple

"""Analytical PLD and epsilon/delta formulas for Gaussian mechanism with subsampling.

Provides: `Gaussian_PLD` on a uniform grid and `Gaussian_epsilon_for_delta` via binary search.
"""

from subsample_pld import stable_subsampling_loss

def Gaussian_PLD(
    sigma: float,
    sampling_prob: float,
    discretization: float,
    remove_direction: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct analytical PLD PMF on a uniform loss grid.

    Uses the stable subsampling loss mapping and Gaussian tails to build a PMF where
    probabilities are computed via forward differences of the CCDF.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if not (0 < sampling_prob <= 1):
        raise ValueError("sampling_prob (q) must be in (0, 1]")
    
    # Use symmetric range based on 50/sigma as requested
    l_max = np.ceil(20.0 / (sigma * discretization)) * discretization
    losses = np.arange(-l_max, l_max + discretization, discretization)

    transformed_losses = stable_subsampling_loss(losses, sampling_prob, remove_direction)
   
    # mapping of loss to normalized Gaussian value:
    #   for the upper distribution: sigma * l - 0.5/sigma
    #   for the lower distribution: sigma * l + 0.5/sigma
    x_upper = sigma * transformed_losses - 0.5/sigma
    x_lower = sigma * transformed_losses + 0.5/sigma
    
    # C-CDF (1 - CDF) of the loss
    # Initialize to 1 outside the valid support so CCDF defaults to 1
    S = np.ones_like(losses)
    if remove_direction:
        # S = (1-q) * S_lower + q * S_upper
        S = (1.0 - sampling_prob) * stats.norm.sf(x_lower) + sampling_prob * stats.norm.sf(x_upper)
    else:
        S = stats.norm.sf(x_upper)
    # PMF by forward tail differences with S[-1] = 1
    probs = np.concatenate(([1.0], S[:-1])) - S

    # Basic sanity checks
    if np.any(S < 0) or np.any(S > 1):
        raise ValueError("CCDF out of [0,1] in subsampled_gaussian_probabilities_from_losses")
    if np.any(probs < -1e-15):
        raise ValueError("Negative probability in subsampled_gaussian_probabilities_from_losses")
   # Check if any numerical issue arised such as sum(p) > 1
    if np.sum(probs) > 1 + 1e-12:
        raise ValueError("sum(probs) > 1 in subsampled_gaussian_probabilities_from_losses")
    if np.size(probs) != np.size(losses):
        raise ValueError("Length mismatch between losses and probs")
    return losses, probs

def Gaussian_delta_from_epsilon(sigma: float, sampling_prob: float, epsilon: float, remove_direction: bool) -> float:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if not (0 < sampling_prob <= 1):
        raise ValueError("sampling_prob (q) must be in (0, 1]")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    # In the non-amplified case: delta(eps) = Phi(1/(2 sigma) - eps * sigma) - e^eps * Phi(-1/(2 sigma) - eps * sigma)
    if sampling_prob == 1.0:
        return stats.norm.cdf(0.5/sigma - epsilon*sigma) - np.exp(epsilon) * stats.norm.cdf(-0.5/sigma - epsilon*sigma)
    # In the remove amplified case (mixture in the first argument): delta(eps) = q * delta(eps_base), where eps_base = log(1 + (e^eps - 1)/q)
    if remove_direction:
        amplified_epsilon = np.log(1.0 + (np.exp(epsilon) - 1.0)/sampling_prob)
        return sampling_prob * Gaussian_delta_from_epsilon(sigma, 1.0, amplified_epsilon, True)
    # In the add amplified case (mixture in the second argument):
    # if epsilon is too large, the delta is 0
    if epsilon >= -np.log(1-sampling_prob):
        return 0.0
    # else: delta(eps) = (1 - e^eps * (1 - q)) * delta(eps_base), where eps_base = -log(1 + (e^eps - 1)/q)
    amplified_epsilon = -np.log(1.0 + (np.exp(-epsilon) - 1.0)/sampling_prob)
    return (1.0 - np.exp(epsilon) * (1.0 - sampling_prob)) * Gaussian_delta_from_epsilon(sigma, 1.0, amplified_epsilon, False)

def Gaussian_epsilon_for_delta(sigma: float, sampling_prob: float, delta: float, remove_direction: bool) -> float:
    """Compute epsilon for the Gaussian mechanism given delta via binary search."""
    def delta_for_eps(eps: float) -> float:
        return Gaussian_delta_from_epsilon(sigma, sampling_prob, eps, remove_direction)
    eps_low = 0.0
    eps_high = 100.0
    if delta_for_eps(eps_high) > delta:
        return float('inf')
    while eps_high - eps_low > 1e-6:
        eps_mid = (eps_low + eps_high) / 2.0
        if delta_for_eps(eps_mid) <= delta:
            eps_high = eps_mid
        else:
            eps_low = eps_mid
    return eps_high