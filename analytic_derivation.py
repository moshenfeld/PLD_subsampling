import numpy as np
from scipy import stats
from typing import Tuple

from subsample_pld_pmf import stable_subsampling_loss

def subsampled_gaussian_probabilities_from_losses(
    sigma: float,
    sampling_prob: float,
    discretization: float,
    remove_direction: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if not (0 < sampling_prob <= 1):
        raise ValueError("sampling_prob (q) must be in (0, 1]")
    
    sigma_rounded = np.round(sigma / discretization) * discretization
    # Use symmetric range based on 50/sigma as requested
    l_bound = 50.0 / sigma
    l_max = np.ceil(l_bound / discretization) * discretization
    losses = np.arange(-l_max, l_max + discretization, discretization)

    # Only losses for which l > ln(1-sampling_prob) might have non-zero probability
    valid = losses > np.log(1 - sampling_prob) if sampling_prob < 1.0 else np.ones_like(losses, dtype=bool)
    transformed_losses = stable_subsampling_loss(losses[valid], sampling_prob, remove_direction)
   
    # mapping of loss to normalized Gaussian value:
    #   for the upper distribution: sigma * l - 0.5/sigma
    #   for the lower distribution: sigma * l + 0.5/sigma
    x_upper = sigma * transformed_losses - 0.5/sigma
    x_lower = sigma * transformed_losses + 0.5/sigma
    
    # P_upper = stats.norm.cdf(x_upper)
    # p_upper = P_upper - np.concatenate(([0.0], P_upper[:-1]))
    # P_lower = stats.norm.cdf(x_lower)
    # p_lower = P_lower - np.concatenate(([0.0], P_lower[:-1]))
    # p_upper_from_lower = np.exp(np.log(p_lower) + transformed_losses)
    # p_lower_from_upper = np.exp(np.log(p_upper) - transformed_losses)
    # P_upper_from_lower = np.cumsum(p_upper_from_lower)
    # P_lower_from_upper = np.cumsum(p_lower_from_upper)
    # print(f'sum(p_upper) = {np.sum(p_upper)}, sum(p_lower) = {np.sum(p_lower)}, sum(p_upper_from_lower) = {np.sum(p_upper_from_lower)}, sum(p_lower_from_upper) = {np.sum(p_lower_from_upper)}')


    # CDF of the loss
    P = np.zeros_like(losses)
    if remove_direction:
        P[valid] = stats.norm.cdf(x_lower) + sampling_prob * (stats.norm.cdf(x_upper) - stats.norm.cdf(x_lower))
    else:
        P[valid] = stats.norm.cdf(x_upper)

    # PMF by forward differences with P[-1] = 0
    probs = P - np.concatenate(([0.0], P[:-1]))

    # Check if any numerical issue arised such as P>1, p < 0, or sum(p) > 1
    if np.any(P > 1):
        raise ValueError("P > 1 in subsampled_gaussian_probabilities_from_losses")
    if np.any(probs < 0):
        raise ValueError("P < 0 in subsampled_gaussian_probabilities_from_losses")
    if np.sum(probs) > 1:
        raise ValueError("sum(P) > 1 in subsampled_gaussian_probabilities_from_losses")
    if np.size(probs) != np.size(losses):
        raise ValueError("Length mismatch between losses and probs")
    return losses, probs

def compute_delta(epsilon: float, noise_multiplier: float) -> float:
    """
    Compute exact delta for the Gaussian mechanism given epsilon and noise multiplier.
    
    This implementation matches the dp-accounting library's implementation.
    
    Args:
        epsilon: The privacy parameter epsilon
        noise_multiplier: The noise multiplier (sigma/sensitivity)
        
    Returns:
        The corresponding delta value
    """
    # The standard deviation is noise_multiplier * sensitivity, but we've already
    # incorporated sensitivity into noise_multiplier
    if noise_multiplier == 0:
        return 0 if epsilon == float('inf') else 1
    
    # The exact formula for delta
    return stats.norm.cdf(0.5/noise_multiplier - epsilon*noise_multiplier) - \
           np.exp(epsilon) * stats.norm.cdf(-0.5/noise_multiplier - epsilon*noise_multiplier)

def gaussian_epsilon_for_delta(sigma: float, sensitivity: float, delta: float) -> float:
    """Compute epsilon for the Gaussian mechanism given delta via binary search."""
    noise_multiplier = sigma / sensitivity
    def delta_for_eps(eps: float) -> float:
        return compute_delta(eps, noise_multiplier)
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

def analytic_subsampled_epsilon_for_delta(
    sigma: float,
    q: float,
    delta: float,
    sensitivity: float = 1.0,
) -> float:
    """Analytical ε(δ) for subsampled Gaussian: ε' = log(1 + q (exp(ε_orig(δ/q)) - 1))."""
    if q <= 0.0 or q > 1.0:
        raise ValueError("Sampling probability must be in (0, 1]")
    if q == 1.0:
        return gaussian_epsilon_for_delta(sigma=sigma, sensitivity=sensitivity, delta=delta)
    adjusted_delta = min(delta / q, 1.0)
    eps_orig = gaussian_epsilon_for_delta(sigma=sigma, sensitivity=sensitivity, delta=adjusted_delta)
    return float(np.log(1.0 + q * (np.exp(eps_orig) - 1.0)))