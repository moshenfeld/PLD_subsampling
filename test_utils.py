import numpy as np
from typing import Dict, List, Tuple, Any
from scipy import stats

from dp_accounting.pld import privacy_loss_distribution

from pmf_utils import create_test_pmfs
from pmf_compare import calc_W1_dist

 

def run_experiment(
    sigma: float,
    sampling_prob: float,
    discretization: float,
    delta_values: List[float],
) -> Dict[str, Any]:
    """
    Build library reference PMF and our transformed PMF and return:
    - w1_distance between our PMF and the library PMF
    - epsilon arrays for analytical, reference (library), and ours for the provided deltas
    """
    from subsample_pld_pmf import subsample_pld_pmf

    # Create the PMFs on the requested grid
    test_pmfs = create_test_pmfs(
        sampling_prob=sampling_prob,
        sigma=sigma,
        sensitivity=1.0,
        discretization=discretization,
    )

    lib_pmf = test_pmfs['ground_truth_pmf']
    base_pmf = test_pmfs['unamplified_pmf']
    our_pmf = subsample_pld_pmf(base_pmf, sampling_prob)

    # W1 distance
    w1_distance = calc_W1_dist(our_pmf, lib_pmf)

    # Epsilon from delta arrays (analytical/ref/ours)
    eps_analytical, eps_ref, eps_ours = epsilon_from_deltas(
        sigma=sigma,
        q=sampling_prob,
        delta_values=delta_values,
        lib_pmf=lib_pmf,
        our_pmf=our_pmf,
        sensitivity=1.0,
    )

    return {
        'w1_distance': w1_distance,
        'epsilon_analytical': eps_analytical,
        'epsilon_ref': eps_ref,
        'epsilon_ours': eps_ours,
        'lib_pmf': lib_pmf,
        'our_pmf': our_pmf,
    }

def epsilon_from_deltas(
    sigma: float,
    q: float,
    delta_values: List[float],
    lib_pmf,
    our_pmf,
    sensitivity: float = 1.0,
) -> Tuple[List[float], List[float], List[float]]:
    """Compute epsilon arrays for analytical, library and ours given deltas."""
    lib_pld = privacy_loss_distribution.PrivacyLossDistribution(pmf_remove=lib_pmf)
    our_pld = privacy_loss_distribution.PrivacyLossDistribution(pmf_remove=our_pmf)
    eps_analytical: List[float] = []
    eps_ref: List[float] = []
    eps_ours: List[float] = []
    for delta in delta_values:
        eps_analytical.append(
            analytic_subsampled_epsilon_for_delta(sigma=sigma, q=q, delta=delta, sensitivity=sensitivity)
        )
        eps_ref.append(lib_pld.get_epsilon_for_delta(delta))
        eps_ours.append(our_pld.get_epsilon_for_delta(delta))
    return eps_analytical, eps_ref, eps_ours

# Analytical privacy calculation functions

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

 