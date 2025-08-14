"""Experiment utilities for constructing PMFs and evaluating metrics.

This module assembles several PMF variants and computes metrics/epsilons for use
in the main experiment runner.
"""

from typing import Dict, List, Any

from dp_accounting_wrappers import create_pld_and_extract_pmf
import numpy as np
from analytic_derivation import Gaussian_PLD
from subsample_pld import dp_accounting_pmf_to_loss_probs, loss_probs_to_dp_accounting_pmf, subsample_losses


def run_experiment(
    sigma: float,
    sampling_prob: float,
    discretization: float,
    delta_values: List[float],
    remove_direction: bool = True,
) -> Dict[str, Any]:
    """Build PMF variants and compute W1 and epsilon(δ) across methods.

    Returns a list of version dictionaries with keys:
      - name: label identifying the method variant
      - pmf: dp-accounting PMF object
      - losses, probs: finite support representation for plotting/comparison
      - eps: list of epsilon values per delta in `delta_values`
      - W1_vs_GT: Wasserstein-1 distance against analytical ground truth
    """
    versions: List[Dict[str, Any]] = []

    TF_subsampled_pmf = create_pld_and_extract_pmf(sigma, 1.0, sampling_prob, discretization, remove_direction)
    TF_subsampled_losses, TF_subsampled_probs = dp_accounting_pmf_to_loss_probs(TF_subsampled_pmf)
    versions.append({
        'name': 'TF_TF',
        'pmf': TF_subsampled_pmf,
        'losses': TF_subsampled_losses,
        'probs': TF_subsampled_probs,
    })

    TF_original_pmf = create_pld_and_extract_pmf(sigma, 1.0, 1.0, discretization, remove_direction)
    TF_original_losses, TF_original_probs = dp_accounting_pmf_to_loss_probs(TF_original_pmf)
    our_TF_subsampling_probs = subsample_losses(TF_original_losses, TF_original_probs, sampling_prob, remove_direction, normalize_lower=True)
    our_TF_pmf = loss_probs_to_dp_accounting_pmf(
        TF_original_losses,
        our_TF_subsampling_probs,
        discretization,
        TF_original_pmf._pessimistic_estimate,
    )
    versions.append({
        'name': 'TF_Our',
        'pmf': our_TF_pmf,
        'losses': TF_original_losses,
        'probs': our_TF_subsampling_probs,
    })

    GT_original_losses, GT_original_probs = Gaussian_PLD(
        sigma=sigma, sampling_prob=1.0, discretization=discretization, remove_direction=remove_direction
    )
    our_GT_subsampling_probs = subsample_losses(GT_original_losses, GT_original_probs, sampling_prob, remove_direction, normalize_lower=True)
    our_GT_pmf = loss_probs_to_dp_accounting_pmf(
        GT_original_losses,
        our_GT_subsampling_probs,
        discretization,
        TF_original_pmf._pessimistic_estimate,
    )
    versions.append({
        'name': 'GT_Our',
        'pmf': our_GT_pmf,
        'losses': GT_original_losses,
        'probs': our_GT_subsampling_probs,
    })

    GT_losses, GT_probs = Gaussian_PLD(
        sigma=sigma, sampling_prob=sampling_prob, discretization=discretization, remove_direction=remove_direction
    )
    GT_pmf = loss_probs_to_dp_accounting_pmf(GT_losses, GT_probs, discretization, TF_original_pmf._pessimistic_estimate)
    versions.append({
        'name': 'GT_GT',
        'pmf': GT_pmf, 
        'losses': GT_losses,
        'probs': GT_probs,
    })

    for version in versions:
        version['eps'] = [version['pmf'].get_epsilon_for_delta(d) for d in delta_values]
        version['W1_vs_GT'] = calc_W1_dist(version['losses'], version['probs'], GT_losses, GT_probs)

    return versions


def calc_W1_dist(losses1, probs1, losses2, probs2) -> float:
    """Compute Wasserstein-1 distance between two discrete distributions.

    - Each pair `(lossesX, probsX)` must have matching lengths.
    - The two pairs may have different lengths and different supports.
    - Losses may include +inf to represent infinity mass; if those masses differ, returns inf.
    """

    losses1 = np.asarray(losses1, dtype=np.float64)
    probs1 = np.asarray(probs1, dtype=np.float64)
    losses2 = np.asarray(losses2, dtype=np.float64)
    probs2 = np.asarray(probs2, dtype=np.float64)

    if losses1.shape[0] != probs1.shape[0]:
        raise ValueError(f'Length mismatch between losses1 and probs1: {losses1.shape[0]} != {probs1.shape[0]}')
    if losses2.shape[0] != probs2.shape[0]:
        raise ValueError(f'Length mismatch between losses2 and probs2: {losses2.shape[0]} != {probs2.shape[0]}')

    # Remove zero-mass entries to simplify alignment
    mask1 = probs1 > 0
    mask2 = probs2 > 0
    losses1 = losses1[mask1]
    probs1 = probs1[mask1]
    losses2 = losses2[mask2]
    probs2 = probs2[mask2]

    # Union grid over finite supports
    all_losses = np.unique(np.concatenate([losses1, losses2]))
    all_losses = np.sort(all_losses)
    finite_mask = np.isfinite(all_losses)
    finite_losses = all_losses[finite_mask]

    # Align probabilities on the union grid
    pmf1_dict = dict(zip(losses1, probs1))
    pmf2_dict = dict(zip(losses2, probs2))
    probs1_finite = np.array([pmf1_dict.get(l, 0.0) for l in finite_losses])
    probs2_finite = np.array([pmf2_dict.get(l, 0.0) for l in finite_losses])

    # Work with complementary CDFs (CCDF = 1 - CDF)
    ccdf1 = 1.0 - np.cumsum(probs1_finite)
    ccdf2 = 1.0 - np.cumsum(probs2_finite)

    # Explicit +inf mass check
    p1_inf = probs1[np.isposinf(losses1)].sum() if np.any(np.isposinf(losses1)) else 0.0
    p2_inf = probs2[np.isposinf(losses2)].sum() if np.any(np.isposinf(losses2)) else 0.0
    if abs(p1_inf - p2_inf) > 0:
        return float('inf')

    if finite_losses.size <= 1:
        return 0.0

    return float(np.sum(np.abs(ccdf1[:-1] - ccdf2[:-1]) * np.diff(finite_losses)))