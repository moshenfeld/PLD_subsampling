import numpy as np
from typing import Dict, List, Any

from ..wrappers.dp_accounting_wrappers import (
    create_pld_and_extract_pmf,
    dp_accounting_pmf_to_loss_probs,
    loss_probs_to_dp_accounting_pmf,
)
from .analytic_Gaussian import Gaussian_PLD
from ..PLD_subsampling_impl import subsample_losses


def run_multiple_experiments(discretizations, q_values, sigma_values, remove_directions, delta_values):
    results = []
    for discretization in discretizations:
        for sigma in sigma_values:
            for q in q_values:
                for remove_direction in remove_directions:
                    versions = run_experiment(sigma, q, discretization, delta_values, remove_direction)
                    results.append({
                        'sigma': sigma,
                        'q': q,
                        'discretization': discretization,
                        'remove_direction': bool(remove_direction),
                        'versions': versions,
                        'delta_values': np.asarray(delta_values, dtype=float),
                    })
    return results


def run_experiment(
    sigma: float,
    sampling_prob: float,
    discretization: float,
    delta_values: List[float],
    remove_direction: bool = True,
) -> Dict[str, Any]:
    versions: List[Dict[str, Any]] = []

    TF_subsampled_pmf = create_pld_and_extract_pmf(sigma, 1.0, sampling_prob, discretization, remove_direction)
    TF_subsampled_losses, TF_subsampled_probs = dp_accounting_pmf_to_loss_probs(TF_subsampled_pmf)
    versions.append({'name': 'TF_TF', 'pmf': TF_subsampled_pmf, 'losses': TF_subsampled_losses, 'probs': TF_subsampled_probs})

    TF_original_pmf = create_pld_and_extract_pmf(sigma, 1.0, 1.0, discretization, remove_direction)
    TF_original_losses, TF_original_probs = dp_accounting_pmf_to_loss_probs(TF_original_pmf)
    our_TF_subsampling_probs = subsample_losses(TF_original_losses, TF_original_probs, sampling_prob, remove_direction, normalize_lower=True)
    our_TF_pmf = loss_probs_to_dp_accounting_pmf(TF_original_losses, our_TF_subsampling_probs, discretization, TF_original_pmf._pessimistic_estimate)
    versions.append({'name': 'TF_Our', 'pmf': our_TF_pmf, 'losses': TF_original_losses, 'probs': our_TF_subsampling_probs})

    GT_original_losses, GT_original_probs = Gaussian_PLD(sigma=sigma, sampling_prob=1.0, discretization=discretization, remove_direction=remove_direction)
    our_GT_subsampling_probs = subsample_losses(GT_original_losses, GT_original_probs, sampling_prob, remove_direction, normalize_lower=True)
    our_GT_pmf = loss_probs_to_dp_accounting_pmf(GT_original_losses, our_GT_subsampling_probs, discretization, TF_original_pmf._pessimistic_estimate)
    versions.append({'name': 'GT_Our', 'pmf': our_GT_pmf, 'losses': GT_original_losses, 'probs': our_GT_subsampling_probs})

    GT_losses, GT_probs = Gaussian_PLD(sigma=sigma, sampling_prob=sampling_prob, discretization=discretization, remove_direction=remove_direction)
    GT_pmf = loss_probs_to_dp_accounting_pmf(GT_losses, GT_probs, discretization, TF_original_pmf._pessimistic_estimate)
    versions.append({'name': 'GT_GT', 'pmf': GT_pmf, 'losses': GT_losses, 'probs': GT_probs})

    for version in versions:
        version['eps'] = [version['pmf'].get_epsilon_for_delta(d) for d in delta_values]

    return versions


def calc_W1_dist(losses1, probs1, losses2, probs2) -> float:
    losses1 = np.asarray(losses1, dtype=np.float64)
    probs1 = np.asarray(probs1, dtype=np.float64)
    losses2 = np.asarray(losses2, dtype=np.float64)
    probs2 = np.asarray(probs2, dtype=np.float64)
    mask1 = probs1 > 0
    mask2 = probs2 > 0
    losses1 = losses1[mask1]
    probs1 = probs1[mask1]
    losses2 = losses2[mask2]
    probs2 = probs2[mask2]
    all_losses = np.unique(np.concatenate([losses1, losses2]))
    all_losses = np.sort(all_losses)
    finite_mask = np.isfinite(all_losses)
    finite_losses = all_losses[finite_mask]
    pmf1_dict = dict(zip(losses1, probs1))
    pmf2_dict = dict(zip(losses2, probs2))
    probs1_finite = np.array([pmf1_dict.get(l, 0.0) for l in finite_losses])
    probs2_finite = np.array([pmf2_dict.get(l, 0.0) for l in finite_losses])
    ccdf1 = 1.0 - np.cumsum(probs1_finite)
    ccdf2 = 1.0 - np.cumsum(probs2_finite)
    if finite_losses.size <= 1:
        return 0.0
    return float(np.sum(np.abs(ccdf1[:-1] - ccdf2[:-1]) * np.diff(finite_losses)))
