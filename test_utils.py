from typing import Dict, List, Any

from dp_accounting.pld import privacy_loss_distribution

from pmf_utils import create_pld_and_extract_pmf
from pmf_compare import calc_W1_dist
from analytic_derivation import subsampled_gaussian_probabilities_from_losses
from subsample_pld_pmf import dp_accounting_pmf_to_loss_probs, loss_probs_to_dp_accounting_pmf, subsample_losses


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
    versions.append({
        'name': 'TF_subsampled',
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
        'name': 'Our_TF_subsampling',
        'pmf': our_TF_pmf,
        'losses': TF_original_losses,
        'probs': our_TF_subsampling_probs,
    })

    GT_original_losses, GT_original_probs = subsampled_gaussian_probabilities_from_losses(sigma=sigma, sampling_prob=1.0, discretization=discretization)
    our_GT_subsampling_probs = subsample_losses(GT_original_losses, GT_original_probs, sampling_prob, remove_direction, normalize_lower=True)
    our_GT_pmf = loss_probs_to_dp_accounting_pmf(
        GT_original_losses,
        our_GT_subsampling_probs,
        discretization,
        TF_original_pmf._pessimistic_estimate,
    )
    versions.append({
        'name': 'Our_GT_subsampling',
        'pmf': our_GT_pmf,
        'losses': GT_original_losses,
        'probs': our_GT_subsampling_probs,
    })

    GT_losses, GT_probs = subsampled_gaussian_probabilities_from_losses(sigma=sigma, sampling_prob=sampling_prob, discretization=discretization)
    GT_pmf = loss_probs_to_dp_accounting_pmf(GT_losses, GT_probs, discretization, TF_original_pmf._pessimistic_estimate)
    versions.append({
        'name': 'GT',
        'pmf': GT_pmf, 
        'losses': GT_losses,
        'probs': GT_probs,
    })

    for version in versions:
        print(f'{version["name"]}')
        version['eps'] = [version['pmf'].get_epsilon_for_delta(d) for d in delta_values]
        version['W1_vs_GT'] = calc_W1_dist(version['losses'], version['probs'], GT_losses, GT_probs)

    return versions