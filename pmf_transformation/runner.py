#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

from .testing import calc_W1_dist
from .wrappers import create_pld_and_extract_pmf, dp_accounting_pmf_to_loss_probs, loss_probs_to_dp_accounting_pmf, amplify_pld_separate_directions
from .analytics import Gaussian_PLD, Gaussian_epsilon_for_delta
from .transforms import subsample_losses
from .plotting import create_pmf_cdf_plot, create_epsilon_delta_plot


def run_all_experiments():
    print("\n\n🔬 EXPERIMENTS: W1 + epsilon(δ)")
    print("=" * 70)

    # One-off amplification test
    test_sigma = 0.5
    test_q = 0.1
    test_disc = 1e-4
    from dp_accounting.pld import privacy_loss_distribution
    base_pld = privacy_loss_distribution.from_gaussian_mechanism(
        standard_deviation=test_sigma,
        sensitivity=1.0,
        value_discretization_interval=test_disc,
        pessimistic_estimate=True,
    )
    amplified = amplify_pld_separate_directions(base_pld, sampling_prob=test_q)
    for d in [1e-2, 3e-3, 9e-4, 3e-4, 8e-5, 2e-5, 7e-6, 2e-6, 6e-7, 2e-7, 5e-8, 2e-8, 5e-9, 1e-9, 4e-10, 1e-10, 4e-11, 1e-11, 3e-12, 1e-12]:
        print(f"  delta={d:.0e}, subsampled_epsilon={amplified['pmf_remove'].get_epsilon_for_delta(d):.6f}, amplified_epsilon={amplified['pmf_remove'].get_epsilon_for_delta(d):.6f}")

    discretizations = [1e-4]
    q_values = [0.1, 0.9]
    sigma_values = [0.5, 2.0]
    remove_directions = [True, False]
    delta_values = np.array([10 ** (-k) for k in range(2, 13)], dtype=float)

    for discretization in discretizations:
        for sigma in sigma_values:
            for q in q_values:
                for remove_direction in remove_directions:
                    dir_tag = 'rem' if remove_direction else 'add'
                    versions = _run_experiment(sigma, q, discretization, delta_values, remove_direction)
                    print(f"\nσ={sigma}, q={q}, disc={discretization:g}, dir={dir_tag}")

                    eps_GT = [
                        Gaussian_epsilon_for_delta(sigma=sigma, sampling_prob=q, delta=float(d), remove_direction=remove_direction)
                        for d in delta_values
                    ]

                    headers = ['Delta'] + [v['name'] for v in versions] + ['GT']
                    col_fmt = "{:<8} " + "{:>15} " * (len(headers) - 1)
                    print(col_fmt.format(*headers))
                    print("-" * (10 + 16 * (len(headers) - 1)))
                    eps_arrays = [np.array(v['eps']) for v in versions] + [np.array(eps_GT)]
                    for row_vals in zip([f"{d:0.0e}" for d in delta_values], *eps_arrays):
                        delta_str = row_vals[0]
                        vals = [f"{x:15.6f}" for x in row_vals[1:]]
                        print(f"{delta_str:<8} " + " ".join(vals))

                    fig_cdf = create_pmf_cdf_plot(
                        versions=versions,
                        title_suffix=f'sigma={sigma}, q={q}, disc={discretization:.0e}, dir={dir_tag}',
                    )
                    os.makedirs('plots', exist_ok=True)
                    fig_cdf.savefig(os.path.join('plots', f'cdf_sigma:{sigma}_q:{q}_d:{discretization:.0e}_dir:{dir_tag}.png'))
                    plt.close(fig_cdf)

                    fig_eps = create_epsilon_delta_plot(
                        delta_values=delta_values,
                        versions=versions,
                        eps_GT=eps_GT,
                        log_x_axis=True,
                        log_y_axis=False,
                        title_suffix=f'sigma={sigma}, q={q}, disc={discretization:.0e}, dir:{dir_tag}',
                    )
                    fig_eps.savefig(os.path.join('plots', f'epsilon_ratios:{sigma}_q:{q}_d:{discretization:.0e}_dir:{dir_tag}.png'))
                    plt.close(fig_eps)


def _run_experiment(sigma: float, sampling_prob: float, discretization: float, delta_values, remove_direction: bool):
    versions = []

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
        version['eps'] = [version['pmf'].get_epsilon_for_delta(float(d)) for d in delta_values]
        version['W1_vs_GT'] = calc_W1_dist(version['losses'], version['probs'], GT_losses, GT_probs)

    return versions


if __name__ == "__main__":
    run_all_experiments()


