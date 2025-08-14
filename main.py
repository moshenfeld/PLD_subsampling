#!/usr/bin/env python3
"""
Entry point to run subsampling experiments and generate figures.

Builds multiple PMF variants (library-based and analytical), computes
Wasserstein-1 distances and epsilon(δ), and saves CDF and ratio plots
under `plots/`.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from test_utils import run_experiment
from plot_utils import create_pmf_cdf_plot, create_epsilon_delta_plot
from analytic_derivation import Gaussian_epsilon_for_delta
from dp_accounting.pld import privacy_loss_distribution
from dp_accounting_wrappers import amplify_pld_separate_directions

def run_all_experiments():
    discretizations = [1e-4]
    q_values = [0.1, 0.9]
    sigma_values = [0.5, 2.0]
    remove_directions = [True, False]
    # Use deltas 1e-1 .. 1e-10
    delta_values = np.array([10 ** (-k) for k in np.linspace(2, 12, 20)], dtype=float)

    # One-off amplification test (separate from plotting loop)
    test_sigma = 0.5
    test_q = 0.1
    test_disc = 1e-4
    base_pld = privacy_loss_distribution.from_gaussian_mechanism(
        standard_deviation=test_sigma,
        sensitivity=1.0,
        value_discretization_interval=test_disc,
        pessimistic_estimate=True,
    )
    amplified_pld = amplify_pld_separate_directions(base_pld, sampling_prob=test_q)
    subsampled_pld = privacy_loss_distribution.from_gaussian_mechanism(
        standard_deviation=test_sigma,
        sensitivity=1.0,
        value_discretization_interval=test_disc,
        sampling_prob=test_q,
        pessimistic_estimate=True,
    )
    #Compare the epsilon(delta) of the amplified PLD with a subsampled PLD
    subsampled_pld_epsilons = [subsampled_pld.get_epsilon_for_delta(delta) for delta in delta_values]
    amplified_pld_epsilons = [amplified_pld["pmf_remove"].get_epsilon_for_delta(delta) for delta in delta_values]
    print(f"One-off amplification test: sigma={test_sigma}, q={test_q}, disc={test_disc:.0e}")
    for delta, subsampled_epsilon, amplified_epsilon in zip(delta_values, subsampled_pld_epsilons, amplified_pld_epsilons):
        print(f"  delta={delta:.0e}, subsampled_epsilon={subsampled_epsilon:.6f}, amplified_epsilon={amplified_epsilon:.6f}")
    
    print("\n\n🔬 EXPERIMENTS: W1 + epsilon(δ)")
    print("=" * 70)
    for discretization in discretizations:
        for sigma in sigma_values:
            for q in q_values:
                for remove_direction in remove_directions:
                    dir_tag = 'rem' if remove_direction else 'add'
                    versions = run_experiment(sigma, q, discretization, delta_values, remove_direction=remove_direction)
                    print(f"\nσ={sigma}, q={q}, disc={discretization:g}, dir={dir_tag}")
                    # Compute GT epsilons analytically (direction-aware)
                    eps_GT = [
                        Gaussian_epsilon_for_delta(sigma=sigma, sampling_prob=q, delta=float(d), remove_direction=remove_direction)
                        for d in delta_values
                    ]

                    # Table header with GT included
                    headers = ['Delta'] + [v['name'] for v in versions] + ['GT']
                    col_fmt = "{:<8} " + "{:>15} " * (len(headers) - 1)
                    print(col_fmt.format(*headers))
                    print("-" * (10 + 16 * (len(headers) - 1)))
                    # Build rows (versions + GT)
                    eps_arrays = [np.array(v['eps']) for v in versions] + [np.array(eps_GT)]
                    for row_vals in zip([f"{d:0.0e}" for d in delta_values], *eps_arrays):
                        delta_str = row_vals[0]
                        vals = [f"{x:15.6f}" for x in row_vals[1:]]
                        print(f"{delta_str:<8} " + " ".join(vals))
                    # Create and save CDF plot using ground truth on union losses
                    fig_cdf = create_pmf_cdf_plot(
                        versions=versions,
                        title_suffix=f'sigma={sigma}, q={q}, disc={discretization:.0e}, dir={dir_tag}',
                    )
                    fig_cdf.savefig(os.path.join('plots', f'cdf_sigma:{sigma}_q:{q}_d:{discretization:.0e}_dir:{dir_tag}.png'))
                    plt.close(fig_cdf)

                    # Create and save epsilon-vs-delta plot
                    fig_eps = create_epsilon_delta_plot(
                        delta_values=delta_values,
                        versions=versions,
                        eps_GT=eps_GT,  
                        log_x_axis=True,
                        log_y_axis=False,
                        title_suffix=f'sigma={sigma}, q={q}, disc={discretization:.0e}, dir={dir_tag}',
                    )
                    fig_eps.savefig(os.path.join('plots', f'epsilon_ratios:{sigma}_q:{q}_d:{discretization:.0e}_dir:{dir_tag}.png'))
                    plt.close(fig_eps)

if __name__ == "__main__":
    run_all_experiments()