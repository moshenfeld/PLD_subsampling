#!/usr/bin/env python3
"""
Test the simple formula implementation with different parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from test_utils import run_experiment
from plot_utils import ensure_plots_dir, create_pmf_cdf_plot, create_epsilon_delta_plot
from subsample_pld_pmf import dp_accounting_pmf_to_loss_probs

def run_all_experiments():
    print("\n\n🔬 EXPERIMENTS: W1 + epsilon(δ)")
    print("=" * 70)
    # discretizations = [1e-3, 1e-4, 1e-5]
    # q_values = [0.1, 0.5, 0.9]
    # sigma_values = [0.5, 1.0, 2.0]
    discretizations = [1e-4]
    q_values = [0.1, 0.5, 0.9]
    sigma_values = [0.5, 1.0, 2.0]
    # delta array: 1e-[1..10]
    delta_values = [10 ** (-k) for k in range(1, 11)]

    for discretization in discretizations:
        for sigma in sigma_values:
            for q in q_values:
                result = run_experiment(sigma, q, discretization, delta_values)
                w1 = result['w1_distance']
                eps_a = result['epsilon_analytical']
                eps_r = result['epsilon_ref']
                eps_o = result['epsilon_ours']
                lib_pmf = result['lib_pmf']
                our_pmf = result['our_pmf']
                print(f"\nσ={sigma}, q={q}, disc={discretization:g}, W1={w1:.6g}")
                print(f"{'Delta':<8} {'Analytical':<15} {'Ref':<15} {'Ours':<15}")
                print("-" * 60)
                for d, ea, er, eo in zip(delta_values, eps_a, eps_r, eps_o):
                    print(f"{d:<8.0e} {ea:<15.6f} {er:<15.6f} {eo:<15.6f}")

                our_losses, our_probs = dp_accounting_pmf_to_loss_probs(our_pmf)
                lib_losses, lib_probs = dp_accounting_pmf_to_loss_probs(lib_pmf)
                # print(np.min(our_losses), np.max(our_losses), np.size(our_losses))
                # print(np.min(lib_losses), np.max(lib_losses), np.size(lib_losses))

                # plt.plot(our_losses, our_probs, 'b-', label='Our PMF', alpha=0.8)
                # plt.plot(lib_losses, lib_probs, 'r--', label='Library PMF', alpha=0.8)
                # plt.legend()
                # plt.savefig(os.path.join('plots', f'pmf_sigma{sigma}_q{q}_d{discretization:e}.png'))
                # plt.close()

                # Generate plots per configuration
                ensure_plots_dir('plots')
                # Create and save CDF plot
                fig_cdf = create_pmf_cdf_plot(our_pmf=our_pmf, library_pmf=lib_pmf, w1=w1, title_suffix=f'sigma={sigma}, q={q}, disc={discretization:e}')
                fig_cdf.savefig(os.path.join('plots', f'pmf_cdf_sigma{sigma}_q{q}_d{discretization:e}.png'))
                plt.close(fig_cdf)
                # Create and save epsilon-vs-delta plot
                fig_eps = create_epsilon_delta_plot(delta_values=delta_values, eps_a=eps_a, eps_r=eps_r, eps_o=eps_o, log_x_axis=True, log_y_axis=True, title_suffix=f'sigma={sigma}, q={q}, disc={discretization:g}')
                fig_eps.savefig(os.path.join('plots', f'epsilon_vs_delta_sigma{sigma}_q{q}_d{discretization:e}.png'))
                plt.close(fig_eps)

if __name__ == "__main__":
    run_all_experiments()