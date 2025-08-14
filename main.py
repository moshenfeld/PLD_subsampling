#!/usr/bin/env python3
"""
Test the simple formula implementation with different parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from test_utils import run_experiment
from plot_utils import create_pmf_cdf_plot, create_epsilon_delta_plot

def run_all_experiments():
    print("\n\n🔬 EXPERIMENTS: W1 + epsilon(δ)")
    print("=" * 70)
    discretizations = [1e-4]
    q_values = [0.1, 0.5, 0.9]
    sigma_values = [0.5, 1.0, 2.0]
    # Use deltas 1e-1 .. 1e-10
    delta_values = np.array([10 ** (-k) for k in range(1, 11)], dtype=float)

    for discretization in discretizations:
        for sigma in sigma_values:
            for q in q_values:
                versions = run_experiment(sigma, q, discretization, delta_values)
                print(f"\nσ={sigma}, q={q}, disc={discretization:g}")
                # Dynamic table header (versions already include any GT entry)
                headers = ['Delta'] + [v['name'] for v in versions]
                col_fmt = "{:<8} " + "{:>15} " * (len(headers) - 1)
                print(col_fmt.format(*headers))
                print("-" * (10 + 16 * (len(headers) - 1)))
                # Build rows
                eps_arrays = [np.array(v['eps']) for v in versions]
                for row_vals in zip([f"{d:0.0e}" for d in delta_values], *eps_arrays):
                    # First item is delta string, others are floats
                    delta_str = row_vals[0]
                    vals = [f"{x:15.6f}" for x in row_vals[1:]]
                    print(f"{delta_str:<8} " + " ".join(vals))

                # Create and save CDF plot using ground truth on union losses
                fig_cdf = create_pmf_cdf_plot(
                    versions=versions,
                    title_suffix=f'sigma={sigma}, q={q}, disc={discretization:.0e}',
                )
                fig_cdf.savefig(os.path.join('plots', f'pmf_cdf_sigma{sigma}_q{q}_d{discretization:.0e}.png'))
                plt.close(fig_cdf)

                # Create and save epsilon-vs-delta plot
                fig_eps = create_epsilon_delta_plot(
                    delta_values=delta_values,
                    versions=versions,
                    sigma=sigma,
                    q=q,
                    log_x_axis=False,
                    log_y_axis=True,
                    title_suffix=f'sigma={sigma}, q={q}, disc={discretization:.0e}'
                )
                fig_eps.savefig(os.path.join('plots', f'epsilon_vs_delta_sigma{sigma}_q{q}_d{discretization:.0e}.png'))
                plt.close(fig_eps)

if __name__ == "__main__":
    run_all_experiments()