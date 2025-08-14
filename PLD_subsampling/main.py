#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

from .testing.test_utils import run_multiple_experiments
from .testing.plot_utils import create_pmf_cdf_plot, create_epsilon_delta_plot, print_experiment_table
from .testing.analytic_Gaussian import Gaussian_epsilon_for_delta


def main():
    # Parameters can be adjusted or wired to CLI later
    discretizations = [1e-4]
    q_values = [0.1, 0.9]
    sigma_values = [0.5, 2.0]
    remove_directions = [True, False]
    delta_values = np.array([10 ** (-k) for k in range(2, 13)], dtype=float)

    results = run_multiple_experiments(discretizations, q_values, sigma_values, remove_directions, delta_values)
    for res in results:
        sigma = res['sigma']; q = res['q']; discretization = res['discretization']; remove_direction = res['remove_direction']
        versions = res['versions']; deltas = res['delta_values']
        dir_tag = 'rem' if remove_direction else 'add'
        print(f"\nσ={sigma}, q={q}, disc={discretization:g}, dir={dir_tag}")
        eps_GT = [
            Gaussian_epsilon_for_delta(sigma=sigma, sampling_prob=q, delta=float(d), remove_direction=remove_direction)
            for d in deltas
        ]
        print_experiment_table(deltas, versions, eps_GT)

        fig_cdf = create_pmf_cdf_plot(versions=versions, title_suffix=f'sigma={sigma}, q={q}, disc={discretization:.0e}, dir={dir_tag}')
        os.makedirs('plots', exist_ok=True)
        fig_cdf.savefig(os.path.join('plots', f'cdf_sigma:{sigma}_q:{q}_d:{discretization:.0e}_dir:{dir_tag}.png'))
        plt.close(fig_cdf)

        fig_eps = create_epsilon_delta_plot(delta_values=deltas, versions=versions, eps_GT=eps_GT, log_x_axis=True, log_y_axis=False, title_suffix=f'sigma={sigma}, q={q}, disc={discretization:.0e}, dir:{dir_tag}')
        fig_eps.savefig(os.path.join('plots', f'epsilon_ratios:{sigma}_q:{q}_d:{discretization:.0e}_dir:{dir_tag}.png'))
        plt.close(fig_eps)


if __name__ == "__main__":
    main()


