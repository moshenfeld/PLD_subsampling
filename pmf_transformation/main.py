#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

from .testing.test_utils import run_all_experiments
from .testing.plot_utils import create_pmf_cdf_plot, create_epsilon_delta_plot
from .analytic_Gaussian import Gaussian_epsilon_for_delta


def main():
    # Parameters can be adjusted or wired to CLI later
    discretizations = [1e-4]
    q_values = [0.1, 0.9]
    sigma_values = [0.5, 2.0]
    remove_directions = [True, False]
    delta_values = np.array([10 ** (-k) for k in range(2, 13)], dtype=float)

    for sigma, q, discretization, dir_tag, versions in run_all_experiments(
        discretizations, q_values, sigma_values, remove_directions, delta_values
    ):
        print(f"\nσ={sigma}, q={q}, disc={discretization:g}, dir={dir_tag}")

        eps_GT = [
            Gaussian_epsilon_for_delta(sigma=sigma, sampling_prob=q, delta=float(d), remove_direction=(dir_tag=='rem'))
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


if __name__ == "__main__":
    main()


