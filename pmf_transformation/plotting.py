from typing import List
import numpy as np
import matplotlib.pyplot as plt


def create_pmf_cdf_plot(versions: List[dict], title_suffix: str = ''):
    from ..plot_utils import create_pmf_cdf_plot as _impl
    return _impl(versions=versions, title_suffix=title_suffix)


def create_epsilon_delta_plot(delta_values, versions, eps_GT, log_x_axis: bool, log_y_axis: bool, title_suffix: str):
    from ..plot_utils import create_epsilon_delta_plot as _impl
    return _impl(delta_values, versions, eps_GT, log_x_axis, log_y_axis, title_suffix)


__all__ = [
    'create_pmf_cdf_plot',
    'create_epsilon_delta_plot',
]


