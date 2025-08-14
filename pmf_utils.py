import numpy as np
from typing import Dict
from dp_accounting.pld import privacy_loss_distribution


def create_pld_and_extract_pmf(
    standard_deviation: float,
    sensitivity: float,
    sampling_prob: float,
    value_discretization_interval: float,
    remove_direction: bool = True,
):
    if sampling_prob < 1.0:
        pld = privacy_loss_distribution.from_gaussian_mechanism(
            standard_deviation=standard_deviation,
            sensitivity=sensitivity,
            sampling_prob=sampling_prob,
            value_discretization_interval=value_discretization_interval,
            pessimistic_estimate=True,
        )
    else:
        pld = privacy_loss_distribution.from_gaussian_mechanism(
            standard_deviation=standard_deviation,
            sensitivity=sensitivity,
            value_discretization_interval=value_discretization_interval,
            pessimistic_estimate=True,
        )
    return pld._pmf_remove if remove_direction else pld._pmf_add


def create_test_pmfs(
    sampling_prob: float,
    sigma: float,
    sensitivity: float = 1.0,
    discretization: float = 1e-3,
) -> Dict[str, object]:
    """Create ground truth and unamplified PMFs for testing."""
    ground_truth_pmf = create_pld_and_extract_pmf(
        sigma, sensitivity, sampling_prob, discretization
    )
    unamplified_pmf = create_pld_and_extract_pmf(
        sigma, sensitivity, 1.0, discretization
    )
    return {
        "ground_truth_pmf": ground_truth_pmf,
        "unamplified_pmf": unamplified_pmf,
    }


