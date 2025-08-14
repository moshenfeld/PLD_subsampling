import numpy as np
from typing import Dict, Any, Union, Tuple

from dp_accounting.pld import privacy_loss_distribution
from dp_accounting.pld.pld_pmf import SparsePLDPmf, DensePLDPmf

from ..transforms import (
    subsample_losses,
)


def create_pld_and_extract_pmf(
    standard_deviation: float,
    sensitivity: float,
    sampling_prob: float,
    value_discretization_interval: float,
    remove_direction: bool = True,
):
    """Create a PLD PMF using dp-accounting and return the internal PMF object."""
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


def amplify_pld_separate_directions(
    base_pld: privacy_loss_distribution.PrivacyLossDistribution,
    sampling_prob: float,
) -> Dict[str, Any]:
    """Amplify a base PLD by subsampling, producing separate remove/add direction PMFs."""
    if not (0.0 < sampling_prob <= 1.0):
        raise ValueError("sampling_prob must be in (0, 1]")

    if sampling_prob == 1.0:
        return {"pmf_remove": base_pld._pmf_remove, "pmf_add": base_pld._pmf_add}

    base_losses, base_probs = dp_accounting_pmf_to_loss_probs(base_pld._pmf_remove)

    probs_remove = subsample_losses(
        losses=base_losses,
        probs=base_probs,
        sampling_prob=sampling_prob,
        remove_direction=True,
        normalize_lower=True,
    )
    probs_add = subsample_losses(
        losses=base_losses,
        probs=base_probs,
        sampling_prob=sampling_prob,
        remove_direction=False,
        normalize_lower=True,
    )

    disc = float(base_pld._pmf_remove._discretization)
    pess = bool(base_pld._pmf_remove._pessimistic_estimate)

    pmf_remove = loss_probs_to_dp_accounting_pmf(
        losses=base_losses,
        probs=probs_remove,
        discretization=disc,
        pessimistic_estimate=pess,
    )
    pmf_add = loss_probs_to_dp_accounting_pmf(
        losses=base_losses,
        probs=probs_add,
        discretization=disc,
        pessimistic_estimate=pess,
    )

    return {"pmf_remove": pmf_remove, "pmf_add": pmf_add}


def dp_accounting_pmf_to_loss_probs(pld_pmf: Union[SparsePLDPmf, DensePLDPmf, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract loss-probability mapping from PLDPmf objects as numpy arrays."""
    if isinstance(pld_pmf, DensePLDPmf):
        probs = pld_pmf._probs
        losses = pld_pmf._lower_loss + np.arange(np.size(probs))
    elif isinstance(pld_pmf, SparsePLDPmf):
        loss_probs = pld_pmf._loss_probs.copy()
        if len(loss_probs) == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        losses_sparse = np.array(list(loss_probs.keys()), dtype=np.int64)
        probs_sparse = np.array(list(loss_probs.values()), dtype=np.float64)
        losses = np.arange(np.min(losses_sparse), np.max(losses_sparse) + 1)
        probs = np.zeros(np.size(losses))
        probs[losses_sparse - np.min(losses_sparse)] = probs_sparse
    else:
        raise AttributeError(f"Unrecognized PMF format: {type(pld_pmf)}. Expected DensePLDPmf or SparsePLDPmf.")

    losses = losses.astype(np.float64) * float(pld_pmf._discretization)
    finite_target = float(max(0.0, 1.0 - pld_pmf._infinity_mass))
    sum_probs = float(np.sum(probs, dtype=np.float64))
    if sum_probs > 0.0:
        probs = probs * (finite_target / sum_probs)
    return losses, probs


def loss_probs_to_dp_accounting_pmf(losses: np.ndarray, probs: np.ndarray, discretization: float, pessimistic_estimate: bool) -> SparsePLDPmf:
    """Convert a loss-probability mapping to a dp-accounting SparsePLDPmf object."""
    pos_ind = probs > 0
    losses = losses[pos_ind]
    probs = probs[pos_ind]
    loss_indices = np.round(losses / discretization).astype(int)
    loss_probs_dict = dict(zip(loss_indices.tolist(), probs.tolist()))
    return SparsePLDPmf(
        loss_probs=loss_probs_dict,
        discretization=discretization,
        infinity_mass=np.maximum(0.0, 1.0 - np.sum(probs)),
        pessimistic_estimate=pessimistic_estimate,
    )


