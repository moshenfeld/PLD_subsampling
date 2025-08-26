import numpy as np
import copy
import warnings
from typing import Dict, Any, Union, Tuple

from dp_accounting.pld import privacy_loss_distribution
from dp_accounting.pld.pld_pmf import SparsePLDPmf, DensePLDPmf

from ..PLD_subsampling_impl import subsample_losses


def create_pld_and_extract_pmf(
    standard_deviation: float,
    sensitivity: float,
    sampling_prob: float,
    value_discretization_interval: float,
    remove_direction: bool = True,
):
    """Create a PLD via dp-accounting and return the internal PMF for one direction.

    When `sampling_prob < 1`, `dp-accounting` constructs the amplified PLD directly. This
    helper returns either the remove-direction PMF (`_pmf_remove`) or the add-direction
    PMF (`_pmf_add`) from that PLD depending on `remove_direction`.
    """
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
) -> privacy_loss_distribution.PrivacyLossDistribution:
    """Amplify a base PLD by subsampling and return a new PLD with separate directions.

    Internally, we read both PMFs from `base_pld`, apply our subsampling
    transform for both directions on the shared loss grid, and reconstruct a
    `PrivacyLossDistribution` with the two transformed PMFs.
    """
    if not (0.0 < sampling_prob <= 1.0):
        raise ValueError("sampling_prob must be in (0, 1]")

    if sampling_prob == 1.0:
        return base_pld


    base_losses_remove, base_probs_remove = dp_accounting_pmf_to_loss_probs(base_pld._pmf_remove)
    probs_remove = subsample_losses(
        losses=base_losses_remove,
        probs=base_probs_remove,
        sampling_prob=sampling_prob,
        remove_direction=True,
        normalize_lower=True,
    )
    pmf_remove = loss_probs_to_dp_accounting_pmf(
        losses=base_losses_remove,
        probs=probs_remove,
        discretization=base_pld._pmf_remove._discretization,
        pessimistic_estimate=base_pld._pmf_remove._pessimistic_estimate,
    )

    base_losses_add, base_probs_add = dp_accounting_pmf_to_loss_probs(base_pld._pmf_add)
    probs_add = subsample_losses(
        losses=base_losses_add,
        probs=base_probs_add,
        sampling_prob=sampling_prob,
        remove_direction=False,
        normalize_lower=True,
    )
    pmf_add = loss_probs_to_dp_accounting_pmf(
        losses=base_losses_add,
        probs=probs_add,
        discretization=base_pld._pmf_add._discretization,
        pessimistic_estimate=base_pld._pmf_add._pessimistic_estimate,
    )

    return privacy_loss_distribution.PrivacyLossDistribution(pmf_remove=pmf_remove, pmf_add=pmf_add)

def dp_accounting_pmf_to_loss_probs(pld_pmf: Union[SparsePLDPmf, DensePLDPmf, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract a dense loss grid and probabilities from a PLD PMF.

    - For dense PMFs, the probabilities are copied from the internal storage.
    - For sparse PMFs, we expand the sparse dictionary to its contiguous loss grid.
    - Probabilities are rescaled to sum to `(1 - infinity_mass)` to represent the finite mass only.
    """
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
    """Convert a loss-probability mapping to a dp-accounting `SparsePLDPmf`.

    The resulting PMF will store mass only on the provided loss indices. The remaining mass,
    if any, is placed at infinity.
    """
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


def scale_pmf_infinity_mass(
    pld_pmf: Union[SparsePLDPmf, DensePLDPmf, Any],
    delta: float,
) -> Union[SparsePLDPmf, DensePLDPmf]:
    """Increase the infinity mass by `delta` and scale finite probabilities accordingly.

    Given base infinity mass β, this returns a new PMF with:
    - infinity mass set to β + δ
    - finite probabilities multiplied by (1 − β − δ) / (1 − β)

    The PMF type (dense or sparse) is preserved.
    Constraints: 0 ≤ δ ≤ 1 − β.
    """

    beta = float(pld_pmf._infinity_mass)
    finite_mass = 1.0 - beta
    if not (0.0 <= delta <= finite_mass + 1e-18):
        raise ValueError(
            f"delta must satisfy 0 <= delta <= 1 - beta (beta={beta}) so that beta+delta <= 1; got {delta}."
        )

    new_infinity_mass = beta + float(delta)
    scale = (1.0 - new_infinity_mass) / finite_mass

    new_pmf = copy.deepcopy(pld_pmf)
    if isinstance(pld_pmf, DensePLDPmf):
        new_pmf._probs = new_pmf._probs * scale
        new_pmf._infinity_mass = new_infinity_mass
        return new_pmf

    if isinstance(pld_pmf, SparsePLDPmf):
        new_pmf._loss_probs = {k: v * scale for k, v in new_pmf._loss_probs.items()}
        new_pmf._infinity_mass = new_infinity_mass
        return new_pmf

    raise AttributeError(
        f"Unrecognized PMF format: {type(pld_pmf)}. Expected DensePLDPmf or SparsePLDPmf."
    )

def scale_pld_infinity_mass(
    pld: privacy_loss_distribution.PrivacyLossDistribution,
    delta: float,
) -> privacy_loss_distribution.PrivacyLossDistribution:
    """Apply the same infinity-mass scaling to both remove/add PMFs of a PLD.

    Each internal PMF is transformed via `scale_pmf_infinity_mass` with the same delta,
    and a new `PrivacyLossDistribution` is returned.
    """
    pmf_remove_scaled = scale_pmf_infinity_mass(pld._pmf_remove, delta)
    pmf_add_scaled = scale_pmf_infinity_mass(pld._pmf_add, delta)
    return privacy_loss_distribution.PrivacyLossDistribution(
        pmf_remove=pmf_remove_scaled, pmf_add=pmf_add_scaled
    )