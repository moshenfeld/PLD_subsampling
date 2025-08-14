import numpy as np
from typing import Tuple

from dp_accounting.pld.pld_pmf import SparsePLDPmf, DensePLDPmf

def subsample_pld_pmf(source_pld: 'PLDPmf', sampling_prob: float, remove_direction: bool = True) -> 'PLDPmf':
    """
    Transform a Privacy Loss Distribution PMF using subsampling transformation.
    
    This version bypasses discretization entirely and creates the PMF directly from the
    transformed arrays. This is suitable for functions that don't change the loss values
    or that handle their own discretization.
    
    Args:
        source_pld: Source PLDPmf object (SparsePLDPmf or DensePLDPmf)
        sampling_prob: Sampling probability q (0 < q <= 1)
        
    Returns:
        New PLDPmf object with transformed distribution
        
    Raises:
        ImportError: If dp_accounting library is not available
    """
    
    # Extract loss-probability data as numpy arrays (only finite values)
    source_losses, source_probs = dp_accounting_pmf_to_loss_probs(source_pld)
    # Apply transformation on the finite distribution
    # transformed_probs = subsample_losses_new(source_losses, source_probs, sampling_prob, remove_direction)
    transformed_probs = subsample_losses(source_losses, source_probs, sampling_prob, remove_direction)
    # Build PMF on the library's discretization grid using integer loss indices
    return loss_probs_to_dp_accounting_pmf(source_losses, transformed_probs, source_pld._discretization, source_pld._pessimistic_estimate)

def dp_accounting_pmf_to_loss_probs(pld_pmf: 'PLDPmf') -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract loss-probability mapping from PLDPmf objects as numpy arrays
    Args:
        pld_pmf: PLDPmf object (DensePLDPmf or SparsePLDPmf)
        
    Returns:
        Tuple of (losses: np.ndarray, probs: np.ndarray) with only finite values
        Only includes non-zero probabilities for efficiency
        
    Raises:
        AttributeError: If the PMF format is not recognized
    """
    # Check if it's DensePLDPmf format (which is what create_pmf typically returns)
    if isinstance(pld_pmf, DensePLDPmf):
        probs = pld_pmf._probs
        losses = pld_pmf._lower_loss + np.arange(np.size(probs))

    # Check if it's SparsePLDPmf format (has _loss_probs directly)
    elif isinstance(pld_pmf, SparsePLDPmf):
        loss_probs = pld_pmf._loss_probs.copy()
        if len(loss_probs) == 0:
            # Empty sparse PMF: return empty arrays
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        # Ensure monotone losses and aligned probs by sorting keys
        losses_sparse = np.array(list(loss_probs.keys()),   dtype=np.int64)
        probs_sparse  = np.array(list(loss_probs.values()), dtype=np.float64)
        losses = np.arange(np.min(losses_sparse), np.max(losses_sparse) + 1)
        probs  = np.zeros(np.size(losses))
        probs[losses_sparse - np.min(losses_sparse)] = probs_sparse

    # If neither format is recognized, raise an error
    else:
        raise AttributeError(f"Unrecognized PMF format: {type(pld_pmf)}. Expected DensePLDPmf or SparsePLDPmf.")

    # Convert to float64 and multiply by discretization
    losses = losses.astype(np.float64) * float(pld_pmf._discretization)

    # Robust normalization of finite probabilities
    finite_target = float(max(0.0, 1.0 - pld_pmf._infinity_mass))
    sum_probs = float(np.sum(probs, dtype=np.float64))
    if sum_probs > 0.0:
        probs = probs * (finite_target / sum_probs)
    return losses, probs

def loss_probs_to_dp_accounting_pmf(losses: np.ndarray, probs: np.ndarray, discretization: float, pessimistic_estimate: bool) -> 'PLDPmf':
    """
    Convert a loss-probability mapping to a PLDPmf object
    Args:
        losses: numpy array of losses
        probs: numpy array of probabilities
        discretization: discretization of the PMF
        pessimistic_estimate: whether to use the pessimistic estimate
    Returns:
        SparsePLDPmf object
    """
    pos_ind = probs > 0
    losses = losses[pos_ind]
    probs  = probs[pos_ind]

    # Build PMF on the library's discretization grid using integer loss indices
    loss_indices = np.round(losses / discretization).astype(int)
    loss_probs_dict = dict(zip(loss_indices.tolist(), probs.tolist()))
    return SparsePLDPmf(
        loss_probs=loss_probs_dict,
        discretization=discretization,
        infinity_mass=np.maximum(0.0, 1.0 - np.sum(probs)),
        pessimistic_estimate=pessimistic_estimate
    )

def subsample_losses(losses: np.ndarray, probs: np.ndarray, sampling_prob: float, remove_direction: bool, normalize_lower: bool) -> np.ndarray:
    """    
    Given a PDF p_i (probs) over l_i (losses) between distributions P and Q and c (sampling_prob),
    compute the PDF p'_i over the same losses between:
        Remove case: P'  = (1-c) * Q + c * P and Q' = Q
        Add case:    P'' = P                 and Q'' = (1-c) * P + c * Q
    using the following formula:
        P_i = sum_{j=0}^{i} p_j
        q_i = e^{-l_i} p_i
        Q_i = sum_{j=0}^{i} q_j
        Remove:
            j(i) = floor(ln(1+(exp(l_i)-1)/c))
            P'_i = (1-c) * Q_{j(i)} + c * P_{j(i)}
            p'_i = P'_i - P'_{i-1} (where P_{-1} = 0)
        Add:
            j(i) = floor(ln(1+(exp(l_i)-1)/c))
            P''_i = P_{j(i)}
            p''_i = P''_i - P''_{i-1} (where P_{-1} = 0)
    """
    # raise an error if:
    #   1. sampling_prob is not in [0, 1]
    #   2. losses are not sorted
    #   3. sum(probs) > 1
    if sampling_prob < 0 or sampling_prob > 1:
        raise ValueError("sampling_prob must be in [0, 1]")
    if not np.all(np.diff(losses) >= 0):
        # find the first example of unsorted values, the two indexes and the two values
        raise ValueError("losses must be sorted")
    total_prob = float(np.sum(probs, dtype=np.float64))
    # Allow tiny FP overshoot and renormalize if needed
    if total_prob > 1.0 + 1e-5:
        raise ValueError(f"sum(probs) = {total_prob} > 1")

    if sampling_prob == 1:
        return probs

    # Transform reference losses using the formula ln(1 + (exp(l_i) - 1)/q)
    transformed_losses = stable_subsampling_loss(losses, sampling_prob, remove_direction)

    # Compute lower distribution weights: q_j = e^{-l_j} p_j
    lower_probs = np.zeros_like(probs)
    lower_probs[probs > 0] = np.exp(np.log(probs[probs > 0]) - losses[probs > 0])
    if normalize_lower:
        lower_probs /= np.maximum(1, np.sum(lower_probs))
        
    if remove_direction:
        # P'_cdf(i) = (1-q) * Q_cdf(j(i)) + q * P_cdf(j(i))
        probs_cumsum = (1.0 - sampling_prob) * np.cumsum(lower_probs) + sampling_prob * np.cumsum(probs)
    else:
        probs_cumsum = np.cumsum(probs)

    # #plot orobs cumsum and lower probs cumsum
    # import matplotlib.pyplot as plt
    # plt.plot(1-probs_cumsum, label='1-upper_probs_cumsum')
    # plt.plot(1-np.cumsum(lower_probs), label='1-lower_probs_cumsum')
    # plt.legend()
    # plt.yscale('log')
    # plt.show()
 
    # Inclusive prefix sums with a leading zero; supports prev = -1 for first interval
    probs_cumsum = np.concatenate(([0.0], probs_cumsum))
    # probs_cumsum *= ((1-sampling_prob) * np.sum(lower_probs) + sampling_prob * np.sum(probs)) / np.max(probs_cumsum)
    # Debug print removed for cleanliness

    # For each transformed loss l, find the largest index i where losses[i] <= l
    indices = np.searchsorted(losses, transformed_losses, side='right') - 1
    # print(f'Max(transformed_losses) = {np.max(transformed_losses)}, max(losses) = {np.max(losses)}, max index = {np.max(indices)-1}, size = {np.size(indices)}, ind max loss = {np.searchsorted(losses, np.max(losses), side="right") - 1}')

    # Compute the previous index while guarding against decreasing I(l_i) due to numerical quirks
    prev_indices = np.minimum(np.concatenate(([-1], indices[:-1])), indices)
    # Bin i mass = sum_{j=prev+1..indices[i]} new_probs[j]
    return probs_cumsum[indices + 1] - probs_cumsum[prev_indices + 1]

def stable_subsampling_loss(losses: np.ndarray, sampling_prob: float = 0.1, remove_direction: bool = True) -> np.ndarray:
    """
    Compute in a stable manner:
        if remove_direction: l'_i =  ln(1+(e^{ l_i}-1)/q)
        else:                l'_i = -ln(1+(e^{-l_i}-1)/q)
    """
    if losses.ndim != 1:
        raise ValueError("losses must be a 1-D array")
    new_losses = np.zeros_like(losses)

    if not remove_direction:
        losses = -losses.copy()

    # Handle the case where ln(1+(e^{l_i}-1)/q) is undefined because exp(l_i) - 1 < q
    undefined_threshold = np.log(1 - sampling_prob) if sampling_prob < 1.0 else -np.inf
    undefined_mask = losses <= undefined_threshold
    new_losses[undefined_mask] = -np.inf

    # For large positive x we use the fact that (q-1)*e^(-l) << 1 so,
    # ln(1+(e^l-1)/q) = ln((q + e^l - 1)/q)
    #                 = ln(e^l * (1 + (q-1)*e^(-l)) / q)
    #                 = l + ln(1 + (q-1)*e^(-l)) - ln(q)
    #                 = l - ln(q) + log1p((q-1)*exp(-l))
    large_loss_ind = losses >= 1
    new_losses[large_loss_ind] = losses[large_loss_ind] - np.log(sampling_prob) \
        + np.log1p((sampling_prob-1)*np.exp(-losses[large_loss_ind]))

    # For large negative x we use the fact that l << 1 and (e^l-1)/q << 1 so,
    # ln(1+(e^l-1)/q) = log1p((e^l-1)/q) = log1p(expm1(l)/q)
    small_loss_ind = (losses <= -1) & (~undefined_mask)
    new_losses[small_loss_ind] = np.log1p(np.expm1(losses[small_loss_ind]) / sampling_prob)

    # For other x we use the original formula
    other_loss_ind = (~large_loss_ind) & (~small_loss_ind) & (~undefined_mask)
    new_losses[other_loss_ind] = np.log(1 + (np.exp(losses[other_loss_ind]) - 1) / sampling_prob)

    if not remove_direction:
        new_losses = -new_losses
    return new_losses