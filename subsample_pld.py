import numpy as np
from typing import Union, Any

from dp_accounting.pld.pld_pmf import SparsePLDPmf, DensePLDPmf


def subsample_losses(losses: np.ndarray, probs: np.ndarray, sampling_prob: float, remove_direction: bool, normalize_lower: bool) -> np.ndarray:
    """
    Compute the subsampled PMF over the same loss grid.

    Using the PLD subsampling identities, we accumulate CDFs/CCDFs and map through the
    stable subsampling loss transform, operating in the exclusive CCDF domain for improved
    numerical stability. Optionally normalizes the lower distribution weights.

    Args:
        losses: Monotone, uniformly spaced loss grid.
        probs: Probabilities aligned with `losses`.
        sampling_prob: Sampling probability q in [0, 1]. If 1, returns `probs`.
        remove_direction: If True, compute the remove-direction transformation; otherwise add-direction.
        normalize_lower: If True, renormalize the lower-probabilities e^{-l} p to sum to 1.

    Returns:
        Transformed probabilities over the same grid.
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
    # Check if the gaps between losses are constant and if so, compute the step
    diffs = np.diff(losses)
    step = np.mean(diffs)
    if not np.allclose(diffs, step, rtol=0.0, atol=1e-12):
        raise ValueError(f"losses must be a uniform grid with constant step, but they are in the range of {np.min(diffs)} to {np.max(diffs)}")
    

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
        
    # Exclusive CCDF with a leading one; supports prev = -1 (S[-1] = 1)
    if remove_direction:
        mix_ccdf = (1.0 - sampling_prob) * exclusive_ccdf_from_pdf(lower_probs) + sampling_prob * exclusive_ccdf_from_pdf(probs)
    else:
        mix_ccdf = exclusive_ccdf_from_pdf(probs)
    ccdf_ext = np.concatenate(([1.0], mix_ccdf))
    # Map transformed losses to indices on a uniform grid [-L, -L+d, ...]
    indices = np.clip(np.floor((transformed_losses - float(losses[0])) / step), -1, losses.size - 1).astype(int)
    # Previous index (j(i-1))
    prev_indices = np.concatenate(([-1], indices[:-1]))
    # Bin i mass using CCDF: p_i = S(prev) - S(curr)
    return ccdf_ext[prev_indices + 1] - ccdf_ext[indices + 1]

def exclusive_ccdf_from_pdf(probs: np.ndarray) -> np.ndarray:
    """Compute exclusive tail S(j) = sum_{k>j} p_k without 1-CDF subtraction."""
    tail_inclusive = np.flip(np.cumsum(np.flip(probs)))
    return tail_inclusive - probs

def stable_subsampling_loss(losses: np.ndarray, sampling_prob: float = 0.1, remove_direction: bool = True) -> np.ndarray:
    """
    Stable implementation of the loss mapping under subsampling.

    Computes:
        remove: l'(l) =  ln(1 + (exp(l) - 1) / q)
        add:    l'(l) = -ln(1 + (exp(-l) - 1) / q)
    with region-specific approximations to avoid catastrophic cancellation.
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