import numpy as np


def stable_subsampling_loss(losses: np.ndarray, sampling_prob: float = 0.1, remove_direction: bool = True) -> np.ndarray:
    if losses.ndim != 1:
        raise ValueError("losses must be a 1-D array")
    new_losses = np.zeros_like(losses)
    if not remove_direction:
        losses = -losses.copy()
    undefined_threshold = np.log(1 - sampling_prob) if sampling_prob < 1.0 else -np.inf
    undefined_mask = losses <= undefined_threshold
    new_losses[undefined_mask] = -np.inf
    large_loss_ind = losses >= 1
    new_losses[large_loss_ind] = losses[large_loss_ind] - np.log(sampling_prob) \
        + np.log1p((sampling_prob - 1) * np.exp(-losses[large_loss_ind]))
    small_loss_ind = (losses <= -1) & (~undefined_mask)
    if np.any(small_loss_ind):
        lvals = losses[small_loss_ind]
        raw_arg = np.expm1(lvals) / sampling_prob
        with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
            safe = raw_arg > -1.0
            out = np.empty_like(lvals)
            out[~safe] = -np.inf
            out[safe] = np.log1p(raw_arg[safe])
        new_losses[small_loss_ind] = out
    other_loss_ind = (~large_loss_ind) & (~small_loss_ind) & (~undefined_mask)
    new_losses[other_loss_ind] = np.log(1 + (np.exp(losses[other_loss_ind]) - 1) / sampling_prob)
    if not remove_direction:
        new_losses = -new_losses
    return new_losses


def exclusive_padded_ccdf_from_pdf(probs: np.ndarray) -> np.ndarray:
    '''
    Given an array of probabilities [p_0, p_1, ..., p_{n-1}], 
    return the array [1, 1-p_0, 1-p_0-p_1, ..., 1-p_0-p_1-...-p_{n-1}]
    '''
    return np.flip(np.cumsum(np.flip(np.concatenate((probs, [1.0-np.sum(probs)])))))


def subsample_losses(losses: np.ndarray, probs: np.ndarray, sampling_prob: float, remove_direction: bool, normalize_lower: bool) -> np.ndarray:
    if sampling_prob < 0 or sampling_prob > 1:
        raise ValueError("sampling_prob must be in [0, 1]")
    if not np.all(np.diff(losses) >= 0):
        raise ValueError("losses must be sorted")
    diffs = np.diff(losses)
    step = np.mean(diffs)
    if not np.allclose(diffs, step, rtol=0.0, atol=1e-12):
        raise ValueError(f"losses must be a uniform grid with constant step, but they are in the range of {np.min(diffs)} to {np.max(diffs)}")
    total_prob = float(np.sum(probs, dtype=np.float64))
    if total_prob > 1.0 + 1e-5:
        raise ValueError(f"sum(probs) = {total_prob} > 1")
    if sampling_prob == 1:
        return probs
    
    transformed_losses = stable_subsampling_loss(losses, sampling_prob, remove_direction)
    lower_probs = np.zeros_like(probs)
    lower_probs[probs > 0] = np.exp(np.log(probs[probs > 0]) - losses[probs > 0])
    lower_probs = lower_probs / np.sum(lower_probs)
    if remove_direction:
        mix_ccdf = (1.0 - sampling_prob) * exclusive_padded_ccdf_from_pdf(lower_probs) + sampling_prob * exclusive_padded_ccdf_from_pdf(probs)
    else:
        mix_ccdf = exclusive_padded_ccdf_from_pdf(probs)
    indices = np.clip(np.floor((transformed_losses - float(losses[0])) / step), -1, losses.size - 1).astype(int)
    prev_indices = np.concatenate(([-1], indices[:-1]))
    return mix_ccdf[prev_indices + 1] - mix_ccdf[indices + 1]