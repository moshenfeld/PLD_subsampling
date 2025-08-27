import numpy as np

def subsample_losses(losses: np.ndarray, probs: np.ndarray, sampling_prob: float, remove_direction: bool) -> np.ndarray:
    '''
    - Transform the probabilities to the CCDF of the new mixture probability
    - Compute the indices of the new losses using the losses transformation
    - Compute the new probabilities by subtracting the CCDF values at the indices
    '''
    ensure_inputs_validity(losses, probs, sampling_prob)
    
    if sampling_prob == 1:
        return probs

    # For the remove direction, P'(l) = (1-q)*Q(l) + q*P(l)
    if remove_direction:
        lower_probs = np.zeros_like(probs)
        lower_probs[probs > 0] = np.exp(np.log(probs[probs > 0]) - losses[probs > 0]) # Q(l) = P(l)*e^{-l}, computed in log space for numerical stability
        lower_probs /= np.sum(lower_probs) # By definition, the PLD under Q sums up to 1, since Q(inf) = 0
        mixed_probs = (1.0 - sampling_prob) * lower_probs + sampling_prob * probs
    # For the add direction, P'(l) = P(l)
    else:
        mixed_probs = probs
    # We use the padded CCDF (see documentation of exclusive_padded_ccdf_from_pdf) to maximize numerical stability in the high losses regime
    mix_ccdf = exclusive_padded_ccdf_from_pdf(mixed_probs)

    # Compute in a stable manner l' = ln(1 + (exp(l) - 1) / q) for the remove direction,
    #                        and l' = -ln(1 + (exp(-l) - 1) / q) for the add direction.
    transformed_losses = stable_subsampling_loss(losses, sampling_prob, remove_direction)
    # I(l) = the index on the maximal loss <= l'
    # which is equivalent to floor((l' - l_0) / step)
    losses_step = np.mean(np.diff(losses))
    indices = np.floor((transformed_losses - float(losses[0])) / losses_step)
    # In the remove direction, the index can't be larger than the highest loss
    if remove_direction:
        indices = np.clip(indices, -1, losses.size-1).astype(int)
    # In the add direction, the index can include the infinity loss
    else:
        indices = np.clip(indices, -1, losses.size).astype(int)
    # p'[i] = P'(I(l_i)) - P'(I(l_{i-1}) + 1)
    prev_indices = np.concatenate(([-1], indices[:-1]))
    return mix_ccdf[prev_indices + 1] - mix_ccdf[indices + 1]


def ensure_inputs_validity(losses: np.ndarray, probs: np.ndarray, sampling_prob: float):
    '''
    Ensure that the inputs are valid.
    - sampling_prob must be in [0, 1]
    - losses must be sorted and have a constant step
    - probs must be non-negative and sum up to 1
    '''
    if sampling_prob < 0 or sampling_prob > 1:
        raise ValueError("sampling_prob must be in [0, 1]")
    if not np.all(np.diff(losses) >= 0):
        first_negative_index = np.argmax(np.diff(losses) < 0)
        raise ValueError(f"losses must be sorted, but losses[{first_negative_index}] = {losses[first_negative_index]} > losses[{first_negative_index+1}] = {losses[first_negative_index+1]}")
    if not np.all(probs >= 0.0):
        first_negative_index = np.argmax(probs < 0)
        raise ValueError(f"probs must be non-negative, but p[{first_negative_index}] = {probs[first_negative_index]} < 0")
    if not np.sum(probs) <= 1.0 + 1e-12:
        raise ValueError(f"sum(probs) = {np.sum(probs)} > 1")
    diffs = np.diff(losses)
    step = np.mean(diffs)
    if not np.allclose(diffs, step, rtol=0.0, atol=1e-12):
        raise ValueError(f"losses must be a uniform grid with constant step, but they are in the range of {np.min(diffs)} to {np.max(diffs)}")


def stable_subsampling_loss(losses: np.ndarray, sampling_prob: float = 0.1, remove_direction: bool = True) -> np.ndarray:
    '''
    Compute in a stable manner l' = ln(1 + (exp(l) - 1) / q) for the remove direction,
                           and l' = -ln(1 + (exp(-l) - 1) / q) for the add direction.
    '''
    if sampling_prob <= 0 or sampling_prob > 1:
        raise ValueError("sampling_prob must be in (0, 1]")
    if sampling_prob == 1:
        return losses
    
    new_losses = np.zeros_like(losses)
    if not remove_direction:
        losses = -losses.copy()
    
    # If l < log(1-q), l' is not defined and is set to -inf
    undefined_threshold = np.log(1 - sampling_prob) if sampling_prob < 1.0 else -np.inf
    undefined_ind = losses <= undefined_threshold
    new_losses[undefined_ind] = -np.inf

    # In case log(1-q) < l < q we simply compute
    # ln(1 + (exp(l) - 1) / q) = log1p(expm1(l)/q)
    small_loss_ind = ~undefined_ind & (losses < sampling_prob)
    new_losses[small_loss_ind] = np.log1p(np.expm1(losses[small_loss_ind])/sampling_prob)

    # In case q < l < 1, we we simply compute
    # ln(1 + (exp(l) - 1) / q) = log(1 + expm1(l)/q)
    medium_loss_ind = ~undefined_ind & ~small_loss_ind & (losses > 1)
    new_losses[medium_loss_ind] = np.log(1 + np.expm1(losses[medium_loss_ind])/sampling_prob)

    # In case l > 1, we use the fact that
    # ln(1 + (exp(l) - 1) / q) = log(exp(l)/q * (q*exp(-l) + 1 - exp(-l)))
    #                          = l - log(q) + log(1 + (q-1)*exp(-l))
    #                          = l - log(q) + log1p((q-1)*exp(-l))
    large_loss_ind = ~undefined_ind & ~small_loss_ind & ~medium_loss_ind
    new_losses[large_loss_ind] = losses[large_loss_ind] - np.log(sampling_prob) \
        + np.log1p((sampling_prob - 1) * np.exp(-losses[large_loss_ind]))
    
    if not remove_direction:
        new_losses = -new_losses
    return new_losses


def exclusive_padded_ccdf_from_pdf(probs: np.ndarray) -> np.ndarray:
    '''
    Given an array of probabilities [p_0, p_1, ..., p_{n-1}], 
    return the array [1, 1-p_0, 1-p_0-p_1, ..., 1-p_0-p_1-...-p_{n-1}, 0],
    which is equivalent to CCDF of [0, p_0, p_1, ...,p_{n-1}, p_{inf}],
    where p_{inf} = 1-p_0-p_1-...-p_{n-1} is the probability of the infinity loss.
    '''
    padded_probs = np.concatenate(([0.0], probs, [1.0-np.sum(probs)]))
    flipped_padded_probs = np.flip(padded_probs)
    padded_cumsum = np.cumsum(flipped_padded_probs) - flipped_padded_probs
    flipped_padded_cumsum = np.flip(padded_cumsum)
    return flipped_padded_cumsum




