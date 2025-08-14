import numpy as np
from typing import Tuple, Any
from subsample_pld_pmf import dp_accounting_pmf_to_loss_probs

def calc_W1_dist(losses1, probs1, losses2, probs2) -> float:
    """Compute Wasserstein-1 distance between two discrete distributions.

    - Each pair `(lossesX, probsX)` must have matching lengths.
    - The two pairs may have different lengths and different supports.
    - Losses may include +inf to represent infinity mass; if those masses differ, returns inf.
    """

    losses1 = np.asarray(losses1, dtype=np.float64)
    probs1 = np.asarray(probs1, dtype=np.float64)
    losses2 = np.asarray(losses2, dtype=np.float64)
    probs2 = np.asarray(probs2, dtype=np.float64)

    if losses1.shape[0] != probs1.shape[0]:
        raise ValueError(f'Length mismatch between losses1 and probs1: {losses1.shape[0]} != {probs1.shape[0]}')
    if losses2.shape[0] != probs2.shape[0]:
        raise ValueError(f'Length mismatch between losses2 and probs2: {losses2.shape[0]} != {probs2.shape[0]}')

    # Remove zero-mass entries to simplify alignment
    mask1 = probs1 > 0
    mask2 = probs2 > 0
    losses1 = losses1[mask1]
    probs1 = probs1[mask1]
    losses2 = losses2[mask2]
    probs2 = probs2[mask2]

    # Union grid over finite supports
    all_losses = np.unique(np.concatenate([losses1, losses2]))
    all_losses = np.sort(all_losses)
    finite_mask = np.isfinite(all_losses)
    finite_losses = all_losses[finite_mask]

    # Align probabilities on the union grid
    pmf1_dict = dict(zip(losses1, probs1))
    pmf2_dict = dict(zip(losses2, probs2))
    probs1_finite = np.array([pmf1_dict.get(l, 0.0) for l in finite_losses])
    probs2_finite = np.array([pmf2_dict.get(l, 0.0) for l in finite_losses])

    # CDFs over the common finite grid
    cdf1 = np.cumsum(probs1_finite)
    cdf2 = np.cumsum(probs2_finite)

    # Explicit +inf mass check
    p1_inf = probs1[np.isposinf(losses1)].sum() if np.any(np.isposinf(losses1)) else 0.0
    p2_inf = probs2[np.isposinf(losses2)].sum() if np.any(np.isposinf(losses2)) else 0.0
    if abs(p1_inf - p2_inf) > 0:
        return float('inf')

    if finite_losses.size <= 1:
        return 0.0

    return float(np.sum(np.abs(cdf1[:-1] - cdf2[:-1]) * np.diff(finite_losses)))


