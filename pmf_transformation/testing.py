import numpy as np
from typing import Dict, List, Any

from .wrappers import (
    create_pld_and_extract_pmf,
    dp_accounting_pmf_to_loss_probs,
    loss_probs_to_dp_accounting_pmf,
)
from .transforms import (
    subsample_losses,
)


def calc_W1_dist(losses1, probs1, losses2, probs2) -> float:
    losses1 = np.asarray(losses1, dtype=np.float64)
    probs1 = np.asarray(probs1, dtype=np.float64)
    losses2 = np.asarray(losses2, dtype=np.float64)
    probs2 = np.asarray(probs2, dtype=np.float64)
    mask1 = probs1 > 0
    mask2 = probs2 > 0
    losses1 = losses1[mask1]
    probs1 = probs1[mask1]
    losses2 = losses2[mask2]
    probs2 = probs2[mask2]
    all_losses = np.unique(np.concatenate([losses1, losses2]))
    all_losses = np.sort(all_losses)
    finite_mask = np.isfinite(all_losses)
    finite_losses = all_losses[finite_mask]
    pmf1_dict = dict(zip(losses1, probs1))
    pmf2_dict = dict(zip(losses2, probs2))
    probs1_finite = np.array([pmf1_dict.get(l, 0.0) for l in finite_losses])
    probs2_finite = np.array([pmf2_dict.get(l, 0.0) for l in finite_losses])
    ccdf1 = 1.0 - np.cumsum(probs1_finite)
    ccdf2 = 1.0 - np.cumsum(probs2_finite)
    if finite_losses.size <= 1:
        return 0.0
    return float(np.sum(np.abs(ccdf1[:-1] - ccdf2[:-1]) * np.diff(finite_losses)))


