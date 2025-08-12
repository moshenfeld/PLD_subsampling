import numpy as np
from subsample_pld_pmf import dp_accounting_pmf_to_loss_probs


def calc_W1_dist(pmf1, pmf2) -> float:
    """Compute Wasserstein-1 (Earth Mover) distance on finite support only."""
    losses1, probs1 = dp_accounting_pmf_to_loss_probs(pmf1)
    losses1 = losses1[probs1 > 0]
    probs1 = probs1[probs1 > 0]
    losses2, probs2 = dp_accounting_pmf_to_loss_probs(pmf2)
    losses2 = losses2[probs2 > 0]
    probs2 = probs2[probs2 > 0]

    all_losses = np.unique(np.concatenate([losses1, losses2]))
    all_losses = np.sort(all_losses)
    finite_mask = np.isfinite(all_losses)
    finite_losses = all_losses[finite_mask]

    pmf1_dict = dict(zip(losses1, probs1))
    pmf2_dict = dict(zip(losses2, probs2))
    probs1_finite = np.array([pmf1_dict.get(l, 0.0) for l in finite_losses])
    probs2_finite = np.array([pmf2_dict.get(l, 0.0) for l in finite_losses])

    cdf1 = np.cumsum(probs1_finite)
    cdf2 = np.cumsum(probs2_finite)

    p1_inf = probs1[np.isposinf(losses1)].sum() if np.any(np.isposinf(losses1)) else 0.0
    p2_inf = probs2[np.isposinf(losses2)].sum() if np.any(np.isposinf(losses2)) else 0.0

    if abs(p1_inf - p2_inf) > 0:
        return float('inf')
    if finite_losses.size <= 1:
        return 0.0
    return float(np.sum(np.abs(cdf1[:-1] - cdf2[:-1]) * np.diff(finite_losses)))


