import numpy as np
from typing import Dict, Any, List, Tuple

from pmf_utils import create_pld_and_extract_pmf
from pmf_compare import calc_W1_dist
from analytic_derivation import subsampled_gaussian_probabilities_from_losses
from subsample_pld_pmf import (
    dp_accounting_pmf_to_loss_probs,
    loss_probs_to_dp_accounting_pmf,
    subsample_losses,
    stable_subsampling_loss,
)


def build_pmf(losses: np.ndarray, probs: np.ndarray, discretization: float, pessimistic: bool):
    return loss_probs_to_dp_accounting_pmf(losses, probs, discretization, pessimistic)


def eps_for_deltas(pmf, deltas: List[float]) -> List[float]:
    return [pmf.get_epsilon_for_delta(d) for d in deltas]


def subsample_on_target_grid(
    base_losses: np.ndarray,
    base_probs: np.ndarray,
    q: float,
    remove_direction: bool,
    target_losses: np.ndarray,
) -> np.ndarray:
    if q < 0.0 or q > 1.0:
        raise ValueError("q must be in [0,1]")
    if not np.all(np.diff(base_losses) >= 0):
        raise ValueError("base_losses must be sorted")
    if not np.all(np.diff(target_losses) >= 0):
        raise ValueError("target_losses must be sorted")

    # Transform target losses via the subsampling map (remove_direction branch only used when requested)
    transformed = stable_subsampling_loss(target_losses, q, remove_direction)

    # Build CDFs on the base grid
    P_cdf = np.cumsum(base_probs)
    lower_probs = np.zeros_like(base_probs)
    mask_pos = base_probs > 0
    lower_probs[mask_pos] = np.exp(np.log(base_probs[mask_pos]) - base_losses[mask_pos])
    Q_cdf = np.cumsum(lower_probs)

    # For each transformed loss, map to the base index j where base_losses[j] <= transformed[i]
    j = np.searchsorted(base_losses, transformed, side='right') - 1
    j = np.clip(j, -1, base_losses.size - 1)

    # Evaluate mixture CDF at those indices (with CDF[-1] = 0)
    P_cdf_ext = np.concatenate(([0.0], P_cdf))
    Q_cdf_ext = np.concatenate(([0.0], Q_cdf))
    # shift by +1 because j==-1 should map to 0
    Pprime_cdf = (1.0 - q) * Q_cdf_ext[j + 1] + q * P_cdf_ext[j + 1]

    # Forward differences along the target grid give target-grid probabilities
    p_target = Pprime_cdf - np.concatenate(([0.0], Pprime_cdf[:-1]))
    return p_target


def compare_one(sigma: float, q: float, d: float, deltas: List[float]) -> None:
    print(f"\n=== sigma={sigma}, q={q}, d={d} ===")
    # Library reference: TF_subsampled
    TF_subsampled_pmf = create_pld_and_extract_pmf(sigma, 1.0, q, d, True)
    TF_subsampled_losses, TF_subsampled_probs = dp_accounting_pmf_to_loss_probs(TF_subsampled_pmf)

    # TF original grid
    TF_original_pmf = create_pld_and_extract_pmf(sigma, 1.0, 1.0, d, True)
    TF_original_losses, TF_original_probs = dp_accounting_pmf_to_loss_probs(TF_original_pmf)

    # GT original grid (analytic)
    GT_original_losses, GT_original_probs = subsampled_gaussian_probabilities_from_losses(sigma, 1.0, d, True)

    # Variant A: Our_TF_subsampling (base grid = TF original), current implementation
    probs_A = subsample_losses(TF_original_losses, TF_original_probs, q, True, normalize_lower=True)
    pmf_A = build_pmf(TF_original_losses, probs_A, d, TF_original_pmf._pessimistic_estimate)

    # Variant B: Our_GT_subsampling (base grid = GT original), current implementation on base grid
    probs_B = subsample_losses(GT_original_losses, GT_original_probs, q, True, normalize_lower=True)
    pmf_B = build_pmf(GT_original_losses, probs_B, d, TF_original_pmf._pessimistic_estimate)

    # Variant C: Our_GT_subsampling on target grid extended by ~-ln(q)
    max_base = GT_original_losses[-1]
    min_base = GT_original_losses[0]
    l_max = max_base - np.log(q) + 10 * d
    l_min = max(min_base, np.log(1.0 - q) - 10 * d) if q < 1.0 else min_base
    l_min = np.floor(l_min / d) * d
    l_max = np.ceil(l_max / d) * d
    target_losses = np.arange(l_min, l_max + d, d)
    probs_C = subsample_on_target_grid(GT_original_losses, GT_original_probs, q, True, target_losses)
    pmf_C = build_pmf(target_losses, probs_C, d, TF_original_pmf._pessimistic_estimate)

    variants: List[Tuple[str, Any, np.ndarray, np.ndarray]] = [
        ("TF_subsampled", TF_subsampled_pmf, TF_subsampled_losses, TF_subsampled_probs),
        ("Our_TF_subsampling", pmf_A, TF_original_losses, probs_A),
        ("Our_GT_subsampling_base", pmf_B, GT_original_losses, probs_B),
        ("Our_GT_subsampling_target", pmf_C, target_losses, probs_C),
    ]

    # Metrics per variant
    for name, pmf, losses, probs in variants:
        try:
            inf_mass = pmf._infinity_mass
        except Exception:
            inf_mass = None
        w1 = calc_W1_dist(losses, probs, TF_subsampled_losses, TF_subsampled_probs)
        eps = eps_for_deltas(pmf, deltas)
        finite_eps = sum(np.isfinite(eps))
        print(f"{name:24s} | W1={w1:.6e} | inf_mass={inf_mass:.3e} | finite_eps={finite_eps}/{len(deltas)}")


def main():
    deltas = [10.0 ** (-k) for k in range(1, 11)]
    cases = [
        (0.5, 0.1, 1e-4),
        (0.5, 0.5, 1e-4),
        (0.5, 0.9, 1e-4),
    ]
    for sigma, q, d in cases:
        compare_one(sigma, q, d, deltas)


if __name__ == "__main__":
    main()


