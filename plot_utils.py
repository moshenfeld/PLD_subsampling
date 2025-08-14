import numpy as np
import matplotlib.pyplot as plt
from typing import List
from analytic_derivation import analytic_subsampled_epsilon_for_delta


def create_pmf_cdf_plot(
    versions: List[dict],
    title_suffix: str = '',
):
    """
    Plot CDFs for a sequence of PLDs provided as (losses, probs) pairs.

    - Union the provided loss grids to form a common grid
    - Compute ground-truth probabilities on this grid using
      subsampled_gaussian_probabilities_from_losses
    - Plot the ground-truth CDF and each PMF's CDF
    - Compute W1 for each PMF relative to ground truth and show it in legend
    """
    # Union of all finite losses across versions and ground truth
    losses_list = []
    for entry in versions:
        losses = np.asarray(entry['losses'], dtype=np.float64)
        finite_mask = np.isfinite(losses)
        losses_list.append(losses[finite_mask])
    union_losses = np.unique(np.concatenate(losses_list)) if losses_list else np.array([], dtype=np.float64)
    union_losses = np.sort(union_losses[np.isfinite(union_losses)])

    # Figure and main panel
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])
    ax_main = fig.add_subplot(gs[0, :])

    # Build lines for plotting: treat all PMFs equally
    lines = []

    colors = ['r', 'b', 'g', 'm']
    for idx, entry in enumerate(versions):
        name = entry.get('name', f'PMF {idx+1}')
        losses = np.asarray(entry['losses'], dtype=np.float64)
        probs = np.asarray(entry['probs'], dtype=np.float64)
        pmf_map = dict(zip(losses, probs))
        grid_probs = np.array([pmf_map.get(x, 0.0) for x in union_losses])
        cdf_vals = np.cumsum(grid_probs)
        color = colors[idx % len(colors)]
        lines.append({
            'label': f'{name} (W1={entry.get("W1_vs_GT", "N/A"):.2e})' if "W1_vs_GT" in entry.keys() else name,
            'cdf': cdf_vals,
            'color': color,
            'style': '--',
            'alpha': 0.85,
        })

    # Plot all lines on main panel
    for line in lines:
        ax_main.plot(union_losses, line['cdf'], linestyle=line['style'], color=line['color'],
                     alpha=line['alpha'], label=line['label'])

    title = 'CDF Comparison'
    if title_suffix:
        title += f' — {title_suffix}'
    ax_main.set_title(title)
    ax_main.set_xlabel('Privacy Loss')
    ax_main.set_ylabel('CDF')
    ax_main.grid(True, alpha=0.3)
    ax_main.legend()
    ax_main.set_ylim(0.0, 1.0)

    # Determine focus centers via largest spread in log space across ALL lines
    finite_losses = union_losses
    tiny = 1e-16
    if finite_losses.size and len(lines) >= 1:
        # Stack CDFs: shape (k, n)
        cdfs = np.vstack([ln['cdf'] for ln in lines]) if len(lines) > 1 else np.vstack([lines[0]['cdf'], lines[0]['cdf']])
        mean_cdf = np.mean(cdfs, axis=0)
        # Left side: where mean CDF <= 0.5, maximize range of log CDFs
        log_cdfs = np.log(np.maximum(cdfs, tiny))
        left_mask = mean_cdf <= 0.5
        left_range = np.max(log_cdfs, axis=0) - np.min(log_cdfs, axis=0)
        left_weighted = np.where(left_mask, left_range, -np.inf)
        # Right side: focus near 1 using log(1-CDF)
        thresholds = [0.999, 0.99, 0.95, 0.9, 0.5]
        right_weighted = np.full_like(left_weighted, -np.inf)
        log_one_minus = np.log(np.maximum(1.0 - cdfs, tiny))
        right_range = np.max(log_one_minus, axis=0) - np.min(log_one_minus, axis=0)
        right_mask = mean_cdf > 0.5
        # Prefer very high CDF regions if available
        for thr in thresholds:
            mask_thr = mean_cdf >= thr
            if np.any(mask_thr):
                cand = np.where(mask_thr, right_range, -np.inf)
                if np.isfinite(np.max(cand)) and np.max(cand) >= 0:
                    right_weighted = cand
                    right_mask = mask_thr
                    break
        if not np.isfinite(np.max(right_weighted)):
            right_weighted = np.where(right_mask, right_range, -np.inf)
        # Fallbacks if side is empty
        left_idx = int(np.nanargmax(left_weighted)) if np.any(left_mask) else 0
        right_idx = int(np.nanargmax(right_weighted)) if np.any(right_mask) else (len(finite_losses) - 1)
        left_loss = float(finite_losses[left_idx])
        right_loss = float(finite_losses[right_idx])
    else:
        left_loss = 0.0
        right_loss = 0.0
    # Derive adaptive x-windows around focus points using log-gap bands
    # Fallback span if bands are degenerate
    global_span = max(float(finite_losses[-1] - finite_losses[0]) if finite_losses.size else 0.0, 1e-12)
    step = float(finite_losses[1] - finite_losses[0]) if finite_losses.size > 1 else 1.0
    default_span = max(200 * step, 0.01 * global_span)
    # Build windows based on where the weighted metrics remain within a fraction of the peak
    def window_from_metric(metric: np.ndarray, center_idx: int, mask: np.ndarray, frac: float = 0.1, pad_bins: int = 50):
        if metric.size == 0 or center_idx < 0:
            return None
        peak = float(metric[center_idx]) if np.isfinite(metric[center_idx]) else -np.inf
        if not np.isfinite(peak) or peak <= 0:
            return None
        thresh = peak * frac
        active = (metric >= thresh) & mask
        # Expand from center to contiguous bounds
        left_b = center_idx
        while left_b - 1 >= 0 and active[left_b - 1]:
            left_b -= 1
        right_b = center_idx
        n = metric.size
        while right_b + 1 < n and active[right_b + 1]:
            right_b += 1
        left_b = max(0, left_b - pad_bins)
        right_b = min(n - 1, right_b + pad_bins)
        return float(finite_losses[left_b]), float(finite_losses[right_b])

    # left tail centered at left_loss
    ax_left = fig.add_subplot(gs[1, 0])
    for line in lines:
        ax_left.plot(finite_losses, line['cdf'], linestyle=line['style'], color=line['color'], alpha=0.8)
    ax_left.set_yscale('log')
    # Focus x-limits via adaptive log-gap band; fallback to default span
    left_win = window_from_metric(left_weighted, left_idx, left_mask) if finite_losses.size else None
    if left_win is not None:
        left_min, left_max = left_win
    else:
        left_min = left_loss - default_span
        left_max = left_loss + default_span
    # Asymmetric focus: on the left subplot, show more to the RIGHT of the main point
    left_width = max(1e-12, left_max - left_min)
    left_min = left_loss - 0.2 * left_width
    left_max = left_loss + 0.8 * left_width
    if finite_losses.size:
        ax_left.set_xlim(max(float(finite_losses.min()), left_min), min(float(finite_losses.max()), left_max))
        # Concentrate Y-range to the relevant CDF values within the left window across all lines
        mask_left = (finite_losses >= ax_left.get_xlim()[0]) & (finite_losses <= ax_left.get_xlim()[1])
        if np.any(mask_left):
            stacked = np.vstack([ln['cdf'][mask_left] for ln in lines]) if lines else np.zeros((1, np.sum(mask_left)))
            y_left_vals = stacked.flatten()
            pos = y_left_vals[y_left_vals > 0.0]
            y_min = float(np.max([np.min(pos) if pos.size else 1e-16, 1e-16]))
            y_max = float(np.max(y_left_vals)) if y_left_vals.size else 1.0
            ax_left.set_ylim(max(y_min * 0.8, 1e-16), min(y_max * 1.25, 1.0))
    ax_left.set_title('Left extreme (log CDF)')
    ax_left.set_xlabel('Privacy Loss')
    ax_left.set_ylabel('CDF (log)')
    ax_left.grid(True, which='both', alpha=0.3)

    # right tail centered at right_loss
    ax_right = fig.add_subplot(gs[1, 1])
    # Plot CDF (not 1-CDF)
    for line in lines:
        ax_right.plot(finite_losses, line['cdf'], linestyle=line['style'], color=line['color'], alpha=0.8)
    # Focus x-limits via adaptive log-gap band; fallback to default span
    right_win = window_from_metric(right_weighted, right_idx, right_mask if 'right_mask' in locals() else (avg_cdf > 0.5)) if finite_losses.size else None
    if right_win is not None:
        right_min, right_max = right_win
    else:
        right_min = right_loss - default_span
        right_max = right_loss + default_span
    # Asymmetric focus: on the right subplot, show more to the LEFT of the main point
    right_width = max(1e-12, right_max - right_min)
    right_min = right_loss - 0.8 * right_width
    right_max = right_loss + 0.2 * right_width
    if finite_losses.size:
        ax_right.set_xlim(max(float(finite_losses.min()), right_min), min(float(finite_losses.max()), right_max))
        # Concentrate Y-range: choose bounds based on 1-CDF within the window, aggregated over all lines
        mask_right = (finite_losses >= ax_right.get_xlim()[0]) & (finite_losses <= ax_right.get_xlim()[1])
        if np.any(mask_right) and lines:
            stacked = np.vstack([ln['cdf'][mask_right] for ln in lines])
            one_minus = 1.0 - stacked
            pos = one_minus[one_minus > 0.0]
            if pos.size:
                min_one = float(np.max([np.min(pos), 1e-16]))
                max_one = float(np.max(pos))
                y_low = max(0.0, 1.0 - 1.25 * max_one)
                y_high = min(1.0, 1.0 - 0.8 * min_one)
                if y_high > y_low:
                    ax_right.set_ylim(y_low, y_high)
    # Set primary y-axis scale to be logarithmic in (1 − CDF) while plotting CDF values
    # Use an increasing transform: z = -log10(1 - y), inverse: y = 1 - 10^{-z}
    def forward_cdf_to_log1mcdf(y):
        return -np.log10(np.maximum(1e-16, 1.0 - y))
    def inverse_log1mcdf_to_cdf(z):
        return 1.0 - np.power(10.0, -z)
    ax_right.set_yscale('function', functions=(forward_cdf_to_log1mcdf, inverse_log1mcdf_to_cdf))
    ax_right.set_title('Right extreme (CDF; scale: log10(1−CDF))')
    ax_right.set_xlabel('Privacy Loss')
    ax_right.set_ylabel('CDF')
    # Configure y-ticks at CDF = 1 - 10^{-k}
    try:
        y0, y1 = ax_right.get_ylim()
        ks = list(range(0, 13))  # 10^0 .. 10^-12
        tick_vals = [1.0 - 10.0**(-k) for k in ks]
        tick_pairs = [(k, y) for k, y in zip(ks, tick_vals) if y0 <= y <= y1]
        if tick_pairs:
            ax_right.set_yticks([y for _, y in tick_pairs])
            ax_right.set_yticklabels([rf'$1-10^{{-{k}}}$' for k, _ in tick_pairs])
    except Exception:
        pass
    ax_right.grid(True, which='both', alpha=0.3)

    fig.tight_layout()
    return fig

def create_epsilon_delta_plot(delta_values, versions, sigma: float, q: float, log_x_axis: bool, log_y_axis: bool, title_suffix: str = ''):
    """Plot epsilon ratios vs delta for each version relative to analytical ground-truth.

    Inputs:
    - delta_values: list/array of deltas
    - versions: list of dicts with fields: {'name': str, 'eps': List[float]}
    - title_suffix/log flags same as before
    Ground-truth is computed via analytic_subsampled_epsilon_for_delta using sigma,q parsed from title_suffix.
    """
    fig = plt.figure(figsize=(10, 6))

    delta_values = np.asarray(delta_values, dtype=np.float64)
    # Analytical ground-truth eps via closed form
    eps_GT = np.array([
        analytic_subsampled_epsilon_for_delta(sigma=sigma, q=q, delta=float(d), sensitivity=1.0)
        for d in delta_values
    ], dtype=np.float64)

    tiny = 1e-15
    colors = ['r', 'b', 'g', 'm']
    line_fn = plt.semilogx if log_x_axis else plt.plot

    # X scale
    if log_x_axis:
        plt.xscale('log')
        plt.xlabel('Epsilon (log scale)')
    else:
        plt.xlabel('Epsilon')

    # Y scale
    if log_y_axis:
        plt.yscale('log')
    plt.ylabel('Epsilon ratio (method / ground truth)')
    # Plot ratios
    valid_mask = eps_GT > tiny
    for idx, entry in enumerate(versions):
        name = entry.get('name', f'PMF {idx+1}')
        eps_v = np.asarray(entry.get('eps', []), dtype=np.float64)
        ratio_v = eps_v[valid_mask] / eps_GT[valid_mask]
        style = ['--', '-', '-.'][idx % 3]
        color = colors[idx % len(colors)]
        line_fn(eps_GT, ratio_v, linestyle=style, color=color, label=f'{name} / Analytical')

    title = 'Epsilon ratio (relative to analytical) vs. analytical epsilon'
    if title_suffix:
        title += f' — {title_suffix}'
    plt.title(title)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    return fig

