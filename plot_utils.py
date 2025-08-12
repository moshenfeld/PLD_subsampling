import os
import numpy as np
import matplotlib.pyplot as plt
from subsample_pld_pmf import dp_accounting_pmf_to_loss_probs
from pmf_compare import calc_W1_dist
from analytic_derivation import subsampled_gaussian_probabilities_from_losses


def ensure_plots_dir(path: str = "plots") -> str:
    os.makedirs(path, exist_ok=True)
    return path


def create_pmf_cdf_plot(
    pmfs: list,
    sigma: float,
    sampling_prob: float,
    title_suffix: str = '',
    method_labels: list | None = None,
):
    """
    Plot CDFs for a sequence of PLDs provided as (losses, probs) pairs.

    - Union the provided loss grids to form a common grid
    - Compute ground-truth probabilities on this grid using
      subsampled_gaussian_probabilities_from_losses
    - Plot the ground-truth CDF and each PMF's CDF
    - Compute W1 for each PMF relative to ground truth and show it in legend
    """
    # Union of all finite losses
    losses_list = []
    for losses, probs in pmfs:
        losses = np.asarray(losses, dtype=np.float64)
        finite_mask = np.isfinite(losses)
        losses_list.append(losses[finite_mask])
    union_losses = np.unique(np.concatenate(losses_list)) if losses_list else np.array([], dtype=np.float64)
    union_losses = np.sort(union_losses[np.isfinite(union_losses)])

    # Ground-truth probabilities on union grid
    gt_probs = subsampled_gaussian_probabilities_from_losses(
        sigma=sigma, sampling_prob=sampling_prob, losses=union_losses
    )
    gt_pair = (union_losses, gt_probs)

    # Figure and main panel
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])
    ax_main = fig.add_subplot(gs[0, :])

    # Build lines for plotting: ground-truth + each PMF
    gt_cdf = np.cumsum(gt_probs)
    lines = [
        {
            'label': 'Ground truth (analytic)',
            'cdf': gt_cdf,
            'color': 'k',
            'style': '-',
            'alpha': 0.9,
        }
    ]

    colors = ['b', 'r', 'g', 'm', 'c', 'y']
    for idx, (losses, probs) in enumerate(pmfs):
        losses = np.asarray(losses, dtype=np.float64)
        probs = np.asarray(probs, dtype=np.float64)
        pmf_map = dict(zip(losses, probs))
        grid_probs = np.array([pmf_map.get(x, 0.0) for x in union_losses])
        cdf_vals = np.cumsum(grid_probs)
        w1 = calc_W1_dist((union_losses, grid_probs), (union_losses, gt_probs))
        color = colors[idx % len(colors)]
        name = method_labels[idx] if (method_labels is not None and idx < len(method_labels)) else f'PMF {idx+1}'
        lines.append({
            'label': f'{name} (W1={w1:.3g})',
            'cdf': cdf_vals,
            'color': color,
            'style': '--',
            'alpha': 0.85,
        })

    # Plot all lines on main panel
    for line in lines:
        ax_main.plot(union_losses, line['cdf'], linestyle=line['style'], color=line['color'],
                     alpha=line['alpha'], label=line['label'])

    title = 'CDF Comparison vs Ground Truth'
    if title_suffix:
        title += f' — {title_suffix}'
    ax_main.set_title(title)
    ax_main.set_xlabel('Privacy Loss')
    ax_main.set_ylabel('CDF')
    ax_main.grid(True, alpha=0.3)
    ax_main.legend()
    ax_main.set_ylim(0.0, 1.0)

    # Determine focus centers via largest gap in log space on each side
    # Tail panels: compute focus windows using first PMF (if any) vs ground-truth
    finite_losses = union_losses
    if len(lines) > 1:
        our_cdf = lines[1]['cdf']  # first provided PMF
    else:
        our_cdf = np.zeros_like(gt_cdf)
    lib_cdf = lines[0]['cdf']  # ground truth
    avg_cdf = 0.5 * (our_cdf + lib_cdf)
    tiny = 1e-16
    if finite_losses.size:
        # Left: maximize |log(CDF_our) - log(CDF_lib)| for avg CDF <= 0.5
        left_mask = avg_cdf <= 0.5
        log_cdf_our = np.log(np.maximum(our_cdf, tiny))
        log_cdf_lib = np.log(np.maximum(lib_cdf, tiny))
        left_metric = np.abs(log_cdf_our - log_cdf_lib)
        left_weighted = np.where(left_mask, left_metric, -np.inf)

        # Right: maximize |log(1-CDF_our) - log(1-CDF_lib)| near the edge (avg CDF high)
        thresholds = [0.999, 0.99, 0.95, 0.9, 0.5]
        right_weighted = np.full_like(left_weighted, -np.inf)
        log_tail_our = np.log(np.maximum(1.0 - our_cdf, tiny))
        log_tail_lib = np.log(np.maximum(1.0 - lib_cdf, tiny))
        right_metric = np.abs(log_tail_our - log_tail_lib)
        for thr in thresholds:
            right_mask = avg_cdf >= thr
            if np.any(right_mask):
                cand = np.where(right_mask, right_metric, -np.inf)
                if np.isfinite(np.max(cand)) and np.max(cand) >= 0:
                    right_weighted = cand
                    break
        if not np.isfinite(np.max(right_weighted)):
            right_mask = avg_cdf > 0.5
            right_weighted = np.where(right_mask, right_metric, -np.inf)
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
        # Concentrate Y-range to the relevant CDF values within the left window
        mask_left = (finite_losses >= ax_left.get_xlim()[0]) & (finite_losses <= ax_left.get_xlim()[1])
        if np.any(mask_left):
            y_left_vals = np.concatenate([our_cdf[mask_left], lib_cdf[mask_left]])
            y_min = float(np.max([np.min(y_left_vals[y_left_vals > 0.0]) if np.any(y_left_vals > 0.0) else 1e-16, 1e-16]))
            y_max = float(np.max(y_left_vals))
            # Pad by a small factor on log scale
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
        # Concentrate Y-range: choose bounds based on 1-CDF within the window, but display CDF
        mask_right = (finite_losses >= ax_right.get_xlim()[0]) & (finite_losses <= ax_right.get_xlim()[1])
        if np.any(mask_right):
            cdf_vals = np.concatenate([our_cdf[mask_right], lib_cdf[mask_right]])
            one_minus = 1.0 - cdf_vals
            # define y-lims on CDF so that one_minus ranges are in view
            if np.any(one_minus > 0):
                min_one = float(np.max([np.min(one_minus[one_minus > 0.0]), 1e-16]))
                max_one = float(np.max(one_minus))
                # Map to CDF band [1 - 1.25*max_one, 1 - 0.8*min_one]
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

def create_epsilon_delta_plot(delta_values, eps_a, eps_r, eps_o, log_x_axis, log_y_axis, title_suffix: str = '', eps_analytic_pmf=None, method_labels: list | None = None, use_log_y: bool | None = None):
    """Create an epsilon ratio plot vs delta: method epsilon divided by ground truth epsilon (analytical).

    method_labels, if provided, should correspond to the numerator series order:
      [label_for_ref, label_for_ours, (optional) label_for_analytic_pmf]
    """
    fig = plt.figure(figsize=(10, 6))
    eps_a = np.asarray(eps_a, dtype=np.float64)
    eps_r = np.asarray(eps_r, dtype=np.float64)
    eps_o = np.asarray(eps_o, dtype=np.float64)
    tiny = 1e-15
    # Avoid division by zero; mark undefined ratios as NaN so they are not drawn
    ratio_ref = np.where(eps_a > tiny, eps_r / eps_a, np.nan)
    ratio_ours = np.where(eps_a > tiny, eps_o / eps_a, np.nan)
    if eps_analytic_pmf is not None:
        eps_analytic_pmf = np.asarray(eps_analytic_pmf, dtype=np.float64)
        ratio_anpmf = np.where(eps_a > tiny, eps_analytic_pmf / eps_a, np.nan)
    else:
        ratio_anpmf = None

    # X scale
    if log_x_axis:
        plt.xscale('log')
        plt.xlabel('Delta (log scale)')
    else:
        plt.xlabel('Delta')

    # Y scale (allow explicit override via use_log_y)
    y_log = use_log_y if use_log_y is not None else log_y_axis
    if y_log:
        plt.yscale('log')
    plt.ylabel('Epsilon ratio (method / ground truth)')

    # Reference line at 1
    try:
        yref = np.ones_like(delta_values, dtype=np.float64)
        plt.semilogx(delta_values, yref, 'k:', alpha=0.6, label='Baseline (1.0)') if log_x_axis else plt.plot(delta_values, yref, 'k:', alpha=0.6, label='Baseline (1.0)')
    except Exception:
        pass

    # Plot ratios
    line_fn = plt.semilogx if log_x_axis else plt.plot
    ref_label = (method_labels[0] if (method_labels and len(method_labels) >= 1) else 'Ref (Lib)') + ' / Analytical'
    ours_label = (method_labels[1] if (method_labels and len(method_labels) >= 2) else 'Ours') + ' / Analytical'
    line_fn(delta_values, ratio_ref, 'r--', label=ref_label)
    line_fn(delta_values, ratio_ours, 'b-', label=ours_label)
    if ratio_anpmf is not None:
        an_label_base = method_labels[2] if (method_labels and len(method_labels) >= 3) else 'Analytic PMF'
        line_fn(delta_values, ratio_anpmf, 'g-.', label=f'{an_label_base} / Analytical')

    title = 'Epsilon ratio vs Delta (relative to analytical)'
    if title_suffix:
        title += f' — {title_suffix}'
    plt.title(title)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    return fig

