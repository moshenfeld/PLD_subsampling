import os
import numpy as np
import matplotlib.pyplot as plt
from subsample_pld_pmf import dp_accounting_pmf_to_loss_probs
from pmf_compare import calc_W1_dist


def ensure_plots_dir(path: str = "plots") -> str:
    os.makedirs(path, exist_ok=True)
    return path


def create_pmf_cdf_plot(our_pmf, library_pmf, w1, title_suffix: str = ''):
    our_losses, our_probs = dp_accounting_pmf_to_loss_probs(our_pmf)
    lib_losses, lib_probs = dp_accounting_pmf_to_loss_probs(library_pmf)

    all_losses = np.unique(np.concatenate([our_losses, lib_losses]))
    finite_losses = np.sort(all_losses[np.isfinite(all_losses)])
    our_map = dict(zip(our_losses, our_probs))
    lib_map = dict(zip(lib_losses, lib_probs))
    our_grid = np.array([our_map.get(x, 0.0) for x in finite_losses])
    lib_grid = np.array([lib_map.get(x, 0.0) for x in finite_losses])
    our_cdf = np.cumsum(our_grid)
    lib_cdf = np.cumsum(lib_grid)

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])
    # main
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.plot(finite_losses, our_cdf, 'b-', label='Our CDF', alpha=0.8)
    ax_main.plot(finite_losses, lib_cdf, 'r--', label='Library CDF', alpha=0.8)
    title = f'CDF Comparison (W1={w1:.6g})'
    if title_suffix:
        title += f' — {title_suffix}'
    ax_main.set_title(title)
    ax_main.set_xlabel('Privacy Loss')
    ax_main.set_ylabel('CDF')
    ax_main.grid(True, alpha=0.3)
    ax_main.legend()
    ax_main.set_ylim(0.0, 1.0)

    # Determine focus centers via largest gap in log space on each side
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
    ax_left.plot(finite_losses, our_cdf, 'b-', alpha=0.8)
    ax_left.plot(finite_losses, lib_cdf, 'r--', alpha=0.8)
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
    ax_right.plot(finite_losses, our_cdf, 'b-', alpha=0.8)
    ax_right.plot(finite_losses, lib_cdf, 'r--', alpha=0.8)
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

def create_epsilon_delta_plot(delta_values, eps_a, eps_r, eps_o, log_x_axis, log_y_axis, title_suffix: str = ''):
    """Create an epsilon-vs-delta figure for analytical/ref/ours and return the figure."""
    fig = plt.figure(figsize=(10, 6))
    plt.semilogx(delta_values, eps_a, 'k-', label='Analytical')
    plt.semilogx(delta_values, eps_r, 'r--', label='Ref (Lib)')
    plt.semilogx(delta_values, eps_o, 'b-', label='Ours')
    if log_x_axis:
        plt.xscale('log')
        plt.xlabel('Delta (log scale)')
    else:
        plt.xlabel('Delta')
    if log_y_axis:
        plt.yscale('log')
        plt.ylabel('Epsilon (log)')
    else:
        plt.ylabel('Epsilon')
    title = 'Epsilon vs Delta'
    if title_suffix:
        title += f' — {title_suffix}'
    plt.title(title)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    return fig

