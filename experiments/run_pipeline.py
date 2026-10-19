"""
remix_pipeline.py — Evaluation pipeline with tissue-aware localization.

Implements the ReMix (SIGCOMM 2018) localization approach, including:
  - Tissue-aware effective distances (refraction + wavelength scaling)
  - Coarse-to-fine phase-ambiguity resolution on depth axis
  - Harmonic-combination trick (Eq. 14) for improved robustness
  - TDOA baseline for lateral axis

Usage:
  python remix_pipeline.py --csv Remix_Dataset.csv --axis y
  python remix_pipeline.py --csv Remix_Dataset.csv --axis x
  python remix_pipeline.py --csv Remix_Dataset.csv --axis y --approach calibrated harmonic_comb tdoa
"""

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.config import (
    PAIR_A, PAIR_B, ALL_HARMONICS,
    CALIBRATION_TARGET_POS, REFERENCE_DEPTH,
    TISSUE_PROFILES, get_tissue_wall, ALPHA,
    C, FREQ_TO_HZ, TX_POS, rx_positions, target_position,
)
from src.data import (
    load_dataset, get_aggregated_data, get_phase_matrix,
    get_available_target_positions,
)
from src.localize import (
    CalibratedLocalizer, GridSearchLocalizer,
    TDOALocalizer, HarmonicCombinationLocalizer,
    per_frequency_grid, TissueModel, phase_cost,
)

warnings.filterwarnings('ignore', category=RuntimeWarning)

FREQ_SETS = {
    'pair_a':       PAIR_A,           # [790, 1700]
    'pair_b':       PAIR_B,           # [910, 1700]
    'all':          ALL_HARMONICS,    # all 7 frequencies
    'close_low':    [790, 910],       # coarse-only, most unambiguous
    'close_high':   [1700, 1780],     # coarse-only, high precision
    'low3':         [790, 910, 1700], # 2 low for coarse + 1 high for precision
    'second_order': [1660, 1700, 1740],
}


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluators
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_calibrated(agg_df, axis, frequencies, category,
                        tp_min=None, tp_max=None):
    """Calibrated localizer with tissue model.

    Optionally restrict the set of target positions to the closed
    interval [tp_min, tp_max] (mm) if either bound is provided.
    """
    target_positions = get_available_target_positions(agg_df)
    if tp_min is not None or tp_max is not None:
        target_positions = [tp for tp in target_positions
                            if (tp_min is None or tp >= tp_min)
                            and (tp_max is None or tp <= tp_max)]
    cal_data = get_phase_matrix(agg_df, CALIBRATION_TARGET_POS, frequencies)
    if cal_data is None:
        return []

    localizer = CalibratedLocalizer(
        cal_data['freqs'], cal_data['ant_pos'],
        dataset_axis=axis, category=category,
    )
    localizer.calibrate(cal_data['phases'], CALIBRATION_TARGET_POS)

    results = []
    for tp in target_positions:
        if tp == CALIBRATION_TARGET_POS:
            continue
        data = get_phase_matrix(agg_df, tp, frequencies)
        if data is None or len(data['freqs']) < 2:
            continue
        try:
            est_tp, cost = localizer.localize(data['phases'], data['powers'])
        except Exception as e:
            print(f"\n    ✗ tp={tp}: {e}")
            continue
        results.append(_make_result(tp, est_tp, cost))
    return results


def evaluate_calibrated_freespace(agg_df, axis, frequencies,
                                    tp_min=None, tp_max=None):
    """Calibrated localizer WITHOUT tissue model (Patch/free-space for all)."""
    return evaluate_calibrated(agg_df, axis, frequencies,
                                category='Patch',
                                tp_min=tp_min, tp_max=tp_max)


def evaluate_grid_search(agg_df, axis, frequencies, category,
                         tp_min=None, tp_max=None):
    target_positions = get_available_target_positions(agg_df)
    if tp_min is not None or tp_max is not None:
        target_positions = [tp for tp in target_positions
                            if (tp_min is None or tp >= tp_min)
                            and (tp_max is None or tp <= tp_max)]
    cal_data = get_phase_matrix(agg_df, CALIBRATION_TARGET_POS, frequencies)
    if cal_data is None:
        return []

    results = []
    for tp in target_positions:
        if tp == CALIBRATION_TARGET_POS:
            continue
        data = get_phase_matrix(agg_df, tp, frequencies)
        if data is None or len(data['freqs']) < 2:
            continue
        localizer = GridSearchLocalizer(
            data['freqs'], data['ant_pos'],
            dataset_axis=axis, category=category,
        )
        try:
            est_tp, cost = localizer.localize(
                data['phases'], cal_data['phases'],
                CALIBRATION_TARGET_POS, step=1.0)
        except Exception:
            continue
        results.append(_make_result(tp, est_tp, cost))
    return results


def evaluate_tdoa(agg_df, axis, frequencies, category, method='de',
                  tp_min=None, tp_max=None):
    target_positions = get_available_target_positions(agg_df)
    if tp_min is not None or tp_max is not None:
        target_positions = [tp for tp in target_positions
                            if (tp_min is None or tp >= tp_min)
                            and (tp_max is None or tp <= tp_max)]
    results = []
    for tp in target_positions:
        if tp == CALIBRATION_TARGET_POS:
            continue  # Keep length consistent with other evaluators
            
        data = get_phase_matrix(agg_df, tp, frequencies)
        if data is None or len(data['freqs']) < 2:
            continue
        localizer = TDOALocalizer(
            data['freqs'], data['ant_pos'],
            dataset_axis=axis, category=category,
        )
        try:
            est_tp, cost = localizer.localize(data['phases'], method=method)
        except Exception:
            continue
        results.append(_make_result(tp, est_tp, cost))
    return results


def evaluate_harmonic_comb(agg_df, axis, category, frequencies,
                           tp_min=None, tp_max=None):
    """
    Harmonic-combination approach (ReMix Eq. 14).
    Dynamically uses the first two frequencies passed.
    """
    target_positions = get_available_target_positions(agg_df)
    if tp_min is not None or tp_max is not None:
        target_positions = [tp for tp in target_positions
                            if (tp_min is None or tp >= tp_min)
                            and (tp_max is None or tp <= tp_max)]
                            
    if len(frequencies) < 2:
        return []
        
    f1, f2 = frequencies[0], frequencies[1]

    # Get calibration data for both required frequencies
    cal_f1 = get_phase_matrix(agg_df, CALIBRATION_TARGET_POS, [f1])
    cal_f2 = get_phase_matrix(agg_df, CALIBRATION_TARGET_POS, [f2])
    
    if cal_f1 is None or cal_f2 is None:
        return []

    localizer = HarmonicCombinationLocalizer(
        cal_f1['ant_pos'], dataset_axis=axis, category=category,
    )
    localizer.calibrate(
        cal_f1['phases'][:, 0], cal_f2['phases'][:, 0],
        CALIBRATION_TARGET_POS,
    )

    results = []
    for tp in target_positions:
        if tp == CALIBRATION_TARGET_POS:
            continue
        data_f1 = get_phase_matrix(agg_df, tp, [f1])
        data_f2 = get_phase_matrix(agg_df, tp, [f2])
        if data_f1 is None or data_f2 is None:
            continue
        try:
            est_tp, cost = localizer.localize(
                data_f1['phases'][:, 0], data_f2['phases'][:, 0])
        except Exception as e:
            print(f"\n    ✗ tp={tp}: {e}")
            continue
        results.append(_make_result(tp, est_tp, cost))
    return results


def _make_result(gt_tp, est_tp, cost):
    return {
        'target_pos': gt_tp,
        'est_target_pos': est_tp,
        'error': abs(est_tp - gt_tp),
        'signed_error': est_tp - gt_tp,
        'cost': cost,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Diagnostic: per-frequency phase curves
# ══════════════════════════════════════════════════════════════════════════════
def plot_phase_diagnostic(agg_df, axis, category, outdir,
                          tp_min=None, tp_max=None):
    """
    Plot measured phase vs target position for each frequency and antenna.
    Overlays the model prediction.  This reveals which freq/antenna combos
    have clean data vs noise or model mismatch.
    """
    from src.data import get_available_target_positions, get_phase_matrix

    target_positions = get_available_target_positions(agg_df)
    if tp_min is not None or tp_max is not None:
        target_positions = [tp for tp in target_positions
                            if (tp_min is None or tp >= tp_min)
                            and (tp_max is None or tp <= tp_max)]
    freqs = ALL_HARMONICS

    cal_data = get_phase_matrix(agg_df, CALIBRATION_TARGET_POS, freqs)
    if cal_data is None:
        return

    tm = TissueModel(category)
    rx_pos = rx_positions()
    tx_pos = TX_POS.copy()
    cal_xy = target_position(CALIBRATION_TARGET_POS, axis)
    cal_D = tm.effective_round_trip(cal_xy, tx_pos, rx_pos)

    n_freq = len(cal_data['freqs'])
    n_ant = len(cal_data['ant_pos'])

    fig, axes = plt.subplots(n_freq, n_ant, figsize=(3.5 * n_ant, 3 * n_freq),
                             squeeze=False, sharex=True)
    fig.suptitle(f'Phase diagnostic: {category} / axis={axis}\n'
                 f'Δφ(measured) vs Δφ(model) relative to TP=0',
                 fontsize=16, y=1.01)

    freq_hz = np.array([FREQ_TO_HZ[f] for f in cal_data['freqs']])

    # Model predictions
    model_tp = np.arange(-180, 181, 1.0)
    model_dphi = np.zeros((len(model_tp), n_ant, n_freq))
    for i, tp in enumerate(model_tp):
        dev = target_position(tp, axis)
        D = tm.effective_round_trip(dev, tx_pos, rx_pos)
        dD = D - cal_D
        for j in range(n_freq):
            model_dphi[i, :, j] = np.degrees(-2 * np.pi * freq_hz[j] / C * dD)

    # Measured data
    for fi in range(n_freq):
        for ai in range(n_ant):
            ax = axes[fi][ai]
            meas_dp = []
            valid_tp = []
            for tp in target_positions:
                data = get_phase_matrix(agg_df, tp, [cal_data['freqs'][fi]])
                if data is None:
                    continue
                cal_ph = cal_data['phases'][ai, fi]
                meas_ph = data['phases'][ai, 0]
                # Wrap to [-180, 180]
                dp = ((meas_ph - cal_ph + 180) % 360) - 180
                meas_dp.append(dp)
                valid_tp.append(tp)

            # Plot model
            model_dp_wrapped = ((model_dphi[:, ai, fi] + 180) % 360) - 180
            ax.plot(model_tp, model_dp_wrapped, 'r-', lw=0.6, alpha=0.5,
                    label='model')
            # Plot measured
            ax.scatter(valid_tp, meas_dp, s=6, alpha=0.7, zorder=3)

            ax.set_ylim(-190, 190)
            ax.grid(True, alpha=0.15)
            if fi == 0:
                ax.set_title(f'Ant {cal_data["ant_pos"][ai]:.0f}mm',
                             fontsize=12)
            if ai == 0:
                ax.set_ylabel(f'{cal_data["freqs"][fi]} MHz\nΔφ (°)',
                              fontsize=12)
            if fi == n_freq - 1:
                ax.set_xlabel('Target Pos (mm)')

    plt.tight_layout()
    fig.savefig(outdir / f'phase_diag_{category}_{axis}.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  Diagnostic: cost landscape
# ══════════════════════════════════════════════════════════════════════════════
def plot_cost_landscape(agg_df, axis, category, outdir, n_examples=4,
                        tp_min=None, tp_max=None):
    """
    Plot the cost landscape for a few target positions.
    Shows per-frequency and joint cost to diagnose ambiguity.
    """
    target_positions = get_available_target_positions(agg_df)
    if tp_min is not None or tp_max is not None:
        target_positions = [tp for tp in target_positions
                            if (tp_min is None or tp >= tp_min)
                            and (tp_max is None or tp <= tp_max)]
    freqs = ALL_HARMONICS
    cal_data = get_phase_matrix(agg_df, CALIBRATION_TARGET_POS, freqs)
    if cal_data is None:
        return

    tm = TissueModel(category)
    rx_pos = rx_positions()
    tx_pos = TX_POS.copy()
    cal_xy = target_position(CALIBRATION_TARGET_POS, axis)
    cal_D = tm.effective_round_trip(cal_xy, tx_pos, rx_pos)
    freq_hz = np.array([FREQ_TO_HZ[f] for f in cal_data['freqs']])
    cal_rad = np.deg2rad(cal_data['phases'])

    # Pick a few example positions spread across the range
    idx = np.linspace(0, len(target_positions) - 1, n_examples, dtype=int)
    example_tps = target_positions[idx]

    grid = np.arange(-200, 201, 1.0)

    fig, axes = plt.subplots(n_examples, 2, figsize=(14, 3.5 * n_examples),
                             squeeze=False)
    fig.suptitle(f'Cost landscape: {category} / axis={axis}', fontsize=16)

    for row, true_tp in enumerate(example_tps):
        data = get_phase_matrix(agg_df, true_tp, freqs)
        if data is None:
            continue

        meas_rad = np.deg2rad(data['phases'])

        # Per-frequency costs
        per_freq = per_frequency_grid(
            grid, axis, freq_hz, rx_pos, tx_pos, cal_D,
            meas_rad, cal_rad, tm)

        # Joint cost
        joint_costs = per_freq.sum(axis=1)

        # Left: per-frequency
        ax = axes[row][0]
        for j in range(len(cal_data['freqs'])):
            normalized = per_freq[:, j] / per_freq[:, j].max()
            ax.plot(grid, normalized, lw=0.8, alpha=0.7,
                    label=f'{cal_data["freqs"][j]}')
        ax.axvline(true_tp, color='k', ls='--', lw=1.5, alpha=0.7,
                   label=f'truth={true_tp:.0f}')
        ax.set_ylabel('Normalized cost')
        ax.set_title(f'TP={true_tp:.0f}mm — per frequency')
        ax.legend(fontsize=12, ncol=4)
        ax.grid(True, alpha=0.15)

        # Right: joint
        ax = axes[row][1]
        ax.plot(grid, joint_costs, 'b-', lw=1)
        ax.axvline(true_tp, color='k', ls='--', lw=1.5, alpha=0.7,
                   label=f'truth={true_tp:.0f}')
        best_est = grid[np.argmin(joint_costs)]
        ax.axvline(best_est, color='r', ls=':', lw=1.5, alpha=0.7,
                   label=f'est={best_est:.0f}')
        ax.set_ylabel('Joint cost')
        ax.set_title(f'TP={true_tp:.0f}mm — joint (est={best_est:.0f}mm, '
                     f'err={abs(best_est-true_tp):.0f}mm)')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.15)

    plt.tight_layout()
    fig.savefig(outdir / f'cost_landscape_{category}_{axis}.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_estimation(all_results, axis, outdir):
    """Estimated Target_Pos vs ground truth."""
    n = len(all_results)
    if n == 0:
        return
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows),
                             squeeze=False)

    for idx, (label, results) in enumerate(all_results.items()):
        ax = axes[idx // cols][idx % cols]
        if not results:
            ax.set_title(f'{label} — no data')
            continue
        df = pd.DataFrame(results)

        ax.scatter(df['target_pos'], df['est_target_pos'],
                   s=18, alpha=0.7, zorder=3)

        lo, hi = -200, 200
        ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.4, label='perfect')

        if len(df) > 2:
            m, b = np.polyfit(df['target_pos'], df['est_target_pos'], 1)
            x_fit = np.linspace(lo, hi, 100)
            ax.plot(x_fit, m * x_fit + b, 'r-', lw=1, alpha=0.6,
                    label=f'fit: y={m:.2f}x{b:+.0f}')

        mean_err = df['error'].mean()
        med_err = df['error'].median()
        dim = 'lateral' if axis == 'x' else 'depth'
        ax.set_xlabel(f'GT Target_Pos (mm) [{dim}]')
        ax.set_ylabel(f'Estimated Target_Pos (mm)')
        ax.set_title(f'{label}\nmean={mean_err:.0f}mm  med={med_err:.0f}mm')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal')

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.tight_layout()
    fig.savefig(outdir / 'estimation.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_error_vs_position(all_results, axis, outdir):
    n = len(all_results)
    if n == 0:
        return
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows),
                             squeeze=False)

    for idx, (label, results) in enumerate(all_results.items()):
        ax = axes[idx // cols][idx % cols]
        if not results:
            ax.set_title(label)
            continue
        df = pd.DataFrame(results)
        ax.plot(df['target_pos'], df['error'], 'o-', ms=3, lw=0.8, alpha=0.8)
        ax.plot(df['target_pos'], df['signed_error'], 'x-', ms=3, lw=0.5,
                alpha=0.5, label='signed error')
        ax.axhline(0, color='k', lw=0.5, alpha=0.3)
        ax.set_xlabel('Target_Pos (mm)')
        ax.set_ylabel('Error (mm)')
        ax.set_title(label)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.25)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.tight_layout()
    fig.savefig(outdir / 'error_vs_position.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_cdf(all_results, outdir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, results in all_results.items():
        if not results:
            continue
        errors = sorted([r['error'] for r in results])
        cdf = np.arange(1, len(errors) + 1) / len(errors)
        ax.plot(errors, cdf, '-', lw=1.5, label=label)

    ax.set_xlabel('Localization Error (mm)')
    ax.set_ylabel('CDF')
    ax.set_title('Cumulative Distribution of Localization Error')
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.25)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(outdir / 'error_cdf.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_summary_bars(all_results, outdir):
    labels, means, medians, p90s = [], [], [], []
    for label, results in all_results.items():
        if not results:
            continue
        errs = [r['error'] for r in results]
        labels.append(label)
        means.append(np.mean(errs))
        medians.append(np.median(errs))
        p90s.append(np.percentile(errs, 90))

    if not labels:
        return

    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.8), 5))
    ax.bar(x - w, means,   w, label='Mean',   alpha=0.85)
    ax.bar(x,     medians, w, label='Median', alpha=0.85)
    ax.bar(x + w, p90s,    w, label='P90',    alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=12)
    ax.set_ylabel('Localization Error (mm)')
    ax.set_title('Localization Error Summary')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.25)
    plt.tight_layout()
    fig.savefig(outdir / 'error_summary.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  Summary
# ══════════════════════════════════════════════════════════════════════════════
def plot_2d_predictions(all_results_x, all_results_y, outdir):
    """
    Plot model predictions in 2D space for both x and y axes.
    Each point is colored by its ground truth position.
    Helps visualize systematic biases (e.g., always predicting center).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    
    # Collect all unique approach labels across both axes
    all_approaches = set(all_results_x.keys()) | set(all_results_y.keys())
    
    for idx, (ax, results_dict, axis_name) in enumerate([
        (axes[0], all_results_x, 'X-axis'),
        (axes[1], all_results_y, 'Y-axis'),
    ]):
        if not results_dict:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            continue
        
        for label, results in results_dict.items():
            if not results:
                continue
            
            df_res = pd.DataFrame(results)
            scatter = ax.scatter(df_res['target_pos'], df_res['est_target_pos'],
                                 s=30, alpha=0.6, label=label)
        
        # Perfect prediction line
        lo, hi = -200, 200
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.3)
        
        ax.set_xlabel('Ground Truth Target Pos (mm)', fontsize=14)
        ax.set_ylabel('Estimated Target Pos (mm)', fontsize=14)
        ax.set_title(f'{axis_name} Dataset', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        if idx == 1:
            ax.legend(fontsize=12, loc='upper left')
    
    plt.tight_layout()
    fig.savefig(outdir / 'predictions_2d.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def print_summary(all_results, axis):
    dim = 'lateral' if axis == 'x' else 'depth'
    header = (f"{'Configuration':<55}  {'N':>4}  {'Mean':>7}  {'Med':>7}"
              f"  {'P90':>7}  {'<20mm':>6}  {'<50mm':>6}")
    print(f'\n  Estimating Target_Pos ({dim} axis) in mm')
    print('═' * len(header))
    print(header)
    print('─' * len(header))
    for label, results in all_results.items():
        if not results:
            print(f"{label:<55}  {'—':>4}")
            continue
        errs = np.array([r['error'] for r in results])
        n = len(errs)
        print(f"{label:<55}  {n:4d}  "
              f"{errs.mean():7.1f}  "
              f"{np.median(errs):7.1f}  "
              f"{np.percentile(errs, 90):7.1f}  "
              f"{np.mean(errs < 20) * 100:5.0f}%  "
              f"{np.mean(errs < 50) * 100:5.0f}%")
    print('═' * len(header))


def save_results_csv(all_results, outdir):
    rows = []
    for label, results in all_results.items():
        for r in results:
            r2 = r.copy()
            r2['config'] = label
            rows.append(r2)
    if rows:
        pd.DataFrame(rows).to_csv(outdir / 'results.csv', index=False)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='ReMix Localization Pipeline (tissue-aware)')
    parser.add_argument('--csv', required=True)
    parser.add_argument('--axis', default=None, choices=['x', 'y'],
                        help='Evaluate on x or y axis. If not specified, evaluates both.')
    parser.add_argument('--categories', nargs='+',
                        default=['Patch', 'Chicken', 'Pork'])
    parser.add_argument('--freqs', nargs='+',
                        default=['pair_a', 'all'],
                        choices=list(FREQ_SETS.keys()))
    parser.add_argument('--approach', nargs='+',
                        default=['calibrated'],
                        choices=['calibrated', 'calibrated_freespace',
                                 'grid_search', 'tdoa', 'harmonic_comb'])
    parser.add_argument('--diagnose', action='store_true',
                        help='Generate per-frequency phase and cost diagnostics')
    parser.add_argument('--method', default='de',
                        choices=['de', 'multistart'])
    parser.add_argument('--tp-range', nargs=2, type=float, metavar=('MIN','MAX'),
                        help='Restrict evaluation to targets between MIN and MAX (mm)')
    parser.add_argument('--output', default='results')
    args = parser.parse_args()

    # If axis not specified, evaluate both
    axes_to_eval = [args.axis] if args.axis else ['x', 'y']

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    tp_min, tp_max = (None, None)
    if args.tp_range:
        tp_min, tp_max = args.tp_range
        print(f"  Restricting target positions to [{tp_min}, {tp_max}] mm")

    print(f"Loading dataset from {args.csv} ...")
    df = load_dataset(args.csv)
    print(f"  {len(df):,} rows")
    print(f"  Categories : {sorted(df['Category'].unique())}")
    print(f"  Frequencies: {sorted(df['Frequency'].unique())} MHz")

    print(f"\n  Dataset 'x' = LATERAL axis (perpendicular to beam)")
    print(f"  Target depth fixed at 850 mm")
    print(f"  Dataset 'y' = DEPTH axis (along beam)")
    print(f"  Depth range: 640 mm to 1000 mm (centre 820 mm)")

    # Print tissue model info
    for cat in args.categories:
        wall = get_tissue_wall(cat)
        layers = TISSUE_PROFILES.get(cat, [])
        if layers:
            desc = ' + '.join(f'{t:.0f}mm {m} (α={ALPHA[m]:.2f})'
                              for m, t in layers)
            print(f"  Tissue [{cat}]: wall={wall:.0f}mm = {desc}")
        else:
            print(f"  Tissue [{cat}]: free space")

    all_results = {}
    all_results_by_axis = {'x': {}, 'y': {}}

    # Evaluate on each axis
    for axis in axes_to_eval:
        for category in args.categories:
            agg = get_aggregated_data(df, category, axis)
            if agg.empty:
                print(f"\n  ⚠ No data for {category}/{axis}")
                continue

            n_pos = len(get_available_target_positions(agg))
            print(f"\n{'━' * 70}")
            print(f"  {category} / axis={axis}  ({n_pos} target positions)")
            print(f"{'━' * 70}")

            for freq_label in args.freqs:
                # decide whether to run this freq label for each approach
                freqs = FREQ_SETS[freq_label]

                for approach in args.approach:

                    tag = f"{category}/{axis}  {freq_label:<12s}  {approach}"
                    print(f"  ▸ {tag}", end='', flush=True)

                    if approach == 'calibrated':
                        res = evaluate_calibrated(agg, axis, freqs, category,
                                                  tp_min=tp_min, tp_max=tp_max)

                    elif approach == 'calibrated_freespace':
                        res = evaluate_calibrated_freespace(agg, axis, freqs,
                                                            tp_min=tp_min, tp_max=tp_max)

                    elif approach == 'grid_search':
                        res = evaluate_grid_search(agg, axis, freqs, category,
                                                   tp_min=tp_min, tp_max=tp_max)

                    elif approach == 'tdoa':
                        res = evaluate_tdoa(agg, axis, freqs, category,
                                            method=args.method,
                                            tp_min=tp_min, tp_max=tp_max)

                    elif approach == 'harmonic_comb':
                        res = evaluate_harmonic_comb(agg, axis, category, freqs,
                                                     tp_min=tp_min, tp_max=tp_max)

                    else:
                        res = []

                    all_results[tag] = res
                    all_results_by_axis[axis][tag] = res

                    if res:
                        errs = [r['error'] for r in res]
                        print(f"  → {len(res)} pts  |  "
                              f"Mean = {np.mean(errs):.1f} mm  |  "
                              f"Median = {np.median(errs):.1f} mm")
                    else:
                        print("  → no results")

            # Diagnostics
            if args.diagnose:
                print(f"  📊 Generating diagnostics for {category}/{axis}...")
                plot_phase_diagnostic(agg, axis, category, outdir,
                                      tp_min=tp_min, tp_max=tp_max)
                plot_cost_landscape(agg, axis, category, outdir,
                                    tp_min=tp_min, tp_max=tp_max)

    # Print summaries for each axis
    for axis in axes_to_eval:
        print_summary({k: v for k, v in all_results.items() if f'/{axis}  ' in k}, axis)
    
    save_results_csv(all_results, outdir)

    if any(v for v in all_results.values()):
        for axis in axes_to_eval:
            axis_results = {k: v for k, v in all_results.items() if f'/{axis}  ' in k}
            plot_estimation(axis_results, axis, outdir)
            plot_error_vs_position(axis_results, axis, outdir)
        
        plot_cdf(all_results, outdir)
        plot_summary_bars(all_results, outdir)
        
        # Generate 2D prediction plot if we have both axes
        if all_results_by_axis['x'] and all_results_by_axis['y']:
            plot_2d_predictions(all_results_by_axis['x'],
                                all_results_by_axis['y'], outdir)
        
        print(f"\nPlots saved to {outdir.resolve()}/")


if __name__ == '__main__':
    main()