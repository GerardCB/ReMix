"""
plot_overview.py — Generate the main overview figure:
  Estimated vs Ground Truth target position for all algorithms and tissues.

  - Rows: X-axis (lateral), Y-axis (depth)
  - Symbols: tissue type (Patch=circle, Chicken=triangle, Pork=square)
  - Colors: algorithm (Calibrated=blue, Harmonic Comb=orange, Ensemble=green)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.config import (
    ALL_HARMONICS, PAIR_A, CALIBRATION_TARGET_POS,
)
from src.data import (
    load_dataset, get_aggregated_data, get_phase_matrix,
    get_available_target_positions,
)
from src.localize import CalibratedLocalizer, HarmonicCombinationLocalizer
from experiments.report_freq_pairs import GeneralizedHarmonicCombLocalizer

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTDIR = REPO_ROOT / 'results'


def run_calibrated(df, axis, category):
    """Run calibrated localizer, return list of (gt, est)."""
    agg = get_aggregated_data(df, category, axis)
    if agg.empty:
        return []
    freqs = ALL_HARMONICS
    cal_data = get_phase_matrix(agg, CALIBRATION_TARGET_POS, freqs)
    if cal_data is None:
        return []
    loc = CalibratedLocalizer(
        cal_data['freqs'], cal_data['ant_pos'],
        dataset_axis=axis, category=category,
    )
    loc.calibrate(cal_data['phases'], CALIBRATION_TARGET_POS)
    results = []
    for tp in get_available_target_positions(agg):
        if tp == CALIBRATION_TARGET_POS:
            continue
        data = get_phase_matrix(agg, tp, freqs)
        if data is None:
            continue
        try:
            est, _ = loc.localize(data['phases'], data['powers'])
            results.append((tp, est))
        except Exception:
            continue
    return results


def run_harmonic_comb(df, axis, category):
    """Run harmonic combination localizer, return list of (gt, est)."""
    agg = get_aggregated_data(df, category, axis)
    if agg.empty:
        return []
    cal_790 = get_phase_matrix(agg, CALIBRATION_TARGET_POS, [790])
    cal_1700 = get_phase_matrix(agg, CALIBRATION_TARGET_POS, [1700])
    if cal_790 is None or cal_1700 is None:
        return []
    loc = HarmonicCombinationLocalizer(
        cal_790['ant_pos'], dataset_axis=axis, category=category,
    )
    loc.calibrate(cal_790['phases'][:, 0], cal_1700['phases'][:, 0])
    results = []
    for tp in get_available_target_positions(agg):
        if tp == CALIBRATION_TARGET_POS:
            continue
        d790 = get_phase_matrix(agg, tp, [790])
        d1700 = get_phase_matrix(agg, tp, [1700])
        if d790 is None or d1700 is None:
            continue
        try:
            est, _ = loc.localize(d790['phases'][:, 0], d1700['phases'][:, 0])
            results.append((tp, est))
        except Exception:
            continue
    return results


def run_ensemble(df, axis, category, n_top=5):
    """Run ensemble of top-5 HC pairs, return list of (gt, est)."""
    freq_csv = REPO_ROOT / 'results' / 'freq_pair_comparison.csv'
    if not freq_csv.exists():
        return []
    freq_df = pd.read_csv(freq_csv)
    mask = (freq_df['axis'] == axis) & (freq_df['category'] == category)
    sub = freq_df[mask].sort_values('median_error')
    if sub.empty:
        return []
    top_pairs = list(zip(sub['freq_a'].astype(int), sub['freq_b'].astype(int)))[:n_top]

    agg = get_aggregated_data(df, category, axis)
    if agg.empty:
        return []

    # Build and calibrate localizers for each pair
    localizers = []
    for fa, fb in top_pairs:
        cal_a = get_phase_matrix(agg, CALIBRATION_TARGET_POS, [fa])
        cal_b = get_phase_matrix(agg, CALIBRATION_TARGET_POS, [fb])
        if cal_a is None or cal_b is None:
            continue
        loc = GeneralizedHarmonicCombLocalizer(
            fa, fb, cal_a['ant_pos'], dataset_axis=axis, category=category,
        )
        loc.calibrate(cal_a['phases'][:, 0], cal_b['phases'][:, 0])
        localizers.append((fa, fb, loc))

    results = []
    for tp in get_available_target_positions(agg):
        if tp == CALIBRATION_TARGET_POS:
            continue
        estimates = []
        for fa, fb, loc in localizers:
            da = get_phase_matrix(agg, tp, [fa])
            db = get_phase_matrix(agg, tp, [fb])
            if da is None or db is None:
                continue
            try:
                est, _ = loc.localize(da['phases'][:, 0], db['phases'][:, 0])
                estimates.append(est)
            except Exception:
                continue
        if estimates:
            results.append((tp, float(np.median(estimates))))
    return results


def main():
    csv_path = str(REPO_ROOT / 'data' / 'Remix_Dataset.csv')
    print(f"Loading dataset from {csv_path} ...")
    df = load_dataset(csv_path)
    print(f"  {len(df):,} rows\n")

    categories = ['Patch', 'Chicken', 'Pork']
    axes = ['x', 'y']

    # ── Marker / color scheme ──────────────────────────────────────
    tissue_markers = {'Patch': 'o', 'Chicken': '^', 'Pork': 's'}
    tissue_sizes   = {'Patch': 38,  'Chicken': 42,  'Pork': 34}
    algo_colors    = {
        'Calibrated':    '#2196F3',   # blue
        'Harmonic Comb': '#FF9800',   # orange
        'Ensemble (5)':  '#4CAF50',   # green
    }

    # ── Collect all results ────────────────────────────────────────
    all_data = {}  # (axis, category, algo) -> [(gt, est), ...]
    for axis in axes:
        for cat in categories:
            print(f"  Running {cat}/{axis}...", end='', flush=True)
            cal_res = run_calibrated(df, axis, cat)
            hc_res = run_harmonic_comb(df, axis, cat)
            ens_res = run_ensemble(df, axis, cat)
            all_data[(axis, cat, 'Calibrated')] = cal_res
            all_data[(axis, cat, 'Harmonic Comb')] = hc_res
            all_data[(axis, cat, 'Ensemble (5)')] = ens_res
            print(f"  cal={len(cal_res)}, hc={len(hc_res)}, ens={len(ens_res)}")

    # ── Plot ───────────────────────────────────────────────────────
    fig, axes_arr = plt.subplots(1, 2, figsize=(14, 6.5))

    for ai, axis in enumerate(axes):
        ax = axes_arr[ai]
        dim = 'Lateral (X)' if axis == 'x' else 'Depth (Y)'

        for algo_name, color in algo_colors.items():
            for cat in categories:
                data = all_data.get((axis, cat, algo_name), [])
                if not data:
                    continue
                gt = np.array([d[0] for d in data])
                est = np.array([d[1] for d in data])
                marker = tissue_markers[cat]
                ms = tissue_sizes[cat]

                ax.scatter(gt, est, s=ms, marker=marker, color=color,
                           alpha=0.6, edgecolors='white', linewidths=0.3,
                           zorder=3)

        # Perfect prediction line
        ax.plot([-200, 200], [-200, 200], 'k--', lw=1.0, alpha=0.3, zorder=1)

        # ±20mm band
        ax.fill_between([-200, 200], [-220, 180], [-180, 220],
                         alpha=0.06, color='green', zorder=0)

        ax.set_xlim(-200, 200)
        ax.set_ylim(-200, 200)
        ax.set_aspect('equal')
        ax.set_xlabel('Ground Truth Position (mm)', fontsize=11)
        ax.set_ylabel('Estimated Position (mm)', fontsize=11)
        ax.set_title(f'{dim} Axis', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.15)

        # Add median error annotations
        annot_lines = []
        for algo_name in algo_colors:
            for cat in categories:
                data = all_data.get((axis, cat, algo_name), [])
                if data:
                    errs = [abs(d[1] - d[0]) for d in data]
                    med = np.median(errs)
                    annot_lines.append(f"{cat[:3]}/{algo_name[:3]}: {med:.0f}")

    # ── Combined legend ────────────────────────────────────────────
    # Algorithm colors
    algo_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
               markersize=9, label=name, markeredgecolor='gray',
               markeredgewidth=0.5)
        for name, c in algo_colors.items()
    ]
    # Tissue markers
    tissue_handles = [
        Line2D([0], [0], marker=m, color='w', markerfacecolor='gray',
               markersize=9, label=cat, markeredgecolor='dimgray',
               markeredgewidth=0.5)
        for cat, m in tissue_markers.items()
    ]
    # Band
    band_handle = [
        Line2D([0], [0], color='green', alpha=0.3, lw=6, label='±20 mm band')
    ]

    all_handles = algo_handles + tissue_handles + band_handle
    fig.legend(handles=all_handles, loc='lower center', ncol=7,
               fontsize=9, frameon=True, fancybox=True,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    outpath = OUTDIR / 'overview_predictions.png'
    fig.savefig(outpath, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved {outpath}")


if __name__ == '__main__':
    main()
