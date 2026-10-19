"""
plot_overview.py — Generate the main overview figure:
  Estimated vs Ground Truth target position.

  Layout: 2 rows (X-axis, Y-axis) × 3 columns (Patch, Chicken, Pork)
  Each subplot: Calibrated (blue) vs Harmonic Comb (orange)
  Perfect-prediction diagonal + ±20 mm shaded band in each panel.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from src.config import ALL_HARMONICS, CALIBRATION_TARGET_POS
from src.data import (
    load_dataset, get_aggregated_data, get_phase_matrix,
    get_available_target_positions,
)
from src.localize import CalibratedLocalizer, HarmonicCombinationLocalizer

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTDIR = REPO_ROOT / 'results'


def run_calibrated(df, axis, category):
    agg = get_aggregated_data(df, category, axis)
    if agg.empty:
        return []
    cal_data = get_phase_matrix(agg, CALIBRATION_TARGET_POS, ALL_HARMONICS)
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
        data = get_phase_matrix(agg, tp, ALL_HARMONICS)
        if data is None:
            continue
        try:
            est, _ = loc.localize(data['phases'], data['powers'])
            results.append((float(tp), float(est)))
        except Exception:
            continue
    return results


def run_harmonic_comb(df, axis, category):
    agg = get_aggregated_data(df, category, axis)
    if agg.empty:
        return []
    cal_790  = get_phase_matrix(agg, CALIBRATION_TARGET_POS, [790])
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
        d790  = get_phase_matrix(agg, tp, [790])
        d1700 = get_phase_matrix(agg, tp, [1700])
        if d790 is None or d1700 is None:
            continue
        try:
            est, _ = loc.localize(d790['phases'][:, 0], d1700['phases'][:, 0])
            results.append((float(tp), float(est)))
        except Exception:
            continue
    return results


def main():
    csv_path = str(REPO_ROOT / 'data' / 'Remix_Dataset.csv')
    print(f"Loading dataset from {csv_path} ...")
    df = load_dataset(csv_path)
    print(f"  {len(df):,} rows\n")

    categories  = ['Patch', 'Chicken', 'Pork']
    axes        = ['x', 'y']
    axis_labels = {'x': 'Lateral (X) Axis', 'y': 'Depth (Y) Axis'}

    algos = {
        'Calibrated':    {'fn': run_calibrated,    'color': '#1565C0', 'zorder': 4},
        'Harmonic Comb': {'fn': run_harmonic_comb, 'color': '#E65100', 'zorder': 3},
    }

    # ── Collect all results ────────────────────────────────────────
    data = {}  # (axis, cat, algo) -> [(gt, est)]
    for axis in axes:
        for cat in categories:
            print(f"  {cat}/{axis} ...", end='', flush=True)
            for algo_name, cfg in algos.items():
                res = cfg['fn'](df, axis, cat)
                data[(axis, cat, algo_name)] = res
                print(f"  {algo_name[:3]}={len(res)}", end='', flush=True)
            print()

    # ── Plot: 2 rows × 3 cols ─────────────────────────────────────
    fig, axes_arr = plt.subplots(
        2, 3, figsize=(13, 8.5),
        sharex=True, sharey=True,
    )
    fig.subplots_adjust(hspace=0.08, wspace=0.06)

    lim  = 200
    band = 20

    for ri, axis in enumerate(axes):
        for ci, cat in enumerate(categories):
            ax = axes_arr[ri][ci]

            # ±20 mm band
            ax.fill_between(
                [-lim, lim],
                [-lim - band, lim - band],
                [-lim + band, lim + band],
                color='#4CAF50', alpha=0.10, zorder=0,
            )

            # Perfect prediction diagonal
            ax.plot([-lim, lim], [-lim, lim],
                    color='#555', lw=0.8, ls='--', alpha=0.4, zorder=1)

            for algo_name, cfg in algos.items():
                pts = data.get((axis, cat, algo_name), [])
                if not pts:
                    continue
                gt      = np.array([p[0] for p in pts])
                est     = np.array([p[1] for p in pts])
                med_err = float(np.median(np.abs(est - gt)))

                ax.scatter(
                    gt, est, s=22,
                    color=cfg['color'], alpha=0.75,
                    edgecolors='white', linewidths=0.25,
                    zorder=cfg['zorder'],
                    label=f'{algo_name}  (med = {med_err:.0f} mm)',
                )

            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.12, lw=0.5)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
            ax.tick_params(labelsize=8)

            # Column title — top row only
            if ri == 0:
                ax.set_title(cat, fontsize=16, fontweight='bold', pad=6)

            # Row label — left column only
            if ci == 0:
                ax.set_ylabel(
                    f'{axis_labels[axis]}\nEstimated (mm)',
                    fontsize=12, labelpad=4,
                )

            # X-axis label — bottom row only
            if ri == len(axes) - 1:
                ax.set_xlabel('Ground Truth (mm)', fontsize=12)

            # Per-panel legend
            leg = ax.legend(
                fontsize=12, loc='upper left',
                framealpha=0.88, edgecolor='#ccc',
                handletextpad=0.4, borderpad=0.5,
                markerscale=1.2,
            )
            leg.get_frame().set_linewidth(0.5)

    fig.suptitle('Estimated vs Ground Truth Position',
                 fontsize=16, fontweight='bold', y=0.998)

    outpath = OUTDIR / 'overview_predictions.png'
    fig.savefig(outpath, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved {outpath}")


if __name__ == '__main__':
    main()
