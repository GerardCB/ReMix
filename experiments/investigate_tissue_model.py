"""
investigate_tissue_model.py — Understand why the tissue model has no effect.

The grid search over tissue thickness found ~zero sensitivity: all 694 configs
produced nearly identical errors. This script investigates WHY by:

1. Quantifying the actual phase difference introduced by the tissue model
   vs. free-space for different tissue thicknesses.
2. Comparing model-predicted vs measured phase to detect systematic offsets.
3. Testing whether ignoring the tissue model entirely (using free-space for
   all categories) changes results.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    C, FREQ_TO_HZ, TX_POS, ALL_HARMONICS,
    rx_positions, target_position,
    DEPTH_SEARCH, LATERAL_SEARCH, CALIBRATION_TARGET_POS,
)
from src.data import (
    load_dataset, get_aggregated_data, get_phase_matrix,
    get_available_target_positions,
)
from src.localize import (
    TissueModel, CalibratedLocalizer, free_space_round_trip,
)

OUTDIR = Path(__file__).resolve().parent.parent / 'results' / 'tissue_investigation'
OUTDIR.mkdir(parents=True, exist_ok=True)


def compare_effective_distances():
    """
    Compare effective round-trip distances for free-space vs tissue models.
    Show how much the tissue correction actually changes the distance estimate.
    """
    rx_pos = rx_positions()
    tx_pos = TX_POS.copy()

    categories = ['Patch', 'Chicken', 'Pork']
    models = {cat: TissueModel(cat) for cat in categories}

    # Test across a range of target positions
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Effective Round-Trip Distance: Free-Space vs Tissue Model',
                 fontsize=18, fontweight='bold')

    for ai, axis in enumerate(['x', 'y']):
        tps = np.arange(-180, 181, 10)
        dim = 'Lateral (X)' if axis == 'x' else 'Depth (Y)'

        for ci, category in enumerate(categories):
            ax = axes[ai][ci]

            # For center antenna
            for ant_idx, ant_label in [(0, 'ant=-150'), (2, 'ant=0'), (4, 'ant=+150')]:
                d_free = []
                d_tissue = []

                for tp in tps:
                    xy = target_position(tp, axis)
                    D_f = free_space_round_trip(xy, tx_pos, rx_pos)
                    D_t = models[category].effective_round_trip(xy, tx_pos, rx_pos)
                    d_free.append(D_f[ant_idx])
                    d_tissue.append(D_t[ant_idx])

                d_free = np.array(d_free)
                d_tissue = np.array(d_tissue)
                delta = (d_tissue - d_free) * 1000  # mm

                ax.plot(tps, delta, label=ant_label, lw=1.5)

            ax.set_xlabel(f'Target Pos (mm)')
            ax.set_ylabel('D_tissue - D_freespace (mm)')
            ax.set_title(f'{category} / {dim}', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.15)
            ax.legend(fontsize=14)
            ax.axhline(0, color='gray', ls='--', alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTDIR / 'distance_difference.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved distance_difference.png")


def compare_phase_impact():
    """
    Show the phase impact of tissue correction at each frequency.
    Compute: Δφ = -2πf/c × (D_tissue - D_free) in degrees.
    This is the TOTAL phase correction the tissue model applies.
    """
    rx_pos = rx_positions()
    tx_pos = TX_POS.copy()

    categories = ['Chicken', 'Pork']
    models = {cat: TissueModel(cat) for cat in categories}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Phase Correction from Tissue Model (center antenna)',
                 fontsize=18, fontweight='bold')

    freqs = ALL_HARMONICS
    freq_hz = np.array([FREQ_TO_HZ[f] for f in freqs])
    ant_idx = 2  # center antenna

    for ai, axis in enumerate(['x', 'y']):
        tps = np.arange(-180, 181, 5)
        dim = 'X' if axis == 'x' else 'Y'

        for ci, category in enumerate(categories):
            ax = axes[ai][ci]

            for fi, (f, f_hz) in enumerate(zip(freqs, freq_hz)):
                dphi_deg = []
                for tp in tps:
                    xy = target_position(tp, axis)
                    D_f = free_space_round_trip(xy, tx_pos, rx_pos)[ant_idx]
                    D_t = models[category].effective_round_trip(
                        xy, tx_pos, rx_pos)[ant_idx]
                    delta_D = D_t - D_f
                    dphi = np.degrees(-2 * np.pi * f_hz / C * delta_D)
                    dphi_deg.append(dphi)

                ax.plot(tps, dphi_deg, label=f'{f} MHz', lw=1)

            ax.set_xlabel('Target Pos (mm)')
            ax.set_ylabel('Phase correction (°)')
            ax.set_title(f'{category} / {dim}-axis', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.15)
            ax.legend(fontsize=12, ncol=2)

    plt.tight_layout()
    fig.savefig(OUTDIR / 'phase_correction.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved phase_correction.png")


def test_free_space_for_all(df):
    """
    CRITICAL TEST: What happens if we ignore the tissue model entirely
    and use free-space distances for Chicken/Pork?
    If results are similar → tissue model is doing nothing useful.
    If results are worse → tissue model is at least partially correct.
    """
    freqs = ALL_HARMONICS
    categories = ['Patch', 'Chicken', 'Pork']

    print(f"\n  {'Condition':<15}  {'With tissue':>12}  {'Free-space':>12}  {'Δ':>8}")
    print(f"  {'─'*15}  {'─'*12}  {'─'*12}  {'─'*8}")

    all_results = []

    for axis in ['x', 'y']:
        for category in categories:
            agg = get_aggregated_data(df, category, axis)
            if agg.empty:
                continue
            target_positions = get_available_target_positions(agg)
            cal_data = get_phase_matrix(agg, CALIBRATION_TARGET_POS, freqs)
            if cal_data is None:
                continue

            # With tissue model
            loc_tissue = CalibratedLocalizer(
                cal_data['freqs'], cal_data['ant_pos'],
                dataset_axis=axis, category=category)
            loc_tissue.calibrate(cal_data['phases'], CALIBRATION_TARGET_POS)

            # Without tissue model (force Patch = free space)
            loc_free = CalibratedLocalizer(
                cal_data['freqs'], cal_data['ant_pos'],
                dataset_axis=axis, category='Patch')  # Force free-space
            loc_free.calibrate(cal_data['phases'], CALIBRATION_TARGET_POS)

            tissue_errors = []
            free_errors = []

            for tp in target_positions:
                if tp == CALIBRATION_TARGET_POS:
                    continue
                data = get_phase_matrix(agg, tp, freqs)
                if data is None:
                    continue
                try:
                    est_t, _ = loc_tissue.localize(data['phases'], data['powers'])
                    est_f, _ = loc_free.localize(data['phases'], data['powers'])
                    tissue_errors.append(abs(est_t - tp))
                    free_errors.append(abs(est_f - tp))
                except:
                    continue

            if tissue_errors:
                med_t = np.median(tissue_errors)
                med_f = np.median(free_errors)
                delta = med_f - med_t
                dim = 'X' if axis == 'x' else 'Y'
                label = f"{category}/{dim}"

                arrow = '↑' if delta > 0 else '↓'
                print(f"  {label:<15}  {med_t:>10.1f}mm  {med_f:>10.1f}mm  "
                      f"{arrow}{abs(delta):.1f}mm")

                all_results.append({
                    'condition': label, 'axis': axis, 'category': category,
                    'tissue_median': med_t, 'free_median': med_f,
                    'delta': delta,
                })

    return all_results


def measure_phase_vs_model(df):
    """
    For Chicken and Pork, compare:
    - Measured phase change (Δφ from calibration)
    - Model-predicted phase change (with tissue)
    - Model-predicted phase change (free-space)

    This will reveal whether the tissue model IMPROVES or HURTS
    the phase prediction accuracy.
    """
    rx_pos = rx_positions()
    tx_pos = TX_POS.copy()
    ant_idx = 2  # center antenna

    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    fig.suptitle('Measured vs Model Phase (Δφ from calibration)\n'
                 '— Center antenna, 1700 MHz —',
                 fontsize=18, fontweight='bold', y=1.01)

    freq = 1700
    f_hz = FREQ_TO_HZ[freq]

    for ci, category in enumerate(['Chicken', 'Pork']):
        for ai, axis in enumerate(['x', 'y']):
            row = ci * 2 + ai
            ax = axes[row][0]  # Wrapped
            ax2 = axes[row][1]  # Residual

            agg = get_aggregated_data(df, category, axis)
            if agg.empty:
                continue
            target_positions = get_available_target_positions(agg)
            cal_data = get_phase_matrix(agg, CALIBRATION_TARGET_POS, [freq])
            if cal_data is None:
                continue

            cal_phase = cal_data['phases'][ant_idx, 0]

            tm_tissue = TissueModel(category)
            tm_free = TissueModel('Patch')
            cal_xy = target_position(CALIBRATION_TARGET_POS, axis)
            D_cal_tissue = tm_tissue.effective_round_trip(
                cal_xy, tx_pos, rx_pos)[ant_idx]
            D_cal_free = tm_free.effective_round_trip(
                cal_xy, tx_pos, rx_pos)[ant_idx]

            tps_arr = []
            meas_dphi = []
            model_tissue_dphi = []
            model_free_dphi = []
            residual_tissue = []
            residual_free = []

            for tp in target_positions:
                if tp == CALIBRATION_TARGET_POS:
                    continue
                data = get_phase_matrix(agg, tp, [freq])
                if data is None:
                    continue

                xy = target_position(tp, axis)
                D_tissue = tm_tissue.effective_round_trip(
                    xy, tx_pos, rx_pos)[ant_idx]
                D_free = tm_free.effective_round_trip(
                    xy, tx_pos, rx_pos)[ant_idx]

                # Measured (wrapped)
                dp_meas = data['phases'][ant_idx, 0] - cal_phase
                dp_meas_w = ((dp_meas + 180) % 360) - 180

                # Model tissue
                dD_t = D_tissue - D_cal_tissue
                dp_model_t = np.degrees(-2 * np.pi * f_hz / C * dD_t)
                dp_model_t_w = ((dp_model_t + 180) % 360) - 180

                # Model free
                dD_f = D_free - D_cal_free
                dp_model_f = np.degrees(-2 * np.pi * f_hz / C * dD_f)
                dp_model_f_w = ((dp_model_f + 180) % 360) - 180

                tps_arr.append(tp)
                meas_dphi.append(dp_meas_w)
                model_tissue_dphi.append(dp_model_t_w)
                model_free_dphi.append(dp_model_f_w)

                # Complex residual (unwrapped)
                res_t = abs(np.exp(1j * np.radians(dp_meas_w))
                           - np.exp(1j * np.radians(dp_model_t_w)))
                res_f = abs(np.exp(1j * np.radians(dp_meas_w))
                           - np.exp(1j * np.radians(dp_model_f_w)))
                residual_tissue.append(res_t)
                residual_free.append(res_f)

            tps_arr = np.array(tps_arr)

            dim = 'X' if axis == 'x' else 'Y'

            # Left: wrapped phase comparison
            ax.scatter(tps_arr, meas_dphi, s=20, zorder=3,
                      label='Measured', alpha=0.8)
            ax.scatter(tps_arr, model_tissue_dphi, s=15, marker='x',
                      alpha=0.6, label='Model (tissue)')
            ax.scatter(tps_arr, model_free_dphi, s=15, marker='+',
                      alpha=0.6, label='Model (free-space)')
            ax.set_xlabel('Target Pos (mm)')
            ax.set_ylabel('Δφ (deg, wrapped)')
            ax.set_title(f'{category}/{dim} — Wrapped Δφ at {freq} MHz',
                        fontsize=14, fontweight='bold')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.15)

            # Right: residual comparison
            ax2.plot(tps_arr, residual_tissue, 'ro-', ms=3, alpha=0.7,
                    label=f'Tissue (mean={np.mean(residual_tissue):.3f})')
            ax2.plot(tps_arr, residual_free, 'bs-', ms=3, alpha=0.7,
                    label=f'Free (mean={np.mean(residual_free):.3f})')
            ax2.set_xlabel('Target Pos (mm)')
            ax2.set_ylabel('Complex residual |e^jΔφ_meas - e^jΔφ_model|')
            ax2.set_title(f'{category}/{dim} — Phase prediction residual',
                         fontsize=14, fontweight='bold')
            ax2.legend(fontsize=12)
            ax2.grid(True, alpha=0.15)
            ax2.set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(OUTDIR / 'phase_meas_vs_model.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved phase_meas_vs_model.png")


def quantify_tissue_correction_magnitude():
    """
    Show the actual magnitude of tissue correction as a percentage
    of the total phase change. If it's tiny, that explains the
    grid search insensitivity.
    """
    rx_pos = rx_positions()
    tx_pos = TX_POS.copy()



    print(f"\n  Phase correction magnitude (tissue model vs free-space):")
    print(f"  {'Category':<10} {'Freq':>6} {'Max Δφ(tissue)':>16} {'Max Δφ(free)':>14} "
          f"{'Correction':>12} {'% of total':>10}")
    print(f"  {'─'*10} {'─'*6} {'─'*16} {'─'*14} {'─'*12} {'─'*10}")

    for category in ['Chicken', 'Pork']:
        tm = TissueModel(category)
        cal_xy = target_position(CALIBRATION_TARGET_POS, 'x')
        D_cal_t = tm.effective_round_trip(cal_xy, tx_pos, rx_pos)[2]
        D_cal_f = free_space_round_trip(cal_xy, tx_pos, rx_pos)[2]

        for freq in [790, 1700]:
            f_hz = FREQ_TO_HZ[freq]

            max_dphi_t = 0
            max_dphi_f = 0

            for tp in range(-180, 181, 10):
                xy = target_position(tp, 'x')
                D_t = tm.effective_round_trip(xy, tx_pos, rx_pos)[2]
                D_f = free_space_round_trip(xy, tx_pos, rx_pos)[2]

                dD_t = D_t - D_cal_t
                dD_f = D_f - D_cal_f

                dphi_t = abs(np.degrees(-2 * np.pi * f_hz / C * dD_t))
                dphi_f = abs(np.degrees(-2 * np.pi * f_hz / C * dD_f))

                max_dphi_t = max(max_dphi_t, dphi_t)
                max_dphi_f = max(max_dphi_f, dphi_f)

            correction = abs(max_dphi_t - max_dphi_f)
            pct = correction / max(max_dphi_f, 1e-10) * 100

            print(f"  {category:<10} {freq:>4}MHz {max_dphi_t:>13.1f}° "
                  f"{max_dphi_f:>12.1f}° {correction:>10.1f}° {pct:>8.1f}%")


def main():
    csv_path = str(Path(__file__).resolve().parent.parent / 'data' / 'Remix_Dataset.csv')
    print(f"Loading dataset from {csv_path} ...")
    df = load_dataset(csv_path)
    print(f"  {len(df):,} rows\n")

    print(f"{'═' * 70}")
    print(f"  TISSUE MODEL INVESTIGATION")
    print(f"{'═' * 70}")

    print(f"\n  1. Distance difference (tissue vs free-space)...")
    compare_effective_distances()

    print(f"\n  2. Phase correction magnitude...")
    compare_phase_impact()

    print(f"\n  3. Quantifying correction magnitude...")
    quantify_tissue_correction_magnitude()

    print(f"\n  4. Free-space vs tissue model performance...")
    test_results = test_free_space_for_all(df)

    print(f"\n  5. Measured vs model phase comparison...")
    measure_phase_vs_model(df)

    # Summary
    print(f"\n{'═' * 70}")
    print(f"  SUMMARY")
    print(f"{'═' * 70}")

    if test_results:
        tissue_wins = sum(1 for r in test_results if r['delta'] > 1)
        free_wins = sum(1 for r in test_results if r['delta'] < -1)
        ties = len(test_results) - tissue_wins - free_wins

        print(f"\n  Tissue model helps in {tissue_wins}/{len(test_results)} conditions")
        print(f"  Free-space better in {free_wins}/{len(test_results)} conditions")
        print(f"  Negligible difference in {ties}/{len(test_results)} conditions")

    print(f"\n✓ All plots saved to {OUTDIR}/")


if __name__ == '__main__':
    main()
