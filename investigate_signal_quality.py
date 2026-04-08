"""
investigate_signal_quality.py — Is the problem attenuation or algorithm?

Analyzes:
1. Received power levels across Patch/Chicken/Pork
2. Power vs localization error correlation
3. Phase coherence / standard deviation across sweeps
4. SNR estimation and breakdown of where signal is reliable vs not
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from remix_config import (
    C, FREQ_TO_HZ, TX_POS, ALL_HARMONICS,
    rx_positions, target_position,
    CALIBRATION_TARGET_POS,
)
from remix_data import (
    load_dataset, get_aggregated_data, get_phase_matrix,
    get_available_target_positions,
)

OUTDIR = Path('results/signal_quality')
OUTDIR.mkdir(parents=True, exist_ok=True)

# Raw CSV columns: Category, Axis, Frequency, Target_Pos, Antenna_Pos, I, Q, dBm, Phase


def analyze_raw_power(df):
    """
    Look at the raw power levels before aggregation.
    """
    print("  Analyzing raw power levels...")

    categories = ['Patch', 'Chicken', 'Pork']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Received Power Distribution by Condition\n'
                 '(All antennas, all frequencies, all positions)',
                 fontsize=14, fontweight='bold')

    for ai, axis in enumerate(['x', 'y']):
        dim = 'X' if axis == 'x' else 'Y'
        for ci, category in enumerate(categories):
            ax = axes[ai][ci]
            sub = df[(df['Category'] == category) & (df['Axis'] == axis)]
            if sub.empty:
                ax.set_title(f'{category}/{dim} — no data')
                continue

            for freq in ALL_HARMONICS:
                fsub = sub[sub['Frequency'] == freq]
                if fsub.empty:
                    continue
                powers = fsub['dBm'].dropna()
                if len(powers) > 0:
                    ax.hist(powers, bins=30, alpha=0.4, label=f'{freq}MHz',
                           density=True)

            ax.set_xlabel('Power (dBm)')
            ax.set_ylabel('Density')
            ax.set_title(f'{category}/{dim}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=6, ncol=2)
            ax.grid(True, alpha=0.15)

    plt.tight_layout()
    fig.savefig(OUTDIR / 'power_distribution.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved power_distribution.png")

    # Summary table
    print(f"\n  {'Category':<10} {'Axis':<5} {'Freq':>6} {'Mean dBm':>10} {'Std':>6} "
          f"{'Min':>8} {'Max':>8} {'N':>8}")
    print(f"  {'─'*10} {'─'*5} {'─'*6} {'─'*10} {'─'*6} {'─'*8} {'─'*8} {'─'*8}")

    summary_data = []
    for category in categories:
        for axis in ['x', 'y']:
            dim = 'X' if axis == 'x' else 'Y'
            for freq in ALL_HARMONICS:
                sub = df[(df['Category'] == category) &
                         (df['Axis'] == axis) &
                         (df['Frequency'] == freq)]
                powers = sub['dBm'].dropna()
                if len(powers) > 0:
                    row = {
                        'category': category, 'axis': dim, 'freq': freq,
                        'mean_dBm': powers.mean(), 'std_dBm': powers.std(),
                        'min_dBm': powers.min(), 'max_dBm': powers.max(),
                        'n_samples': len(powers),
                    }
                    summary_data.append(row)
                    if freq in [790, 1700]:
                        print(f"  {category:<10} {dim:<5} {freq:>4}MHz "
                              f"{powers.mean():>9.1f} {powers.std():>5.1f} "
                              f"{powers.min():>7.1f} {powers.max():>7.1f} "
                              f"{len(powers):>7}")

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(OUTDIR / 'power_summary.csv', index=False)
    return summary_df


def analyze_phase_stability(df):
    """
    Phase std dev across repeated sweeps → signal coherence.
    """
    print("\n  Analyzing phase stability across sweeps...")

    categories = ['Patch', 'Chicken', 'Pork']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Phase Standard Deviation Across Sweeps\n'
                 '(Low = reliable signal, High = noise-dominated)',
                 fontsize=14, fontweight='bold')

    for ai, axis in enumerate(['x', 'y']):
        dim = 'X' if axis == 'x' else 'Y'
        for ci, category in enumerate(categories):
            ax = axes[ai][ci]

            for freq in [790, 910, 1700]:
                sub = df[(df['Category'] == category) &
                         (df['Axis'] == axis) &
                         (df['Frequency'] == freq)]
                if sub.empty:
                    continue

                grouped = sub.groupby(['Target_Pos', 'Antenna_Pos'])['Phase'].std()
                pos_std = grouped.groupby('Target_Pos').mean()

                if len(pos_std) > 0:
                    ax.plot(pos_std.index, pos_std.values, 'o-', ms=3,
                           label=f'{freq}MHz', alpha=0.7)

            ax.set_xlabel('Target Position (mm)')
            ax.set_ylabel('Phase Std Dev (°)')
            ax.set_title(f'{category}/{dim}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.15)
            ax.axhline(10, color='green', ls='--', alpha=0.4)
            ax.axhline(30, color='orange', ls='--', alpha=0.4)
            ax.axhline(90, color='red', ls='--', alpha=0.4)

    plt.tight_layout()
    fig.savefig(OUTDIR / 'phase_stability.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved phase_stability.png")


def analyze_power_vs_position(df):
    """
    Power vs target position — shows position-dependent attenuation.
    """
    print("\n  Analyzing power vs target position...")

    categories = ['Patch', 'Chicken', 'Pork']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Mean Received Power vs Target Position\n'
                 '(Center antenna — Antenna_Pos closest to 0)',
                 fontsize=14, fontweight='bold')

    for ai, axis in enumerate(['x', 'y']):
        dim = 'X' if axis == 'x' else 'Y'
        for ci, category in enumerate(categories):
            ax = axes[ai][ci]

            # Find center antenna position
            positions = df[(df['Category'] == category) &
                          (df['Axis'] == axis)]['Antenna_Pos'].unique()
            if len(positions) == 0:
                ax.set_title(f'{category}/{dim} — no data')
                continue
            center_ant = positions[np.argmin(np.abs(positions))]

            for freq in ALL_HARMONICS:
                sub = df[(df['Category'] == category) &
                         (df['Axis'] == axis) &
                         (df['Frequency'] == freq) &
                         (df['Antenna_Pos'] == center_ant)]
                if sub.empty:
                    continue

                mean_power = sub.groupby('Target_Pos')['dBm'].mean()
                tp_vals = mean_power.index[mean_power.index > -900]
                if len(tp_vals) > 0:
                    ax.plot(tp_vals, mean_power[tp_vals], 'o-', ms=3,
                           label=f'{freq}MHz', alpha=0.7, lw=1)

            ax.set_xlabel('Target Position (mm)')
            ax.set_ylabel('Mean Power (dBm)')
            ax.set_title(f'{category}/{dim}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=6, ncol=2)
            ax.grid(True, alpha=0.15)

    plt.tight_layout()
    fig.savefig(OUTDIR / 'power_vs_position.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved power_vs_position.png")


def analyze_attenuation_by_tissue(df):
    """
    Direct comparison: same position, same frequency, different tissue.
    """
    print("\n  Computing tissue attenuation...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Tissue Attenuation Relative to Free-Space (Patch)\n'
                 'Power_Patch - Power_Tissue at same position (1700 MHz)',
                 fontsize=14, fontweight='bold')

    for ai, axis in enumerate(['x', 'y']):
        dim = 'X' if axis == 'x' else 'Y'
        ax = axes[ai]

        for category in ['Chicken', 'Pork']:
            freq = 1700
            sub_patch = df[(df['Category'] == 'Patch') &
                          (df['Axis'] == axis) &
                          (df['Frequency'] == freq)]
            sub_tissue = df[(df['Category'] == category) &
                           (df['Axis'] == axis) &
                           (df['Frequency'] == freq)]

            if sub_patch.empty or sub_tissue.empty:
                continue

            patch_power = sub_patch.groupby('Target_Pos')['dBm'].mean()
            tissue_power = sub_tissue.groupby('Target_Pos')['dBm'].mean()

            # Only real target positions
            patch_power = patch_power[patch_power.index > -900]
            tissue_power = tissue_power[tissue_power.index > -900]

            common = patch_power.index.intersection(tissue_power.index)
            if len(common) == 0:
                continue

            attenuation = patch_power[common] - tissue_power[common]
            ax.plot(common, attenuation, 'o-', ms=4,
                   label=f'{category} (mean={attenuation.mean():.1f}dB)', lw=1.5)

        ax.set_xlabel('Target Position (mm)')
        ax.set_ylabel('Attenuation (dB)')
        ax.set_title(f'{dim}-axis / 1700 MHz', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.15)
        ax.axhline(0, color='gray', ls='--', alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTDIR / 'tissue_attenuation.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved tissue_attenuation.png")


def correlate_power_and_error(df):
    """
    Scatter: power vs localization error.
    Positive correlation → attenuation is the bottleneck.
    No correlation → algorithm is the bottleneck.
    """
    print("\n  Correlating power with localization error...")

    try:
        results_df = pd.read_csv('results/results.csv')
    except FileNotFoundError:
        print("  ⚠ results/results.csv not found, skipping")
        return

    categories = ['Patch', 'Chicken', 'Pork']

    fig, axes_arr = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Localization Error vs Mean Received Power\n'
                 'Negative correlation → weaker signal = more error = attenuation problem',
                 fontsize=14, fontweight='bold')

    # Pre-compute mean power per (category, axis, target_pos)
    agg_power = df.groupby(['Category', 'Axis', 'Target_Pos'])['dBm'].mean()

    for ai, axis in enumerate(['x', 'y']):
        dim = 'X' if axis == 'x' else 'Y'
        for ci, category in enumerate(categories):
            ax = axes_arr[ai][ci]

            for approach, color, marker in [
                ('harmonic_comb', 'blue', 'o'),
                ('calibrated', 'red', 's')
            ]:
                mask = (results_df['config'].str.contains(f'{category}/{axis}') &
                        results_df['config'].str.contains(approach))
                res_sub = results_df[mask]

                if res_sub.empty:
                    continue

                powers = []
                errors = []
                for _, row in res_sub.iterrows():
                    tp = row['target_pos']
                    try:
                        pwr = agg_power.loc[(category, axis, tp)]
                        powers.append(pwr)
                        errors.append(row['error'])
                    except KeyError:
                        continue

                if powers:
                    ax.scatter(powers, errors, alpha=0.5, s=20,
                             color=color, marker=marker, label=approach)

                    if len(powers) > 3:
                        corr = np.corrcoef(powers, errors)[0, 1]
                        y_pos = 0.98 if approach == 'harmonic_comb' else 0.90
                        ax.text(0.02, y_pos,
                               f'{approach}: r={corr:.2f}',
                               transform=ax.transAxes, fontsize=8, va='top',
                               color=color)

            ax.set_xlabel('Mean Power (dBm)')
            ax.set_ylabel('Error (mm)')
            ax.set_title(f'{category}/{dim}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.15)

    plt.tight_layout()
    fig.savefig(OUTDIR / 'power_error_correlation.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved power_error_correlation.png")


def analyze_phase_residuals(df):
    """
    Phase prediction residual per position — reveals model quality.
    """
    print("\n  Analyzing phase model residuals...")

    rx_pos = rx_positions()
    tx_pos = TX_POS.copy()

    from remix_localize import TissueModel

    categories = ['Patch', 'Chicken', 'Pork']
    freqs = ALL_HARMONICS

    fig, axes_arr = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Phase Prediction Residual vs Target Position\n'
                 '|exp(jΔφ_meas) - exp(jΔφ_model)| avg.  '
                 '0 = perfect, 2 = completely wrong',
                 fontsize=14, fontweight='bold')

    residual_data = []

    for ai, axis in enumerate(['x', 'y']):
        dim = 'X' if axis == 'x' else 'Y'
        for ci, category in enumerate(categories):
            ax = axes_arr[ai][ci]

            agg = get_aggregated_data(df, category, axis)
            if agg.empty:
                ax.set_title(f'{category}/{dim} — no data')
                continue

            target_positions = get_available_target_positions(agg)
            cal_data = get_phase_matrix(agg, CALIBRATION_TARGET_POS, freqs)
            if cal_data is None:
                continue

            tm = TissueModel('Patch')
            cal_xy = target_position(CALIBRATION_TARGET_POS, axis)
            D_cal = tm.effective_round_trip(cal_xy, tx_pos, rx_pos)
            cal_phases = np.deg2rad(cal_data['phases'])
            freq_hz = np.array([FREQ_TO_HZ[f] for f in freqs])

            tps_list = []
            mean_residuals = []
            per_freq_residuals = {f: [] for f in freqs}

            for tp in target_positions:
                if tp == CALIBRATION_TARGET_POS:
                    continue
                data = get_phase_matrix(agg, tp, freqs)
                if data is None:
                    continue

                xy = target_position(tp, axis)
                D_new = tm.effective_round_trip(xy, tx_pos, rx_pos)
                delta_D = D_new - D_cal

                meas_rad = np.deg2rad(data['phases'])
                delta_phi_model = (-2.0 * np.pi * freq_hz[np.newaxis, :] / C
                                   * delta_D[:, np.newaxis])
                delta_phi_meas = meas_rad - cal_phases

                residual = np.abs(
                    np.exp(1j * delta_phi_meas) - np.exp(1j * delta_phi_model))

                mean_res = np.mean(residual)
                tps_list.append(tp)
                mean_residuals.append(mean_res)

                for fi, f in enumerate(freqs):
                    per_freq_residuals[f].append(np.mean(residual[:, fi]))

                residual_data.append({
                    'category': category, 'axis': dim, 'tp': tp,
                    'mean_residual': mean_res,
                    'max_residual': float(np.max(residual)),
                    'min_residual': float(np.min(residual)),
                })

            if tps_list:
                ax.plot(tps_list, mean_residuals, 'ko-', ms=4, lw=1.5,
                       label='All freqs avg', zorder=5)
                for f in [790, 1700]:
                    if per_freq_residuals[f]:
                        ax.plot(tps_list, per_freq_residuals[f], '--', ms=2,
                               alpha=0.5, label=f'{f}MHz', lw=1)

            ax.set_xlabel('Target Position (mm)')
            ax.set_ylabel('Mean Residual')
            ax.set_title(f'{category}/{dim}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.15)
            ax.set_ylim(0, 2.1)
            ax.axhspan(0, 0.5, alpha=0.05, color='green')
            ax.axhspan(0.5, 1.0, alpha=0.05, color='orange')
            ax.axhspan(1.0, 2.0, alpha=0.05, color='red')

            if mean_residuals:
                mean_overall = np.mean(mean_residuals)
                ax.text(0.98, 0.98, f'Mean: {mean_overall:.2f}',
                       transform=ax.transAxes, fontsize=10, va='top', ha='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fig.savefig(OUTDIR / 'phase_residuals.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved phase_residuals.png")

    if residual_data:
        pd.DataFrame(residual_data).to_csv(
            OUTDIR / 'phase_residuals.csv', index=False)

    return residual_data


def create_diagnosis(df, residual_data):
    """
    Final diagnosis per condition.
    """
    print(f"\n{'═' * 70}")
    print(f"  DIAGNOSIS: ATTENUATION vs ALGORITHM")
    print(f"{'═' * 70}")

    categories = ['Patch', 'Chicken', 'Pork']

    print(f"\n  {'Condition':<12} {'MeanPwr':>8} {'PhStd':>7} {'Residual':>9} "
          f"{'Signal':>8} {'Model':>8} {'Diagnosis'}")
    print(f"  {'─'*12} {'─'*8} {'─'*7} {'─'*9} {'─'*8} {'─'*8} {'─'*35}")

    for axis in ['x', 'y']:
        dim = 'X' if axis == 'x' else 'Y'
        for category in categories:
            label = f"{category}/{dim}"

            sub = df[(df['Category'] == category) & (df['Axis'] == axis)]
            if sub.empty:
                continue

            mean_power = sub['dBm'].mean()

            # Phase std across sweeps
            phase_std = sub.groupby(
                ['Target_Pos', 'Antenna_Pos'])['Phase'].std().mean()

            # Model residual
            res = [r for r in residual_data
                   if r['category'] == category and r['axis'] == dim]
            mean_res = np.mean([r['mean_residual'] for r in res]) if res else 999

            signal_ok = '✅' if phase_std < 20 else ('⚠️' if phase_std < 40 else '❌')
            model_ok = '✅' if mean_res < 0.5 else ('⚠️' if mean_res < 1.0 else '❌')

            if phase_std < 20 and mean_res < 0.5:
                diagnosis = "Algorithm (phase ambiguity)"
            elif phase_std >= 40 or mean_res >= 1.0:
                diagnosis = "SIGNAL (noise-dominated)"
            elif phase_std >= 20:
                diagnosis = "Signal + Algorithm"
            else:
                diagnosis = "Algorithm (model mismatch)"

            print(f"  {label:<12} {mean_power:>6.1f}dB {phase_std:>5.1f}° "
                  f"{mean_res:>8.2f}  {signal_ok:>6} {model_ok:>6}  {diagnosis}")

    print()


def main():
    csv_path = 'Remix_Dataset.csv'
    print(f"Loading dataset from {csv_path} ...")
    df = load_dataset(csv_path)
    print(f"  {len(df):,} rows\n")

    print(f"{'═' * 70}")
    print(f"  SIGNAL QUALITY INVESTIGATION")
    print(f"{'═' * 70}")

    summary = analyze_raw_power(df)
    analyze_phase_stability(df)
    analyze_power_vs_position(df)
    analyze_attenuation_by_tissue(df)
    correlate_power_and_error(df)
    residual_data = analyze_phase_residuals(df)
    create_diagnosis(df, residual_data)

    print(f"✓ All plots saved to {OUTDIR}/")


if __name__ == '__main__':
    main()
