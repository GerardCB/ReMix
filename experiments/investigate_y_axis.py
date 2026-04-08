"""
investigate_y_axis.py — Debug the Y-axis phase wrapping failures.

Generates cost landscape plots for positions where both algorithms fail,
showing exactly where/why the optimizer gets trapped.
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
    DEPTH_SEARCH, CALIBRATION_TARGET_POS,
)
from src.data import (
    load_dataset, get_aggregated_data, get_phase_matrix,
    get_available_target_positions,
)
from src.localize import (
    TissueModel, CalibratedLocalizer, per_frequency_grid,
)

OUTDIR = Path(__file__).resolve().parent.parent / 'results' / 'y_axis_investigation'
OUTDIR.mkdir(parents=True, exist_ok=True)


def investigate_cost_landscape(df, category='Patch'):
    """
    Detailed cost landscape analysis for every Y-axis target position.
    Shows: per-frequency costs, joint cost, and where the optimizer lands.
    """
    axis = 'y'
    freqs = ALL_HARMONICS
    
    agg = get_aggregated_data(df, category, axis)
    target_positions = get_available_target_positions(agg)
    
    cal_data = get_phase_matrix(agg, CALIBRATION_TARGET_POS, freqs)
    if cal_data is None:
        print(f"  No calibration data for {category}/{axis}")
        return
    
    tm = TissueModel(category)
    rx_pos = rx_positions()
    tx_pos = TX_POS.copy()
    cal_xy = target_position(CALIBRATION_TARGET_POS, axis)
    cal_D = tm.effective_round_trip(cal_xy, tx_pos, rx_pos)
    freq_hz = np.array([FREQ_TO_HZ[f] for f in cal_data['freqs']])
    cal_rad = np.deg2rad(cal_data['phases'])
    
    # Also run the localizer to get its estimates
    localizer = CalibratedLocalizer(
        cal_data['freqs'], cal_data['ant_pos'],
        dataset_axis=axis, category=category,
    )
    localizer.calibrate(cal_data['phases'], CALIBRATION_TARGET_POS)
    
    grid = np.arange(-200, 201, 0.5)
    
    # Select positions that include failures AND successes
    test_positions = target_positions[target_positions != CALIBRATION_TARGET_POS]
    
    # Create a comprehensive cost landscape figure for ALL positions
    n_pos = len(test_positions)
    n_cols = 6
    n_rows = (n_pos + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten() if n_pos > 1 else [axes]
    fig.suptitle(f'{category} / Y-axis: Joint Cost Landscape for Each Position',
                 fontsize=14, fontweight='bold', y=1.01)
    
    errors = []
    
    for pi, tp in enumerate(test_positions):
        ax = axes[pi] if pi < len(axes) else None
        if ax is None:
            break
        
        data = get_phase_matrix(agg, tp, freqs)
        if data is None:
            ax.set_title(f'TP={tp:.0f} — no data', fontsize=8)
            continue
        
        meas_rad = np.deg2rad(data['phases'])
        
        # Per-frequency costs on the grid
        per_freq = per_frequency_grid(
            grid, axis, freq_hz, rx_pos, tx_pos, cal_D,
            meas_rad, cal_rad, tm)
        
        joint = per_freq.sum(axis=1)
        
        # Get localizer estimate
        try:
            est_tp, cost = localizer.localize(data['phases'], data['powers'])
        except:
            est_tp = np.nan
        
        error = abs(est_tp - tp) if not np.isnan(est_tp) else np.nan
        errors.append({'tp': tp, 'est': est_tp, 'error': error})
        
        # Plot joint cost
        ax.plot(grid, joint, 'b-', lw=0.8, alpha=0.8)
        ax.axvline(tp, color='green', ls='--', lw=1.5, alpha=0.7, label=f'true')
        if not np.isnan(est_tp):
            ax.axvline(est_tp, color='red', ls=':', lw=1.5, alpha=0.7, label=f'est')
        
        # Mark global minimum of joint cost
        grid_min = grid[np.argmin(joint)]
        ax.axvline(grid_min, color='orange', ls='-.', lw=1, alpha=0.5)
        
        # Color title by error severity
        if np.isnan(error):
            color = 'gray'
        elif error < 20:
            color = 'green'
        elif error < 50:
            color = 'orange'
        else:
            color = 'red'
        
        ax.set_title(f'TP={tp:.0f}  est={est_tp:.0f}  err={error:.0f}mm',
                     fontsize=7, color=color, fontweight='bold')
        ax.set_xlim(-200, 200)
        ax.grid(True, alpha=0.15)
        ax.tick_params(labelsize=6)
    
    # Hide unused axes
    for pi in range(len(test_positions), len(axes)):
        axes[pi].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(OUTDIR / f'cost_landscape_all_{category}.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved cost_landscape_all_{category}.png")
    
    return errors


def investigate_phase_wrapping(df, category='Patch'):
    """
    Show the phase vs position curves to visualize wrapping behavior.
    Plot: unwrapped measured phase difference (from calibration) vs TP,
    overlaid with the model prediction.
    """
    axis = 'y'
    freqs = ALL_HARMONICS
    
    agg = get_aggregated_data(df, category, axis)
    target_positions = get_available_target_positions(agg)
    test_positions = target_positions[target_positions != CALIBRATION_TARGET_POS]
    
    cal_data = get_phase_matrix(agg, CALIBRATION_TARGET_POS, freqs)
    if cal_data is None:
        return
    
    tm = TissueModel(category)
    rx_pos = rx_positions()
    tx_pos = TX_POS.copy()
    cal_xy = target_position(CALIBRATION_TARGET_POS, axis)
    cal_D = tm.effective_round_trip(cal_xy, tx_pos, rx_pos)
    freq_hz = np.array([FREQ_TO_HZ[f] for f in cal_data['freqs']])
    
    n_freq = len(cal_data['freqs'])
    n_ant = len(cal_data['ant_pos'])
    
    # For the center antenna (index 2, position 0mm)
    ant_idx = 2
    
    fig, axes = plt.subplots(n_freq, 2, figsize=(14, 3 * n_freq), squeeze=False)
    fig.suptitle(f'{category} / Y-axis: Phase Measurements vs Model\n'
                 f'(Center antenna, ant_pos=0mm)',
                 fontsize=13, fontweight='bold', y=1.01)
    
    model_tp = np.arange(-200, 201, 1.0)
    
    for fi in range(n_freq):
        freq = cal_data['freqs'][fi]
        f_hz = freq_hz[fi]
        
        # Model prediction (wrapped delta-phase)
        model_dphi = []
        for tp in model_tp:
            dev = target_position(tp, axis)
            D = tm.effective_round_trip(dev, tx_pos, rx_pos)
            dD = D[ant_idx] - cal_D[ant_idx]
            dphi = np.degrees(-2 * np.pi * f_hz / C * dD)
            model_dphi.append(dphi)
        model_dphi = np.array(model_dphi)
        
        # Measured delta-phase
        meas_tp = []
        meas_dphi_wrapped = []
        meas_dphi_unwrapped = []
        
        for tp in test_positions:
            data = get_phase_matrix(agg, tp, [freq])
            if data is None:
                continue
            cal_ph = cal_data['phases'][ant_idx, fi]
            meas_ph = data['phases'][ant_idx, 0]
            dp = meas_ph - cal_ph
            dp_wrapped = ((dp + 180) % 360) - 180
            meas_tp.append(tp)
            meas_dphi_wrapped.append(dp_wrapped)
            meas_dphi_unwrapped.append(dp)
        
        meas_tp = np.array(meas_tp)
        meas_dphi_wrapped = np.array(meas_dphi_wrapped)
        
        # Left: wrapped
        ax = axes[fi][0]
        model_wrapped = ((model_dphi + 180) % 360) - 180
        ax.plot(model_tp, model_wrapped, 'r-', lw=0.8, alpha=0.6, label='model')
        ax.scatter(meas_tp, meas_dphi_wrapped, s=12, alpha=0.7,
                  zorder=3, label='measured')
        ax.set_ylim(-190, 190)
        ax.set_ylabel(f'{freq} MHz\nΔφ (°) wrapped')
        ax.set_xlabel('Target Pos (mm)')
        ax.grid(True, alpha=0.15)
        ax.legend(fontsize=7)
        if fi == 0:
            ax.set_title('Wrapped Δφ', fontsize=10)
        
        # Right: unwrapped model to show how many times it wraps
        ax = axes[fi][1]
        ax.plot(model_tp, model_dphi, 'r-', lw=1, alpha=0.7, label='model (unwrapped)')
        
        # Show wrap period
        wrap_period = C / (2 * f_hz) * 1000  # mm
        n_wraps = 400 / wrap_period
        ax.set_title(f'Unwrapped model: λ_wrap={wrap_period:.0f}mm, '
                    f'{n_wraps:.1f} wraps over ±200mm', fontsize=8)
        ax.set_xlabel('Target Pos (mm)')
        ax.set_ylabel('Δφ (°) unwrapped')
        ax.grid(True, alpha=0.15)
        ax.legend(fontsize=7)
    
    plt.tight_layout()
    fig.savefig(OUTDIR / f'phase_wrapping_{category}.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved phase_wrapping_{category}.png")


def investigate_coarse_vs_fine(df, category='Patch'):
    """
    Show the coarse (790+910) and fine (all) cost separately.
    Diagnose whether the coarse stage gives the right basin.
    """
    axis = 'y'
    
    agg = get_aggregated_data(df, category, axis)
    target_positions = get_available_target_positions(agg)
    test_positions = target_positions[target_positions != CALIBRATION_TARGET_POS]
    
    # Pick 8 representative positions (spread across range)
    idx = np.linspace(0, len(test_positions) - 1, 8, dtype=int)
    selected = test_positions[idx]
    
    cal_data = get_phase_matrix(agg, CALIBRATION_TARGET_POS, ALL_HARMONICS)
    if cal_data is None:
        return
    
    tm = TissueModel(category)
    rx_pos = rx_positions()
    tx_pos = TX_POS.copy()
    cal_xy = target_position(CALIBRATION_TARGET_POS, axis)
    cal_D = tm.effective_round_trip(cal_xy, tx_pos, rx_pos)
    freq_hz = np.array([FREQ_TO_HZ[f] for f in cal_data['freqs']])
    cal_rad = np.deg2rad(cal_data['phases'])
    
    # Indices for coarse pair (790, 910) = indices 0, 1
    coarse_freqs = [790, 910]
    coarse_idx = [cal_data['freqs'].index(f) for f in coarse_freqs if f in cal_data['freqs']]
    
    grid = np.arange(-200, 201, 1.0)
    
    fig, axes_arr = plt.subplots(len(selected), 3, 
                                  figsize=(18, 3 * len(selected)),
                                  squeeze=False)
    fig.suptitle(f'{category} / Y-axis: Coarse vs Fine Cost Landscape',
                 fontsize=14, fontweight='bold', y=1.01)
    
    for ri, tp in enumerate(selected):
        data = get_phase_matrix(agg, tp, ALL_HARMONICS)
        if data is None:
            continue
        
        meas_rad = np.deg2rad(data['phases'])
        
        per_freq = per_frequency_grid(
            grid, axis, freq_hz, rx_pos, tx_pos, cal_D,
            meas_rad, cal_rad, tm)
        
        # Coarse cost (790 + 910 only)
        coarse_cost = per_freq[:, coarse_idx].sum(axis=1)
        
        # Fine cost (remaining frequencies)
        fine_idx = [i for i in range(len(freq_hz)) if i not in coarse_idx]
        fine_cost = per_freq[:, fine_idx].sum(axis=1)
        
        # Joint cost
        joint_cost = per_freq.sum(axis=1)
        
        # Coarse minimum
        coarse_min = grid[np.argmin(coarse_cost)]
        joint_min = grid[np.argmin(joint_cost)]
        
        # Plot coarse
        ax = axes_arr[ri][0]
        ax.plot(grid, coarse_cost, 'b-', lw=1)
        ax.axvline(tp, color='green', ls='--', lw=1.5, label=f'true={tp:.0f}')
        ax.axvline(coarse_min, color='red', ls=':', lw=1.5,
                  label=f'min={coarse_min:.0f}')
        ax.set_title(f'TP={tp:.0f} — Coarse (790+910)', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.15)
        
        # Plot fine
        ax = axes_arr[ri][1]
        ax.plot(grid, fine_cost, 'orange', lw=1)
        ax.axvline(tp, color='green', ls='--', lw=1.5)
        fine_min = grid[np.argmin(fine_cost)]
        ax.axvline(fine_min, color='red', ls=':', lw=1.5,
                  label=f'min={fine_min:.0f}')
        ax.set_title(f'Fine (5 higher freqs)', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.15)
        
        # Plot joint
        ax = axes_arr[ri][2]
        ax.plot(grid, joint_cost, 'purple', lw=1)
        ax.axvline(tp, color='green', ls='--', lw=1.5)
        ax.axvline(joint_min, color='red', ls=':', lw=1.5,
                  label=f'min={joint_min:.0f}')
        coarse_err = abs(coarse_min - tp)
        joint_err = abs(joint_min - tp)
        ax.set_title(f'Joint — c_err={coarse_err:.0f}, j_err={joint_err:.0f}',
                     fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.15)
    
    plt.tight_layout()
    fig.savefig(OUTDIR / f'coarse_vs_fine_{category}.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved coarse_vs_fine_{category}.png")


def main():
    csv_path = str(Path(__file__).resolve().parent.parent / 'data' / 'Remix_Dataset.csv')
    print(f"Loading dataset from {csv_path} ...")
    df = load_dataset(csv_path)
    print(f"  {len(df):,} rows\n")
    
    for category in ['Patch', 'Chicken', 'Pork']:
        print(f"\n{'═' * 60}")
        print(f"  Investigating {category} / Y-axis")
        print(f"{'═' * 60}")
        
        print(f"\n  1. Cost landscape for ALL positions...")
        errors = investigate_cost_landscape(df, category)
        
        if errors:
            err_df = pd.DataFrame(errors)
            print(f"\n  Error statistics (calibrated, all freqs):")
            print(f"    Positions < 20mm error: "
                  f"{(err_df['error'] < 20).sum()}/{len(err_df)}")
            print(f"    Positions < 50mm error: "
                  f"{(err_df['error'] < 50).sum()}/{len(err_df)}")
            print(f"    Median error: {err_df['error'].median():.1f}mm")
            # Show which positions fail
            fails = err_df[err_df['error'] > 50].sort_values('error', ascending=False)
            if len(fails) > 0:
                print(f"    FAILING positions (>50mm):")
                for _, row in fails.head(10).iterrows():
                    print(f"      TP={row['tp']:>6.0f} → est={row['est']:>6.0f}  "
                          f"err={row['error']:>6.0f}mm")
        
        print(f"\n  2. Phase wrapping analysis...")
        investigate_phase_wrapping(df, category)
        
        print(f"\n  3. Coarse vs fine analysis...")
        investigate_coarse_vs_fine(df, category)
    
    print(f"\n✓ All investigation plots saved to {OUTDIR}/")


if __name__ == '__main__':
    main()
