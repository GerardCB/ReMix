"""
report_freq_pairs.py — Compare harmonic comb performance with different frequency pairs.

The harmonic combination trick from the ReMix paper combines two harmonics
algebraically. Given harmonics at frequencies with coefficients (n,m) where
freq = n*f1 + m*f2, combining phases lets us create virtual measurements
at higher effective frequencies.

For pair (A, B) with coefficients (nA, mA) and (nB, mB):
  φ_A + φ_B → effective at (nA+nB)*f1 + (mA+mB)*f2
  2φ_A - φ_B → effective at (2nA-nB)*f1 + (2mA-mB)*f2
  2φ_B - φ_A → effective at (2nB-nA)*f1 + (2mB-mA)*f2

This script tests ALL sensible pairs and reports their performance.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    C, FREQ_TO_HZ, HARMONIC_COEFFS, TX_POS,
    rx_positions, target_position,
    DEPTH_SEARCH, LATERAL_SEARCH, CALIBRATION_TARGET_POS,
)
from src.data import (
    load_dataset, get_aggregated_data, get_phase_matrix,
    get_available_target_positions,
)
from src.localize import TissueModel, _find_local_minima

OUTDIR = Path(__file__).resolve().parent.parent / 'results'
OUTDIR.mkdir(exist_ok=True)

F1 = 830e6
F2 = 870e6


class GeneralizedHarmonicCombLocalizer:
    """
    Generalized harmonic combination localizer that works with ANY pair
    of measured harmonics.
    
    Given two harmonics at measured frequencies fA and fB with known
    harmonic coefficients (nA, mA) and (nB, mB), we form two combinations:
      combo1 = φ_A + φ_B → effective at (nA+nB)f1 + (mA+mB)f2
      combo2 = 2φ_B - φ_A → effective at (2nB-nA)f1 + (2mB-mA)f2
    """
    
    def __init__(self, freq_a_mhz, freq_b_mhz, ant_positions,
                 dataset_axis='y', category='Patch'):
        self.freq_a = freq_a_mhz
        self.freq_b = freq_b_mhz
        self.rx_pos = rx_positions()
        self.tx_pos = TX_POS.copy()
        self.axis = dataset_axis
        self.tissue = TissueModel(category)
        self.n_ant = len(ant_positions)
        
        # Get harmonic coefficients
        nA, mA = HARMONIC_COEFFS[freq_a_mhz]
        nB, mB = HARMONIC_COEFFS[freq_b_mhz]
        
        # Effective frequencies for the two combinations
        # combo1: φA + φB
        n1, m1 = nA + nB, mA + mB
        f_eff_1 = n1 * F1 + m1 * F2
        
        # combo2: 2φB - φA (use this order for variety)
        n2, m2 = 2*nB - nA, 2*mB - mA
        f_eff_2 = n2 * F1 + m2 * F2
        
        self.f_eff = np.array([f_eff_1, f_eff_2])
        self.combo_labels = [
            f"({n1},{m1}) = {f_eff_1/1e6:.0f}MHz",
            f"({n2},{m2}) = {f_eff_2/1e6:.0f}MHz",
        ]
        
        self._cal_D = None
        self._cal_combined_rad = None
    
    def calibrate(self, phase_deg_a, phase_deg_b,
                  cal_target_pos=CALIBRATION_TARGET_POS):
        cal_xy = target_position(cal_target_pos, self.axis)
        self._cal_D = self.tissue.effective_round_trip(
            cal_xy, self.tx_pos, self.rx_pos)
        phi_a = np.deg2rad(phase_deg_a)
        phi_b = np.deg2rad(phase_deg_b)
        # combo1 = φA + φB, combo2 = 2φB - φA
        self._cal_combined_rad = np.column_stack([
            phi_a + phi_b,
            2 * phi_b - phi_a,
        ])
    
    @property
    def is_calibrated(self):
        return self._cal_D is not None
    
    def _cost_at(self, tp, delta_combined):
        target_xy = target_position(tp, self.axis)
        D_new = self.tissue.effective_round_trip(
            target_xy, self.tx_pos, self.rx_pos)
        delta_D = D_new - self._cal_D
        delta_model = -2.0 * np.pi * self.f_eff[np.newaxis, :] / C \
                      * delta_D[:, np.newaxis]
        residual = np.exp(1j * delta_combined) - np.exp(1j * delta_model)
        return np.sum(np.abs(residual) ** 2)
    
    def localize(self, phase_deg_a, phase_deg_b):
        if not self.is_calibrated:
            raise RuntimeError("Call calibrate() first.")
        phi_a = np.deg2rad(phase_deg_a)
        phi_b = np.deg2rad(phase_deg_b)
        meas_combined = np.column_stack([
            phi_a + phi_b,
            2 * phi_b - phi_a,
        ])
        delta = meas_combined - self._cal_combined_rad
        
        if self.axis == 'y':
            sb = DEPTH_SEARCH
        else:
            sb = LATERAL_SEARCH
        
        grid = np.arange(sb[0], sb[1] + 1, 1.0)
        costs = np.array([self._cost_at(tp, delta) for tp in grid])
        candidates = _find_local_minima(costs, grid, n_candidates=10)
        
        best_tp, best_cost = candidates[0], np.inf
        for cand in candidates:
            fine = np.arange(cand - 5, cand + 5.5, 0.5)
            fine = fine[(fine >= sb[0]) & (fine <= sb[1])]
            for tp in fine:
                c = self._cost_at(tp, delta)
                if c < best_cost:
                    best_cost = c
                    best_tp = tp
        return best_tp, best_cost


def evaluate_pair(df, freq_a, freq_b, axis, category):
    """Evaluate a frequency pair on all positions."""
    agg = get_aggregated_data(df, category, axis)
    if agg.empty:
        return []
    
    target_positions = get_available_target_positions(agg)
    
    cal_a = get_phase_matrix(agg, CALIBRATION_TARGET_POS, [freq_a])
    cal_b = get_phase_matrix(agg, CALIBRATION_TARGET_POS, [freq_b])
    if cal_a is None or cal_b is None:
        return []
    
    localizer = GeneralizedHarmonicCombLocalizer(
        freq_a, freq_b, cal_a['ant_pos'],
        dataset_axis=axis, category=category,
    )
    localizer.calibrate(
        cal_a['phases'][:, 0], cal_b['phases'][:, 0],
        CALIBRATION_TARGET_POS,
    )
    
    results = []
    for tp in target_positions:
        if tp == CALIBRATION_TARGET_POS:
            continue
        data_a = get_phase_matrix(agg, tp, [freq_a])
        data_b = get_phase_matrix(agg, tp, [freq_b])
        if data_a is None or data_b is None:
            continue
        try:
            est_tp, cost = localizer.localize(
                data_a['phases'][:, 0], data_b['phases'][:, 0])
        except Exception:
            continue
        results.append({
            'target_pos': tp,
            'est_target_pos': est_tp,
            'error': abs(est_tp - tp),
            'signed_error': est_tp - tp,
            'cost': cost,
        })
    
    return results, localizer


def main():
    csv_path = str(Path(__file__).resolve().parent.parent / 'data' / 'Remix_Dataset.csv')
    print(f"Loading dataset from {csv_path} ...")
    df = load_dataset(csv_path)
    print(f"  {len(df):,} rows\n")
    
    # All available frequencies
    all_freqs = sorted(HARMONIC_COEFFS.keys())
    all_pairs = list(combinations(all_freqs, 2))
    
    print(f"Testing {len(all_pairs)} frequency pairs:")
    for fa, fb in all_pairs:
        nA, mA = HARMONIC_COEFFS[fa]
        nB, mB = HARMONIC_COEFFS[fb]
        f_eff_1 = (nA + nB) * F1 + (mA + mB) * F2
        f_eff_2 = (2*nB - nA) * F1 + (2*mB - mA) * F2
        print(f"  ({fa}, {fb}) MHz → effective ({f_eff_1/1e6:.0f}, {f_eff_2/1e6:.0f}) MHz")
    
    # Evaluate all pairs
    categories = ['Patch', 'Chicken', 'Pork']
    axes_to_test = ['x', 'y']
    
    all_data = []
    
    for axis in axes_to_test:
        for category in categories:
            print(f"\n{'─' * 60}")
            print(f"  {category} / axis={axis}")
            print(f"{'─' * 60}")
            
            for freq_a, freq_b in all_pairs:
                try:
                    results, loc = evaluate_pair(df, freq_a, freq_b, axis, category)
                except Exception as e:
                    print(f"  ({freq_a}, {freq_b}): ERROR {e}")
                    continue
                
                if not results:
                    continue
                
                errors = [r['error'] for r in results]
                nA, mA = HARMONIC_COEFFS[freq_a]
                nB, mB = HARMONIC_COEFFS[freq_b]
                
                row = {
                    'axis': axis,
                    'category': category,
                    'freq_a': freq_a,
                    'freq_b': freq_b,
                    'n_points': len(errors),
                    'mean_error': np.mean(errors),
                    'median_error': np.median(errors),
                    'p25_error': np.percentile(errors, 25),
                    'p75_error': np.percentile(errors, 75),
                    'p90_error': np.percentile(errors, 90),
                    'min_error': np.min(errors),
                    'max_error': np.max(errors),
                    'f_eff_1': (nA + nB) * F1 + (mA + mB) * F2,
                    'f_eff_2': (2*nB - nA) * F1 + (2*mB - mA) * F2,
                    'pair_label': f"({freq_a},{freq_b})",
                }
                all_data.append(row)
                
                for r in results:
                    r['pair'] = f"({freq_a},{freq_b})"
                    r['axis'] = axis
                    r['category'] = category
                
                print(f"  ({freq_a:>4}, {freq_b:>4}) → "
                      f"median={np.median(errors):6.1f}mm  "
                      f"mean={np.mean(errors):6.1f}mm  "
                      f"n={len(errors)}")
    
    # Save summary
    summary_df = pd.DataFrame(all_data)
    summary_df.to_csv(OUTDIR / 'freq_pair_comparison.csv', index=False)
    print(f"\n✓ Saved {OUTDIR / 'freq_pair_comparison.csv'}")
    
    # ──────────────────────────────────────────────────────────────
    # Visualization
    # ──────────────────────────────────────────────────────────────
    
    # 1. Heatmap: median error for each pair, per category/axis
    for axis in axes_to_test:
        dim = 'Lateral (X)' if axis == 'x' else 'Depth (Y)'
        sub = summary_df[summary_df['axis'] == axis]
        
        fig, axes_arr = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f'Harmonic Comb: Median Error by Frequency Pair — {dim} Axis',
                     fontsize=14, fontweight='bold')
        
        for ci, cat in enumerate(categories):
            ax = axes_arr[ci]
            cat_sub = sub[sub['category'] == cat]
            
            if cat_sub.empty:
                ax.set_title(f'{cat} — no data')
                continue
            
            # Create matrix
            freqs = sorted(all_freqs)
            n = len(freqs)
            matrix = np.full((n, n), np.nan)
            
            for _, row in cat_sub.iterrows():
                i = freqs.index(row['freq_a'])
                j = freqs.index(row['freq_b'])
                matrix[i, j] = row['median_error']
                matrix[j, i] = row['median_error']
            
            im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='equal',
                          vmin=0, vmax=min(200, np.nanmax(matrix)))
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(freqs, rotation=45, fontsize=8)
            ax.set_yticklabels(freqs, fontsize=8)
            ax.set_title(f'{cat}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Freq B (MHz)')
            ax.set_ylabel('Freq A (MHz)')
            
            # Annotate cells
            for i in range(n):
                for j in range(n):
                    if not np.isnan(matrix[i, j]):
                        val = matrix[i, j]
                        color = 'white' if val > 100 else 'black'
                        ax.text(j, i, f'{val:.0f}',
                               ha='center', va='center',
                               fontsize=7, color=color, fontweight='bold')
            
            plt.colorbar(im, ax=ax, label='Median Error (mm)', shrink=0.8)
        
        plt.tight_layout()
        fig.savefig(OUTDIR / f'freq_pair_heatmap_{axis}.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Saved {OUTDIR / f'freq_pair_heatmap_{axis}.png'}")
    
    # 2. Bar chart: top-5 best and worst pairs per condition
    for axis in axes_to_test:
        dim = 'Lateral (X)' if axis == 'x' else 'Depth (Y)'
        sub = summary_df[summary_df['axis'] == axis]
        
        fig, axes_arr = plt.subplots(3, 1, figsize=(14, 12))
        fig.suptitle(f'Best & Worst Frequency Pairs — {dim} Axis',
                     fontsize=14, fontweight='bold')
        
        for ci, cat in enumerate(categories):
            ax = axes_arr[ci]
            cat_sub = sub[sub['category'] == cat].sort_values('median_error')
            
            if cat_sub.empty:
                ax.set_title(f'{cat} — no data')
                continue
            
            # Show all pairs sorted
            labels = cat_sub['pair_label'].values
            medians = cat_sub['median_error'].values
            
            colors = ['#4caf50' if m < 30 else '#ff9800' if m < 80 
                      else '#f44336' for m in medians]
            
            ax.barh(range(len(labels)), medians, color=colors, alpha=0.8)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel('Median Error (mm)')
            ax.set_title(f'{cat}', fontsize=12, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(True, axis='x', alpha=0.2)
            
            # Annotate values
            for i, (lbl, med) in enumerate(zip(labels, medians)):
                ax.text(med + 2, i, f'{med:.1f}mm', va='center', fontsize=8)
        
        plt.tight_layout()
        fig.savefig(OUTDIR / f'freq_pair_ranking_{axis}.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Saved {OUTDIR / f'freq_pair_ranking_{axis}.png'}")
    
    # 3. Print best pair per condition
    print(f"\n{'═' * 70}")
    print(f"  BEST FREQUENCY PAIRS (by median error)")
    print(f"{'═' * 70}")
    print(f"  {'Condition':<20}  {'Best Pair':>12}  {'Median':>10}  {'Eff. Freqs':>20}")
    print(f"  {'─' * 20}  {'─' * 12}  {'─' * 10}  {'─' * 20}")
    
    for axis in axes_to_test:
        for cat in categories:
            mask = (summary_df['axis'] == axis) & (summary_df['category'] == cat)
            sub = summary_df.loc[mask]
            if sub.empty:
                continue
            best = sub.loc[sub['median_error'].idxmin()]
            dim = 'X' if axis == 'x' else 'Y'
            eff = f"{best['f_eff_1']/1e6:.0f},{best['f_eff_2']/1e6:.0f}"
            print(f"  {cat}/{dim:<16}  {best['pair_label']:>12}  "
                  f"{best['median_error']:>8.1f}mm  {eff:>20}")
    
    print(f"{'═' * 70}")


if __name__ == '__main__':
    main()
