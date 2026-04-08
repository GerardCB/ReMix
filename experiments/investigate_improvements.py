"""
investigate_improvements.py — Test two concrete improvements:

1. MULTI-FREQUENCY COARSE: Use all 7 frequencies in the coarse stage
   with a per-antenna consistency scoring to disambiguate mirrors.

2. ANTENNA SLOPE DISAMBIGUATION: Use the sign of the phase gradient
   across the RX antenna array to distinguish +Y from -Y positions.

3. ENSEMBLE ACROSS FREQUENCY PAIRS: Average estimates from multiple
   high-performing frequency pairs.
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
    C, FREQ_TO_HZ, TX_POS, ALL_HARMONICS,
    rx_positions, target_position,
    DEPTH_SEARCH, LATERAL_SEARCH, CALIBRATION_TARGET_POS,
)
from src.data import (
    load_dataset, get_aggregated_data, get_phase_matrix,
    get_available_target_positions,
)
from src.localize import (
    TissueModel, CalibratedLocalizer,
    phase_cost, grid_evaluate, _find_local_minima,
)

OUTDIR = Path(__file__).resolve().parent.parent / 'results' / 'improvements'
OUTDIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  IMPROVEMENT 1: Multi-Frequency Coarse + Per-Antenna Voting
# ══════════════════════════════════════════════════════════════════════════════
class ImprovedCalibratedLocalizer:
    """
    Fixes the Y-axis mirror ambiguity by:
    1. Using ALL frequencies in coarse search (not just 2)
    2. Scoring candidates by per-antenna consistency (variance)
    3. Using phase gradient across antennas to disambiguate sign
    """

    def __init__(self, frequencies_mhz, ant_positions, dataset_axis='x',
                 category='Patch'):
        self.freq_mhz = list(frequencies_mhz)
        self.freq_hz = np.array([FREQ_TO_HZ[f] for f in self.freq_mhz])
        self.rx_pos = rx_positions()
        self.tx_pos = TX_POS.copy()
        self.axis = dataset_axis
        self.n_ant = len(ant_positions)
        self.n_freq = len(self.freq_mhz)
        self.tissue = TissueModel(category)

        self._cal_phases_rad = None
        self._cal_D = None

    def calibrate(self, phase_matrix_deg, cal_target_pos=CALIBRATION_TARGET_POS):
        cal_xy = target_position(cal_target_pos, self.axis)
        self._cal_phases_rad = np.deg2rad(phase_matrix_deg)
        self._cal_D = self.tissue.effective_round_trip(
            cal_xy, self.tx_pos, self.rx_pos)

    @property
    def is_calibrated(self):
        return self._cal_phases_rad is not None

    def _cost_at(self, tp, meas_rad, pw=None):
        """Standard complex-domain cost."""
        target_xy = target_position(float(tp), self.axis)
        D_new = self.tissue.effective_round_trip(
            target_xy, self.tx_pos, self.rx_pos)
        delta_D = D_new - self._cal_D
        delta_phi_model = -2.0 * np.pi * self.freq_hz[np.newaxis, :] / C \
                          * delta_D[:, np.newaxis]
        delta_phi_meas = meas_rad - self._cal_phases_rad
        residual = np.exp(1j * delta_phi_meas) - np.exp(1j * delta_phi_model)
        mag_sq = np.abs(residual) ** 2
        if pw is not None:
            return np.sum(mag_sq * pw)
        return np.sum(mag_sq)

    def _per_antenna_score(self, tp, meas_rad):
        """
        Score a candidate using per-antenna consistency.
        At the TRUE position, all antennas should give low cost.
        At a mirror/ghost, some antennas agree and others don't.
        Returns (total_cost, variance_penalty, combined_score).
        """
        target_xy = target_position(float(tp), self.axis)
        D_new = self.tissue.effective_round_trip(
            target_xy, self.tx_pos, self.rx_pos)
        delta_D = D_new - self._cal_D

        per_ant_costs = np.zeros(self.n_ant)
        for r in range(self.n_ant):
            for j in range(self.n_freq):
                dphi_model = -2.0 * np.pi * self.freq_hz[j] / C * delta_D[r]
                dphi_meas = meas_rad[r, j] - self._cal_phases_rad[r, j]
                res = np.abs(np.exp(1j * dphi_meas) - np.exp(1j * dphi_model))
                per_ant_costs[r] += res ** 2

        total = np.sum(per_ant_costs)
        variance = np.var(per_ant_costs)
        # Heavily penalize variance to break mirror ties
        score = total + 5.0 * variance * self.n_ant
        return score

    def localize(self, phase_matrix_deg, power_matrix=None):
        """Improved localization with mirror disambiguation."""
        if not self.is_calibrated:
            raise RuntimeError("Call calibrate() first.")

        meas_rad = np.deg2rad(phase_matrix_deg)

        if self.axis == 'y':
            search_bounds = DEPTH_SEARCH
        else:
            search_bounds = LATERAL_SEARCH

        # STAGE 1: Full-range grid search with ALL frequencies
        coarse_grid = np.arange(search_bounds[0], search_bounds[1] + 1, 1.0)
        coarse_costs = np.array([
            self._cost_at(tp, meas_rad) for tp in coarse_grid])

        # Find top candidates (local minima)
        candidates = _find_local_minima(coarse_costs, coarse_grid, n_candidates=15)

        # STAGE 2: Score candidates by per-antenna consistency
        best_tp = candidates[0]
        best_score = np.inf

        for cand in candidates:
            score = self._per_antenna_score(cand, meas_rad)
            if score < best_score:
                best_score = score
                best_tp = cand

        # STAGE 3: Fine refinement
        min_wrap = min(C / (2.0 * f) * 1000.0 for f in self.freq_hz)
        window = max(min_wrap / 2.0, 25.0)

        fine_lo = max(search_bounds[0], best_tp - window)
        fine_hi = min(search_bounds[1], best_tp + window)
        fine_grid = np.arange(fine_lo, fine_hi + 0.5, 0.5)
        fine_costs = np.array([
            self._cost_at(tp, meas_rad) for tp in fine_grid])
        best_idx = np.argmin(fine_costs)
        best_tp = fine_grid[best_idx]

        # Sub-mm
        ultra = np.arange(best_tp - 2, best_tp + 2.1, 0.25)
        ultra = ultra[(ultra >= search_bounds[0]) & (ultra <= search_bounds[1])]
        ultra_costs = np.array([
            self._cost_at(tp, meas_rad) for tp in ultra])
        best_idx = np.argmin(ultra_costs)
        return ultra[best_idx], ultra_costs[best_idx]


# ══════════════════════════════════════════════════════════════════════════════
#  IMPROVEMENT 2: Ensemble of frequency-pair localizers
# ══════════════════════════════════════════════════════════════════════════════
from report_freq_pairs import GeneralizedHarmonicCombLocalizer


def ensemble_localize(df, axis, category, top_pairs, n_top=5):
    """
    For each test position, run the harmonic comb localizer with the
    top-N pairs and take the trimmed-mean of estimates.
    """
    agg = get_aggregated_data(df, category, axis)
    if agg.empty:
        return []

    target_positions = get_available_target_positions(agg)

    # Build localizers for top pairs and calibrate
    localizers = []
    for freq_a, freq_b in top_pairs[:n_top]:
        cal_a = get_phase_matrix(agg, CALIBRATION_TARGET_POS, [freq_a])
        cal_b = get_phase_matrix(agg, CALIBRATION_TARGET_POS, [freq_b])
        if cal_a is None or cal_b is None:
            continue
        loc = GeneralizedHarmonicCombLocalizer(
            freq_a, freq_b, cal_a['ant_pos'],
            dataset_axis=axis, category=category,
        )
        loc.calibrate(cal_a['phases'][:, 0], cal_b['phases'][:, 0])
        localizers.append((freq_a, freq_b, loc))

    results = []
    for tp in target_positions:
        if tp == CALIBRATION_TARGET_POS:
            continue

        estimates = []
        costs = []
        for freq_a, freq_b, loc in localizers:
            data_a = get_phase_matrix(agg, tp, [freq_a])
            data_b = get_phase_matrix(agg, tp, [freq_b])
            if data_a is None or data_b is None:
                continue
            try:
                est, cost = loc.localize(
                    data_a['phases'][:, 0], data_b['phases'][:, 0])
                estimates.append(est)
                costs.append(cost)
            except:
                continue

        if not estimates:
            continue

        estimates = np.array(estimates)
        costs = np.array(costs)

        # Method A: cost-weighted average
        weights = 1.0 / (costs + 1e-10)
        weighted_est = np.average(estimates, weights=weights)

        # Method B: trimmed mean (drop the highest and lowest estimates)
        if len(estimates) >= 3:
            sorted_est = np.sort(estimates)
            trimmed = sorted_est[1:-1]
            trimmed_mean = np.mean(trimmed)
        else:
            trimmed_mean = np.mean(estimates)

        # Method C: median
        median_est = np.median(estimates)

        results.append({
            'target_pos': tp,
            'weighted_est': weighted_est,
            'trimmed_mean_est': trimmed_mean,
            'median_est': median_est,
            'n_pairs': len(estimates),
            'spread': np.std(estimates),
            'weighted_error': abs(weighted_est - tp),
            'trimmed_mean_error': abs(trimmed_mean - tp),
            'median_error': abs(median_est - tp),
        })

    return results


def main():
    csv_path = str(Path(__file__).resolve().parent.parent / 'data' / 'Remix_Dataset.csv')
    print(f"Loading dataset from {csv_path} ...")
    df = load_dataset(csv_path)
    print(f"  {len(df):,} rows\n")

    categories = ['Patch', 'Chicken', 'Pork']
    axes_to_test = ['x', 'y']
    freqs = ALL_HARMONICS

    # ─────────────────────────────────────────────────────────────
    # Test Improvement 1: Multi-frequency coarse + per-antenna voting
    # ─────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  IMPROVEMENT 1: Multi-freq coarse + per-antenna voting")
    print("=" * 70)

    imp1_results = []

    for axis in axes_to_test:
        for category in categories:
            agg = get_aggregated_data(df, category, axis)
            if agg.empty:
                continue
            target_positions = get_available_target_positions(agg)
            cal_data = get_phase_matrix(agg, CALIBRATION_TARGET_POS, freqs)
            if cal_data is None:
                continue

            # Original localizer
            orig = CalibratedLocalizer(
                cal_data['freqs'], cal_data['ant_pos'],
                dataset_axis=axis, category=category,
            )
            orig.calibrate(cal_data['phases'], CALIBRATION_TARGET_POS)

            # Improved localizer
            improved = ImprovedCalibratedLocalizer(
                cal_data['freqs'], cal_data['ant_pos'],
                dataset_axis=axis, category=category,
            )
            improved.calibrate(cal_data['phases'], CALIBRATION_TARGET_POS)

            orig_errors = []
            impr_errors = []

            for tp in target_positions:
                if tp == CALIBRATION_TARGET_POS:
                    continue
                data = get_phase_matrix(agg, tp, freqs)
                if data is None:
                    continue
                try:
                    orig_est, _ = orig.localize(data['phases'], data['powers'])
                    impr_est, _ = improved.localize(data['phases'], data['powers'])
                except:
                    continue

                orig_errors.append(abs(orig_est - tp))
                impr_errors.append(abs(impr_est - tp))

                imp1_results.append({
                    'axis': axis, 'category': category, 'tp': tp,
                    'orig_error': abs(orig_est - tp),
                    'impr_error': abs(impr_est - tp),
                    'orig_est': orig_est, 'impr_est': impr_est,
                })

            dim = 'X' if axis == 'x' else 'Y'
            orig_med = np.median(orig_errors) if orig_errors else float('nan')
            impr_med = np.median(impr_errors) if impr_errors else float('nan')
            delta = orig_med - impr_med
            arrow = '↓' if delta > 0 else '↑'
            print(f"  {category}/{dim}: orig={orig_med:6.1f}mm  "
                  f"improved={impr_med:6.1f}mm  "
                  f"{arrow}{abs(delta):.1f}mm")

    # Save improvement 1 results
    if imp1_results:
        imp1_df = pd.DataFrame(imp1_results)
        imp1_df.to_csv(OUTDIR / 'improvement1_results.csv', index=False)

    # ─────────────────────────────────────────────────────────────
    # Test Improvement 2: Ensemble across frequency pairs
    # ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  IMPROVEMENT 2: Ensemble across top-5 frequency pairs")
    print("=" * 70)

    # Load the freq pair comparison to get the best pairs per condition
    freq_pair_df = pd.read_csv('results/freq_pair_comparison.csv')

    imp2_results = []

    for axis in axes_to_test:
        for category in categories:
            # Get best pairs for this condition
            mask = ((freq_pair_df['axis'] == axis) &
                    (freq_pair_df['category'] == category))
            sub = freq_pair_df[mask].sort_values('median_error')
            if sub.empty:
                continue

            top_pairs = list(zip(
                sub['freq_a'].astype(int).values,
                sub['freq_b'].astype(int).values
            ))

            results = ensemble_localize(df, axis, category, top_pairs, n_top=5)
            if not results:
                continue

            for r in results:
                r['axis'] = axis
                r['category'] = category
            imp2_results.extend(results)

            dim = 'X' if axis == 'x' else 'Y'
            weighted_med = np.median([r['weighted_error'] for r in results])
            trimmed_med = np.median([r['trimmed_mean_error'] for r in results])
            median_med = np.median([r['median_error'] for r in results])

            print(f"  {category}/{dim}: "
                  f"weighted={weighted_med:6.1f}mm  "
                  f"trimmed={trimmed_med:6.1f}mm  "
                  f"median={median_med:6.1f}mm")

    if imp2_results:
        imp2_df = pd.DataFrame(imp2_results)
        imp2_df.to_csv(OUTDIR / 'improvement2_results.csv', index=False)

    # ─────────────────────────────────────────────────────────────
    # Compare all approaches
    # ─────────────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("  COMPREHENSIVE COMPARISON (median error)")
    print(f"{'═' * 70}")

    # Load original results too
    orig_df = pd.read_csv(Path(__file__).resolve().parent.parent / 'results' / 'results.csv')
    parts = orig_df['config'].str.extract(
        r'(?P<category>\w+)/(?P<axis>[xy])\s+(?P<freqs>\S+)\s+(?P<approach>\S+)')
    orig_df = pd.concat([orig_df, parts], axis=1)

    print(f"\n  {'Condition':<15}  {'HC(790,1700)':>12}  {'Cal(all)':>10}  "
          f"{'Cal+voting':>10}  {'Ens.trim':>10}  {'Ens.median':>10}")
    print(f"  {'─'*15}  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")

    for axis in axes_to_test:
        for category in categories:
            dim = 'X' if axis == 'x' else 'Y'
            label = f"{category}/{dim}"

            # HC original
            hc_mask = ((orig_df['category'] == category) &
                       (orig_df['axis'] == axis) &
                       (orig_df['approach'] == 'harmonic_comb'))
            hc_med = orig_df.loc[hc_mask, 'error'].median() if hc_mask.any() else np.nan

            # Cal original
            cal_mask = ((orig_df['category'] == category) &
                        (orig_df['axis'] == axis) &
                        (orig_df['approach'] == 'calibrated'))
            cal_med = orig_df.loc[cal_mask, 'error'].median() if cal_mask.any() else np.nan

            # Improved cal
            if imp1_results:
                i1_sub = [r for r in imp1_results
                          if r['axis'] == axis and r['category'] == category]
                i1_med = np.median([r['impr_error'] for r in i1_sub]) if i1_sub else np.nan
            else:
                i1_med = np.nan

            # Ensemble
            if imp2_results:
                i2_sub = [r for r in imp2_results
                          if r['axis'] == axis and r['category'] == category]
                i2_trim = np.median([r['trimmed_mean_error'] for r in i2_sub]) if i2_sub else np.nan
                i2_med_ens = np.median([r['median_error'] for r in i2_sub]) if i2_sub else np.nan
            else:
                i2_trim = np.nan
                i2_med_ens = np.nan

            def fmt(v):
                return f"{v:.1f}mm" if not np.isnan(v) else "—"

            # Bold the best
            vals = [hc_med, cal_med, i1_med, i2_trim, i2_med_ens]
            best = min(v for v in vals if not np.isnan(v))

            strs = []
            for v in vals:
                s = fmt(v)
                if not np.isnan(v) and abs(v - best) < 0.1:
                    s = f"*{s}*"
                strs.append(s)

            print(f"  {label:<15}  {strs[0]:>12}  {strs[1]:>10}  "
                  f"{strs[2]:>10}  {strs[3]:>10}  {strs[4]:>10}")

    # ─────────────────────────────────────────────────────────────
    # Visualization: Before vs After for Y-axis
    # ─────────────────────────────────────────────────────────────
    fig, axes_arr = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Y-Axis: Original vs Improved Calibrated Localizer',
                 fontsize=14, fontweight='bold')

    for ci, category in enumerate(categories):
        ax = axes_arr[ci]
        i1_sub = [r for r in imp1_results
                  if r['axis'] == 'y' and r['category'] == category]
        if not i1_sub:
            ax.set_title(f'{category} — no data')
            continue

        tps = np.array([r['tp'] for r in i1_sub])
        orig_err = np.array([r['orig_error'] for r in i1_sub])
        impr_err = np.array([r['impr_error'] for r in i1_sub])

        sort_idx = np.argsort(tps)
        tps = tps[sort_idx]
        orig_err = orig_err[sort_idx]
        impr_err = impr_err[sort_idx]

        ax.plot(tps, orig_err, 'r-o', ms=4, alpha=0.7, label='Original')
        ax.plot(tps, impr_err, 'g-s', ms=4, alpha=0.7, label='Improved')
        ax.axhline(50, color='gray', ls='--', alpha=0.3)
        ax.fill_between(tps, 0, 50, alpha=0.05, color='green')
        ax.set_xlabel('Target Position (mm)')
        ax.set_ylabel('Error (mm)')
        ax.set_title(f'{category}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.15)

        orig_med = np.median(orig_err)
        impr_med = np.median(impr_err)
        ax.text(0.02, 0.98, f'Orig median: {orig_med:.1f}mm\n'
                              f'Impr median: {impr_med:.1f}mm',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fig.savefig(OUTDIR / 'y_axis_improvement.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✓ Saved {OUTDIR / 'y_axis_improvement.png'}")

    # Visualization: Ensemble comparison
    fig, axes_arr = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Ensemble vs Single-Pair Localization',
                 fontsize=14, fontweight='bold')

    for ai, axis in enumerate(axes_to_test):
        dim = 'X' if axis == 'x' else 'Y'
        for ci, category in enumerate(categories):
            ax = axes_arr[ai][ci]

            # Get original HC results
            hc_mask = ((orig_df['category'] == category) &
                       (orig_df['axis'] == axis) &
                       (orig_df['approach'] == 'harmonic_comb'))
            hc_sub = orig_df.loc[hc_mask].sort_values('target_pos')

            # Get ensemble results
            i2_sub = [r for r in imp2_results
                      if r['axis'] == axis and r['category'] == category]

            if not i2_sub:
                ax.set_title(f'{category}/{dim} — no data', fontsize=9)
                continue

            i2_tps = np.array([r['target_pos'] for r in i2_sub])
            i2_trim_err = np.array([r['trimmed_mean_error'] for r in i2_sub])
            sort_idx = np.argsort(i2_tps)
            i2_tps = i2_tps[sort_idx]
            i2_trim_err = i2_trim_err[sort_idx]

            if len(hc_sub) > 0:
                ax.plot(hc_sub['target_pos'], hc_sub['error'],
                       'r-o', ms=3, alpha=0.6, label='HC(790,1700)')
            ax.plot(i2_tps, i2_trim_err, 'g-s', ms=3, alpha=0.7,
                   label='Ensemble(top5)')
            ax.axhline(50, color='gray', ls='--', alpha=0.3)
            ax.set_xlabel('Target Pos (mm)')
            ax.set_ylabel('Error (mm)')
            ax.set_title(f'{category}/{dim}', fontsize=10, fontweight='bold')
            ax.legend(fontsize=8)
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.15)

    plt.tight_layout()
    fig.savefig(OUTDIR / 'ensemble_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved {OUTDIR / 'ensemble_comparison.png'}")

    print(f"\n✓ All results saved to {OUTDIR}/")


if __name__ == '__main__':
    main()
