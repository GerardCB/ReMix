"""
remix_localize.py — Localization algorithms with tissue propagation model
                     and hierarchical phase-ambiguity resolution.

Key insight for phase ambiguity on the depth axis:
  Each frequency f has a phase wrap period of λ_wrap = c/(2f) mm of target
  displacement.  Over the ±200mm measurement range:
    790 MHz → λ_wrap = 190mm → ~2.1 wraps  (smooth cost, few minima)
    910 MHz → λ_wrap = 165mm → ~2.4 wraps
   1700 MHz → λ_wrap =  88mm → ~4.5 wraps  (many minima, easy to trap)

  Using two frequencies, the unambiguous range is c/(2·Δf):
    [790, 910]:  Δf = 120 MHz → 1250mm  ≫ 400mm search range  
    [790, 1700]: Δf = 910 MHz →  165mm  < 400mm → ghosts!     
    [1620,1660]: Δf =  40 MHz → 3750mm  ✓ but noisy high-freq

  Strategy:
    1. COARSE: Use the two lowest available frequencies [790, 910]
       → unambiguous, smooth cost, robust to noise
    2. FINE: Refine in ±50mm window using ALL frequencies
       → high precision (sub-mm in theory), ghosts excluded by window
    3. If only 2 frequencies available (pair_a): use antenna diversity
       to pick the best among local minima
"""

import numpy as np
from scipy.optimize import brentq, differential_evolution

from src.config import (
    C, FREQ_TO_HZ, TX_POS,
    rx_positions, target_position,
    DEPTH_SEARCH, LATERAL_SEARCH, CALIBRATION_TARGET_POS,
    get_tissue_layers, get_tissue_wall,
)


# ══════════════════════════════════════════════════════════════════════════════
#  TISSUE-AWARE DISTANCE: Refraction solver (Snell's law)
# ══════════════════════════════════════════════════════════════════════════════
class TissueModel:
    """
    Models RF propagation through a layered tissue slab.
    See ReMix 7.2: piecewise-linear spline with Snell's law at interfaces.
    """

    def __init__(self, category='Patch'):
        self.category = category
        self.layers = get_tissue_layers(category)
        self.wall = get_tissue_wall(category)
        self._is_free_space = (len(self.layers) == 0 or self.wall <= 0)

    def effective_distance_one_way(self, antenna_xy, device_xy):
        x_a, y_a = antenna_xy
        x_d, y_d = device_xy

        if self._is_free_space:
            return np.sqrt((x_d - x_a)**2 + (y_d - y_a)**2) / 1000.0

        y_surface = y_d - self.wall
        h_air = y_surface - y_a
        dx = x_d - x_a

        if h_air <= 0:
            d_geom = np.sqrt(dx**2 + (y_d - y_a)**2)
            alpha_avg = np.mean([a for a, _ in self.layers]) if self.layers else 1.0
            return (alpha_avg * d_geom) / 1000.0

        abs_dx = abs(dx)
        if abs_dx < 1e-6:
            d_eff = h_air
            for alpha_i, h_i in self.layers:
                d_eff += alpha_i * h_i
            return d_eff / 1000.0

        layer_alphas = np.array([a for a, _ in self.layers])
        layer_thicks = np.array([h for _, h in self.layers])

        def _horizontal_displacement(kappa):
            if kappa >= 1.0 - 1e-15:
                return 1e12
            disp = h_air * kappa / np.sqrt(1.0 - kappa**2)
            for alpha_i, h_i in zip(layer_alphas, layer_thicks):
                sin_i = kappa / alpha_i
                if sin_i >= 1.0 - 1e-15:
                    return 1e12
                disp += h_i * sin_i / np.sqrt(1.0 - sin_i**2)
            return disp

        kappa_max = 1.0 - 1e-12
        disp_max = _horizontal_displacement(kappa_max)

        if disp_max < abs_dx:
            d_geom = np.sqrt(dx**2 + (y_d - y_a)**2) / 1000.0
            tissue_fraction = self.wall / (self.wall + h_air)
            alpha_avg = np.mean(layer_alphas) if len(layer_alphas) else 1.0
            return d_geom * (1.0 + (alpha_avg - 1.0) * tissue_fraction)

        kappa_sol = brentq(lambda k: _horizontal_displacement(k) - abs_dx,
                           0.0, kappa_max, xtol=1e-14, maxiter=200)

        cos_air = np.sqrt(1.0 - kappa_sol**2)
        d_eff = h_air / cos_air
        for alpha_i, h_i in zip(layer_alphas, layer_thicks):
            sin_i = kappa_sol / alpha_i
            cos_i = np.sqrt(1.0 - sin_i**2)
            d_eff += alpha_i * (h_i / cos_i)
        return d_eff / 1000.0

    def effective_round_trip(self, device_xy, tx_xy, rx_xy_array):
        d_tx = self.effective_distance_one_way(tx_xy, device_xy)
        n_ant = rx_xy_array.shape[0]
        D_eff = np.empty(n_ant)
        for r in range(n_ant):
            D_eff[r] = d_tx + self.effective_distance_one_way(rx_xy_array[r], device_xy)
        return D_eff


FREE_SPACE = TissueModel('Patch')


def free_space_round_trip(device_xy, tx_xy, rx_xy_array):
    d_tx = np.linalg.norm(device_xy - tx_xy) / 1000.0
    d_rx = np.linalg.norm(rx_xy_array - device_xy, axis=1) / 1000.0
    return d_tx + d_rx


# ══════════════════════════════════════════════════════════════════════════════
#  Phase cost functions
# ══════════════════════════════════════════════════════════════════════════════
def phase_cost(tp, axis, freq_hz, rx_pos, tx_pos, cal_D, meas_rad, cal_rad,
               tissue_model, power_weights=None):
    """
    Complex-domain phase cost for a candidate Target_Pos.
    cost = Σ_{r,f} | exp(j·Δφ_meas) - exp(j·Δφ_model) |²
    """
    tp = float(tp)
    target_xy = target_position(tp, axis)
    D_new = tissue_model.effective_round_trip(target_xy, tx_pos, rx_pos)
    delta_D = D_new - cal_D

    delta_phi_model = -2.0 * np.pi * freq_hz[np.newaxis, :] / C \
                      * delta_D[:, np.newaxis]
    delta_phi_meas = meas_rad - cal_rad

    residual = np.exp(1j * delta_phi_meas) - np.exp(1j * delta_phi_model)
    mag_sq = np.abs(residual) ** 2

    if power_weights is not None:
        return np.sum(mag_sq * power_weights)
    return np.sum(mag_sq)


def grid_evaluate(search_range, axis, freq_hz, rx_pos, tx_pos, cal_D,
                  meas_rad, cal_rad, tissue_model, pw=None):
    """Evaluate phase_cost over a 1-D grid."""
    costs = np.empty(len(search_range))
    for i, tp in enumerate(search_range):
        costs[i] = phase_cost(tp, axis, freq_hz, rx_pos, tx_pos, cal_D,
                              meas_rad, cal_rad, tissue_model, pw)
    return costs


def per_frequency_grid(search_range, axis, freq_hz, rx_pos, tx_pos, cal_D,
                       meas_rad, cal_rad, tissue_model):
    """Per-frequency cost over a grid. Returns (n_grid, n_freq)."""
    n_grid = len(search_range)
    n_freq = len(freq_hz)
    costs = np.empty((n_grid, n_freq))
    for i, tp in enumerate(search_range):
        target_xy = target_position(float(tp), axis)
        D_new = tissue_model.effective_round_trip(target_xy, tx_pos, rx_pos)
        delta_D = D_new - cal_D
        delta_phi_model = -2.0 * np.pi * freq_hz[np.newaxis, :] / C \
                          * delta_D[:, np.newaxis]
        delta_phi_meas = meas_rad - cal_rad
        residual = np.exp(1j * delta_phi_meas) - np.exp(1j * delta_phi_model)
        costs[i, :] = np.sum(np.abs(residual) ** 2, axis=0)
    return costs


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════
def _find_local_minima(costs, grid, n_candidates=10):
    """Find top-N local minima in cost curve, sorted by cost."""
    n = len(costs)
    if n < 3:
        return np.array([grid[np.argmin(costs)]])
    is_min = np.zeros(n, dtype=bool)
    is_min[0]    = costs[0] <= costs[1]
    is_min[-1]   = costs[-1] <= costs[-2]
    is_min[1:-1] = (costs[1:-1] <= costs[:-2]) & (costs[1:-1] <= costs[2:])
    min_idx = np.where(is_min)[0]
    if len(min_idx) == 0:
        return np.array([grid[np.argmin(costs)]])
    sorted_order = np.argsort(costs[min_idx])
    top_idx = min_idx[sorted_order[:n_candidates]]
    return grid[top_idx]


def _select_coarse_pair(freq_mhz_list):
    """
    Select the best coarse pair for unambiguous estimation.

    Priority: the two LOWEST frequencies, because they have:
      1. Longest individual wrap periods → smoothest cost landscape
      2. Their beat period is c/(2·Δf) which for low freqs with
         moderate spacing gives a long unambiguous range.

    For [790, 910, ...]: picks (790, 910), beat = 1250mm 
    For [790, 1700]:     picks (790, 1700), beat = 165mm  (best available)
    For [1620, ...]:     picks two lowest from what's available.
    """
    sorted_freqs = sorted(set(freq_mhz_list))
    if len(sorted_freqs) < 2:
        return None

    f_lo, f_hi = sorted_freqs[0], sorted_freqs[1]
    idx_lo = freq_mhz_list.index(f_lo)
    idx_hi = freq_mhz_list.index(f_hi)
    return (idx_lo, idx_hi)


def _beat_period_mm(f1_hz, f2_hz):
    """Beat wrap period in mm: c / (2 × |f1−f2|) × 1000."""
    df = abs(f1_hz - f2_hz)
    if df < 1.0:
        return np.inf
    return C / (2.0 * df) * 1000.0


# ══════════════════════════════════════════════════════════════════════════════
#  Parametric Tissue Model Adapter
# ══════════════════════════════════════════════════════════════════════════════
class ParametricTissueModelAdapter:
    """
    Wraps a parametric tissue model (e.g., from grid search) and provides
    the same interface as TissueModel for use in localizers.
    """
    
    def __init__(self, category, parametric_model):
        self.category = category
        self.parametric_model = parametric_model
        self.layers = parametric_model.get_tissue_layers(category)
        self.wall = sum(thick for _, thick in self.layers)
        self._is_free_space = (len(self.layers) == 0 or self.wall <= 0)
    
    def effective_distance_one_way(self, antenna_xy, device_xy):
        """Delegate to a TissueModel built from parametric layers."""
        # Create a temporary TissueModel-like object with the parametric layers
        tm = TissueModel.__new__(TissueModel)
        tm.layers = self.layers
        tm.wall = self.wall
        tm._is_free_space = self._is_free_space
        return tm.effective_distance_one_way(antenna_xy, device_xy)
    
    def effective_round_trip(self, device_xy, tx_xy, rx_pos_array):
        """Delegate to a TissueModel built from parametric layers."""
        tm = TissueModel.__new__(TissueModel)
        tm.layers = self.layers
        tm.wall = self.wall
        tm._is_free_space = self._is_free_space
        return tm.effective_round_trip(device_xy, tx_xy, rx_pos_array)


# ══════════════════════════════════════════════════════════════════════════════
#  Calibrated Localizer — hierarchical coarse-to-fine
# ══════════════════════════════════════════════════════════════════════════════
class CalibratedLocalizer:
    """
    Hierarchical coarse-to-fine localizer.

    When ≥ 3 frequencies are available (including 2 low ones for coarse):
      Stage 1 — COARSE: Grid search with the 2 lowest frequencies.
          These have the longest wrap periods → smoothest cost landscape.
          If their beat period > search range, the estimate is unambiguous.
      Stage 2 — FINE: Refine in ±window around coarse estimate using
          ALL frequencies for maximum precision.

    When only 2 frequencies are available:
      Full-range search with both.  Use per-antenna consistency to
      pick among local minima.

    The tissue model accounts for changed wavelength inside biological tissue.
    """

    def __init__(self, frequencies_mhz, ant_positions, dataset_axis='x',
                 category='Patch', tissue_model_override=None):
        self.freq_mhz = list(frequencies_mhz)
        self.freq_hz  = np.array([FREQ_TO_HZ[f] for f in self.freq_mhz])
        self.rx_pos   = rx_positions()
        self.tx_pos   = TX_POS.copy()
        self.axis     = dataset_axis
        self.n_ant    = len(ant_positions)
        self.n_freq   = len(self.freq_mhz)
        
        # Allow override of tissue model (e.g., for grid search with custom params)
        if tissue_model_override is not None:
            self.tissue   = ParametricTissueModelAdapter(category, tissue_model_override)
        else:
            self.tissue   = TissueModel(category)

        self._cal_phases_rad = None
        self._cal_D = None

        # Select coarse pair (two lowest frequencies)
        self._coarse_pair = _select_coarse_pair(self.freq_mhz)
        if self._coarse_pair is not None:
            ci, cj = self._coarse_pair
            self._coarse_beat_mm = _beat_period_mm(
                self.freq_hz[ci], self.freq_hz[cj])
        else:
            self._coarse_beat_mm = 0.0

        # Remaining (non-coarse) frequencies for fine stage
        if self._coarse_pair is not None and self.n_freq > 2:
            ci, cj = self._coarse_pair
            self._fine_idx = np.array([i for i in range(self.n_freq)
                                       if i != ci and i != cj])
            self._has_fine = len(self._fine_idx) > 0
        else:
            self._fine_idx = np.arange(self.n_freq)
            self._has_fine = False

    def calibrate(self, phase_matrix_deg, cal_target_pos=CALIBRATION_TARGET_POS):
        cal_xy = target_position(cal_target_pos, self.axis)
        self._cal_phases_rad = np.deg2rad(phase_matrix_deg)
        self._cal_D = self.tissue.effective_round_trip(cal_xy, self.tx_pos, self.rx_pos)

    @property
    def is_calibrated(self):
        return self._cal_phases_rad is not None

    def localize(self, phase_matrix_deg, power_matrix=None):
        """Estimate Target_Pos.  Returns (est_tp, cost)."""
        if not self.is_calibrated:
            raise RuntimeError("Call calibrate() first.")

        meas_rad = np.deg2rad(phase_matrix_deg)

        pw = None
        if power_matrix is not None:
            lin = 10.0 ** (power_matrix / 10.0)
            lin = np.maximum(lin, 1e-30)
            pw = lin / lin.max()

        if self.axis == 'y':
            search_bounds = DEPTH_SEARCH
        else:
            search_bounds = LATERAL_SEARCH

        search_span = search_bounds[1] - search_bounds[0]

        can_do_hierarchical = (
            self._coarse_pair is not None
            and self.n_freq >= 3
            and self._has_fine
        )

        if can_do_hierarchical:
            return self._hierarchical(meas_rad, pw, search_bounds, search_span)
        else:
            return self._two_freq_robust(meas_rad, pw, search_bounds, search_span)

    def _hierarchical(self, meas_rad, pw, search_bounds, search_span):
        """
        Stage 1: Coarse with 2 lowest frequencies (unambiguous).
        Stage 2: Fine with ALL frequencies in narrow window.
        """
        ci, cj = self._coarse_pair
        coarse_idx = np.array([ci, cj])

        # ── STAGE 1: Coarse grid search ──────────────────────────────
        coarse_grid = np.arange(search_bounds[0], search_bounds[1] + 1, 1.0)
        coarse_costs = grid_evaluate(
            coarse_grid, self.axis,
            self.freq_hz[coarse_idx], self.rx_pos, self.tx_pos, self._cal_D,
            meas_rad[:, coarse_idx],
            self._cal_phases_rad[:, coarse_idx],
            self.tissue,
            pw[:, coarse_idx] if pw is not None else None,
        )

        # If coarse pair is unambiguous, take global min directly
        if self._coarse_beat_mm > search_span:
            coarse_tp = coarse_grid[np.argmin(coarse_costs)]
        else:
            # Multiple minima possible — pick the one that also works
            # well with the fine frequencies
            candidates = _find_local_minima(coarse_costs, coarse_grid,
                                            n_candidates=8)
            best_tp = candidates[0]
            best_cost = np.inf
            for cand in candidates:
                c = phase_cost(
                    cand, self.axis,
                    self.freq_hz, self.rx_pos, self.tx_pos, self._cal_D,
                    meas_rad, self._cal_phases_rad, self.tissue, pw)
                if c < best_cost:
                    best_cost = c
                    best_tp = cand
            coarse_tp = best_tp

        # ── STAGE 2: Fine refinement with ALL frequencies ────────────
        # Window: half the shortest individual wrap period
        min_wrap = min(C / (2.0 * f) * 1000.0 for f in self.freq_hz)
        window = max(min_wrap / 2.0, 30.0)

        fine_lo = max(search_bounds[0], coarse_tp - window)
        fine_hi = min(search_bounds[1], coarse_tp + window)
        fine_grid = np.arange(fine_lo, fine_hi + 0.5, 0.5)

        fine_costs = grid_evaluate(
            fine_grid, self.axis,
            self.freq_hz, self.rx_pos, self.tx_pos, self._cal_D,
            meas_rad, self._cal_phases_rad, self.tissue, pw,
        )
        best_idx = np.argmin(fine_costs)
        best_tp = fine_grid[best_idx]

        # Sub-mm refinement
        ultra = np.arange(best_tp - 2, best_tp + 2.1, 0.25)
        ultra = ultra[(ultra >= search_bounds[0]) & (ultra <= search_bounds[1])]
        ultra_costs = grid_evaluate(
            ultra, self.axis,
            self.freq_hz, self.rx_pos, self.tx_pos, self._cal_D,
            meas_rad, self._cal_phases_rad, self.tissue, pw,
        )
        best_idx = np.argmin(ultra_costs)
        return ultra[best_idx], ultra_costs[best_idx]

    def _two_freq_robust(self, meas_rad, pw, search_bounds, search_span):
        """
        When only 2 frequencies available: full-range search.
        Find all local minima, score each using per-antenna consistency.
        """
        grid = np.arange(search_bounds[0], search_bounds[1] + 1, 1.0)
        costs = grid_evaluate(
            grid, self.axis,
            self.freq_hz, self.rx_pos, self.tx_pos, self._cal_D,
            meas_rad, self._cal_phases_rad, self.tissue, pw,
        )

        # Find local minima
        candidates = _find_local_minima(costs, grid, n_candidates=10)

        if len(candidates) <= 1:
            best_tp = candidates[0]
        else:
            # Score each candidate by per-antenna agreement
            # At the true position, ALL antennas should agree (low variance)
            # At a ghost, some antennas agree and others don't
            best_tp = candidates[0]
            best_score = np.inf

            for cand in candidates:
                target_xy = target_position(float(cand), self.axis)
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

                # Combine: total cost + variance penalty
                # High variance means inconsistent across antennas → ghost
                total = np.sum(per_ant_costs)
                variance = np.var(per_ant_costs)
                score = total + 2.0 * variance * self.n_ant

                if score < best_score:
                    best_score = score
                    best_tp = cand

        # Refine
        fine = np.arange(best_tp - 3, best_tp + 3.25, 0.25)
        fine = fine[(fine >= search_bounds[0]) & (fine <= search_bounds[1])]
        fine_costs = grid_evaluate(
            fine, self.axis,
            self.freq_hz, self.rx_pos, self.tx_pos, self._cal_D,
            meas_rad, self._cal_phases_rad, self.tissue, pw,
        )
        return fine[np.argmin(fine_costs)], fine_costs.min()


# ══════════════════════════════════════════════════════════════════════════════
#  Grid Search Localizer (single-stage baseline)
# ══════════════════════════════════════════════════════════════════════════════
class GridSearchLocalizer:
    def __init__(self, frequencies_mhz, ant_positions, dataset_axis='x',
                 category='Patch'):
        self.freq_mhz = list(frequencies_mhz)
        self.freq_hz  = np.array([FREQ_TO_HZ[f] for f in self.freq_mhz])
        self.rx_pos   = rx_positions()
        self.tx_pos   = TX_POS.copy()
        self.axis     = dataset_axis
        self.tissue   = TissueModel(category)

    def localize(self, phase_matrix_deg, cal_phase_matrix_deg,
                 cal_target_pos=CALIBRATION_TARGET_POS, step=1.0):
        cal_xy = target_position(cal_target_pos, self.axis)
        cal_D = self.tissue.effective_round_trip(cal_xy, self.tx_pos, self.rx_pos)
        meas_rad = np.deg2rad(phase_matrix_deg)
        cal_rad  = np.deg2rad(cal_phase_matrix_deg)
        if self.axis == 'y':
            search = np.arange(DEPTH_SEARCH[0], DEPTH_SEARCH[1] + step, step)
        else:
            search = np.arange(LATERAL_SEARCH[0], LATERAL_SEARCH[1] + step, step)
        costs = grid_evaluate(search, self.axis, self.freq_hz, self.rx_pos,
                              self.tx_pos, cal_D, meas_rad, cal_rad, self.tissue)
        return search[np.argmin(costs)], costs.min()


# ══════════════════════════════════════════════════════════════════════════════
#  TDOA Localizer (phase differences between RX antennas)
# ══════════════════════════════════════════════════════════════════════════════
class TDOALocalizer:
    def __init__(self, frequencies_mhz, ant_positions, dataset_axis='y',
                 ref_idx=2, category='Patch'):
        self.freq_mhz = list(frequencies_mhz)
        self.freq_hz  = np.array([FREQ_TO_HZ[f] for f in self.freq_mhz])
        self.rx_pos   = rx_positions()
        self.tx_pos   = TX_POS.copy()
        self.axis     = dataset_axis
        self.ref_idx  = ref_idx
        self.n_ant    = len(ant_positions)
        self.tissue   = TissueModel(category)

    def _cost(self, tp_candidate, delta_phases_rad):
        tp = float(np.asarray(tp_candidate).ravel()[0])
        target_xy = target_position(tp, self.axis)
        d_rx = np.empty(self.n_ant)
        for r in range(self.n_ant):
            d_rx[r] = self.tissue.effective_distance_one_way(
                self.rx_pos[r], target_xy)
        delta_d = d_rx - d_rx[self.ref_idx]
        delta_phi_model = -2.0 * np.pi * self.freq_hz[np.newaxis, :] / C \
                          * delta_d[:, np.newaxis]
        residual = np.exp(1j * delta_phases_rad) - np.exp(1j * delta_phi_model)
        mask = np.ones(self.n_ant, dtype=bool)
        mask[self.ref_idx] = False
        return np.sum(np.abs(residual[mask, :]) ** 2)

    def localize(self, phase_matrix_deg, method='de'):
        ref_row = phase_matrix_deg[self.ref_idx, :]
        delta_rad = np.deg2rad(phase_matrix_deg - ref_row[np.newaxis, :])
        if self.axis == 'y':
            bounds = [DEPTH_SEARCH]
        else:
            bounds = [LATERAL_SEARCH]
        result = differential_evolution(
            self._cost, bounds, args=(delta_rad,),
            maxiter=500, tol=1e-12, seed=42, polish=True)
        return result.x[0], result.fun


# ══════════════════════════════════════════════════════════════════════════════
#  Harmonic Combination Localizer (ReMix Eq. 14)
# ══════════════════════════════════════════════════════════════════════════════
class HarmonicCombinationLocalizer:
    """
    Combines φ(f₁+f₂) and ψ(2f₁-f₂) into equations at 3f₁ and 3f₂.
    Requires frequencies 790 and 1700 in the data.
    """

    def __init__(self, ant_positions, dataset_axis='y', category='Patch'):
        self.rx_pos   = rx_positions()
        self.tx_pos   = TX_POS.copy()
        self.axis     = dataset_axis
        self.tissue   = TissueModel(category)
        self.n_ant    = len(ant_positions)
        self.f_eff = np.array([3.0 * 830e6, 3.0 * 870e6])
        self._cal_D = None
        self._cal_combined_rad = None

    def calibrate(self, phase_deg_790, phase_deg_1700,
                  cal_target_pos=CALIBRATION_TARGET_POS):
        cal_xy = target_position(cal_target_pos, self.axis)
        self._cal_D = self.tissue.effective_round_trip(cal_xy, self.tx_pos, self.rx_pos)
        phi = np.deg2rad(phase_deg_1700)
        psi = np.deg2rad(phase_deg_790)
        self._cal_combined_rad = np.column_stack([phi + psi, 2*phi - psi])

    @property
    def is_calibrated(self):
        return self._cal_D is not None

    def _cost_at(self, tp, delta_combined):
        target_xy = target_position(tp, self.axis)
        D_new = self.tissue.effective_round_trip(target_xy, self.tx_pos, self.rx_pos)
        delta_D = D_new - self._cal_D
        delta_model = -2.0 * np.pi * self.f_eff[np.newaxis, :] / C \
                      * delta_D[:, np.newaxis]
        residual = np.exp(1j * delta_combined) - np.exp(1j * delta_model)
        return np.sum(np.abs(residual) ** 2)

    def localize(self, phase_deg_790, phase_deg_1700):
        if not self.is_calibrated:
            raise RuntimeError("Call calibrate() first.")
        phi = np.deg2rad(phase_deg_1700)
        psi = np.deg2rad(phase_deg_790)
        meas_combined = np.column_stack([phi + psi, 2*phi - psi])
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