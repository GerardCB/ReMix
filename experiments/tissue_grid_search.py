"""
tissue_grid_search.py — Grid search over tissue thickness parameters.

Searches for optimal fat and muscle thickness values for Chicken and Pork
tissues to minimize localization errors.
"""

import argparse
import sys
import warnings
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.config import (
    PAIR_A, ALL_HARMONICS, CALIBRATION_TARGET_POS,
    REFERENCE_DEPTH, ALPHA, C, FREQ_TO_HZ, TX_POS, rx_positions,
    target_position, get_tissue_wall, EPS_R,
)
from src.data import (
    load_dataset, get_aggregated_data, get_phase_matrix,
    get_available_target_positions,
)
from src.localize import (
    CalibratedLocalizer, TissueModel, phase_cost,
)

warnings.filterwarnings('ignore', category=RuntimeWarning)


class ParametricTissueModel:
    """
    Tissue model where Chicken and Pork parameters can be modified.
    
    Chicken: muscle thickness (mm)
    Pork: fat thickness (mm) + muscle thickness (mm)
    """
    
    def __init__(self):
        self.chicken_muscle_mm = 15.0  # default
        self.pork_fat_mm = 10.0        # default
        self.pork_muscle_mm = 20.0     # default
    
    def get_tissue_profile(self, category):
        """Return [(material, thickness_mm), ...] for category."""
        if category == 'Patch':
            return []
        elif category == 'Chicken':
            return [('muscle', self.chicken_muscle_mm)]
        elif category == 'Pork':
            return [('fat', self.pork_fat_mm), ('muscle', self.pork_muscle_mm)]
        else:
            return []
    
    def get_tissue_layers(self, category):
        """Return [(alpha, thickness_mm), ...] for category."""
        profile = self.get_tissue_profile(category)
        return [(np.sqrt(EPS_R[mat]), thick) for mat, thick in profile]


def evaluate_config(agg_df, axis, freqs, category, params, tp_min=None, tp_max=None):
    """
    Evaluate localization error for a given tissue parameter configuration.
    Returns mean error, median error, and count of successful localizations.
    """
    target_positions = get_available_target_positions(agg_df)
    if tp_min is not None or tp_max is not None:
        target_positions = [tp for tp in target_positions
                            if (tp_min is None or tp >= tp_min)
                            and (tp_max is None or tp <= tp_max)]
    
    cal_data = get_phase_matrix(agg_df, CALIBRATION_TARGET_POS, freqs)
    if cal_data is None:
        return None, None, 0
    
    # Create parametric tissue model with current config
    tissue_model = ParametricTissueModel()
    tissue_model.chicken_muscle_mm = params.get('chicken_muscle', 15.0)
    tissue_model.pork_fat_mm = params.get('pork_fat', 10.0)
    tissue_model.pork_muscle_mm = params.get('pork_muscle', 20.0)
    
    # Create localizer with parametric tissue model
    localizer = CalibratedLocalizer(
        cal_data['freqs'], cal_data['ant_pos'],
        dataset_axis=axis, category=category,
        tissue_model_override=tissue_model,
    )
    localizer.calibrate(cal_data['phases'], CALIBRATION_TARGET_POS)
    
    errors = []
    for tp in target_positions:
        data = get_phase_matrix(agg_df, tp, freqs)
        if data is None or len(data['freqs']) < 2:
            continue
        try:
            est_tp, cost = localizer.localize(data['phases'], data['powers'])
            error = abs(est_tp - tp)
            errors.append(error)
        except Exception:
            continue
    
    if not errors:
        return None, None, 0
    
    return np.mean(errors), np.median(errors), len(errors)


def main():
    parser = argparse.ArgumentParser(
        description='Grid search over tissue thickness parameters')
    parser.add_argument('--csv', required=True)
    parser.add_argument('--axis', default='y', choices=['x', 'y'])
    parser.add_argument('--output', default='grid_search_results')
    parser.add_argument('--chicken-range', nargs=2, type=float,
                        default=[5.0, 25.0],
                        metavar=('MIN', 'MAX'),
                        help='Muscle thickness range for Chicken (mm)')
    parser.add_argument('--chicken-step', type=float, default=5.0,
                        help='Step size for Chicken muscle thickness')
    parser.add_argument('--pork-fat-range', nargs=2, type=float,
                        default=[5.0, 20.0],
                        metavar=('MIN', 'MAX'),
                        help='Fat thickness range for Pork (mm)')
    parser.add_argument('--pork-fat-step', type=float, default=2.5,
                        help='Step size for Pork fat thickness')
    parser.add_argument('--pork-muscle-range', nargs=2, type=float,
                        default=[10.0, 35.0],
                        metavar=('MIN', 'MAX'),
                        help='Muscle thickness range for Pork (mm)')
    parser.add_argument('--pork-muscle-step', type=float, default=5.0,
                        help='Step size for Pork muscle thickness')
    parser.add_argument('--tp-range', nargs=2, type=float, metavar=('MIN','MAX'),
                        help='Restrict evaluation to targets between MIN and MAX')
    parser.add_argument('--freqs', nargs='+', default=['all'],
                        choices=['pair_a', 'all'])
    args = parser.parse_args()
    
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Parse frequency set
    if args.freqs[0] == 'pair_a':
        freqs = PAIR_A
    else:
        freqs = ALL_HARMONICS
    
    tp_min, tp_max = (None, None)
    if args.tp_range:
        tp_min, tp_max = args.tp_range
    
    print(f"Loading dataset from {args.csv} ...")
    df = load_dataset(args.csv)
    print(f"  {len(df):,} rows")
    
    # Generate parameter grid
    chicken_values = np.arange(args.chicken_range[0],
                               args.chicken_range[1] + args.chicken_step / 2,
                               args.chicken_step)
    pork_fat_values = np.arange(args.pork_fat_range[0],
                                args.pork_fat_range[1] + args.pork_fat_step / 2,
                                args.pork_fat_step)
    pork_muscle_values = np.arange(args.pork_muscle_range[0],
                                    args.pork_muscle_range[1] + args.pork_muscle_step / 2,
                                    args.pork_muscle_step)
    
    print(f"\nChicken muscle: {chicken_values}")
    print(f"Pork fat: {pork_fat_values}")
    print(f"Pork muscle: {pork_muscle_values}")
    print(f"Total combinations: {len(chicken_values) * len(pork_fat_values) * len(pork_muscle_values)}")
    
    results = []
    
    dim = 'lateral' if args.axis == 'x' else 'depth'
    print(f"\nSearching for best parameters ({dim} axis)...\n")
    
    # Loop over all parameter combinations
    for chk_m, pork_f, pork_m in product(chicken_values, pork_fat_values, pork_muscle_values):
        params = {
            'chicken_muscle': chk_m,
            'pork_fat': pork_f,
            'pork_muscle': pork_m,
        }
        
        row = {
            'chicken_muscle': chk_m,
            'pork_fat': pork_f,
            'pork_muscle': pork_m,
        }
        
        # Evaluate Chicken
        agg_chk = get_aggregated_data(df, 'Chicken', args.axis)
        if not agg_chk.empty:
            mean_err, med_err, n = evaluate_config(agg_chk, args.axis, freqs,
                                                    'Chicken', params, tp_min, tp_max)
            if n > 0:
                row['chicken_mean'] = mean_err
                row['chicken_median'] = med_err
                row['chicken_n'] = n
        
        # Evaluate Pork
        agg_pork = get_aggregated_data(df, 'Pork', args.axis)
        if not agg_pork.empty:
            mean_err, med_err, n = evaluate_config(agg_pork, args.axis, freqs,
                                                    'Pork', params, tp_min, tp_max)
            if n > 0:
                row['pork_mean'] = mean_err
                row['pork_median'] = med_err
                row['pork_n'] = n
        
        # Combined metric
        if 'chicken_mean' in row and 'pork_mean' in row:
            row['combined_mean'] = (row['chicken_mean'] + row['pork_mean']) / 2
            row['combined_median'] = (row['chicken_median'] + row['pork_median']) / 2
        
        results.append(row)
        
        # Progress indicator every 10 iterations
        if len(results) % 10 == 0:
            print(f"  Evaluated {len(results)} configurations...", end='\r')
    
    print(f"  Evaluated {len(results)} configurations.        ")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_csv = outdir / 'tissue_grid_search.csv'
    results_df.to_csv(results_csv, index=False)
    print(f"\nResults saved to {results_csv}")
    
    # Find best configurations
    print(f"\n{'=' * 80}")
    print("BEST CONFIGURATIONS (by combined mean error)")
    print(f"{'=' * 80}")
    
    best_by_combined = results_df.nsmallest(5, 'combined_mean')
    for idx, row in best_by_combined.iterrows():
        print(f"\n  Chicken muscle: {row['chicken_muscle']:.1f} mm")
        print(f"  Pork fat:       {row['pork_fat']:.1f} mm")
        print(f"  Pork muscle:    {row['pork_muscle']:.1f} mm")
        print(f"  Chicken: mean={row['chicken_mean']:.1f}mm, median={row['chicken_median']:.1f}mm (n={int(row['chicken_n'])})")
        print(f"  Pork:    mean={row['pork_mean']:.1f}mm, median={row['pork_median']:.1f}mm (n={int(row['pork_n'])})")
        print(f"  Combined mean: {row['combined_mean']:.1f}mm")
    
    # Plot heatmaps of mean errors
    print(f"\nGenerating heatmaps...")
    
    # For each combination of pork parameters, plot chicken muscle vs mean error
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Chicken mean error heatmap
    chicken_pivot = results_df.pivot_table(
        values='chicken_mean',
        index=['pork_fat', 'pork_muscle'],
        columns='chicken_muscle',
        aggfunc='first'
    )
    if not chicken_pivot.empty:
        ax = axes[0]
        im = ax.imshow(chicken_pivot, cmap='RdYlGn_r', aspect='auto', origin='lower')
        ax.set_xlabel('Chicken muscle thickness (mm)')
        ax.set_ylabel('(Pork fat, Pork muscle) (mm)')
        ax.set_xticklabels([f'{v:.0f}' for v in chicken_pivot.columns], rotation=45)
        ax.set_yticklabels([f'{i}' for i in range(len(chicken_pivot))], rotation=0)
        ax.set_title('Chicken Mean Error (mm)')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Error (mm)')
    
    # Pork mean error heatmap
    pork_pivot = results_df.pivot_table(
        values='pork_mean',
        index=['chicken_muscle'],
        columns=['pork_fat', 'pork_muscle'],
        aggfunc='first'
    )
    if not pork_pivot.empty:
        ax = axes[1]
        im = ax.imshow(pork_pivot, cmap='RdYlGn_r', aspect='auto', origin='lower')
        ax.set_xlabel('(Pork fat, Pork muscle) (mm)')
        ax.set_ylabel('Chicken muscle thickness (mm)')
        ax.set_yticklabels([f'{v:.0f}' for v in pork_pivot.index], rotation=0)
        ax.set_xticklabels([f'{i}' for i in range(len(pork_pivot.columns))], rotation=45)
        ax.set_title('Pork Mean Error (mm)')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Error (mm)')
    
    plt.tight_layout()
    plt.savefig(outdir / 'tissue_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Plot convergence of combined error
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(range(len(results_df)), results_df['combined_mean'], s=20, alpha=0.6)
    ax.axhline(results_df['combined_mean'].min(), color='r', ls='--', 
               label=f'Best: {results_df["combined_mean"].min():.1f}mm')
    ax.set_xlabel('Configuration index')
    ax.set_ylabel('Combined mean error (mm)')
    ax.set_title('Tissue parameter grid search: combined error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / 'tissue_search_convergence.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nPlots saved to {outdir}/")


if __name__ == '__main__':
    main()
