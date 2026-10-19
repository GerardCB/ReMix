"""
report_comparisons.py — Generate comparison reports:
  1. Median-only comparison table: harmonic_comb vs calibrated, for X and Y
  2. Boxplots with quartiles + outliers for all conditions

Uses existing results.csv data.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.stats as stats


OUTDIR = Path(__file__).resolve().parent.parent / 'results'
OUTDIR.mkdir(exist_ok=True)


def load_results():
    df = pd.read_csv(OUTDIR / 'results.csv')
    # Parse config into components
    parts = df['config'].str.extract(
        r'(?P<category>\w+)/(?P<axis>[xy])\s+(?P<freqs>\S+)\s+(?P<approach>\S+)')
    df = pd.concat([df, parts], axis=1)
    return df


def make_median_tables(df):
    """Create median comparison tables for X and Y axes."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, axis in enumerate(['x', 'y']):
        ax = axes[idx]
        sub = df[df['axis'] == axis]
        
        # Pivot: rows = category, columns = approach, values = median error
        table_data = []
        categories = ['Patch', 'Chicken', 'Pork']
        approaches = ['harmonic_comb', 'calibrated']
        
        for cat in categories:
            row = {'Category': cat}
            for app in approaches:
                mask = (sub['category'] == cat) & (sub['approach'] == app)
                errors = sub.loc[mask, 'error']
                if len(errors) > 0:
                    row[app] = errors.median()
                else:
                    row[app] = np.nan
            table_data.append(row)
        
        table_df = pd.DataFrame(table_data)
        
        # Plot as a styled table
        ax.axis('off')
        dim = 'Lateral (X)' if axis == 'x' else 'Depth (Y)'
        ax.set_title(f'Median Localization Error — {dim} Axis',
                     fontsize=20, fontweight='bold', pad=20)
        
        cell_text = []
        for _, row in table_df.iterrows():
            cell_text.append([
                row['Category'],
                f"{row['harmonic_comb']:.1f} mm" if not np.isnan(row['harmonic_comb']) else '—',
                f"{row['calibrated']:.1f} mm" if not np.isnan(row['calibrated']) else '—',
            ])
        
        # Color cells by error magnitude
        colors = []
        for _, row in table_df.iterrows():
            row_colors = ['#f0f0f0']  # category column
            for app in approaches:
                val = row[app]
                if np.isnan(val):
                    row_colors.append('#f0f0f0')
                elif val < 20:
                    row_colors.append('#c8e6c9')  # green
                elif val < 50:
                    row_colors.append('#fff9c4')  # yellow
                elif val < 100:
                    row_colors.append('#ffccbc')  # orange
                else:
                    row_colors.append('#ef9a9a')  # red
                    
            colors.append(row_colors)
        
        table = ax.table(
            cellText=cell_text,
            colLabels=['Category', 'Harmonic Comb\n(pair_a)', 'Calibrated\n(all freqs)'],
            cellColours=colors,
            loc='center',
            cellLoc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(18)
        table.scale(1.0, 2.0)
        
        # Bold header
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(fontweight='bold')
                cell.set_facecolor('#bbdefb')
    
    plt.tight_layout()
    fig.savefig(OUTDIR / 'median_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved {OUTDIR / 'median_comparison.png'}")


def make_boxplots(df):
    """Create boxplots showing quartiles + outliers for all conditions."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    for idx, axis in enumerate(['x', 'y']):
        ax = axes[idx]
        sub = df[df['axis'] == axis]
        
        categories = ['Patch', 'Chicken', 'Pork']
        approaches = ['harmonic_comb', 'calibrated']
        approach_labels = ['Harmonic\nComb', 'Calibrated']
        colors = ['#5c6bc0', '#ef5350']  # indigo, red
        
        # Collect data for grouped boxplot
        positions = []
        data_list = []
        box_colors = []
        
        group_width = 2.5
        box_width = 0.8
        
        for gi, cat in enumerate(categories):
            center = gi * group_width
            for ai, (app, label) in enumerate(zip(approaches, approach_labels)):
                mask = (sub['category'] == cat) & (sub['approach'] == app)
                errors = sub.loc[mask, 'error'].values
                
                pos = center + (ai - 0.5) * 1.0
                positions.append(pos)
                box_colors.append(colors[ai])
                
                if len(errors) > 0:
                    data_list.append(errors)
                else:
                    data_list.append([])
        
        dim = 'Lateral (X)' if axis == 'x' else 'Depth (Y)'
        ax.set_title(f'{dim} Axis — Error Distribution', fontsize=18, fontweight='bold')
        
        # Modified condition logic: Only assign the label string if it is the left plot (idx == 0)
        ax.set_ylabel('Localization Error (mm)' if idx == 0 else '', fontsize=18)
        
        # Set tick parameters globally for this axis
        ax.tick_params(axis='both', which='major', labelsize=15)
        
        if any(len(d) > 0 for d in data_list):
            bp = ax.boxplot(data_list, positions=positions, widths=box_width,
                           patch_artist=True, showfliers=True,
                           flierprops=dict(marker='o', markersize=4, alpha=0.5),
                           medianprops=dict(color='black', linewidth=2),
                           whiskerprops=dict(linewidth=1.2),
                           capprops=dict(linewidth=1.2))
            
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        else:
            ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', 
                    transform=ax.transAxes, fontsize=18, color='gray', style='italic')
        
        cat_positions = [gi * group_width for gi in range(len(categories))]
        ax.set_xticks(cat_positions)
        ax.set_xticklabels(categories, fontsize=18) 
        
        ax.grid(True, axis='y', alpha=0.2)
        ax.set_ylim(bottom=0)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=colors[0], alpha=0.7, label='Harmonic Comb'),
            Patch(facecolor=colors[1], alpha=0.7, label='Calibrated (all)'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=16)
    
    plt.tight_layout()
    fig.savefig(OUTDIR / 'boxplot_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved {OUTDIR / 'boxplot_comparison.png'}")


def make_detailed_boxplots(df):
    """One boxplot per axis, with individual conditions broken out."""
    
    for axis in ['x', 'y']:
        sub = df[df['axis'] == axis]
        
        categories = ['Patch', 'Chicken', 'Pork']
        approaches = ['harmonic_comb', 'calibrated']
        colors = {'harmonic_comb': '#5c6bc0', 'calibrated': '#ef5350'}
        
        fig, axes_arr = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        dim = 'Lateral (X)' if axis == 'x' else 'Depth (Y)'
        fig.suptitle(f'{dim} Axis — Per-Category Error Boxplots', 
                     fontsize=22, fontweight='bold', y=1.02)
        
        for ci, cat in enumerate(categories):
            ax = axes_arr[ci]
            data_list = []
            labels = []
            box_colors_list = []
            
            ax.tick_params(axis='both', which='major', labelsize=15)
            
            for app in approaches:
                mask = (sub['category'] == cat) & (sub['approach'] == app)
                errors = sub.loc[mask, 'error'].values
                
                data_list.append(errors if len(errors) > 0 else [])
                
                nice_name = 'Harmonic Comb' if app == 'harmonic_comb' else 'Calibrated'
                labels.append(nice_name)
                box_colors_list.append(colors[app])
            
            if any(len(d) > 0 for d in data_list):
                bp = ax.boxplot(data_list, labels=labels,
                               patch_artist=True, showfliers=True,
                               notch=True,
                               flierprops=dict(marker='D', markersize=5, alpha=0.6,
                                             markerfacecolor='#333'),
                               medianprops=dict(color='black', linewidth=2.5),
                               whiskerprops=dict(linewidth=1.5),
                               capprops=dict(linewidth=1.5))
                
                for patch, color in zip(bp['boxes'], box_colors_list):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                for i, d in enumerate(data_list):
                    if len(d) > 0:
                        med = np.median(d)
                        ax.annotate(f'{med:.1f}mm', xy=(i + 1, med),
                                   xytext=(15, 5), textcoords='offset points',
                                   fontsize=15, fontweight='bold',
                                   arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
            else:
                ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', 
                        transform=ax.transAxes, fontsize=15, color='gray', style='italic')
                ax.set_xticks([1, 2])
                ax.set_xticklabels(labels, fontsize=15)
            
            ax.set_title(cat, fontsize=20, fontweight='bold')
            ax.set_ylabel('Error (mm)' if ci == 0 else '', fontsize=18)
            
            ax.grid(True, axis='y', alpha=0.2)
            ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        fig.savefig(OUTDIR / f'boxplot_detailed_{axis}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved {OUTDIR / f'boxplot_detailed_{axis}.png'}")


def print_median_table(df):
    """Print text-based median comparison tables."""
    for axis in ['x', 'y']:
        sub = df[df['axis'] == axis]
        dim = 'LATERAL (X)' if axis == 'x' else 'Depth (Y)'
        
        print(f"\n{'═' * 60}")
        print(f"  MEDIAN ERROR COMPARISON — {dim} AXIS")
        print(f"{'═' * 60}")
        print(f"  {'Category':<12}  {'Harmonic Comb':>15}  {'Calibrated':>15}")
        print(f"  {'─' * 12}  {'─' * 15}  {'─' * 15}")
        
        for cat in ['Patch', 'Chicken', 'Pork']:
            hc_mask = (sub['category'] == cat) & (sub['approach'] == 'harmonic_comb')
            cal_mask = (sub['category'] == cat) & (sub['approach'] == 'calibrated')
            
            hc_med = sub.loc[hc_mask, 'error'].median() if hc_mask.any() else float('nan')
            cal_med = sub.loc[cal_mask, 'error'].median() if cal_mask.any() else float('nan')
            
            hc_str = f"{hc_med:.1f} mm" if not np.isnan(hc_med) else "—"
            cal_str = f"{cal_med:.1f} mm" if not np.isnan(cal_med) else "—"
            
            print(f"  {cat:<12}  {hc_str:>15}  {cal_str:>15}")
        
        print(f"{'═' * 60}")




def run_statistical_analysis(df):
    """Outputs Mean, Std, Variance, and Wilcoxon p-values for all conditions."""
    print(f"\n{'═' * 80}")
    print(f"  EXTENSIVE STATISTICAL ANALYSIS (Mean, Std, Variance, p-values)")
    print(f"{'═' * 80}")
    
    stats_data = []
    for axis in ['x', 'y']:
        sub = df[df['axis'] == axis]
        print(f"\n  --- {axis.upper()} AXIS ---")
        
        for cat in ['Patch', 'Chicken', 'Pork']:
            hc_mask = (sub['category'] == cat) & (sub['approach'] == 'harmonic_comb')
            cal_mask = (sub['category'] == cat) & (sub['approach'] == 'calibrated')
            
            hc_df = sub[hc_mask][['target_pos', 'error']].rename(columns={'error': 'hc_err'})
            cal_df = sub[cal_mask][['target_pos', 'error']].rename(columns={'error': 'cal_err'})
            merged = pd.merge(hc_df, cal_df, on='target_pos', how='inner')
            
            p_val = float('nan')
            if len(merged) > 5:
                try:
                    _, p_val = stats.wilcoxon(merged['hc_err'], merged['cal_err'])
                except ValueError:
                    pass
            
            hc_errs, cal_errs = merged['hc_err'].values, merged['cal_err'].values
            print(f"  {cat}:")
            if len(hc_errs) > 0:
                print(f"    HC   -> Median: {np.median(hc_errs):6.1f} | Mean: {np.mean(hc_errs):6.1f} | Std: {np.std(hc_errs):6.1f} | Var: {np.var(hc_errs):7.1f}")
            if len(cal_errs) > 0:
                print(f"    Cal  -> Median: {np.median(cal_errs):6.1f} | Mean: {np.mean(cal_errs):6.1f} | Std: {np.std(cal_errs):6.1f} | Var: {np.var(cal_errs):7.1f}")
            print(f"    Wilcoxon p-value (HC vs Cal): {p_val:.4e}\n")
            
            # Save to list for CSV export...


if __name__ == '__main__':
    print("Loading results...")
    df = load_results()
    print(f"  {len(df)} rows loaded\n")
    
    print("1. Median comparison tables...")
    print_median_table(df)
    make_median_tables(df)
    
    print("\n2. Boxplots...")
    make_boxplots(df)
    make_detailed_boxplots(df)

    run_statistical_analysis(df)
    
    print("\nDone!")