"""
remix_data.py — Data loading and preprocessing for the ReMix localization pipeline.

The CSV dataset has columns:
    Category, Axis, Frequency, Target_Pos, Antenna_Pos, I, Q, dBm, Phase

Each unique (Category, Axis, Frequency, Target_Pos, Antenna_Pos) combination
may contain up to 1001 temporal IQ samples.  This module aggregates them by
coherent averaging (mean of complex IQ), then recomputes phase and power.
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
#  Loading
# ─────────────────────────────────────────────────────────────────────────────
def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load the backscatter dataset CSV and coerce column types."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    df['Frequency']   = df['Frequency'].astype(int)
    df['Target_Pos']  = pd.to_numeric(df['Target_Pos'],   errors='coerce')
    df['Antenna_Pos'] = pd.to_numeric(df['Antenna_Pos'],  errors='coerce')
    df['I']           = pd.to_numeric(df['I'],            errors='coerce')
    df['Q']           = pd.to_numeric(df['Q'],            errors='coerce')
    df['dBm']         = pd.to_numeric(df['dBm'],          errors='coerce')
    df['Phase']       = pd.to_numeric(df['Phase'],        errors='coerce')
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Aggregation (coherent IQ averaging over temporal samples)
# ─────────────────────────────────────────────────────────────────────────────
def _agg_iq(group: pd.DataFrame) -> pd.Series:
    """Aggregate a group of temporal IQ samples into one measurement."""
    mean_i = group['I'].mean()
    mean_q = group['Q'].mean()
    z = complex(mean_i, mean_q)

    mag = np.abs(z)
    phase_deg = np.degrees(np.angle(z)) % 360.0
    power_dbm = 10.0 * np.log10(mag ** 2) if mag > 0 else -200.0

    return pd.Series({
        'I_mean':    mean_i,
        'Q_mean':    mean_q,
        'Phase_deg': phase_deg,
        'dBm':       power_dbm,
        'n_samples': len(group),
    })


def get_aggregated_data(df: pd.DataFrame,
                        category: str,
                        axis: str) -> pd.DataFrame:
    """
    Filter by category/axis, then aggregate temporal IQ samples.

    Returns a DataFrame indexed by (Frequency, Target_Pos, Antenna_Pos) with
    columns: I_mean, Q_mean, Phase_deg, dBm, n_samples.
    """
    mask = (df['Category'] == category) & (df['Axis'] == axis)
    subset = df.loc[mask]
    if subset.empty:
        return pd.DataFrame()

    grouped = subset.groupby(['Frequency', 'Target_Pos', 'Antenna_Pos'],
                             sort=True)
    result = grouped.apply(_agg_iq).reset_index()
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Extraction helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_phase_matrix(agg_df: pd.DataFrame,
                     target_pos: float,
                     frequencies: list) -> dict:
    """
    Build a phase matrix for one target position across the requested
    frequencies.

    Parameters
    ----------
    agg_df : DataFrame returned by get_aggregated_data()
    target_pos : target displacement in mm
    frequencies : list of int — MHz values (e.g. [790, 1700])

    Returns
    -------
    dict with keys:
        'phases'      — np.ndarray (n_ant, n_freq) in degrees [0, 360)
        'powers'      — np.ndarray (n_ant, n_freq) in dBm
        'ant_pos'     — np.ndarray (n_ant,) antenna positions in mm
        'freqs'       — list of int MHz that actually had data
    or None if no data available.
    """
    phases_cols, power_cols = [], []
    ant_pos = None
    valid_freqs = []

    for freq_mhz in frequencies:
        mask = ((agg_df['Target_Pos'] == target_pos) &
                (agg_df['Frequency'] == freq_mhz))
        sub = agg_df.loc[mask].sort_values('Antenna_Pos')
        if sub.empty:
            continue

        if ant_pos is None:
            ant_pos = sub['Antenna_Pos'].values.copy()

        phases_cols.append(sub['Phase_deg'].values)
        power_cols.append(sub['dBm'].values)
        valid_freqs.append(freq_mhz)

    if not phases_cols:
        return None

    return {
        'phases':  np.column_stack(phases_cols),   # (n_ant, n_freq)
        'powers':  np.column_stack(power_cols),    # (n_ant, n_freq)
        'ant_pos': ant_pos,                        # (n_ant,)
        'freqs':   valid_freqs,
    }


def get_available_target_positions(agg_df: pd.DataFrame) -> np.ndarray:
    """Return sorted unique target positions, excluding baseline (−999)."""
    tp = agg_df['Target_Pos'].unique()
    tp = tp[tp > -900]
    return np.sort(tp)


def get_baseline_phases(df: pd.DataFrame, frequency: int) -> dict:
    """
    Get the baseline (static) measurement for a given frequency.
    Returns dict with 'phase_deg' and 'dbm' (scalars, averaged over samples).
    """
    mask = (df['Category'] == 'Baseline') & (df['Frequency'] == frequency)
    sub = df.loc[mask]
    if sub.empty:
        return None

    mean_i = sub['I'].mean()
    mean_q = sub['Q'].mean()
    z = complex(mean_i, mean_q)
    return {
        'phase_deg': np.degrees(np.angle(z)) % 360.0,
        'dbm': 10.0 * np.log10(np.abs(z)**2) if np.abs(z) > 0 else -200.0,
    }