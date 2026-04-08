"""
remix_config.py — Physical setup configuration.

Coordinate system (2D, same elevation):
    X-axis: lateral (perpendicular to beam) — RX and target move here
    Y-axis: depth (along beam, from antenna plane toward target)

Dataset axis mapping:
    Axis 'x' in CSV → LATERAL (target moves in X, same direction as RX)
        Target at (X = Target_Pos, Y = 850mm)
        Depth FIXED at 850mm

    Axis 'y' in CSV → DEPTH (target moves toward/away from RX)
        Target at (X = 0, Y = 820 + Target_Pos)
        At TP = -180: depth = 640mm (closest)
        At TP =    0: depth = 820mm (center)
        At TP = +180: depth = 1000mm (farthest)

    TX: fixed at (X = 0, Y = 0) — centered laterally, beneath table,
        aligned in Y with RX rail
    RX: at (X = Antenna_Pos, Y = 0) — moves ±150mm laterally

All spatial units in mm.
"""

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  FREQUENCIES
# ══════════════════════════════════════════════════════════════════════════════
F1 = 830e6   # Hz
F2 = 870e6   # Hz
C  = 3e8     # m/s

# Map from MHz label → Hz
FREQ_TO_HZ = {
    790:  790e6,
    910:  910e6,
    1620: 1620e6,
    1660: 1660e6,
    1700: 1700e6,
    1740: 1740e6,
    1780: 1780e6,
}

# Harmonic decomposition: freq_mhz → (n, m) where freq = n*f1 + m*f2
HARMONIC_COEFFS = {
    790:  ( 2, -1),   # 2f1 - f2   
    910:  (-1,  2),   # 2f2 - f1   
    1620: ( 3, -1),   # 3f1 - f2   
    1660: ( 2,  0),   # 2f1        
    1700: ( 1,  1),   # f1 + f2    
    1740: ( 0,  2),   # 2f2        
    1780: (-1,  3),   # 3f2 - f1   
}

# Useful frequency groupings
PAIR_A = [790, 1700]
PAIR_B = [910, 1700]
ALL_HARMONICS = [790, 910, 1620, 1660, 1700, 1740, 1780]

# ══════════════════════════════════════════════════════════════════════════════
#  PHYSICAL SETUP (mm)
# ══════════════════════════════════════════════════════════════════════════════
# TX horn antenna — fixed, centered laterally, at antenna plane
TX_POS = np.array([0.0, 0.0])   # (X, Y) = (lateral, depth)

# RX horn antenna — moves laterally (X), at Y=0
RX_X_VALUES = np.array([-150.0, -75.0, 0.0, 75.0, 150.0])  # Antenna_Pos in X

# Depth when target moves laterally ('x' dataset)
LATERAL_FIXED_DEPTH = 850.0   # mm (85 cm from RX)

# Center depth when target moves in depth ('y' dataset)
DEPTH_CENTER = 820.0   # mm (82 cm at TP=0; min 64 cm at TP=-180)

# ══════════════════════════════════════════════════════════════════════════════
#  TISSUE MODEL — Relative permittivity and layer profiles
# ══════════════════════════════════════════════════════════════════════════════
# Relative permittivity ε_r
EPS_R = {
    'air':    1.0,
    'muscle': 54.81,
    'fat':    5.447,
}

# Phase scaling factor α = √ε_r  (since ε_r is given as real)
ALPHA = {k: np.sqrt(v) for k, v in EPS_R.items()}
# ALPHA['muscle'] ≈ 7.403,  ALPHA['fat'] ≈ 2.334

# Tissue profiles: list of (material_name, thickness_mm) layers.
TISSUE_PROFILES = {
    'Patch':   [],                                        # free space
    'Chicken': [('muscle', 15.0)],                        # 1.5 cm muscle
    'Pork':    [('fat', 10.0), ('muscle', 20.0)],         # 1 cm fat + 2 cm muscle
}


def get_tissue_layers(category):
    """Return list of (alpha, thickness_mm) for a given category."""
    profile = TISSUE_PROFILES.get(category, [])
    return [(ALPHA[mat], thick) for mat, thick in profile]


def get_tissue_wall(category):
    """Total one-way tissue wall thickness in mm."""
    profile = TISSUE_PROFILES.get(category, [])
    return sum(t for _, t in profile)


# ══════════════════════════════════════════════════════════════════════════════
#  DERIVED POSITIONS
# ══════════════════════════════════════════════════════════════════════════════
def rx_positions():
    """(n_ant, 2) array of [X, Y] for each RX antenna."""
    n = len(RX_X_VALUES)
    pos = np.zeros((n, 2))
    pos[:, 0] = RX_X_VALUES   # X (lateral)
    pos[:, 1] = 0.0           # Y (at antenna plane)
    return pos


def target_position(target_pos_mm, dataset_axis):
    """
    Map a dataset Target_Pos value to physical (X, Y) coordinates.

    dataset_axis='x' → LATERAL: target at (X=TP, Y=850mm)
    dataset_axis='y' → DEPTH:   target at (X=0,  Y=820+TP)
    """
    if dataset_axis == 'x':
        return np.array([target_pos_mm, LATERAL_FIXED_DEPTH])
    else:
        return np.array([0.0, DEPTH_CENTER + target_pos_mm])


# ══════════════════════════════════════════════════════════════════════════════
#  SEARCH BOUNDS (mm) — in Target_Pos space
# ══════════════════════════════════════════════════════════════════════════════
DEPTH_SEARCH   = (-200.0, 200.0)
LATERAL_SEARCH = (-200.0, 200.0)

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
CALIBRATION_TARGET_POS = 0.0
CATEGORIES = ['Patch', 'Chicken', 'Pork']
REFERENCE_DEPTH = DEPTH_CENTER