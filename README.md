# ReMix — RF Localization Through Tissue

RF-based localization of a diode/patch antenna using backscatter measurements, evaluated under free-space and tissue-covered conditions (chicken and pork).

## Quick Start

```bash
# 1. Set up environment
conda create -n remix python=3.10
conda activate remix
pip install numpy pandas matplotlib

# 2. Place the dataset in the repo root
#    (Remix_Dataset.csv — not tracked by Git due to size)

# 3. Run the main pipeline
python remix_pipeline.py

# 4. Run specific experiments
python report_comparisons.py        # Median tables + boxplots
python report_freq_pairs.py         # Frequency pair optimization
python investigate_y_axis.py        # Y-axis failure analysis
python investigate_signal_quality.py # Signal quality investigation
python investigate_tissue_model.py   # Tissue model effectiveness
python investigate_improvements.py   # Algorithmic improvements
```

## Repository Structure

```
ReMix/
├── REPORT.md                      # Full experiment report with figures
├── README.md                      # This file
│
├── remix_config.py                # Physical setup, coordinates, frequencies
├── remix_data.py                  # Data loading & coherent IQ averaging
├── remix_localize.py              # Localization algorithms (HC, Calibrated, etc.)
├── remix_pipeline.py              # Main evaluation pipeline
│
├── report_comparisons.py          # Exp 1: Median tables + boxplots
├── report_freq_pairs.py           # Exp 2: Frequency pair optimization
├── investigate_y_axis.py          # Exp 3: Y-axis mirror ambiguity
├── investigate_signal_quality.py  # Exp 4: Signal quality analysis
├── investigate_tissue_model.py    # Exp 5: Tissue model effectiveness
├── investigate_improvements.py    # Exp 6: Per-antenna voting + ensemble
├── tissue_grid_search.py          # Grid search over tissue parameters
│
├── results/                       # Generated plots & CSVs
│   ├── signal_quality/            #   Power, phase stability, attenuation
│   ├── tissue_investigation/      #   Tissue model analysis
│   ├── improvements/              #   Algorithmic improvement results
│   └── y_axis_investigation/      #   Cost landscape & phase wrapping
│
├── grid_search_results/           # Tissue grid search outputs
├── Chicken.png                    # Setup photos
├── Patch.png
└── Pork.png
```

## Dataset

The dataset (`Remix_Dataset.csv`, ~608 MB, **not included in this repo**) contains 7.78M rows of backscatter measurements:

| Column | Description |
|--------|-------------|
| `Category` | Tissue condition: Patch, Chicken, Pork |
| `Axis` | Scan axis: x (lateral) or y (depth) |
| `Frequency` | Harmonic frequency in MHz |
| `Target_Pos` | Target position (mm) |
| `Antenna_Pos` | Rx antenna position (mm) |
| `I`, `Q` | In-phase / quadrature components |
| `dBm` | Received power |
| `Phase` | Received phase (degrees) |

## Key Results

| Condition | Best X-axis | Best Y-axis | Bottleneck |
|-----------|:-----------:|:-----------:|------------|
| **Patch** | **21.6 mm** | **8.3 mm** | Algorithm (mirror ambiguity) |
| **Chicken** | **20.2 mm** | 76.8 mm | Signal (partial attenuation) |
| **Pork** | 89.9 mm | 67.0 mm | Signal (at noise floor) |

See [REPORT.md](REPORT.md) for the full analysis.

## Algorithms

1. **Harmonic Combination (HC)** — From the original ReMix paper. Phase difference between two harmonics creates a virtual low-frequency signal.
2. **Calibrated Phase Matching** — All 7 frequencies, hierarchical coarse-to-fine search with complex-domain residual.
3. **Per-Antenna Voting** — Improved calibrated approach that penalizes per-antenna variance to reject mirror ghosts.
4. **Ensemble (Top-5 Pairs)** — Median of HC estimates from the 5 best frequency pairs.
