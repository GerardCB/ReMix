# ReMix — RF Localization Through Tissue

RF-based localization of a diode/patch antenna using backscatter measurements, evaluated under free-space and tissue-covered conditions (chicken and pork).

## Setup

![Experimental setup](figures/RemixSetup.jpeg)

## Repository Structure

```
ReMix/
├── REPORT.md                          # Full experiment report with figures
├── README.md                          # This file
│
├── src/                               # Core library
│   ├── __init__.py
│   ├── config.py                      # Physical setup, coordinates, frequencies
│   ├── data.py                        # Data loading & coherent IQ averaging
│   └── localize.py                    # Localization algorithms (HC, Calibrated, etc.)
│
├── experiments/                       # Runnable experiment scripts
│   ├── run_pipeline.py                # Main evaluation pipeline
│   ├── report_comparisons.py          # Exp 1: Median tables + boxplots
│   ├── report_freq_pairs.py           # Exp 2: Frequency pair optimization
│   ├── investigate_y_axis.py          # Exp 3: Y-axis mirror ambiguity
│   ├── investigate_signal_quality.py  # Exp 4: Signal quality analysis
│   ├── investigate_tissue_model.py    # Exp 5: Tissue model effectiveness
│   ├── investigate_improvements.py    # Exp 6: Per-antenna voting + ensemble
│   └── tissue_grid_search.py          # Grid search over tissue parameters
│
├── data/                              # Dataset (not tracked by Git)
│   └── README.md                      # Instructions for obtaining the data
│
├── results/                           # Generated plots & CSVs
│   ├── signal_quality/
│   ├── tissue_investigation/
│   ├── improvements/
│   ├── y_axis_investigation/
│   └── grid_search/
│
└── figures/                           # Setup photos
    ├── RemixSetup.jpeg
    ├── Chicken.png
    ├── Patch.png
    └── Pork.png
```

## Quick Start

```bash
# 1. Set up environment
conda create -n remix python=3.10
conda activate remix
pip install numpy pandas matplotlib scipy

# 2. Place the dataset
#    Copy Remix_Dataset.csv into the data/ directory

# 3. Run the main pipeline
python experiments/run_pipeline.py --csv data/Remix_Dataset.csv --axis y

# 4. Run specific experiments
python experiments/report_comparisons.py
python experiments/report_freq_pairs.py
python experiments/investigate_y_axis.py
python experiments/investigate_signal_quality.py
python experiments/investigate_tissue_model.py
python experiments/investigate_improvements.py
```

All experiment scripts can be run from the repository root. Results are saved to `results/`.

## Key Results

See [REPORT.md](REPORT.md) for the full analysis.

| Condition | Best X-axis | Best Y-axis |
|-----------|:-----------:|:-----------:|
| **Patch** (free-space) | 21.6 mm | 8.3 mm |
| **Chicken** (~5 mm tissue) | 20.2 mm | 76.8 mm |
| **Pork** (~10 mm tissue) | 89.9 mm | 67.0 mm |

## Algorithms

1. **Harmonic Combination (HC)** — From the original ReMix paper. Phase difference between two harmonics creates a virtual low-frequency signal.
2. **Calibrated Phase Matching** — All 7 frequencies, hierarchical coarse-to-fine search with complex-domain residual.
3. **Per-Antenna Voting** — Improved calibrated approach that penalizes per-antenna variance to reject mirror ghosts.
4. **Ensemble (Top-5 Pairs)** — Median of HC estimates from the 5 best frequency pairs.
