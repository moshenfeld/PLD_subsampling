### PLD_subsampling

Implements and evaluates privacy amplification by subsampling for Privacy Loss Distribution (PLD) probability mass functions (PMFs). Generates CDF plots and epsilon ratio plots comparing analytical ground truth, dp-accounting, and our direct subsampling implementation.

### Package layout

- `PLD_subsampling/`
  - `PLD_subsampling.py`: Core subsampling primitives
    - `stable_subsampling_loss`: numerically stable loss mapping
    - `exclusive_ccdf_from_pdf`: CCDF helper (exclusive tail)
    - `subsample_losses`: transforms a PMF on a uniform loss grid
  - `wrappers/dp_accounting_wrappers.py`: Thin wrappers around dp-accounting (construct PLDs, amplify PLDs separately for remove/add), plus PMF bridge utilities
  - `testing/`
    - `analytic_Gaussian.py`: Analytical PLD and epsilon(δ) formulas for Gaussian mechanism
    - `test_utils.py`: Builds method variants for experiments, parameterized `run_all_experiments`
    - `plot_utils.py`: Plotting (CDF with focused x-range, epsilon ratio)
  - `main.py`: Runs experiments and saves figures to `plots/`

### Quickstart

1) Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2) Run experiments and generate plots

```bash
python -m PLD_subsampling.main
```

Figures are written to `plots/` (treat this directory as build output).

### Notes

- CDF plots automatically focus the main x-axis on the transition region and add slight y-padding to show the 0 and 1 limits clearly.
- Epsilon-ratio plots show method/GT vs analytical epsilon over log-scale epsilon.
- All heavy computations use vectorized NumPy operations with careful numerical handling in tail regions.