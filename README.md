### PLD_subsampling

Implements and evaluates privacy amplification by subsampling for Privacy Loss Distribution (PLD) probability mass functions (PMFs). Generates CDF plots and epsilon ratio plots comparing analytical ground truth, `dp-accounting`, and our direct subsampling implementation.

### Package layout

- `PLD_subsampling/`
  - `PLD_subsampling_impl.py`: Core subsampling primitives
    - `stable_subsampling_loss`: numerically stable loss mapping
    - `exclusive_ccdf_from_pdf`: CCDF helper (exclusive tail)
    - `subsample_losses`: transforms a PMF on a uniform loss grid
  - `wrappers/dp_accounting_wrappers.py`: Thin wrappers around `dp-accounting` (construct PLDs, amplify PLDs separately for remove/add), plus PMF/PLD utilities
    - `amplify_pld_separate_directions(base_pld, sampling_prob) -> PrivacyLossDistribution`: returns a PLD with amplified remove/add PMFs
    - `scale_pmf_infinity_mass(pmf, delta) -> PMF`: increases the infinity mass by `delta` and scales all finite probabilities by `(1-β-δ)/(1-β)` preserving PMF type (dense/sparse)
    - `scale_pld_infinity_mass(pld, delta) -> PrivacyLossDistribution`: applies the same infinity-mass change to both directions of a PLD and returns a new PLD
  - `testing/`
    - `analytic_Gaussian.py`: Analytical PLD and epsilon(δ) formulas for Gaussian mechanism
    - `test_utils.py`: Experiment runners (`run_experiment`, `run_multiple_experiments`)
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

2) Editable install for local development (optional)

```bash
pip install -e .
```

3) Run experiments and generate plots

```bash
python -m PLD_subsampling.main
```

Figures are written to `plots/` (treat this directory as build output).

### Usage examples

Scale infinity mass on a PMF and a PLD (see `PLD_subsampling/example.ipynb` for a full demo):

```python
from PLD_subsampling.wrappers.dp_accounting_wrappers import (
    scale_pmf_infinity_mass,
    scale_pld_infinity_mass,
)
from dp_accounting.pld import privacy_loss_distribution

# Build a fresh PLD
pld = privacy_loss_distribution.from_gaussian_mechanism(
    standard_deviation=1.0,
    sensitivity=1.0,
    value_discretization_interval=1e-4,
    sampling_prob=0.1,
    pessimistic_estimate=True,
)

# Scale a single PMF
pmf_scaled = scale_pmf_infinity_mass(pld._pmf_remove, delta=1e-4)

# Scale both directions of the PLD
pld_scaled = scale_pld_infinity_mass(pld, delta=1e-4)
```

### Notes

- CDF plots automatically focus the main x-axis on the transition region and add slight y-padding to show the 0 and 1 limits clearly.
- Epsilon-ratio plots show method/GT vs analytical epsilon over log-scale epsilon.
- All heavy computations use vectorized NumPy operations with careful numerical handling in tail regions.

### Build a package

```bash
python -m pip install --upgrade build
python -m build
```

Artifacts will be created under `dist/`. To upload to PyPI/TestPyPI, use `twine` with an API token.