# Econformal

Conformal Inference Toolkit for Econometric Models

[![PyPI version](https://badge.fury.io/py/econformal.svg)](https://badge.fury.io/py/econformal)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Introduction

Econformal is a Python toolkit that combines conformal inference with econometric models, providing statistically-guaranteed uncertainty quantification (confidence intervals) for treatment effects in DID, Synthetic Control, and other econometric models.

## Installation

```bash
pip install econformal
```

## Quick Start

The following four examples progressively demonstrate Econformal's core functionality. Each code block can be run independently.

### 1. Understanding Panel Data Structure

Before using any econometric model, it's essential to understand the panel data format Econformal requires.

```python
from econformal.tools.generate_data import generate_test_panel_data

# Generate a small panel dataset (5 units, 2 treated, 6 periods)
data = generate_test_panel_data(
    n_ids=5, n_treated=2, start_year=2000,
    pre_periods=4, post_periods=2, x_num=1, seed=99
)

print(data.head(12))
```

**Column Descriptions:**

| Column | Description |
|------|------|
| `year` | Time variable, must be sortable (int/float/datetime) |
| `id` | Unit identifier, each unit has a unique ID |
| `Y` | Dependent/outcome variable |
| `Treat` | Treatment indicator, **must be binary 0/1**. 0 = control group (never treated), 1 = treatment group (treated after treatment onset) |
| `X1`, `X2`, ... | Covariates/control variables (optional), used to control for confounding factors |

**Data Constraints:**
- Must be a strongly balanced panel: each `(id, time)` combination is unique
- The `Treat` column must contain both 0 and 1
- Column names must not contain the `'T_'` prefix (DID internal reserved prefix)
- No missing values (NaN)

---

### 2. SC + Full Conformal — The Most Rigorous Combination

Synthetic Control is suited for scenarios with only 1 (or very few) treated units. Full Conformal searches for confidence intervals over a grid of null hypotheses via permutation tests, providing exact finite-sample coverage guarantees.

```python
import numpy as np
import matplotlib.pyplot as plt
from econformal import Econformal
from econformal.tools.generate_data import generate_test_panel_data

# 1. Generate data: 30 units, only 1 treated
data = generate_test_panel_data(
    n_ids=30, n_treated=1, start_year=2000,
    pre_periods=10, post_periods=5, x_num=0, seed=99
)

# 2. Initialize (SC does not use covariates, controls_col=None)
model = Econformal(
    data=data, time='year', id='id',
    y_col='Y', treat_col='Treat', controls_col=None
)

# 3. Set the null hypothesis grid
# nulls is an array of candidate treatment effect values. Full Conformal runs
# a permutation test for each value, retaining those with p-value >= (1-coverage)
# and taking min/max as the confidence interval.
# Range too narrow -> all nulls rejected -> bounds become NaN
# Range too wide -> increased computation
nulls = np.linspace(-10, 10, 50)

# 4. Run conformal inference (SC + Full Conformal, 90% confidence level)
result = model.conformal_inference(
    econ_model='sc', conformal_model='full',
    nulls=nulls, coverage=0.9
)
print(result.round(4))

# 5. Plot confidence intervals
fig = model.plot_ci_interval()
fig.savefig('sc_full_conformal.png', dpi=150, bbox_inches='tight')
plt.close()
```

**Output Column Descriptions:**

| Column | Description |
|------|------|
| `year` | Time |
| `effect` | Estimated treatment effect (actual Y − SC synthetic predicted Y) |
| `predictions` | SC synthetic predicted values |
| `std_error` | Standard error |
| `p-value` | P-value |
| `90%_conformal_lower` | Conformal 90% CI lower bound (only post-treatment periods have values) |
| `90%_conformal_upper` | Conformal 90% CI upper bound |

> ⚠ Full Conformal number of fits = post-treatment periods × number of nulls. This example performs ~250 SC fits and may take tens of seconds.

---

### 3. DID + Split Conformal — Fast Version for Large Datasets

Difference-in-Differences is one of the most widely used causal inference methods. Split Conformal splits pre-treatment periods into training and calibration sets, requiring only **2** model fits and delivering results in seconds.

```python
import numpy as np
import matplotlib.pyplot as plt
from econformal import Econformal
from econformal.tools.generate_data import generate_test_panel_data

# 1. Generate large-sample data with covariates
data = generate_test_panel_data(
    n_ids=200, n_treated=50, start_year=2010,
    pre_periods=15, post_periods=8, x_num=3, seed=123
)

# 2. Initialize (include covariates to control for confounding)
model = Econformal(
    data=data, time='year', id='id', y_col='Y',
    treat_col='Treat', controls_col=['X1', 'X2', 'X3']
)

# 3. Run conformal inference
# split_rate=0.7: first 70% of pre-treatment periods for training,
#                 last 30% for calibrating quantiles
# Split Conformal does not require the nulls parameter
result = model.conformal_inference(
    econ_model='did', conformal_model='split',
    split_rate=0.7, coverage=0.90
)
print(result.round(4))

# 4. Comparative plot: Conformal CI (I-bars) vs Traditional CI (shaded band)
fig = model.plot_ci_interval(traditional=True)
fig.savefig('did_split_conformal.png', dpi=150, bbox_inches='tight')
plt.close()
```

**Interpreting Results:**
- Pre-treatment (left of dashed line): effect should be near 0 (parallel trends assumption)
- Post-treatment (right of dashed line): effect deviating from 0 indicates treatment effect
- Conformal CI (I-bars) is generally wider than traditional CI (shaded band) — more honest about uncertainty
- Traditional CI relies on model assumptions (normality); conformal CI provides distribution-free coverage guarantees

---

### 4. SDID + CV+ Conformal — Modern Method Combination

Synthetic Difference-in-Differences (Arkhangelsky et al., 2021) combines the strengths of DID and SC. CV+ is a K-fold generalization of Jackknife+, reducing the number of fits from n_pre to K (default 5).

```python
import numpy as np
import matplotlib.pyplot as plt
from econformal import Econformal
from econformal.tools.generate_data import generate_test_panel_data

# 1. Generate data: 100 units, 30 treated (SDID supports multiple treated units)
data = generate_test_panel_data(
    n_ids=100, n_treated=30, start_year=2010,
    pre_periods=12, post_periods=6, x_num=2, seed=42
)

# 2. Initialize
model = Econformal(
    data=data, time='year', id='id', y_col='Y',
    treat_col='Treat', controls_col=['X1', 'X2']
)

# 3. Run conformal inference (SDID + CV+)
# cv_folds=5         : K=5 fold cross-validation
# cv_strategy='block' : contiguous time blocks (preserves temporal dependence, recommended for panel data)
#                       'random' for shuffled folds (assumes exchangeability)
result = model.conformal_inference(
    econ_model='sdid', conformal_model='cv_plus',
    coverage=0.90, cv_folds=5, cv_strategy='block', random_state=42
)
print(result.round(4))

# 4. Plot CI chart
fig = model.plot_ci_interval(traditional=True)
fig.savefig('sdid_cvplus_conformal.png', dpi=150, bbox_inches='tight')
plt.close()
```

> ⚠ Methodological note: DID + JK+/CV+ uses proxy residuals (not standard held-out prediction errors), so the coverage guarantee does not strictly hold. It is recommended to pair SC/SDID with JK+/CV+.

---

## Method Selection Guide

Three econometric models × five conformal methods = 15 possible combinations. Choose based on the scenarios below:

### Econometric Model Comparison

| Model | Use Case | Treated Units | Covariate Support | Core Idea |
|------|---------|-----------|-----------|---------|
| **SC** | Single or very few treated units | 1 | Limited | Convex combination of control units for counterfactual |
| **DID** | Large panels, many treated units | Any | ✓ | Difference-in-differences + event study |
| **SDID** | Medium scale, robustness-focused | Multiple | ✓ | SC weights + DID time weights |

### Conformal Method Comparison

| Method | Fits | Interval Shape | Coverage Guarantee | Use Case |
|------|---------|---------|---------|---------|
| **Full** | n_post × n_nulls | Varies by time point | Exact finite-sample | Small samples, maximum rigor |
| **Split** | 2 | Equal width across time | Approximate | Large data, fast iteration |
| **LOO** | n_pre | Equal width across time | Conservative in simulation | Medium data, robustness-focused |
| **JK+** | n_pre | Asymmetric by time point | P(Y∈CI) ≥ 1−2α | Medium data, heteroskedastic settings |
| **CV+** | K (default 5) | Asymmetric by time point | P(Y∈CI) ≥ 1−2α (approx.) | Many time periods, efficiency-focused |

### Recommended Combinations

| Scenario | Recommended | Rationale |
|------|---------|------|
| Small samples, rigor | SC + Full | SC has 1 treated unit; Full provides exact finite-sample guarantee |
| Large data, fast iteration | DID + Split | 2 fits, results in seconds |
| Routine analysis, balanced | SDID + CV+ | Modern method + K-fold calibration, suitable for formal reports |
| Theoretical guarantees | SC + JK+ / CV+ | SC enables true leave-one-out prediction |

---

## Practical Tips

### Hyperparameter Tuning

- **nulls range too narrow** → CI bounds are NaN → expand `np.linspace` range
- **Full Conformal too slow** → reduce nulls points (e.g., 100 → 30) or switch to split
- **Split intervals too wide** → increase `split_rate` (more training data improves model fit)
- **CV+ fold selection** → use `block` + few folds for few pre-treatment periods; use `random` + more folds for many periods
- **coverage selection** → 90% is the common default; 95% is more conservative; 80% is narrower but lower coverage

### Using Your Own Data

```python
import pandas as pd
import numpy as np
from econformal import Econformal

my_data = pd.read_csv('my_panel_data.csv')
model = Econformal(
    data=my_data,
    time='year',           # your time column name
    id='state',            # your unit identifier column name
    y_col='outcome',       # your outcome column name
    treat_col='treated',   # your treatment indicator column name (must be 0/1)
    controls_col=['gdp', 'population']  # optional covariates
)
result = model.conformal_inference(
    econ_model='did',
    conformal_model='full',
    nulls=np.linspace(-20, 20, 100),
    coverage=0.95
)
print(result)
```

### Saving High-Resolution Figures

```python
fig = model.plot_ci_interval(traditional=True)
fig.savefig('my_ci_plot.png', dpi=300, bbox_inches='tight')
plt.close()
```

---

## Core Modules

- **conformal_methods**: Conformal inference methods
  - `full` — Full Conformal (null grid search + permutation test)
  - `split` — Split Conformal (train/calibration split + quantiles)
  - `loo` — Leave-One-Out Conformal (LOO calibration)
  - `jk_plus` — Jackknife+ Conformal (asymmetric intervals)
  - `cv_plus` — CV+ Conformal (K-fold cross-validation)

- **econometrics_methods**: Econometric methods
  - `did` — Difference-in-Differences (event study + PanelOLS)
  - `sc` — Synthetic Control (cvxpy convex optimization for weights)
  - `sdid` — Synthetic Difference-in-Differences (unit/time double weighting)

- **tools**: Utility functions
  - `check` — Data validation (strong panel check, column name check, missing value check)
  - `generate_data` — Synthetic panel data generation
  - `plot` — Visualization (effect curve + conformal confidence intervals)
  - `model_registration` — Dynamic model loading

## Example Script

`example.py` provides a runnable 3-section simple example (SC/DID/SDID + Full Conformal).

## Contributing

Contributions, bug reports, and suggestions are welcome!

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Contact

- Author: Forry Wu
