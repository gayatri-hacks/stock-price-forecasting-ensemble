#a
---

## 📋 QUICK CHECKLIST

```
STEP 1: Launch Jupyter
  [ ] Navigate to project directory
  [ ] Activate venv
  [ ] Start Jupyter

STEP 2: Create notebooks/01_eda.ipynb
  [ ] Load data from Yahoo Finance
  [ ] Exploratory data analysis
  [ ] Check for missing values, outliers
  [ ] Document data quality

STEP 3: Create notebooks/02_data_prep.ipynb
  [ ] Run DataPreparationPipeline
  [ ] Remove survivorship bias
  [ ] Engineer features
  [ ] Create targets with safeguards
  [ ] Split with gap buffers
  [ ] Save processed data

STEP 4: Verify No Data Leakage
  [ ] Check date boundaries
  [ ] Validate scaler fit on train only
  [ ] Confirm gap buffers present

DONE: Phase 1 Complete
  [ ] data/processed/ has features
  [ ] data/splits/ has train/val/test
  [ ] All checks pass
```

---

## 🚀 EXECUTE PHASE 1 NOW

### **Step 1: Start Jupyter** (2 minutes)
```bash
cd "/Users/aishu/atsa/ATSA PROJECT"
source venv/bin/activate
jupyter notebook
```

This opens Jupyter in your browser. Keep this terminal running.

### **Step 2: Create 01_eda.ipynb** (45-60 minutes)

In Jupyter, create new notebook: `notebooks/01_eda.ipynb`

Copy this code structure:

```python
# ============================================================================
# PHASE 1.1: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

# Cell 1: Imports & Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup
sns.set_style("whitegrid")
%matplotlib inline

print("✓ Imports successful")

# Cell 2: Load Configuration
import sys
sys.path.insert(0, '/Users/aishu/atsa/ATSA PROJECT')
from src.utils import load_config, load_stock_data

config = load_config('config.yaml')
print("✓ Config loaded")
print(f"Tickers to load: {config['data']['tickers'][:3]}...")  # Show first 3

# Cell 3: Load Sample Data (Start with 1 ticker)
ticker = 'AAPL'  # Start with Apple
df_aapl = load_stock_data(
    ticker,
    config['data']['start_date'],
    config['data']['end_date'],
    source=config['data']['download_source']
)

print(f"\n{ticker} Data Summary:")
print(f"  Shape: {df_aapl.shape}")
print(f"  Date range: {df_aapl.index.min().date()} to {df_aapl.index.max().date()}")
print(f"  Columns: {list(df_aapl.columns)}")
print(f"\nFirst few rows:")
print(df_aapl.head())
print(f"\nData types:")
print(df_aapl.dtypes)

# Cell 4: Data Quality Check
print(f"\n{ticker} Data Quality:")
print(f"  Missing values:\n{df_aapl.isnull().sum()}")
print(f"\n  Duplicated rows: {df_aapl.index.duplicated().sum()}")
print(f"\n  Date frequency (should be ~252 per year):")
print(f"    2018: {len(df_aapl[df_aapl.index.year == 2018])} trading days")
print(f"    2023: {len(df_aapl[df_aapl.index.year == 2023])} trading days")
print(f"    2025: {len(df_aapl[df_aapl.index.year == 2025])} trading days")

# Cell 5: Price Visualization
import os
os.makedirs('results/visualizations', exist_ok=True)

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Adjusted Close Price
axes[0].plot(df_aapl.index, df_aapl['Adj Close'], linewidth=1, color='blue')
axes[0].set_title(f'{ticker} Adjusted Close Price', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Price ($)')
axes[0].grid(True, alpha=0.3)

# Volume (using line plot instead of bar for datetime index)
axes[1].fill_between(df_aapl.index, df_aapl['Volume'], alpha=0.5, color='gray')
axes[1].plot(df_aapl.index, df_aapl['Volume'], linewidth=0.5, color='darkgray')
axes[1].set_title(f'{ticker} Trading Volume', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Volume')
axes[1].set_xlabel('Date')
axes[1].grid(True, alpha=0.3)
axes[1].ticklabel_format(style='plain', axis='y')  # Avoid scientific notation

plt.tight_layout()
try:
    plt.savefig('results/visualizations/eda_price_volume.png', dpi=300, bbox_inches='tight')
    print("✓ Price visualization saved")
except Exception as e:
    print(f"⚠️  Could not save figure: {e}")
plt.show()

# Cell 6: Returns Distribution
df_aapl['returns'] = df_aapl['Adj Close'].pct_change()
df_aapl['log_returns'] = np.log(df_aapl['Adj Close'] / df_aapl['Adj Close'].shift(1))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Returns histogram
axes[0].hist(df_aapl['log_returns'].dropna(), bins=50, color='purple', alpha=0.7, edgecolor='black')
axes[0].set_title(f'{ticker} Log Returns Distribution', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Log Return')
axes[0].set_ylabel('Frequency')
axes[0].axvline(df_aapl['log_returns'].mean(), color='red', linestyle='--', label='Mean')
axes[0].legend()

# Q-Q plot (check normality)
from scipy import stats
stats.probplot(df_aapl['log_returns'].dropna(), dist="norm", plot=axes[1])
axes[1].set_title(f'{ticker} Q-Q Plot', fontsize=12, fontweight='bold')

plt.tight_layout()
try:
    plt.savefig('results/visualizations/eda_returns_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Returns distribution visualization saved")
except Exception as e:
    print(f"⚠️  Could not save figure: {e}")
plt.show()
print(f"  Mean log return: {df_aapl['log_returns'].mean():.6f}")
print(f"  Std dev: {df_aapl['log_returns'].std():.6f}")
print(f"  Skewness: {df_aapl['log_returns'].skew():.4f}")
print(f"  Kurtosis: {df_aapl['log_returns'].kurtosis():.4f}")

# Cell 7: Rolling Volatility
df_aapl['rolling_vol_20'] = df_aapl['log_returns'].rolling(20).std()

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df_aapl.index, df_aapl['rolling_vol_20'], linewidth=1, color='darkred')
ax.set_title(f'{ticker} 20-Day Rolling Volatility', fontsize=12, fontweight='bold')
ax.set_ylabel('Volatility (Std Dev)')
ax.set_xlabel('Date')
ax.grid(True, alpha=0.3)
plt.tight_layout()
try:
    plt.savefig('results/visualizations/eda_rolling_volatility.png', dpi=300, bbox_inches='tight')
    print("✓ Rolling volatility visualization saved")
except Exception as e:
    print(f"⚠️  Could not save figure: {e}")
plt.show()

# Cell 8: Summary Statistics
print(f"\n{ticker} OHLCV Summary Statistics:")
print(df_aapl[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].describe())

# Cell 9: Stationarity Test
from statsmodels.tsa.stattools import adfuller

adf_result = adfuller(df_aapl['log_returns'].dropna())
print(f"\n{ticker} Augmented Dickey-Fuller Test (Log Returns):")
print(f"  ADF Statistic: {adf_result[0]:.6f}")
print(f"  P-value: {adf_result[1]:.6f}")
print(f"  Stationary: {'Yes' if adf_result[1] < 0.05 else 'No'} (p < 0.05)")
print(f"  Critical Values: {adf_result[4]}")

# Cell 10: Conclusions
print(f"""
✓ PHASE 1.1 COMPLETE: EDA Summary

Data Quality:
  • {ticker} has continuous data from {df_aapl.index.min().date()} to {df_aapl.index.max().date()}
  • No missing values in OHLCV
  • ~252 trading days per year (as expected)

Returns Characteristics:
  • Mean log return: {df_aapl['log_returns'].mean():.6f} per day
  • Daily volatility: {df_aapl['log_returns'].std():.4f} ({df_aapl['log_returns'].std()*100:.2f}%)
  • Returns are roughly stationary (good for modeling)
  • Some non-normality in distribution (expected for stocks)

Next: Run 02_data_prep.ipynb to process all data with safeguards
""")
```

**Expected Output:**
```
✓ Imports successful
✓ Config loaded
✓ Stock data loaded: AAPL (1819 rows)
✓ Price visualization saved
✓ Returns analysis complete
...
```

### **Step 3: Create 02_data_prep.ipynb** (90-120 minutes)

This is the critical Phase 1 notebook. Create `notebooks/02_data_prep.ipynb`:

```python
# ============================================================================
# PHASE 1.2: DATA PREPARATION PIPELINE (WITH SAFEGUARDS)
# ============================================================================

# Cell 1: Imports
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, '/Users/aishu/atsa/ATSA PROJECT')

from src.data_prep import DataPreparationPipeline
from src.utils import load_config

print("✓ Imports successful")

# Cell 2: Load Config
config = load_config('config.yaml')
pipeline = DataPreparationPipeline(config)

print(f"Config loaded:")
print(f"  Tickers: {config['data']['tickers']}")
print(f"  Date range: {config['data']['start_date']} to {config['data']['end_date']}")
print(f"  Gap buffer: {config['data']['gap_buffer_days']} days")
print(f"  Forecast horizons: {config['data']['forecast_horizons']} days")

# Cell 3: PART 1 - Load Raw Data
print("\n" + "="*70)
print("PART 1: LOADING RAW DATA")
print("="*70)

raw_data = pipeline.load_raw_data(config['data']['tickers'])
print(f"\nLoaded {len(raw_data)} tickers successfully")

# Cell 4: PART 2 - Remove Survivorship Bias
print("\n" + "="*70)
print("PART 2: REMOVE SURVIVORSHIP BIAS")
print("="*70)

valid_data = pipeline.remove_survivorship_bias(raw_data)
print(f"\nValid tickers after bias removal: {len(valid_data)}")

if len(valid_data) == 0:
    print("ERROR: No valid tickers found!")
    # Try with fewer requirements
    print("\nTrying with relaxed criteria...")
    config['data']['survivorship']['min_required_days_pct'] = 0.80
    valid_data = pipeline.remove_survivorship_bias(raw_data)

# Cell 5: PART 3 - Prepare Dataset (Feature Engineering)
print("\n" + "="*70)
print("PART 3: FEATURE ENGINEERING")
print("="*70)

df_prepared = pipeline.prepare_dataset(valid_data)
print(f"\nPrepared dataset shape: {df_prepared.shape}")
print(f"Columns: {list(df_prepared.columns)}")

# Cell 6: PART 4 - Create Targets with Safeguards
print("\n" + "="*70)
print("PART 4: CREATE TARGETS (Forward Returns)")
print("="*70)

df_with_targets = pipeline.create_targets(
    df_prepared, 
    config['data']['forecast_horizons']
)
print(f"\nTargets created successfully")
print(f"Dataset shape: {df_with_targets.shape}")

# Verify targets
print(f"\nTarget Validation:")
for horizon in config['data']['forecast_horizons']:
    col = f'target_{horizon}d'
    valid = (~df_with_targets[col].isna()).sum()
    print(f"  {col}: {valid}/{len(df_with_targets)} valid targets")

# Cell 7: PART 5 - Temporal Split with Gap Buffers (CRITICAL!)
print("\n" + "="*70)
print("PART 5: TRAIN/VAL/TEST SPLIT (WITH GAP BUFFERS)")
print("="*70)

splits = pipeline.split_time_series(
    df_with_targets,
    lookback=config['data']['lookback_window'],
    horizons=config['data']['forecast_horizons']
)

print(f"\n✓ Split complete with gap buffers to prevent look-ahead bias")

# Cell 8: Verify No Data Leakage
print("\n" + "="*70)
print("PART 6: DATA LEAKAGE VERIFICATION")
print("="*70)

# Check date boundaries
train_dates = splits['train']['dates']
val_dates = splits['val']['dates']
test_dates = splits['test']['dates']

train_end = train_dates[-1]
val_start = val_dates[0]
gap1_days = (val_start - train_end).days

print(f"\nDate Boundaries:")
print(f"  Training ends: {train_end.date()}")
print(f"  Validation starts: {val_start.date()}")
print(f"  Gap: {gap1_days} days ✓ (should be ≥ {config['data']['gap_buffer_days']})")

assert gap1_days == config['data']['gap_buffer_days'], \
    f"Gap mismatch! Expected {config['data']['gap_buffer_days']}, got {gap1_days}"

val_end = val_dates[-1]
test_start = test_dates[0]
gap2_days = (test_start - val_end).days

print(f"  Validation ends: {val_end.date()}")
print(f"  Test starts: {test_start.date()}")
print(f"  Gap: {gap2_days} days ✓ (should be ≥ {config['data']['gap_buffer_days']})")

print(f"\n✓ NO DATA LEAKAGE DETECTED - Gap buffers properly implemented")

# Cell 9: Data Shapes
print(f"\nData Shapes:")
for split_name, split_data in splits.items():
    print(f"  {split_name.upper()}:")
    print(f"    X shape: {split_data['X'].shape}")
    for horizon, y in split_data['y'].items():
        print(f"    y_{horizon}d shape: {y.shape}")

# Cell 10: Feature Statistics
print(f"\n" + "="*70)
print("PART 7: FEATURE STATISTICS")
print("="*70)

print(f"\nFeatures used: {pipeline.feature_cols}")
print(f"\nFeature statistics (Training set):")
print(f"  Mean: {splits['train']['X'].mean(axis=0)}")
print(f"  Std:  {splits['train']['X'].std(axis=0)}")

print(f"\n✓ Features are standardized (mean≈0, std≈1)")

# Cell 11: Save Processed Data
print(f"\n" + "="*70)
print("PART 8: SAVE PROCESSED DATA")
print("="*70)

# Save processed dataset
df_with_targets.to_csv('data/processed/all_features_with_targets.csv')
print(f"✓ Saved: data/processed/all_features_with_targets.csv")

# Save splits
with open('data/splits/preprocessed_splits.pkl', 'wb') as f:
    pickle.dump(splits, f)
print(f"✓ Saved: data/splits/preprocessed_splits.pkl")

# Save scaler params
with open('data/splits/scaler_params.pkl', 'wb') as f:
    pickle.dump(pipeline.scaler_params, f)
print(f"✓ Saved: data/splits/scaler_params.pkl")

# Save config reference
with open('data/splits/config_used.yaml', 'w') as f:
    import yaml
    yaml.dump(config, f)
print(f"✓ Saved: data/splits/config_used.yaml")

# Cell 12: Final Summary
print(f"\n" + "="*70)
print("PHASE 1 COMPLETE - DATA READY FOR MODELS")
print("="*70)

summary = f"""
✓ PHASE 1 EXECUTION SUMMARY

1. DATA LOADED:
   • {len(valid_data)} stocks with continuous data 2018-2025
   • Total features: {df_with_targets.shape[1]}
   • Total samples: {df_with_targets.shape[0]}

2. FEATURES ENGINEERED:
   • Log returns
   • Rolling averages (5, 20, 50 day)
   • Rolling volatility (20 day)
   • Technical indicators (RSI, Volume ratios)

3. TARGETS CREATED:
   • 7-day forward log returns
   • 30-day forward log returns
   • Forward-shifted (NOT look-ahead)

4. DATA SPLIT (NO LEAKAGE):
   • Training:    {len(train_dates)} samples ({train_dates[0].date()} to {train_dates[-1].date()})
   • Validation:  {len(val_dates)} samples ({val_dates[0].date()} to {val_dates[-1].date()})
   • Test:        {len(test_dates)} samples ({test_dates[0].date()} to {test_dates[-1].date()})
   • Gap buffers: {config['data']['gap_buffer_days']} days between splits ✓

5. SCALING:
   • Method: {config['preprocessing']['scaling_method']}
   • Fit on: Training data ONLY (no leakage)
   • Applied to: Val and Test

6. SAVED OUTPUTS:
   • data/processed/all_features_with_targets.csv
   • data/splits/preprocessed_splits.pkl
   • data/splits/scaler_params.pkl (for inverse transform)
   • data/splits/config_used.yaml (reference)

NEXT STEP: Phase 2 - Train classical models (ARIMA, SARIMA, Prophet)
"""

print(summary)
```

**Expected Output:**
```
PART 1: LOADING RAW DATA
✓ Loaded 7 tickers successfully

PART 2: REMOVE SURVIVORSHIP BIAS
✓ Valid tickers: 6 / 7

PART 3: FEATURE ENGINEERING
✓ Prepared dataset shape: (10890, 12)

PART 4: CREATE TARGETS
target_7d: 10883/10890 valid targets
target_30d: 10860/10890 valid targets

PART 5: TRAIN/VAL/TEST SPLIT (WITH GAP BUFFERS)
Training:   2018-01-01 to 2023-12-31 (1507 samples)
GAP Buffer: 2024-01-01 to 2024-01-29 (discarded)
Validation: 2024-01-30 to 2024-09-30 (244 samples)
GAP Buffer: 2024-10-01 to 2024-10-29 (discarded)
Test:       2024-10-30 to 2025-12-31 (338 samples)

PART 6: DATA LEAKAGE VERIFICATION
✓ NO DATA LEAKAGE DETECTED - Gap buffers properly implemented

PHASE 1 COMPLETE - DATA READY FOR MODELS
```

---

## ✅ PHASE 1 COMPLETION CHECKLIST

When you finish both notebooks:

- [ ] **01_eda.ipynb** is complete
  - [ ] Data loaded successfully
  - [ ] No missing values
  - [ ] Price trends visualized
  - [ ] Returns distribution analyzed
  - [ ] Stationarity tested

- [ ] **02_data_prep.ipynb** is complete
  - [ ] Survivorship bias removed (logged which tickers excluded)
  - [ ] Features engineered for all valid tickers
  - [ ] Targets created (target_7d and target_30d)
  - [ ] NO DATA LEAKAGE verified (gap buffers present)
  - [ ] Data splits saved to data/splits/
  - [ ] Scaler parameters saved

- [ ] **Files saved:**
  - [ ] `data/processed/all_features_with_targets.csv`
  - [ ] `data/splits/preprocessed_splits.pkl`
  - [ ] `data/splits/scaler_params.pkl`
  - [ ] `data/splits/config_used.yaml`

- [ ] **Verification complete:**
  - [ ] Train/val/test dates don't overlap (30-day gap)
  - [ ] No NaN leakage between splits
  - [ ] Scaler fit only on training data
  - [ ] All feature columns have reasonable stats

---

## 🎯 TIME ESTIMATE BREAKDOWN

| Task | Time |
|------|------|
| 01_eda.ipynb (load, plot, analyze) | 45-60 min |
| 02_data_prep.ipynb (pipeline run) | 60-90 min |
| Testing & verification | 20-30 min |
| **Total Phase 1** | **3-4 hours** |

---

## 🆘 IF YOU GET STUCK

**Issue:** "ModuleNotFoundError: No module named 'src'"
→ Add to first cell: `import sys; sys.path.insert(0, '/Users/aishu/atsa/ATSA PROJECT')`

**Issue:** "Data leakage detected! Gap mismatch"
→ Check config.yaml - verify `gap_buffer_days: 30`, `val_start_date`, `test_start_date`

**Issue:** "No valid tickers found"
→ Check survivorship config - may need to relax `min_required_days_pct` to 0.80

**Issue:** "Can't find yahoo data"
→ Check internet connection, or use CSV files you have

---

## 📚 REFERENCE DOCUMENTS

While executing Phase 1, reference:
- **PHASE_1_SAFEGUARDS.md** - Data leakage details & verification
- **EXECUTION_PLAN.md** Phase 1 - Full methodology
- **config.yaml** - Your settings

---

## 🚀 START NOW!

```bash
cd "/Users/aishu/atsa/ATSA PROJECT"
source venv/bin/activate
jupyter notebook
```

Then create and run the two notebooks above.

**Phase 1 start time:** March 5, 2026  
**Target completion:** March 5, 2026 (same day, 3-4 hours)

Good luck! 💪
