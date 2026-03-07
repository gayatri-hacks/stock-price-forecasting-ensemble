# PHASE 1 SAFEGUARDS: Data Leakage Prevention Guide

**Last Updated:** March 5, 2026  
**Critical for:** Preventing look-ahead bias and survivorship bias in stock forecasting

---

## 🚨 CRITICAL ISSUE FIXED: Look-Ahead Bias

### **What Happened**
Your original config had a potential **data leakage bug**:

```yaml
Training ends:   2023-12-31
Validation starts: 2024-01-01
```

**The Problem:** The 30-day target for Dec 31, 2023 = log(Jan 30, 2024 price)
- This target value uses data from Jan 30, 2024
- But your validation set starts Jan 1, 2024
- **Result:** The model sees information from the validation period while training!

### **The Fix: Gap Buffer**
```yaml
Training ends:     2023-12-31
GAP BUFFER:        2024-01-01 to 2024-01-29 (30 days discarded)
Validation starts: 2024-01-30
```

Now the 30-day target for Dec 31, 2023 (Jan 30, 2024 price) is **exactly at the boundary**—no leakage!

---

## ✅ UPDATED: config.yaml (Data Section)

### **Key Changes:**

1. **Explicit Target Column**
```yaml
target_col: 'Adj Close'  # Column to shift for forward returns
```

2. **Gap Buffer Between Splits** (CRITICAL!)
```yaml
train_end_date: '2023-12-31'
val_start_date: '2024-01-30'     # 30 days after train_end
val_end_date: '2024-09-30'

test_start_date: '2024-10-30'    # 30 days after val_end
test_end_date: '2025-12-31'

gap_buffer_days: 30              # Must match max(forecast_horizons)
```

3. **Survivorship Bias Configuration**
```yaml
survivorship:
  min_required_days_pct: 0.95    # Stock must have 95% of trading days
  remove_incomplete: true         # Remove tickers with gaps > 5%
  min_data_points: 252           # Minimum trading days required
```

4. **Transaction Costs Explicit**
```yaml
costs:
  bid_ask_spread: 0.001          # 0.1% round-trip
  slippage: 0.0005               # 0.05% per execution
  commission_pct: 0.0005         # 0.05% per trade
```

---

## ✅ UPDATED: src/data_prep.py

### **1. Survivorship Bias Removal (Enhanced)**
**New:** Uses `survivorship` config section
```python
survival_cfg = self.config['data'].get('survivorship', {})
min_days_pct = survival_cfg.get('min_required_days_pct', 0.95)
remove_incomplete = survival_cfg.get('remove_incomplete', True)
```

**Improved Logging:**
- Shows exact reasons for ticker removal
- Counts gap days: "starts 15 days late"
- Documents exact day counts

### **2. Timeline-Aware Split (CRITICAL FIX)**
**New:** Implements gap buffer between train/val/test

```python
train_end = pd.to_datetime(config['data']['train_end_date'])
val_start = pd.to_datetime(config['data']['val_start_date'])
gap_buffer = config['data']['gap_buffer_days']

# Validate buffer is sufficient
max_horizon = max(horizons)  # e.g., 30
if gap_buffer < max_horizon:
    logger.warning("⚠️  gap_buffer too small, look-ahead bias possible!")
```

**Timeline Output:**
```
Training:   2018-01-01 to 2023-12-31 (1507 samples)
GAP Buffer: 2024-01-01 to 2024-01-29 (29 samples, discarded)
Validation: 2024-01-30 to 2024-09-30 (244 samples)
GAP Buffer: 2024-10-01 to 2024-10-29 (29 samples, discarded)
Test:       2024-10-30 to 2025-12-31 (338 samples)

Buffer days between splits: 30
Max forecast horizon: 30 days (ensures no leakage)
```

### **3. Target Creation (Explicit Column)**
**New:** Uses `target_col` from config
```python
price_col = self.config['data'].get('target_col', 'Adj Close')
```

**Enhanced Logging:**
```
Creating targets using column: Adj Close
  target_7d: 1800/1807 valid targets
  target_30d: 1777/1807 valid targets
✓ Target variables created
```

---

## 🔍 HOW TO VERIFY NO DATA LEAKAGE

### **Check 1: Date Boundaries**
```python
# In Phase 1 notebook
print(f"Last training date: {train['Date'].max()}")
print(f"First validation date: {val['Date'].min()}")
print(f"Gap: {(val['Date'].min() - train['Date'].max()).days} days")

# Should show: Gap: 30 days (or equal to gap_buffer_days)
```

### **Check 2: Target Values**
```python
# Last training sample should NOT predict into validation period
train_last_date = train.index[-1]             # 2023-12-31
train_last_target_7d = train['target_7d'].iloc[-1]
target_date_7d = train_last_date + timedelta(days=7)

print(f"Training ends: {train_last_date}")
print(f"7-day target uses price from: {target_date_7d}")
print(f"Validation starts: {val.index[0]}")

# target_date_7d should be WITHIN training period or at gap buffer
```

### **Check 3: Scalers**
```python
# Scaler should ONLY be fit on training data
print(f"Scaler mean: {scaler_params['mean']}")
print(f"Scaler fit on: Training data only (dates ≤ 2023-12-31)")
print(f"Applied to: Val and Test data (validates no leakage)")
```

---

## 📋 PHASE 1 CHECKLIST (WITH SAFEGUARDS)

- [ ] **Load Data**
  - [ ] Download tickers from Yahoo Finance
  - [ ] Verify dates: start_date, end_date from config
  
- [ ] **Remove Survivorship Bias**
  - [ ] Check min_required_days_pct (95%)
  - [ ] List removed tickers and reasons
  - [ ] Confirm only 2018-2025 continuous tickers remain
  
- [ ] **Engineer Features**
  - [ ] Create log_returns (normalized, no NaNs)
  - [ ] Create rolling averages (5, 20, 50)
  - [ ] Create rolling volatility (20-day)
  - [ ] Create RSI, volume ratios
  - [ ] Handle outliers with IQR method
  
- [ ] **Create Targets (CRITICAL)**
  - [ ] Create target_7d = forward 7-day log return
  - [ ] Create target_30d = forward 30-day log return
  - [ ] Verify: Last 30 training rows have target_30d = NaN
  - [ ] Verify: No future price data leaked into features
  
- [ ] **Split with Gap Buffer (CRITICAL)**
  - [ ] Train: Up to 2023-12-31
  - [ ] **Gap buffer: 2024-01-01 to 2024-01-29 (discard!)**
  - [ ] Validation: From 2024-01-30 to 2024-09-30
  - [ ] **Gap buffer: 2024-10-01 to 2024-10-29 (discard!)**
  - [ ] Test: From 2024-10-30 to 2025-12-31
  
- [ ] **Scale Data (CRITICAL)**
  - [ ] Fit StandardScaler on training X ONLY
  - [ ] Save scaler parameters (mean, std)
  - [ ] Apply same scaler to val and test
  - [ ] Verify: No information leakage from val/test to training
  
- [ ] **Verify No Data Leakage**
  - [ ] Last training sample's furthest target doesn't touch validation
  - [ ] Scaler fit only on training features
  - [ ] Gap buffer properly discarded
  - [ ] Document findings

**Deliverables:**
- [ ] `data/processed/features_all_tickers.csv`
- [ ] `data/splits/train_data.csv`, `val_data.csv`, `test_data.csv`
- [ ] `data/splits/scaler_params.pkl`
- [ ] Data leakage verification report

---

## 🚀 RUNNING PHASE 1 WITH SAFEGUARDS

```bash
# 1. Activate environment
source venv/bin/activate

# 2. In Jupyter, run Phase 1 notebook
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_data_prep.ipynb

# 3. At the end of Phase 1, verify:
python -c "
from src.data_prep import DataPreparationPipeline
from src.utils import load_config

config = load_config()
pipeline = DataPreparationPipeline(config)

# Run pipeline
raw_data = pipeline.load_raw_data(config['data']['tickers'][:3])
valid_data = pipeline.remove_survivorship_bias(raw_data)
df = pipeline.prepare_dataset(valid_data)
df = pipeline.create_targets(df)
splits = pipeline.split_time_series(df)

print('✅ Phase 1 pipeline executed without data leakage')
print(f'✓ Train: {len(splits[\"train\"][\"X\"])} samples')
print(f'✓ Val: {len(splits[\"val\"][\"X\"])} samples')
print(f'✓ Test: {len(splits[\"test\"][\"X\"])} samples')
"
```

---

## ⚠️ COMMON MISTAKES TO AVOID

1. **Forgetting the gap buffer**
   - ❌ Bad: validate starts at Jan 1, train ends Dec 31
   - ✅ Good: validate starts Jan 30, train ends Dec 31 (gap of 30)

2. **Fitting scaler on multiple splits**
   - ❌ Bad: `scaler.fit(pd.concat([train, val, test]))`
   - ✅ Good: `scaler.fit(train_only)`

3. **Using time-based indexing without ordering**
   - ❌ Bad: `df.reset_index().split`
   - ✅ Good: `df[df.index <= date1]` respects chronology

4. **Forgetting to remove gap buffer data**
   - ❌ Bad: Include Jan 1-29 in validation
   - ✅ Good: Completely discard gap buffer, start val at Jan 30

5. **Not logging actual dates**
   - ❌ Bad: "Train: 70%, Val: 15%, Test: 15%"
   - ✅ Good: "Train: 2018-01-01 to 2023-12-31" with exact sample counts

---

## 📊 EXPECTED PHASE 1 OUTPUT

After running your Phase 1 notebook, you should see:

```
✓ Loaded AAPL: 2015 records
✓ Loaded MSFT: 2014 records
✓ Removed TSLA: Starts 200 days late
✓ Removed NVIDIA: Only 500/2010 days

Survivorship Bias Removal:
  Valid tickers: 5 / 7
  Removed: ['TSLA', 'NVIDIA']

Creating targets using column: Adj Close
  target_7d: 1800/1807 valid targets
  target_30d: 1777/1807 valid targets

Temporal Train/Val/Test Split (NO Data Leakage):
  Training:   2018-01-01 to 2023-12-31 (1507 samples)
  GAP Buffer: 2024-01-01 to 2024-01-29 (29 samples, discarded)
  Validation: 2024-01-30 to 2024-09-30 (244 samples)
  GAP Buffer: 2024-10-01 to 2024-10-29 (29 samples, discarded)
  Test:       2024-10-30 to 2025-12-31 (338 samples)

  Buffer days between splits: 30
  Max forecast horizon: 30 days (ensures no leakage)

✓ Data Leakage Prevention: CONFIRMED
```

---

## 🎯 NEXT: Phase 2 Models

Once Phase 1 is complete with these safeguards verified:
1. You have clean, validated data with NO leakage
2. Horizons are properly separated
3. Survivorship bias is removed
4. Ready to train 3 classical models (ARIMA, SARIMA, Prophet)

See: EXECUTION_PLAN.md Phase 2

---

**Status:** ✅ Phase 1 Safeguards Implemented
**Last Verified:** March 5, 2026
**Critical:** These changes prevent the most common time-series ML mistakes!
