# PROJECT 4: Multi-Horizon Stock Forecasting with LSTM + Transformers + Prophet
## Executive Execution Plan

**Project Goal:** Predict 7-day and 30-day future prices using classical and deep learning models with explainability analysis.

---

## PHASE 0: PROJECT SETUP & ENVIRONMENT (~1-2 hours)

### 0.1 Directory Structure
```
project_root/
├── data/
│   ├── raw/                    # Original stock data
│   ├── processed/              # Cleaned, bias-removed data
│   └── splits/                 # Train/val/test sets
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory data analysis
│   ├── 02_data_prep.ipynb     # Data preparation & validation
│   ├── 03_classical_models.ipynb
│   ├── 04_lstm_models.ipynb
│   ├── 05_transformer.ipynb
│   └── 06_evaluation_viz.ipynb
├── src/
│   ├── data_prep.py           # Data handling utilities
│   ├── models/
│   │   ├── classical.py       # ARIMA, SARIMA, Prophet
│   │   ├── lstm.py            # LSTM models
│   │   └── transformer.py     # Transformer models
│   ├── evaluation.py           # Metrics (MAPE, RMSE, directional accuracy)
│   ├── explainability.py       # Attention visualization, SHAP
│   └── utils.py               # Helpers (scaling, windowing, etc.)
├── results/
│   ├── predictions/            # Model forecasts
│   ├── metrics/               # Evaluation results
│   ├── visualizations/         # Plots & attention heatmaps
│   └── reports/               # Summary findings
├── requirements.txt
├── config.yaml                 # Hyperparameters
└── README.md
```

### 0.2 Dependencies Installation
```
Core: pandas, numpy, scikit-learn
Time Series: statsmodels, fbprophet, pmdarima
Deep Learning: tensorflow/torch, keras
Explainability: shap, lime
Visualization: matplotlib, seaborn, plotly
Misc: pyyaml, tqdm, joblib
```

### 0.3 Configuration File (config.yaml)
- Train/val/test split ratios
- Hyperparameters for each model
- Scaling strategy
- Lookback window size (e.g., 60 days)
- Forecast horizons (7 & 30 days)
- Transaction costs & slippage %

**Deliverables:**
- [ ] Project directory created with proper structure
- [ ] requirements.txt finalized
- [ ] config.yaml with all hyperparameters
- [ ] Environment activated, dependencies installed

---

## PHASE 1: DATA PREPARATION (~3-4 hours)

### 1.1 Load & Exploratory Analysis
**Input:** Raw stock data (OHLCV)
**Output:** EDA report with statistics
- Load multiple tickers to simulate universe
- Check for gaps, missing values, outliers
- Plot price series, returns distribution
- Calculate rolling volatility

### 1.2 Remove Survivorship Bias
**Critical Step:** Ensure only include stocks that existed during entire training period
- Document delisted/merged stocks separately
- Verify date range consistency across tickers
- Create a "valid ticker list" for analysis

### 1.3 Feature Engineering
```
Primary Features:
- Returns (log returns)
- Rolling mean (5, 20, 50 days)
- Rolling volatility (20 days)
- RSI, MACD, Bollinger Bands (optional)
- Volume indicators

Target:
- 7-day forward return (% change or log return)
- 30-day forward return
```

### 1.4 Train-Test Split (NO LOOK-AHEAD BIAS)
```
Timeline: [2018-01-01 to 2025-12-31]
Training:   2018-01-01 to 2023-12-31 (70%)
Validation: 2024-01-01 to 2024-09-30 (15%)
Testing:    2024-10-01 to 2025-12-31 (15%)

Key: Use walk-forward validation for time series
- NO mixing of future data with past
- Each data point's features use only past data
```

### 1.5 Adjust for Slippage & Transaction Costs
```
Realistic Adjustments:
- Bid-ask spread: ±0.1% on entry
- Slippage: ±0.05% per trade
- Commissions: 0.05% per transaction
Apply to returns calculations for realistic evaluation
```

### 1.6 Data Scaling & Normalization
```
Strategy:
- Use StandardScaler (fit only on training data)
- Save scalers for inverse transformation
- For Prophet: use raw prices (it handles scaling internally)
```

**Deliverables:**
- [ ] EDA notebook completed
- [ ] Survivorship bias removed
- [ ] Features engineered & validated
- [ ] Train/val/test splits created (non-overlapping, temporally ordered)
- [ ] Scalers fit and saved
- [ ] Feature statistics documented (mean, std, missing %)

---

## PHASE 2: CLASSICAL MODELS (~4-5 hours)

### 2.1 ARIMA Model
```
Workflow:
1. Test stationarity (ADF test, KPSS)
2. Determine (p,d,q) using auto_arima (pmdarima)
3. Fit on training data
4. Generate 7-day & 30-day forecasts
5. Evaluate on test set

Output: Point forecasts, confidence intervals
```

### 2.2 SARIMA Model
```
Enhancements over ARIMA:
- Include seasonal components (P,D,Q,m)
- Use auto_arima with seasonal=True
- m=252 (trading days per year) or m=5 (weekly seasonality)
- Fit & forecast (same as ARIMA)

Output: Decomposed seasonality visualization
```

### 2.3 Facebook Prophet
```
Workflow:
1. Aggregate to daily level if needed
2. Add holiday effects (NYSE holidays)
3. Configure trend (linear or logistic)
4. Add yearly/weekly seasonality
5. Fit model with training data
6. Generate future dataframe for 7 & 30 day horizons
7. Include 80%/95% confidence intervals

Key Features:
- Automatic trend detection
- Holiday support
- Robust to outliers & missing data
```

### 2.4 Model Comparison
```
Store results:
- predictions_7day_classical.csv
- predictions_30day_classical.csv

Include:
- Actual prices
- ARIMA forecast
- SARIMA forecast
- Prophet forecast
- Confidence intervals
```

**Deliverables:**
- [ ] ARIMA notebook: hyperparameters, diagnostics, forecasts
- [ ] SARIMA notebook: seasonality analysis, diagnostics
- [ ] Prophet notebook: trend/seasonality components, forecast plots
- [ ] Comparative CSV with all forecasts
- [ ] Statistical test results (ADF, ACF plots)

---

## PHASE 3: DEEP LEARNING - LSTM (~5-6 hours)

### 3.1 Data Windowing
```
Create overlapping windows:
- lookback = 60 days (adjustable)
- Each sample: X[t-60:t] → y[t:t+horizon]
- For 7-day: y = mean/last price from [t:t+7]
- For 30-day: y = mean/last price from [t:t+30]

Validation: Ensure no temporal leakage
```

### 3.2 LSTM Architecture (Single-Step)
```
Model 1: Single-step forecasting
Input:  (batch, 60, n_features)
├── LSTM(64, return_sequences=True)
├── Dropout(0.2)
├── LSTM(32)
├── Dropout(0.2)
└── Dense(1)

Output: Single value for t+7 or t+30

Loss: MSE
Optimizer: Adam(lr=0.001)
```

### 3.3 LSTM Architecture (Multi-Step / Seq2Seq)
```
Model 2: Sequence-to-sequence (multi-step output)
Encoder:
├── LSTM(64, return_sequences=True) on X[t-60:t]
└── LSTM(32)

Decoder:
├── RepeatVector(7 or 30)
├── LSTM(32, return_sequences=True)
└── TimeDistributed(Dense(1))

Output: (batch, 7, 1) or (batch, 30, 1)
```

### 3.4 Training & Validation
```
- Batch size: 32
- Epochs: 100 (with early stopping)
- Validation split: 20% of training set
- Metrics: Track RMSE, MAE during training
- Save best model based on val_loss
```

### 3.5 Hyperparameter Tuning (Optional)
```
Grid/Random Search over:
- LSTM units: [32, 64, 128]
- Dropout rates: [0.1, 0.2, 0.3]
- Lookback window: [30, 60, 90]
- Learning rate: [0.001, 0.005, 0.01]
```

**Deliverables:**
- [ ] Windowed data pipeline created & tested
- [ ] Single-step LSTM trained & validated
- [ ] Multi-step LSTM trained & validated
- [ ] Training curves (loss, metrics) plotted
- [ ] Predictions saved: lstm_7day.csv, lstm_30day.csv
- [ ] Model weights saved for later evaluation

---

## PHASE 4: TRANSFORMER ENCODER (~5-6 hours)

### 4.1 Positional Encoding
```python
# Create positional embeddings for sequence position information
# Essential for Transformer as it has no recurrence
```

### 4.2 Transformer Encoder Architecture
```
Input: (batch, 60, n_features)
├── Linear embedding layer
├── Positional encoding
├── Transformer Encoder Block (stack of 2-3):
│   ├── Multi-Head Self-Attention (8 heads)
│   ├── Feed-forward network (hidden_dim=256)
│   ├── Layer normalization
│   └── Residual connections
├── Global average pooling
├── Dense(64) → ReLU
├── Dense(1)

Output: Single scalar (for 7-day) or (batch, 7, 1) for multi-step
```

### 4.3 Implementation Details
```
- Attention heads: 8
- Model dimension: 64
- FF hidden dimension: 256
- Dropout in attention: 0.1
- Loss: MSE
- Optimizer: Adam(lr=0.001)
```

### 4.4 Attention Visualization
```
Extract attention weights from encoder:
- Shape: (batch, heads, seq_len, seq_len)
- Average over batch & heads
- Plot heatmap showing which timesteps model attends to
- Visualize for both 7-day and 30-day forecasts
```

**Deliverables:**
- [ ] Transformer encoder implemented & tested
- [ ] Training curves saved
- [ ] Predictions: transformer_7day.csv, transformer_30day.csv
- [ ] Attention weight heatmaps generated
- [ ] Attention analysis report (which time periods most important?)

---

## PHASE 5: EVALUATION FRAMEWORK (~3-4 hours)

### 5.1 Metrics Implementation
```
For each model & horizon (7-day, 30-day):

1. MAPE (Mean Absolute Percentage Error)
   MAPE = 100 * mean(|y_true - y_pred| / |y_true|)
   
2. RMSE (Root Mean Squared Error)
   RMSE = sqrt(mean((y_true - y_pred)^2))
   
3. MAE (Mean Absolute Error)
   MAE = mean(|y_true - y_pred|)
   
4. Directional Accuracy
   DA = % of correct up/down predictions
   DA = sum(sign(y_true) == sign(y_pred)) / len(y_true)
   
5. Hit Rate (within ±2% of actual)
   HR = sum(|y_true - y_pred| <= 0.02 * y_true) / len(y_true)
```

### 5.2 Evaluation Results Table
```
Create comparison table:
                 MAPE(7d) | RMSE(7d) | DA(7d) | MAPE(30d) | RMSE(30d) | DA(30d)
ARIMA            ...
SARIMA           ...
Prophet          ...
LSTM (single)    ...
LSTM (multi)     ...
Transformer      ...

Identify: Best for each horizon & metric
```

### 5.3 Statistical Significance Testing
```
- Paired t-test between model predictions
- Diebold-Mariano test for forecast accuracy
- Confidence intervals for each metric
```

### 5.4 Backtesting (Optional)
```
Simulate trading based on directional predictions:
- Long/short based on predicted direction
- Account for transaction costs (slippage, commissions)
- Calculate cumulative returns vs. buy-and-hold
- Plot equity curves
```

**Deliverables:**
- [ ] metrics.py with all evaluation functions
- [ ] Comprehensive evaluation_results.csv
- [ ] Statistical significance test results
- [ ] Backtesting report (if implemented)
- [ ] Summary table with rankings

---

## PHASE 6: EXPLAINABILITY & VISUALIZATION (~4-5 hours)

### 6.1 LSTM Attention/Visualization
```
Option 1: Grad-CAM (Gradient-weighted Class Activation Mapping)
- Compute gradients of output w.r.t. LSTM hidden states
- Visualize which input timesteps had highest impact

Option 2: Layer-wise Relevance Propagation (LRP)
- Backpropagate predictions to input
- Show relevance scores for each input feature
```

### 6.2 Transformer Attention Heatmaps
```
For each prediction:
1. Extract attention weights from encoder
2. Average over heads & batch
3. Create heatmap: x-axis = input timesteps, y-axis = attention
4. Identify which past prices model "looks at"
5. Compare attention patterns for rising vs. falling markets
```

### 6.3 Prophet Component Analysis
```
prophet_plot_components():
- Trend: Long-term directional movement
- Seasonality: Yearly/weekly patterns
- Holiday effects: Named event impacts
- Residuals: Unexplained variance

Compare with LSTM learned patterns
```

### 6.4 Feature Importance (SHAP/Permutation)
```
SHAP analysis:
1. Calculate SHAP values for each feature
2. Feature importance bar plot
3. Dependence plots: how each feature affects predictions
4. Summary plot: impact of features on output

Alternative: Permutation importance
```

### 6.5 Comparative Visualizations
```
Plots to create:
1. Actual vs. Predicted (line plots) for each model
2. Residual distribution (histograms)
3. Residual autocorrelation (to check independence)
4. Q-Q plots for normality
5. Model comparison: side-by-side MAPE/RMSE bars
6. Attention heatmaps (Transformer)
7. Trend/seasonality decomposition (Prophet)
```

**Deliverables:**
- [ ] Attention heatmaps for Transformer (top 10 predictions)
- [ ] SHAP feature importance plots
- [ ] Prophet component decomposition plots
- [ ] LSTM/Transformer comparison dashboard
- [ ] Residual analysis plots
- [ ] explainability_report.md with insights

---

## PHASE 7: COMPARATIVE ANALYSIS & DOCUMENTATION (~3-4 hours)

### 7.1 Model Comparison Report
```
Create comprehensive report:
1. Executive Summary
   - Best model for 7-day, 30-day forecasts
   - Key findings vs. baselines
   
2. Methodology Section
   - Data sources & preprocessing
   - Model architectures
   - Evaluation metrics
   
3. Results & Analysis
   - Performance tables (MAPE, RMSE, DA)
   - Statistical significance tests
   - Computational efficiency (training time, inference)
   
4. Explainability Insights
   - What do models learn?
   - Which features matter most?
   - Attention patterns in Transformer
   - Trends in Prophet
   
5. Limitations & Future Work
   - Assumptions made
   - Data constraints
   - Recommendations (ensemble models, retraining frequency)
```

### 7.2 Recommendation Framework
```
Decision tree for practitioners:
- If seeking interpretability → Prophet
- If seeking accuracy → Ensemble (best classical + deep)
- If seeking explainability → Transformer (attention) + SHAP
- For production: Consider ensemble with voting/stacking
```

### 7.3 Code Documentation
```
- Docstrings for all functions
- Type hints for clarity
- Configuration file documentation
- README with running instructions
```

### 7.4 Final Deliverables Package
```
results/
├── reports/
│   ├── final_report.pdf
│   ├── executive_summary.md
│   └── technical_appendix.md
├── predictions/
│   ├── all_predictions.csv (merged forecasts)
│   └── model_comparison_metrics.csv
├── visualizations/
│   ├── actual_vs_predicted/
│   ├── attention_heatmaps/
│   ├── feature_importance/
│   └── residual_analysis/
└── code_artifacts/
    ├── trained_models/ (saved weights)
    └── config_final.yaml
```

**Deliverables:**
- [ ] final_report.md with all findings
- [ ] Comparative performance table
- [ ] Recommendation framework documented
- [ ] All visualizations organized in results/
- [ ] README.md with usage instructions
- [ ] requirements.txt finalized

---

## EXECUTION TIMELINE & MILESTONES

| Phase | Duration | Start | End | Milestone |
|-------|----------|-------|-----|-----------|
| 0 | 1-2h | Day 1 | Day 1 | Environment ready |
| 1 | 3-4h | Day 1 | Day 2 | Clean data splits ready |
| 2 | 4-5h | Day 2 | Day 3 | Classical models trained |
| 3 | 5-6h | Day 3 | Day 4 | LSTM models trained |
| 4 | 5-6h | Day 4 | Day 5 | Transformer trained + attention viz |
| 5 | 3-4h | Day 5 | Day 6 | Evaluation metrics computed |
| 6 | 4-5h | Day 6 | Day 7 | Explainability analysis done |
| 7 | 3-4h | Day 7 | Day 7 | Final report & documentation |
| **Total** | **28-36 hours** | | | **Complete project** |

---

## CRITICAL SUCCESS FACTORS

1. **No Data Leakage:** Ensure test set truly unseen during model development
2. **Proper Scaling:** Fit scalers ONLY on training data
3. **Walk-Forward Validation:** Don't use future data for predictions
4. **Document Assumptions:** Transaction costs, slippage, holiday calendars
5. **Version Control:** Track model iterations, hyperparameters
6. **Reproducibility:** Set random seeds, save configs, document data sources

---

## CONTINGENCY NOTES

- If deep learning models underperform: Try simpler architectures, longer lookback windows, or ensemble with classical models
- If Transformer overfits: Increase dropout, reduce heads, use more regularization
- If Prophet doesn't capture dynamics: Tweak seasonality periods, add more holidays, use local linear trend
- If evaluation shows poor directional accuracy: Focus on volatility forecasting instead, or use different loss functions

---

## NEXT STEPS

1. **Start Phase 0 immediately:** Set up directory structure and dependencies
2. **Data loading:** Identify stock data source (Yahoo Finance, Quandl, etc.)
3. **Iterate:** Complete each phase with checkpoints before moving to next

Good luck!
