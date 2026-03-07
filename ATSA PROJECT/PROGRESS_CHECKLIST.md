# PROJECT PROGRESS CHECKLIST
**Track your progress through all 7 phases**

---

## PHASE 0: PROJECT SETUP & ENVIRONMENT (1-2 hours)
**Status: IN PROGRESS / NOT STARTED / COMPLETED**

- [ ] Virtual environment created: `python3 -m venv venv`
- [ ] Environment activated: `source venv/bin/activate`
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Verify imports: `python -c "import pandas, numpy, tensorflow; print('✓')"`
- [ ] config.yaml customized with your tickers
- [ ] Directory structure created (auto-done)
- [ ] requirements.txt finalized
- [ ] README.md reviewed

**Phase 0 Sign-Off:**
- [ ] Environment is ready for code execution
- [ ] All imports work without errors
- [ ] config.yaml is customized for your use case
- [ ] Can run: `python src/utils.py` without errors

**Estimated Time: 1-2 hours**  
**Date Started: ___________**  
**Date Completed: ___________**

---

## PHASE 1: DATA PREPARATION (3-4 hours)
**Status: NOT STARTED / IN PROGRESS / COMPLETED**

### 1.1 Load & EDA
- [ ] Create `notebooks/01_eda.ipynb`
- [ ] Load sample stock data (at least 3 tickers)
- [ ] Plot price series (all tickers)
- [ ] Plot returns distribution (normal, log-normal check)
- [ ] Calculate rolling volatility
- [ ] Check for gaps and missing data
- [ ] Document data issues and solutions

**Deliverable:** `01_eda.ipynb` with visualizations

### 1.2 Remove Survivorship Bias
- [ ] List all tickers in date range [2018-2025]
- [ ] Verify each ticker has full date coverage
- [ ] Remove tickers with gaps > 5%
- [ ] Document removed tickers and reasons
- [ ] Save "valid_tickers.txt"

**Deliverable:** List of 10-50 valid tickers

### 1.3 Feature Engineering
- [ ] Calculate log returns
- [ ] Create rolling averages (5, 20, 50 days)
- [ ] Create rolling volatility (20-day std)
- [ ] Calculate RSI (14-day)
- [ ] Create volume ratios
- [ ] Handle NaN values properly (forward/backward fill)

**Deliverable:** Feature engineering notebook complete

### 1.4 Train-Test Split (NO LOOK-AHEAD BIAS)
- [ ] Split dates: Train [2018-2023], Val [2024-09 30], Test [2024-10 to 2025-12]
- [ ] Verify NO temporal overlap
- [ ] Verify ALL training data before 2024
- [ ] Verify ALL test data after 2024-09-30
- [ ] Check train/val/test sizes (approximate 70/15/15)

**Deliverable:** Three separate CSV files saved

### 1.5 Scaling & Normalization
- [ ] Fit StandardScaler on TRAINING DATA ONLY
- [ ] Store scaler parameters for later inverse transform
- [ ] Scale validation data with training scaler
- [ ] Scale test data with training scaler
- [ ] Verify no information leakage

**Deliverable:** Scaled datasets, scaler parameters saved

### 1.6 Create Targets (Forward Returns)
- [ ] For 7-day horizon: y_7d = log(P_t+7 / P_t)
- [ ] For 30-day horizon: y_30d = log(P_t+30 / P_t)
- [ ] Verify no look-ahead (targets use FUTURE prices only)
- [ ] Remove last 30 rows (no future data for predictions)

**Deliverable:** `02_data_prep.ipynb` complete with targets

**Phase 1 Sign-Off:**
- [ ] All data preparation code written & tested
- [ ] Train/val/test splits saved to CSV
- [ ] Features and targets created
- [ ] No data leakage detected
- [ ] Survivorship bias removed

**Estimated Time: 3-4 hours**  
**Date Started: ___________**  
**Date Completed: ___________**

---

## PHASE 2: CLASSICAL MODELS (4-5 hours)
**Status: NOT STARTED / IN PROGRESS / COMPLETED**

### 2.1 ARIMA Model
- [ ] Import: `from pmdarima import auto_arima`
- [ ] Test stationarity with ADF test
- [ ] Use `auto_arima()` to find optimal (p,d,q)
- [ ] Fit ARIMA on training data
- [ ] Generate 7-day forecast (on test set)
- [ ] Generate 30-day forecast (on test set)
- [ ] Plot actual vs predicted
- [ ] Save predictions to CSV

**Deliverable:** ARIMA forecasts for both horizons

### 2.2 SARIMA Model
- [ ] Extend ARIMA with seasonal components
- [ ] Use `auto_arima(seasonal=True)`
- [ ] Set seasonal period m=252 (trading days) or m=5 (weekly)
- [ ] Fit SARIMA on training data
- [ ] Generate forecasts for 7-day & 30-day
- [ ] Plot seasonality decomposition
- [ ] Compare SARIMA vs ARIMA

**Deliverable:** SARIMA forecasts, decomposition plots

### 2.3 Prophet Model
- [ ] Import: `from prophet import Prophet`
- [ ] Create Prophet DataFrame (columns: ds, y)
- [ ] Add US holidays to Prophet
- [ ] Configure trend (linear or logistic)
- [ ] Add yearly and weekly seasonality
- [ ] Fit on training data
- [ ] Create future dataframe for 7 & 30 day horizons
- [ ] Generate forecasts with 80% confidence intervals
- [ ] Plot trend and seasonal components
- [ ] Analyze residuals

**Deliverable:** Prophet forecasts, component plots

### 2.4 Model Comparison
- [ ] Create comparison CSV: all_classical_predictions.csv
- [ ] Columns: date, actual, arima_pred, sarima_pred, prophet_pred
- [ ] Calculate MAPE for each model
- [ ] Calculate RMSE for each model
- [ ] Create comparison plot (actual vs 3 models)

**Deliverable:** `03_classical_models.ipynb` complete

**Phase 2 Sign-Off:**
- [ ] All 3 classical models implemented & trained
- [ ] Forecasts generated for both horizons
- [ ] Comparison table created
- [ ] Plots saved to results/visualizations/
- [ ] CSV predictions saved to results/predictions/

**Estimated Time: 4-5 hours**  
**Date Started: ___________**  
**Date Completed: ___________**

---

## PHASE 3: DEEP LEARNING - LSTM (5-6 hours)
**Status: NOT STARTED / IN PROGRESS / COMPLETED**

### 3.1 Data Windowing
- [ ] Create sliding windows of lookback=60
- [ ] Each X: (batch, 60, n_features)
- [ ] Each y: single value for 7-day horizon
- [ ] Verify no data leakage
- [ ] Create separate windows for 7-day and 30-day
- [ ] Apply to train/val/test sets

**Deliverable:** Windowed datasets saved

### 3.2 LSTM Single-Step Model
- [ ] Import TensorFlow/Keras
- [ ] Build architecture:
  - Input: (batch, 60, n_features)
  - LSTM(64) + Dropout(0.2)
  - LSTM(32) + Dropout(0.2)
  - Dense(1) output
- [ ] Compile: optimizer='adam', loss='mse'
- [ ] Fit on training data with early stopping
- [ ] Monitor val_loss for convergence
- [ ] Save model weights

**Deliverable:** Trained LSTM model saved

### 3.3 LSTM Multi-Step Model (Seq2Seq)
- [ ] Encoder: LSTM(64) on input sequence
- [ ] Decoder: RepeatVector + LSTM(32)
- [ ] Output: (batch, 30, 1) for 30-step sequence
- [ ] Fit on training data
- [ ] Generate multi-step forecasts
- [ ] Compare single vs multi-step

**Deliverable:** Multi-step LSTM model & predictions

### 3.4 Training & Validation
- [ ] Plot training curves (loss vs epoch)
- [ ] Check for overfitting (train vs val loss divergence)
- [ ] Save models: lstm_single_7day.h5, lstm_multi_30day.h5
- [ ] Verify early stopping triggers properly

**Deliverable:** Training history plots, model comparison

### 3.5 Generate Predictions
- [ ] Load trained models
- [ ] Predict on test set
- [ ] Inverse scale predictions (back to price space)
- [ ] Save forecasts: lstm_7day_preds.csv, lstm_30day_preds.csv

**Deliverable:** `04_lstm_models.ipynb` complete

**Phase 3 Sign-Off:**
- [ ] Single-step LSTM trained & saved
- [ ] Multi-step LSTM trained & saved
- [ ] Training curves show convergence
- [ ] Predictions generated for both horizons
- [ ] LSTM predictions saved to CSV

**Estimated Time: 5-6 hours**  
**Date Started: ___________**  
**Date Completed: ___________**

---

## PHASE 4: TRANSFORMER ENCODER (5-6 hours)
**Status: NOT STARTED / IN PROGRESS / COMPLETED**

### 4.1 Positional Encoding
- [ ] Implement sinusoidal positional encoding
- [ ] Create position embeddings: max_len=512, d_model=64
- [ ] Verify positions scale correctly

**Deliverable:** Positional encoding function

### 4.2 Transformer Encoder Architecture
- [ ] Input embedding: d_model=64
- [ ] Add positional encoding
- [ ] Multi-head attention: 8 heads
- [ ] Feed-forward network: hidden_dim=256
- [ ] Layer normalization & residual connections
- [ ] Stack: 2-3 encoder layers
- [ ] Global average pooling
- [ ] Output dense layer for regression

**Deliverable:** Transformer encoder class

### 4.3 Training
- [ ] Fit on X_train (60 timesteps, n_features)
- [ ] Predict y_train_7day (single value)
- [ ] Use early stopping on validation loss
- [ ] Plot training curves
- [ ] Save model weights

**Deliverable:** Trained Transformer model

### 4.4 Attention Visualization
- [ ] Extract attention weights from encoder layers
- [ ] Shape: (batch_size, num_heads, seq_len, seq_len)
- [ ] Average over batch and heads
- [ ] Create heatmap: x=input timesteps, y=attention weights
- [ ] Plot top 10 predictions with their attention patterns
- [ ] Save heatmaps as PNG/PDF

**Deliverable:** Attention heatmaps for key predictions

### 4.5 Predictions
- [ ] Predict on test set
- [ ] Inverse scale back to price space
- [ ] Save forecasts: transformer_7day_preds.csv

**Deliverable:** `05_transformer.ipynb` complete

**Phase 4 Sign-Off:**
- [ ] Transformer encoder implemented & trained
- [ ] Attention weights extracted & visualized
- [ ] Heatmaps saved to results/visualizations/
- [ ] Predictions generated for test set
- [ ] Model analysis complete (which timesteps matter most?)

**Estimated Time: 5-6 hours**  
**Date Started: ___________**  
**Date Completed: ___________**

---

## PHASE 5: EVALUATION FRAMEWORK (3-4 hours)
**Status: NOT STARTED / IN PROGRESS / COMPLETED**

### 5.1 Load All Predictions
- [ ] Load classical model predictions (ARIMA, SARIMA, Prophet)
- [ ] Load LSTM predictions (single & multi-step)
- [ ] Load Transformer predictions
- [ ] Merge into single DataFrame
- [ ] Verify alignments (same dates, same actuals)

**Deliverable:** all_predictions_merged.csv

### 5.2 Calculate All Metrics
- [ ] For 7-day horizon:
  - [ ] MAPE for each model
  - [ ] RMSE for each model
  - [ ] MAE for each model
  - [ ] Directional accuracy (% correct up/down)
  - [ ] Hit rate (% within ±2%)
- [ ] Repeat for 30-day horizon

**Deliverable:** metrics_comparison.csv with all values

### 5.3 Create Comparison Table
- [ ] Rank models by MAPE (ascending = best)
- [ ] Show RMSE, MAE, DA, HR for each
- [ ] Highlight best model for each metric
- [ ] Create separate tables for 7-day and 30-day

**Deliverable:** comparison_table_7day.csv, comparison_table_30day.csv

### 5.4 Statistical Significance Testing
- [ ] Perform Diebold-Mariano test: Transformer vs LSTM
- [ ] Calculate p-values
- [ ] Document significant differences (p < 0.05)
- [ ] Paired t-test for additional validation

**Deliverable:** Statistical test results document

### 5.5 Visualization
- [ ] Actual vs predicted line plots (6 plots, one per model)
- [ ] Model comparison bar chart (MAPE, RMSE)
- [ ] Ranking table visualization

**Deliverable:** Comparison plots saved

**Phase 5 Sign-Off:**
- [ ] All metrics calculated for 6 models × 2 horizons
- [ ] Comparison table with ranking
- [ ] Statistical significance tests completed
- [ ] Visualizations saved

**Estimated Time: 3-4 hours**  
**Date Started: ___________**  
**Date Completed: ___________**

---

## PHASE 6: EXPLAINABILITY & VISUALIZATION (4-5 hours)
**Status: NOT STARTED / IN PROGRESS / COMPLETED**

### 6.1 Transformer Attention Analysis
- [ ] Plot attention heatmaps for top 10 predictions
- [ ] Analyze: Which past timesteps does model attend to?
- [ ] Compare attention patterns: Bull vs bear markets
- [ ] Document insights

**Deliverable:** Attention analysis report + plots

### 6.2 SHAP Feature Importance
- [ ] Calculate SHAP values for LSTM on test set
- [ ] Feature importance ranking
- [ ] Dependence plots: how each feature affects output
- [ ] Summary plot

**Deliverable:** SHAP plots + feature ranking

### 6.3 Prophet Component Analysis
- [ ] Plot trend component (long-term movement)
- [ ] Plot yearly seasonality (annual patterns)
- [ ] Plot weekly seasonality (intra-week patterns)
- [ ] Compare to LSTM learned patterns

**Deliverable:** Component decomposition plots

### 6.4 Residual Analysis
- [ ] Calculate residuals for best model
- [ ] Plot residual distribution (histogram)
- [ ] ACF/PACF plots (test independence)
- [ ] Q-Q plot (test normality)
- [ ] Plot residuals over time

**Deliverable:** Residual diagnostics plots

### 6.5 Model Behavior Comparison
- [ ] Plot actual vs predicted for all models (6 subplots)
- [ ] Highlight prediction errors
- [ ] Identify when each model fails
- [ ] Document failure modes

**Deliverable:** Comprehensive visualization dashboard

**Phase 6 Sign-Off:**
- [ ] Attention heatmaps explained
- [ ] Feature importance ranked
- [ ] Prophet components visualized
- [ ] Residuals analyzed
- [ ] Model comparison dashboard created

**Estimated Time: 4-5 hours**  
**Date Started: ___________**  
**Date Completed: ___________**

---

## PHASE 7: FINAL REPORT & DOCUMENTATION (3-4 hours)
**Status: NOT STARTED / IN PROGRESS / COMPLETED**

### 7.1 Executive Summary
- [ ] Write 1-page summary
- [ ] Highlight best performing model(s)
- [ ] Key findings vs classical vs deep learning
- [ ] Actionable recommendations

**Deliverable:** Executive summary

### 7.2 Methodology Section
- [ ] Document data sources (Yahoo Finance dates)
- [ ] Data preprocessing steps
- [ ] Model architectures (concise descriptions)
- [ ] Evaluation methodology

**Deliverable:** Methodology documentation

### 7.3 Results Section
- [ ] Comparison table: all models × both horizons
- [ ] Statistical significance findings
- [ ] Performance ranking
- [ ] Computational costs (training time, inference speed)

**Deliverable:** Results tables & analysis

### 7.4 Explainability Section
- [ ] Transformer attention insights
- [ ] Feature importance rankings
- [ ] Prophet seasonality findings
- [ ] Comparison: What do models learn differently?

**Deliverable:** Explainability analysis

### 7.5 Limitations & Future Work
- [ ] Data assumptions (survivorship bias, gaps)
- [ ] Model constraints (lookback window, forecast horizon)
- [ ] Suggestions for improvement:
  - Ensemble methods
  - Different architectures
  - Retraining frequency
  - Multi-step training

**Deliverable:** Limitations document

### 7.6 Code & Configure Organization
- [ ] Ensure all code is documented
- [ ] Type hints on all functions
- [ ] config.yaml fully annotated
- [ ] README, SETUP_SUMMARY complete
- [ ] All notebooks have markdown explanations

**Deliverable:** Complete, documented codebase

### 7.7 Final Package Creation
- [ ] All results in results/ directory
- [ ] predictions/ contains all forecasts
- [ ] metrics/ contains all evaluation tables
- [ ] visualizations/ contains all plots
- [ ] reports/ contains final_report.pdf

**Deliverable:** `06_evaluation_viz.ipynb` complete

**Phase 7 Sign-Off:**
- [ ] Final report written (5-10 pages)
- [ ] All findings documented
- [ ] Recommendations provided
- [ ] Complete results package ready

**Estimated Time: 3-4 hours**  
**Date Started: ___________**  
**Date Completed: ___________**

---

## 📊 OVERALL PROJECT SUMMARY

| Phase | Status | Duration | Start | End |
|-------|--------|----------|-------|-----|
| 0: Setup | [ ] | 1-2h | _____ | _____ |
| 1: Data | [ ] | 3-4h | _____ | _____ |
| 2: Classical | [ ] | 4-5h | _____ | _____ |
| 3: LSTM | [ ] | 5-6h | _____ | _____ |
| 4: Transformer | [ ] | 5-6h | _____ | _____ |
| 5: Evaluation | [ ] | 3-4h | _____ | _____ |
| 6: Explainability | [ ] | 4-5h | _____ | _____ |
| 7: Report | [ ] | 3-4h | _____ | _____ |
| **TOTAL** | | **28-36h** | _____ | _____ |

**Project Start Date:** ___________  
**Project Completion Date:** ___________  
**Total Days Elapsed:** ___________

---

## 🎉 FINAL COMPLETION CHECKLIST

When all phases are complete, verify:

- [ ] 6 trained models saved (ARIMA, SARIMA, Prophet, LSTM single, LSTM multi, Transformer)
- [ ] Forecasts for 7-day and 30-day horizons generated
- [ ] MAPE < 5% achieved on test set (or documented why not)
- [ ] Directional accuracy > 52% (or documented limitations)
- [ ] All evaluation metrics computed
- [ ] Statistical significance tests completed
- [ ] Transformer attention visualizations created
- [ ] Feature importance analysis done
- [ ] Prophet components analyzed
- [ ] Residual diagnostics plotted
- [ ] Final report written (5-10 pages)
- [ ] All notebooks runnable end-to-end
- [ ] All code documented & type-hinted
- [ ] All visualizations saved to results/
- [ ] All predictions saved to CSV
- [ ] No data leakage confirmed
- [ ] Survivorship bias removed
- [ ] Transaction costs accounted for

**READY FOR PRESENTATION: [ ]**

---

## 📝 NOTES & OBSERVATIONS

As you progress, document:

### Phase-by-phase insights:
- Data quality observations:
- Model performance surprises:
- Attention patterns discovered:
- Feature importance findings:

---

**Good luck!** Follow this checklist phase by phase, marking completion as you go.
