# VISUAL PROJECT ROADMAP
**Multi-Horizon Stock Forecasting: Complete Execution Plan**

---

## 🎯 PROJECT GOAL
Predict 7-day and 30-day future stock prices using **6 models** across 3 approaches:

```
                        ┌─────────────────────────────────────────────┐
                        │  STOCK PRICE PREDICTION (7-day & 30-day)    │
                        └──────────────────┬──────────────────────────┘
                                          │
                ┌─────────────────────────┼─────────────────────────┐
                │                         │                         │
        ┌───────▼────────┐      ┌────────▼────────┐      ┌────────▼────────┐
        │  CLASSICAL     │      │   DEEP LEARNING │      │   STATE-OF-ART │
        │                │      │                 │      │                 │
        │ • ARIMA        │      │ • LSTM (single) │      │ • Transformer   │
        │ • SARIMA       │      │ • LSTM (multi)  │      │   (+ Attention) │
        │ • Prophet      │      │                 │      │                 │
        │   (trend,      │      │                 │      │  → Explainable! │
        │    seasonality)│      │                 │      │  → Interpretable│
        └────────────────┘      └─────────────────┘      └─────────────────┘
```

---

## 📅 7-PHASE TIMELINE (28-36 Hours)

```
PHASE 0: SETUP (1-2h)
├─ Virtual environment
├─ Install dependencies
├─ Customize config.yaml
└─ ✓ Ready to code

PHASE 1: DATA PREP (3-4h)
├─ Load stock data
├─ Remove survivorship bias
├─ Engineer features (RSI, moving avg, vol)
├─ Create targets (forward returns)
├─ Train/Val/Test split (no leakage)
└─ ✓ Clean dataset ready

PHASE 2: CLASSICAL (4-5h)
├─ Auto ARIMA (p,d,q optimization)
├─ SARIMA + seasonality
├─ Prophet + holidays
└─ ✓ 3 baseline models trained

PHASE 3: LSTM (5-6h)
├─ Single-step LSTM (7d & 30d)
├─ Multi-step Seq2Seq LSTM
├─ Training & validation
└─ ✓ 2 deep learning models trained

PHASE 4: TRANSFORMER (5-6h)
├─ Positional encoding
├─ Multi-head attention encoder
├─ Residual connections
├─ Attention visualization
└─ ✓ 1 state-of-art model + explainability

PHASE 5: EVALUATION (3-4h)
├─ Calculate metrics (MAPE, RMSE, DA)
├─ Model comparison table
├─ Statistical significance testing
└─ ✓ Ranked all 6 models

PHASE 6: EXPLAINABILITY (4-5h)
├─ Transformer attention heatmaps
├─ SHAP feature importance
├─ Prophet components
├─ Residual analysis
└─ ✓ Understand what models learn

PHASE 7: FINAL REPORT (3-4h)
├─ Executive summary
├─ Methodology documentation
├─ Results tables & insights
├─ Recommendations
└─ ✓ Complete project package
```

---

## 📊 DATA FLOW ARCHITECTURE

```
Raw Stock Data (Yahoo Finance)
         │
         ▼
  ┌──────────────────┐
  │ SURVIVORSHIP     │  Remove delisted/merged companies
  │ BIAS CHECK       │  Verify continuous data 2018-2025
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │ FEATURE ENGINE   │  • Log returns
  │                  │  • Moving averages (5,20,50d)
  │                  │  • Volatility (20d rolling std)
  │                  │  • RSI (14), Volume ratios
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │ CREATE TARGETS   │  • target_7d = log(P_t+7 / P_t)
  │                  │  • target_30d = log(P_t+30 / P_t)
  │  (Forward)       │  [NO LOOK-AHEAD BIAS!]
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │ TRAIN/VAL/TEST   │  Train: 2018-12-31 (70%)
  │ SPLIT            │  Val:   2024-09-30 (15%)
  │ (Temporal Order) │  Test:  2025-12-31 (15%)
  └────────┬─────────┘
           │
           ├─────────────────┬─────────────────┬─────────────────┐
           ▼                 ▼                 ▼                 ▼
        TRAIN          VALIDATION           TEST            METRICS
        Data           Data                Data
        (Fit scalers)  (Monitoring)        (Evaluation)
```

---

## 🔄 MODEL ARCHITECTURE AT A GLANCE

### **Classical Models**
```
ARIMA (7-day)          SARIMA (30-day)        Prophet (both)
└─ Auto ARIMA          └─ Seasonal ARIMA       └─ Trend
   (p,d,q) fit)           (P,D,Q,m) fit)          Seasonality
   Forecast (1-7)         Forecast (1-30)        Holidays
                                                 Confidence int.
```

### **Deep Learning: LSTM**
```
Single-Step (7-day)              Multi-Step Seq2Seq (30-day)
Input: (60, 5) features          Input: (60, 5) features
  │                               │
  ├─ LSTM(64)                    ├─ ENCODER:
  │  └─ Dropout(0.2)             │   LSTM(64)
  │                               │   └─ Dropout(0.2)
  ├─ LSTM(32)                    │
  │  └─ Dropout(0.2)             ├─ DECODER:
  │                               │   RepeatVector(30)
  └─ Dense(1)                    │   LSTM(32)
                                  │   TimeDistributed(Dense)
     Output: (1) single value    └─ Output: (30) sequence
```

### **State-of-Art: Transformer**
```
Input: Price time series (60 timesteps × 5 features)
       │
       ├─ Embed + Positional Encoding
       │
       └─ Encoder Block (×2-3):
           ├─ Multi-Head Attention (8 heads)
           │
           ├─ Feed-Forward (256 hidden)
           │
           └─ Residual + LayerNorm
           
Output: Prediction (7-day or 30-day price)

BONUS: Can extract & visualize attention weights!
       → Shows which past timesteps model focuses on
```

---

## 📈 EVALUATION METRICS SUMMARY

For **each model** × **both horizons** (7-day & 30-day):

```
Metric                  Formula                      Target       Interpretation
─────────────────────────────────────────────────────────────────────────────────
MAPE                    mean(|y-ŷ|/|y|) × 100        < 5%        % prediction error
RMSE                    √(mean((y-ŷ)²))               < 2.0       Magnitude of errors
MAE                     mean(|y-ŷ|)                  < 1.5       Avg absolute error
Directional Accuracy    sum(sign(y)==sign(ŷ))/n ×100 > 52%       Correct up/down
Hit Rate (±2%)          sum(|y-ŷ| ≤ 0.02y)/n ×100   > 40%       Close predictions

Best Model: Ranked by MAPE (primary) → RMSE (tiebreaker)
```

---

## 🔍 EXPLAINABILITY FRAMEWORK

### **What We'll Understand:**

```
1. TRANSFORMER ATTENTION
   ├─ Which timesteps matter most?
   ├─ Do patterns differ in bull vs bear markets?
   └─ Attention heatmap visualization

2. SHAP FEATURE IMPORTANCE
   ├─ Ranking: returns > volatility > moving averages?
   ├─ How does each feature affect predictions?
   └─ Dependence plots

3. PROPHET COMPONENTS
   ├─ Trend: Linear? Accelerating? Regime changes?
   ├─ Seasonality: Yearly patterns? Weekly effects?
   └─ Holiday impact: Do markets react to NYSE holidays?

4. RESIDUAL ANALYSIS
   ├─ Are errors normally distributed?
   ├─ Any remaining patterns (autocorrelation)?
   └─ Identify failure modes for each model

5. COMPARATIVE INSIGHTS
   ├─ Classical: Good at baseline trends
   ├─ LSTM: Captures non-linear dynamics
   └─ Transformer: Most explainable + accurate?
```

---

## 📁 FINAL DELIVERABLES STRUCTURE

```
results/
├── predictions/
│   ├── arima_7day_forecasts.csv
│   ├── sarima_7day_forecasts.csv
│   ├── prophet_7day_forecasts.csv
│   ├── lstm_7day_forecasts.csv
│   ├── transformer_7day_forecasts.csv
│   └── [duplicate for 30-day]
│
├── metrics/
│   ├── comparison_table_7day.csv   ← MAPE, RMSE, DA
│   ├── comparison_table_30day.csv
│   ├── statistical_tests.txt       ← p-values, significance
│   └── ranking_best_models.csv
│
├── visualizations/
│   ├── actual_vs_predicted/        ← 6 line plots
│   ├── attention_heatmaps/         ← Transformer focus
│   ├── feature_importance/         ← SHAP rankings
│   ├── prophet_components/         ← Trend, seasonality
│   ├── residual_analysis/          ← Diagnostics
│   └── model_comparison/           ← Performance bars
│
└── reports/
    ├── FINAL_REPORT.pdf            ← 10-15 pages
    ├── executive_summary.txt
    ├── methodology.txt
    ├── findings_insights.txt
    ├── recommendations.txt
    └── limitations_future_work.txt
```

---

## ⚠️ CRITICAL SUCCESS FACTORS

```
✓ DATA LEAKAGE PREVENTION
  └─ Scalers fit ONLY on training data
  └─ Test set never touches training
  └─ Each prediction uses ONLY past data (60-day lookback)

✓ SURVIVORSHIP BIAS REMOVAL
  └─ Remove stocks delisted during 2018-2025
  └─ Use only tickers with full date coverage
  └─ Document exclusions

✓ REALISTIC MARKET CONDITIONS
  └─ Bid-ask spread: 0.1%
  └─ Slippage: 0.05%
  └─ Commission: 0.05%
  └─ Applied to return calculations

✓ TEMPORAL ORDERING (TIME SERIES!)
  └─ Train: 2018-2023
  └─ Val: 2024 Q1-Q3
  └─ Test: 2024 Q4 - 2025 Q4
  └─ NO overlapping dates
```

---

## 🚀 HOW TO GET STARTED RIGHT NOW

```
STEP 1: Setup
$ cd "/Users/aishu/atsa/ATSA PROJECT"
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt

STEP 2: Verify
$ python -c "import pandas, numpy, tensorflow; print('✓')"

STEP 3: Customize
$ vim config.yaml  # Edit your tickers and dates

STEP 4: Start Phase 1
$ jupyter notebook
→ Create notebooks/01_eda.ipynb
→ Follow EXECUTION_PLAN.md step-by-step
```

---

## 📊 EXPECTED OUTCOMES

After completing all 7 phases, you will have:

```
Models Trained              6
├─ ARIMA                   1
├─ SARIMA                  1
├─ Prophet                 1
├─ LSTM (single)           1
├─ LSTM (multi)            1
└─ Transformer             1

Forecasts Generated       12 (2 horizons × 6 models)

Metrics Calculated        8 per model (MAPE, RMSE, ...)

Visualizations Created    20+
├─ Attention heatmaps     10
├─ Feature importance      3
├─ Prophet components      3
├─ Residual plots          4
└─ Comparisons             5

Reports Generated          5
├─ Executive summary       1
├─ Final report            1
├─ Technical appendix      1
└─ Limitations & outlook   2

Lines of Code             1500+

Documented Insights        50+
```

---

## 🎓 SKILLS YOU'LL GAIN

```
Data Science
├─ Time series preprocessing
├─ Feature engineering
├─ Survivorship bias removal
└─ Scaling/normalization

Classical Forecasting
├─ ARIMA: Trend + differencing
├─ SARIMA: Seasonal patterns
└─ Prophet: Trend + seasonality + holidays

Deep Learning
├─ LSTM architectures
├─ Sequence-to-sequence models
└─ Transformers & attention

Model Evaluation
├─ Multiple metrics (MAPE, RMSE, DA)
├─ Statistical significance testing
└─ Paired comparisons

Explainability
├─ Attention visualization
├─ SHAP feature importance
└─ Residual diagnostics
```

---

## 📚 FILES YOU HAVE

```
✓ EXECUTION_PLAN.md          ← Ultimate guide (2000+ lines)
✓ README.md                  ← Quick reference (600+ lines)
✓ config.yaml               ← All settings (300+ lines)
✓ SETUP_SUMMARY.md          ← Current status & next steps
✓ PROGRESS_CHECKLIST.md     ← Track your work ← YOU ARE HERE
✓ VISUAL_ROADMAP.md         ← This file
✓ requirements.txt          ← Dependencies
✓ src/utils.py             ← Data utilities (400 lines)
✓ src/evaluation.py        ← Metrics framework (400 lines)
✓ src/data_prep.py         ← Full pipeline (400 lines)
```

Still to create:
```
△ notebooks/01_eda.ipynb            ← Start here!
△ notebooks/02_data_prep.ipynb
△ notebooks/03_classical_models.ipynb
△ notebooks/04_lstm_models.ipynb
△ notebooks/05_transformer.ipynb
△ notebooks/06_evaluation_viz.ipynb
```

---

## ⏱️ TIME BREAKDOWN

```
Phase 0: Setup & Environment          1-2 hours
Phase 1: Data Preparation             3-4 hours  ← Most important!
Phase 2: Classical Models             4-5 hours
Phase 3: LSTM                         5-6 hours
Phase 4: Transformer                  5-6 hours
Phase 5: Evaluation                   3-4 hours
Phase 6: Explainability               4-5 hours
Phase 7: Final Report                 3-4 hours
─────────────────────────────────────────────────
TOTAL                                28-36 hours
```

**Recommended pace:** 4 hours/day × 7 days = Complete in a week

---

## 🎯 SUCCESS CONDITION

Your project is complete when:

```
✓ All 6 models trained and saved
✓ Forecasts generated for 7-day and 30-day horizons
✓ MAPE < 5% achieved (or limitations documented)
✓ Directional accuracy > 52% (better than coin flip)
✓ Attention weights visualized
✓ Feature importance ranked
✓ Statistical significance tests completed
✓ Final report documenting all findings
✓ All code documented and reproducible
✓ No data leakage confirmed
✓ Survivorship bias removed
```

---

**YOU ARE READY. START PHASE 0 NOW!** 🚀

Reference: `EXECUTION_PLAN.md` | `PROGRESS_CHECKLIST.md` | `README.md`
