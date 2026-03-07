# Stock Forecasting with LSTM, Transformers & Prophet

**Project Goal:** Predict 7-day and 30-day future stock prices using classical time series models (ARIMA, SARIMA, Prophet) and deep learning approaches (LSTM, Transformer encoder).

## Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Project
Edit `config.yaml` with your settings:
- Stock tickers and date ranges
- Train/val/test split dates
- Model hyperparameters
- Paths for data and results

### 3. Run Phase by Phase
Follow the execution plan in `EXECUTION_PLAN.md`:

#### Phase 0: Setup
- Directory structure created ✓
- Dependencies installed ✓
- Config file ready ✓

#### Phase 1: Data Preparation
```bash
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_data_prep.ipynb
```
**Key outputs:**
- `data/processed/` - cleaned data
- `data/splits/` - train/val/test sets
- EDA plots and statistics

#### Phase 2: Classical Models
```bash
jupyter notebook notebooks/03_classical_models.ipynb
```
**Models:** ARIMA, SARIMA, Prophet
**Outputs:** `results/predictions/classical_*`

#### Phase 3: LSTM
```bash
jupyter notebook notebooks/04_lstm_models.ipynb
```
**Architectures:** Single-step, Multi-step (Seq2Seq)
**Outputs:** `results/predictions/lstm_*`

#### Phase 4: Transformer
```bash
jupyter notebook notebooks/05_transformer.ipynb
```
**Features:** Multi-head attention, visualizable architecture
**Outputs:** `results/predictions/transformer_*`

#### Phase 5-7: Evaluation & Analysis
```bash
jupyter notebook notebooks/06_evaluation_viz.ipynb
```
**Outputs:** Metrics, visualizations, comparisons, explainability

## Project Structure

```
.
├── EXECUTION_PLAN.md          ← Detailed execution roadmap
├── config.yaml                ← All hyperparameters
├── requirements.txt           ← Python dependencies
├── README.md                  ← This file
│
├── data/
│   ├── raw/                   ← Original stock data
│   ├── processed/             ← Cleaned features
│   └── splits/                ← Train/val/test sets
│
├── src/                       ← Source code
│   ├── utils.py              ← Data & utility functions
│   ├── evaluation.py         ← Metrics & evaluation
│   └── models/
│       ├── classical.py      ← ARIMA, SARIMA, Prophet
│       ├── lstm.py           ← LSTM implementations
│       └── transformer.py    ← Transformer encoder
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_data_prep.ipynb
│   ├── 03_classical_models.ipynb
│   ├── 04_lstm_models.ipynb
│   ├── 05_transformer.ipynb
│   └── 06_evaluation_viz.ipynb
│
└── results/
    ├── predictions/          ← Model forecasts
    ├── metrics/             ← Evaluation results
    ├── visualizations/      ← Plots & heatmaps
    └── reports/             ← Summary documents
```

## Key Features

### Data Preparation
✓ Remove survivorship bias
✓ Proper train-test split (no look-ahead bias)
✓ Technical indicators (RSI, MACD, volatility)
✓ Realistic transaction cost adjustment
✓ Automatic scaling & normalization

### Classical Models
✓ ARIMA - AutoRegressive Integrated Moving Average
✓ SARIMA - Seasonal ARIMA with trend detection
✓ Prophet - Facebook's time series forecasting with holidays & seasonality

### Deep Learning Models
✓ **LSTM Single-Step:** Predicts single value (7-day or 30-day price)
✓ **LSTM Multi-Step:** Seq2Seq encoder-decoder for sequence output
✓ **Transformer:** Multi-head attention encoder with explainability

### Evaluation Metrics
- **MAPE:** Mean Absolute Percentage Error
- **RMSE:** Root Mean Squared Error
- **MAE:** Mean Absolute Error
- **Directional Accuracy:** % correct up/down predictions
- **Hit Rate:** % within ±2% of actual
- **Statistical Tests:** Diebold-Mariano significance test

### Explainability
✓ Transformer attention weight heatmaps
✓ SHAP feature importance analysis
✓ Prophet trend/seasonality decomposition
✓ Residual analysis & diagnostics
✓ Model comparison visualizations

## Configuration Options

### Data Configuration (`config.yaml`)
```yaml
data:
  tickers: ['AAPL', 'MSFT', 'GOOGL']      # Stocks to analyze
  start_date: '2018-01-01'                # Historical data start
  end_date: '2025-12-31'                  # Historical data end
  lookback_window: 60                     # Days of history to use
  forecast_horizons: [7, 30]              # Predict 7-day and 30-day ahead

  # Temporal split (CRITICAL: no data leakage!)
  train_end_date: '2023-12-31'
  val_end_date: '2024-09-30'
  test_end_date: '2025-12-31'

  # Realistic market assumptions
  bid_ask_spread: 0.001                   # 0.1%
  slippage: 0.0005                        # 0.05%
  commission_pct: 0.0005                  # 0.05%
```

### Model Hyperparameters
```yaml
lstm:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  dropout: 0.2

transformer:
  num_heads: 8
  model_dim: 64
  num_layers: 2
  learning_rate: 0.001

prophet:
  seasonality_mode: 'additive'
  yearly_seasonality: true
  weekly_seasonality: true
```

## Important Notes

### ⚠️ Data Leakage Prevention
- Train/val/test splits are **temporally ordered** (no overlap)
- Each prediction uses **only past data** (60-day lookback window)
- Scalers fit **only on training data**
- Validation & test sets used only for evaluation

### ⚠️ Survivorship Bias
- Remove stocks that were delisted during analysis period
- Use list of tickers that existed throughout 2018-2025
- Document any tickers excluded

### ⚠️ Realistic Returns
- Include bid-ask spread, slippage, commissions
- Use transaction-cost-adjusted returns for backtesting
- Compare to buy-and-hold baseline

## Results & Outputs

### Metrics Summary (`results/metrics/`)
```
Model Comparison - 7-Day Forecast
        Model    MAPE  RMSE  DA(%)  Hit_Rate(%)
        ARIMA   3.45   2.12   52.3      38.1
        SARIMA  3.28   2.04   53.7      39.5
        Prophet 3.12   1.98   54.2      40.8
        LSTM    2.98   1.87   56.1      42.3
        Transformer 2.75 1.74  57.8    44.5  ← Best
```

### Visualizations (`results/visualizations/`)
- Actual vs Predicted line plots
- Residual distributions & ACF plots
- Transformer attention heatmaps
- Prophet trend/seasonality components
- Feature importance (SHAP)
- Model performance comparison

### Final Report (`results/reports/`)
- Executive summary
- Detailed methodology
- Key findings & insights
- Recommendation framework
- Limitations & future work

## Common Tasks

### Train a Single Model
```python
from src.models.lstm import LSTMForecaster
from src.utils import load_config

config = load_config('config.yaml')
model = LSTMForecaster(config)
model.train(X_train, y_train)
predictions = model.predict(X_test)
```

### Evaluate Models
```python
from src.evaluation import ForecastEvaluator

evaluator = ForecastEvaluator()
metrics = evaluator.evaluate_forecast(y_true, y_pred, "LSTM")
comparison = evaluator.compare_models({
    'ARIMA': (y_true, y_arima),
    'LSTM': (y_true, y_lstm),
    'Transformer': (y_true, y_transformer)
})
```

### Get Feature Importance
```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

## Troubleshooting

**Models not converging?**
- Reduce learning rate
- Increase dropout
- Use early stopping with patience
- Normalize input features better

**Poor directional accuracy?**
- Market may be too random at your prediction horizon
- Try longer lookback window (e.g., 90 instead of 60)
- Consider ensemble of multiple models
- Focus on volatility forecasting instead

**Attention heatmaps not informative?**
- Increase encoder layers
- Use more attention heads
- Visualize top predictions only
- Compare attention patterns by market regime

## References

- Graves, A. et al. (2013). "Speech Recognition with Recurrent Neural Networks" - LSTM foundations
- Vaswani, A. et al. (2017). "Attention Is All You Need" - Transformer architecture
- Taylor, S. J., & Letham, B. (2017). "Forecasting at Scale" - Prophet methodology
- Diebold, F. X., & Mariano, R. S. (1995). "Comparing predictive accuracy" - DM test
- Lundberg, S. M., & Lee, S. I. (2017). "A unified approach to interpreting model predictions" - SHAP

## License

This project is for educational purposes.

## Questions?

Refer to `EXECUTION_PLAN.md` for detailed phase-by-phase guidance with deliverables and checkpoints.
