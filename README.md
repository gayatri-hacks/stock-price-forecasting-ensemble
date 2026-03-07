# Advanced Time Series Analysis (ATSA) Project
## Faculty Presentation Pipeline

This is a comprehensive stock price forecasting system implementing five phases of analysis: data foundation, baseline models, deep learning, backtesting, and visualization.

## Project Structure

```
atsa/
├── phase1_data_foundation/        # Data extraction and preprocessing
│   └── data_foundation.py         # Ticker selection, feature engineering, scaling, splitting
├── phase2_baseline_models/        # Classical statistical models
│   └── baseline_models.py         # ARIMA, SARIMA, Prophet
├── phase3_deep_learning/          # Neural network models
│   └── deep_learning.py           # LSTM (7-day), Transformer (30-day)
├── phase4_backtesting/            # Trading strategy backtesting
│   └── backtesting.py             # Signal logic, market friction, performance metrics
├── phase5_visualization/          # Results synthesis and visualization
│   └── visualization.py           # Dashboards, leaderboards, attention heatmaps
├── utils/
│   ├── config.py                  # Global configuration and parameters
│   ├── helpers.py                 # Utility functions
│   └── __init__.py
├── data/                          # Preprocessed data storage
├── outputs/                       # Results, visualizations, and reports
├── main.py                        # Master orchestration script
└── requirements.txt               # Python dependencies
```

## Installation

1. **Clone/navigate to project directory:**
   ```bash
   cd /Users/aishu/atsa
   ```

2. **Create virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run the Complete Pipeline
```bash
python main.py
```

This will execute all 5 phases in sequence and generate results.

### Run Individual Phases

**Phase 1: Data Foundation**
```python
from phase1_data_foundation.data_foundation import run_phase1
data = run_phase1()
```

**Phase 2: Baseline Models**
```python
from phase2_baseline_models.baseline_models import run_phase2
baseline_results = run_phase2()
```

**Phase 3: Deep Learning**
```python
from phase3_deep_learning.deep_learning import run_phase3
dl_results = run_phase3()
```

**Phase 4: Backtesting**
```python
from phase4_backtesting.backtesting import run_phase4
backtest_results = run_phase4(baseline_results, dl_results)
```

**Phase 5: Visualization**
```python
from phase5_visualization.visualization import run_phase5
viz_results = run_phase5(backtest_results, baseline_results, dl_results)
```

## Phase Details

### Phase 1: Data Foundation
**Role:** Data & Integrity Engineer

- **Ticker Selection:** AAPL (stability), NVDA (high volatility), SPY (index)
- **Feature Extraction:** Downloads OHLCV data using yfinance for 2020-2025
- **Technical Indicators:** 
  - 14-day RSI (momentum)
  - 50-day Moving Average (trend)
- **Data Integrity:** Min-Max scaling fitted on training data only (prevents leakage)
- **Data Split:** 60% training, 20% validation, 20% testing
- **Sequences:** 60-day lookback windows for deep learning models

### Phase 2: Baseline Models
**Role:** Classical Stats Specialist

- **7-Day ARIMA:** Auto-regressive model with rolling 7-day forecasts
- **7-Day SARIMA:** Seasonal variant capturing weekly patterns
- **30-Day Prophet:** Facebook's Prophet with:
  - Monthly seasonality detection
  - US market holidays incorporated
  - 95% confidence intervals
- **Benchmarking:** RMSE, MAE, directional accuracy compared

### Phase 3: Deep Learning
**Role:** Deep Learning Architect

- **LSTM 7-Day Model:**
  - Many-to-Many sequence model
  - 60-day input → 7-day forecast
  - 2 LSTM layers (64 → 32 units)
  - Captures short-term momentum patterns
  
- **Transformer 30-Day Model:**
  - Multi-head attention mechanism (4 heads)
  - 60-day context window
  - 2 Transformer blocks
  - Learns which historical patterns match current volatility

### Phase 4: Backtesting
**Role:** Quantitative/Backtesting Analyst

- **Signal Logic:**
  - Buy when predicted > current + threshold
  - Sell when predicted < current - threshold
  - Different thresholds for 7-day vs 30-day
  
- **Market Friction:**
  - Slippage: 0.05% (buy higher, sell lower)
  - Transaction costs: $1.00 per trade
  
- **Performance Metrics:**
  - Total Return (%)
  - Sharpe Ratio (risk-adjusted returns)
  - Maximum Drawdown (worst peak-to-trough)
  - Win Rate & Trade Count

### Phase 5: Visualization & Explainability
**Role:** Integration & Visualization Lead

- **Explainability:**
  - Transformer attention heatmaps (past days → future prices)
  - Time series decomposition (trend vs seasonality)
  - Forecast vs actual overlay plots
  
- **Comparison Dashboard:**
  - Model leaderboard (return, Sharpe, drawdown, trades)
  - Performance summary tables
  - 4-panel comparison (Return, Sharpe, Drawdown, Activity)
  
- **Integration:**
  - Master results table with all metrics
  - CSV exports for further analysis
  - PNG visualizations for presentations

## Output Files

### Data
- `data/AAPL_phase1.pkl` - Preprocessed AAPL data with sequences
- `data/NVDA_phase1.pkl` - Preprocessed NVDA data with sequences
- `data/SPY_phase1.pkl` - Preprocessed SPY data with sequences

### Results (JSON)
- `outputs/phase2_baseline_{TICKER}.json` - Baseline model metrics
- `outputs/phase3_deep_learning_{TICKER}.json` - Deep learning metrics
- `outputs/phase4_backtest_{TICKER}.json` - Backtest performance

### Summary Tables (CSV)
- `outputs/performance_summary.csv` - All model performance metrics
- `outputs/master_results.csv` - Comprehensive results table

### Visualizations (PNG)
- `outputs/model_comparison_leaderboard.png` - 4-panel leaderboard
- `outputs/{TICKER}_attention_heatmap.png` - Transformer attention weights
- `outputs/{TICKER}_decomposition.png` - Time series decomposition
- `outputs/{TICKER}_ARIMA_7day_forecast.png` - ARIMA predictions
- `outputs/{TICKER}_Prophet_30day_forecast.png` - Prophet predictions
- `outputs/{TICKER}_LSTM_7day_forecast.png` - LSTM predictions
- `outputs/{TICKER}_Transformer_30day_forecast.png` - Transformer predictions
- `outputs/{TICKER}_{MODEL}_portfolio.png` - Portfolio value over time

## Configuration

Edit `utils/config.py` to customize:
- Stock tickers
- Date ranges
- Train/val/test split ratios
- Technical indicator periods
- Model hyperparameters
- Backtesting parameters

## Key Concepts

### Data Leakage Prevention
The Min-Max scaler is fit ONLY on training data. Validation and test data are transformed using training statistics. This prevents the model from "knowing" about future price ranges.

### Rolling Forecasts
Baseline models use rolling forecasts where the model is retrained at each step, mimicking real-world prediction scenarios.

### Market Friction
Realistic modeling includes:
- Slippage: Assuming execution at less favorable prices
- Transaction costs: Fixed per-trade fees

### Evaluation Metrics
- **RMSE:** Overall price prediction accuracy
- **Directional Accuracy:** Did the model get "up/down" right?
- **Sharpe Ratio:** Returns per unit of risk
- **Maximum Drawdown:** Severity of losses

## Dependencies

Core libraries:
- `yfinance`: Stock data download
- `pandas/numpy`: Data manipulation
- `scikit-learn`: Preprocessing and metrics
- `tensorflow/keras`: Deep learning
- `torch`: Alternative deep learning backend
- `prophet`: Facebook's forecasting library
- `statsmodels`: ARIMA/SARIMA models
- `matplotlib/seaborn`: Visualization

## Implementation Notes

1. **Memory Management:** Large pickle files (>50MB) may be created for sequences
2. **Training Time:** Full pipeline runs 15-30 minutes depending on hardware
3. **GPU Acceleration:** TensorFlow will use GPU if available
4. **Reproducibility:** Set RANDOM_SEED in config.py

## Faculty Presentation Talking Points

1. **Why Multiple Timeframes?**
   - 7-day: Momentum trading, short-term swings
   - 30-day: Trend-following, seasonal patterns

2. **Classical vs AI:**
   - ARIMA/Prophet: Explainable, interpretable
   - LSTM/Transformer: Captures complex patterns, higher accuracy

3. **Backtesting Reality:**
   - Real trading has friction (slippage, fees)
   - Sharpe ratio accounts for risk
   - Drawdown shows worst-case scenarios

4. **Attention Mechanisms:**
   - Transformer learns WHICH past patterns matter
   - Visualize as heatmaps: which days influenced predictions?

5. **Risk Management:**
   - Maximum drawdown controls loss
   - Trade count shows activity level
   - Win rate validates signal quality

## Troubleshooting

**Issue: `ModuleNotFoundError`**
- Ensure all dependencies installed: `pip install -r requirements.txt`
- Check Python version (3.8+)

**Issue: Out of Memory**
- Reduce BATCH_SIZE in config.py
- Lower EPOCHS
- Process one ticker at a time

**Issue: yfinance Connection Error**
- Check internet connection
- Market data may be delayed on weekends/holidays
- Try different ticker

**Issue: Model not converging**
- Increase EPOCHS
- Adjust learning rate in deep_learning.py
- Check data scaling

## Future Enhancements

1. Ensemble models combining all predictions
2. Reinforcement learning for dynamic risk management
3. Real-time streaming data integration
4. Interactive Dash/Streamlit dashboard
5. Custom loss functions for risk-adjusted optimization
6. Multi-step ahead predictions with uncertainty quantification

## Author Notes

This implementation is designed for educational purposes and faculty presentations. It demonstrates:
- End-to-end data science pipeline
- Multiple modeling paradigms
- Practical considerations (market friction, data leakage)
- Comprehensive visualization and reporting

For production use, additional considerations are needed:
- Transaction cost models
- Liquidity constraints
- Portfolio optimization
- Real-time monitoring
- Regulatory compliance

## License

For academic and educational use.
