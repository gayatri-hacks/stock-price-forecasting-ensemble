"""Configuration and constants for the entire project."""

# Ticker selection: stability, high volatility, and index
TICKERS = ['AAPL', 'NVDA', 'SPY']

# Date range for data collection
START_DATE = '2020-01-01'
END_DATE = '2025-12-31'

# Data split percentages
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# Technical indicators parameters
RSI_PERIOD = 14
MA_PERIOD = 50

# LSTM/Transformer parameters
LOOKBACK_WINDOW = 60
FORECAST_HORIZONS = {
    'short': 7,    # 7-day LSTM forecast
    'long': 30     # 30-day Transformer forecast
}

# Backtesting parameters
SLIPPAGE_RATE = 0.0005  # 0.05%
TRANSACTION_COST = 1.00  # $1.00 per trade

# Model training parameters
BATCH_SIZE = 32
EPOCHS = 100
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# Output paths
DATA_PATH = './data'
MODELS_PATH = './models'
OUTPUTS_PATH = './outputs'
