"""Phase 1: Data Foundation - Extract, preprocess, and split data."""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utils.config import (
    TICKERS, START_DATE, END_DATE, TRAIN_RATIO, VAL_RATIO, 
    RSI_PERIOD, MA_PERIOD, LOOKBACK_WINDOW, RANDOM_SEED
)
from utils.helpers import save_preprocessed_data, ensure_directories


def fetch_stock_data(ticker):
    """Fetch OHLCV data from yfinance."""
    print(f"Fetching data for {ticker} from {START_DATE} to {END_DATE}...")
    data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
    return data


def add_technical_indicators(df):
    """Add RSI and Moving Average to dataframe."""
    df = df.copy()
    
    # Calculate RSI using Pandas
    close = df['Close']
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    df['RSI'] = rsi
    df['MA50'] = df['Close'].rolling(window=MA_PERIOD).mean()
    return df


def scale_data(train_data, val_data, test_data):
    """Apply Min-Max scaling using training data parameters only."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit scaler on training data only (prevent data leakage)
    scaler.fit(train_data)
    
    # Transform all datasets
    train_scaled = scaler.transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)
    
    return train_scaled, val_scaled, test_scaled, scaler


def split_train_val_test(df, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    """Split data into train, validation, and test sets."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
    
    return train_df, val_df, test_df


def prepare_data_for_modeling(data, lookback=LOOKBACK_WINDOW):
    """
    Prepare sequences for LSTM/Transformer models.
    Creates overlapping windows of lookback length.
    """
    # If data is a DataFrame, convert to values
    if hasattr(data, 'values'):
        data = data.values
    
    data = np.array(data, dtype=np.float32)
    X, y = [], []
    
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback, 3])  # Close price is at index 3 (OHLCV)
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def process_single_ticker(ticker):
    """Process a single ticker through Phase 1."""
    print(f"\n{'='*60}")
    print(f"Processing {ticker}")
    print(f"{'='*60}")
    
    # 1. Fetch raw data
    raw_data = fetch_stock_data(ticker)
    print(f"Raw data shape: {raw_data.shape}")
    
    # 2. Add technical indicators
    data_with_indicators = add_technical_indicators(raw_data)
    
    # 3. Drop NaN values (from moving average calculations)
    data_clean = data_with_indicators.dropna().reset_index(drop=True)
    print(f"Data after dropping NaN: {data_clean.shape}")
    
    # 4. Select relevant features: Open, High, Low, Close, Volume, RSI, MA50
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MA50']
    data_subset = data_clean[features].copy()
    
    # 5. Split into train/val/test
    train_df, val_df, test_df = split_train_val_test(data_subset)
    
    # 6. Scale data using training statistics only
    train_scaled, val_scaled, test_scaled, scaler = scale_data(
        train_df.values,
        val_df.values,
        test_df.values
    )
    
    # 7. Create sequences for LSTM/Transformer
    X_train, y_train = prepare_data_for_modeling(train_scaled)
    X_val, y_val = prepare_data_for_modeling(val_scaled)
    X_test, y_test = prepare_data_for_modeling(test_scaled)
    
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    # 8. Save all data for downstream phases
    data_dict = {
        'raw_data': raw_data,
        'data_with_indicators': data_with_indicators,
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'train_scaled': train_scaled,
        'val_scaled': val_scaled,
        'test_scaled': test_scaled,
        'scaler': scaler,
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'features': features,
        'dates': data_clean.index
    }
    
    save_preprocessed_data(data_dict, ticker, phase='phase1')
    
    return data_dict


def run_phase1(tickers=TICKERS):
    """Run Phase 1 for all tickers."""
    ensure_directories()
    
    print("\n" + "="*60)
    print("PHASE 1: DATA FOUNDATION")
    print("="*60)
    
    all_data = {}
    for ticker in tickers:
        all_data[ticker] = process_single_ticker(ticker)
    
    print("\n" + "="*60)
    print("PHASE 1 COMPLETE: Data foundation established")
    print("="*60)
    
    return all_data


if __name__ == "__main__":
    phase1_data = run_phase1()
