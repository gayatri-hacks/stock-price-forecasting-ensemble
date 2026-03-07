"""
Utility functions for stock forecasting project
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict, List, Optional
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


def load_config(config_path: str = 'config.yaml') -> Dict:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_directories(paths: Dict[str, str]) -> None:
    """Create all necessary directories"""
    for path_value in paths.values():
        Path(path_value).mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory ready: {path_value}")


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except:
        pass
    logger.info(f"Random seed set to {seed}")


def load_stock_data(ticker: str, 
                   start_date: str, 
                   end_date: str,
                   source: str = 'yfinance') -> pd.DataFrame:
    """
    Load stock data from various sources
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    source : str
        Data source ('yfinance', 'csv', etc.)
    
    Returns:
    --------
    df : pd.DataFrame
        OHLCV data with datetime index
    """
    
    if source.lower() == 'yfinance':
        try:
            import yfinance as yf
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            df.index.name = 'Date'
            logger.info(f"Downloaded {ticker}: {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Error downloading {ticker}: {e}")
            return pd.DataFrame()
    
    elif source.lower() == 'csv':
        try:
            df = pd.read_csv(f'data/raw/{ticker}.csv', index_col='Date', parse_dates=True)
            logger.info(f"Loaded CSV {ticker}: {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV {ticker}: {e}")
            return pd.DataFrame()
    
    else:
        raise ValueError(f"Unknown data source: {source}")


def check_stationarity(series: pd.Series, name: str = '') -> Dict:
    """
    Check stationarity using ADF test
    
    Returns:
    --------
    dict with adf_stat, p_value, is_stationary
    """
    from statsmodels.tsa.stattools import adfuller
    
    result = adfuller(series.dropna())
    
    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'is_stationary': result[1] < 0.05,
        'critical_values': result[4]
    }


def calculate_returns(prices: pd.Series, 
                     method: str = 'log') -> pd.Series:
    """
    Calculate returns
    
    Parameters:
    -----------
    prices : pd.Series
        Price series
    method : str
        'simple' for (P_t - P_{t-1}) / P_{t-1}
        'log' for log(P_t / P_{t-1})
    
    Returns:
    --------
    returns : pd.Series
    """
    if method == 'log':
        returns = np.log(prices / prices.shift(1))
    elif method == 'simple':
        returns = prices.pct_change()
    else:
        raise ValueError(f"Unknown return method: {method}")
    
    return returns


def create_features(df: pd.DataFrame,
                   feature_cols: List[str],
                   lookback: int = 60) -> pd.DataFrame:
    """
    Create technical indicators and features
    
    Common features:
    - Rolling averages (SMA)
    - Rolling volatility (std)
    - RSI
    - MACD
    - Bollinger Bands
    """
    
    df = df.copy()
    
    # Log returns
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Rolling averages
    df['rolling_mean_5'] = df['Close'].rolling(5).mean()
    df['rolling_mean_20'] = df['Close'].rolling(20).mean()
    df['rolling_mean_50'] = df['Close'].rolling(50).mean()
    
    # Rolling volatility
    df['rolling_vol_20'] = df['log_returns'].rolling(20).std()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Volume ratio
    df['volume_sma_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    return df


def remove_outliers(series: pd.Series, 
                   method: str = 'iqr',
                   threshold: float = 3.0) -> pd.Series:
    """
    Remove outliers from a series
    
    Parameters:
    -----------
    series : pd.Series
    method : str
        'iqr' or 'zscore'
    threshold : float
        IQR multiplier or z-score threshold
    
    Returns:
    --------
    cleaned series with outliers set to NaN
    """
    
    series = series.copy()
    
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        series = series.mask((series < lower) | (series > upper))
    
    elif method == 'zscore':
        z_scores = np.abs((series - series.mean()) / series.std())
        series = series.mask(z_scores > threshold)
    
    return series


def normalize_scale(data: np.ndarray,
                   method: str = 'standard',
                   fit_data: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
    """
    Scale data
    
    Parameters:
    -----------
    data : np.ndarray
        Data to scale
    method : str
        'standard', 'minmax', 'robust'
    fit_data : np.ndarray
        Data to compute scaling params (if None, uses data)
    
    Returns:
    --------
    scaled_data : np.ndarray
    scaler_params : dict
    """
    
    if fit_data is None:
        fit_data = data
    
    if method == 'standard':
        mean = fit_data.mean(axis=0)
        std = fit_data.std(axis=0)
        scaled = (data - mean) / (std + 1e-8)
        return scaled, {'method': 'standard', 'mean': mean, 'std': std}
    
    elif method == 'minmax':
        min_val = fit_data.min(axis=0)
        max_val = fit_data.max(axis=0)
        scaled = (data - min_val) / (max_val - min_val + 1e-8)
        return scaled, {'method': 'minmax', 'min': min_val, 'max': max_val}
    
    elif method == 'robust':
        q1 = np.percentile(fit_data, 25, axis=0)
        q3 = np.percentile(fit_data, 75, axis=0)
        iqr = q3 - q1
        median = np.percentile(fit_data, 50, axis=0)
        scaled = (data - median) / (iqr + 1e-8)
        return scaled, {'method': 'robust', 'median': median, 'iqr': iqr}
    
    else:
        raise ValueError(f"Unknown scaling method: {method}")


def inverse_scale(data: np.ndarray, scaler_params: Dict) -> np.ndarray:
    """Inverse scaling transformation"""
    
    method = scaler_params['method']
    
    if method == 'standard':
        return data * scaler_params['std'] + scaler_params['mean']
    
    elif method == 'minmax':
        return data * (scaler_params['max'] - scaler_params['min']) + scaler_params['min']
    
    elif method == 'robust':
        return data * scaler_params['iqr'] + scaler_params['median']


def create_sequences(data: np.ndarray,
                    lookback: int,
                    forecast_horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for time series
    
    Parameters:
    -----------
    data : np.ndarray
        Shape (n_samples, n_features)
    lookback : int
        Number of past timesteps to use
    forecast_horizon : int
        Number of future timesteps to predict
    
    Returns:
    --------
    X : np.ndarray
        Shape (n_sequences, lookback, n_features)
    y : np.ndarray
        Shape (n_sequences, forecast_horizon) or (n_sequences,)
    """
    
    X, y = [], []
    
    for i in range(len(data) - lookback - forecast_horizon + 1):
        X.append(data[i:i+lookback])
        # For single-step: take last value of horizon
        y.append(data[i+lookback:i+lookback+forecast_horizon, 0])  # Assumes Close is first feature
    
    return np.array(X), np.array(y)


def train_val_test_split(df: pd.DataFrame,
                        train_end: str,
                        val_end: str,
                        test_end: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Temporal train/val/test split with no data leakage
    
    Returns:
    --------
    train_df, val_df, test_df
    """
    
    train = df[df.index <= train_end]
    val = df[(df.index > train_end) & (df.index <= val_end)]
    test = df[(df.index > val_end) & (df.index <= test_end)]
    
    logger.info(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    
    return train, val, test


def calculate_transaction_costs(returns: np.ndarray,
                               bid_ask: float = 0.001,
                               slippage: float = 0.0005,
                               commission: float = 0.0005) -> np.ndarray:
    """
    Adjust returns for realistic transaction costs
    
    Parameters:
    -----------
    returns : np.ndarray or pd.Series
        Return series
    bid_ask : float
        Bid-ask spread as proportion
    slippage : float
        Slippage as proportion
    commission : float
        Commission as proportion
    
    Returns:
    --------
    adjusted_returns
    """
    
    total_cost = bid_ask + slippage + commission
    adjusted = returns - total_cost
    
    return adjusted


if __name__ == "__main__":
    # Example usage
    config = load_config('config.yaml')
    print(f"Config loaded: {list(config.keys())}")
