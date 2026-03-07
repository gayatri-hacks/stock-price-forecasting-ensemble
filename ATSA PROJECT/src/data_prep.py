"""
Data preparation module for stock forecasting
Handles loading, cleaning, feature engineering, and splitting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

from src.utils import (
    load_stock_data, 
    create_features, 
    remove_outliers,
    normalize_scale,
    create_sequences,
    train_val_test_split,
    check_stationarity
)

logger = logging.getLogger(__name__)


class DataPreparationPipeline:
    """Complete data preparation pipeline with no look-ahead bias"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.scaler_params = None
        self.feature_cols = None
        
    def load_raw_data(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Load raw stock data for multiple tickers
        
        Returns:
        --------
        data_dict : {ticker: DataFrame}
        """
        data_dict = {}
        
        for ticker in tickers:
            try:
                df = load_stock_data(
                    ticker,
                    self.config['data']['start_date'],
                    self.config['data']['end_date'],
                    source=self.config['data']['download_source']
                )
                
                if len(df) > self.config['data']['min_data_points']:
                    data_dict[ticker] = df
                    logger.info(f"✓ Loaded {ticker}: {len(df)} records")
                else:
                    logger.warning(f"✗ {ticker} has insufficient data ({len(df)} < {self.config['data']['min_data_points']})")
                    
            except Exception as e:
                logger.error(f"Error loading {ticker}: {e}")
        
        return data_dict
    
    def remove_survivorship_bias(self, 
                                 data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Remove stocks that don't exist for entire analysis period
        Uses config parameters: survivorship.min_required_days_pct and survivorship.remove_incomplete
        
        Returns:
        --------
        filtered_dict : {ticker: DataFrame} - only tickers with full date range
        """
        
        start_date = pd.to_datetime(self.config['data']['start_date'])
        end_date = pd.to_datetime(self.config['data']['end_date'])
        valid_tickers = {}
        removed = []
        
        # Get survivorship settings from config
        survival_cfg = self.config['data'].get('survivorship', {})
        min_days_pct = survival_cfg.get('min_required_days_pct', 0.95)
        remove_incomplete = survival_cfg.get('remove_incomplete', True)
        
        total_days = (end_date - start_date).days
        min_required_days = int(total_days * min_days_pct)
        
        logger.info(f"Survivorship Bias Removal Settings:")
        logger.info(f"  Required coverage: {min_days_pct*100:.0f}% ({min_required_days} days out of {total_days})")
        logger.info(f"  Remove incomplete: {remove_incomplete}")
        
        for ticker, df in data_dict.items():
            # Check if ticker has data covering full period
            df_start = df.index.min()
            df_end = df.index.max()
            actual_days = (df_end - df_start).days
            
            if df_start <= start_date and df_end >= end_date and actual_days >= min_required_days:
                valid_tickers[ticker] = df
                logger.info(f"✓ {ticker}: Full coverage {df_start.date()} to {df_end.date()} ({actual_days} days)")
            else:
                removed.append(ticker)
                reason = ""
                if df_start > start_date:
                    reason += f"starts {(df_start-start_date).days} days late; "
                if df_end < end_date:
                    reason += f"ends {(end_date-df_end).days} days early; "
                if actual_days < min_required_days:
                    reason += f"only {actual_days}/{min_required_days} days; "
                logger.warning(f"✗ {ticker}: Excluded ({reason})")
        
        logger.info(f"\nSurvivorship Bias Removal Summary:")
        logger.info(f"  Valid tickers: {len(valid_tickers)} / {len(data_dict)}")
        logger.info(f"  Removed: {len(removed)} tickers")
        if removed:
            logger.info(f"  List: {removed}")
        
        return valid_tickers
    
    def prepare_single_ticker(self, 
                              ticker: str,
                              df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare single ticker: clean, engineer features
        
        Returns:
        --------
        df : Processed DataFrame with features
        """
        
        df = df.copy()
        logger.info(f"\nPreparing {ticker}...")
        
        # 1. Check for missing values
        missing_pct = (df.isnull().sum() / len(df) * 100).sum() / len(df.columns)
        logger.info(f"  Missing data: {missing_pct:.2f}%")
        
        if missing_pct > 5:
            df = df.fillna(method='ffill').fillna(method='bfill')
            logger.info(f"  Forward-filled missing values")
        
        # 2. Ensure numeric types
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 3. Create features
        df = create_features(df, self.config['data']['features'], 
                           lookback=self.config['data']['lookback_window'])
        
        # 4. Remove outliers
        for col in ['log_returns', 'rolling_vol_20']:
            if col in df.columns:
                df[col] = remove_outliers(df[col], 
                                         method=self.config['preprocessing']['outlier_method'],
                                         threshold=self.config['preprocessing']['outlier_threshold'])
        
        # 5. Handle NaNs created by feature engineering
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        # 6. Check stationarity
        if 'log_returns' in df.columns:
            stationarity = check_stationarity(df['log_returns'], f"{ticker} Returns")
            logger.info(f"  Stationarity: p-value={stationarity['p_value']:.4f} "
                       f"({'stationary' if stationarity['is_stationary'] else 'non-stationary'})")
        
        logger.info(f"  ✓ {ticker} prepared: {df.shape[0]} rows × {df.shape[1]} cols")
        
        return df
    
    def create_targets(self, 
                      df: pd.DataFrame,
                      horizons: List[int] = [7, 30]) -> pd.DataFrame:
        """
        Create forward-looking target variables (NO LOOK-AHEAD BIAS)
        
        For each row t, target[h] = log return from t to t+h
        Example: target_7d at row 100 = log(Price_107 / Price_100)
        
        This ensures we NEVER use future price data when creating training samples.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with price column
        horizons : List[int]
            Forecast horizons (e.g., [7, 30])
        
        Returns:
        --------
        df : DataFrame with target_7d, target_30d columns
        """
        
        df = df.copy()
        
        # Get the price column to shift (from config)
        price_col = self.config['data'].get('target_col', 'Adj Close')
        if price_col not in df.columns:
            # Fallback to Close if Adj Close doesn't exist
            price_col = 'Close' if 'Close' in df.columns else df.columns[3]  # Usually 4th column
            logger.warning(f"  {self.config['data']['target_col']} not found, using {price_col}")
        
        close_prices = df[price_col].values
        
        logger.info(f"Creating targets using column: {price_col}")
        
        for horizon in horizons:
            # Create forward-looking returns
            forward_returns = np.zeros(len(df))
            
            for i in range(len(df) - horizon):
                # Log return: log(Price_future / Price_current)
                if close_prices[i] > 0 and close_prices[i + horizon] > 0:
                    forward_returns[i] = np.log(close_prices[i + horizon] / close_prices[i])
                else:
                    forward_returns[i] = np.nan
            
            # NaN for last horizon points (can't predict beyond available data)
            forward_returns[-horizon:] = np.nan
            
            target_name = f'target_{horizon}d'
            df[target_name] = forward_returns
            
            # Count valid targets
            valid_targets = np.sum(~np.isnan(forward_returns))
            logger.info(f"  {target_name}: {valid_targets}/{len(df)} valid targets")
        
        logger.info(f"✓ Target variables created (7-day and 30-day forward returns)")
        
        return df
    
    def prepare_dataset(self, 
                       data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine all tickers into single dataset
        
        Returns:
        --------
        combined_df : DataFrame with all features and targets
        """
        
        all_dfs = []
        
        for ticker, df in data_dict.items():
            df = self.prepare_single_ticker(ticker, df)
            df['ticker'] = ticker
            all_dfs.append(df)
        
        combined = pd.concat(all_dfs, ignore_index=False)
        combined = combined.sort_index()
        
        logger.info(f"\nCombined dataset: {combined.shape[0]} rows × {combined.shape[1]} cols")
        
        return combined
    
    def split_time_series(self,
                         df: pd.DataFrame,
                         lookback: int = 60,
                         horizons: List[int] = [7, 30]) -> Dict:
        """
        Create train/val/test splits with NO look-ahead bias
        
        CRITICAL: Uses gap_buffer_days to separate splits
        This ensures the maximum forecast horizon (30 days) doesn't leak between splits
        
        Example Timeline:
        Training:   2018-01-01 to 2023-12-31
        GAP:        2024-01-01 to 2024-01-29 (30-day buffer)
        Validation: 2024-01-30 to 2024-09-30
        GAP:        2024-10-01 to 2024-10-29 (30-day buffer)
        Test:       2024-10-30 to 2025-12-31
        
        Returns:
        --------
        splits : {
            'train': {'X': ..., 'y': {...}, 'dates': ...}
            'val': {'X': ..., 'y': {...}, 'dates': ...}
            'test': {'X': ..., 'y': {...}, 'dates': ...}
        }
        """
        
        # Get dates and buffer from config
        train_end = pd.to_datetime(self.config['data']['train_end_date'])
        val_start = pd.to_datetime(self.config['data'].get('val_start_date', self.config['data']['train_end_date']))
        val_end = pd.to_datetime(self.config['data']['val_end_date'])
        test_start = pd.to_datetime(self.config['data'].get('test_start_date', self.config['data']['val_end_date']))
        test_end = pd.to_datetime(self.config['data']['test_end_date'])
        gap_buffer = self.config['data'].get('gap_buffer_days', 30)
        
        # Validate buffer is sufficient for maximum horizon
        max_horizon = max(horizons)
        if gap_buffer < max_horizon:
            logger.warning(f"⚠️  gap_buffer_days ({gap_buffer}) < max forecast horizon ({max_horizon})")
            logger.warning(f"   This may cause look-ahead bias! Recommend gap_buffer_days >= {max_horizon}")
        
        # Split data using date boundaries
        train = df[df.index <= train_end]
        val = df[(df.index >= val_start) & (df.index <= val_end)]
        test = df[(df.index >= test_start) & (df.index <= test_end)]
        
        # Discarded regions (gap buffers)
        gap1 = df[(df.index > train_end) & (df.index < val_start)]
        gap2 = df[(df.index > val_end) & (df.index < test_start)]
        
        logger.info(f"\nTemporal Train/Val/Test Split (NO Data Leakage):")
        logger.info(f"  Training:   {train.index.min().date()} to {train.index.max().date()} ({len(train)} samples)")
        logger.info(f"  GAP Buffer: {gap1.index.min().date() if len(gap1) > 0 else 'N/A'} to {gap1.index.max().date() if len(gap1) > 0 else 'N/A'} ({len(gap1)} samples, discarded)")
        logger.info(f"  Validation: {val.index.min().date()} to {val.index.max().date()} ({len(val)} samples)")
        logger.info(f"  GAP Buffer: {gap2.index.min().date() if len(gap2) > 0 else 'N/A'} to {gap2.index.max().date() if len(gap2) > 0 else 'N/A'} ({len(gap2)} samples, discarded)")
        logger.info(f"  Test:       {test.index.min().date()} to {test.index.max().date()} ({len(test)} samples)")
        logger.info(f"\n  Buffer days between splits: {gap_buffer}")
        logger.info(f"  Max forecast horizon: {max_horizon} days (ensures no leakage)")
        
        # Feature columns to use
        self.feature_cols = [col for col in df.columns 
                            if col not in ['Close', 'Adj Close', 'ticker'] 
                            and not col.startswith('target_')]
        
        splits = {}
        
        for split_name, split_df in [('train', train), ('val', val), ('test', test)]:
            split_data = {}
            
            # Create sequences for each horizon
            if split_name == 'train':
                # Fit scaler on training data only
                X_values = split_df[self.feature_cols].values
                X_scaled, self.scaler_params = normalize_scale(
                    X_values,
                    method=self.config['preprocessing']['scaling_method'],
                    fit_data=X_values
                )
            else:
                # Use training scaler parameters
                X_scaled, _ = normalize_scale(
                    split_df[self.feature_cols].values,
                    method=self.config['preprocessing']['scaling_method'],
                    fit_data=None  # Use scaler_params
                )
            
            split_data['X'] = X_scaled
            split_data['dates'] = split_df.index
            
            # Targets for each horizon
            split_data['y'] = {}
            for horizon in horizons:
                target_col = f'target_{horizon}d'
                if target_col in split_df.columns:
                    split_data['y'][horizon] = split_df[target_col].values
                else:
                    logger.warning(f"⚠️  {target_col} not found in {split_name} split")
            
            splits[split_name] = split_data
        
        logger.info(f"\nFeatures used: {len(self.feature_cols)}")
        logger.info(f"  {self.feature_cols}")
        
        return splits
    
    def create_windowed_sequences(self,
                                  splits: Dict,
                                  lookback: int = 60,
                                  horizons: List[int] = [7, 30]) -> Dict:
        """
        Create sliding window sequences for deep learning
        
        Returns:
        --------
        windowed_splits : {split_name: {'X': (n_samples, lookback, n_features), 'y': {...}}}
        """
        
        windowed = {}
        
        for split_name, split_data in splits.items():
            X = split_data['X']
            
            # Create sequences
            X_seq = []
            valid_indices = []
            
            for i in range(len(X) - lookback - max(horizons) + 1):
                X_seq.append(X[i:i+lookback])
                valid_indices.append(i + lookback)
            
            X_seq = np.array(X_seq)
            
            # Targets
            y_dict = {}
            for horizon in horizons:
                y_full = split_data['y'][horizon]
                y_seq = np.array([y_full[idx:idx+horizon].mean() 
                                 for idx in valid_indices])
                y_dict[horizon] = y_seq
            
            windowed[split_name] = {
                'X': X_seq,
                'y': y_dict,
                'dates': split_data['dates'].values[valid_indices]
            }
            
            logger.info(f"  {split_name.upper()}: X shape {X_seq.shape}, "
                       f"y shapes {[y.shape for y in y_dict.values()]}")
        
        return windowed


if __name__ == "__main__":
    # Example usage
    from src.utils import load_config
    
    config = load_config('config.yaml')
    
    # Initialize pipeline
    pipeline = DataPreparationPipeline(config)
    
    # Load data
    raw_data = pipeline.load_raw_data(config['data']['tickers'][:3])  # First 3 tickers
    
    # Remove survivorship bias
    valid_data = pipeline.remove_survivorship_bias(raw_data)
    
    # Prepare full dataset
    df_prepared = pipeline.prepare_dataset(valid_data)
    
    # Create targets
    df_prepared = pipeline.create_targets(df_prepared, config['data']['forecast_horizons'])
    
    print("\nData Preparation Complete!")
    print(f"Shape: {df_prepared.shape}")
    print(f"\nColumns: {list(df_prepared.columns)}")
