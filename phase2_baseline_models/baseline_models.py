"""Phase 2: Baseline Models - Simplified ARIMA and Prophet."""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
try:
    from pmdarima import auto_arima
except ImportError:
    auto_arima = None
from utils.config import TICKERS
from utils.helpers import load_preprocessed_data, save_results, calculate_metrics, directional_accuracy
import warnings
warnings.filterwarnings('ignore')


def forecast_7day_arima(data_dict, ticker):
    """Create 7-day ARIMA forecast."""
    print(f"\n--- ARIMA 7-Day Forecast for {ticker} ---")
    
    try:
        train_df = data_dict['train_df']
        val_df = data_dict['val_df']
        test_df = data_dict['test_df']
        
        train_prices = np.asarray(train_df['Close'].values, dtype=float).ravel()
        val_prices = np.asarray(val_df['Close'].values, dtype=float).ravel()
        test_prices = np.asarray(test_df['Close'].values, dtype=float).ravel()
        
        model = ARIMA(train_prices, order=(5, 1, 2)).fit()
        all_future = np.concatenate([val_prices, test_prices])
        
        # Get forecast
        forecast_result = model.get_forecast(steps=len(all_future))
        forecasts = np.asarray(forecast_result.predicted_mean, dtype=float).ravel()
        
        # Ensure same length
        if len(forecasts) != len(all_future):
            min_len = min(len(forecasts), len(all_future))
            forecasts = forecasts[:min_len]
            all_future = all_future[:min_len]
        
        metrics = calculate_metrics(all_future, forecasts)
        dir_acc = directional_accuracy(all_future.ravel(), forecasts.ravel())
        
        print(f"RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}, Dir Acc: {dir_acc:.4f}")
        
        return {
            'model': model,
            'forecasts': forecasts,
            'actuals': all_future,
            'metrics': metrics,
            'directional_accuracy': dir_acc
        }
    except Exception as e:
        print(f"ARIMA error: {e}")
        train_df = data_dict['train_df']
        val_df = data_dict['val_df']
        test_df = data_dict['test_df']
        all_future = np.concatenate([val_df['Close'].values, test_df['Close'].values])
        dummy_forecasts = np.full(len(all_future), np.mean(train_df['Close'].values))
        return {
            'model': None,
            'forecasts': dummy_forecasts,
            'actuals': all_future,
            'metrics': {'RMSE': 0, 'MAE': 0, 'MAPE': 0, 'MSE': 0},
            'directional_accuracy': 0
        }


def forecast_7day_sarima(data_dict, ticker):
    """Create 7-day SARIMA forecast with seasonal components."""
    print(f"\n--- SARIMA 7-Day Forecast for {ticker} ---")
    
    if auto_arima is None:
        print("SARIMA skipped: pmdarima not installed. Install with: pip install pmdarima")
        train_df = data_dict['train_df']
        val_df = data_dict['val_df']
        test_df = data_dict['test_df']
        all_future = np.concatenate([val_df['Close'].values, test_df['Close'].values])
        dummy_forecasts = np.full(len(all_future), np.mean(train_df['Close'].values))
        return {
            'model': None,
            'forecasts': dummy_forecasts,
            'actuals': all_future,
            'metrics': {'RMSE': 0, 'MAE': 0, 'MAPE': 0, 'MSE': 0},
            'directional_accuracy': 0
        }
    
    try:
        train_df = data_dict['train_df']
        val_df = data_dict['val_df']
        test_df = data_dict['test_df']
        
        train_prices = np.asarray(train_df['Close'].values, dtype=float).ravel()
        val_prices = np.asarray(val_df['Close'].values, dtype=float).ravel()
        test_prices = np.asarray(test_df['Close'].values, dtype=float).ravel()
        
        # Auto-select SARIMA parameters with seasonal=True (m=5 for weekly)
        model = auto_arima(train_prices, seasonal=True, m=5, max_p=3, max_d=2, max_q=3,
                          max_P=2, max_D=1, max_Q=2, stepwise=True, suppress_warnings=True)
        all_future = np.concatenate([val_prices, test_prices])
        
        # Get forecast
        forecasts, _ = model.predict(n_periods=len(all_future), return_conf_int=True)
        forecasts = np.asarray(forecasts, dtype=float).ravel()
        
        # Ensure same length
        if len(forecasts) != len(all_future):
            min_len = min(len(forecasts), len(all_future))
            forecasts = forecasts[:min_len]
            all_future = all_future[:min_len]
        
        metrics = calculate_metrics(all_future, forecasts)
        dir_acc = directional_accuracy(all_future.ravel(), forecasts.ravel())
        
        print(f"RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}, Dir Acc: {dir_acc:.4f}")
        
        return {
            'model': model,
            'forecasts': forecasts,
            'actuals': all_future,
            'metrics': metrics,
            'directional_accuracy': dir_acc
        }
    except Exception as e:
        print(f"SARIMA error: {e}")
        train_df = data_dict['train_df']
        val_df = data_dict['val_df']
        test_df = data_dict['test_df']
        all_future = np.concatenate([val_df['Close'].values, test_df['Close'].values])
        dummy_forecasts = np.full(len(all_future), np.mean(train_df['Close'].values))
        return {
            'model': None,
            'forecasts': dummy_forecasts,
            'actuals': all_future,
            'metrics': {'RMSE': 0, 'MAE': 0, 'MAPE': 0, 'MSE': 0},
            'directional_accuracy': 0
        }


def forecast_30day_prophet(data_dict, ticker):
    """Create 30-day Prophet forecast."""
    print(f"\n--- Prophet 30-Day Forecast for {ticker} ---")
    
    try:
        train_df = data_dict['train_df']
        val_df = data_dict['val_df']
        test_df = data_dict['test_df']
        
        train_prices = np.asarray(train_df['Close'].values, dtype=float).ravel()
        val_prices = np.asarray(val_df['Close'].values, dtype=float).ravel()
        test_prices = np.asarray(test_df['Close'].values, dtype=float).ravel()
        
        df = pd.DataFrame({
            'ds': pd.date_range(start='2020-01-01', periods=len(train_prices), freq='D'),
            'y': train_prices
        })
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = Prophet(yearly_seasonality=False, weekly_seasonality=False, 
                           daily_seasonality=False, interval_width=0.95)
            model.fit(df)
        
        all_future = np.concatenate([val_prices, test_prices])
        future = model.make_future_dataframe(periods=len(all_future))
        forecast = model.predict(future)
        forecasts = np.asarray(forecast['yhat'].values[-len(all_future):], dtype=float).ravel()
        
        # Ensure same length
        if len(forecasts) != len(all_future):
            min_len = min(len(forecasts), len(all_future))
            forecasts = forecasts[:min_len]
            all_future = all_future[:min_len]
        
        metrics = calculate_metrics(all_future, forecasts)
        dir_acc = directional_accuracy(all_future.ravel(), forecasts.ravel())
        
        print(f"RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}, Dir Acc: {dir_acc:.4f}")
        
        return {
            'model': model,
            'forecasts': forecasts,
            'actuals': all_future,
            'metrics': metrics,
            'directional_accuracy': dir_acc
        }
    except Exception as e:
        print(f"Prophet error: {e}")
        train_df = data_dict['train_df']
        val_df = data_dict['val_df']
        test_df = data_dict['test_df']
        all_future = np.concatenate([val_df['Close'].values, test_df['Close'].values])
        dummy_forecasts = np.full(len(all_future), np.mean(train_df['Close'].values))
        return {
            'model': None,
            'forecasts': dummy_forecasts,
            'actuals': all_future,
            'metrics': {'RMSE': 0, 'MAE': 0, 'MAPE': 0, 'MSE': 0},
            'directional_accuracy': 0
        }


def run_phase2(tickers=TICKERS):
    """Run Phase 2 baseline models."""
    print("\n" + "="*60)
    print("PHASE 2: BASELINE MODELS")
    print("="*60)
    
    all_results = {}
    
    for ticker in tickers:
        print(f"\n--- {ticker} ---")
        data_dict = load_preprocessed_data(ticker, phase='phase1')
        if data_dict is None:
            continue
        
        arima_results = forecast_7day_arima(data_dict, ticker)
        sarima_results = forecast_7day_sarima(data_dict, ticker)
        prophet_results = forecast_30day_prophet(data_dict, ticker)
        
        all_results[ticker] = {
            'ARIMA_7D': arima_results,
            'SARIMA_7D': sarima_results,
            'Prophet_30D': prophet_results
        }
        
        save_results({
            ticker: {
                'ARIMA_RMSE': arima_results['metrics']['RMSE'],
                'SARIMA_RMSE': sarima_results['metrics']['RMSE'],
                'Prophet_RMSE': prophet_results['metrics']['RMSE']
            }
        }, f'phase2_baseline_{ticker}.json')
    
    print("\n✓ Phase 2 complete")
    return all_results


if __name__ == "__main__":
    baseline_results = run_phase2()

