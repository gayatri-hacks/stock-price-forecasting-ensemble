"""Ensemble model combining all forecasts with learned weights."""

import numpy as np
import pandas as pd
from utils.helpers import calculate_metrics, directional_accuracy, save_results


class EnsembleForecaster:
    """Combine multiple forecasts with weighted averaging."""
    
    def __init__(self, weights=None):
        """
        Initialize ensemble with optional custom weights.
        
        Default weights:
        - ARIMA (0.15): Classical short-term
        - SARIMA (0.15): Classical seasonal short-term
        - Prophet (0.20): Classical long-term
        - LSTM (0.25): Deep learning short-term
        - Transformer (0.25): Deep learning long-term
        """
        if weights is None:
            self.weights = {
                'ARIMA_7D': 0.15,
                'SARIMA_7D': 0.15,
                'Prophet_30D': 0.20,
                'LSTM_7D': 0.25,
                'Transformer_30D': 0.25
            }
        else:
            self.weights = weights
        
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
        print(f"Ensemble weights: {self.weights}")
    
    def combine_forecasts(self, baseline_results, dl_results, ticker):
        """
        Combine forecasts from all models.
        
        Args:
            baseline_results: Dict with ARIMA_7D, SARIMA_7D, Prophet_30D
            dl_results: Dict with LSTM_7D, Transformer_30D
            ticker: Stock ticker
        
        Returns:
            Dict with ensemble forecasts and metrics
        """
        forecasts_list = []
        actuals_list = []
        
        # Collect all forecasts
        predictions = {}
        
        # ARIMA 7-day
        if 'ARIMA_7D' in baseline_results and baseline_results['ARIMA_7D'] is not None:
            arima = baseline_results['ARIMA_7D']
            predictions['ARIMA_7D'] = arima['forecasts']
            actuals = arima['actuals']
        
        # SARIMA 7-day
        if 'SARIMA_7D' in baseline_results and baseline_results['SARIMA_7D'] is not None:
            sarima = baseline_results['SARIMA_7D']
            if sarima['model'] is not None:  # Only if successfully trained
                predictions['SARIMA_7D'] = sarima['forecasts']
                actuals = sarima['actuals']
        
        # Prophet 30-day
        if 'Prophet_30D' in baseline_results and baseline_results['Prophet_30D'] is not None:
            prophet = baseline_results['Prophet_30D']
            predictions['Prophet_30D'] = prophet['forecasts']
            actuals = prophet['actuals']
        
        # LSTM 7-day
        if 'LSTM_7D' in dl_results and dl_results['LSTM_7D'] is not None:
            lstm = dl_results['LSTM_7D']
            predictions['LSTM_7D'] = lstm['predictions'][:, 0]
            actuals = lstm['actuals'][:, 0]
        
        # Transformer 30-day
        if 'Transformer_30D' in dl_results and dl_results['Transformer_30D'] is not None:
            transformer = dl_results['Transformer_30D']
            predictions['Transformer_30D'] = transformer['predictions'][:, 0]
            actuals = transformer['actuals'][:, 0]
        
        # Align all to same length
        min_len = min([len(p) for p in predictions.values()] + [len(actuals)])
        
        predictions = {k: v[:min_len] for k, v in predictions.items()}
        actuals = actuals[:min_len]
        
        # Weighted average
        ensemble_forecast = np.zeros(len(actuals))
        total_weight = 0
        
        for model_name, forecast in predictions.items():
            if model_name in self.weights:
                weight = self.weights[model_name]
                ensemble_forecast += weight * forecast
                total_weight += weight
        
        # Normalize by sum of applicable weights
        if total_weight > 0:
            ensemble_forecast = ensemble_forecast / total_weight
        
        # Calculate metrics
        metrics = calculate_metrics(actuals, ensemble_forecast)
        dir_acc = directional_accuracy(actuals, ensemble_forecast)
        
        print(f"\n=== ENSEMBLE FORECAST FOR {ticker} ===")
        print(f"RMSE: {metrics['RMSE']:.4f}")
        print(f"MAE: {metrics['MAE']:.4f}")
        print(f"MAPE: {metrics['MAPE']:.4f}")
        print(f"Directional Accuracy: {dir_acc:.4f}")
        
        return {
            'ticker': ticker,
            'ensemble_forecast': ensemble_forecast,
            'actuals': actuals,
            'individual_predictions': predictions,
            'metrics': metrics,
            'directional_accuracy': dir_acc
        }
    
    def optimize_weights(self, baseline_results, dl_results, ticker, metric='RMSE'):
        """
        Optimize ensemble weights using grid search on validation set.
        
        This is computationally expensive but finds better weight combinations.
        """
        from itertools import combinations
        
        print(f"\nOptimizing ensemble weights for {ticker}...")
        
        # Get all predictions
        predictions = {}
        
        if 'ARIMA_7D' in baseline_results:
            predictions['ARIMA_7D'] = baseline_results['ARIMA_7D']['forecasts']
            actuals = baseline_results['ARIMA_7D']['actuals']
        
        if 'SARIMA_7D' in baseline_results and baseline_results['SARIMA_7D']['model'] is not None:
            predictions['SARIMA_7D'] = baseline_results['SARIMA_7D']['forecasts']
        
        if 'Prophet_30D' in baseline_results:
            predictions['Prophet_30D'] = baseline_results['Prophet_30D']['forecasts']
        
        if 'LSTM_7D' in dl_results:
            predictions['LSTM_7D'] = dl_results['LSTM_7D']['predictions'][:, 0]
        
        if 'Transformer_30D' in dl_results:
            predictions['Transformer_30D'] = dl_results['Transformer_30D']['predictions'][:, 0]
        
        best_score = float('inf') if metric == 'RMSE' else 0
        best_weights = self.weights
        
        # Test different weight combinations
        model_names = list(predictions.keys())
        for weights_idx in range(10):  # Limited iterations for speed
            test_weights = {m: np.random.rand() for m in model_names}
            total = sum(test_weights.values())
            test_weights = {k: v/total for k, v in test_weights.items()}
            
            # Align lengths
            min_len = min([len(p) for p in predictions.values()] + [len(actuals)])
            aligned_preds = {k: v[:min_len] for k, v in predictions.items()}
            aligned_actuals = actuals[:min_len]
            
            # Calculate weighted forecast
            weighted_forecast = sum(
                test_weights.get(m, 0) * aligned_preds[m]
                for m in aligned_preds.keys()
            )
            
            # Evaluate
            metrics = calculate_metrics(aligned_actuals, weighted_forecast)
            score = metrics[metric]
            
            # Update best if better
            if metric == 'RMSE' and score < best_score:
                best_score = score
                best_weights = test_weights
            elif metric != 'RMSE' and score > best_score:
                best_score = score
                best_weights = test_weights
        
        self.weights = best_weights
        print(f"Optimized weights: {self.weights}")
        return best_weights


def ensemble_backtest(ensemble_results, initial_capital=10000):
    """Backtest ensemble forecasts."""
    from utils.config import SLIPPAGE_RATE, TRANSACTION_COST
    from utils.helpers import sharpe_ratio
    
    ensemble_forecast = ensemble_results['ensemble_forecast']
    actuals = ensemble_results['actuals']
    
    # Generate simple signals
    signals = np.zeros(len(ensemble_forecast))
    for i in range(len(ensemble_forecast)):
        if ensemble_forecast[i] > actuals[i] * 1.005:
            signals[i] = 1  # Buy
        elif ensemble_forecast[i] < actuals[i] * 0.995:
            signals[i] = -1  # Sell
    
    # Simple portfolio simulation
    portfolio_value = [initial_capital]
    position = 0
    
    for i in range(len(signals) - 1):
        current_price = actuals[i]
        next_price = actuals[i + 1]
        signal = signals[i]
        
        if signal == 1 and position == 0:
            position = 1
            entry_price = current_price * (1 + SLIPPAGE_RATE)
        elif signal == -1 and position == 1:
            exit_price = current_price * (1 - SLIPPAGE_RATE)
            pnl = (exit_price - entry_price) * 100 - TRANSACTION_COST
            portfolio_value.append(portfolio_value[-1] + pnl)
            position = 0
            continue
        
        if position == 1:
            unrealized = (current_price - entry_price) * 100
            portfolio_value.append(portfolio_value[-1] + unrealized)
        else:
            portfolio_value.append(portfolio_value[-1])
    
    total_return = (portfolio_value[-1] - initial_capital) / initial_capital
    returns = np.diff(portfolio_value) / portfolio_value[:-1]
    returns = returns[returns != 0] if len(returns[returns != 0]) > 0 else returns
    
    try:
        sharpe = sharpe_ratio(returns)
    except:
        sharpe = 0
    
    return {
        'portfolio_value': portfolio_value,
        'total_return': total_return,
        'sharpe_ratio': sharpe
    }


if __name__ == "__main__":
    print("Ensemble module loaded successfully")
