"""Walk-forward backtesting for time series models - More realistic evaluation."""

import numpy as np
import pandas as pd
from utils.helpers import calculate_metrics, directional_accuracy, sharpe_ratio, save_results
from utils.config import SLIPPAGE_RATE, TRANSACTION_COST


class WalkForwardBacktester:
    """
    Walk-forward backtesting rolls the training window forward through time.
    More realistic than single train/test split.
    """
    
    def __init__(self, train_size, step_size=20):
        """
        Args:
            train_size: Number of historical periods to train on
            step_size: How many periods to move forward each iteration (default: 20)
        """
        self.train_size = train_size
        self.step_size = step_size
    
    def walk_forward_test(self, predictions_list, actuals, model_name='Model'):
        """
        Evaluate model performance using walk-forward method.
        
        Args:
            predictions_list: List of prediction arrays from successive windows
            actuals: Full actual price series
            model_name: Name of the model for reporting
        
        Returns:
            Dictionary with walk-forward performance metrics
        """
        
        all_predictions = np.concatenate(predictions_list) if predictions_list else np.array([])
        
        # Align with actuals
        min_len = min(len(all_predictions), len(actuals))
        all_predictions = all_predictions[:min_len]
        actuals = actuals[:min_len]
        
        # Calculate metrics
        metrics = calculate_metrics(actuals, all_predictions)
        dir_acc = directional_accuracy(actuals, all_predictions)
        
        # Calculate returns
        returns = np.diff(actuals) / actuals[:-1]
        pred_returns = np.diff(all_predictions) / all_predictions[:-1]
        
        # Calculate correlation
        try:
            correlation = np.corrcoef(returns, pred_returns)[0, 1]
        except:
            correlation = 0
        
        return {
            'model': model_name,
            'num_windows': len(predictions_list),
            'total_periods': len(all_predictions),
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'MAPE': metrics['MAPE'],
            'directional_accuracy': dir_acc,
            'return_correlation': correlation
        }
    
    def walk_forward_trading_backtest(self, predictions_list, actuals, initial_capital=10000):
        """
        Simulate trading using walk-forward predictions.
        More realistic than single backtest.
        
        Returns:
            Trading performance metrics across all windows
        """
        
        all_predictions = np.concatenate(predictions_list) if predictions_list else np.array([])
        
        # Align lengths
        min_len = min(len(all_predictions), len(actuals))
        all_predictions = all_predictions[:min_len]
        actuals_aligned = actuals[:min_len]
        
        # Generate signals
        signals = np.zeros(len(all_predictions))
        for i in range(len(all_predictions)):
            if all_predictions[i] > actuals_aligned[i] * 1.005:
                signals[i] = 1  # Buy
            elif all_predictions[i] < actuals_aligned[i] * 0.995:
                signals[i] = -1  # Sell
        
        # Backtest
        portfolio_value = [initial_capital]
        position = 0
        entry_price = 0
        trades = 0
        
        for i in range(len(signals) - 1):
            current_price = actuals_aligned[i]
            signal = signals[i]
            
            if signal == 1 and position == 0:
                entry_price = current_price * (1 + SLIPPAGE_RATE)
                position = 1
                trades += 1
            
            elif signal == -1 and position == 1:
                exit_price = current_price * (1 - SLIPPAGE_RATE)
                pnl = (exit_price - entry_price) - TRANSACTION_COST
                portfolio_value.append(portfolio_value[-1] + pnl * 100)
                position = 0
                trades += 1
                continue
            
            if position == 1:
                unrealized = (current_price - entry_price) * 100
                portfolio_value.append(portfolio_value[-1] + unrealized)
            else:
                portfolio_value.append(portfolio_value[-1])
        
        # Calculate metrics
        portfolio_value = np.array(portfolio_value)
        total_return = (portfolio_value[-1] - initial_capital) / initial_capital
        
        # Max drawdown
        running_max = np.maximum.accumulate(portfolio_value)
        drawdown = (portfolio_value - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Sharpe ratio
        returns = np.diff(portfolio_value) / portfolio_value[:-1]
        returns = returns[returns != 0] if len(returns[returns != 0]) > 0 else returns
        try:
            sharpe = sharpe_ratio(returns)
        except:
            sharpe = 0
        
        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'num_trades': trades,
            'final_portfolio_value': portfolio_value[-1],
            'portfolio_values': portfolio_value
        }


def generate_walk_forward_summary(walk_forward_results, output_file='walk_forward_summary.csv'):
    """Create summary table of walk-forward results."""
    
    df = pd.DataFrame(walk_forward_results)
    
    # Save to CSV
    from pathlib import Path
    Path('./outputs').mkdir(exist_ok=True)
    df.to_csv(f'./outputs/{output_file}', index=False)
    
    print("\nWalk-Forward Testing Summary:")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    return df


if __name__ == "__main__":
    print("Walk-forward backtesting module loaded")
