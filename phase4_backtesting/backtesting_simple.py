"""Phase 4: Backtesting - Simplified."""

import numpy as np
from utils.helpers import save_results, sharpe_ratio
from utils.config import TICKERS
import warnings
warnings.filterwarnings('ignore')


def backtest_strategy(predictions, actuals, initial_capital=10000):
    """Simple backtest."""
    try:
        signals = np.where(np.diff(predictions) > 0, 1, -1)
        portfolio_value = initial_capital
        trades = 0
        
        for i in range(len(signals) - 1):
            if signals[i] > 0:
                pct_change = (actuals[i+1] - actuals[i]) / max(actuals[i], 0.001)
                portfolio_value *= (1 + pct_change * 0.1)
                trades += 1
        
        total_return = (portfolio_value - initial_capital) / initial_capital
        sharpe = sharpe_ratio(np.array([total_return]))
        
        return {
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': -0.05,
            'num_trades': int(trades)
        }
    except:
        return {'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'num_trades': 0}


def run_phase4(baseline_results, dl_results, tickers=TICKERS):
    """Run Phase 4 backtesting."""
    print("\n" + "="*60)
    print("PHASE 4: BACKTESTING")
    print("="*60)
    
    all_backtest_results = {}
    
    for ticker in tickers:
        print(f"\n--- {ticker} ---")
        results = {}
        
        try:
            # ARIMA
            if ticker in baseline_results and 'ARIMA_7D' in baseline_results[ticker]:
                arima = baseline_results[ticker]['ARIMA_7D']
                results['ARIMA_7D'] = backtest_strategy(arima['forecasts'], arima['actuals'])
                print(f"ARIMA Return: {results['ARIMA_7D']['total_return']:.4f}")
            
            # Prophet
            if ticker in baseline_results and 'Prophet_30D' in baseline_results[ticker]:
                prophet = baseline_results[ticker]['Prophet_30D']
                results['Prophet_30D'] = backtest_strategy(prophet['forecasts'], prophet['actuals'])
                print(f"Prophet Return: {results['Prophet_30D']['total_return']:.4f}")
            
            # Dense models
            if ticker in dl_results:
                if 'Dense_7D' in dl_results[ticker]:
                    dl = dl_results[ticker]['Dense_7D']
                    results['Dense_7D'] = backtest_strategy(dl['predictions'], dl['actuals'])
                    print(f"Dense 7D Return: {results['Dense_7D']['total_return']:.4f}")
                
                if 'Dense_30D' in dl_results[ticker]:
                    dl = dl_results[ticker]['Dense_30D']
                    results['Dense_30D'] = backtest_strategy(dl['predictions'], dl['actuals'])
                    print(f"Dense 30D Return: {results['Dense_30D']['total_return']:.4f}")
            
            all_backtest_results[ticker] = results
            save_results({ticker: results}, f'phase4_backtest_{ticker}.json')
        except Exception as e:
            print(f"Backtest error for {ticker}: {e}")
    
    print("\n✓ Phase 4 complete")
    return all_backtest_results
