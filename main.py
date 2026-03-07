"""Main orchestration script - Runs all phases in sequence with ensemble & validation."""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

from phase1_data_foundation.data_foundation import run_phase1
from phase2_baseline_models.baseline_models import run_phase2
from phase3_deep_learning.deep_learning import run_phase3
from phase4_backtesting.backtesting import run_phase4
from phase5_visualization.visualization import run_phase5, plot_attention_heatmap
from ensemble import EnsembleForecaster, ensemble_backtest
from utils.config import TICKERS
from utils.helpers import ensure_directories, validate_data, statistical_comparison


def main():
    """Run the complete ATSA pipeline."""
    
    print("\n" + "="*60)
    print("ADVANCED TIME SERIES ANALYSIS (ATSA) PROJECT")
    print("Faculty Presentation Pipeline")
    print("="*60)
    
    ensure_directories()
    
    try:
        # Phase 1: Data Foundation
        print("\n>>> Starting Phase 1: Data Foundation...")
        phase1_data = run_phase1(TICKERS)
        print("✓ Phase 1 complete")
        
        # Phase 2: Baseline Models
        print("\n>>> Starting Phase 2: Baseline Models (including SARIMA)...")
        baseline_results = run_phase2(TICKERS)
        print("✓ Phase 2 complete")
        
        # Phase 3: Deep Learning Models
        print("\n>>> Starting Phase 3: Deep Learning Models...")
        dl_results = run_phase3(TICKERS)
        print("✓ Phase 3 complete")
        
        # Phase 4: Backtesting
        print("\n>>> Starting Phase 4: Backtesting...")
        backtest_results = run_phase4(baseline_results, dl_results, TICKERS)
        print("✓ Phase 4 complete")
        
        # ENSEMBLE: Combine all model predictions
        print("\n>>> Creating Ensemble Models...")
        ensemble_forecaster = EnsembleForecaster()
        ensemble_results = {}
        for ticker in TICKERS:
            try:
                if ticker in baseline_results and ticker in dl_results:
                    ensemble_result = ensemble_forecaster.combine_forecasts(
                        baseline_results[ticker],
                        dl_results[ticker],
                        ticker
                    )
                    ensemble_results[ticker] = ensemble_result
            except Exception as e:
                print(f"⚠ Ensemble failed for {ticker}: {e}")
        print("✓ Ensemble complete")
        
        # STATISTICAL TESTS: Compare model accuracy
        print("\n>>> Running Statistical Comparisons...")
        for ticker in TICKERS:
            try:
                if ticker in baseline_results:
                    predictions = {
                        'ARIMA': baseline_results[ticker].get('ARIMA_7D', {}).get('forecasts', []),
                        'Prophet': baseline_results[ticker].get('Prophet_30D', {}).get('forecasts', [])
                    }
                    if predictions['ARIMA'] is not None and len(predictions['ARIMA']) > 0:
                        actuals = baseline_results[ticker]['ARIMA_7D']['actuals']
                        comparison = statistical_comparison(actuals, predictions)
                        print(f"\n{ticker} Diebold-Mariano Test:")
                        for model_pair, result in comparison.items():
                            sig = "✓ Significant" if result['significant'] else "✗ Not significant"
                            print(f"  {model_pair}: p-value={result['p_value']:.4f} ({sig})")
            except Exception as e:
                print(f"⚠ Statistical test failed for {ticker}: {e}")
        print("✓ Statistical tests complete")
        
        # Phase 5: Visualization & Explainability
        print("\n>>> Starting Phase 5: Visualization & Explainability...")
        visualization_results = run_phase5(backtest_results, baseline_results, dl_results, TICKERS)
        
        # ATTENTION VISUALIZATION: Plot Transformer attention heatmaps
        print("\nGenerating attention heatmaps...")
        for ticker in TICKERS:
            try:
                if ticker in dl_results and 'Transformer_30D' in dl_results[ticker]:
                    transformer = dl_results[ticker]['Transformer_30D']
                    if hasattr(transformer['model'], 'attention_blocks'):
                        phase1_data_tuple = phase1_data.get(ticker, {})
                        X_test = phase1_data_tuple.get('X_test', None)
                        if X_test is not None:
                            plot_attention_heatmap(transformer['model'], X_test, ticker, top_n_samples=2)
            except Exception as e:
                print(f"⚠ Attention visualization skipped for {ticker}: {e}")
        
        print("✓ Phase 5 complete")
        
        print("\n" + "="*60)
        print("ALL PHASES COMPLETE!")
        print("="*60)
        print("\nResults saved to:")
        print("  - Data: ./data/")
        print("  - Results: ./outputs/")
        print("\nKey outputs:")
        print("  - Performance Summary: ./outputs/performance_summary.csv")
        print("  - Master Results: ./outputs/master_results.csv")
        print("  - Model Leaderboard: ./outputs/model_comparison_leaderboard.png")
        print("  - Visualizations: ./outputs/*.png")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
