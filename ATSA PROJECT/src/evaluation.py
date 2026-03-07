"""
Evaluation metrics and functions for stock forecasting
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate various forecasting metrics"""
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Percentage Error
        MAPE = 100 * mean(|y_true - y_pred| / |y_true|)
        
        Good for percentage errors, but problematic when y_true ≈ 0
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Avoid division by zero
        mask = y_true != 0
        if mask.sum() == 0:
            return np.nan
        
        ape = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
        return 100.0 * np.mean(ape)
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error"""
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Symmetric Mean Absolute Percentage Error
        Avoid issues with MAPE when true values are near zero
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        numerator = 2 * np.abs(y_pred - y_true)
        denominator = np.abs(y_true) + np.abs(y_pred)
        
        mask = denominator != 0
        return 100.0 * np.mean(numerator[mask] / denominator[mask])
    
    @staticmethod
    def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Percentage of correct directional predictions (up/down)
        DA = sum(sign(y_true) == sign(y_pred)) / len(y_true)
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        correct = np.sign(y_true) == np.sign(y_pred)
        return 100.0 * np.sum(correct) / len(y_true)
    
    @staticmethod
    def hit_rate(y_true: np.ndarray, y_pred: np.ndarray, tolerance: float = 0.02) -> float:
        """
        Percentage of predictions within tolerance of actual
        Default: within ±2% of actual value
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        hits = np.abs(y_pred - y_true) <= tolerance * np.abs(y_true)
        return 100.0 * np.sum(hits) / len(y_true)
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error"""
        return mean_squared_error(y_true, y_pred)
    
    @staticmethod
    def corr_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Pearson correlation coefficient between true and predicted"""
        return np.corrcoef(y_true, y_pred)[0, 1]


class ForecastEvaluator:
    """Comprehensive forecast evaluation"""
    
    def __init__(self):
        self.metrics_calc = MetricsCalculator()
    
    def evaluate_forecast(self, 
                         y_true: np.ndarray,
                         y_pred: np.ndarray,
                         model_name: str = "Model") -> Dict[str, float]:
        """
        Evaluate a single forecast with all metrics
        
        Parameters:
        -----------
        y_true : np.ndarray
            Actual values
        y_pred : np.ndarray
            Predicted values
        model_name : str
            Name of the model
        
        Returns:
        --------
        metrics_dict : Dict
            All computed metrics
        """
        
        results = {
            'model': model_name,
            'mape': self.metrics_calc.mape(y_true, y_pred),
            'rmse': self.metrics_calc.rmse(y_true, y_pred),
            'mae': self.metrics_calc.mae(y_true, y_pred),
            'smape': self.metrics_calc.smape(y_true, y_pred),
            'mse': self.metrics_calc.mse(y_true, y_pred),
            'directional_accuracy': self.metrics_calc.directional_accuracy(y_true, y_pred),
            'hit_rate': self.metrics_calc.hit_rate(y_true, y_pred),
            'correlation': self.metrics_calc.corr_coefficient(y_true, y_pred),
        }
        
        logger.info(f"{model_name} - MAPE: {results['mape']:.2f}%, "
                   f"RMSE: {results['rmse']:.4f}, "
                   f"DA: {results['directional_accuracy']:.2f}%")
        
        return results
    
    def compare_models(self,
                      predictions_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
                      horizon: str = "7-day") -> pd.DataFrame:
        """
        Compare multiple models
        
        Parameters:
        -----------
        predictions_dict : Dict
            {model_name: (y_true, y_pred)}
        horizon : str
            Forecast horizon for reporting
        
        Returns:
        --------
        comparison_df : pd.DataFrame
            Comparison table
        """
        
        results_list = []
        
        for model_name, (y_true, y_pred) in predictions_dict.items():
            metrics = self.evaluate_forecast(y_true, y_pred, model_name)
            results_list.append(metrics)
        
        df = pd.DataFrame(results_list)
        df['horizon'] = horizon
        
        # Sort by MAPE (ascending = better)
        df = df.sort_values('mape')
        
        logger.info(f"\n{'='*80}\n{horizon} Forecast Evaluation\n{'='*80}")
        logger.info(df.to_string(index=False))
        
        return df
    
    def statistical_significance_test(self,
                                      y_true: np.ndarray,
                                      y_pred1: np.ndarray,
                                      y_pred2: np.ndarray,
                                      model1_name: str = "Model 1",
                                      model2_name: str = "Model 2") -> Dict:
        """
        Diebold-Mariano test for forecast accuracy comparison
        
        H0: Both forecasts have equal accuracy
        H1: Forecasts have different accuracy
        """
        
        from scipy import stats
        
        # Compute forecast errors
        e1 = y_true - y_pred1
        e2 = y_true - y_pred2
        
        # Loss differential (using MSE)
        d = e1**2 - e2**2
        
        # DM test statistic
        mean_d = np.mean(d)
        var_d = np.var(d, ddof=1)
        dm_stat = mean_d / np.sqrt(var_d / len(d))
        
        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
        
        results = {
            'model1': model1_name,
            'model2': model2_name,
            'dm_statistic': dm_stat,
            'p_value': p_value,
            'significant_at_0.05': p_value < 0.05,
            'interpretation': f"{model1_name} is better" if mean_d > 0 else f"{model2_name} is better"
        }
        
        logger.info(f"\nDiebold-Mariano Test:")
        logger.info(f"{model1_name} vs {model2_name}")
        logger.info(f"DM Statistic: {dm_stat:.4f}, p-value: {p_value:.4f}")
        logger.info(f"Result: {results['interpretation']} (p < 0.05: {results['significant_at_0.05']})")
        
        return results
    
    def forecast_skill(self,
                      y_true: np.ndarray,
                      y_pred: np.ndarray,
                      baseline_pred: np.ndarray) -> float:
        """
        Calculate forecast skill (improvement over baseline)
        Skill = 1 - (MSE_model / MSE_baseline)
        
        Skill > 0: Model better than baseline
        Skill = 0: Model equals baseline
        Skill < 0: Model worse than baseline
        """
        
        mse_model = mean_squared_error(y_true, y_pred)
        mse_baseline = mean_squared_error(y_true, baseline_pred)
        
        if mse_baseline == 0:
            return np.nan
        
        skill = 1 - (mse_model / mse_baseline)
        return skill


def create_evaluation_report(results_dict: Dict) -> pd.DataFrame:
    """
    Create comprehensive evaluation report
    
    Parameters:
    -----------
    results_dict : Dict
        {horizon: {model: metrics}}
    
    Returns:
    --------
    report_df : pd.DataFrame
    """
    
    rows = []
    
    for horizon, models in results_dict.items():
        for model_name, metrics in models.items():
            row = {
                'Horizon': horizon,
                'Model': model_name,
                **metrics
            }
            rows.append(row)
    
    report_df = pd.DataFrame(rows)
    
    return report_df


def rank_models(evaluation_df: pd.DataFrame,
               primary_metric: str = 'mape',
               secondary_metric: str = 'rmse') -> pd.DataFrame:
    """
    Rank models based on metrics
    
    Parameters:
    -----------
    evaluation_df : pd.DataFrame
        Evaluation results
    primary_metric : str
        Main ranking metric
    secondary_metric : str
        Secondary tiebreaker
    
    Returns:
    --------
    ranked_df : pd.DataFrame
        Sorted by rank
    """
    
    # Sort by primary then secondary
    ranked = evaluation_df.sort_values([primary_metric, secondary_metric])
    ranked['rank'] = range(1, len(ranked) + 1)
    
    return ranked


if __name__ == "__main__":
    # Example usage
    y_true = np.array([100, 102, 105, 103, 104])
    y_pred = np.array([101, 103, 104, 102, 105])
    
    evaluator = ForecastEvaluator()
    metrics = evaluator.evaluate_forecast(y_true, y_pred, "Test Model")
    
    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
