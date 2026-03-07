"""Helper functions for the project."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import json
import pickle
import warnings


def ensure_directories():
    """Ensure all output directories exist."""
    Path('./data').mkdir(exist_ok=True)
    Path('./models').mkdir(exist_ok=True)
    Path('./outputs').mkdir(exist_ok=True)


def save_preprocessed_data(data_dict, ticker, phase='phase1'):
    """Save preprocessed data to disk."""
    ensure_directories()
    path = Path('./data') / f'{ticker}_{phase}.pkl'
    pd.to_pickle(data_dict, path)
    print(f"Saved data for {ticker} to {path}")


def load_preprocessed_data(ticker, phase='phase1'):
    """Load preprocessed data from disk."""
    path = Path('./data') / f'{ticker}_{phase}.pkl'
    if path.exists():
        return pd.read_pickle(path)
    return None


def calculate_metrics(y_true, y_pred):
    """Calculate common error metrics."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }


def directional_accuracy(y_true, y_pred):
    """Calculate if the model got the direction (up/down) correct."""
    # Ensure arrays are flattened and proper length
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if len(y_true) < 2 or len(y_pred) < 2:
        return 0.5
    
    # Ensure same length
    if len(y_true) != len(y_pred):
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
    
    # Take differences to get direction
    true_direction = np.diff(y_true) > 0
    pred_direction = np.diff(y_pred) > 0
    
    # Both should now be same length
    if len(true_direction) != len(pred_direction):
        min_len = min(len(true_direction), len(pred_direction))
        true_direction = true_direction[:min_len]
        pred_direction = pred_direction[:min_len]
    
    if len(true_direction) == 0:
        return 0.5
    
    accuracy = np.mean(true_direction == pred_direction)
    return accuracy


def sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate Sharpe ratio."""
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)


def save_results(results_dict, filename):
    """Save results to JSON, converting numpy types to native Python types."""
    import numpy as np
    
    def convert_to_native_types(obj):
        """Recursively convert numpy/tensorflow types to native Python types."""
        if isinstance(obj, dict):
            return {k: convert_to_native_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native_types(item) for item in obj]
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return convert_to_native_types(obj.tolist())
        else:
            return obj
    
    ensure_directories()
    path = Path('./outputs') / filename
    converted_dict = convert_to_native_types(results_dict)
    with open(path, 'w') as f:
        json.dump(converted_dict, f, indent=4)
    print(f"Saved results to {path}")


def load_results(filename):
    """Load results from JSON."""
    path = Path('./outputs') / filename
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return None


# ============================================================================
# MODEL PERSISTENCE (Save/Load trained models)
# ============================================================================

def save_model(model, model_name, model_type='tensorflow'):
    """Save trained model to disk."""
    ensure_directories()
    path = Path('./models') / f'{model_name}.pkl'
    try:
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ Model saved: {path}")
    except Exception as e:
        print(f"✗ Error saving model: {e}")


def load_model(model_name):
    """Load trained model from disk."""
    path = Path('./models') / f'{model_name}.pkl'
    if path.exists():
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
            print(f"✓ Model loaded: {path}")
            return model
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return None
    else:
        print(f"Model not found: {path}")
        return None


# ============================================================================
# STATISTICAL TESTS
# ============================================================================

def diebold_mariano_test(errors1, errors2, h=1):
    """
    Diebold-Mariano test for comparing forecast accuracy.
    H0: Both forecasts have equal accuracy
    If p-value < 0.05, reject H0 (one is significantly better)
    """
    from scipy import stats
    
    errors1 = np.asarray(errors1).flatten()
    errors2 = np.asarray(errors2).flatten()
    
    # Make same length
    min_len = min(len(errors1), len(errors2))
    errors1 = errors1[:min_len]
    errors2 = errors2[:min_len]
    
    # DM test statistic
    d = errors1**2 - errors2**2
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    
    if var_d == 0:
        dm_stat = 0
        p_value = 1.0
    else:
        dm_stat = mean_d / np.sqrt(var_d / len(d))
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    
    return {'dm_statistic': dm_stat, 'p_value': p_value, 'significant': p_value < 0.05}


def confidence_interval(y_true, y_pred, confidence=0.95):
    """Calculate confidence interval for forecast errors."""
    from scipy import stats
    
    errors = np.asarray(y_true).flatten() - np.asarray(y_pred).flatten()
    mean_error = np.mean(errors)
    std_error = np.std(errors, ddof=1)
    n = len(errors)
    
    margin = std_error * stats.t.ppf((1 + confidence) / 2, n - 1) / np.sqrt(n)
    
    return {
        'mean_error': mean_error,
        'margin': margin,
        'lower_bound': mean_error - margin,
        'upper_bound': mean_error + margin,
        'confidence': confidence
    }


def statistical_comparison(y_true, predictions_dict):
    """
    Compare multiple models statistically.
    Returns pairwise Diebold-Mariano test results.
    """
    results = {}
    models = list(predictions_dict.keys())
    
    for i, model1 in enumerate(models):
        for model2 in models[i+1:]:
            errors1 = np.asarray(y_true).flatten() - np.asarray(predictions_dict[model1]).flatten()
            errors2 = np.asarray(y_true).flatten() - np.asarray(predictions_dict[model2]).flatten()
            
            dm_result = diebold_mariano_test(errors1, errors2)
            results[f'{model1}_vs_{model2}'] = dm_result
    
    return results


# ============================================================================
# DATA VALIDATION
# ============================================================================

def validate_data(df, min_rows=100):
    """Validate data quality before processing."""
    warnings_list = []
    
    # Check for missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        warnings_list.append(f"⚠ Missing values found: {missing}")
    
    # Check for negative prices
    if (df['Close'] < 0).any():
        warnings_list.append("⚠ Negative prices detected")
    
    # Check minimum length
    if len(df) < min_rows:
        warnings_list.append(f"⚠ Insufficient data: {len(df)} rows < {min_rows} required")
    
    # Check for duplicates
    if df.index.duplicated().any():
        warnings_list.append("⚠ Duplicate timestamps found")
    
    # Check chronological order
    if not df.index.is_monotonic_increasing:
        warnings_list.append("⚠ Data not in chronological order")
    
    if warnings_list:
        print("\nData Validation Warnings:")
        for w in warnings_list:
            print(f"  {w}")
        return False
    else:
        print("✓ Data validation passed")
        return True


def validate_predictions(y_true, y_pred, min_samples=10):
    """Validate prediction arrays."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    issues = []
    
    if len(y_true) < min_samples:
        issues.append(f"Insufficient samples: {len(y_true)}")
    
    if len(y_true) != len(y_pred):
        issues.append(f"Length mismatch: {len(y_true)} vs {len(y_pred)}")
    
    if np.isnan(y_pred).any():
        issues.append("NaN values in predictions")
    
    if np.isinf(y_pred).any():
        issues.append("Infinite values in predictions")
    
    if issues:
        print("⚠ Prediction validation warnings:")
        for iss in issues:
            print(f"  - {iss}")
        return False
    
    return True


# ============================================================================
# HYPERPARAMETER TUNING UTILITIES
# ============================================================================

def grid_search_params(param_grid):
    """Generate all combinations of hyperparameters."""
    from itertools import product
    
    param_names = list(param_grid.keys())
    param_values = [param_grid[p] for p in param_names]
    
    combinations = []
    for vals in product(*param_values):
        param_dict = dict(zip(param_names, vals))
        combinations.append(param_dict)
    
    return combinations


def log_hyperparameters(params, metrics, filename='hyperparameter_log.json'):
    """Log hyperparameter experiments for tracking."""
    ensure_directories()
    path = Path('./outputs') / filename
    
    log_entry = {
        'params': params,
        'metrics': metrics,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    logs = []
    if path.exists():
        with open(path, 'r') as f:
            logs = json.load(f)
    
    logs.append(log_entry)
    
    with open(path, 'w') as f:
        json.dump(logs, f, indent=4)


# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

def feature_correlation(X, y, feature_names=None):
    """Calculate correlation between features and target."""
    X_array = np.asarray(X)
    y_array = np.asarray(y).flatten()
    
    correlations = {}
    for i in range(X_array.shape[1]):
        if feature_names:
            fname = feature_names[i]
        else:
            fname = f'Feature_{i}'
        
        corr = np.corrcoef(X_array[:, i], y_array)[0, 1]
        correlations[fname] = corr
    
    return correlations


# ============================================================================
# WALK-FORWARD BACKTESTING UTILITIES
# ============================================================================

def walk_forward_split(X, y, train_size, step_size=1):
    """
    Generate walk-forward train/test splits for time series.
    
    Args:
        X, y: Full dataset
        train_size: Training period length
        step_size: Steps to move forward each iteration
    
    Yields:
        (X_train, X_test, y_train, y_test)
    """
    for t in range(train_size, len(X), step_size):
        X_train, X_test = X[:t], X[t:t+step_size]
        y_train, y_test = y[:t], y[t:t+step_size]
        
        if len(X_test) > 0:
            yield X_train, X_test, y_train, y_test
