"""Unit tests for ATSA project components."""

import numpy as np
import pandas as pd
from pathlib import Path


def test_calculate_metrics():
    """Test metric calculation."""
    from utils.helpers import calculate_metrics
    
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])
    
    metrics = calculate_metrics(y_true, y_pred)
    
    assert 'RMSE' in metrics
    assert 'MAE' in metrics
    assert 'MAPE' in metrics
    assert metrics['RMSE'] < 0.2, "RMSE should be less than 0.2"
    assert metrics['MAE'] < 0.2, "MAE should be less than 0.2"
    
    print("✓ test_calculate_metrics passed")


def test_directional_accuracy():
    """Test directional accuracy calculation."""
    from utils.helpers import directional_accuracy
    
    # Perfect predictions
    y_true = np.array([1, 2, 3, 2, 1])
    y_pred = np.array([1.1, 2.1, 3.1, 2.1, 1.1])
    acc = directional_accuracy(y_true, y_pred)
    
    assert 0 <= acc <= 1, "Accuracy should be between 0 and 1"
    assert acc > 0.5, "Should have decent accuracy on similar values"
    
    print("✓ test_directional_accuracy passed")


def test_sharpe_ratio():
    """Test Sharpe ratio calculation."""
    from utils.helpers import sharpe_ratio
    
    returns = np.array([0.01, 0.02, -0.01, 0.03, 0.02])
    sharpe = sharpe_ratio(returns)
    
    assert isinstance(sharpe, (int, float, np.number)), "Sharpe should be a number"
    
    print("✓ test_sharpe_ratio passed")


def test_data_validation():
    """Test data validation function."""
    from utils.helpers import validate_data
    
    # Valid data
    df = pd.DataFrame({
        'Close': np.linspace(100, 150, 200),
        'Volume': np.random.rand(200) * 1000000
    })
    df.index = pd.date_range('2020-01-01', periods=200)
    
    result = validate_data(df)
    assert result == True, "Valid data should pass validation"
    
    print("✓ test_data_validation passed")


def test_diebold_mariano():
    """Test Diebold-Mariano statistical test."""
    from utils.helpers import diebold_mariano_test
    
    errors1 = np.array([0.1, 0.15, 0.12, 0.08, 0.11])
    errors2 = np.array([0.2, 0.25, 0.22, 0.18, 0.21])  # Larger errors
    
    result = diebold_mariano_test(errors1, errors2)
    
    assert 'dm_statistic' in result
    assert 'p_value' in result
    assert 'significant' in result
    
    print("✓ test_diebold_mariano passed")


def test_model_persistence():
    """Test model save/load functionality."""
    from utils.helpers import save_model, load_model
    
    # Create a simple model dict
    test_model = {'weights': [0.1, 0.2, 0.3], 'name': 'test'}
    
    save_model(test_model, 'test_model_temporary')
    loaded_model = load_model('test_model_temporary')
    
    assert loaded_model == test_model, "Loaded model should match saved model"
    
    # Cleanup
    model_path = Path('./models/test_model_temporary.pkl')
    if model_path.exists():
        model_path.unlink()
    
    print("✓ test_model_persistence passed")


def test_ensemble_weights():
    """Test ensemble weighting logic."""
    from ensemble import EnsembleForecaster
    
    ensemble = EnsembleForecaster()
    
    # Weights should sum to 1
    total_weight = sum(ensemble.weights.values())
    assert abs(total_weight - 1.0) < 0.001, "Weights should sum to 1"
    
    print("✓ test_ensemble_weights passed")


def test_walk_forward_generator():
    """Test walk-forward data splitting."""
    from utils.helpers import walk_forward_split
    
    X = np.arange(100).reshape(-1, 1)
    y = np.arange(100)
    train_size = 70
    
    splits = list(walk_forward_split(X, y, train_size, step_size=10))
    
    assert len(splits) > 0, "Should generate at least one split"
    assert splits[0][0].shape[0] == 70, "Training set should be 70 samples"
    
    print("✓ test_walk_forward_generator passed")


def test_grid_search_params():
    """Test hyperparameter grid generation."""
    from utils.helpers import grid_search_params
    
    param_grid = {
        'layer1': [32, 64],
        'layer2': [16, 32],
        'lr': [0.001, 0.01]
    }
    
    combos = grid_search_params(param_grid)
    
    expected = 2 * 2 * 2  # 8 combinations
    assert len(combos) == expected, f"Should generate {expected} combinations"
    
    print("✓ test_grid_search_params passed")


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "="*60)
    print("RUNNING UNIT TESTS")
    print("="*60 + "\n")
    
    tests = [
        test_calculate_metrics,
        test_directional_accuracy,
        test_sharpe_ratio,
        test_data_validation,
        test_diebold_mariano,
        test_model_persistence,
        test_ensemble_weights,
        test_walk_forward_generator,
        test_grid_search_params
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} ERROR: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"TESTS COMPLETE: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
