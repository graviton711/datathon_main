import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(y_true, y_pred):
    """
    Calculates standard regression metrics: MAE, RMSE, R2.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }

def calculate_bias(y_true, y_pred):
    """
    Calculates Bias Ratio: Sum(Predictions) / Sum(Actuals).
    > 1 : Over-predicting (Optimistic)
    < 1 : Under-predicting (Pessimistic)
    """
    sum_true = np.sum(y_true)
    sum_pred = np.sum(y_pred)
    
    if sum_true == 0:
        return np.nan
        
    return sum_pred / sum_true

def evaluate_predictions(y_true, y_pred):
    """Returns a comprehensive dictionary of all metrics."""
    metrics = calculate_metrics(y_true, y_pred)
    metrics['Bias_Ratio'] = calculate_bias(y_true, y_pred)
    return metrics
