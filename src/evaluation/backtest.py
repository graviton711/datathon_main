import pandas as pd
import numpy as np
from src.evaluation.metrics import evaluate_predictions
from src.utils.logger import setup_logger

logger = setup_logger("backtester")

class TimeSeriesBacktester:
    def __init__(self, date_col='Date', target_col='Revenue'):
        self.date_col = date_col
        self.target_col = target_col

    def get_folds(self, df):
        """
        Generates 2 specific evaluation folds simulating the 2023 forecasting task.
        Fold 1: Train (2012-07-04 -> 2021-12-31), Test (2022-01-01 -> 2022-06-30)
        Fold 2: Train (2012-07-04 -> 2022-06-30), Test (2022-07-01 -> 2022-12-31)
        """
        df = df.sort_values(self.date_col)
        
        folds = [
            {
                'train_end': pd.to_datetime('2021-12-31'),
                'test_start': pd.to_datetime('2022-01-01'),
                'test_end': pd.to_datetime('2022-06-30')
            },
            {
                'train_end': pd.to_datetime('2022-06-30'),
                'test_start': pd.to_datetime('2022-07-01'),
                'test_end': pd.to_datetime('2022-12-31')
            }
        ]
        
        for fold in folds:
            train_mask = df[self.date_col] <= fold['train_end']
            test_mask = (df[self.date_col] >= fold['test_start']) & (df[self.date_col] <= fold['test_end'])
            
            yield df[train_mask].copy(), df[test_mask].copy(), fold

    def run_eval(self, model_class, df, features, **model_kwargs):
        """
        Runs the backtesting framework using a provided modeling class/architecture.
        """
        logger.info(f"Starting Backtest on {len(df)} records using features: {features}")
        
        fold_results = []
        
        for fold_idx, (train_df, test_df, fold_info) in enumerate(self.get_folds(df)):
            logger.info(f"--- Fold {fold_idx + 1} ---")
            logger.info(f"Train: {train_df[self.date_col].min().date()} to {train_df[self.date_col].max().date()} ({len(train_df)} rows)")
            logger.info(f"Test:  {test_df[self.date_col].min().date()} to {test_df[self.date_col].max().date()} ({len(test_df)} rows)")
            
            X_train = train_df[features]
            y_train = train_df[self.target_col]
            
            X_test = test_df[features]
            y_test = test_df[self.target_col]
            
            # Initialize and train model
            model = model_class(**model_kwargs)
            model.train(X_train, y_train)
            
            # Predict
            preds = model.predict(X_test)
            test_df['Predictions'] = preds
            
            # Naive Baseline (Mean of Train Target) for comparison
            naive_pred = y_train.mean()
            naive_preds = np.full(len(y_test), naive_pred)
            
            # Evaluate
            metrics = evaluate_predictions(y_test, preds)
            naive_metrics = evaluate_predictions(y_test, naive_preds)
            
            logger.info(f"Model MAE: {metrics['MAE']:,.2f} | Naive MAE: {naive_metrics['MAE']:,.2f}")
            logger.info(f"Model Bias: {metrics['Bias_Ratio']:.3f} | Naive Bias: {naive_metrics['Bias_Ratio']:.3f}")
            
            fold_results.append({
                'fold': fold_idx + 1,
                'metrics': metrics,
                'naive_metrics': naive_metrics,
                'predictions': test_df[[self.date_col, self.target_col, 'Predictions']]
            })
            
        # Summary
        avg_mae = np.mean([res['metrics']['MAE'] for res in fold_results])
        avg_naive_mae = np.mean([res['naive_metrics']['MAE'] for res in fold_results])
        avg_bias = np.mean([res['metrics']['Bias_Ratio'] for res in fold_results])
        
        logger.info("=== Backtest Summary ===")
        logger.info(f"Average MAE: {avg_mae:,.2f}")
        logger.info(f"Average Naive MAE: {avg_naive_mae:,.2f}")
        logger.info(f"Average Bias Ratio: {avg_bias:.3f}")
        
        if avg_mae < avg_naive_mae:
            logger.info("-> MODEL BEATS NAIVE BASELINE. Good to proceed.")
        else:
            logger.warning("-> MODEL UNDERPERFORMS NAIVE BASELINE! Needs tuning or better features.")
            
        return fold_results
