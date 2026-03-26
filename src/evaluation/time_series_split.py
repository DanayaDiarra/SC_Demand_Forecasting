"""
Time series cross-validation for forecasting
With 1-day prediction horizon, we need to be careful about leakage
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from typing import Tuple, Dict

class ForecastingValidator:
    """
    Custom time series split for daily forecasting
    """
    
    def __init__(self, n_splits: int = 3, test_size_days: int = 7):
        """
        n_splits: number of cross-validation folds
        test_size_days: how many days to predict (1 for your case, but can be more)
        """
        self.n_splits = n_splits
        self.test_size_days = test_size_days
    
    def create_walk_forward_validation(self, 
                                       df: pd.DataFrame, 
                                       date_col: str = 'Date',
                                       store_col: str = 'Store') -> list:
        """
        Create walk-forward validation splits
        Essential for time series forecasting
        """
        unique_dates = sorted(df[date_col].unique())
        
        splits = []
        for i in range(self.n_splits):
            # Train: all data up to split point
            train_end_idx = -(self.test_size_days * (self.n_splits - i))
            train_dates = unique_dates[:train_end_idx]
            
            # Test: next test_size_days
            test_dates = unique_dates[train_end_idx:train_end_idx + self.test_size_days]
            
            splits.append({
                'train': df[df[date_col].isin(train_dates)],
                'test': df[df[date_col].isin(test_dates)],
                'train_dates': (train_dates[0], train_dates[-1]),
                'test_dates': (test_dates[0], test_dates[-1])
            })
        
        return splits
    
    def validate_no_leakage(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
        """
        Critical check: ensure no future data leaks into training
        """
        train_max_date = train_df['Date'].max()
        test_min_date = test_df['Date'].min()
        
        # For 1-day forecast, test should be exactly 1 day after training
        if (test_min_date - train_max_date).days != 1:
            print(f"Warning: Gap is {(test_min_date - train_max_date).days} days")
            return False
        
        return True