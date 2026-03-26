"""
XGBoost with lag features - most powerful model
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XGBoostForecaster:
    """
    XGBoost model with engineered features
    """
    
    def __init__(self, feature_cols: List[str], params: Dict = None):
        self.feature_cols = feature_cols
        self.params = params or {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        self.models = {}  # One model per store
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """
        Separate features and target
        """
        X = df[self.feature_cols].copy()
        y = df['Sales'].copy() if 'Sales' in df.columns else None
        
        # Handle missing values
        X = X.fillna(X.median())
        
        return X, y
    
    def fit(self, df: pd.DataFrame, store_col: str = 'Store'):
        """
        Train one XGBoost model per store
        """
        self.stores = df[store_col].unique()
        
        for store in self.stores:
            logger.info(f"Training XGBoost for store {store}")
            
            # Get store data
            store_data = df[df[store_col] == store].sort_values('Date')
            
            # Skip if not enough data
            if len(store_data) < 50:
                logger.warning(f"Store {store} has insufficient data")
                continue
            
            # Prepare features
            X, y = self.prepare_features(store_data)
            
            # Remove rows with NaN in target
            valid_idx = y.notna()
            X = X[valid_idx]
            y = y[valid_idx]
            
            if len(X) < 30:
                continue
            
            # Train model
            model = XGBRegressor(**self.params)
            model.fit(X, y)
            
            self.models[store] = model
            
        return self
    
    def predict(self, df: pd.DataFrame, steps: int = 7) -> pd.DataFrame:
        """
        Forecast next steps days
        """
        predictions = []
        
        for store in self.stores:
            if store not in self.models:
                continue
            
            model = self.models[store]
            
            # Get last row of data
            last_data = df[df['Store'] == store].sort_values('Date').iloc[-1:]
            last_date = last_data['Date'].iloc[0]
            
            # Iteratively predict (for multi-step forecast)
            current_data = last_data.copy()
            for i in range(steps):
                # Prepare features for prediction
                X_pred, _ = self.prepare_features(current_data)
                
                # Predict
                pred = model.predict(X_pred)[0]
                
                # Create next date
                next_date = last_date + pd.Timedelta(days=i+1)
                
                # Store prediction
                predictions.append({
                    'Store': store,
                    'Date': next_date,
                    'Sales_Predicted': max(0, pred),
                    'Model': 'XGBoost'
                })
                
                # Update current_data for next iteration (simulate feature update)
                # This is where you'd update lag features, etc.
                new_row = current_data.copy()
                new_row['Date'] = next_date
                new_row['Sales'] = pred  # Use prediction as future sales for next step
                current_data = new_row
        
        return pd.DataFrame(predictions)