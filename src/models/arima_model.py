"""
ARIMA/SARIMA model for baseline forecasting
"""

import pandas as pd
import numpy as np
from pmdarima import auto_arima
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ARIMAForecaster:
    """
    ARIMA model with automatic parameter selection
    """
    
    def __init__(self, seasonal_period: int = 7):
        self.seasonal_period = seasonal_period
        self.models = {}  # Store one model per store
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame, store_col: str = 'Store', target_col: str = 'Sales'):
        """
        Fit ARIMA model for each store
        """
        self.stores = df[store_col].unique()
        
        for store in self.stores:
            logger.info(f"Fitting ARIMA for store {store}")
            
            # Get store data
            store_data = df[df[store_col] == store].sort_values('Date')
            
            # Skip if not enough data
            if len(store_data) < 30:
                logger.warning(f"Store {store} has insufficient data")
                continue
            
            # Fit auto_arima
            try:
                model = auto_arima(
                    store_data[target_col],
                    start_p=1, start_q=1,
                    max_p=3, max_q=3,
                    seasonal=True,
                    m=self.seasonal_period,
                    start_P=0, start_Q=0,
                    max_P=2, max_Q=2,
                    trace=False,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True,
                    n_fits=50
                )
                self.models[store] = model
                
            except Exception as e:
                logger.error(f"Failed to fit ARIMA for store {store}: {e}")
                continue
        
        self.is_fitted = True
        return self
    
    def predict(self, df: pd.DataFrame, steps: int = 7) -> pd.DataFrame:
        """
        Predict next 'steps' days
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        predictions = []
        
        for store in self.stores:
            if store not in self.models:
                continue
            
            model = self.models[store]
            
            # Get last date
            last_date = df[df['Store'] == store]['Date'].max()
            
            # Make forecast
            forecast = model.predict(n_periods=steps)
            
            # Create prediction dataframe
            for i, pred in enumerate(forecast):
                pred_date = last_date + pd.Timedelta(days=i+1)
                predictions.append({
                    'Store': store,
                    'Date': pred_date,
                    'Sales_Predicted': max(0, pred),  # No negative sales
                    'Model': 'ARIMA'
                })
        
        return pd.DataFrame(predictions)