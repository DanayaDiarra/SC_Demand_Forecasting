"""
Main training script for all forecasting models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.features.build_features import DemandFeatureBuilder
from src.models.arima_model import ARIMAForecaster
from src.models.xgboost_model import XGBoostForecaster
from src.evaluation.time_series_split import ForecastingValidator

def main():
    # Load your processed data
    # You'll need to create this file from your data processing step
    train_data = pd.read_csv('data/processed/train_processed.csv')
    train_data['Date'] = pd.to_datetime(train_data['Date'])
    
    # Create features
    feature_builder = DemandFeatureBuilder()
    train_with_features = feature_builder.create_forecasting_features(train_data)
    
    # Save features for reuse
    train_with_features.to_csv('data/processed/train_with_features.csv', index=False)
    
    # Define feature columns for XGBoost
    feature_cols = [
        'day_of_week', 'month', 'day_of_month', 'week_of_year',
        'day_of_week_sin', 'day_of_week_cos',
        'is_month_start', 'is_month_end', 'days_until_month_end',
        'Promo', 'promo_effect', 'days_since_promo',
        'lag_1', 'lag_7', 'lag_14', 'lag_28',
        'rolling_mean_7', 'rolling_std_7',
        'rolling_mean_14', 'rolling_std_14',
        'rolling_mean_30', 'rolling_std_30'
    ]
    
    # Filter to available columns
    available_cols = [col for col in feature_cols if col in train_with_features.columns]
    print(f"Using {len(available_cols)} features: {available_cols}")
    
    # Train ARIMA
    print("\n" + "="*50)
    print("Training ARIMA models...")
    arima = ARIMAForecaster(seasonal_period=7)
    arima.fit(train_with_features)
    
    # Train XGBoost
    print("\n" + "="*50)
    print("Training XGBoost models...")
    xgb = XGBoostForecaster(feature_cols=available_cols)
    xgb.fit(train_with_features)
    
    # Save models (you'll need to implement model saving)
    import joblib
    joblib.dump(arima, 'models/arima_model.pkl')
    joblib.dump(xgb, 'models/xgboost_model.pkl')
    
    print("\n✅ Models trained and saved successfully!")

if __name__ == "__main__":
    main()