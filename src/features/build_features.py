import pandas as pd
import numpy as np
from typing import List

class DemandFeatureBuilder:
    def __init__(self, lags=[1, 7, 14, 28], rolling_windows=[7, 14, 30]):
        self.lags = lags
        self.rolling_windows = rolling_windows
    
    def add_temporal_features(self, df):
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['month'] = df['Date'].dt.month
        df['day_of_month'] = df['Date'].dt.day
        df['week_of_year'] = df['Date'].dt.isocalendar().week
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        return df
    
    def add_lag_features(self, df):
        df = df.copy()
        df = df.sort_values(['Store', 'Date'])
        for lag in self.lags:
            df[f'lag_{lag}'] = df.groupby('Store')['Sales'].shift(lag)
        return df
    
    def add_rolling_features(self, df):
        df = df.copy()
        for window in self.rolling_windows:
            df[f'rolling_mean_{window}'] = df.groupby('Store')['Sales'].transform(
                lambda x: x.rolling(window, min_periods=1).mean())
            df[f'rolling_std_{window}'] = df.groupby('Store')['Sales'].transform(
                lambda x: x.rolling(window, min_periods=1).std())
        return df
    
    def create_features(self, df):
        df = self.add_temporal_features(df)
        df = self.add_lag_features(df)
        df = self.add_rolling_features(df)
        return df