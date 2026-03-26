"""
Optimized data preparation with proper data types
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_and_merge_data(raw_path='data/raw/'):
    """
    Load raw data with optimized dtypes to save memory
    """
    print("Loading raw data with optimized dtypes...")
    
    # Define dtypes to save memory
    train_dtypes = {
        'Store': 'int16',
        'DayOfWeek': 'int8',
        'Sales': 'int32',
        'Customers': 'int32',
        'Open': 'int8',
        'Promo': 'int8',
        'StateHoliday': 'category',
        'SchoolHoliday': 'int8'
    }
    
    store_dtypes = {
        'Store': 'int16',
        'StoreType': 'category',
        'Assortment': 'category',
        'CompetitionDistance': 'float32',
        'CompetitionOpenSinceMonth': 'float32',
        'CompetitionOpenSinceYear': 'float32',
        'Promo2': 'int8',
        'Promo2SinceWeek': 'float32',
        'Promo2SinceYear': 'float32',
        'PromoInterval': 'category'
    }
    
    # Load with optimized dtypes
    train = pd.read_csv(f'{raw_path}/train_raw.csv', dtype=train_dtypes, low_memory=False)
    test = pd.read_csv(f'{raw_path}/test_raw.csv')
    store = pd.read_csv(f'{raw_path}/store_raw.csv', dtype=store_dtypes)
    
    # Convert dates
    train['Date'] = pd.to_datetime(train['Date'])
    test['Date'] = pd.to_datetime(test['Date'])
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Store shape: {store.shape}")
    
    # Merge store data
    print("\nMerging store data...")
    train_merged = train.merge(store, on='Store', how='left')
    
    # Handle missing values
    print("\nHandling missing values...")
    
    # Competition distance
    train_merged['CompetitionDistance'] = train_merged['CompetitionDistance'].fillna(
        train_merged['CompetitionDistance'].median()
    )
    
    # Competition open dates
    for col in ['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear']:
        train_merged[col] = train_merged[col].fillna(0).astype('int32')
    
    # Promo2
    train_merged['Promo2'] = train_merged['Promo2'].fillna(0).astype('int8')
    train_merged['Promo2SinceWeek'] = train_merged['Promo2SinceWeek'].fillna(0).astype('int32')
    train_merged['Promo2SinceYear'] = train_merged['Promo2SinceYear'].fillna(0).astype('int32')
    train_merged['PromoInterval'] = train_merged['PromoInterval'].fillna('None')
    
    # Verify date gap
    train_max_date = train_merged['Date'].max()
    test_min_date = test['Date'].min()
    gap_days = (test_min_date - train_max_date).days
    
    print(f"\n✅ Train ends: {train_max_date.date()}")
    print(f"✅ Test starts: {test_min_date.date()}")
    print(f"✅ Gap: {gap_days} day(s)")
    
    # Save
    print("\nSaving processed data...")
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    train_merged.to_csv('data/processed/train_processed.csv', index=False)
    test.to_csv('data/processed/test_original.csv', index=False)
    
    # Memory usage report
    memory_usage = train_merged.memory_usage(deep=True).sum() / 1024**2
    print(f"\n💾 Memory usage: {memory_usage:.2f} MB")
    
    return train_merged, test

if __name__ == "__main__":
    train_data, test_data = load_and_merge_data()
    
    print("\n" + "="*50)
    print("DATA SUMMARY")
    print("="*50)
    print(f"Total records: {len(train_data):,}")
    print(f"Unique stores: {train_data['Store'].nunique()}")
    print(f"Date range: {train_data['Date'].min().date()} to {train_data['Date'].max().date()}")
    print(f"\nSales statistics:")
    print(f"  Mean: {train_data['Sales'].mean():.0f}")
    print(f"  Median: {train_data['Sales'].median():.0f}")
    print(f"  Std: {train_data['Sales'].std():.0f}")
    print(f"  Max: {train_data['Sales'].max():,}")
