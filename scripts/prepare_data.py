"""
Data preparation pipeline for Rossmann dataset
Merges store data and handles missing values
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def load_and_merge_data(raw_path='data/raw/'):
    """
    Load raw data and merge store information
    """
    print("Loading raw data...")
    
    # Load files
    train = pd.read_csv(f'{raw_path}/train_raw.csv')
    test = pd.read_csv(f'{raw_path}/test_raw.csv')
    store = pd.read_csv(f'{raw_path}/store_raw.csv')
    
    # Convert dates
    train['Date'] = pd.to_datetime(train['Date'])
    test['Date'] = pd.to_datetime(test['Date'])
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Store shape: {store.shape}")
    
    # Merge store data with train
    print("\nMerging store data with training...")
    train_merged = train.merge(store, on='Store', how='left')
    
    # Handle missing values in store data
    print("\nHandling missing values...")
    
    # Competition distance - fill with median (business logic: assume average competition)
    train_merged['CompetitionDistance'] = train_merged['CompetitionDistance'].fillna(
        train_merged['CompetitionDistance'].median()
    )
    
    # Competition open since - fill with 0 (means not applicable)
    train_merged['CompetitionOpenSinceMonth'] = train_merged['CompetitionOpenSinceMonth'].fillna(0)
    train_merged['CompetitionOpenSinceYear'] = train_merged['CompetitionOpenSinceYear'].fillna(0)
    
    # Promo2 - fill with 0 (no promo2)
    train_merged['Promo2'] = train_merged['Promo2'].fillna(0)
    train_merged['Promo2SinceWeek'] = train_merged['Promo2SinceWeek'].fillna(0)
    train_merged['Promo2SinceYear'] = train_merged['Promo2SinceYear'].fillna(0)
    train_merged['PromoInterval'] = train_merged['PromoInterval'].fillna('None')
    
    # Verify date range for 1-day gap
    train_max_date = train_merged['Date'].max()
    test_min_date = test['Date'].min()
    gap_days = (test_min_date - train_max_date).days
    
    print(f"\n✅ Train ends: {train_max_date.date()}")
    print(f"✅ Test starts: {test_min_date.date()}")
    print(f"✅ Gap: {gap_days} day(s)")
    
    # Save processed data
    print("\nSaving processed data...")
    train_merged.to_csv('data/processed/train_processed.csv', index=False)
    test.to_csv('data/processed/test_original.csv', index=False)
    
    print("✅ Data preparation complete!")
    
    return train_merged, test

if __name__ == "__main__":
    train_data, test_data = load_and_merge_data()
    
    # Print summary statistics
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