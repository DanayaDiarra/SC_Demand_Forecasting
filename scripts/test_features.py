import pandas as pd
import sys
sys.path.append('.')

from src.features.build_features import DemandFeatureBuilder

print("Loading data...")
df = pd.read_csv('data/processed/train_processed.csv', nrows=100000)
df['Date'] = pd.to_datetime(df['Date'])

# Pick a store
store_id = df['Store'].value_counts().index[0]
store_data = df[df['Store'] == store_id].sort_values('Date')

print(f"Store {store_id}: {len(store_data)} records")
print(f"Date range: {store_data['Date'].min()} to {store_data['Date'].max()}")

# Create features
builder = DemandFeatureBuilder(lags=[1, 7, 14], rolling_windows=[7, 14])
store_features = builder.create_features(store_data)

new_cols = [c for c in store_features.columns if c not in store_data.columns]
print(f"\n✅ Created {len(new_cols)} new features:")
for col in new_cols[:10]:
    print(f"  - {col}")

print(f"\nSample data:")
print(store_features[['Date', 'Sales', 'lag_1', 'rolling_mean_7']].tail(10))
