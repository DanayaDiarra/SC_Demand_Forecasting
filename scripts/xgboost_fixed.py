import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')
import os

print("="*60)
print("XGBOOST DEMAND FORECASTING - FIXED VERSION")
print("="*60)

# Load data
print("\n1. Loading data...")
df = pd.read_csv('data/processed/train_processed.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Use Store 1
store_data = df[df['Store'] == 1].sort_values('Date').copy()

print(f"\n2. Store 1 Analysis")
print(f"   Total records: {len(store_data)}")
print(f"   Date range: {store_data['Date'].min().date()} to {store_data['Date'].max().date()}")
print(f"   Sales - Mean: {store_data['Sales'].mean():.0f}, Std: {store_data['Sales'].std():.0f}")

# Create features (but keep the original for reference)
print("\n3. Creating features...")

# Temporal features
store_data['day_of_week'] = store_data['Date'].dt.dayofweek
store_data['month'] = store_data['Date'].dt.month
store_data['day'] = store_data['Date'].dt.day
store_data['week'] = store_data['Date'].dt.isocalendar().week
store_data['quarter'] = store_data['Date'].dt.quarter

# Lag features (these will create NaN at the beginning)
store_data['lag_1'] = store_data['Sales'].shift(1)
store_data['lag_7'] = store_data['Sales'].shift(7)
store_data['lag_14'] = store_data['Sales'].shift(14)
store_data['lag_28'] = store_data['Sales'].shift(28)

# Rolling statistics (these will create NaN at the beginning)
store_data['roll_mean_7'] = store_data['Sales'].rolling(7, min_periods=1).mean()
store_data['roll_mean_14'] = store_data['Sales'].rolling(14, min_periods=1).mean()
store_data['roll_std_7'] = store_data['Sales'].rolling(7, min_periods=1).std()
store_data['roll_std_14'] = store_data['Sales'].rolling(14, min_periods=1).std()

# Promo features
store_data['promo_active'] = store_data['Promo'].astype(int)

# Check NaN counts before dropping
print(f"\n   NaN counts before dropping:")
print(f"   lag_1: {store_data['lag_1'].isna().sum()} rows")
print(f"   lag_7: {store_data['lag_7'].isna().sum()} rows")
print(f"   lag_14: {store_data['lag_14'].isna().sum()} rows")
print(f"   lag_28: {store_data['lag_28'].isna().sum()} rows")

# Instead of dropping all NaN, let's only keep rows where all features are available
# But we need at least 28 days of history for all lags
min_required_days = 28
store_data_clean = store_data.iloc[min_required_days:].copy()

print(f"\n   After keeping data from day {min_required_days}+: {len(store_data_clean)} records")

# Define features
features = [
    'day_of_week', 'month', 'day', 'week', 'quarter',
    'lag_1', 'lag_7', 'lag_14', 'lag_28',
    'roll_mean_7', 'roll_mean_14',
    'roll_std_7', 'roll_std_14',
    'promo_active'
]

# Verify no NaN in features
print(f"\n   Final NaN check:")
for col in features:
    na_count = store_data_clean[col].isna().sum()
    if na_count > 0:
        print(f"   WARNING: {col} has {na_count} NaN values")

# Split into train and test (last 30 days for testing)
train = store_data_clean.iloc[:-30]
test = store_data_clean.iloc[-30:]

print(f"\n4. Data Split:")
print(f"   Train: {len(train)} records ({train['Date'].min().date()} to {train['Date'].max().date()})")
print(f"   Test: {len(test)} records ({test['Date'].min().date()} to {test['Date'].max().date()})")
print(f"   Test period: Last 30 days of available data")

# Train XGBoost
print("\n5. Training XGBoost model...")
model = XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

model.fit(train[features], train['Sales'])

# Predict
print("\n6. Making predictions...")
predictions = model.predict(test[features])

# Calculate metrics
rmse = np.sqrt(mean_squared_error(test['Sales'], predictions))
mape = mean_absolute_percentage_error(test['Sales'], predictions) * 100
bias = (test['Sales'] - predictions).mean()
error_std = (test['Sales'] - predictions).std()
safety_stock = 1.65 * error_std

print("\n" + "="*60)
print("XGBOOST RESULTS")
print("="*60)
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"Forecast Accuracy: {100-mape:.1f}%")
print(f"Bias: {bias:.2f}")
if bias > 0:
    print(f"  → Underforecasting by {bias:.0f} units/day (Risk: Stockouts)")
else:
    print(f"  → Overforecasting by {abs(bias):.0f} units/day (Risk: Excess inventory)")
print(f"Error Std Dev: {error_std:.2f}")
print(f"Safety Stock (95% service level): {safety_stock:.0f} units")

# Feature importance
print("\n7. Feature Importance:")
importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
for i, row in importance.head(10).iterrows():
    print(f"   {row['feature']}: {row['importance']:.4f}")

# Show predictions
print("\n8. Sample Predictions (Last 7 days of test):")
results_df = pd.DataFrame({
    'Date': test['Date'].tail(7).dt.date,
    'Actual': test['Sales'].tail(7).values,
    'Predicted': predictions[-7:],
    'Error': (test['Sales'].tail(7).values - predictions[-7:])
})
results_df['Error%'] = np.abs(results_df['Error'] / results_df['Actual']) * 100
print(results_df.to_string(index=False))

# Save results
os.makedirs('results', exist_ok=True)
full_results = pd.DataFrame({
    'Date': test['Date'],
    'Actual': test['Sales'],
    'Predicted': predictions,
    'Error': test['Sales'] - predictions
})
full_results.to_csv('results/store1_xgboost_results.csv', index=False)
print(f"\n✅ Results saved to results/store1_xgboost_results.csv")

print("\n" + "="*60)
print("BUSINESS INSIGHTS")
print("="*60)
if mape < 15:
    print("✅ EXCELLENT forecast accuracy! (<15% MAPE)")
    print("   This model can reduce inventory costs significantly.")
elif mape < 25:
    print("📊 GOOD forecast accuracy (15-25% MAPE)")
    print("   Recommended for production with regular retraining.")
else:
    print("⚠️ MODERATE forecast accuracy (>25% MAPE)")
    print("   Consider adding more features (competition, holidays, etc.)")

avg_daily_sales = test['Sales'].mean()
print(f"\n📦 Recommended Safety Stock: {safety_stock:.0f} units")
print(f"   (This covers {safety_stock/avg_daily_sales*100:.1f}% of average daily demand)")
