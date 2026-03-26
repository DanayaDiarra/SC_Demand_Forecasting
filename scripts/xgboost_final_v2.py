import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')
import os

print("="*60)
print("XGBOOST DEMAND FORECASTING - WITH BUSINESS METRICS")
print("="*60)

# Load data
df = pd.read_csv('data/processed/train_processed.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Use Store 1
store_data = df[df['Store'] == 1].sort_values('Date').copy()

print(f"\nStore 1: {len(store_data)} days of data")
print(f"Date range: {store_data['Date'].min().date()} to {store_data['Date'].max().date()}")

# Create features
store_data['day_of_week'] = store_data['Date'].dt.dayofweek
store_data['month'] = store_data['Date'].dt.month
store_data['day'] = store_data['Date'].dt.day
store_data['week'] = store_data['Date'].dt.isocalendar().week
store_data['quarter'] = store_data['Date'].dt.quarter

# Lag features
store_data['lag_1'] = store_data['Sales'].shift(1)
store_data['lag_7'] = store_data['Sales'].shift(7)
store_data['lag_14'] = store_data['Sales'].shift(14)
store_data['lag_28'] = store_data['Sales'].shift(28)

# Rolling statistics
store_data['roll_mean_7'] = store_data['Sales'].rolling(7, min_periods=1).mean()
store_data['roll_mean_14'] = store_data['Sales'].rolling(14, min_periods=1).mean()
store_data['roll_std_7'] = store_data['Sales'].rolling(7, min_periods=1).std()
store_data['roll_std_14'] = store_data['Sales'].rolling(14, min_periods=1).std()

# Promo features
store_data['promo_active'] = store_data['Promo'].astype(int)

# Keep data from day 28 onwards (ensures all lags available)
store_data_clean = store_data.iloc[28:].copy()

# Features
features = [
    'day_of_week', 'month', 'day', 'week', 'quarter',
    'lag_1', 'lag_7', 'lag_14', 'lag_28',
    'roll_mean_7', 'roll_mean_14', 'roll_std_7', 'roll_std_14',
    'promo_active'
]

# Split: last 30 days for testing
train = store_data_clean.iloc[:-30]
test = store_data_clean.iloc[-30:]

print(f"\nTrain: {len(train)} days ({train['Date'].min().date()} to {train['Date'].max().date()})")
print(f"Test: {len(test)} days ({test['Date'].min().date()} to {test['Date'].max().date()})")

# Train model
print("\nTraining XGBoost...")
model = XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
model.fit(train[features], train['Sales'])

# Predict
predictions = model.predict(test[features])

# Calculate metrics (handle zero values in MAPE)
def safe_mape(actual, predicted):
    """Calculate MAPE while handling zero actual values"""
    # Filter out zero actual values for MAPE calculation
    mask = actual != 0
    if mask.sum() == 0:
        return float('inf')
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

rmse = np.sqrt(mean_squared_error(test['Sales'], predictions))
mape = safe_mape(test['Sales'].values, predictions)
bias = (test['Sales'] - predictions).mean()
error_std = (test['Sales'] - predictions).std()
safety_stock = 1.65 * error_std  # 95% service level

# Weighted MAPE (better for supply chain)
def wmape(actual, predicted):
    """Weighted MAPE - more business relevant"""
    return np.sum(np.abs(actual - predicted)) / np.sum(np.abs(actual)) * 100

wmape_value = wmape(test['Sales'].values, predictions)

print("\n" + "="*60)
print("MODEL PERFORMANCE METRICS")
print("="*60)
print(f"RMSE: {rmse:.2f} units")
print(f"MAPE: {mape:.2f}% (excluding zero sales days)")
print(f"WMAPE: {wmape_value:.2f}% (weighted by sales volume)")
print(f"Bias: {bias:.2f} units/day")
print(f"  → {'Underforecasting' if bias > 0 else 'Overforecasting'} by {abs(bias):.0f} units/day")
print(f"Forecast Error Std Dev: {error_std:.2f} units")

# Safety stock
print(f"\n" + "="*60)
print("INVENTORY RECOMMENDATIONS")
print("="*60)
print(f"📦 Safety Stock (95% service level): {safety_stock:.0f} units")
print(f"   This means we expect to have stock available 95% of the time")
print(f"   Safety stock as % of avg daily sales: {safety_stock/test['Sales'].mean()*100:.1f}%")

# Feature importance
print(f"\n" + "="*60)
print("TOP DRIVERS OF DEMAND")
print("="*60)
importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
for i, row in importance.head(8).iterrows():
    # Add interpretation
    if row['feature'] == 'day_of_week':
        print(f"   📅 {row['feature']}: {row['importance']:.3f} (Day patterns - highest on weekends?)")
    elif row['feature'] == 'promo_active':
        print(f"   🎯 {row['feature']}: {row['importance']:.3f} (Promotion impact)")
    elif 'lag' in row['feature']:
        print(f"   ⏰ {row['feature']}: {row['importance']:.3f} (Previous sales impact)")
    elif 'roll' in row['feature']:
        print(f"   📊 {row['feature']}: {row['importance']:.3f} (Recent sales trend)")
    else:
        print(f"   🔍 {row['feature']}: {row['importance']:.3f}")

# Show predictions
print(f"\n" + "="*60)
print("LAST 7 DAYS FORECAST vs ACTUAL")
print("="*60)
results_df = pd.DataFrame({
    'Date': test['Date'].tail(7).dt.date,
    'Actual': test['Sales'].tail(7).values,
    'Predicted': predictions[-7:].round(0),
    'Error': (test['Sales'].tail(7).values - predictions[-7:]).round(0)
})
results_df['Error%'] = (np.abs(results_df['Error']) / results_df['Actual'].replace(0, np.nan) * 100).round(1)
results_df['Error%'] = results_df['Error%'].fillna(0)
print(results_df.to_string(index=False))

# Business impact
print(f"\n" + "="*60)
print("BUSINESS IMPACT ANALYSIS")
print("="*60)

# Calculate potential inventory savings
avg_daily_sales = test['Sales'].mean()
current_inventory = avg_daily_sales * 30  # Assuming 30 days of inventory
recommended_inventory = avg_daily_sales * 14 + safety_stock  # 14 days + safety stock
inventory_reduction = current_inventory - recommended_inventory
inventory_reduction_pct = (inventory_reduction / current_inventory) * 100

print(f"Current inventory (30 days): {current_inventory:.0f} units")
print(f"Recommended inventory (14 days + safety stock): {recommended_inventory:.0f} units")
print(f"Potential inventory reduction: {inventory_reduction:.0f} units ({inventory_reduction_pct:.1f}%)")

# Stockout risk
stockout_risk = (bias / error_std) if error_std > 0 else 0
print(f"\nStockout Risk Score: {stockout_risk:.2f}")
if stockout_risk > 1:
    print("⚠️ HIGH stockout risk - increase safety stock")
elif stockout_risk < -1:
    print("✅ LOW stockout risk - potential to reduce inventory")
else:
    print("📊 MODERATE stockout risk - current safety stock is adequate")

# Save results
os.makedirs('results', exist_ok=True)
full_results = pd.DataFrame({
    'Date': test['Date'],
    'Actual': test['Sales'],
    'Predicted': predictions,
    'Error': test['Sales'] - predictions
})
full_results.to_csv('results/store1_xgboost_final.csv', index=False)
print(f"\n✅ Results saved to results/store1_xgboost_final.csv")

print("\n" + "="*60)
print("NEXT STEPS")
print("="*60)
print("1. Add more features: holidays, competition distance, store type")
print("2. Test on other stores to validate generalization")
print("3. Build Streamlit dashboard for interactive forecasting")
print("4. Add what-if analysis for promotion scenarios")
