import pandas as pd
import numpy as np
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')
import os

print("="*60)
print("ARIMA DEMAND FORECASTING MODEL")
print("="*60)

print("\n1. Loading data...")
df = pd.read_csv('data/processed/train_processed.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Find store with most data
store_counts = df.groupby('Store').size().sort_values(ascending=False)
store_id = store_counts.index[0]
store_data = df[df['Store'] == store_id].sort_values('Date')

print(f"\n2. Analyzing Store {store_id}")
print(f"   Total days: {len(store_data)}")
print(f"   Date range: {store_data['Date'].min().date()} to {store_data['Date'].max().date()}")

# Split data (last 30 days for validation)
train = store_data.iloc[:-30]
val = store_data.iloc[-30:]

print(f"\n3. Data Split")
print(f"   Training: {train['Date'].min().date()} to {train['Date'].max().date()} ({len(train)} days)")
print(f"   Validation: {val['Date'].min().date()} to {val['Date'].max().date()} ({len(val)} days)")
print(f"   Gap between train and validation: {(val['Date'].min() - train['Date'].max()).days} days")

# Train ARIMA
print("\n4. Training ARIMA model...")
print("   (This may take 1-2 minutes)")

model = auto_arima(
    train['Sales'],
    start_p=1, start_q=1,
    max_p=3, max_q=3,
    seasonal=True,
    m=7,
    start_P=0, start_Q=0,
    max_P=2, max_Q=2,
    trace=False,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True,
    n_fits=30
)

print(f"\n   Best model found: ARIMA{model.order} x {model.seasonal_order}")

# Make predictions
print("\n5. Making predictions for validation period...")
preds = model.predict(n_periods=len(val))

# Calculate metrics
rmse = np.sqrt(mean_squared_error(val['Sales'], preds))
mape = mean_absolute_percentage_error(val['Sales'], preds) * 100
bias = (val['Sales'] - preds).mean()
error_std = (val['Sales'] - preds).std()
safety_stock = 1.65 * error_std

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"Forecast Accuracy: {100-mape:.1f}%")
print(f"Bias: {bias:.2f} ({'Underforecast' if bias > 0 else 'Overforecast'})")
print(f"Forecast Error Std Dev: {error_std:.2f}")
print(f"Safety Stock (95% service level): {safety_stock:.0f} units")

# Show actual vs predicted
print("\n6. Sample of predictions (last 5 days):")
comparison = pd.DataFrame({
    'Date': val['Date'],
    'Actual': val['Sales'],
    'Predicted': preds,
    'Error': val['Sales'] - preds,
    'Error%': np.abs((val['Sales'] - preds) / val['Sales']) * 100
})
print(comparison.tail().round(2))

# Save results
os.makedirs('results', exist_ok=True)
comparison.to_csv(f'results/store_{store_id}_arima_results.csv', index=False)
print(f"\n✅ Results saved to results/store_{store_id}_arima_results.csv")

print("\n" + "="*60)
print("BUSINESS INSIGHTS")
print("="*60)
if bias > 0:
    print(f"⚠️  WARNING: Systematic UNDERforecasting detected!")
    print(f"   On average, we're under-predicting by {bias:.0f} units/day")
    print(f"   → Risk: Potential stockouts and lost sales")
else:
    print(f"✅ Systematic OVERforecasting detected")
    print(f"   On average, we're over-predicting by {abs(bias):.0f} units/day")
    print(f"   → Risk: Excess inventory and higher carrying costs")

print(f"\n📊 Recommended Actions:")
print(f"   1. Keep {safety_stock:.0f} units as safety stock")
print(f"   2. Monitor forecast accuracy (target >80%)")
print(f"   3. Consider XGBoost for potentially better accuracy")
