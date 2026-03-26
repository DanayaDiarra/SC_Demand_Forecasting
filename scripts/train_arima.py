import pandas as pd
import numpy as np
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
df = pd.read_csv('data/processed/train_processed.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Select store with most data
store_counts = df.groupby('Store').size().sort_values(ascending=False)
store_id = store_counts.index[0]
store_data = df[df['Store'] == store_id].sort_values('Date')

print(f"\nStore {store_id}: {len(store_data)} days")
print(f"Date range: {store_data['Date'].min().date()} to {store_data['Date'].max().date()}")

# Split: last 30 days for validation
train = store_data.iloc[:-30]
val = store_data.iloc[-30:]

print(f"\nTrain: {train['Date'].min().date()} to {train['Date'].max().date()}")
print(f"Val: {val['Date'].min().date()} to {val['Date'].max().date()}")
print(f"Gap: {(val['Date'].min() - train['Date'].max()).days} days")

# Train ARIMA
print("\n" + "="*50)
print("Training ARIMA model...")
print("="*50)

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

print(f"\nBest model: ARIMA{model.order} x {model.seasonal_order}")

# Predict
print("\nMaking predictions...")
preds = model.predict(n_periods=len(val))

# Metrics
rmse = np.sqrt(mean_squared_error(val['Sales'], preds))
mape = mean_absolute_percentage_error(val['Sales'], preds) * 100
bias = (val['Sales'] - preds).mean()

print("\n" + "="*50)
print("RESULTS")
print("="*50)
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"Bias: {bias:.2f}")
print(f"Direction: {'Underforecast' if bias > 0 else 'Overforecast'}")

# Safety stock
error_std = (val['Sales'] - preds).std()
safety_stock = 1.65 * error_std
print(f"\nSafety Stock (95% service level): {safety_stock:.0f} units")
print(f"Forecast Accuracy: {100-mape:.1f}%")

# Save results
import os
os.makedirs('results', exist_ok=True)

results = pd.DataFrame({
    'Date': val['Date'],
    'Actual': val['Sales'],
    'Predicted': preds,
    'Error': val['Sales'] - preds
})
results.to_csv(f'results/store_{store_id}_arima.csv', index=False)
print(f"\n✅ Saved to results/store_{store_id}_arima.csv")