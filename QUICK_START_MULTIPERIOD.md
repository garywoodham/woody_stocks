# Quick Start - Multi-Period Stock Prediction

## What Changed?

### 1. Data Period: 5 years → 10 years ✓
- More historical data for better training
- Run: `python download_stock_data.py`

### 2. New Time Periods: Daily, Weekly, Monthly ✓
- Daily data aggregated into weekly (Fridays) and monthly (month-end)
- Run: `python aggregate_data.py`

### 3. New Training System ✓
- Matches data granularity to prediction horizon
- Daily → predict days ahead
- Weekly → predict weeks ahead  
- Monthly → predict months ahead
- Run: `python train_model_multiperiod.py`

## Quick Commands

```bash
# 1. Download 10 years of data
python download_stock_data.py

# 2. Create weekly and monthly aggregations
python aggregate_data.py

# 3. Train models on all time periods
python train_model_multiperiod.py
```

## File Structure

```
data/
  ├── multi_sector_stocks.csv          (daily - 49,668 records)
  ├── multi_sector_stocks_weekly.csv   (weekly - 10,297 records)
  └── multi_sector_stocks_monthly.csv  (monthly - 2,385 records)

models/
  ├── TICKER_daily_models.joblib       (1d, 5d, 21d horizons)
  ├── TICKER_weekly_models.joblib      (1w, 4w, 12w horizons)
  └── TICKER_monthly_models.joblib     (1m, 3m, 6m horizons)
```

## Why This Approach?

**Old way:** Using daily data to predict 90 days ahead
- ❌ Mismatched time scales
- ❌ Indicators calibrated for daily movements
- ❌ Poor long-term predictions

**New way:** Match data period to prediction horizon
- ✅ Daily data predicts days
- ✅ Weekly data predicts weeks
- ✅ Monthly data predicts months
- ✅ Indicators adjusted per time period
- ✅ Better accuracy at each time scale

## Example Model Usage

```python
import joblib
import pandas as pd

# Load a trained model
models = joblib.load('models/AAPL_weekly_models.joblib')

# Get the 4-week horizon model
week4_model = models[4]
model = week4_model['model']
scaler = week4_model['scaler']
feature_cols = week4_model['feature_cols']

# Make prediction on new data
# (after creating features with create_features())
X = data[feature_cols].values
X_scaled = scaler.transform(X)
prediction = model.predict(X_scaled)
```

## What's Next?

- Test the models on new data
- Update prediction and backtesting scripts
- Integrate with dashboard
- Consider ensemble methods combining multiple time periods
