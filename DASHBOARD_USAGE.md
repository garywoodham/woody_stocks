# Multi-Period Stock Prediction Dashboard - Updated Workflow

## Quick Start

### 1. Download 10 Years of Data
```bash
python download_stock_data.py
```
Downloads daily stock data for 20 stocks across 4 sectors (Defence, Banking, Pharma, Technology).

### 2. Aggregate Data to Multiple Time Periods
```bash
python aggregate_data.py
```
Creates:
- `data/multi_sector_stocks.csv` (daily)
- `data/multi_sector_stocks_weekly.csv` (weekly - Friday close)
- `data/multi_sector_stocks_monthly.csv` (monthly - month-end)

### 3. Train Multi-Period Models
```bash
python train_model_multiperiod.py
```
Trains separate models for each time period:
- **Daily models**: Predict 1, 5, 21 days ahead
- **Weekly models**: Predict 1, 4, 12 weeks ahead
- **Monthly models**: Predict 1, 3, 6 months ahead

Models saved to: `models/TICKER_daily_models.joblib`, etc.

### 4. Generate Multi-Period Predictions
```bash
python predict_multiperiod.py
```
Generates predictions across all time periods and saves to `predictions_multiperiod.csv`.

### 5. Launch Dashboard
```bash
python dashboard.py
```
Opens at http://localhost:8050

The dashboard automatically detects whether you're using:
- Multi-period predictions (recommended) - from `predictions_multiperiod.csv`
- Legacy predictions - from `predictions_summary.csv`

## Dashboard Features

### Multi-Period View
When using multi-period predictions, the dashboard displays:

**Daily Predictions:**
- 1d - Next day
- 5d - Next week (5 trading days)
- 21d - Next month (21 trading days)

**Weekly Predictions:**
- 1w - Next week
- 4w - Next month
- 12w - Next quarter

**Monthly Predictions:**
- 1m - Next month
- 3m - Next quarter
- 6m - Next half year

### Dashboard Tabs

1. **📊 Predictions & Charts**
   - Interactive candlestick charts
   - Volume analysis
   - Filter by sector and stock
   - KPI cards with predictions
   - Full stock overview table

2. **🚦 Trading Signals**
   - Daily trading signals
   - Signal strength indicators
   - Actionable buy/sell recommendations

3. **🎯 Backtest Results**
   - Strategy performance metrics
   - Win rates and Sharpe ratios
   - Comparative analysis

## Key Improvements

### Old Approach Problems
❌ Used daily data to predict 90 days ahead  
❌ Mismatched time scales (daily indicators for long-term predictions)  
❌ Only 5 years of data  
❌ Single model type for all horizons  

### New Approach Benefits
✅ Time-scale matched: daily data predicts days, weekly predicts weeks, monthly predicts months  
✅ 10 years of historical data  
✅ Technical indicators adjusted per time period  
✅ Separate models optimized for each horizon  
✅ Better prediction accuracy at each time scale  

## Files Generated

```
data/
  ├── multi_sector_stocks.csv           # Daily data (49,668 records)
  ├── multi_sector_stocks_weekly.csv    # Weekly data (10,297 records)
  └── multi_sector_stocks_monthly.csv   # Monthly data (2,385 records)

models/
  ├── TICKER_daily_models.joblib        # Daily predictions (1d, 5d, 21d)
  ├── TICKER_weekly_models.joblib       # Weekly predictions (1w, 4w, 12w)
  └── TICKER_monthly_models.joblib      # Monthly predictions (1m, 3m, 6m)

predictions_multiperiod.csv              # All predictions across time periods
```

## Backward Compatibility

The dashboard supports both:
1. **New multi-period format** (recommended) - `predictions_multiperiod.csv`
2. **Legacy format** - `predictions_summary.csv`

It automatically detects which format is available and adjusts the display accordingly.

## Technical Details

### Time Period Configurations

| Period | Horizons | Moving Averages | RSI Periods | MACD |
|--------|----------|-----------------|-------------|------|
| Daily | 1, 5, 21 days | 5, 10, 20, 50 | 7, 14, 21 | 12/26/9 |
| Weekly | 1, 4, 12 weeks | 4, 8, 13, 26 | 4, 9, 14 | 6/13/4 |
| Monthly | 1, 3, 6 months | 3, 6, 12, 24 | 3, 6, 9 | 3/6/2 |

### Model Details
- Algorithm: LightGBM (gradient boosting)
- Task: Binary classification (price direction)
- Features: 40-50 technical indicators per time period
- Validation: Time-series split (80/20)
- Output: Direction (UP/DOWN) + Probability + Confidence

## Usage Tips

1. **Short-term trading**: Focus on daily predictions (1d, 5d)
2. **Swing trading**: Use weekly predictions (1w, 4w)
3. **Position trading**: Use monthly predictions (1m, 3m, 6m)
4. **Confirmation strategy**: Look for alignment across time periods
5. **High confidence**: Filter predictions with confidence > 0.3

## Automation

To automate daily updates:

```bash
# Update data and predictions
python download_stock_data.py
python aggregate_data.py
python predict_multiperiod.py
```

Add to crontab for daily execution:
```cron
0 18 * * 1-5 cd /path/to/woody_stocks && python download_stock_data.py && python aggregate_data.py && python predict_multiperiod.py
```

## Troubleshooting

### Dashboard shows "No predictions available"
Run: `python predict_multiperiod.py`

### Missing model files
Run: `python train_model_multiperiod.py`

### Missing aggregated data
Run: `python aggregate_data.py`

### Need fresh data
Run: `python download_stock_data.py`

## Next Steps

1. Generate predictions for all stocks
2. Implement weekly/monthly model training (currently only daily is complete)
3. Update backtest scripts to use multi-period models
4. Create ensemble predictions combining multiple time periods
5. Add confidence-weighted recommendations
