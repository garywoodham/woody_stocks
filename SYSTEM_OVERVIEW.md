# Stock Prediction & Portfolio Management System

Complete automated system for stock predictions, recommendations, and portfolio management.

## 📊 System Components

### 1. **Prediction Pipeline**
- **Script**: `predict_refined.py`
- **Output**: `predictions_refined.csv`
- **Function**: Generates 1-day, 5-day, and 21-day predictions for 20 stocks
- **Features**: 55 technical indicators, LightGBM models, accuracy tracking

### 2. **Recommendation Engine**
- **Script**: `generate_recommendations.py`
- **Output**: `stock_recommendations.csv`
- **Function**: Converts predictions into BUY/HOLD/SELL recommendations
- **Logic**: 
  - Combines 1d (20%), 5d (30%), 21d (50%) predictions
  - Weights by confidence levels
  - Score range: -1.0 (strong sell) to +1.0 (strong buy)
  - **BUY**: Score > 0.05 | **HOLD**: -0.05 to 0.05 | **SELL**: Score < -0.05

### 3. **Portfolio Manager**
- **Script**: `portfolio_manager.py`
- **Output**: `portfolio_allocation.csv`
- **Function**: Allocates capital across BUY recommendations
- **Features**:
  - Position sizing based on recommendation strength
  - Max 15% per stock, 35% per sector
  - Maintains 10% cash reserve
  - Diversification rules
  - **Default capital**: $100,000

### 4. **Backtesting**
- **Script**: `backtest_recommendations.py`
- **Output**: `backtest_recommendations.csv`
- **Function**: Tests historical performance of recommendations
- **Metrics**: Total return, win rate, Sharpe ratio, drawdowns

### 5. **Dashboard**
- **Script**: `dashboard.py`
- **URL**: http://localhost:8050
- **Features**:
  - Interactive charts with technical indicators
  - Stock recommendations with visual signals
  - Trading signals by horizon (1d, 5d, 21d)
  - Backtest results
  - Sector analysis

### 6. **Automation**
- **Workflow**: `.github/workflows/daily_update.yml`
- **Schedule**: 
  - **Daily** (6 PM UTC): Predictions → Recommendations → Portfolio → Signals
  - **Weekly** (Sundays): Model retraining + Backtesting
- **Auto-commits**: Updates pushed to GitHub automatically

## 🚀 Quick Start

### Daily Workflow (Manual)
```bash
# Run complete pipeline
./run_daily_workflow.sh

# With backtesting (slower)
./run_daily_workflow.sh --backtest
```

### Individual Components
```bash
# 1. Generate predictions
python predict_refined.py

# 2. Generate recommendations
python generate_recommendations.py

# 3. Generate portfolio
python portfolio_manager.py

# 4. Run backtest
python backtest_recommendations.py

# 5. Launch dashboard
python dashboard.py
```

## 📈 Current Portfolio Status

**Latest Allocation** (from portfolio_allocation.csv):
- **Total Capital**: $100,000
- **Invested**: ~$65,000 (65%)
- **Cash Reserve**: ~$35,000 (35%)
- **Positions**: 9 stocks
- **Sectors**: Technology (34%), Defence (16%), Pharma (15%)

**Top Holdings**:
1. Pfizer (PFE) - $14,831 - Strong BUY
2. NVIDIA (NVDA) - $8,876 - Strong BUY
3. Microsoft (MSFT) - $7,094 - Moderate BUY

## 📊 Performance Metrics

### Latest Recommendations (stock_recommendations.csv)
- **BUY**: 9 stocks (45%)
- **HOLD**: 10 stocks (50%)
- **SELL**: 1 stock (5%)

**Strongest BUYs**:
- Pfizer: Score 0.3221 (3/3 UP)
- NVIDIA: Score 0.1899 (3/3 UP)
- Microsoft: Score 0.1395 (3/3 UP)

### Backtest Results (backtest_recommendations.csv)
- **Avg Return**: 2.83% (over test period)
- **Best Performer**: Alphabet (33.48%)
- **Win Rate**: Varies by stock
- **Sharpe Ratio**: 0.43 average

## 🔧 Configuration

### Portfolio Manager Settings
```python
PortfolioManager(
    total_capital=100000,      # $100k portfolio
    max_position_pct=0.15,     # Max 15% per stock
    max_sector_pct=0.35,       # Max 35% per sector
    min_score=0.05,            # Min BUY score threshold
    cash_reserve=0.10          # Keep 10% cash
)
```

### Recommendation Thresholds
- **Strong Signal**: |Score| > 0.15
- **Moderate Signal**: 0.05 < |Score| < 0.15
- **Weak/Neutral**: |Score| < 0.05

## 📁 Key Files

### Data Files
- `predictions_refined.csv` - AI predictions for all stocks
- `stock_recommendations.csv` - BUY/HOLD/SELL recommendations
- `portfolio_allocation.csv` - Optimal portfolio allocation
- `backtest_recommendations.csv` - Backtest results
- `daily_signals.csv` - Trading signals

### Model Files
- `models/*_daily_refined.joblib` - Trained LightGBM models (20 stocks)

### Scripts
- `train_refined_models.py` - Train/retrain models
- `predict_refined.py` - Generate predictions
- `generate_recommendations.py` - Generate recommendations
- `portfolio_manager.py` - Portfolio allocation
- `backtest_recommendations.py` - Backtest system
- `dashboard.py` - Interactive dashboard
- `run_daily_workflow.sh` - Complete automation

## 🔄 Automation Schedule

### Daily (6 PM UTC)
1. Download latest stock data
2. Generate predictions
3. Create recommendations
4. Allocate portfolio
5. Generate trading signals
6. Commit results to GitHub

### Weekly (Sundays)
1. All daily tasks +
2. Retrain all models
3. Run comprehensive backtests
4. Performance analysis

## 🎯 Next Steps

### Potential Enhancements
1. **Email/Slack Alerts**
   - Daily recommendation summaries
   - Strong BUY/SELL alerts
   - Portfolio rebalancing notifications

2. **Advanced Backtesting**
   - Walk-forward optimization
   - Monte Carlo simulation
   - Regime detection

3. **Risk Management**
   - Stop-loss automation
   - Volatility-based position sizing
   - Correlation analysis

4. **Expansion**
   - More stocks/sectors
   - Options strategies
   - International markets

## 📚 Documentation

- `QUICK_START.md` - Getting started guide
- `AUTOMATION.md` - Automation details
- `DASHBOARD_USAGE.md` - Dashboard features
- `MODEL_PERFORMANCE_REPORT.md` - Model accuracy

## 🔐 Security Notes

- API keys stored in environment variables (not committed)
- GitHub Actions uses repository secrets
- No sensitive data in version control

## 📞 Support

For issues or questions:
1. Check documentation files
2. Review GitHub Actions logs
3. Check dashboard for real-time status

---

**Last Updated**: $(date)
**System Status**: ✅ Operational
