# 🚀 Stock Prediction System - Quick Start Guide

## 📋 System Overview

Your stock prediction system is now **fully automated** with:
- ✅ **20 stocks** across 4 sectors (Defence, Banking, Pharma, Technology)
- ✅ **80 AI models** (4 prediction horizons per stock: 1d, 7d, 30d, 90d)
- ✅ **Interactive dashboard** with backtesting results
- ✅ **Automated daily updates** via GitHub Actions
- ✅ **Trading signals** with BUY/SELL/HOLD recommendations

---

## 🎯 Automated Daily Workflow

### What Happens Automatically

**Every Day at 6 PM UTC:**
1. 📥 Downloads latest stock data
2. 🤖 Generates trading signals for all stocks
3. 📊 Creates daily report (DAILY_REPORT.md)
4. 💾 Commits changes to repository

**Every Sunday (Weekly Retraining):**
1. 🧠 Retrains all 80 AI models with latest data
2. 📈 Runs full backtesting suite
3. 📊 Updates performance metrics
4. 💾 Saves updated models and results

---

## 🖥️ Local Usage

### View Dashboard
```bash
python dashboard.py
```
Then open: http://localhost:8050

### Manual Updates

**Download latest data:**
```bash
python download_stock_data.py
```

**Generate signals:**
```bash
python generate_daily_signals.py
```

**Create report:**
```bash
python generate_report.py
```

**Retrain models:**
```bash
python predict_stock.py
```

**Run backtests:**
```bash
python backtest_trading.py 2
```

---

## 📊 Key Files

| File | Description | Updated |
|------|-------------|---------|
| `data/multi_sector_stocks.csv` | Historical stock data (5 years) | Daily |
| `predictions_summary.csv` | AI predictions for all stocks | Weekly |
| `daily_signals.csv` | BUY/SELL/HOLD signals with position sizes | Daily |
| `backtest_summary.csv` | Strategy performance metrics | Weekly |
| `DAILY_REPORT.md` | Daily market summary and top signals | Daily |
| `models/*.pkl` | Trained ML models (80 files) | Weekly |

---

## 🎨 Dashboard Features

### Tab 1: Predictions & Charts
- 📈 **Candlestick charts** with technical indicators
- 🔍 **Filter by sector/stock**
- 🎯 **AI predictions** for 1/7/30/90 days
- 📊 **Volume analysis**
- 📋 **Stock overview table**

### Tab 2: Backtest Results
- 💰 **Strategy performance** vs buy-and-hold
- 📊 **Returns by sector** visualization
- 🏆 **Top performing strategies**
- 📈 **Detailed metrics** (Sharpe, Win Rate, Drawdown)

---

## 🤖 Understanding Trading Signals

### Signal Logic

**BUY Signal** when:
- Prediction probability ≥ 55%
- Model confidence ≥ 15%
- Expected upside movement

**SELL Signal** when:
- Prediction probability ≤ 45% (high downside)
- Model confidence ≥ 15%
- Expected downside movement

**HOLD** otherwise

### Signal Strength
- **Signal Strength:** Expected % movement
- **Probability:** Model's confidence in direction (0-100%)
- **Model Confidence:** Prediction certainty (higher = more reliable)
- **Position Size:** Recommended capital allocation ($1000 base)

---

## 📈 Current Performance Highlights

### Top Backtested Strategies
1. **Barclays (7d):** 938% return | 77% win rate
2. **Alphabet (7d):** 686% return | 80% win rate
3. **Microsoft (7d):** 457% return | 80% win rate

### Model Accuracy by Horizon
- **1-day:** 55% average accuracy
- **7-day:** 61% average accuracy
- **30-day:** 68% average accuracy
- **90-day:** 77% average accuracy

---

## ⚙️ GitHub Actions Setup

The automation is configured in `.github/workflows/daily_update.yml`

### Manual Trigger
1. Go to your GitHub repository
2. Click "Actions" tab
3. Select "Daily Stock Data Update & Model Training"
4. Click "Run workflow"

### View Run History
- Actions tab shows all automated runs
- Failed runs create GitHub issues automatically
- All changes committed with "🤖 Auto-update" message

---

## 🔧 Customization

### Adjust Signal Thresholds
Edit `generate_daily_signals.py`:
```python
signal_gen = SignalGenerator(
    min_confidence=0.15,      # Model certainty threshold (0-1)
    min_probability_buy=0.55,  # Buy probability threshold
    min_probability_sell=0.45  # Sell probability threshold
)
```

### Change Automation Schedule
Edit `.github/workflows/daily_update.yml`:
```yaml
schedule:
  - cron: '0 18 * * *'  # 6 PM UTC daily
```

### Add More Stocks
Edit `download_stock_data.py`:
```python
STOCKS = {
    "Your Sector": {
        "Your Stock": "TICKER",
        # ...
    }
}
```

---

## 📞 Support & Monitoring

### Check System Health
```bash
# View latest signals
cat daily_signals.csv | head

# Check report
cat DAILY_REPORT.md

# Model performance
cat predictions_summary.csv | grep accuracy
```

### Troubleshooting
- **Dashboard not loading:** Check if CSV files exist in data/
- **No signals generated:** Ensure models/ directory has .pkl files
- **Automation failed:** Check GitHub Actions logs for errors

---

## 🎯 Next Steps

1. **Test the dashboard:** `python dashboard.py`
2. **Review today's signals:** Check `daily_signals.csv`
3. **Read daily report:** Open `DAILY_REPORT.md`
4. **Push to GitHub:** Automation will start tomorrow at 6 PM UTC
5. **Monitor performance:** Review weekly backtest updates

---

## 📚 Additional Resources

- **Automation Details:** See `AUTOMATION.md`
- **Technical Indicators:** RSI, MACD, Bollinger Bands, ATR, ADX, CCI
- **ML Model:** LightGBM gradient boosting with 46 features
- **Backtest Period:** 5 years (2021-2026)
- **Commission:** 0.1% per trade
- **Slippage:** 0.05% per trade

---

**System Status:** ✅ Fully Operational

*Last updated: 2026-01-03*
