# 🎯 Woody Stocks - Project Status Summary
**Date:** January 12, 2026  
**Created for:** Quick context and starting point

---

## 📋 What This Project Is

A **stock prediction trading system** that uses machine learning (LightGBM) to predict stock price movements across multiple time horizons (1-day, 5-day, 21-day) for 35 stocks across various sectors.

---

## 🎯 Current System Status

### ✅ What's Working
- **35 stocks** being tracked across Defence, Banking, Pharma, Tech, and Meme sectors
- **105 ML models** trained (35 stocks × 3 horizons)
- **Market regime detection** (Currently: BULL market, 100% confidence)
- **Earnings calendar integration** (3-day safety buffer)
- **Stock quality tiers** (4-tier system based on accuracy)
- **Interactive dashboard** available
- **Automated filtering** system in place

### 📊 Current Performance
**Overall Average:** 55.08%

By horizon:
- 1-day predictions: **52.10%** (barely better than random)
- 5-day predictions: **53.31%** (slightly better)
- 21-day predictions: **59.83%** (best performance) ⭐

### 🏆 Best Performing Models
1. **Standard Chartered (STAN.L)** - 21-day: **78.66%** ✨
2. **GOOGL** - 21-day: **69.73%**
3. **HSBC** - 21-day: **67.89%**
4. **HSBC** - 5-day: **65.65%**
5. **AMC** - 21-day: **65.64%**

### ⚠️ Worst Performing Models
- **Rolls-Royce (RR.L)** - 21-day: **24.80%**
- **Lloyds Banking** - 21-day: **38.62%**
- Several models below 45% (avoid trading these)

---

## 🔥 Recent Improvements Implemented

### 1. **Volume Enhancement** (Jan 5)
- Added **15 advanced volume indicators**
- Key features: MFI, CMF, AD (Accumulation/Distribution)
- Result: +0.65% improvement to 55.66%

### 2. **Market Regime + Earnings Integration** (Jan 7) - MOST RECENT
- Added **9 regime features** (market_regime, regime_confidence, etc.)
- Added **5 earnings features** (days_to_earnings, is_earnings_week, etc.)
- Result: 
  - 21-day: **+1.80%** improvement ⭐
  - 1-day/5-day: slight decline (~-1.4%)
  - Overall: -0.32% (mixed results)

**Key Finding:** Earnings features work VERY well for 21-day predictions (position trading), but not as helpful for short-term trading.

---

## 🎨 System Architecture

```
Data Sources:
├── Stock OHLCV data (5 years history)
├── Market regime history (303 days) → current_market_regime.json
├── Earnings calendar (35 stocks) → earnings_calendar.csv
├── SPY & VIX data (market context)
└── Sentiment data (building organically, currently 0% importance)

Features (per model):
├── 55 technical indicators (RSI, MACD, Bollinger, etc.)
├── 15 volume features (MFI, CMF, AD, OBV, etc.)
├── 9 market regime features
├── 5 earnings features
├── 7 market context features (SPY, VIX)
└── ~99 total features per model

Models:
├── 105 LightGBM models (35 stocks × 3 horizons)
└── Saved in /workspaces/woody_stocks/models/*.joblib

Filtering System:
├── Layer 1: Stock Quality (4 tiers by accuracy)
├── Layer 2: Earnings Safety (skip if <3 days to earnings)
├── Layer 3: Market Regime (BULL=trade, BEAR=hold)
└── Layer 4: Position Sizing (tier × regime multiplier)
```

---

## 🚀 How to Use the System

### Quick Daily Workflow
```bash
# Generate predictions and trading plan
./run_daily_trading.sh

# View results
cat todays_trading_plan.csv
```

### Individual Commands
```bash
# 1. Generate predictions
python3 generate_daily_signals.py

# 2. Update earnings calendar
python3 create_earnings_calendar.py

# 3. Filter top stocks (quality tier)
python3 filter_top_stocks.py

# 4. Detect market regime
python3 detect_market_regime.py

# 5. View dashboard
python3 dashboard.py
```

### Key Output Files
- `predictions_refined.csv` - Raw predictions (all 35 stocks)
- `predictions_top_stocks.csv` - Quality filtered (Tier 1-2 only)
- `predictions_regime_filtered.csv` - Market regime filtered
- `todays_trading_plan.csv` - **FINAL TRADING PLAN** (top 10 stocks)
- `current_market_regime.json` - Current market status
- `earnings_calendar.csv` - Upcoming earnings dates

---

## 📈 Improvement Roadmap

### ✅ Completed
1. ✅ Volume Enhancement (15 features) - +0.65%
2. ✅ Market Regime Integration (9 features)
3. ✅ Earnings Calendar Integration (5 features) - +1.80% for 21-day

### 🎯 Next Priority (From Improvement Plan)

**Priority #3: Cross-Stock Correlation Features** 🎯
- Expected improvement: +1 to 1.5%
- Effort: Medium (45-60 minutes)
- Features to add:
  - Sector daily returns
  - Stock vs sector relative strength
  - Correlation to sector leaders
  - Cross-stock momentum signals

**Priority #4: Volatility Regime Classification**
- Expected improvement: +1%
- Better than binary BULL/BEAR/SIDEWAYS
- VIX-based buckets: Low (<15), Normal (15-25), High (>25)

**Priority #5: Ensemble Stacking**
- Expected improvement: +1-2%
- Stack predictions from multiple models
- Reduce overfitting

### 🔬 Long-term Goals
- **Target accuracy:** 70-75% (currently 55%)
- **Professional grade:** 80%+ (theoretical maximum ~85%)

---

## 🧠 Key Insights & Lessons Learned

### What Works ✅
1. **21-day predictions are most accurate** (59.83% vs 52-53% for short-term)
2. **Earnings features are powerful** - 22.9% of models use them in top 5 features
3. **Volume indicators matter** - MFI, CMF, AD consistently rank high
4. **Stock quality varies greatly** - Focus on high-accuracy stocks (STAN.L, GOOGL, HSBC)

### What Doesn't Work ❌
1. **Sentiment features** - Currently 0% importance (need better data sources)
2. **Market regime features** - Only 1% usage (needs more regime changes in history)
3. **Short-term predictions** - Still only slightly better than random
4. **Low-quality stocks** - Some stocks just aren't predictable (RR.L, LLOY.L)

### Critical Findings 🔍
1. **Earnings dates are predictable events** - Models learn to avoid/exploit volatility
2. **Volume features > sentiment** (for now) - 2-4% importance vs 0%
3. **Feature interactions help** - mfi_rsi_divergence ranks #3 overall
4. **Not all stocks are predictable** - Focus portfolio on proven winners

---

## 🎯 Current Market Status

**Date:** January 7, 2026  
**Regime:** BULL (100% confidence)  
**Score:** 75.0  
**Should Trade:** YES  
**Position Multiplier:** 1.0x (full allocation)

**Signals:**
- ✅ Above 200 SMA
- ✅ Above 50 SMA
- ✅ Golden Cross
- ✅ 20 SMA > 50 SMA
- ✅ MACD bullish

**Current Prices:**
- SPY: $692.28
- VIX: 15.04 (low volatility)

---

## 🛠️ Technical Stack

- **Language:** Python 3.12.1
- **ML Framework:** LightGBM (primary), XGBoost, RandomForest
- **Data:** yfinance, pandas, numpy
- **Visualization:** Plotly, Dash
- **Features:** 99 total (technical + volume + regime + earnings)
- **Models:** 58 .joblib files (some legacy, 105 active models)

---

## 📁 Important Files

### Configuration
- `stock_tiers.json` - Stock quality classification (4 tiers)
- `earnings_calendar.json` - Earnings dates for all stocks
- `current_market_regime.json` - Current market status

### Data
- `data/multi_sector_stocks.csv` - Historical stock data
- `data/spy_data.csv` - S&P 500 data
- `data/vix_data.csv` - VIX volatility data
- `market_regime_history.csv` - 303 days of regime history

### Models
- `models/*.joblib` - 58 trained models
- `models/multitimeframe/` - Multi-timeframe ensemble models
- `models/sector_specific/` - Sector-specific models

### Results
- `model_results_with_regime_earnings.csv` - Latest accuracy results
- `feature_importance_summary.csv` - Feature importance analysis
- `backtest_summary.csv` - Historical performance

### Documentation
- `FINAL_SUMMARY.md` - Volume enhancement results
- `REGIME_EARNINGS_IMPLEMENTATION_SUMMARY.md` - Most recent changes (Jan 7)
- `ACCURACY_IMPROVEMENT_PLAN.md` - Future improvement roadmap
- `QUICK_REFERENCE.md` - Command cheat sheet
- `QUICK_START.md` - Getting started guide

---

## ⚠️ Known Issues

### Major Issues
1. **Class imbalance problem** - Models predict 100% DOWN, 0% UP
   - Happens when train/test distributions differ
   - Using `scale_pos_weight` to mitigate
   - Still an ongoing challenge

2. **Short-term predictions barely beat random** (52-53%)
   - Market noise dominates at 1-day horizon
   - Need more features or different approach

3. **Sentiment data useless** (0% importance)
   - Current sources not informative
   - Need better sentiment sources (Twitter, news, etc.)

### Minor Issues
1. Some stocks consistently underperform (<40% accuracy)
2. Market regime features underutilized (only 1% usage)
3. Dashboard may have stale data if not regenerated

---

## 🎓 Next Steps & Recommendations

### Immediate Actions
1. **Focus on 21-day predictions** - Best performance, earnings features work well
2. **Implement Cross-Stock Correlation features** (Priority #3 from roadmap)
3. **Filter out low-quality stocks** (<45% accuracy) from trading
4. **Use earnings calendar** to avoid volatile periods

### Medium-term (This Week)
1. Implement Volatility Regime classification
2. Add ensemble stacking (multiple model types)
3. Better hyperparameter tuning
4. Walk-forward validation for robust testing

### Long-term (This Month)
1. Better sentiment data sources (Twitter API, news APIs)
2. Deep learning models (LSTM/Transformer)
3. Additional data sources (options, insider trading)
4. Production deployment with automated trading

---

## 💡 Quick Tips

### Trading Strategy
- **Focus on Tier 1 stocks:** STAN.L, GOOGL, HSBC (65%+ accuracy)
- **Use 21-day predictions:** Most reliable (59.83% avg)
- **Avoid earnings weeks:** Skip stocks within 3 days of earnings
- **Bull market:** Trade actively (current regime)
- **Position sizing:** Larger positions for high-accuracy stocks

### Model Retraining
- **Weekly:** Download new data and regenerate predictions
- **Monthly:** Retrain models with latest data
- **Quarterly:** Review and update feature engineering

### Monitoring
- Track predictions vs actual results
- Monitor accuracy trends over time
- Watch for regime changes (BULL → SIDEWAYS → BEAR)
- Update earnings calendar monthly

---

## 📞 Quick Reference Commands

```bash
# Daily workflow
./run_daily_trading.sh

# Manual steps
python3 generate_daily_signals.py      # Predictions
python3 create_earnings_calendar.py    # Earnings
python3 detect_market_regime.py        # Market regime
python3 filter_top_stocks.py           # Quality filter

# View results
cat todays_trading_plan.csv            # Trading plan
cat current_market_regime.json         # Market status
python3 dashboard.py                   # Dashboard

# Model training
python3 train_refined_models.py        # Standard training
python3 train_multitimeframe_ensemble.py  # Advanced (recommended)

# Analysis
python3 analyze_feature_importance.py  # Feature analysis
python3 backtest_trading.py            # Performance testing
python3 compare_model_performance.py   # Model comparison
```

---

## 🎯 Summary

**You have:** A working stock prediction system with 105 ML models covering 35 stocks across 3 time horizons.

**Current accuracy:** 55% overall (21-day: 59.83%, 1-day: 52.10%)

**Recent work:** Just implemented market regime + earnings features (Jan 7) with +1.80% improvement on 21-day predictions.

**Next priority:** Implement cross-stock correlation features for expected +1-1.5% improvement.

**Target:** Reach 70-75% accuracy (professional grade) through systematic feature engineering and ensemble methods.

**Best stocks:** STAN.L (78.66%), GOOGL (69.73%), HSBC (67.89%)

**Current market:** BULL regime, safe to trade, VIX at 15 (low volatility)

---

*This document provides a comprehensive starting point for understanding the woody_stocks project. For specific details, refer to the individual documentation files listed above.*
