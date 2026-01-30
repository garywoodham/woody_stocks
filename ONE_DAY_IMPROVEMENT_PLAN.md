# 1-Day Prediction Improvement Plan
**Focus:** Improve short-term (1-day) accuracy from 53% to 60%+  
**Date:** January 17, 2026

---

## 🎯 Current Situation

**Current 1-Day Accuracy:**
- Average: **53.2%** (barely better than random 50%)
- Best performing: 56-60% (only a few stocks)
- Worst performing: 43-50% (many stocks)

**Problem:**
- Technical indicators alone cannot predict daily movements reliably
- Market is too efficient at daily scale
- Need **alternative data sources** that capture information faster than price

---

## 🚀 Recommended Next Steps (Prioritized)

### **TIER 1: High Impact, Moderate Effort** ⭐⭐⭐

#### 1. **News Sentiment - Real-Time** (Expected: +3-5% accuracy)

**Current Issue:** You have sentiment data but it's aggregated/delayed

**Implementation:**
```python
# Use real-time news sentiment from multiple sources
- NewsAPI (already have)
- Alpha Vantage News Sentiment
- Finnhub News Sentiment
- Twitter/X API (official)

# Key features to add:
- Breaking news in last 1 hour
- Sentiment change in last 24 hours
- News surprise score (actual vs expected)
- Source credibility weighting
```

**Why It Works:**
- News drives immediate price movements
- You have ~4-6 hours reaction time before close
- High sentiment + high volume = strong signal

**Files to Create:**
- `collect_realtime_sentiment.py` - Fetch every hour
- `sentiment_features_1day.py` - 1-day specific features

**Estimated Time:** 1-2 days  
**Cost:** $50-100/month for APIs

---

#### 2. **Pre-Market & After-Hours Signals** (Expected: +2-4% accuracy)

**Concept:** Use overnight price action to predict next-day movement

**Implementation:**
```python
# Features to add:
- Pre-market % change (9:00 AM price vs yesterday close)
- After-hours % change (yesterday)
- Gap size at open
- Pre-market volume vs average
- Futures movement (SPY, QQQ)

# Signal examples:
- Large pre-market gap + high volume = continuation
- Small gap + low volume = mean reversion
- Futures up + stock down in pre-market = buying opportunity
```

**Data Source:**
- yfinance has pre-market data
- Yahoo Finance API
- Interactive Brokers API (free paper trading account)

**Files to Create:**
- `fetch_premarket_data.py`
- `premarket_features.py`

**Estimated Time:** 1 day  
**Cost:** Free

---

#### 3. **Market Regime + VIX Enhancement** (Expected: +2-3% accuracy)

**Current:** You have `detect_market_regime.py` but not fully integrated

**Implementation:**
```python
# Train separate 1-day models for:
1. High volatility (VIX > 25) - Mean reversion works better
2. Low volatility (VIX < 15) - Momentum works better
3. Rising VIX (fear increasing) - Bearish bias
4. Falling VIX (fear decreasing) - Bullish bias

# Additional features:
- VIX term structure (VIX vs VXV)
- Put/call ratio changes
- SKEW index (tail risk)
```

**Model Structure:**
```python
if current_vix > 25:
    model = high_volatility_model
elif current_vix < 15:
    model = low_volatility_model
else:
    model = normal_model
```

**Files to Modify:**
- `train_refined_models.py` - Add regime-specific models
- `predict_refined.py` - Select model based on regime

**Estimated Time:** 2-3 days  
**Cost:** Free

---

### **TIER 2: Medium Impact, Low-Medium Effort** ⭐⭐

#### 4. **Earnings Calendar Integration** (Expected: +1-3% accuracy)

**You already have:** `earnings_calendar.csv`

**Implementation:**
```python
# Features to add:
- Days until earnings (avoid 1-2 days before)
- Historical earnings beat/miss pattern
- Post-earnings drift direction
- Earnings surprise magnitude

# Strategy:
- AVOID predictions 1-2 days before earnings (too unpredictable)
- INCREASE confidence right after earnings (direction clearer)
```

**Files to Create:**
- `earnings_features.py`
- Integrate into `create_optimized_features()`

**Estimated Time:** 0.5-1 day  
**Cost:** Free

---

#### 5. **Intraday VWAP & Order Flow Proxies** (Expected: +1-2% accuracy)

**Concept:** Use intraday patterns without needing tick data

**Implementation:**
```python
# Collect at market close (4 PM):
- Current price vs today's VWAP
- Current price vs day's high/low
- Volume profile (morning vs afternoon)
- Large block trades (>100k shares)

# Signals:
- Price > VWAP + strong close = bullish continuation
- Price < VWAP + weak close = bearish continuation
```

**Data Source:**
- yfinance intraday data (free)
- Volume profile from daily bars

**Files to Create:**
- `fetch_intraday_patterns.py`
- `intraday_close_features.py`

**Estimated Time:** 1 day  
**Cost:** Free

---

#### 6. **Social Media Sentiment** (Expected: +2-5% for meme stocks)

**Target:** TLRY, AMC, GME, PLTR, SOFI, etc.

**Implementation:**
```python
# Data sources:
1. Reddit (r/wallstreetbets, r/stocks)
   - Post volume about ticker
   - Sentiment of top posts
   - Upvote velocity

2. Twitter/X
   - Mentions last 24h
   - Sentiment analysis
   - Influencer mentions

3. StockTwits
   - Bull/bear sentiment
   - Message volume
```

**Tools:**
- PRAW (Reddit API) - Free
- Twitter API - $100/month for basic
- StockTwits API - Free

**Files to Create:**
- `collect_social_sentiment.py`
- `social_features.py`

**Estimated Time:** 2-3 days  
**Cost:** $100/month

---

### **TIER 3: Experimental, Higher Effort** ⭐

#### 7. **Options Flow Analysis** (Expected: +3-7% accuracy)

**Concept:** Follow the smart money

**Implementation:**
```python
# Data needed:
- Unusual options activity
- Large institutional trades
- Put/Call ratio by strike
- Implied volatility changes

# Signals:
- Large call buying = bullish
- Large put buying = bearish
- IV spike = big move coming
```

**Data Source:**
- Tradier API (free delayed options data)
- CBOE data (free put/call ratios)
- Unusual Whales ($50/month)

**Files to Create:**
- `fetch_options_flow.py`
- `options_features.py`

**Estimated Time:** 3-5 days  
**Cost:** $50-200/month

---

#### 8. **Insider Trading Signals** (Expected: +2-4% accuracy)

**Concept:** Follow corporate insiders

**Implementation:**
```python
# Data from SEC Form 4 filings:
- Recent insider buys (bullish)
- Recent insider sells (bearish if unusual)
- Multiple insiders buying = strong signal

# Signals:
- Insider buy in last 7 days = bullish
- Cluster of buys = very bullish
- CEO buying = strongest signal
```

**Data Source:**
- SEC EDGAR API (free)
- FinViz (free, delayed)
- QuiverQuant API ($40/month)

**Files to Create:**
- `fetch_insider_trades.py`
- `insider_features.py`

**Estimated Time:** 2-3 days  
**Cost:** $0-40/month

---

#### 9. **Sector Rotation & Relative Strength** (Expected: +1-3% accuracy)

**Concept:** Trade based on sector momentum

**Implementation:**
```python
# Features:
- Stock performance vs sector ETF (XLF, XLK, XLE, etc.)
- Sector ETF vs SPY
- Relative strength index (not RSI, actual RS)
- Sector rotation signals

# Strategy:
- Strong stock in strong sector = best setup
- Weak stock in weak sector = avoid
```

**Data Source:**
- yfinance (sector ETFs are free)

**Files to Create:**
- `sector_rotation_features.py`

**Estimated Time:** 1-2 days  
**Cost:** Free

---

#### 10. **Deep Learning Model** (Expected: +5-10% accuracy, high variance)

**Only if above fail**

**Implementation:**
```python
# Model options:
1. LSTM - Capture sequential patterns
2. Transformer - Attention mechanism
3. 1D CNN - Pattern recognition

# Pros:
- Can learn complex non-linear patterns
- Handles sequences better than LightGBM

# Cons:
- Needs lots of data
- Easy to overfit
- Slower to train
- Harder to interpret
```

**Estimated Time:** 1-2 weeks  
**Cost:** Compute costs

---

## 📊 Implementation Roadmap

### **Week 1: Quick Wins (Free)**
1. ✅ Pre-market/after-hours data
2. ✅ Market regime models (VIX-based)
3. ✅ Earnings calendar integration
4. ✅ Sector rotation signals

**Expected Gain:** +4-8% accuracy → **57-61%**

---

### **Week 2-3: Alternative Data ($50-200/month)**
5. ✅ Real-time news sentiment
6. ✅ Social media sentiment (meme stocks)
7. ✅ Intraday VWAP patterns

**Expected Gain:** +5-10% accuracy → **62-68%**

---

### **Week 4: Advanced (Optional)**
8. ✅ Options flow analysis
9. ✅ Insider trading signals
10. ⚠️ Deep learning (only if needed)

**Expected Gain:** +3-7% accuracy → **65-72%**

---

## 🎯 Recommended Starting Point

**Start Here (Today):**

### **1. Pre-Market Data + Market Regime Models**

**Why:**
- Free
- Easy to implement
- Proven to work
- 4-5% improvement expected

**Implementation Plan:**

```bash
# Step 1: Create pre-market fetcher
python3 << 'PREMARKET'
import yfinance as yf
import pandas as pd
from datetime import datetime

def fetch_premarket_data(ticker):
    """Fetch pre-market price action"""
    stock = yf.Ticker(ticker)
    
    # Get pre-market data (if available)
    hist = stock.history(period="2d", interval="1m", prepost=True)
    
    # Calculate pre-market gap
    today_premarket = hist[hist.index.hour < 9].tail(10)
    yesterday_close = hist[hist.index.date == hist.index.date[-2]][-1]['Close']
    
    if len(today_premarket) > 0:
        premarket_price = today_premarket['Close'].iloc[-1]
        gap = (premarket_price / yesterday_close - 1) * 100
        volume = today_premarket['Volume'].sum()
        
        return {
            'premarket_gap': gap,
            'premarket_volume': volume,
            'has_premarket_data': True
        }
    
    return {'has_premarket_data': False}

# Test
result = fetch_premarket_data('AAPL')
print(result)
PREMARKET
```

```bash
# Step 2: Create regime-based training
# Modify train_refined_models.py to train 3 models per stock:
# - high_volatility_model (VIX > 25)
# - low_volatility_model (VIX < 15)  
# - normal_model (VIX 15-25)
```

**This alone should get you to 57-58% accuracy within a week.**

---

### **2. Then Add Real-Time Sentiment (Week 2)**

**Why:**
- News drives immediate reactions
- You already have sentiment infrastructure
- Just need to make it real-time

**Implementation:**
```bash
# Create hourly sentiment collector
# Run as cron job: */1 * * * * (every hour)
python collect_realtime_sentiment.py

# Key features:
- Sentiment in last 1 hour
- Sentiment change last 4 hours (market hours)
- Breaking news indicator
- Volume spike + positive sentiment = strong buy
```

**This should get you to 60-62% accuracy.**

---

## 🧪 Testing Protocol

**Before Deploying Any Change:**

1. **Backtest on test data**
   ```bash
   python test_short_term_improvements.py --feature new_feature
   ```

2. **Validate with cross-validation**
   ```bash
   # Should show in train_refined_models.py output
   # Look for CV accuracy, not test accuracy
   ```

3. **Paper trade for 5 days**
   ```bash
   # Track predictions vs actual
   # Measure improvement
   ```

4. **Deploy if:**
   - CV accuracy improves by >1%
   - Paper trading confirms improvement
   - Doesn't break existing functionality

---

## 📈 Expected Results Timeline

| Timeframe | Improvements | Expected Accuracy | Confidence |
|-----------|-------------|-------------------|------------|
| **Current** | None | 53% | Low |
| **Week 1** | Pre-market + Regime | 57-58% | High |
| **Week 2** | + Real-time Sentiment | 60-62% | High |
| **Week 3** | + Social Sentiment | 62-64% | Medium |
| **Week 4** | + Options Flow | 65-68% | Medium |

---

## ⚠️ Reality Check

**Be Skeptical:**
- 1-day predictions are HARD
- Even 60% would be excellent
- Above 65% is rare even for pros
- Don't expect 70%+ from technical data alone

**But Achievable:**
- 57-60% is realistic with effort
- Alternative data makes the difference
- Market regime awareness helps
- Focus on best stocks (already at 56-60%)

---

## 💰 Cost-Benefit Analysis

| Improvement | Cost/Month | Time | Expected Gain | ROI |
|-------------|-----------|------|---------------|-----|
| Pre-market | $0 | 1 day | +4% | ∞ |
| Market Regime | $0 | 2 days | +3% | ∞ |
| Earnings | $0 | 1 day | +2% | ∞ |
| Real-time News | $50 | 2 days | +5% | High |
| Social Media | $100 | 3 days | +3% | Medium |
| Options Flow | $100 | 4 days | +5% | High |

**Best ROI: Start with free improvements first.**

---

## 🎯 Your Action Plan (This Week)

### Monday-Tuesday:
1. Implement pre-market data fetcher
2. Add pre-market features to training
3. Test on 5-10 stocks

### Wednesday-Thursday:
4. Implement regime-based models (VIX)
5. Train 3 models per stock
6. Validate with cross-validation

### Friday:
7. Integrate earnings calendar features
8. Run full backtest
9. Compare before/after

**Expected by end of week:** 57-58% accuracy (up from 53%)

---

## 📝 Next Steps After This Week

**If 57-58% achieved:**
- Add real-time sentiment (Week 2)
- Add social sentiment (Week 3)
- Consider options flow (Week 4)

**If stuck at 53-55%:**
- Deep dive into which stocks improved
- Focus only on those stocks for 1-day
- Use 5-day/21-day for others

---

## 🔍 Monitoring Success

**Track These Metrics:**

```python
# After each improvement:
- CV accuracy (honest metric)
- Accuracy by sector
- Accuracy by stock
- Win rate in paper trading
- Sharpe ratio of signals

# Red flags:
- Test accuracy >> CV accuracy (overfitting)
- Improvement not consistent across stocks
- Paper trading worse than backtest
```

---

## Summary

**Best Next Steps for 1-Day Improvements:**

1. ⭐⭐⭐ **Start with pre-market + regime models** (free, high impact)
2. ⭐⭐⭐ **Add real-time sentiment** (low cost, high impact)
3. ⭐⭐ **Add social sentiment for meme stocks** (medium cost, medium impact)
4. ⭐ **Consider options flow** (higher cost, high impact for certain stocks)

**Realistic Target:** 60-62% by end of month (up from 53%)

**Focus Areas:**
- Financials sector (already at 55%)
- Large cap tech (54%)
- Avoid Pharma (52%, too hard)
