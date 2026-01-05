# 🎯 Performance Tracking & Analytics - Quick Reference

## ✅ What We Built

### 1. **Performance Tracking System** 📊
Tracks if predictions actually work by comparing predicted vs actual outcomes.

**Features:**
- Logs daily predictions with timestamps
- Evaluates predictions after maturity (1/5/21 days)
- Calculates accuracy by direction (UP vs DOWN)
- Measures actual returns
- Analyzes high-confidence predictions separately
- Sector-level performance breakdown

**Files Created:**
- `track_performance.py` - Main tracking script
- `data/predictions_log.csv` - Historical predictions log
- `data/performance_summary.csv` - Aggregated accuracy metrics

---

### 2. **Sentiment Analytics Dashboard** 📰
Visualize sentiment data and trends across all stocks.

**Features:**
- Current sentiment distribution histogram
- Sentiment by sector comparison
- Time-series trend chart (when history available)
- Top positive/negative stocks
- Summary statistics

**What You See:**
- Average sentiment score across portfolio
- Positive/Neutral/Negative stock counts
- Sector sentiment comparison (which sectors are bullish/bearish)
- Historical sentiment trends (builds over time)

---

### 3. **Performance Tracking Dashboard** 📈
Monitor prediction accuracy and model performance.

**Features:**
- Accuracy trends over time by horizon
- Metrics cards (1d/5d/21d accuracy)
- UP vs DOWN prediction accuracy
- Average return analysis
- Recent predictions log

**What You See:**
- How accurate models are for each timeframe
- Whether models are improving over time
- Which predictions are working best
- Actual returns from predictions

---

## 📊 Dashboard Overview

### **5 Tabs Now Available:**

1. **📊 Predictions & Charts**
   - Stock price charts with support/resistance zones
   - Technical indicators (SMA, volume)
   - Multi-period predictions
   - BUY/HOLD/SELL recommendations
   - Full stock overview table

2. **📰 Sentiment Analytics** 🆕
   - Sentiment distribution across stocks
   - Sector sentiment comparison
   - Time-series sentiment trends
   - Top bullish/bearish stocks

3. **📈 Performance Tracking** 🆕
   - Prediction accuracy tracking
   - Accuracy trends over time
   - Return analysis
   - Predictions log

4. **🚦 Trading Signals**
   - Daily BUY/HOLD/SELL signals
   - Signal strength ranking
   - Top opportunities by horizon

5. **🎯 Backtest Results**
   - Historical trading performance
   - Sharpe ratios
   - Win rates
   - Maximum drawdown

---

## 🚀 How to Use

### **Daily Routine**

**Morning Check:**
```bash
# 1. System health
python check_health.py

# 2. View dashboard
python dashboard.py
# Then open http://localhost:8050
```

**What to Look At:**
1. **Sentiment Analytics Tab** - Check market mood
   - Is sector sentiment changing?
   - Any major sentiment shifts overnight?
   - Which stocks have positive news?

2. **Performance Tracking Tab** - Review accuracy
   - Are predictions getting better?
   - Which timeframe (1d/5d/21d) is most accurate?
   - Should I trust high-confidence predictions more?

3. **Predictions Tab** - Today's opportunities
   - Top BUY recommendations
   - Support/resistance levels on charts
   - Sector trends

---

### **Manual Operations**

**Log Today's Predictions:**
```bash
python track_performance.py 1
```
Saves today's predictions for future evaluation.

**Evaluate Past Predictions:**
```bash
python track_performance.py 2
```
Compares old predictions to actual outcomes.

**Both (Recommended):**
```bash
python track_performance.py 3
# Or just run with no args (defaults to 3)
python track_performance.py
```

**Update Sentiment:**
```bash
python fetch_sentiment_historical.py
```
Fetches latest sentiment, appends to history.

**System Health:**
```bash
python check_health.py
```
Quick overview of all system components.

---

## 📈 Understanding Performance Metrics

### **Accuracy**
- **Overall**: % of correct UP/DOWN predictions
- **UP Accuracy**: How often UP predictions are correct
- **DOWN Accuracy**: How often DOWN predictions are correct
- **High Confidence**: Accuracy when confidence > 15%

**Good Performance:**
- Overall accuracy > 55% (better than coin flip)
- High confidence accuracy > 60%
- Consistent across horizons

### **Returns**
- **Avg Return**: Average price change across all predictions
- **Avg Return (correct)**: Return when prediction was right
- **Avg Return (wrong)**: Return when prediction was wrong

**What to Expect:**
- Positive avg return on correct predictions
- Correct predictions should have higher returns than wrong ones
- Meme stocks may have higher volatility

### **By Horizon**
- **1-day**: Most volatile, harder to predict
- **5-day**: Good balance of signal vs noise
- **21-day**: Longer trends, more stable

---

## 🎯 What to Watch For

### **Positive Signals**
✅ **Accuracy trending up** - Models learning
✅ **High confidence working** - Trust strong signals
✅ **Consistent sector performance** - Sector-specific patterns emerging
✅ **Positive returns on correct predictions** - Good risk/reward

### **Warning Signs**
⚠️ **Accuracy declining** - May need retraining
⚠️ **High confidence failing** - Over-confident models
⚠️ **Sector divergence** - Some sectors not working
⚠️ **Negative returns on correct predictions** - Risk management issue

---

## 📅 Data Timeline

### **Day 1 (Today)**
- ✅ Predictions logged
- ✅ Sentiment tracked (1 day)
- ⏳ Performance: waiting for maturity

### **Day 2**
- ✅ 1-day predictions can be evaluated
- ✅ Sentiment: 2 days history
- ✅ First accuracy metrics available

### **Day 6**
- ✅ 5-day predictions can be evaluated
- ✅ Sentiment: 6 days history
- ✅ Better accuracy metrics

### **Day 22**
- ✅ 21-day predictions can be evaluated
- ✅ Sentiment: 22 days history
- ✅ Full performance tracking active
- ✅ Sentiment trends visible

### **Month 1**
- ✅ 30 days of performance data
- ✅ 30 days of sentiment history
- ✅ Meaningful accuracy trends
- ✅ Sentiment momentum features active

---

## 🤖 Automation

**GitHub Actions runs daily at 6 PM UTC:**

1. Downloads latest stock data
2. Fetches sentiment (builds history)
3. Generates predictions
4. **Logs predictions** 🆕
5. **Evaluates past predictions** 🆕
6. Updates recommendations
7. Recalculates portfolio

**Weekly (Sundays at 2 AM UTC):**
- Retrains models with accumulated data
- Backtests strategies

**You don't need to do anything** - system builds performance and sentiment history automatically!

---

## 💡 Pro Tips

### **Using Sentiment**
1. **Sentiment + Technical Agreement** = Stronger signal
   - BUY + Positive sentiment = High confidence
   - SELL + Negative sentiment = Avoid
   
2. **Sentiment Divergence** = Opportunity
   - Price down + Positive sentiment = Potential buy
   - Price up + Negative sentiment = Take profits

3. **Sentiment Momentum** = Trend confirmation
   - Improving sentiment = Bullish
   - Declining sentiment = Bearish

### **Using Performance Tracking**
1. **Trust the data** - If 1d accuracy is 50%, don't overtrade short-term
2. **Follow high confidence** - If high-conf accuracy > 65%, focus on those
3. **Sector allocation** - Invest more in sectors with better performance
4. **Timeframe selection** - Trade the horizon with best accuracy for each stock

### **Combining Both**
1. Check sentiment analytics for market mood
2. Review performance to know which predictions to trust
3. Use predictions tab for specific opportunities
4. Verify with support/resistance zones
5. Check portfolio allocation recommendations

---

## 🎓 Learning Over Time

### **Week 1: Establish Baseline**
- Get comfortable with dashboard
- Understand metrics
- Note initial accuracy

### **Weeks 2-4: Pattern Recognition**
- Which stocks perform best?
- Which sectors are predictable?
- How does sentiment correlate with returns?

### **Month 2+: Optimization**
- Adjust position sizes based on performance
- Focus on high-performing sectors
- Use sentiment momentum for timing

---

## 🔧 Troubleshooting

**"No performance data available"**
- Need 1+ days for 1d evaluation
- Need 5+ days for 5d evaluation
- Need 21+ days for 21d evaluation
- Solution: Wait or run `python track_performance.py 1` to start logging

**"No sentiment data available"**
- Run: `python fetch_sentiment_historical.py`
- Check API key: `echo $NEWS_API_KEY`
- Verify API limit not exceeded (100/day)

**"Dashboard not showing new tabs"**
- Restart: `pkill -f dashboard.py && python dashboard.py`
- Check browser cache (hard refresh: Ctrl+Shift+R)

**"Accuracy seems wrong"**
- Check if enough time passed for evaluation
- Verify predictions_log.csv has data
- Run evaluation: `python track_performance.py 2`

---

## 📊 Example Workflows

### **Morning Trading Routine**
```bash
# 1. Health check
python check_health.py

# 2. Launch dashboard
python dashboard.py &
open http://localhost:8050

# 3. Check sentiment analytics
# - Look for overnight sentiment changes
# - Note sector trends

# 4. Check performance tracking
# - Review yesterday's accuracy
# - Check high-confidence predictions

# 5. Review predictions
# - Check BUY recommendations
# - Verify with support/resistance
# - Cross-reference with sentiment

# 6. Make trading decisions
```

### **Weekly Review**
```bash
# 1. Check cumulative performance
python track_performance.py 2

# 2. Review sentiment trends
# - View sentiment analytics tab
# - Note momentum changes

# 3. Evaluate model performance
# - Which stocks to focus on?
# - Which timeframes working best?
# - Any sector rotation needed?

# 4. Adjust strategy accordingly
```

---

## 🎯 Success Metrics

**After 1 Week:**
- [ ] Predictions logged for 7 days
- [ ] 1d accuracy calculated
- [ ] 5d accuracy starting to calculate
- [ ] Sentiment history building

**After 1 Month:**
- [ ] Full performance metrics available
- [ ] Accuracy trends visible
- [ ] Sentiment patterns emerging
- [ ] High-confidence strategy validated

**After 3 Months:**
- [ ] 2-5% accuracy improvement
- [ ] Clear sector preferences identified
- [ ] Sentiment-price correlation understood
- [ ] Profitable trading strategy developed

---

## 🚀 Next Enhancements

**Potential Additions:**
1. Email/SMS alerts for strong signals
2. Portfolio optimization based on performance
3. Correlation analysis (sentiment vs returns)
4. Machine learning for optimal position sizing
5. Earnings calendar integration
6. Social media sentiment (Twitter/Reddit)

---

## 📞 Quick Reference Commands

```bash
# System Health
python check_health.py

# Dashboard
python dashboard.py

# Performance Tracking
python track_performance.py        # Both log and evaluate
python track_performance.py 1      # Log only
python track_performance.py 2      # Evaluate only

# Sentiment
python fetch_sentiment_historical.py

# Full Update Pipeline
python download_stock_data.py
python fetch_sentiment_historical.py
python train_refined_models.py
python predict_refined.py
python generate_recommendations.py
python portfolio_manager.py
python track_performance.py

# Dashboard Access
http://localhost:8050
```

---

✅ **You now have professional-grade analytics and performance tracking!** 🎉

The system will automatically build performance and sentiment history. Just check the dashboard daily and watch your predictions improve over time!
