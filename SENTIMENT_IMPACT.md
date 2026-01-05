# 📰 Sentiment Analysis Impact Report

## Implementation Summary

**Date Integrated:** January 5, 2026
**Sentiment Source:** News API + VADER
**Stocks Analyzed:** 20 (Banking, Defence, Pharma, Technology)

---

## Current Sentiment Snapshot

### Overall Market Sentiment
- **Average Sentiment:** +0.278 (POSITIVE)
- **Distribution:** 20 POSITIVE, 0 NEUTRAL, 0 NEGATIVE
- **Average News Coverage:** 22.6 articles/stock
- **Analysis Window:** Last 7 days

### Top Positive Sentiment
1. **BAE Systems (BA.L):** +0.584 (12 articles)
2. **Barclays (BARC.L):** +0.407 (20 articles)  
3. **Lloyds Banking (LLOY.L):** +0.395 (11 articles)

### Least Positive Sentiment
1. **GSK (GSK.L):** +0.142 (11 articles)
2. **Johnson & Johnson (JNJ):** +0.145 (27 articles)
3. **Pfizer (PFE):** +0.163 (19 articles)

---

## Features Added to Models

### 7 New Sentiment Features

1. **sentiment_score** - Overall news sentiment (-1 to +1)
   - Direct signal: positive = bullish, negative = bearish
   
2. **news_volume** - Normalized article count
   - High volume = increased attention/volatility
   
3. **positive_ratio** - Percentage of bullish articles
   - Measures consensus strength
   
4. **negative_ratio** - Percentage of bearish articles
   - Risk/concern indicator
   
5. **latest_sentiment** - Most recent article sentiment
   - Captures breaking news impact
   
6. **sentiment_rsi** - Sentiment × RSI interaction
   - Combines fundamental + technical signals
   
7. **sentiment_momentum** - Sentiment × 1-day price momentum
   - Detects confirmation/divergence

**Total Features:** 55 technical + 7 sentiment = **62 features**

---

## Integration Points

### Training Pipeline
- `train_refined_models.py` loads `sentiment_data.csv`
- Sentiment features added to all 20 stock models
- Models trained with 62-feature dataset
- Feature importance analysis includes sentiment

### Prediction Pipeline
- `predict_refined.py` uses latest sentiment data
- Sentiment updated before each prediction run
- Fresh sentiment = better real-time signals

### Automation
- `run_daily_workflow.sh` fetches sentiment first
- GitHub Actions runs `fetch_sentiment.py` daily
- Sentiment data committed to repo for tracking

---

## Expected Impact

### Hypothesis: Sentiment Improves Short-Term Predictions

**Why sentiment should help:**
- News drives short-term price movements
- Sentiment captures market psychology
- Combines with technical indicators for stronger signals

**Where it should help most:**
- ✅ 1-day predictions (immediate news impact)
- ✅ 5-day predictions (short-term trend)
- ⚠️ 21-day predictions (technical factors dominate longer term)

**Best use cases:**
- Earnings announcements
- Product launches
- Regulatory news
- Executive changes
- Market-moving events

---

## Testing Plan

### Phase 1: Baseline Comparison (Current Models)
```bash
# Run predictions with existing models + sentiment data
python predict_refined.py
```

**Compare:**
- Accuracy before sentiment integration (from last run)
- Accuracy with sentiment features (this run)
- Change in recommendation distribution

### Phase 2: Retrained Models
```bash
# Retrain all models with sentiment features
python train_refined_models.py
```

**Measure:**
- Feature importance rankings (are sentiment features used?)
- Cross-validation accuracy improvement
- UP/DOWN prediction balance

### Phase 3: Live Performance
```bash
# Run complete workflow for 1 week
./run_daily_workflow.sh
```

**Track:**
- Daily sentiment changes
- Prediction accuracy over time
- Correlation: sentiment → recommendations → returns

---

## Success Metrics

### Immediate (Week 1)
- ✅ Sentiment data fetches successfully daily
- ✅ Models train without errors
- ✅ Predictions include sentiment features

### Short-term (Month 1)
- 🎯 Target: +2-5% accuracy improvement on 1d/5d predictions
- 🎯 Sentiment features in top 20 by importance
- 🎯 Better early detection of trend reversals

### Long-term (Quarter 1)
- 🎯 Improved Sharpe ratio in backtests
- 🎯 Higher win rate on BUY recommendations
- 🎯 Better portfolio returns vs baseline

---

## Monitoring & Maintenance

### Daily Checks
```bash
# Verify sentiment data is fresh
ls -lh sentiment_data.csv

# Check API usage
grep "Total Stocks Analyzed" <(python fetch_sentiment.py)
```

### Weekly Reviews
- Sentiment vs price correlation analysis
- News coverage gaps (stocks with 0 articles)
- API rate limit status (40/100 requests used)

### Monthly Analysis
- Feature importance trends
- Sentiment prediction accuracy by stock/sector
- Compare sentiment-enhanced vs baseline models

---

## Potential Enhancements

### Short-term
1. **Sentiment momentum** - Track 7-day sentiment change
2. **Sector sentiment** - Aggregate sentiment by sector
3. **Sentiment volatility** - Measure consensus stability

### Medium-term
1. **Multiple sources** - Add Reddit (retail sentiment)
2. **Source-specific sentiment** - Weight by news source quality
3. **Sentiment events** - Flag sudden sentiment shifts

### Long-term
1. **Upgrade to FinBERT** - Finance-specific sentiment model
2. **Entity recognition** - Extract key entities from news
3. **Causal analysis** - Link specific news to price moves

---

## Cost & Scalability

### Current Setup (Free Tier)
- **News API:** 100 requests/day
- **Usage:** ~40 requests/day (20 stocks × 2 queries)
- **Headroom:** 60% remaining capacity
- **Cost:** $0/month

### If Scaling Needed
- **50 stocks:** 100 requests/day (at limit, still free)
- **100 stocks:** Upgrade to $49/month (500 req/day)
- **500 stocks:** Consider batch processing or alternative sources

### Alternative Free Sources
- Yahoo Finance RSS feeds
- Reddit PRAW API
- Twitter/X free tier
- Finviz news scraping

---

## Documentation

### Files Created
- `fetch_sentiment.py` - Main sentiment fetcher
- `sentiment_data.csv` - Current sentiment scores
- `SENTIMENT_SETUP.md` - Setup instructions
- `SENTIMENT_IMPACT.md` - This file

### Files Modified
- `train_refined_models.py` - Added sentiment features
- `predict_refined.py` - Uses sentiment in predictions
- `run_daily_workflow.sh` - Fetches sentiment first
- `.github/workflows/daily_update.yml` - Daily automation
- `requirements.txt` - Added vaderSentiment, requests

---

## Next Steps

1. **Choose testing approach:**
   - A. Quick test with existing models
   - B. Full retrain with sentiment (recommended)
   - C. Complete workflow run

2. **Add GitHub Secret:**
   - Set `NEWS_API_KEY` in repo settings
   - Enable automated daily updates

3. **Monitor results:**
   - Track accuracy changes
   - Review feature importance
   - Compare recommendations

4. **Iterate based on data:**
   - If sentiment helps: expand features
   - If neutral: try FinBERT or alternative sources
   - If hurts: consider removing or adjusting weights

---

**Status:** ✅ Ready for testing
**Integration:** ✅ Complete  
**Automation:** ✅ Configured
**Next Action:** Choose testing option A, B, or C above
