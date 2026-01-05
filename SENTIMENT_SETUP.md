# 📰 Sentiment Analysis Setup Guide

## Quick Start (5 minutes)

### 1️⃣ Get Free API Key
Visit: https://newsapi.org/register
- Enter your email
- Get instant API key
- **Free tier: 100 requests/day** (perfect for 20 stocks)

### 2️⃣ Install Dependencies
```bash
pip install vaderSentiment requests
```

### 3️⃣ Set API Key
```bash
# Option A: Environment variable (recommended)
export NEWS_API_KEY='your_api_key_here'

# Option B: Add to .env file
echo "NEWS_API_KEY=your_api_key_here" >> .env
```

### 4️⃣ Test Sentiment Fetching
```bash
python fetch_sentiment.py
```

---

## What You Get

### Sentiment Features (per stock):
- **sentiment_score**: -1 (negative) to +1 (positive)
- **news_count**: Number of recent articles
- **sentiment_label**: POSITIVE / NEUTRAL / NEGATIVE
- **positive_ratio**: % of positive articles
- **negative_ratio**: % of negative articles
- **latest_sentiment**: Most recent article sentiment

### Example Output:
```
Processing Apple (AAPL)... 🟢 POSITIVE  | Score:  0.245 | Articles:  12
Processing Microsoft (MSFT)... 🟢 POSITIVE  | Score:  0.189 | Articles:   8
Processing Amazon (AMZN)... ⚪ NEUTRAL   | Score:  0.045 | Articles:  15
```

---

## Integration with Models

### Add Sentiment to Training

I'll update `train_refined_models.py` to include sentiment features:

**New features added:**
1. `sentiment_score` - Primary sentiment signal
2. `news_volume` - News coverage intensity
3. `sentiment_momentum` - 7-day vs 1-day sentiment change
4. `positive_ratio` - Bullish news percentage
5. `negative_ratio` - Bearish news percentage

**Feature count:** 55 technical + 5 sentiment = **60 total features**

### Daily Workflow

Update `run_daily_workflow.sh`:
```bash
# Before predictions, fetch latest sentiment
python fetch_sentiment.py

# Then predictions will use fresh sentiment data
python predict_refined.py
```

---

## Cost Analysis

### News API Free Tier:
- **100 requests/day**
- 20 stocks × 2 queries each = 40 requests/day
- **Plenty of headroom!** ✅

### VADER Sentiment:
- **Completely free**
- No API key needed
- Runs locally

### Total Cost: $0 💰

---

## GitHub Actions Integration

Add to `.github/workflows/daily_update.yml`:

```yaml
- name: Set News API Key
  env:
    NEWS_API_KEY: ${{ secrets.NEWS_API_KEY }}
  run: echo "NEWS_API_KEY set"

- name: Fetch Sentiment Data
  run: python fetch_sentiment.py
```

**Required:** Add `NEWS_API_KEY` to GitHub Secrets:
1. Go to repo Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Name: `NEWS_API_KEY`
4. Value: Your News API key

---

## Monitoring

### Daily Checks:
- ✅ Are all 20 stocks getting sentiment data?
- ✅ Is news coverage reasonable? (avg 5-15 articles/stock)
- ✅ API rate limit OK? (should use ~40/100 requests)

### Weekly Reviews:
- Sentiment vs price movement correlation
- Coverage gaps (stocks with 0 articles)
- Outlier sentiment (unusually high/low)

### Files to Monitor:
- `sentiment_data.csv` - Latest sentiment scores
- `fetch_sentiment.py` logs - API status

---

## Troubleshooting

### "News API key not found"
```bash
# Check if set
echo $NEWS_API_KEY

# Set temporarily
export NEWS_API_KEY='your_key'

# Set permanently (add to ~/.bashrc)
echo 'export NEWS_API_KEY="your_key"' >> ~/.bashrc
source ~/.bashrc
```

### "Too many requests" (429 error)
- Free tier = 100/day
- Reduce stocks or increase sleep time
- Consider upgrading ($49/mo for 500 req/day)

### "No articles found"
- Check ticker/company name spelling
- UK stocks may have less English news
- Increase `days_back` parameter

### Sentiment seems off
- VADER is trained on social media (shorter text)
- Financial news is more formal
- Consider upgrading to FinBERT later (finance-specific)

---

## Next Steps After Setup

1. ✅ **Test**: Run `python fetch_sentiment.py`
2. 🔄 **Integrate**: I'll add sentiment to model training
3. 📈 **Compare**: Check if sentiment improves predictions
4. 🤖 **Automate**: Add to GitHub Actions workflow
5. 📊 **Monitor**: Track sentiment vs actual returns

---

## Advanced: Upgrade to FinBERT (Optional)

If VADER sentiment doesn't improve predictions, upgrade to **FinBERT** (finance-specific):

```bash
pip install transformers torch
```

FinBERT understands:
- "beat earnings" → POSITIVE
- "missed expectations" → NEGATIVE
- "volatility" → context-dependent

But requires more compute (slower, ~10x).

---

## Support

### Resources:
- **News API Docs**: https://newsapi.org/docs
- **VADER GitHub**: https://github.com/cjhutto/vaderSentiment
- **VADER Paper**: http://eegilbert.org/papers/icwsm14.vader.hutto.pdf

### Alternative Free Sources:
- **Yahoo Finance News** (via yfinance)
- **Reddit** (via PRAW - free)
- **Twitter/X** (free API tier limited)

Let me know if you want to add these too!
