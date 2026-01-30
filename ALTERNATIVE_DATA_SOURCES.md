# Alternative Data Sources for 1-Day Predictions
**Goal:** Boost accuracy from 53% → 65%+  
**Focus:** Data that moves faster than price

---

## 🎯 Tier 1: High Impact, Proven Sources

### 1. **News Sentiment - Real-Time** ⭐⭐⭐

#### **A. News API (What you have, but need real-time)**
```python
# Current: Daily aggregation
# Problem: Too slow, misses intraday moves

# Solution: Fetch every hour during market
import requests
from datetime import datetime, timedelta

API_KEY = 'your_newsapi_key'

def fetch_realtime_news(ticker, hours_back=4):
    """Fetch news from last N hours"""
    url = f'https://newsapi.org/v2/everything'
    params = {
        'q': f'{ticker} OR {company_name}',
        'from': (datetime.now() - timedelta(hours=hours_back)).isoformat(),
        'sortBy': 'publishedAt',
        'language': 'en',
        'apiKey': API_KEY
    }
    response = requests.get(url, params=params)
    return response.json()

# Key features:
- News in last 1 hour (breaking news)
- Sentiment spike (change vs yesterday)
- Headline sentiment (title only, fast)
- Source credibility (WSJ > random blog)
```

**Cost:** $449/month for Business plan (unlimited requests)  
**Free Tier:** 100 requests/day (not enough for real-time)  
**Expected Impact:** +3-5% accuracy  
**Best For:** All stocks, especially large caps

---

#### **B. Alpha Vantage News Sentiment**
```python
# Better sentiment analysis than NewsAPI
# Includes relevance scores

import requests

API_KEY = 'your_alpha_vantage_key'

def get_alpha_vantage_sentiment(ticker):
    url = f'https://www.alphavantage.co/query'
    params = {
        'function': 'NEWS_SENTIMENT',
        'tickers': ticker,
        'time_from': '20260117T0000',
        'apikey': API_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    # Features:
    sentiment = data['feed'][0]['overall_sentiment_score']
    relevance = data['feed'][0]['ticker_sentiment'][0]['relevance_score']
    
    return {
        'sentiment_score': sentiment,
        'relevance_score': relevance,
        'weighted_sentiment': sentiment * relevance
    }
```

**Cost:** Free (25 requests/day) or $49.99/month (1,200 requests/day)  
**Expected Impact:** +2-4% accuracy  
**Pros:** Better sentiment analysis, relevance scoring  
**Cons:** Slower updates than NewsAPI

---

#### **C. Finnhub News Sentiment**
```python
# Best real-time news + sentiment
# Includes market buzz score

import finnhub

finnhub_client = finnhub.Client(api_key="your_finnhub_key")

def get_finnhub_sentiment(ticker):
    # News sentiment
    news = finnhub_client.company_news(ticker, 
                                       _from="2026-01-17", 
                                       to="2026-01-17")
    
    # Social sentiment (buzz score)
    social = finnhub_client.social_sentiment(ticker)
    
    return {
        'news_count': len(news),
        'reddit_mentions': social['reddit']['mention'],
        'twitter_mentions': social['twitter']['mention'],
        'buzz_score': social['reddit']['score']
    }
```

**Cost:** Free (60 calls/min) or $59.99/month (300 calls/min)  
**Expected Impact:** +3-5% accuracy  
**Best Feature:** Social sentiment included  
**Best For:** Meme stocks (TLRY, AMC, GME)

---

### 2. **Social Media Sentiment** ⭐⭐⭐

#### **A. Reddit (r/wallstreetbets)**
```python
# FREE - Most impactful for meme stocks
import praw

reddit = praw.Reddit(
    client_id='your_client_id',
    client_secret='your_secret',
    user_agent='your_app'
)

def get_reddit_sentiment(ticker):
    subreddit = reddit.subreddit('wallstreetbets')
    
    # Get posts mentioning ticker
    mentions = 0
    positive = 0
    negative = 0
    
    for post in subreddit.hot(limit=100):
        if ticker in post.title.upper() or ticker in post.selftext.upper():
            mentions += 1
            score = post.score
            if score > 100:
                positive += 1
            elif score < 0:
                negative += 1
    
    return {
        'reddit_mentions_1h': mentions,
        'reddit_sentiment': (positive - negative) / (mentions + 1),
        'reddit_momentum': mentions / 100  # % of hot posts
    }
```

**Cost:** FREE (need Reddit account)  
**Expected Impact:** +5-10% for meme stocks, +1-2% for others  
**API Limits:** 60 requests/minute  
**Best For:** TLRY, AMC, GME, PLTR, RIVN, SOFI

**Key Subreddits:**
- r/wallstreetbets (meme stocks)
- r/stocks (general)
- r/investing (conservative)
- Stock-specific subreddits (r/PLTR, r/amcstock)

---

#### **B. Twitter/X Sentiment**
```python
# Requires Twitter API v2

import tweepy

client = tweepy.Client(bearer_token='your_bearer_token')

def get_twitter_sentiment(ticker):
    # Search recent tweets
    query = f'${ticker} -is:retweet lang:en'
    tweets = client.search_recent_tweets(
        query=query,
        max_results=100,
        tweet_fields=['created_at', 'public_metrics']
    )
    
    # Analyze
    total_likes = sum(t.public_metrics['like_count'] for t in tweets.data)
    volume = len(tweets.data)
    
    return {
        'twitter_volume': volume,
        'twitter_engagement': total_likes,
        'twitter_momentum': volume / 100  # vs average
    }
```

**Cost:**  
- Basic: $100/month (10,000 tweets/month)  
- Pro: $5,000/month (1M tweets/month)

**Expected Impact:** +2-5% accuracy  
**Best For:** High-volume stocks (AAPL, TSLA, NVDA)

**Alternative (FREE):** Scrape Twitter search (slower, legal gray area)

---

#### **C. StockTwits**
```python
# FREE social sentiment specifically for stocks

import requests

def get_stocktwits_sentiment(ticker):
    url = f'https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json'
    response = requests.get(url)
    data = response.json()
    
    # Messages have sentiment labels
    bullish = 0
    bearish = 0
    
    for msg in data['messages']:
        if msg.get('entities', {}).get('sentiment'):
            if msg['entities']['sentiment']['basic'] == 'Bullish':
                bullish += 1
            elif msg['entities']['sentiment']['basic'] == 'Bearish':
                bearish += 1
    
    return {
        'stocktwits_bullish_pct': bullish / (bullish + bearish + 1),
        'stocktwits_volume': len(data['messages']),
        'stocktwits_sentiment': (bullish - bearish) / len(data['messages'])
    }
```

**Cost:** FREE  
**Expected Impact:** +2-4% accuracy  
**API Limits:** 200 requests/hour  
**Best For:** All stocks, especially retail favorites

---

### 3. **Options Flow Data** ⭐⭐⭐

#### **A. Unusual Whales**
```python
# Best retail options flow data

# Example data structure:
{
    'ticker': 'AAPL',
    'unusual_activity': True,
    'call_volume': 150000,
    'put_volume': 50000,
    'put_call_ratio': 0.33,  # Bullish
    'premium_spent': 5000000,
    'sentiment': 'bullish'
}

# Key signals:
- Large call buying = bullish
- Large put buying = bearish  
- Put/call ratio < 0.7 = bullish
- Put/call ratio > 1.0 = bearish
- Unusual activity + volume = strong signal
```

**Cost:** $50/month (API access)  
**Expected Impact:** +5-8% accuracy  
**Data Updates:** Real-time during market hours  
**Best For:** Tech stocks, high IV names

**Features:**
- Unusual options activity alerts
- Flow tracker (big money moves)
- Put/call ratios
- Historical patterns

---

#### **B. CBOE Data (FREE)**
```python
# Official exchange data - free but delayed

import pandas as pd

def get_cboe_putcall():
    # Total market put/call ratio
    url = 'http://www.cboe.com/publish/scheduledtask/mktdata/datahouse/totalpc.csv'
    df = pd.read_csv(url, skiprows=3)
    
    current_ratio = df.iloc[-1]['CALL/PUT']
    
    return {
        'market_putcall_ratio': current_ratio,
        'putcall_extreme_fear': 1 if current_ratio > 1.2 else 0,
        'putcall_extreme_greed': 1 if current_ratio < 0.6 else 0
    }
```

**Cost:** FREE  
**Expected Impact:** +1-2% accuracy (market-wide signal)  
**Limitation:** Market-level only, not stock-specific

---

#### **C. Tradier Options API**
```python
# Free delayed options data

import requests

def get_tradier_options(ticker):
    url = f'https://sandbox.tradier.com/v1/markets/options/chains'
    headers = {'Authorization': 'Bearer YOUR_TOKEN'}
    params = {'symbol': ticker, 'expiration': '2026-01-24'}
    
    response = requests.get(url, params=params, headers=headers)
    options = response.json()
    
    # Calculate put/call volume
    call_volume = sum(o['volume'] for o in options if o['option_type'] == 'call')
    put_volume = sum(o['volume'] for o in options if o['option_type'] == 'put')
    
    return {
        'options_call_volume': call_volume,
        'options_put_volume': put_volume,
        'options_putcall_ratio': put_volume / (call_volume + 1)
    }
```

**Cost:** FREE (15min delay) or $75/month (real-time)  
**Expected Impact:** +3-5% accuracy  
**Best For:** Liquid stocks with active options

---

### 4. **Insider Trading** ⭐⭐

#### **A. SEC EDGAR (FREE)**
```python
# Official SEC filings - Form 4

import requests
from sec_api import QueryApi

def get_insider_trades(ticker, days=7):
    # Query recent Form 4 filings
    query = {
        "query": f'ticker:{ticker} AND formType:"4"',
        "from": "0",
        "size": "10",
        "sort": [{"filedAt": {"order": "desc"}}]
    }
    
    # Check for insider buying
    buys = 0
    sells = 0
    
    for filing in filings:
        if filing['transactionCode'] == 'P':  # Purchase
            buys += 1
        elif filing['transactionCode'] == 'S':  # Sale
            sells += 1
    
    return {
        'insider_buys_7d': buys,
        'insider_sells_7d': sells,
        'insider_signal': 1 if buys > sells else -1
    }
```

**Cost:** FREE (SEC-API free tier: 10 calls/day) or $49/month (100 calls/day)  
**Expected Impact:** +2-4% accuracy  
**Best For:** Small/mid caps (insiders more informed)

**Key Signals:**
- Multiple insiders buying = very bullish
- CEO buying = strongest signal
- CFO selling = often routine, ignore
- Cluster of buys = strong conviction

---

#### **B. QuiverQuant**
```python
# Cleaned insider data + congressional trading

import requests

def get_quiver_insider(ticker):
    url = f'https://api.quiverquant.com/beta/insider/{ticker}'
    headers = {'Authorization': 'Bearer YOUR_TOKEN'}
    
    response = requests.get(url, headers=headers)
    data = response.json()
    
    # Recent insider activity
    recent_buys = sum(1 for t in data if t['transaction'] == 'P-Purchase')
    recent_sells = sum(1 for t in data if t['transaction'] == 'S-Sale')
    
    return {
        'insider_buy_volume': recent_buys,
        'insider_sell_volume': recent_sells,
        'insider_bullish': 1 if recent_buys > recent_sells * 2 else 0
    }
```

**Cost:** $40/month (insider + congress trading)  
**Expected Impact:** +3-5% accuracy  
**Added Bonus:** Congressional trading data (very predictive)

---

### 5. **Short Interest & Dark Pool** ⭐⭐

#### **A. Fintel Short Interest**
```python
# Short interest changes signal reversals

import requests

def get_short_interest(ticker):
    # Requires Fintel API
    url = f'https://fintel.io/api/shortInterest/{ticker}'
    
    data = requests.get(url).json()
    
    return {
        'short_interest_pct': data['shortInterestPercent'],
        'short_interest_change': data['change'],
        'days_to_cover': data['daysToCover'],
        'squeeze_risk': 1 if data['daysToCover'] > 7 else 0
    }
```

**Cost:** $30/month  
**Expected Impact:** +2-4% for high SI stocks  
**Best For:** TLRY, AMC, GME (squeeze candidates)

---

#### **B. Dark Pool Data**
```python
# Unusual dark pool activity

# Key metrics:
- Dark pool volume as % of total
- Large block trades (>100k shares)
- Premium/discount to market price

# Signal interpretation:
- Increasing dark pool volume = accumulation (bullish)
- Large blocks at premium = strong demand
- Decreasing dark pool = retail buying (neutral)
```

**Source:** Unusual Whales ($50/month) or QuiverQuant ($40/month)  
**Expected Impact:** +2-3% accuracy  
**Best For:** Large caps with deep dark pool activity

---

## 🎯 Tier 2: Experimental/Advanced

### 6. **Google Trends**
```python
from pytrends.request import TrendReq

pytrends = TrendReq()

def get_google_trends(ticker, company_name):
    pytrends.build_payload([company_name], timeframe='now 1-d')
    trends = pytrends.interest_over_time()
    
    return {
        'search_interest': trends[company_name].iloc[-1],
        'search_momentum': trends[company_name].pct_change().iloc[-1]
    }
```

**Cost:** FREE  
**Expected Impact:** +1-2% accuracy  
**Best For:** Consumer brands (AAPL, TSLA, AMZN)

---

### 7. **Credit Card Data**
```python
# Consumer spending patterns (via aggregators)

# Example providers:
- Earnest Research (institutional, expensive)
- Second Measure (app usage + spending)
- Facteus (credit card panel data)

# Use cases:
- AAPL: iPhone sales estimates
- WMT: Foot traffic trends
- AMZN: Online shopping velocity
```

**Cost:** $1,000-10,000/month (institutional only)  
**Expected Impact:** +5-10% for retail stocks  
**Limitation:** Very expensive

---

### 8. **Satellite Imagery**
```python
# Parking lot traffic, shipping activity

# Providers:
- Orbital Insight
- RS Metrics  
- SpaceKnow

# Use cases:
- WMT: Parking lot fullness
- TSLA: Factory activity
- Target shipping container counts
```

**Cost:** $500-5,000/month  
**Expected Impact:** +3-7% for specific sectors  
**Best For:** Retail, manufacturing

---

## 💰 Recommended Budget Allocation

### **Month 1: Free tier ($0)**
1. ✅ Reddit sentiment (FREE)
2. ✅ StockTwits (FREE)
3. ✅ CBOE put/call (FREE)
4. ✅ Google Trends (FREE)
5. ✅ SEC EDGAR insider (FREE with limits)

**Expected: 53% → 58-60% accuracy**

---

### **Month 2: Basic tier ($50-150/month)**
6. ✅ Unusual Whales ($50)
7. ✅ Finnhub ($60)
8. ✅ QuiverQuant ($40)

**Total: $150/month**  
**Expected: 60% → 63-66% accuracy**

---

### **Month 3: Pro tier ($200-500/month)**
9. ✅ News API Business ($449)
10. ✅ Twitter API Basic ($100)
11. Keep Unusual Whales + Finnhub + QuiverQuant

**Total: $699/month**  
**Expected: 66% → 68-70% accuracy**

---

## 🔥 Recommended Starting Point

### **Start Here (Free):**

**This Weekend:**
```bash
1. Sign up for Reddit API (FREE)
2. Sign up for StockTwits API (FREE)
3. Sign up for Finnhub free tier (FREE, 60 calls/min)
4. Implement in quick_start_1day_features.py
```

**Expected Improvement:** +4-6% accuracy (53% → 57-59%)

**Code to Add:**
```python
# Add to quick_start_1day_features.py

def get_social_sentiment(ticker):
    """Get free social sentiment"""
    
    # Reddit
    reddit_score = get_reddit_sentiment(ticker)
    
    # StockTwits  
    stocktwits_score = get_stocktwits_sentiment(ticker)
    
    # Combine
    return {
        'social_volume': reddit_score['mentions'] + stocktwits_score['volume'],
        'social_sentiment': (reddit_score['sentiment'] + stocktwits_score['sentiment']) / 2,
        'social_bullish': 1 if (reddit_score['sentiment'] + stocktwits_score['sentiment']) > 0.2 else 0
    }
```

---

### **Week 2: Add Paid (High ROI)**

**Best $50 you can spend:**
```bash
1. Unusual Whales ($50/month)
   - Options flow = immediate edge
   - Best for tech stocks
   - Real-time alerts
```

**Expected Improvement:** +3-5% accuracy (59% → 62-64%)

---

## 📊 Expected Results by Source

| Data Source | Cost/Month | Impact | Best For | Difficulty |
|-------------|-----------|--------|----------|-----------|
| **Reddit** | $0 | +5-10% | Meme stocks | Easy |
| **StockTwits** | $0 | +2-4% | All stocks | Easy |
| **CBOE P/C** | $0 | +1-2% | Market timing | Easy |
| **Finnhub** | $0-60 | +3-5% | All stocks | Easy |
| **Unusual Whales** | $50 | +5-8% | Options stocks | Medium |
| **QuiverQuant** | $40 | +3-5% | All stocks | Medium |
| **News API** | $450 | +3-5% | Large caps | Medium |
| **Twitter** | $100 | +2-5% | Popular stocks | Medium |
| **Credit Card** | $1,000+ | +5-10% | Retail | Hard |

---

## 🎯 Your Action Plan

### **Today:**
1. Sign up for free APIs (Reddit, StockTwits, Finnhub)
2. Test data quality with quick_start_1day_features.py
3. Validate you can fetch data

### **This Weekend:**
4. Add social sentiment features to training
5. Backtest improvement (should see +4-6%)
6. Deploy if validates

### **Week 2:**
7. Subscribe to Unusual Whales ($50)
8. Add options flow features
9. Backtest (should see another +3-5%)

### **Month 2 Decision:**
10. If at 62-64% accuracy: Consider News API ($450)
11. If stuck at 58-60%: Focus on best-performing stocks only
12. If failing: Revert to 21-day focus (already at 59%)

---

## ⚠️ Important Notes

**Data Quality > Quantity:**
- Better to have 3 good sources than 10 mediocre ones
- Reddit + Options Flow might be all you need

**Focus on Meme Stocks:**
- Your best 1-day performers are meme stocks (TLRY 60%, AMC 58%)
- Social sentiment has highest impact here
- Don't bother with sentiment for boring stocks

**Watch for Overfitting:**
- Always use cross-validation
- Paper trade before going live
- Some "signals" are just noise

**ROI Matters:**
- Free sources first
- Only pay if backtest shows improvement
- $50-150/month is the sweet spot

---

## 🎓 Summary

**Best Bang for Buck:**
1. Reddit + StockTwits (FREE) → +4-6% for meme stocks
2. Unusual Whales ($50) → +5-8% for options activity
3. Pre-market data (FREE) → +2-3% for all stocks

**Realistic Target:**
- With FREE data: 53% → 58-60%
- With $50/month: 60% → 63-65%
- With $150/month: 65% → 67-69%
- Above 70% is very hard without deep learning

**Start today with FREE sources, add paid only if validated.**
