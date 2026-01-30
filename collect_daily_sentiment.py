#!/usr/bin/env python3
"""
Collect fresh sentiment data for today using NewsAPI.
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time
import os

# NewsAPI key
API_KEY = os.getenv('NEWS_API_KEY', '2937fcb16c7f40d493cca9777bf825bb')
BASE_URL = "https://newsapi.org/v2/everything"

analyzer = SentimentIntensityAnalyzer()

# Load stock list
df_stocks = pd.read_csv('data/multi_sector_stocks.csv', index_col=0, parse_dates=True)
tickers_df = df_stocks[['Stock', 'Ticker']].drop_duplicates()
tickers = tickers_df['Ticker'].unique()[:35]  # Limit to 35 to manage API calls

print("="*80)
print(f"COLLECTING SENTIMENT DATA - {datetime.now().strftime('%Y-%m-%d')}")
print("="*80)
print(f"\nCollecting sentiment for {len(tickers)} stocks using NewsAPI...")

sentiment_results = []

# Get date range (last 7 days)
to_date = datetime.now()
from_date = to_date - timedelta(days=7)

for i, ticker in enumerate(tickers, 1):
    stock_name = tickers_df[tickers_df['Ticker'] == ticker]['Stock'].iloc[0]
    print(f"\n[{i}/{len(tickers)}] {stock_name} ({ticker})...", end=" ")
    
    # Search with stock name and ticker
    query = f'"{stock_name}" OR {ticker}'
    
    params = {
        'q': query,
        'from': from_date.strftime('%Y-%m-%d'),
        'to': to_date.strftime('%Y-%m-%d'),
        'language': 'en',
        'sortBy': 'publishedAt',
        'pageSize': 10,
        'apiKey': API_KEY
    }
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        articles = data.get('articles', [])
        
        if not articles:
            print("No news")
            sentiment_results.append({
                'ticker': ticker,
                'news_count': 0,
                'sentiment_score': 0.0,
                'sentiment_label': 'NEUTRAL',
                'positive_ratio': 0.0,
                'negative_ratio': 0.0,
                'avg_compound': 0.0,
                'latest_sentiment': 0.0
            })
            time.sleep(0.5)  # Rate limiting
            continue
        
        # Analyze sentiment
        sentiments = []
        for article in articles[:10]:  # Limit to 10
            text = f"{article.get('title', '')} {article.get('description', '')}"
            if text.strip():
                scores = analyzer.polarity_scores(text)
                sentiments.append(scores['compound'])
        
        if not sentiments:
            sentiments = [0.0]
        
        avg_compound = sum(sentiments) / len(sentiments)
        positive_ratio = sum(1 for s in sentiments if s > 0.05) / len(sentiments)
        negative_ratio = sum(1 for s in sentiments if s < -0.05) / len(sentiments)
        
        # Determine label
        if avg_compound > 0.05:
            label = 'POSITIVE'
        elif avg_compound < -0.05:
            label = 'NEGATIVE'
        else:
            label = 'NEUTRAL'
        
        sentiment_results.append({
            'ticker': ticker,
            'news_count': len(sentiments),
            'sentiment_score': round(avg_compound, 4),
            'sentiment_label': label,
            'positive_ratio': round(positive_ratio, 4),
            'negative_ratio': round(negative_ratio, 4),
            'avg_compound': round(avg_compound, 4),
            'latest_sentiment': round(sentiments[0] if sentiments else 0.0, 4)
        })
        
        print(f"✓ {len(sentiments)} articles, {label} ({avg_compound:.3f})")
        time.sleep(0.5)  # Rate limiting
        
    except Exception as e:
        print(f"✗ Error: {str(e)[:50]}")
        sentiment_results.append({
            'ticker': ticker,
            'news_count': 0,
            'sentiment_score': 0.0,
            'sentiment_label': 'NEUTRAL',
            'positive_ratio': 0.0,
            'negative_ratio': 0.0,
            'avg_compound': 0.0,
            'latest_sentiment': 0.0
        })

# Save results
df_sentiment = pd.DataFrame(sentiment_results)
df_sentiment.to_csv('sentiment_data.csv', index=False)

print("\n" + "="*80)
print("✅ SENTIMENT DATA SAVED")
print("="*80)
print(f"\nTotal stocks: {len(df_sentiment)}")
print(f"With news: {len(df_sentiment[df_sentiment['news_count'] > 0])}")
print(f"\nSentiment distribution:")
print(df_sentiment['sentiment_label'].value_counts())
print(f"\nAverage sentiment score: {df_sentiment[df_sentiment['news_count'] > 0]['sentiment_score'].mean():.3f}")
print(f"\nSaved to: sentiment_data.csv")
print("="*80)
