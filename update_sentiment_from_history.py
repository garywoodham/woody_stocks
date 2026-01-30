#!/usr/bin/env python3
"""
Update sentiment_data.csv from historical sentiment data.
Uses the most recent 7 days of sentiment for each ticker.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("="*80)
print("UPDATING SENTIMENT DATA FROM HISTORICAL RECORDS")
print("="*80)

# Load historical sentiment data
print("\n📊 Loading historical sentiment data...")
df_hist = pd.read_csv('data/sentiment_history_complete.csv')
df_hist['date'] = pd.to_datetime(df_hist['date'])

print(f"✓ Loaded {len(df_hist):,} records")
print(f"  Date range: {df_hist['date'].min()} to {df_hist['date'].max()}")
print(f"  Tickers: {df_hist['ticker'].nunique()}")

# Get most recent date
latest_date = df_hist['date'].max()
lookback_date = latest_date - timedelta(days=7)

print(f"\n📅 Analyzing last 7 days of data:")
print(f"  From: {lookback_date.date()}")
print(f"  To: {latest_date.date()}")

# Filter to recent data
df_recent = df_hist[df_hist['date'] >= lookback_date].copy()
print(f"\n✓ Filtered to {len(df_recent):,} recent records")

# Aggregate by ticker
print("\n🔄 Aggregating sentiment by ticker...")

sentiment_summary = []

for ticker in df_recent['ticker'].unique():
    ticker_data = df_recent[df_recent['ticker'] == ticker]
    
    # Get records with actual news (news_count > 0)
    ticker_with_news = ticker_data[ticker_data['news_count'] > 0]
    
    if len(ticker_with_news) > 0:
        # Calculate weighted average (more recent = more weight)
        ticker_with_news = ticker_with_news.sort_values('date')
        weights = np.arange(1, len(ticker_with_news) + 1)  # Linear weights
        
        avg_sentiment = np.average(ticker_with_news['sentiment_score'], weights=weights)
        total_news = ticker_with_news['news_count'].sum()
        
        # Calculate sentiment label
        if avg_sentiment > 0.05:
            label = 'POSITIVE'
        elif avg_sentiment < -0.05:
            label = 'NEGATIVE'
        else:
            label = 'NEUTRAL'
        
        # Get latest sentiment
        latest = ticker_with_news.iloc[-1]
        
        sentiment_summary.append({
            'ticker': ticker,
            'news_count': int(total_news),
            'sentiment_score': round(avg_sentiment, 4),
            'sentiment_label': label,
            'positive_ratio': round(ticker_with_news[ticker_with_news['sentiment_score'] > 0.05].shape[0] / len(ticker_with_news), 4),
            'negative_ratio': round(ticker_with_news[ticker_with_news['sentiment_score'] < -0.05].shape[0] / len(ticker_with_news), 4),
            'avg_compound': round(avg_sentiment, 4),
            'latest_sentiment': round(latest['sentiment_score'], 4)
        })
    else:
        # No news in recent period
        sentiment_summary.append({
            'ticker': ticker,
            'news_count': 0,
            'sentiment_score': 0.0,
            'sentiment_label': 'NEUTRAL',
            'positive_ratio': 0.0,
            'negative_ratio': 0.0,
            'avg_compound': 0.0,
            'latest_sentiment': 0.0
        })

# Create DataFrame
df_sentiment = pd.DataFrame(sentiment_summary)
df_sentiment = df_sentiment.sort_values('ticker')

# Save
df_sentiment.to_csv('sentiment_data.csv', index=False)

print(f"✓ Aggregated {len(df_sentiment)} tickers")

print("\n📊 Sentiment Summary:")
print(f"  Total tickers: {len(df_sentiment)}")
print(f"  With news: {len(df_sentiment[df_sentiment['news_count'] > 0])}")
print(f"  Without news: {len(df_sentiment[df_sentiment['news_count'] == 0])}")

print(f"\n📈 Sentiment Distribution:")
print(df_sentiment['sentiment_label'].value_counts())

print(f"\n💰 Top 10 Most Positive:")
top_positive = df_sentiment[df_sentiment['news_count'] > 0].nlargest(10, 'sentiment_score')
for _, row in top_positive.iterrows():
    print(f"  {row['ticker']:8} {row['sentiment_score']:6.3f}  ({row['news_count']} articles)")

print(f"\n⚠️  Top 10 Most Negative:")
top_negative = df_sentiment[df_sentiment['news_count'] > 0].nsmallest(10, 'sentiment_score')
for _, row in top_negative.iterrows():
    print(f"  {row['ticker']:8} {row['sentiment_score']:6.3f}  ({row['news_count']} articles)")

print("\n" + "="*80)
print("✅ SENTIMENT DATA UPDATED")
print("="*80)
print(f"\nSaved to: sentiment_data.csv")
print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
