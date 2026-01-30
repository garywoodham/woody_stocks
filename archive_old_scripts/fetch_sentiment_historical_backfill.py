#!/usr/bin/env python3
"""
Backfill historical sentiment data using RSS feeds.
Fetches sentiment for specific date ranges to build historical data.
"""

import feedparser
import pandas as pd
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time
import os
from urllib.parse import quote

class HistoricalSentimentBackfill:
    """
    Backfill sentiment data for historical dates.
    """
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.history_file = 'data/sentiment_history.csv'
        
    def fetch_news_for_date_range(self, query, start_date, end_date, max_articles=10):
        """
        Fetch news for a specific date range using Google News RSS.
        Note: RSS feeds are limited - we'll get whatever's available.
        """
        articles = []
        
        try:
            encoded_query = quote(f"{query} after:{start_date.strftime('%Y-%m-%d')} before:{end_date.strftime('%Y-%m-%d')}")
            google_rss = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
            
            feed = feedparser.parse(google_rss)
            
            for entry in feed.entries[:max_articles]:
                articles.append({
                    'title': entry.get('title', ''),
                    'description': entry.get('summary', ''),
                    'published': entry.get('published', ''),
                    'link': entry.get('link', '')
                })
            
            time.sleep(0.5)  # Be respectful
            
        except Exception as e:
            print(f"⚠️  RSS fetch error for {query}: {str(e)}")
        
        return articles
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using VADER"""
        if not text:
            return {'compound': 0, 'pos': 0, 'neu': 0, 'neg': 0}
        return self.analyzer.polarity_scores(text)
    
    def calculate_stock_sentiment_for_date(self, ticker, company_name, target_date):
        """Calculate sentiment for a stock for a specific date"""
        
        # Search for news around the target date (±2 days window)
        start_date = target_date - timedelta(days=2)
        end_date = target_date + timedelta(days=2)
        
        articles = self.fetch_news_for_date_range(f"{company_name} stock", start_date, end_date)
        
        if len(articles) < 3:
            articles.extend(self.fetch_news_for_date_range(f"{ticker} stock", start_date, end_date))
        
        if not articles:
            return {
                'ticker': ticker,
                'news_count': 0,
                'sentiment_compound': 0.0,
                'sentiment_positive': 0.0,
                'sentiment_negative': 0.0,
                'sentiment_neutral': 1.0,
                'date': target_date.strftime('%Y-%m-%d')
            }
        
        # Analyze sentiment
        sentiments = []
        for article in articles:
            text = f"{article['title']} {article['description']}"
            sentiment = self.analyze_sentiment(text)
            sentiments.append(sentiment)
        
        # Calculate averages
        avg_sentiment = {
            'ticker': ticker,
            'news_count': len(articles),
            'sentiment_compound': sum(s['compound'] for s in sentiments) / len(sentiments),
            'sentiment_positive': sum(s['pos'] for s in sentiments) / len(sentiments),
            'sentiment_negative': sum(s['neg'] for s in sentiments) / len(sentiments),
            'sentiment_neutral': sum(s['neu'] for s in sentiments) / len(sentiments),
            'date': target_date.strftime('%Y-%m-%d')
        }
        
        return avg_sentiment
    
    def backfill_month(self, year, month, stock_list, sample_days=4):
        """
        Backfill sentiment for a specific month.
        Only samples a few days per month to avoid overwhelming the system.
        
        Args:
            year: Year (e.g., 2025)
            month: Month (1-12)
            stock_list: List of (ticker, name) tuples
            sample_days: Number of days to sample per month (default: 4, weekly samples)
        """
        from calendar import monthrange
        
        # Get the days in the month
        _, days_in_month = monthrange(year, month)
        
        # Sample evenly throughout the month (e.g., week 1, 2, 3, 4)
        sample_interval = days_in_month // sample_days
        sample_dates = []
        for i in range(sample_days):
            day = min((i * sample_interval) + 7, days_in_month)  # Start from day 7
            sample_dates.append(datetime(year, month, day))
        
        print(f"\nBackfilling {year}-{month:02d} with {sample_days} sample dates:")
        for d in sample_dates:
            print(f"  • {d.strftime('%Y-%m-%d')}")
        
        # Load existing history
        if os.path.exists(self.history_file):
            df_history = pd.read_csv(self.history_file)
            df_history['date'] = pd.to_datetime(df_history['date'], format='mixed').dt.strftime('%Y-%m-%d')
        else:
            df_history = pd.DataFrame()
            os.makedirs('data', exist_ok=True)
        
        results = []
        total_ops = len(stock_list) * len(sample_dates)
        current_op = 0
        
        for target_date in sample_dates:
            date_str = target_date.strftime('%Y-%m-%d')
            print(f"\n📅 Fetching sentiment for {date_str}")
            print("-" * 80)
            
            for ticker, name in stock_list:
                current_op += 1
                
                # Check if data already exists
                if not df_history.empty:
                    existing = df_history[(df_history['ticker'] == ticker) & (df_history['date'] == date_str)]
                    if not existing.empty:
                        print(f"[{current_op}/{total_ops}] {ticker:6s} - already exists, skipping")
                        continue
                
                print(f"[{current_op}/{total_ops}] Processing {ticker:6s} ({name:25s})...", end=' ')
                
                sentiment = self.calculate_stock_sentiment_for_date(ticker, name, target_date)
                results.append(sentiment)
                
                # Display sentiment
                comp = sentiment['sentiment_compound']
                icon = '🟢' if comp > 0.1 else '🔴' if comp < -0.1 else '⚪'
                print(f"{icon} {comp:+.3f} | {sentiment['news_count']:3d} articles")
                
                # Be respectful with delays (1 second between stocks)
                time.sleep(1)
        
        # Append to history
        if results:
            df_new = pd.DataFrame(results)
            if not df_history.empty:
                df_combined = pd.concat([df_history, df_new], ignore_index=True)
            else:
                df_combined = df_new
            
            # Remove duplicates
            df_combined = df_combined.drop_duplicates(subset=['ticker', 'date'], keep='last')
            
            # Save
            df_combined.to_csv(self.history_file, index=False)
            print(f"\n✓ Added {len(results)} sentiment records for {year}-{month:02d}")
        else:
            print(f"\n⚠️  No new records added for {year}-{month:02d}")
        
        return results


def main():
    # Load stock tickers
    df_stocks = pd.read_csv('data/multi_sector_stocks.csv')
    stock_list = df_stocks[['Ticker', 'Stock']].drop_duplicates().values.tolist()
    
    print("=" * 80)
    print("📊 HISTORICAL SENTIMENT BACKFILL")
    print("=" * 80)
    print(f"\nStocks to process: {len(stock_list)}")
    print("Source: Google News RSS (free, no API key required)")
    print("\n⏱️  Estimated time: ~2-3 minutes per sample day")
    print("=" * 80)
    
    # Backfill December 2025 as a trial (4 sample days)
    backfiller = HistoricalSentimentBackfill()
    backfiller.backfill_month(2025, 12, stock_list, sample_days=4)
    
    print("\n" + "=" * 80)
    print("✓ Historical backfill complete!")
    print("=" * 80)
    print("\nTo backfill more months, edit the script and add more backfill_month() calls")
    print("Example: backfiller.backfill_month(2025, 11, stock_list, sample_days=4)")


if __name__ == '__main__':
    main()
