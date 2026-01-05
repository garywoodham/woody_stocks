#!/usr/bin/env python3
"""
Fetch and track historical sentiment data for stocks.
Appends daily sentiment scores to build time-series data.
Free tier friendly: No backfilling, builds history organically.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time
import os

class HistoricalSentimentTracker:
    """
    Track sentiment over time by appending daily scores.
    """
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('NEWS_API_KEY')
        if not self.api_key:
            raise ValueError("News API key required. Set NEWS_API_KEY environment variable")
        
        self.base_url = "https://newsapi.org/v2/everything"
        self.analyzer = SentimentIntensityAnalyzer()
        self.history_file = 'data/sentiment_history.csv'
        
    def fetch_news(self, query, days_back=7, max_articles=20):
        """Fetch recent news articles"""
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        params = {
            'q': query,
            'from': from_date,
            'language': 'en',
            'sortBy': 'relevancy',
            'pageSize': min(max_articles, 100),
            'apiKey': self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == 'ok':
                return data.get('articles', [])
            else:
                print(f"⚠️  API error for {query}: {data.get('message')}")
                return []
                
        except Exception as e:
            print(f"⚠️  Failed to fetch for {query}: {str(e)}")
            return []
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using VADER"""
        if not text:
            return {'compound': 0, 'pos': 0, 'neu': 0, 'neg': 0}
        return self.analyzer.polarity_scores(text)
    
    def calculate_stock_sentiment(self, ticker, company_name, days_back=7):
        """Calculate current sentiment for a stock"""
        # Fetch news
        articles_ticker = self.fetch_news(ticker, days_back=days_back)
        articles_company = self.fetch_news(company_name, days_back=days_back)
        
        # Combine and deduplicate
        all_articles = articles_ticker + articles_company
        seen_titles = set()
        unique_articles = []
        
        for article in all_articles:
            title = article.get('title', '')
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_articles.append(article)
        
        if not unique_articles:
            return {
                'ticker': ticker,
                'news_count': 0,
                'sentiment_compound': 0.0,
                'sentiment_positive': 0.0,
                'sentiment_negative': 0.0,
                'sentiment_neutral': 1.0
            }
        
        # Analyze each article
        sentiments = []
        for article in unique_articles:
            text = f"{article.get('title', '')} {article.get('description', '')}"
            sentiment = self.analyze_sentiment(text)
            sentiments.append(sentiment)
        
        # Aggregate
        avg_compound = sum(s['compound'] for s in sentiments) / len(sentiments)
        avg_positive = sum(s['pos'] for s in sentiments) / len(sentiments)
        avg_negative = sum(s['neg'] for s in sentiments) / len(sentiments)
        avg_neutral = sum(s['neu'] for s in sentiments) / len(sentiments)
        
        return {
            'ticker': ticker,
            'news_count': len(unique_articles),
            'sentiment_compound': round(avg_compound, 4),
            'sentiment_positive': round(avg_positive, 4),
            'sentiment_negative': round(avg_negative, 4),
            'sentiment_neutral': round(avg_neutral, 4)
        }
    
    def fetch_and_append(self, stocks_list):
        """
        Fetch sentiment for all stocks and append to history.
        
        Args:
            stocks_list: List of (company_name, ticker) tuples
        """
        today = datetime.now().strftime('%Y-%m-%d')
        
        print(f"\n{'='*80}")
        print(f"📰 DAILY SENTIMENT TRACKING - {today}")
        print(f"{'='*80}\n")
        
        # Check if we already ran today
        if os.path.exists(self.history_file):
            df_history = pd.read_csv(self.history_file)
            if today in df_history['date'].values:
                print(f"✓ Sentiment already collected for {today}")
                print(f"  (Found {len(df_history[df_history['date'] == today])} stocks in history)")
                
                response = input("\nRun again anyway? (y/n): ").lower()
                if response != 'y':
                    print("Skipping. Use yesterday's data.")
                    return
                else:
                    # Remove today's data to re-fetch
                    df_history = df_history[df_history['date'] != today]
                    df_history.to_csv(self.history_file, index=False)
        
        print(f"Fetching sentiment for {len(stocks_list)} stocks...")
        print(f"News window: Last 7 days")
        print(f"{'-'*80}\n")
        
        results = []
        for company_name, ticker in stocks_list:
            print(f"Processing {ticker:6s} ({company_name[:30]:30s})...", end=' ')
            
            sentiment = self.calculate_stock_sentiment(ticker, company_name, days_back=7)
            sentiment['date'] = today
            results.append(sentiment)
            
            # Display
            score = sentiment['sentiment_compound']
            emoji = '🟢' if score > 0.1 else '🔴' if score < -0.1 else '⚪'
            print(f"{emoji} {score:>6.3f} | {sentiment['news_count']:>3d} articles")
            
            time.sleep(1)  # API rate limiting
        
        # Create DataFrame
        df_today = pd.DataFrame(results)
        
        # Append to history
        if os.path.exists(self.history_file):
            df_history = pd.read_csv(self.history_file)
            df_combined = pd.concat([df_history, df_today], ignore_index=True)
        else:
            df_combined = df_today
            print(f"\n✨ Creating new sentiment history file")
        
        # Save
        df_combined.to_csv(self.history_file, index=False)
        
        print(f"\n{'-'*80}")
        print(f"✓ Saved to {self.history_file}")
        print(f"  Total records: {len(df_combined)}")
        print(f"  Unique dates: {df_combined['date'].nunique()}")
        print(f"  Date range: {df_combined['date'].min()} to {df_combined['date'].max()}")
        
        # Summary
        print(f"\n{'='*80}")
        print("TODAY'S SENTIMENT SUMMARY")
        print(f"{'='*80}\n")
        
        avg_sentiment = df_today['sentiment_compound'].mean()
        print(f"Average Sentiment: {avg_sentiment:>6.3f}")
        print(f"Average Coverage:  {df_today['news_count'].mean():>6.1f} articles/stock")
        
        positive = (df_today['sentiment_compound'] > 0.1).sum()
        neutral = ((df_today['sentiment_compound'] >= -0.1) & (df_today['sentiment_compound'] <= 0.1)).sum()
        negative = (df_today['sentiment_compound'] < -0.1).sum()
        
        print(f"\nDistribution:")
        print(f"  Positive: {positive:>3d} stocks")
        print(f"  Neutral:  {neutral:>3d} stocks")
        print(f"  Negative: {negative:>3d} stocks")
        
        print(f"\n📈 Most Positive:")
        top_pos = df_today.nlargest(3, 'sentiment_compound')
        for _, row in top_pos.iterrows():
            print(f"   {row['ticker']:6s}  {row['sentiment_compound']:>6.3f}")
        
        print(f"\n📉 Most Negative:")
        top_neg = df_today.nsmallest(3, 'sentiment_compound')
        for _, row in top_neg.iterrows():
            print(f"   {row['ticker']:6s}  {row['sentiment_compound']:>6.3f}")
        
        print(f"\n{'='*80}\n")

def main():
    # Check for API key
    api_key = os.getenv('NEWS_API_KEY')
    if not api_key:
        print("\n❌ News API key not found!")
        print("Set environment variable: export NEWS_API_KEY='your_key'")
        return
    
    # Load stock list
    try:
        df_stocks = pd.read_csv('data/multi_sector_stocks.csv')
        stocks = df_stocks[['Stock', 'Ticker']].drop_duplicates().values.tolist()
    except:
        print("❌ Could not load stock list from data/multi_sector_stocks.csv")
        return
    
    # Track sentiment
    tracker = HistoricalSentimentTracker(api_key)
    tracker.fetch_and_append(stocks)
    
    print("💡 Next Steps:")
    print("   • Run this daily to build history (automate with cron/GitHub Actions)")
    print("   • Sentiment features will automatically improve as history grows")
    print("   • Retrain models weekly to incorporate new sentiment data")
    print()

if __name__ == '__main__':
    main()
