#!/usr/bin/env python3
"""
Fetch and analyze news sentiment for stocks.
Uses News API (free tier) + VADER sentiment analysis.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time
import os

class NewsSentimentAnalyzer:
    """
    Fetch news and calculate sentiment scores for stocks.
    """
    
    def __init__(self, api_key=None):
        """
        Initialize sentiment analyzer.
        
        Args:
            api_key: News API key (get free at https://newsapi.org)
                    If None, tries to get from environment variable NEWS_API_KEY
        """
        self.api_key = api_key or os.getenv('NEWS_API_KEY')
        if not self.api_key:
            raise ValueError(
                "News API key required. Get free key at https://newsapi.org\n"
                "Set as environment variable: export NEWS_API_KEY='your_key'\n"
                "Or pass directly: NewsSentimentAnalyzer(api_key='your_key')"
            )
        
        self.base_url = "https://newsapi.org/v2/everything"
        self.analyzer = SentimentIntensityAnalyzer()
        
    def fetch_news(self, query, days_back=7, language='en', max_articles=20):
        """
        Fetch news articles for a stock.
        
        Args:
            query: Search query (stock ticker or company name)
            days_back: How many days of history to fetch
            language: Article language
            max_articles: Maximum articles to retrieve
            
        Returns:
            List of article dictionaries
        """
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        params = {
            'q': query,
            'from': from_date,
            'language': language,
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
                print(f"⚠️  News API error for {query}: {data.get('message')}")
                return []
                
        except Exception as e:
            print(f"⚠️  Failed to fetch news for {query}: {str(e)}")
            return []
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of text using VADER.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if not text:
            return {'compound': 0, 'pos': 0, 'neu': 0, 'neg': 0}
        
        scores = self.analyzer.polarity_scores(text)
        return scores
    
    def calculate_stock_sentiment(self, ticker, company_name, days_back=7):
        """
        Calculate comprehensive sentiment for a stock.
        
        Args:
            ticker: Stock ticker symbol
            company_name: Full company name
            days_back: Days of news history
            
        Returns:
            Dictionary with sentiment metrics
        """
        # Fetch news (try both ticker and company name)
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
                'sentiment_score': 0.0,
                'sentiment_label': 'NEUTRAL',
                'positive_ratio': 0.0,
                'negative_ratio': 0.0,
                'avg_compound': 0.0,
                'latest_sentiment': 0.0
            }
        
        # Analyze sentiment of each article
        sentiments = []
        for article in unique_articles:
            # Combine title and description for better sentiment
            text = f"{article.get('title', '')} {article.get('description', '')}"
            sentiment = self.analyze_sentiment(text)
            sentiment['publishedAt'] = article.get('publishedAt')
            sentiments.append(sentiment)
        
        # Calculate aggregate metrics
        compounds = [s['compound'] for s in sentiments]
        avg_sentiment = sum(compounds) / len(compounds)
        
        positive_count = sum(1 for c in compounds if c > 0.05)
        negative_count = sum(1 for c in compounds if c < -0.05)
        
        # Latest article sentiment (most recent)
        latest_sentiment = compounds[0] if compounds else 0.0
        
        # Determine label
        if avg_sentiment > 0.1:
            label = 'POSITIVE'
        elif avg_sentiment < -0.1:
            label = 'NEGATIVE'
        else:
            label = 'NEUTRAL'
        
        return {
            'ticker': ticker,
            'news_count': len(unique_articles),
            'sentiment_score': round(avg_sentiment, 4),
            'sentiment_label': label,
            'positive_ratio': round(positive_count / len(unique_articles), 4),
            'negative_ratio': round(negative_count / len(unique_articles), 4),
            'avg_compound': round(avg_sentiment, 4),
            'latest_sentiment': round(latest_sentiment, 4)
        }

def main():
    print("\n" + "="*80)
    print("📰 NEWS SENTIMENT ANALYSIS")
    print("="*80 + "\n")
    
    # Check for API key
    api_key = os.getenv('NEWS_API_KEY')
    if not api_key:
        print("❌ News API key not found!")
        print("\n🔑 To get started:")
        print("   1. Get free API key: https://newsapi.org/register")
        print("   2. Set environment variable:")
        print("      export NEWS_API_KEY='your_api_key_here'")
        print("   3. Run this script again\n")
        print("Free tier: 100 requests/day (plenty for 20 stocks daily)")
        return
    
    # Load stock list
    try:
        df_stocks = pd.read_csv('data/multi_sector_stocks.csv')
        stocks = df_stocks[['Stock', 'Ticker']].drop_duplicates().values.tolist()
    except:
        try:
            df_recommendations = pd.read_csv('stock_recommendations.csv')
            stocks = df_recommendations[['Stock', 'Ticker']].values.tolist()
        except:
            # Fallback stock list
            stocks = [
                ('Apple', 'AAPL'),
                ('Microsoft', 'MSFT'),
                ('Alphabet', 'GOOGL'),
                ('Amazon', 'AMZN'),
                ('NVIDIA', 'NVDA')
            ]
            print("⚠️  Using sample stocks (add data/multi_sector_stocks.csv for full list)\n")
    
    print(f"Analyzing sentiment for {len(stocks)} stocks...")
    print(f"News window: Last 7 days")
    print("-" * 80)
    
    # Initialize analyzer
    analyzer = NewsSentimentAnalyzer(api_key)
    
    # Analyze each stock
    results = []
    for company_name, ticker in stocks:
        print(f"Processing {company_name} ({ticker})...", end=' ')
        
        sentiment = analyzer.calculate_stock_sentiment(ticker, company_name, days_back=7)
        results.append(sentiment)
        
        # Display result
        emoji = '🟢' if sentiment['sentiment_score'] > 0.1 else '🔴' if sentiment['sentiment_score'] < -0.1 else '⚪'
        print(f"{emoji} {sentiment['sentiment_label']:8s} | Score: {sentiment['sentiment_score']:>6.3f} | Articles: {sentiment['news_count']:>3d}")
        
        # Rate limiting (free tier: 100 req/day)
        time.sleep(1)  # Be nice to the API
    
    # Create DataFrame
    df_sentiment = pd.DataFrame(results)
    
    # Save results
    df_sentiment.to_csv('sentiment_data.csv', index=False)
    print("\n" + "-" * 80)
    print(f"✓ Saved sentiment data to sentiment_data.csv\n")
    
    # Summary statistics
    print("="*80)
    print("SENTIMENT SUMMARY")
    print("="*80 + "\n")
    
    print(f"Total Stocks Analyzed:    {len(df_sentiment)}")
    print(f"Avg Sentiment Score:      {df_sentiment['sentiment_score'].mean():>6.3f}")
    print(f"Avg News Coverage:        {df_sentiment['news_count'].mean():>6.1f} articles/stock")
    print(f"\nSentiment Distribution:")
    print(f"  Positive:  {(df_sentiment['sentiment_label'] == 'POSITIVE').sum():>3d} stocks")
    print(f"  Neutral:   {(df_sentiment['sentiment_label'] == 'NEUTRAL').sum():>3d} stocks")
    print(f"  Negative:  {(df_sentiment['sentiment_label'] == 'NEGATIVE').sum():>3d} stocks")
    
    # Top positive/negative
    print(f"\n📈 Most Positive Sentiment:")
    top_positive = df_sentiment.nlargest(3, 'sentiment_score')
    for _, row in top_positive.iterrows():
        print(f"   {row['ticker']:6s}  {row['sentiment_score']:>6.3f}  ({row['news_count']} articles)")
    
    print(f"\n📉 Most Negative Sentiment:")
    top_negative = df_sentiment.nsmallest(3, 'sentiment_score')
    for _, row in top_negative.iterrows():
        print(f"   {row['ticker']:6s}  {row['sentiment_score']:>6.3f}  ({row['news_count']} articles)")
    
    print("\n" + "="*80)
    print("✓ SENTIMENT ANALYSIS COMPLETE!")
    print("="*80 + "\n")
    
    print("💡 Next Steps:")
    print("   1. Review sentiment_data.csv")
    print("   2. Add sentiment features to model training")
    print("   3. Update automation to run daily")
    print()

if __name__ == '__main__':
    main()
