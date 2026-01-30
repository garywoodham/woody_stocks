#!/usr/bin/env python3
"""
Backfill sentiment data for the 25 new stocks added to the portfolio.
This will fetch historical sentiment for weekly intervals going back to match stock data.
"""

import feedparser
import pandas as pd
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time
from urllib.parse import quote
import random

class NewStocksSentimentBackfill:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.history_file = 'data/sentiment_history.csv'
        
        # The 25 new stocks
        self.new_stocks = {
            'Energy': {
                'XOM': 'Exxon Mobil',
                'CVX': 'Chevron',
                'COP': 'ConocoPhillips',
                'SLB': 'Schlumberger',
                'OXY': 'Occidental Petroleum'
            },
            'Consumer Staples': {
                'WMT': 'Walmart',
                'PG': 'Procter & Gamble',
                'KO': 'Coca-Cola',
                'PEP': 'PepsiCo',
                'COST': 'Costco'
            },
            'Industrials': {
                'CAT': 'Caterpillar',
                'GE': 'General Electric',
                'HON': 'Honeywell',
                'UPS': 'United Parcel Service',
                'MMM': '3M Company'
            },
            'Financials': {
                'JPM': 'JPMorgan Chase',
                'BAC': 'Bank of America',
                'GS': 'Goldman Sachs',
                'MS': 'Morgan Stanley',
                'BLK': 'BlackRock'
            },
            'Semiconductors': {
                'AMD': 'Advanced Micro Devices',
                'INTC': 'Intel',
                'TSM': 'Taiwan Semiconductor',
                'AVGO': 'Broadcom',
                'QCOM': 'Qualcomm'
            }
        }
    
    def fetch_news_for_date_range(self, query, start_date, end_date, max_articles=10):
        """Fetch news from Google RSS for a date range"""
        try:
            search_query = f"{query} after:{start_date.strftime('%Y-%m-%d')} before:{end_date.strftime('%Y-%m-%d')}"
            encoded_query = quote(search_query)
            url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
            
            feed = feedparser.parse(url)
            articles = []
            
            for entry in feed.entries[:max_articles]:
                articles.append({
                    'title': entry.get('title', ''),
                    'published': entry.get('published', ''),
                    'source': entry.get('source', {}).get('title', 'Unknown')
                })
            
            time.sleep(random.uniform(1.5, 3.0))  # Rate limiting
            return articles
            
        except Exception as e:
            print(f"    ⚠️  Error fetching news: {e}")
            return []
    
    def calculate_sentiment_for_week(self, ticker, company_name, week_date):
        """Calculate sentiment for a specific week"""
        week_end = week_date + timedelta(days=6)
        
        # Fetch news for ticker and company
        articles_ticker = self.fetch_news_for_date_range(f"{ticker} stock", week_date, week_end, max_articles=8)
        articles_company = self.fetch_news_for_date_range(company_name, week_date, week_end, max_articles=7)
        
        all_articles = articles_ticker + articles_company
        
        if not all_articles:
            return {
                'ticker': ticker,
                'date': week_date.strftime('%Y-%m-%d'),
                'sentiment_compound': 0.0,
                'sentiment_score': 0.0,
                'positive_ratio': 0.0,
                'negative_ratio': 0.0,
                'neutral_ratio': 1.0,
                'news_count': 0
            }
        
        # Analyze sentiment
        sentiments = []
        for article in all_articles:
            text = article['title']
            scores = self.analyzer.polarity_scores(text)
            sentiments.append(scores)
        
        # Aggregate
        avg_compound = sum(s['compound'] for s in sentiments) / len(sentiments)
        avg_pos = sum(s['pos'] for s in sentiments) / len(sentiments)
        avg_neg = sum(s['neg'] for s in sentiments) / len(sentiments)
        avg_neu = sum(s['neu'] for s in sentiments) / len(sentiments)
        
        # Normalize to -1 to 1 score
        sentiment_score = avg_compound
        
        return {
            'ticker': ticker,
            'date': week_date.strftime('%Y-%m-%d'),
            'sentiment_compound': round(avg_compound, 4),
            'sentiment_score': round(sentiment_score, 4),
            'positive_ratio': round(avg_pos, 4),
            'negative_ratio': round(avg_neg, 4),
            'neutral_ratio': round(avg_neu, 4),
            'news_count': len(all_articles)
        }
    
    def backfill_stock(self, ticker, company_name, sector):
        """Backfill sentiment for one stock - weekly intervals for past 3 months"""
        print(f"\n{'='*80}")
        print(f"Backfilling {company_name} ({ticker}) - {sector}")
        print(f"{'='*80}")
        
        # Generate weekly dates for past 3 months
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)  # 3 months
        
        current_date = start_date
        week_dates = []
        while current_date <= end_date:
            week_dates.append(current_date)
            current_date += timedelta(days=7)
        
        print(f"Fetching sentiment for {len(week_dates)} weeks ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")
        
        results = []
        for i, week_date in enumerate(week_dates, 1):
            print(f"  [{i}/{len(week_dates)}] Week of {week_date.strftime('%Y-%m-%d')}...", end=' ')
            
            sentiment = self.calculate_sentiment_for_week(ticker, company_name, week_date)
            results.append(sentiment)
            
            print(f"✓ Score: {sentiment['sentiment_score']:+.3f}, News: {sentiment['news_count']}")
        
        return results
    
    def run_backfill(self):
        """Run the full backfill process"""
        print("="*80)
        print("SENTIMENT BACKFILL FOR NEW STOCKS")
        print("="*80)
        print(f"Stocks to backfill: 25 (across 5 new sectors)")
        print(f"Time period: Past 3 months (weekly intervals)")
        print("="*80)
        
        # Load existing sentiment data
        try:
            df_existing = pd.read_csv(self.history_file)
            print(f"\n✓ Loaded existing sentiment data: {len(df_existing)} records")
        except FileNotFoundError:
            df_existing = pd.DataFrame()
            print("\n⚠️  No existing sentiment data found, creating new file")
        
        all_results = []
        stock_counter = 0
        total_stocks = sum(len(stocks) for stocks in self.new_stocks.values())
        
        for sector, stocks in self.new_stocks.items():
            for ticker, company_name in stocks.items():
                stock_counter += 1
                print(f"\n[{stock_counter}/{total_stocks}]")
                
                results = self.backfill_stock(ticker, company_name, sector)
                all_results.extend(results)
        
        # Combine with existing data
        df_new = pd.DataFrame(all_results)
        
        if not df_existing.empty:
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=['ticker', 'date'], keep='last')
            df_combined = df_combined.sort_values(['ticker', 'date'])
        else:
            df_combined = df_new.sort_values(['ticker', 'date'])
        
        # Save
        df_combined.to_csv(self.history_file, index=False)
        
        print(f"\n{'='*80}")
        print("BACKFILL COMPLETE")
        print(f"{'='*80}")
        print(f"✓ Saved {len(df_combined)} total records to {self.history_file}")
        print(f"✓ New records added: {len(df_new)}")
        print(f"✓ Unique stocks: {df_combined['ticker'].nunique()}")
        
        # Show summary
        print(f"\n{'='*80}")
        print("SENTIMENT SUMMARY BY NEW SECTOR")
        print(f"{'='*80}")
        
        for sector, stocks in self.new_stocks.items():
            print(f"\n{sector}:")
            for ticker in stocks.keys():
                ticker_data = df_combined[df_combined['ticker'] == ticker]
                if len(ticker_data) > 0:
                    avg_sentiment = ticker_data['sentiment_score'].mean()
                    print(f"  {ticker:8s}: {len(ticker_data)} weeks, Avg sentiment: {avg_sentiment:+.3f}")

if __name__ == '__main__':
    backfiller = NewStocksSentimentBackfill()
    backfiller.run_backfill()
