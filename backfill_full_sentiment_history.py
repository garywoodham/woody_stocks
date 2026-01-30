#!/usr/bin/env python3
"""
Backfill complete historical sentiment data for all stocks.
Creates time-series sentiment data matching the full stock history period.
Uses RSS feeds with daily granularity.
"""

import feedparser
import pandas as pd
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time
import os
from urllib.parse import quote
import json

class FullHistoricalSentimentBackfill:
    """
    Comprehensive sentiment backfill for entire stock history.
    """
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.history_file = 'data/sentiment_history_complete.csv'
        self.progress_file = 'data/sentiment_backfill_progress.json'
        
        # Load stocks from download_stock_data.py structure
        self.stocks = {
            'Defence': {
                'BA.L': 'BAE Systems',
                'LMT': 'Lockheed Martin',
                'NOC': 'Northrop Grumman',
                'RTX': 'Raytheon Technologies',
                'RR.L': 'Rolls-Royce'
            },
            'Banking': {
                'BARC.L': 'Barclays',
                'HSBA.L': 'HSBC',
                'LLOY.L': 'Lloyds Banking',
                'NWG.L': 'NatWest Group',
                'STAN.L': 'Standard Chartered'
            },
            'Pharma': {
                'AZN.L': 'AstraZeneca',
                'GSK.L': 'GSK',
                'PFE': 'Pfizer',
                'JNJ': 'Johnson & Johnson',
                'MRNA': 'Moderna'
            },
            'Technology': {
                'AAPL': 'Apple',
                'MSFT': 'Microsoft',
                'NVDA': 'NVIDIA',
                'GOOGL': 'Alphabet',
                'AMZN': 'Amazon'
            },
            'Meme/Speculative': {
                'GME': 'GameStop',
                'AMC': 'AMC Entertainment',
                'BB': 'BlackBerry',
                'PLTR': 'Palantir Technologies',
                'SOFI': 'SoFi Technologies',
                'RIVN': 'Rivian Automotive',
                'NIO': 'NIO Inc',
                'LCID': 'Lucid Group',
                'SPCE': 'Virgin Galactic',
                'PLUG': 'Plug Power',
                'HOOD': 'Robinhood Markets',
                'COIN': 'Coinbase Global',
                'RIOT': 'Riot Platforms',
                'MARA': 'Marathon Digital',
                'TLRY': 'Tilray Brands'
            },
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
        
    def load_progress(self):
        """Load progress from checkpoint"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {'completed_dates': [], 'last_ticker': None, 'last_date': None}
    
    def save_progress(self, progress):
        """Save progress checkpoint"""
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def fetch_news_for_date_range(self, query, start_date, end_date, max_articles=20):
        """
        Fetch news for a specific date range using Google News RSS.
        With timeout handling.
        """
        articles = []
        
        try:
            # Try multiple query variations for better coverage
            queries = [
                f"{query} stock",
                f"{query}",
                f"{query} market"
            ]
            
            for q in queries[:1]:  # OPTIMIZED: Use only 1 query for speed
                try:
                    encoded_query = quote(f"{q} stock after:{start_date.strftime('%Y-%m-%d')} before:{end_date.strftime('%Y-%m-%d')}")
                    google_rss = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
                    
                    # Set timeout on feedparser
                    import socket
                    socket.setdefaulttimeout(5)  # 5 second timeout
                    
                    feed = feedparser.parse(google_rss)
                    
                    for entry in feed.entries[:max_articles]:
                        articles.append({
                            'title': entry.get('title', ''),
                            'description': entry.get('summary', ''),
                            'published': entry.get('published', ''),
                            'link': entry.get('link', '')
                        })
                    
                    if len(articles) >= 10:  # Found enough
                        break
                except Exception as query_error:
                    # Skip this query variation if it fails
                    continue
                    
                time.sleep(0.05)  # OPTIMIZED: Reduced sleep for 2x speed
            
        except Exception as e:
            # Silent fail - just return empty articles
            pass
        
        return articles[:max_articles]
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using VADER"""
        if not text:
            return {'compound': 0, 'pos': 0, 'neu': 0, 'neg': 0}
        return self.analyzer.polarity_scores(text)
    
    def calculate_stock_sentiment_for_week(self, ticker, company_name, start_date, end_date):
        """
        Calculate sentiment for a stock for a week.
        Returns daily sentiment for each trading day in the week.
        """
        
        # Fetch news for the entire week
        articles_ticker = self.fetch_news_for_date_range(ticker, start_date, end_date, max_articles=15)
        articles_company = self.fetch_news_for_date_range(company_name, start_date, end_date, max_articles=15)
        
        # Combine and deduplicate
        all_articles = articles_ticker + articles_company
        seen_titles = set()
        unique_articles = []
        
        for article in all_articles:
            title = article.get('title', '')
            if title and title not in seen_titles and len(title) > 10:
                seen_titles.add(title)
                unique_articles.append(article)
        
        # If we have no articles, return neutral sentiment for each day
        if not unique_articles:
            daily_sentiments = []
            current_date = start_date
            while current_date <= end_date:
                # Only include trading days (Mon-Fri)
                if current_date.weekday() < 5:
                    daily_sentiments.append({
                        'ticker': ticker,
                        'date': current_date.strftime('%Y-%m-%d'),
                        'news_count': 0,
                        'sentiment_compound': 0.0,
                        'sentiment_positive': 0.0,
                        'sentiment_negative': 0.0,
                        'sentiment_neutral': 1.0,
                        'sentiment_score': 0.0
                    })
                current_date += timedelta(days=1)
            return daily_sentiments
        
        # Analyze sentiment for all articles
        sentiments = []
        for article in unique_articles:
            text = f"{article.get('title', '')} {article.get('description', '')}"
            sentiment = self.analyze_sentiment(text)
            sentiments.append(sentiment)
        
        # Calculate weekly average sentiment
        avg_compound = sum(s['compound'] for s in sentiments) / len(sentiments)
        avg_positive = sum(s['pos'] for s in sentiments) / len(sentiments)
        avg_negative = sum(s['neg'] for s in sentiments) / len(sentiments)
        avg_neutral = sum(s['neu'] for s in sentiments) / len(sentiments)
        
        # Apply weekly average to each trading day in the week
        daily_sentiments = []
        current_date = start_date
        while current_date <= end_date:
            # Only include trading days (Mon-Fri)
            if current_date.weekday() < 5:
                daily_sentiments.append({
                    'ticker': ticker,
                    'date': current_date.strftime('%Y-%m-%d'),
                    'news_count': len(unique_articles),
                    'sentiment_compound': avg_compound,
                    'sentiment_positive': avg_positive,
                    'sentiment_negative': avg_negative,
                    'sentiment_neutral': avg_neutral,
                    'sentiment_score': avg_compound  # Primary score
                })
            current_date += timedelta(days=1)
        
        return daily_sentiments
    
    def backfill_all_stocks(self, start_date_str='2016-01-01', end_date_str=None):
        """
        Intelligently backfill sentiment for all stocks.
        - Detects missing date ranges for each stock
        - Only fills gaps, not complete datasets
        - Can be run regularly to catch up on new data
        """
        
        if end_date_str is None:
            end_date = datetime.now()
        else:
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        
        print("="*80)
        print("INTELLIGENT SENTIMENT BACKFILL - ALL 60 STOCKS")
        print("="*80)
        print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"Total stocks: {sum(len(stocks) for stocks in self.stocks.values())}")
        print(f"Strategy: Detect gaps and fill only missing data")
        print("="*80)
        print()
        
        # Load existing data if any
        existing_data_by_ticker = {}
        if os.path.exists(self.history_file):
            print(f"✓ Found existing data: {self.history_file}")
            existing_df = pd.read_csv(self.history_file)
            print(f"  Total records: {len(existing_df)}")
            print(f"  Stocks with data: {existing_df['ticker'].nunique()}")
            
            # Group by ticker and get date ranges
            for ticker in existing_df['ticker'].unique():
                ticker_df = existing_df[existing_df['ticker'] == ticker]
                existing_dates = set(pd.to_datetime(ticker_df['date']).dt.date)
                existing_data_by_ticker[ticker] = {
                    'dates': existing_dates,
                    'count': len(ticker_df),
                    'min_date': ticker_df['date'].min(),
                    'max_date': ticker_df['date'].max()
                }
            
            all_sentiment_data = existing_df.to_dict('records')
        else:
            all_sentiment_data = []
        
        total_stocks = sum(len(stocks) for stocks in self.stocks.values())
        stock_counter = 0
        
        # Generate all trading days (Mon-Fri) in the range
        all_trading_days = []
        current = start_date
        while current <= end_date:
            if current.weekday() < 5:  # Mon-Fri
                all_trading_days.append(current.date())
            current += timedelta(days=1)
        
        print(f"\nTotal trading days in range: {len(all_trading_days)}")
        print(f"From {all_trading_days[0]} to {all_trading_days[-1]}")
        
        # Calculate what needs to be done
        work_summary = []
        for sector, stocks in self.stocks.items():
            for ticker, company_name in stocks.items():
                if ticker in existing_data_by_ticker:
                    existing_dates = existing_data_by_ticker[ticker]['dates']
                    missing_count = len([d for d in all_trading_days if d not in existing_dates])
                else:
                    missing_count = len(all_trading_days)
                
                if missing_count > 0:
                    work_summary.append((ticker, company_name, missing_count))
        
        if work_summary:
            print(f"\n{'='*80}")
            print(f"WORK PLAN: {len(work_summary)} stocks need updates")
            print(f"{'='*80}")
            total_missing = sum(x[2] for x in work_summary)
            print(f"Total dates to fill: {total_missing:,}")
            print(f"\nStocks needing updates:")
            for ticker, name, count in work_summary[:10]:  # Show first 10
                print(f"  {ticker:8s} {name:30s} - {count:4d} dates")
            if len(work_summary) > 10:
                print(f"  ... and {len(work_summary) - 10} more")
        else:
            print(f"\n✓ ALL DATA UP TO DATE - Nothing to backfill!")
            return pd.read_csv(self.history_file)
        
        print()
        
        # Process each stock
        for sector, stocks in self.stocks.items():
            print(f"\n{'='*80}")
            print(f"SECTOR: {sector.upper()}")
            print(f"{'='*80}")
            
            for ticker, company_name in stocks.items():
                stock_counter += 1
                
                # Determine what dates are missing for this stock
                if ticker in existing_data_by_ticker:
                    existing_dates = existing_data_by_ticker[ticker]['dates']
                    missing_dates = [d for d in all_trading_days if d not in existing_dates]
                    
                    print(f"\n[{stock_counter}/{total_stocks}] {company_name} ({ticker})")
                    print(f"  Existing: {existing_data_by_ticker[ticker]['count']} records")
                    print(f"  Range: {existing_data_by_ticker[ticker]['min_date']} to {existing_data_by_ticker[ticker]['max_date']}")
                    print(f"  Missing: {len(missing_dates)} dates")
                    
                    if len(missing_dates) == 0:
                        print(f"  ✓ COMPLETE - No gaps to fill")
                        continue
                else:
                    missing_dates = all_trading_days
                    print(f"\n[{stock_counter}/{total_stocks}] {company_name} ({ticker})")
                    print(f"  NEW STOCK - Need all {len(missing_dates)} dates")
                
                # Group missing dates into weekly batches for efficient fetching
                if len(missing_dates) == 0:
                    continue
                
                print(f"  {'─'*60}")
                
                # Sort missing dates
                missing_dates.sort()
                
                # Process in weekly batches
                batch_start_idx = 0
                week_count = 0
                stock_sentiment_count = 0
                
                while batch_start_idx < len(missing_dates):
                    # Find a week's worth of contiguous or near-contiguous dates
                    batch_end_idx = min(batch_start_idx + 7, len(missing_dates))
                    batch_dates = missing_dates[batch_start_idx:batch_end_idx]
                    
                    # Use the first and last date of the batch for RSS query
                    batch_start_date = datetime.combine(batch_dates[0], datetime.min.time())
                    batch_end_date = datetime.combine(batch_dates[-1], datetime.min.time())
                    
                    week_count += 1
                    
                    # Get sentiment for this week
                    weekly_sentiments = self.calculate_stock_sentiment_for_week(
                        ticker, company_name, batch_start_date, batch_end_date
                    )
                    
                    # Only keep sentiments for dates we actually need
                    filtered_sentiments = [
                        s for s in weekly_sentiments 
                        if datetime.strptime(s['date'], '%Y-%m-%d').date() in batch_dates
                    ]
                    
                    all_sentiment_data.extend(filtered_sentiments)
                    stock_sentiment_count += len(filtered_sentiments)
                    
                    # Progress indicator every 50 weeks
                    if week_count % 50 == 0:
                        print(f"  Week {week_count}: {batch_start_date.strftime('%Y-%m-%d')} | Filled: {stock_sentiment_count}", flush=True)
                    
                    # Save checkpoint every 50 weeks (more frequent saves)
                    if week_count % 50 == 0:
                        temp_df = pd.DataFrame(all_sentiment_data)
                        # Remove duplicates before saving
                        temp_df = temp_df.drop_duplicates(subset=['ticker', 'date'], keep='last')
                        temp_df.to_csv(self.history_file, index=False)
                        print(f"  💾 Checkpoint saved: {len(temp_df)} total records", flush=True)
                    
                    batch_start_idx = batch_end_idx
                    time.sleep(0.3)  # Rate limiting (reduced from 0.5)
                
                print(f"  ✓ Filled {stock_sentiment_count} missing dates")
                
                # Save after each stock
                temp_df = pd.DataFrame(all_sentiment_data)
                temp_df = temp_df.drop_duplicates(subset=['ticker', 'date'], keep='last')
                temp_df.to_csv(self.history_file, index=False)

        
        # Final save
        print(f"\n{'='*80}")
        print("FINALIZING DATA")
        print(f"{'='*80}")
        
        df = pd.DataFrame(all_sentiment_data)
        
        # Remove duplicates (keep last)
        df = df.drop_duplicates(subset=['ticker', 'date'], keep='last')
        
        # Sort by ticker and date
        df = df.sort_values(['ticker', 'date'])
        
        # Save final version
        df.to_csv(self.history_file, index=False)
        
        print(f"\n✓ Complete sentiment history saved: {self.history_file}")
        print(f"Total records: {len(df):,}")
        print(f"Unique stocks: {df['ticker'].nunique()}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Summary statistics
        print(f"\n{'='*80}")
        print("SUMMARY BY STOCK")
        print(f"{'='*80}")
        
        for ticker in sorted(df['ticker'].unique()):
            ticker_data = df[df['ticker'] == ticker]
            avg_sentiment = ticker_data['sentiment_compound'].mean()
            days_with_news = len(ticker_data[ticker_data['news_count'] > 0])
            print(f"{ticker:8s} | {len(ticker_data):5d} days | Avg: {avg_sentiment:+.3f} | News coverage: {days_with_news:5d} days ({100*days_with_news/len(ticker_data):.1f}%)")
        
        print(f"\n{'='*80}")
        print("✓ BACKFILL COMPLETE!")
        print(f"{'='*80}")
        
        return df

def main():
    """
    Main execution: Backfill sentiment for all stocks from 2016 to present.
    """
    backfiller = FullHistoricalSentimentBackfill()
    
    # Run backfill for entire stock history period
    sentiment_df = backfiller.backfill_all_stocks(
        start_date_str='2016-01-06',  # Match stock data start
        end_date_str=None  # Through today
    )
    
    print("\n" + "="*80)
    print("READY FOR TRAINING")
    print("="*80)
    print(f"Sentiment data: data/sentiment_history_complete.csv")
    print(f"Stock data: data/multi_sector_stocks.csv")
    print("\nNext steps:")
    print("1. Merge sentiment with stock data by ticker and date")
    print("2. Use sentiment features in your ML models")
    print("3. Analyze sentiment impact on returns")
    print("="*80)

if __name__ == "__main__":
    main()
