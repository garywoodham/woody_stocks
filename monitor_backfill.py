#!/usr/bin/env python3
"""
Monitor the sentiment backfill progress by checking file size and records.
"""

import pandas as pd
import os
import time
from datetime import datetime

def check_progress():
    """Check and display current backfill progress"""
    
    file_path = 'data/sentiment_history_complete.csv'
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Get stats
    total_records = len(df)
    unique_dates = df['date'].nunique()
    unique_stocks = df['ticker'].nunique()
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    
    # Expected totals
    target_stocks = 60
    target_dates = 2613  # Trading days from 2016-01-06 to 2026-01-11
    target_records = target_stocks * target_dates  # ~156,780
    
    # Calculate progress
    progress_pct = (total_records / target_records) * 100
    
    # File size
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    
    # Display
    print(f"\n{'='*70}")
    print(f"⏱️  UPDATE - {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*70}")
    print(f"File:           {file_path}")
    print(f"Size:           {file_size_mb:.2f} MB")
    print(f"Total records:  {total_records:,} / {target_records:,} ({progress_pct:.1f}%)")
    print(f"Unique stocks:  {unique_stocks} / {target_stocks}")
    print(f"Unique dates:   {unique_dates} / {target_dates}")
    print(f"Date range:     {min_date} to {max_date}")
    
    # Show per-stock status
    stocks_per_date = df.groupby('ticker').size()
    complete_stocks = sum(stocks_per_date >= target_dates)
    incomplete_stocks = len(stocks_per_date) - complete_stocks
    
    print(f"\nStock status:")
    print(f"  ✅ Complete:    {complete_stocks} stocks")
    print(f"  🔄 In progress: {incomplete_stocks} stocks")
    
    if incomplete_stocks > 0:
        print(f"\n  Stocks being filled:")
        incomplete = stocks_per_date[stocks_per_date < target_dates].sort_values(ascending=False)
        for ticker, count in incomplete.head(10).items():
            pct = (count / target_dates) * 100
            print(f"    {ticker:8s} {count:5d}/{target_dates} ({pct:5.1f}%)")
    
    print(f"{'='*70}\n")

if __name__ == "__main__":
    # Run continuous monitoring
    print("Starting sentiment backfill monitor...")
    print("Updates every 60 seconds. Press Ctrl+C to stop.\n")
    
    try:
        while True:
            check_progress()
            time.sleep(60)  # Update every minute
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
