#!/usr/bin/env python3
"""
Monitor the 10-year sentiment backfill progress.
"""

import pandas as pd
import os
import time
from datetime import datetime

def monitor_progress():
    """Monitor and display backfill progress"""
    
    # All 60 stocks
    all_stocks = [
        'BA.L', 'LMT', 'NOC', 'RTX', 'RR.L',
        'BARC.L', 'HSBA.L', 'LLOY.L', 'NWG.L', 'STAN.L',
        'AZN.L', 'GSK.L', 'PFE', 'JNJ', 'MRNA',
        'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN',
        'GME', 'AMC', 'BB', 'PLTR', 'SOFI', 'RIVN', 'NIO', 'LCID', 'SPCE', 'PLUG',
        'HOOD', 'COIN', 'RIOT', 'MARA', 'TLRY',
        'XOM', 'CVX', 'COP', 'SLB', 'OXY',
        'WMT', 'PG', 'KO', 'PEP', 'COST',
        'CAT', 'GE', 'HON', 'UPS', 'MMM',
        'JPM', 'BAC', 'GS', 'MS', 'BLK',
        'DIS', 'NFLX', 'CMCSA', 'T', 'VZ'
    ]
    
    data_file = 'data/sentiment_history_complete.csv'
    pid_file = 'backfill.pid'
    log_file = 'sentiment_backfill_10yr.log'
    
    print("="*80)
    print("10-YEAR SENTIMENT BACKFILL MONITOR")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check if process is running
    if os.path.exists(pid_file):
        with open(pid_file) as f:
            pid = f.read().strip()
        
        try:
            os.kill(int(pid), 0)  # Check if process exists
            print(f"✅ Backfill process RUNNING (PID: {pid})")
        except:
            print(f"❌ Backfill process NOT RUNNING (last PID: {pid})")
    else:
        print("❌ No PID file found")
    
    print()
    
    # Check data file
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return
    
    # Load and analyze data
    df = pd.read_csv(data_file)
    
    print(f"📊 PROGRESS SUMMARY")
    print("="*80)
    print(f"Total records: {len(df):,}")
    print(f"Stocks completed: {df['ticker'].nunique()}/60")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print()
    
    # Show completion status
    ticker_counts = df['ticker'].value_counts().sort_index()
    
    completed = []
    in_progress = []
    not_started = []
    
    for ticker in all_stocks:
        if ticker not in ticker_counts.index:
            not_started.append(ticker)
        elif ticker_counts[ticker] < 2000:
            in_progress.append((ticker, ticker_counts[ticker]))
        else:
            completed.append(ticker)
    
    print(f"✅ Completed: {len(completed)}/60 stocks")
    print(f"🔄 In Progress: {len(in_progress)} stocks")
    print(f"⏳ Not Started: {len(not_started)} stocks")
    print()
    
    if in_progress:
        print("STOCKS IN PROGRESS:")
        for ticker, count in in_progress:
            pct = (count / 2613) * 100
            print(f"  {ticker:8s} - {count:4d}/2613 records ({pct:5.1f}%)")
        print()
    
    if not_started:
        print(f"NEXT UP ({len(not_started)} remaining):")
        for ticker in not_started[:10]:
            print(f"  {ticker}")
        if len(not_started) > 10:
            print(f"  ... and {len(not_started) - 10} more")
        print()
    
    # Show recent log entries
    if os.path.exists(log_file):
        print("RECENT LOG (last 5 lines):")
        print("-"*80)
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in lines[-5:]:
                print(line.rstrip())
        print("-"*80)
        print()
    
    # Estimate completion
    if in_progress or not_started:
        stocks_remaining = len(in_progress) + len(not_started)
        # Assume each stock takes ~5 minutes on average
        estimated_minutes = stocks_remaining * 5
        estimated_hours = estimated_minutes / 60
        
        print(f"⏱️  ESTIMATED TIME REMAINING:")
        print(f"   {stocks_remaining} stocks × ~5 min/stock = ~{estimated_minutes} minutes ({estimated_hours:.1f} hours)")
        print()
    
    print("="*80)
    print("To monitor in real-time:")
    print("  tail -f sentiment_backfill_10yr.log")
    print()
    print("To check progress again:")
    print("  python3 monitor_10yr_backfill.py")
    print("="*80)

if __name__ == "__main__":
    monitor_progress()
