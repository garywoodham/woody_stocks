#!/bin/bash
# Watch backfill progress with minute-by-minute updates

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║     SENTIMENT BACKFILL MONITOR - Updates Every 60 Seconds         ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

while true; do
    # Get current timestamp
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Run Python analysis
    python3 << 'PYEOF'
import pandas as pd
import os
from datetime import datetime

file_path = 'data/sentiment_history_complete.csv'

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    
    total = len(df)
    target = 156780
    pct = (total / target) * 100
    stocks = df['ticker'].nunique()
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    
    # Check progress on incomplete stocks
    stocks_per_date = df.groupby('ticker').size()
    complete = sum(stocks_per_date >= 2613)
    
    print(f"║ Time:     {datetime.now().strftime('%H:%M:%S')}")
    print(f"║ Progress: {total:,} / {target:,} records ({pct:.1f}%)")
    print(f"║ Stocks:   {complete}/60 complete, {stocks - complete} in progress")
    print(f"║ File:     {size_mb:.2f} MB")
    
    # Show current work
    incomplete = stocks_per_date[stocks_per_date < 2613].sort_values(ascending=False)
    if len(incomplete) > 0:
        print(f"║")
        print(f"║ Current:")
        for ticker, count in incomplete.head(3).items():
            pct = (count / 2613) * 100
            bar = '█' * int(pct/5) + '░' * (20 - int(pct/5))
            print(f"║   {ticker:6s} [{bar}] {pct:5.1f}%")
else:
    print("║ ⚠️  File not found")
PYEOF
    
    echo "╠════════════════════════════════════════════════════════════════════╣"
    
    # Wait 60 seconds
    sleep 60
done
