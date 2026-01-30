#!/bin/bash
# Watch backfill progress with updates every 3 minutes

echo "========================================================================"
echo "10-YEAR BACKFILL MONITOR - Updates Every 3 Minutes"
echo "========================================================================"
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    clear
    echo "========================================================================"
    echo "10-YEAR SENTIMENT BACKFILL - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================================================"
    echo ""
    
    # Check if process is running
    if [ -f backfill.pid ]; then
        PID=$(cat backfill.pid)
        if ps -p $PID > /dev/null 2>&1; then
            RUNTIME=$(ps -p $PID -o etime= | tr -d ' ')
            echo "✅ Process RUNNING (PID: $PID, Runtime: $RUNTIME)"
        else
            echo "❌ Process STOPPED (Last PID: $PID)"
        fi
    else
        echo "❌ No PID file found"
    fi
    echo ""
    
    # Show data stats
    if [ -f data/sentiment_history_complete.csv ]; then
        python3 << 'EOF'
import pandas as pd

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

df = pd.read_csv('data/sentiment_history_complete.csv')
ticker_counts = df['ticker'].value_counts()

total_records = len(df)
total_target = 60 * 2613
pct_complete = (total_records / total_target) * 100

print(f"📊 PROGRESS:")
print(f"   Total Records: {total_records:,} / {total_target:,} ({pct_complete:.1f}%)")
print(f"   Stocks: {df['ticker'].nunique()}/60")
print()

completed = [t for t in all_stocks if t in ticker_counts.index and ticker_counts[t] >= 2000]
in_progress = [(t, ticker_counts[t]) for t in all_stocks if t in ticker_counts.index and ticker_counts[t] < 2000]
not_started = [t for t in all_stocks if t not in ticker_counts.index]

print(f"   ✅ Completed: {len(completed)}/60")
print(f"   🔄 In Progress: {len(in_progress)}")
print(f"   ⏳ Not Started: {len(not_started)}")
print()

if in_progress:
    print("CURRENT STOCK:")
    for t, count in in_progress:
        pct = (count/2613)*100
        bar_length = 40
        filled = int(bar_length * count / 2613)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"   {t:8s} [{bar}] {count:4d}/2613 ({pct:.1f}%)")
    print()

remaining = len(in_progress) + len(not_started)
estimated_min = remaining * 5
estimated_hrs = estimated_min / 60

print(f"⏱️  ESTIMATED TIME: ~{estimated_min} min ({estimated_hrs:.1f} hrs) for {remaining} stocks")
EOF
    else
        echo "❌ Data file not found"
    fi
    
    echo ""
    echo "------------------------------------------------------------------------"
    echo "RECENT ACTIVITY (last 8 lines):"
    echo "------------------------------------------------------------------------"
    tail -8 sentiment_backfill_10yr.log 2>/dev/null || echo "No log file"
    echo "------------------------------------------------------------------------"
    echo ""
    echo "Next update in 3 minutes... (Press Ctrl+C to stop)"
    
    sleep 180
done
