#!/bin/bash
# Monitor sentiment backfill progress every 2 minutes

echo "Starting backfill monitor - updates every 2 minutes"
echo "Press Ctrl+C to stop monitoring"
echo "========================================"

while true; do
    clear
    echo "=== SENTIMENT BACKFILL STATUS ==="
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Check if process is running
    if ps aux | grep -v grep | grep backfill_full_sentiment_history.py > /dev/null; then
        PID=$(ps aux | grep -v grep | grep backfill_full_sentiment_history.py | awk '{print $2}')
        CPU=$(ps aux | grep -v grep | grep backfill_full_sentiment_history.py | awk '{print $3}')
        RUNTIME=$(ps -p $PID -o etime= | xargs)
        echo "✅ Process Running: PID $PID, CPU: ${CPU}%, Runtime: $RUNTIME"
    else
        echo "❌ Process NOT running"
    fi
    echo ""
    
    # Count total records
    if [ -f /workspaces/woody_stocks/data/sentiment_history_complete.csv ]; then
        TOTAL=$(wc -l < /workspaces/woody_stocks/data/sentiment_history_complete.csv)
        echo "📊 Total Records: $TOTAL"
        echo ""
        
        # Show latest entries
        echo "Latest entry:"
        tail -1 /workspaces/woody_stocks/data/sentiment_history_complete.csv | awk -F',' '{print "  " $1 " - " $2}'
        echo ""
        
        # Count unique tickers
        TICKERS=$(cut -d',' -f1 /workspaces/woody_stocks/data/sentiment_history_complete.csv | sort -u | grep -v ticker | wc -l)
        echo "📈 Stocks with data: $TICKERS/60"
        echo ""
        
        # Show which tickers have data
        echo "Completed/In Progress:"
        cut -d',' -f1 /workspaces/woody_stocks/data/sentiment_history_complete.csv | sort -u | grep -v ticker | xargs
    else
        echo "⚠️  Data file not found"
    fi
    
    echo ""
    echo "========================================"
    echo "Next update in 2 minutes..."
    
    sleep 120
done
