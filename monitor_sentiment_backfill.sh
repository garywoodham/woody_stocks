#!/bin/bash
# Monitor the sentiment backfill process and notify when complete

echo "Monitoring sentiment backfill process..."
echo "Process PID: $(pgrep -f backfill_full_sentiment)"
echo ""

while true; do
    # Check if process is still running
    if ! pgrep -f "backfill_full_sentiment" > /dev/null; then
        echo ""
        echo "================================"
        echo "✓ BACKFILL PROCESS COMPLETED!"
        echo "================================"
        echo ""
        echo "Final log output:"
        tail -50 sentiment_backfill.log
        echo ""
        echo "Results:"
        if [ -f data/sentiment_history_complete.csv ]; then
            lines=$(wc -l < data/sentiment_history_complete.csv)
            echo "  Records created: $((lines - 1))"
            echo "  File: data/sentiment_history_complete.csv"
            ls -lh data/sentiment_history_complete.csv
        fi
        break
    fi
    
    # Show progress every 60 seconds
    echo "[$(date '+%H:%M:%S')] Still running... Latest output:"
    tail -5 sentiment_backfill.log | grep -E "SECTOR|^\[|Week|Completed" | tail -3
    echo ""
    
    sleep 60
done
