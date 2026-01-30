#!/bin/bash
# Auto-restart script for sentiment backfill
# Ensures the process keeps running even if it crashes

LOG_FILE="sentiment_backfill.log"
PID_FILE="backfill.pid"
SCRIPT="backfill_full_sentiment_history.py"

echo "=== SENTIMENT BACKFILL KEEPER ==="
echo "Started: $(date)"
echo "Log: $LOG_FILE"
echo "=================================="

while true; do
    # Check if process is running
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            echo "[$(date '+%H:%M:%S')] Process running (PID: $PID)"
            sleep 300  # Check every 5 minutes
            continue
        else
            echo "[$(date '+%H:%M:%S')] Process died, restarting..."
        fi
    fi
    
    # Start the process
    echo "[$(date '+%H:%M:%S')] Starting backfill process..."
    nohup python3 "$SCRIPT" >> "$LOG_FILE" 2>&1 &
    NEW_PID=$!
    echo $NEW_PID > "$PID_FILE"
    echo "[$(date '+%H:%M:%S')] Started with PID: $NEW_PID"
    
    sleep 300  # Check again in 5 minutes
done
