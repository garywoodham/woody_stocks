#!/usr/bin/env python3
"""
Monitor model training progress and compare sentiment impact
"""

import time
import os
import re
from datetime import datetime

def parse_log(log_file):
    """Extract training metrics from log"""
    if not os.path.exists(log_file):
        return None
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract key metrics
    stocks_trained = len(re.findall(r'REFINED TRAINING: DAILY -', content))
    accuracies = re.findall(r'Accuracy: ([\d.]+)%', content)
    
    # Check for sentiment loading
    sentiment_line = re.search(r'Loaded (COMPLETE|HISTORICAL|PARTIAL|STATIC) sentiment', content)
    sentiment_type = sentiment_line.group(1) if sentiment_line else "NONE"
    sentiment_records = re.search(r'Records: ([\d,]+)', content)
    
    if accuracies:
        accuracies = [float(a) for a in accuracies]
        avg_acc = sum(accuracies) / len(accuracies)
    else:
        avg_acc = 0
    
    return {
        'stocks_trained': stocks_trained,
        'total_models': len(accuracies),
        'avg_accuracy': avg_acc,
        'sentiment_type': sentiment_type,
        'sentiment_records': sentiment_records.group(1) if sentiment_records else "0",
        'timestamp': datetime.now().strftime('%H:%M:%S')
    }

def monitor_training(log_file='model_training_with_sentiment.log', interval=30):
    """Monitor training progress"""
    print("🔍 Monitoring Model Training Progress")
    print("="*70)
    print(f"Log file: {log_file}")
    print(f"Update interval: {interval} seconds")
    print("Press Ctrl+C to stop monitoring\n")
    
    last_stocks = 0
    start_time = time.time()
    
    try:
        while True:
            stats = parse_log(log_file)
            
            if stats:
                # Calculate progress
                total_stocks = 35  # Expected total
                progress_pct = (stats['stocks_trained'] / total_stocks) * 100
                elapsed = time.time() - start_time
                
                # Clear and update
                print(f"\r[{stats['timestamp']}] Progress: {stats['stocks_trained']}/{total_stocks} stocks ({progress_pct:.0f}%)", end='')
                
                if stats['stocks_trained'] > last_stocks:
                    print(f"\n  Sentiment: {stats['sentiment_type']} ({stats['sentiment_records']} records)")
                    print(f"  Models trained: {stats['total_models']}")
                    print(f"  Avg accuracy: {stats['avg_accuracy']:.2f}%")
                    print(f"  Elapsed: {elapsed/60:.1f} min")
                    last_stocks = stats['stocks_trained']
                
                # Check if complete
                if stats['stocks_trained'] >= total_stocks:
                    print(f"\n\n✅ Training Complete!")
                    print(f"   Total models: {stats['total_models']}")
                    print(f"   Average accuracy: {stats['avg_accuracy']:.2f}%")
                    print(f"   Total time: {elapsed/60:.1f} minutes")
                    print(f"   Sentiment used: {stats['sentiment_type']}")
                    break
            else:
                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] Waiting for log file...", end='')
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n⏸️  Monitoring stopped")
        if stats:
            print(f"Last status: {stats['stocks_trained']} stocks trained, {stats['total_models']} models")

if __name__ == '__main__':
    import sys
    log_file = sys.argv[1] if len(sys.argv) > 1 else 'model_training_with_sentiment.log'
    monitor_training(log_file, interval=20)
