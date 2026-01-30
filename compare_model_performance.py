#!/usr/bin/env python3
"""
Compare model performance with and without sentiment history
"""

import re
import pandas as pd
from datetime import datetime

def extract_metrics_from_log(log_file):
    """Extract all training metrics from log file"""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        return None
    
    # Find sentiment type
    sentiment_match = re.search(r'Loaded (COMPLETE|HISTORICAL|PARTIAL|STATIC) sentiment', content)
    sentiment_type = sentiment_match.group(1) if sentiment_match else "NONE"
    
    records_match = re.search(r'Records: ([\d,]+)', content)
    sentiment_records = records_match.group(1) if records_match else "0"
    
    # Extract all stock results
    stock_pattern = r'REFINED TRAINING: DAILY - (.+?)\n.*?Samples: (\d+)'
    stocks = re.findall(stock_pattern, content)
    
    # Extract accuracies with horizons
    horizon_pattern = r'Horizon: (\d+)-d.*?Accuracy: ([\d.]+)%'
    accuracies = re.findall(horizon_pattern, content)
    
    results = []
    stock_idx = 0
    acc_idx = 0
    
    for stock, samples in stocks:
        stock_accs = {
            'stock': stock,
            'samples': int(samples),
        }
        
        # Each stock has 3 horizons (1d, 5d, 21d)
        for _ in range(3):
            if acc_idx < len(accuracies):
                horizon, acc = accuracies[acc_idx]
                stock_accs[f'acc_{horizon}d'] = float(acc)
                acc_idx += 1
        
        results.append(stock_accs)
    
    df = pd.DataFrame(results)
    
    return {
        'sentiment_type': sentiment_type,
        'sentiment_records': sentiment_records,
        'results': df,
        'avg_1d': df['acc_1d'].mean() if 'acc_1d' in df else 0,
        'avg_5d': df['acc_5d'].mean() if 'acc_5d' in df else 0,
        'avg_21d': df['acc_21d'].mean() if 'acc_21d' in df else 0,
        'total_stocks': len(df),
        'total_models': len(df) * 3
    }

def compare_performance():
    """Compare old vs new model performance"""
    print("="*70)
    print(" "*15 + "MODEL PERFORMANCE COMPARISON")
    print("="*70)
    
    # Load new results (with complete sentiment)
    new_results = extract_metrics_from_log('model_training_with_sentiment.log')
    
    # Try to find old results
    old_results = None
    for old_log in ['model_training.log', 'training_log.txt', 'backtest_log.txt']:
        old_results = extract_metrics_from_log(old_log)
        if old_results:
            break
    
    if new_results:
        print(f"\n📊 NEW TRAINING (with {new_results['sentiment_type']} sentiment)")
        print(f"   Records: {new_results['sentiment_records']}")
        print(f"   Stocks: {new_results['total_stocks']}")
        print(f"   Models: {new_results['total_models']}")
        print(f"\n   Average Accuracy:")
        print(f"     1-day:  {new_results['avg_1d']:.2f}%")
        print(f"     5-day:  {new_results['avg_5d']:.2f}%")
        print(f"     21-day: {new_results['avg_21d']:.2f}%")
        print(f"     Overall: {(new_results['avg_1d'] + new_results['avg_5d'] + new_results['avg_21d'])/3:.2f}%")
    
    if old_results:
        print(f"\n📊 OLD TRAINING (with {old_results['sentiment_type']} sentiment)")
        print(f"   Stocks: {old_results['total_stocks']}")
        print(f"   Models: {old_results['total_models']}")
        print(f"\n   Average Accuracy:")
        print(f"     1-day:  {old_results['avg_1d']:.2f}%")
        print(f"     5-day:  {old_results['avg_5d']:.2f}%")
        print(f"     21-day: {old_results['avg_21d']:.2f}%")
        print(f"     Overall: {(old_results['avg_1d'] + old_results['avg_5d'] + old_results['avg_21d'])/3:.2f}%")
    
    if new_results and old_results:
        print(f"\n📈 IMPROVEMENT:")
        delta_1d = new_results['avg_1d'] - old_results['avg_1d']
        delta_5d = new_results['avg_5d'] - old_results['avg_5d']
        delta_21d = new_results['avg_21d'] - old_results['avg_21d']
        delta_overall = ((new_results['avg_1d'] + new_results['avg_5d'] + new_results['avg_21d'])/3 - 
                        (old_results['avg_1d'] + old_results['avg_5d'] + old_results['avg_21d'])/3)
        
        print(f"     1-day:  {delta_1d:+.2f}%")
        print(f"     5-day:  {delta_5d:+.2f}%")
        print(f"     21-day: {delta_21d:+.2f}%")
        print(f"     Overall: {delta_overall:+.2f}%")
        
        if delta_overall > 0:
            print(f"\n✅ COMPLETE sentiment history improved accuracy by {delta_overall:.2f}%!")
        elif delta_overall < -1:
            print(f"\n⚠️  Accuracy decreased by {abs(delta_overall):.2f}%")
        else:
            print(f"\n➖ No significant change in accuracy")
    
    # Show top performers
    if new_results and 'results' in new_results and not new_results['results'].empty:
        df = new_results['results']
        if 'acc_1d' in df and 'acc_5d' in df and 'acc_21d' in df:
            df['avg_acc'] = (df['acc_1d'] + df['acc_5d'] + df['acc_21d']) / 3
            top5 = df.nlargest(5, 'avg_acc')[['stock', 'acc_1d', 'acc_5d', 'acc_21d', 'avg_acc']]
            
            print(f"\n🏆 TOP 5 PERFORMERS:")
            for idx, row in top5.iterrows():
                print(f"   {row['stock']:20} 1d:{row['acc_1d']:5.1f}% 5d:{row['acc_5d']:5.1f}% 21d:{row['acc_21d']:5.1f}% Avg:{row['avg_acc']:5.1f}%")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    compare_performance()
