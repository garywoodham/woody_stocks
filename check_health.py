#!/usr/bin/env python3
"""
Quick system health check - run anytime to see status
"""

import pandas as pd
import os
from datetime import datetime, timedelta

def check_system_health():
    print("\n" + "="*80)
    print("🏥 STOCK PREDICTION SYSTEM - HEALTH CHECK")
    print("="*80 + "\n")
    
    # 1. Data Files
    print("📁 DATA FILES:")
    files = [
        ('data/multi_sector_stocks.csv', 'Stock data'),
        ('data/sentiment_history.csv', 'Sentiment history'),
        ('sentiment_data.csv', 'Latest sentiment'),
        ('stock_recommendations.csv', 'Recommendations'),
        ('portfolio_allocation.csv', 'Portfolio'),
        ('predictions_refined.csv', 'Predictions')
    ]
    
    for file, desc in files:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024
            df = pd.read_csv(file)
            print(f"  ✅ {desc:20s} {len(df):>6d} rows, {size:>7.1f} KB")
        else:
            print(f"  ❌ {desc:20s} MISSING")
    
    # 2. Models
    print(f"\n🤖 MODELS:")
    if os.path.exists('models'):
        models = [f for f in os.listdir('models') if f.endswith('_refined.joblib')]
        print(f"  ✅ {len(models)} trained models")
    else:
        print(f"  ❌ Models directory missing")
    
    # 3. Sentiment History
    print(f"\n📰 SENTIMENT TRACKING:")
    if os.path.exists('data/sentiment_history.csv'):
        df_sent = pd.read_csv('data/sentiment_history.csv')
        df_sent['date'] = pd.to_datetime(df_sent['date'])
        
        print(f"  Total records:    {len(df_sent):>6d}")
        print(f"  Unique dates:     {df_sent['date'].nunique():>6d}")
        print(f"  Date range:       {df_sent['date'].min().date()} to {df_sent['date'].max().date()}")
        print(f"  Days of history:  {(df_sent['date'].max() - df_sent['date'].min()).days:>6d}")
        
        # Check if current
        latest = df_sent['date'].max().date()
        today = datetime.now().date()
        days_old = (today - latest).days
        
        if days_old == 0:
            print(f"  Status:           ✅ UP TO DATE (today)")
        elif days_old == 1:
            print(f"  Status:           ⚠️  1 day old (run fetch_sentiment_historical.py)")
        else:
            print(f"  Status:           ❌ {days_old} days old (automation issue?)")
        
        # Sentiment stats
        avg_sentiment = df_sent['sentiment_compound'].mean()
        print(f"  Avg sentiment:    {avg_sentiment:>6.3f}")
        
    else:
        print(f"  ❌ No historical sentiment yet")
        print(f"     Run: python fetch_sentiment_historical.py")
    
    # 4. Recommendations
    print(f"\n🎯 LATEST RECOMMENDATIONS:")
    if os.path.exists('stock_recommendations.csv'):
        df_rec = pd.read_csv('stock_recommendations.csv')
        buy = (df_rec['Recommendation'] == 'BUY').sum()
        hold = (df_rec['Recommendation'] == 'HOLD').sum()
        sell = (df_rec['Recommendation'] == 'SELL').sum()
        
        print(f"  BUY:  {buy:>3d} stocks")
        print(f"  HOLD: {hold:>3d} stocks")
        print(f"  SELL: {sell:>3d} stocks")
        
        # Top picks
        top_buys = df_rec[df_rec['Recommendation'] == 'BUY'].nlargest(3, 'Score')
        if not top_buys.empty:
            print(f"\n  Top BUY signals:")
            for _, row in top_buys.iterrows():
                print(f"    {row['Ticker']:6s} - {row['Stock'][:25]:25s} Score: {row['Score']:.3f}")
    
    # 5. Portfolio
    print(f"\n💰 PORTFOLIO ALLOCATION:")
    if os.path.exists('portfolio_allocation.csv'):
        df_port = pd.read_csv('portfolio_allocation.csv')
        total = df_port['Allocation_Amount'].sum()
        positions = len(df_port)
        
        print(f"  Total invested:   ${total:>10,.0f}")
        print(f"  Positions:        {positions:>6d}")
        print(f"  Cash reserve:     ${100000-total:>10,.0f}")
        
        # Top allocations
        print(f"\n  Top allocations:")
        for _, row in df_port.nlargest(3, 'Allocation_Amount').iterrows():
            pct = (row['Allocation_Amount'] / 100000) * 100
            print(f"    {row['Ticker']:6s} ${row['Allocation_Amount']:>8,.0f} ({pct:>5.1f}%)")
    
    print(f"\n{'='*80}")
    print("✅ SYSTEM HEALTH CHECK COMPLETE")
    print("="*80 + "\n")
    
    print("💡 Quick Actions:")
    print("   • Update sentiment:  python fetch_sentiment_historical.py")
    print("   • Retrain models:    python train_refined_models.py")
    print("   • New predictions:   python predict_refined.py")
    print("   • View dashboard:    python dashboard.py")
    print()

if __name__ == '__main__':
    check_system_health()
