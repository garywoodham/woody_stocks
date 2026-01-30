"""
Analyze results from horizon-optimized training
Compares models with horizon-specific features vs. all features for all horizons
"""

import pandas as pd
import re

def parse_training_log(log_file):
    """Parse training log and extract accuracies"""
    results = []
    
    with open(log_file, 'r') as f:
        content = f.read()
        
    # Split by stock sections
    sections = content.split('REFINED TRAINING: DAILY - ')
    
    for section in sections[1:]:
        lines = section.split('\n')
        stock_name = lines[0].strip()
        
        ticker = None
        for line in lines:
            if '✓ Saved to models/' in line:
                ticker = line.split('models/')[1].split('_daily')[0]
                break
        
        # Extract accuracies for each horizon
        horizon_data = {}
        current_horizon = None
        features_count = {}
        
        for line in lines:
            if 'Horizon:' in line:
                current_horizon = line.split('Horizon:')[1].strip().split()[0]
            elif 'Features:' in line and 'Samples:' in line:
                try:
                    feat_count = int(line.split('Features:')[1].strip())
                    if current_horizon:
                        features_count[current_horizon] = feat_count
                except:
                    pass
            elif 'Accuracy:' in line and '%' in line and current_horizon:
                try:
                    acc = float(line.split('Accuracy:')[1].split('%')[0].strip())
                    horizon_data[current_horizon] = acc
                except:
                    pass
        
        if ticker and horizon_data:
            for h, acc in horizon_data.items():
                results.append({
                    'Ticker': ticker,
                    'Stock': stock_name,
                    'Horizon': h,
                    'Accuracy': acc,
                    'Features': features_count.get(h, 0)
                })
    
    return pd.DataFrame(results)

print("="*100)
print("HORIZON-OPTIMIZED TRAINING RESULTS")
print("Feature Strategy: 1d/5d use Technical+Sentiment only, 21d adds Regime+Earnings")
print("="*100)

# Parse new results
df_new = parse_training_log('model_training_horizon_optimized.log')

# Parse old results for comparison
df_old = parse_training_log('model_training_with_regime_earnings.log')

if not df_new.empty:
    print(f"\n✓ Successfully trained: {df_new['Ticker'].nunique()} stocks")
    print(f"✓ Total models: {len(df_new)}")
    
    print("\n" + "="*100)
    print("ACCURACY COMPARISON: BEFORE vs AFTER Horizon Optimization")
    print("="*100)
    
    print(f"\n{'Horizon':<10} {'Before (All Features)':<25} {'After (Optimized)':<25} {'Change':<15}")
    print("-"*100)
    
    for horizon in ['1-d', '5-d', '21-d']:
        new_h = df_new[df_new['Horizon'] == horizon]
        old_h = df_old[df_old['Horizon'] == horizon] if not df_old.empty else pd.DataFrame()
        
        if not new_h.empty:
            new_avg = new_h['Accuracy'].mean()
            new_feat = new_h['Features'].iloc[0] if 'Features' in new_h.columns else 0
            
            if not old_h.empty:
                old_avg = old_h['Accuracy'].mean()
                change = new_avg - old_avg
                change_str = f"{change:+.2f}% {'🟢' if change > 0 else '🔴' if change < 0 else '⚪'}"
            else:
                old_avg = 0
                change_str = "N/A"
            
            print(f"{horizon:<10} {old_avg:>6.2f}% (99 feat)    →    {new_avg:>6.2f}% ({new_feat} feat)    {change_str}")
    
    # Overall comparison
    new_overall = df_new['Accuracy'].mean()
    old_overall = df_old['Accuracy'].mean() if not df_old.empty else 0
    change_overall = new_overall - old_overall
    
    print("-"*100)
    print(f"{'Overall':<10} {old_overall:>6.2f}%                 →    {new_overall:>6.2f}%               {change_overall:+.2f}% {'🟢' if change_overall > 0 else '🔴'}")
    
    print("\n" + "="*100)
    print("FEATURE USAGE ANALYSIS")
    print("="*100)
    
    # Count feature usage by horizon
    for horizon in ['1-d', '5-d', '21-d']:
        h_df = df_new[df_new['Horizon'] == horizon]
        if not h_df.empty:
            feat_count = h_df['Features'].iloc[0] if 'Features' in h_df.columns else 0
            print(f"\n{horizon}:")
            print(f"  Features: {feat_count}")
            if horizon in ['1-d', '5-d']:
                print(f"  Includes: Technical (55) + Sentiment (7) + Interactions (27)")
                print(f"  Excludes: Regime (9) + Earnings (5) - TOO NOISY FOR SHORT-TERM")
            else:
                print(f"  Includes: Technical (55) + Sentiment (7) + Regime (9) + Earnings (5) + Interactions (23)")
                print(f"  Strategy: Earnings events matter at 3-week horizon")
    
    print("\n" + "="*100)
    print("TOP PERFORMERS")
    print("="*100)
    
    top_10 = df_new.nlargest(10, 'Accuracy')[['Stock', 'Horizon', 'Accuracy', 'Features']]
    print("\n" + top_10.to_string(index=False))
    
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
    
    print(f"\nMedian Accuracy: {df_new['Accuracy'].median():.2f}%")
    print(f"Models above 55%: {(df_new['Accuracy'] > 55).sum()}/{len(df_new)} ({(df_new['Accuracy'] > 55).sum()/len(df_new)*100:.1f}%)")
    print(f"Models above 60%: {(df_new['Accuracy'] > 60).sum()}/{len(df_new)} ({(df_new['Accuracy'] > 60).sum()/len(df_new)*100:.1f}%)")
    print(f"Models above 65%: {(df_new['Accuracy'] > 65).sum()}/{len(df_new)} ({(df_new['Accuracy'] > 65).sum()/len(df_new)*100:.1f}%)")
    
    # Save results
    df_new.to_csv('model_results_horizon_optimized.csv', index=False)
    print(f"\n✓ Results saved to model_results_horizon_optimized.csv")
    
    print("\n" + "="*100)
    print("KEY INSIGHT:")
    print("="*100)
    print("\n✅ By removing regime/earnings features from 1-day and 5-day predictions,")
    print("   we reduce noise and let short-term technical patterns dominate.")
    print("\n✅ Keeping regime/earnings for 21-day captures long-term earnings cycles")
    print("   and market regime effects that matter at position-trading horizons.")
    print("\n" + "="*100)

else:
    print("\n⚠️  No results found in log file yet. Training may still be in progress.")
