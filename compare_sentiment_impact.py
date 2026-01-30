#!/usr/bin/env python3
"""
A/B Test: Model Performance WITH vs WITHOUT Sentiment Features
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

# Simple test on a few stocks
test_stocks = [
    ('NVDA', 'NVIDIA'),
    ('AAPL', 'Apple'),
    ('BA.L', 'BAE Systems'),
    ('BARC.L', 'Barclays'),
]

print("="*70)
print("SENTIMENT IMPACT COMPARISON")
print("="*70)
print("\nComparing accuracy of models trained WITH vs WITHOUT sentiment\n")

# We'll look at the training log to extract WITH sentiment results
# Then train a quick model WITHOUT sentiment for comparison

print("Loading WITH sentiment results from training log...")

import re
with open('model_training_with_sentiment.log', 'r') as f:
    log = f.read()

# Extract results for our test stocks
with_sentiment = {}
for ticker, name in test_stocks:
    # Find the section for this stock
    pattern = rf'REFINED TRAINING: DAILY - {name}.*?(?=REFINED TRAINING:|COMPARISON|$)'
    section = re.search(pattern, log, re.DOTALL)
    
    if section:
        text = section.group(0)
        # Extract accuracies
        accs = re.findall(r'Accuracy: ([\d.]+)%', text)
        if accs:
            accs_float = [float(a) for a in accs]
            with_sentiment[ticker] = {
                'name': name,
                'acc_1d': accs_float[0] if len(accs_float) > 0 else 0,
                'acc_5d': accs_float[1] if len(accs_float) > 1 else 0,
                'acc_21d': accs_float[2] if len(accs_float) > 2 else 0,
                'average': sum(accs_float) / len(accs_float)
            }

print(f"✓ Loaded {len(with_sentiment)} stocks with sentiment results\n")

# Now train WITHOUT sentiment for comparison
print("Training models WITHOUT sentiment features...\n")

from train_refined_models import create_optimized_features

without_sentiment = {}

for ticker, name in test_stocks:
    print(f"Training {name} ({ticker}) WITHOUT sentiment...")
    
    try:
        # Load data
        df = pd.read_csv(f'data/{ticker.replace(".", "_")}_data.csv', parse_dates=['Date'], index_col='Date')
        df['Ticker'] = ticker
        
        # Create features WITHOUT sentiment (pass None)
        df_featured = create_optimized_features(df.copy(), 'daily', sentiment_df=None)
        
        # Create targets
        for h in [1, 5, 21]:
            df_featured[f'target_{h}d'] = (df_featured['Close'].shift(-h) > df_featured['Close']).astype(int)
        
        df_clean = df_featured.dropna()
        
        # Feature columns
        feature_cols = [col for col in df_clean.columns 
                       if col not in ['target_1d', 'target_5d', 'target_21d', 'Ticker', 
                                     'Open', 'High', 'Low', 'Close', 'Volume'] 
                       and 'sentiment' not in col.lower()]
        
        X = df_clean[feature_cols].values
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train/test split
        split_idx = int(len(X_scaled) * 0.8)
        
        results = []
        
        for h in [1, 5, 21]:
            y = df_clean[f'target_{h}d'].values
            X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Quick LightGBM
            up_ratio = y_train.mean()
            class_weight = (1 - up_ratio) / up_ratio if up_ratio > 0 else 1.0
            
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_test, label=y_test)
            
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'verbose': -1,
                'scale_pos_weight': class_weight,
            }
            
            model = lgb.train(params, train_data, num_boost_round=50, 
                            valid_sets=[valid_data], 
                            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
            
            y_pred = (model.predict(X_test) > 0.5).astype(int)
            accuracy = (y_pred == y_test).mean() * 100
            results.append(accuracy)
        
        without_sentiment[ticker] = {
            'name': name,
            'acc_1d': results[0],
            'acc_5d': results[1],
            'acc_21d': results[2],
            'average': sum(results) / len(results),
            'features': len(feature_cols)
        }
        
        print(f"  ✓ 1d: {results[0]:.2f}%, 5d: {results[1]:.2f}%, 21d: {results[2]:.2f}%, Avg: {sum(results)/3:.2f}%")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")

# Comparison table
print("\n" + "="*70)
print("RESULTS COMPARISON")
print("="*70)

print(f"\n{'Stock':<15} {'Horizon':<10} {'WITHOUT':<12} {'WITH':<12} {'Difference'}")
print("-" * 70)

all_diffs = []

for ticker in with_sentiment:
    if ticker in without_sentiment:
        w = with_sentiment[ticker]
        wo = without_sentiment[ticker]
        
        for h, label in [(1, '1d'), (5, '5d'), (21, '21d')]:
            acc_key = f'acc_{label}'
            without_acc = wo[acc_key]
            with_acc = w[acc_key]
            diff = with_acc - without_acc
            all_diffs.append(diff)
            
            print(f"{ticker:<15} {label:<10} {without_acc:6.2f}%     {with_acc:6.2f}%     {diff:+6.2f}%")

# Overall summary
print("-" * 70)
print(f"\n{'Stock':<15} {'WITHOUT Avg':<15} {'WITH Avg':<15} {'Difference'}")
print("-" * 70)

avg_diffs = []
for ticker in with_sentiment:
    if ticker in without_sentiment:
        without_avg = without_sentiment[ticker]['average']
        with_avg = with_sentiment[ticker]['average']
        diff = with_avg - without_avg
        avg_diffs.append(diff)
        
        impact = "✅" if diff > 0.5 else ("➖" if diff > -0.5 else "⚠️ ")
        print(f"{ticker:<15} {without_avg:6.2f}%        {with_avg:6.2f}%        {diff:+6.2f}%  {impact}")

print("-" * 70)
overall_diff = sum(avg_diffs) / len(avg_diffs) if avg_diffs else 0
print(f"{'OVERALL':<15} {'':15} {'':15} {overall_diff:+6.2f}%")

print(f"\n{'='*70}")
print("CONCLUSION")
print(f"{'='*70}\n")

if overall_diff > 1.0:
    print(f"✅ Sentiment features IMPROVE accuracy by {overall_diff:.2f}% on average")
elif overall_diff > 0.2:
    print(f"➕ Sentiment provides MODEST improvement (+{overall_diff:.2f}%)")
elif overall_diff > -0.2:
    print(f"➖ Sentiment has NEUTRAL impact ({overall_diff:+.2f}%)")
else:
    print(f"⚠️  Sentiment DECREASES accuracy by {abs(overall_diff):.2f}%")

positive = sum(1 for d in avg_diffs if d > 0.5)
neutral = sum(1 for d in avg_diffs if -0.5 <= d <= 0.5)
negative = sum(1 for d in avg_diffs if d < -0.5)

print(f"\nStock breakdown:")
print(f"  Improved with sentiment: {positive}/{len(avg_diffs)}")
print(f"  Neutral:                 {neutral}/{len(avg_diffs)}")
print(f"  Worse with sentiment:    {negative}/{len(avg_diffs)}")

print(f"\n{'='*70}")
