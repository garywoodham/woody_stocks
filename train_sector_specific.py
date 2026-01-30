#!/usr/bin/env python3
"""
Sector-Specific Model Training
===============================
Trains specialized models for each sector to capture sector-specific patterns
- Banking stocks behave differently than Tech stocks
- Pharma has different drivers than Energy
- Separate models allow for specialized feature importance
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
import joblib
import os
from collections import defaultdict

warnings.filterwarnings('ignore')

from train_refined_models import create_optimized_features, create_targets, get_best_lgb_params

print("=" * 80)
print("SECTOR-SPECIFIC MODEL TRAINING")
print("=" * 80)

# Load data
df = pd.read_csv('data/multi_sector_stocks.csv')
df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

# Load sentiment
try:
    sentiment_df = pd.read_csv('data/sentiment_history.csv')
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'], utc=True).dt.tz_localize(None)
    has_sentiment = True
    print(f"✓ Loaded sentiment data")
except FileNotFoundError:
    sentiment_df = pd.DataFrame()
    has_sentiment = False

HORIZONS = [1, 5, 21]

# Get sector breakdown
sector_counts = df.groupby('Sector')['Ticker'].nunique().sort_values(ascending=False)
print(f"\nSector Breakdown:")
for sector, count in sector_counts.items():
    print(f"  {sector:20s}: {count:2d} stocks")

sectors = df['Sector'].unique()

# Train sector-specific models
os.makedirs('models/sector_specific', exist_ok=True)

sector_models = {}
sector_performance = defaultdict(lambda: defaultdict(list))

for sector_idx, sector in enumerate(sectors, 1):
    print(f"\n{'='*80}")
    print(f"SECTOR {sector_idx}/{len(sectors)}: {sector}")
    print(f"{'='*80}")
    
    # Get stocks in this sector
    sector_stocks = df[df['Sector'] == sector]['Ticker'].unique()
    print(f"Training on {len(sector_stocks)} stocks")
    
    sector_models[sector] = {}
    
    for stock_idx, ticker in enumerate(sector_stocks, 1):
        print(f"\n  [{stock_idx}/{len(sector_stocks)}] {ticker}")
        
        df_stock = df[df['Ticker'] == ticker].copy()
        
        if len(df_stock) < 300:
            print(f"    ⚠️ Insufficient data ({len(df_stock)} days)")
            continue
        
        # Set index
        if 'Date' in df_stock.columns:
            df_stock = df_stock.set_index('Date')
        
        # Create features
        try:
            df_stock = create_optimized_features(df_stock, 'daily', sentiment_df if has_sentiment else None)
            df_stock, targets = create_targets(df_stock, HORIZONS)
        except Exception as e:
            print(f"    ⚠️ Feature creation failed: {str(e)[:50]}")
            continue
        
        # Clean data
        df_stock = df_stock.replace([np.inf, -np.inf], np.nan)
        feature_cols = [col for col in df_stock.columns 
                       if col not in ['target_1', 'target_5', 'target_21', 'Ticker', 'Stock', 'Sector']]
        
        stock_models = {}
        
        for horizon in HORIZONS:
            target_col = f'target_{horizon}'
            if target_col not in df_stock.columns:
                continue
            
            # Prepare data
            df_clean = df_stock[feature_cols + [target_col]].dropna()
            if len(df_clean) < 100:
                continue
            
            X = df_clean[feature_cols].values
            y = df_clean[target_col].values
            
            # Train/test split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Calculate class weight
            pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1]) if y_train.sum() > 0 else 1.0
            
            # Train model with sector-optimized parameters
            params = get_best_lgb_params(horizon)
            params['scale_pos_weight'] = pos_weight
            
            train_data = lgb.Dataset(X_train, label=y_train)
            test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=400,
                valid_sets=[test_data],
                callbacks=[lgb.early_stopping(stopping_rounds=40), lgb.log_evaluation(period=0)]
            )
            
            # Evaluate
            y_pred_proba = model.predict(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int)
            accuracy = np.mean(y_pred == y_test)
            
            # Direction-specific accuracy
            up_mask = y_test == 1
            down_mask = y_test == 0
            up_acc = np.mean(y_pred[up_mask] == y_test[up_mask]) if up_mask.sum() > 0 else 0
            down_acc = np.mean(y_pred[down_mask] == y_test[down_mask]) if down_mask.sum() > 0 else 0
            
            print(f"    {horizon:2d}d: {accuracy:.2%} (UP: {up_acc:.2%}, DOWN: {down_acc:.2%})")
            
            stock_models[horizon] = {
                'model': model,
                'feature_cols': feature_cols,
                'accuracy': accuracy,
                'up_accuracy': up_acc,
                'down_accuracy': down_acc
            }
            
            # Track sector performance
            sector_performance[sector][horizon].append(accuracy)
        
        # Save stock models
        if stock_models:
            sector_models[sector][ticker] = stock_models

# Save sector-specific models
for sector, stocks in sector_models.items():
    if stocks:
        sector_safe = sector.replace('/', '_').replace(' ', '_')
        filepath = f'models/sector_specific/{sector_safe}.pkl'
        joblib.dump(stocks, filepath)
        print(f"✓ Saved {len(stocks)} models for {sector}")

# Performance summary
print(f"\n{'='*80}")
print("SECTOR PERFORMANCE SUMMARY")
print(f"{'='*80}\n")

summary_data = []
for sector in sectors:
    if sector in sector_performance:
        row = {'Sector': sector}
        for horizon in HORIZONS:
            accs = sector_performance[sector][horizon]
            if accs:
                row[f'{horizon}d_accuracy'] = f"{np.mean(accs):.2%}"
                row[f'{horizon}d_count'] = len(accs)
            else:
                row[f'{horizon}d_accuracy'] = 'N/A'
                row[f'{horizon}d_count'] = 0
        summary_data.append(row)

df_summary = pd.DataFrame(summary_data)
print(df_summary.to_string(index=False))

# Save summary
df_summary.to_csv('models/sector_specific/performance_summary.csv', index=False)

print(f"\n{'='*80}")
print(f"✓ Training complete")
print(f"  • Trained models for {len(sector_models)} sectors")
print(f"  • Total stock models: {sum(len(stocks) for stocks in sector_models.values())}")
print(f"  • Saved to models/sector_specific/")
print(f"{'='*80}\n")

# Save configuration
config = {
    'sectors': list(sector_models.keys()),
    'horizons': HORIZONS,
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
    'num_stocks_per_sector': {sector: len(stocks) for sector, stocks in sector_models.items()}
}

joblib.dump(config, 'models/sector_specific/config.pkl')
print("✓ Saved configuration\n")
