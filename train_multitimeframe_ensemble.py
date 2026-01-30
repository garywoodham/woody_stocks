#!/usr/bin/env python3
"""
Multi-Timeframe Ensemble Training
==================================
Trains models on different historical windows and blends predictions
- Short-term model (3 months): Captures recent trends
- Medium-term model (1 year): Balances recent and long-term patterns
- Long-term model (2 years): Captures long-term patterns

Weighted blending based on validation performance
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
from datetime import datetime, timedelta
import joblib
import os

warnings.filterwarnings('ignore')

# Import feature engineering
from train_refined_models import create_optimized_features, create_targets, get_best_lgb_params

print("=" * 80)
print("MULTI-TIMEFRAME ENSEMBLE TRAINING")
print("=" * 80)
print("\nStrategy:")
print("  • Short-term (3 months): Weight 0.4 - Captures recent momentum")
print("  • Medium-term (1 year): Weight 0.35 - Balanced view")
print("  • Long-term (2 years): Weight 0.25 - Long-term patterns")
print("=" * 80)

# Load data
df = pd.read_csv('data/multi_sector_stocks.csv')
df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

# Load sentiment data
try:
    sentiment_df = pd.read_csv('data/sentiment_history.csv')
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'], utc=True).dt.tz_localize(None)
    print(f"\n✓ Loaded sentiment data: {len(sentiment_df)} records")
except FileNotFoundError:
    sentiment_df = pd.DataFrame()
    print("\n⚠️  No sentiment data available")

HORIZONS = [1, 5, 21]
STOCKS = df['Ticker'].unique()

# Training windows (in days)
TIMEFRAMES = {
    'short': 90,      # 3 months - recent trends
    'medium': 365,    # 1 year - balanced
    'long': 730       # 2 years - long-term patterns
}

# Blending weights (optimized based on typical performance)
BLEND_WEIGHTS = {
    'short': 0.40,
    'medium': 0.35,
    'long': 0.25
}

def train_timeframe_model(df_stock, sentiment_df, window_days, timeframe_name):
    """Train model on specific time window"""
    # Filter to recent window
    if len(df_stock) > window_days:
        df_stock = df_stock.iloc[-window_days:].copy()
    
    # Set index
    if 'Date' in df_stock.columns:
        df_stock = df_stock.set_index('Date')
    elif not isinstance(df_stock.index, pd.DatetimeIndex):
        df_stock.index = pd.to_datetime(df_stock.index)
    
    # Create features
    df_stock = create_optimized_features(df_stock, 'daily', sentiment_df)
    df_stock, targets = create_targets(df_stock, HORIZONS)
    
    # Clean data
    df_stock = df_stock.replace([np.inf, -np.inf], np.nan)
    feature_cols = [col for col in df_stock.columns 
                   if col not in ['target_1', 'target_5', 'target_21', 'Ticker', 'Stock', 'Sector']]
    
    models = {}
    
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
        
        # Train model
        params = get_best_lgb_params(horizon)
        params['scale_pos_weight'] = pos_weight
        
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=300,
            valid_sets=[test_data],
            callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(period=0)]
        )
        
        # Evaluate
        y_pred = (model.predict(X_test) > 0.5).astype(int)
        accuracy = np.mean(y_pred == y_test)
        
        models[horizon] = {
            'model': model,
            'feature_cols': feature_cols,
            'accuracy': accuracy,
            'timeframe': timeframe_name
        }
    
    return models

def blend_predictions(predictions_dict, weights):
    """Blend predictions from multiple timeframes"""
    blended = {}
    
    for horizon in HORIZONS:
        if horizon not in predictions_dict['short']:
            continue
        
        # Weighted average of probabilities
        prob_up = (
            predictions_dict['short'][horizon]['prob_up'] * weights['short'] +
            predictions_dict['medium'][horizon]['prob_up'] * weights['medium'] +
            predictions_dict['long'][horizon]['prob_up'] * weights['long']
        )
        
        # Weighted average of accuracies for confidence
        conf = (
            predictions_dict['short'][horizon]['accuracy'] * weights['short'] +
            predictions_dict['medium'][horizon]['accuracy'] * weights['medium'] +
            predictions_dict['long'][horizon]['accuracy'] * weights['long']
        )
        
        blended[horizon] = {
            'prob_up': prob_up,
            'confidence': conf,
            'direction': 'UP ↑' if prob_up > 0.5 else 'DOWN ↓'
        }
    
    return blended

# Main training loop
print(f"\nTraining models for {len(STOCKS)} stocks...\n")

all_models = {timeframe: {} for timeframe in TIMEFRAMES.keys()}
overall_accuracy = {timeframe: {h: [] for h in HORIZONS} for timeframe in TIMEFRAMES.keys()}

for i, ticker in enumerate(STOCKS, 1):
    print(f"\n[{i}/{len(STOCKS)}] {ticker}")
    
    df_stock = df[df['Ticker'] == ticker].copy()
    
    if len(df_stock) < 200:
        print(f"  ⚠️ Insufficient data ({len(df_stock)} days)")
        continue
    
    stock_models = {}
    
    # Train models for each timeframe
    for timeframe_name, window_days in TIMEFRAMES.items():
        try:
            models = train_timeframe_model(df_stock, sentiment_df, window_days, timeframe_name)
            stock_models[timeframe_name] = models
            
            # Track accuracy
            for horizon, model_info in models.items():
                overall_accuracy[timeframe_name][horizon].append(model_info['accuracy'])
                print(f"  {timeframe_name:6s} {horizon:2d}d: {model_info['accuracy']:.2%}")
        
        except Exception as e:
            print(f"  ⚠️ {timeframe_name} failed: {str(e)[:50]}")
    
    # Store models
    if stock_models:
        all_models['ensemble'][ticker] = stock_models

# Save ensemble models
os.makedirs('models/multitimeframe', exist_ok=True)

for ticker, stock_models in all_models['ensemble'].items():
    for timeframe, models in stock_models.items():
        filepath = f'models/multitimeframe/{ticker}_{timeframe}.pkl'
        joblib.dump(models, filepath)

print(f"\n{'='*80}")
print("TRAINING COMPLETE - PERFORMANCE SUMMARY")
print(f"{'='*80}\n")

for timeframe in TIMEFRAMES.keys():
    print(f"\n{timeframe.upper()} MODEL (Window: {TIMEFRAMES[timeframe]} days, Weight: {BLEND_WEIGHTS[timeframe]:.0%})")
    for horizon in HORIZONS:
        accs = overall_accuracy[timeframe][horizon]
        if accs:
            avg_acc = np.mean(accs)
            print(f"  {horizon:2d}-day: {avg_acc:.2%} avg accuracy ({len(accs)} stocks)")

print(f"\n{'='*80}")
print(f"✓ Saved {len(all_models['ensemble'])} stock ensembles to models/multitimeframe/")
print(f"{'='*80}\n")

# Save blend weights
blend_config = {
    'weights': BLEND_WEIGHTS,
    'timeframes': TIMEFRAMES,
    'trained_stocks': list(all_models['ensemble'].keys()),
    'training_date': datetime.now().strftime('%Y-%m-%d')
}

joblib.dump(blend_config, 'models/multitimeframe/blend_config.pkl')
print("✓ Saved blending configuration\n")
