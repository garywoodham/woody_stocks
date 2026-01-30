"""
Enhanced Model Training - With Microstructure Features for 5-Day Predictions
Based on test results showing 0.40% improvement
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import talib
import warnings
import joblib
import os
warnings.filterwarnings('ignore')

# Import the original functions we need
from train_refined_models import (
    load_sentiment_data, 
    create_optimized_features as create_baseline_features,
    _add_neutral_sentiment
)

def add_microstructure_features(df):
    """Add microstructure features that improved 5-day predictions by 0.40%"""
    df = df.copy()
    
    # Bid-Ask Spread Proxy
    df['spread_proxy'] = (df['High'] - df['Low']) / (df['Close'] + 1e-10)
    df['spread_ma_5'] = df['spread_proxy'].rolling(5).mean()
    df['spread_volatility'] = df['spread_proxy'].rolling(10).std()
    
    # Price Clustering (round numbers)
    df['round_number_distance'] = df['Close'] % 10
    df['near_round_10'] = (df['round_number_distance'] < 1).astype(int)
    df['near_round_50'] = (df['Close'] % 50 < 5).astype(int)
    
    # Volume Clustering
    volume_std = df['Volume'].rolling(5).std()
    df['volume_clustering'] = df['Volume'] / (volume_std + 1)
    
    # Price Jumps
    df['price_jump'] = abs(df['High'] - df['Low'].shift(1)) / (df['Close'].shift(1) + 1e-10)
    df['price_jump_up'] = (df['Low'] > df['High'].shift(1)).astype(int)
    df['price_jump_down'] = (df['High'] < df['Low'].shift(1)).astype(int)
    
    return df

def create_enhanced_features_5day(df, time_period='daily', sentiment_df=None):
    """
    Enhanced feature creation specifically for 5-day predictions
    Adds microstructure features on top of baseline
    """
    # Start with baseline features
    df = create_baseline_features(df, time_period, sentiment_df)
    
    # Add microstructure features for 5-day predictions
    df = add_microstructure_features(df)
    
    return df

def create_optimized_features(df, time_period='daily', sentiment_df=None, horizon=None):
    """
    Smart feature selection based on horizon:
    - 1-day: baseline features (no improvement from additions)
    - 5-day: baseline + microstructure (+0.40%)
    - 21-day: baseline features (additions made it worse)
    """
    if horizon == 5 or horizon == '5d':
        # Use enhanced features for 5-day predictions
        return create_enhanced_features_5day(df, time_period, sentiment_df)
    else:
        # Use baseline for 1-day and 21-day predictions
        return create_baseline_features(df, time_period, sentiment_df)

def train_models_for_ticker(ticker_data, ticker, horizons=[1, 5, 21], time_period='daily', 
                            sentiment_df=None, save_dir='models'):
    """
    Train models for a single ticker across multiple horizons with horizon-specific features
    """
    os.makedirs(save_dir, exist_ok=True)
    results = {}
    
    for horizon in horizons:
        print(f"\n  Training {horizon}-day model for {ticker}...")
        
        # Create features with horizon-specific enhancements
        df_features = create_optimized_features(ticker_data.copy(), time_period, 
                                               sentiment_df, horizon=horizon)
        
        # Create target
        future_return = df_features['Close'].shift(-horizon) / df_features['Close'] - 1
        df_features[f'target_{horizon}d'] = (future_return > 0).astype(int)
        
        # Clean data
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        df_features = df_features.dropna()
        
        if len(df_features) < 100:
            print(f"    ⚠️  Insufficient data for {ticker} ({len(df_features)} rows)")
            continue
        
        # Feature columns
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 
                       'Stock Splits', 'Stock', 'Ticker', 'Sector'] + \
                      [f'target_{h}d' for h in [1, 5, 21]]
        feature_cols = [col for col in df_features.columns if col not in exclude_cols]
        
        # Prepare data
        X = df_features[feature_cols].values
        y = df_features[f'target_{horizon}d'].values
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Horizon-optimized hyperparameters
        if horizon == 1:
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'learning_rate': 0.02,
                'num_leaves': 80,
                'n_estimators': 400,
                'feature_fraction': 0.6,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_data_in_leaf': 10,
                'max_depth': -1,
                'verbose': -1
            }
        elif horizon == 5:
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'learning_rate': 0.03,
                'num_leaves': 50,
                'n_estimators': 300,
                'feature_fraction': 0.7,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_data_in_leaf': 15,
                'max_depth': -1,
                'verbose': -1
            }
        else:  # 21-day
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'learning_rate': 0.05,
                'num_leaves': 31,
                'n_estimators': 200,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_data_in_leaf': 20,
                'max_depth': -1,
                'verbose': -1
            }
        
        # Cross-validation
        accuracies = []
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train)
            
            accuracy = model.score(X_test, y_test)
            accuracies.append(accuracy)
        
        avg_accuracy = np.mean(accuracies)
        print(f"    ✓ Accuracy: {avg_accuracy*100:.2f}% ({'ENHANCED' if horizon == 5 else 'BASELINE'})")
        
        # Train final model on all data
        model = lgb.LGBMClassifier(**params)
        model.fit(X_scaled, y)
        
        # Save model, scaler, and feature names
        model_filename = os.path.join(save_dir, f'{ticker}_{horizon}d_model.pkl')
        scaler_filename = os.path.join(save_dir, f'{ticker}_{horizon}d_scaler.pkl')
        features_filename = os.path.join(save_dir, f'{ticker}_{horizon}d_features.pkl')
        
        joblib.dump(model, model_filename)
        joblib.dump(scaler, scaler_filename)
        joblib.dump(feature_cols, features_filename)
        
        results[horizon] = {
            'accuracy': avg_accuracy,
            'model_file': model_filename,
            'scaler_file': scaler_filename,
            'features_file': features_filename,
            'feature_count': len(feature_cols),
            'enhanced': horizon == 5
        }
    
    return results

def train_all_models(data_file='data/stock_data_processed.csv', save_dir='models'):
    """Train enhanced models for all tickers"""
    print("="*80)
    print("ENHANCED MODEL TRAINING - With Microstructure Features for 5-Day")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv(data_file, parse_dates=['Date'], index_col='Date')
    print(f"✓ Loaded {len(df):,} rows for {df['Ticker'].nunique()} tickers")
    
    # Load sentiment
    print("\nLoading sentiment data...")
    sentiment_df, sentiment_type = load_sentiment_data()
    
    # Train models
    all_results = {}
    tickers = df['Ticker'].unique()
    
    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] Processing {ticker}...")
        ticker_data = df[df['Ticker'] == ticker].copy()
        
        if len(ticker_data) < 200:
            print(f"  ⚠️  Skipping {ticker} - insufficient data ({len(ticker_data)} rows)")
            continue
        
        results = train_models_for_ticker(ticker_data, ticker, [1, 5, 21], 
                                         'daily', sentiment_df, save_dir)
        all_results[ticker] = results
    
    # Summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    
    for ticker, results in all_results.items():
        print(f"\n{ticker}:")
        for horizon, info in results.items():
            enhanced = " (MICROSTRUCTURE)" if info['enhanced'] else ""
            print(f"  {horizon}-day: {info['accuracy']*100:.2f}% - {info['feature_count']} features{enhanced}")
    
    return all_results

if __name__ == '__main__':
    results = train_all_models()
