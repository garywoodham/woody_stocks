"""
Enhanced Stock Price Prediction Training with Multiple Time Periods

This script trains models on daily, weekly, and monthly data.
Each time period uses appropriate horizons for prediction:
- Daily data: predict 1, 5, 21 days ahead (1 day, 1 week, 1 month)
- Weekly data: predict 1, 4, 12 weeks ahead (1 week, 1 month, 3 months)
- Monthly data: predict 1, 3, 6 months ahead

This addresses the concern that we should match the granularity of data to the prediction horizon.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import talib
import warnings
import joblib
import os
warnings.filterwarnings('ignore')

def create_features(df, time_period='daily'):
    """
    Create technical indicators and features for prediction.
    
    Args:
        df: DataFrame with OHLCV data
        time_period: 'daily', 'weekly', or 'monthly' - adjusts indicator parameters
    
    Returns:
        DataFrame with added feature columns
    """
    df = df.copy()
    
    # Adjust parameters based on time period
    if time_period == 'daily':
        short_windows = [5, 10, 20, 50]
        vol_windows = [5, 10, 20]
        rsi_periods = [7, 14, 21]
        mom_periods = [1, 5, 10, 20]
    elif time_period == 'weekly':
        short_windows = [4, 8, 13, 26]  # ~1, 2, 3, 6 months
        vol_windows = [4, 8, 13]
        rsi_periods = [4, 9, 14]
        mom_periods = [1, 4, 8, 13]
    else:  # monthly
        short_windows = [3, 6, 12, 24]  # 3m, 6m, 1y, 2y
        vol_windows = [3, 6, 12]
        rsi_periods = [3, 6, 9]
        mom_periods = [1, 3, 6, 12]
    
    # Basic price features
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Moving averages
    for window in short_windows:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
        df[f'price_to_sma_{window}'] = df['Close'] / df[f'SMA_{window}']
    
    # Volatility
    for window in vol_windows:
        df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()
    
    # RSI
    for period in rsi_periods:
        df[f'RSI_{period}'] = talib.RSI(df['Close'].values, timeperiod=period)
    
    # MACD
    if time_period == 'daily':
        macd, signal, hist = talib.MACD(df['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
    elif time_period == 'weekly':
        macd, signal, hist = talib.MACD(df['Close'].values, fastperiod=6, slowperiod=13, signalperiod=4)
    else:  # monthly
        macd, signal, hist = talib.MACD(df['Close'].values, fastperiod=3, slowperiod=6, signalperiod=2)
    
    df['MACD'] = macd
    df['MACD_signal'] = signal
    df['MACD_hist'] = hist
    
    # Bollinger Bands
    upper, middle, lower = talib.BBANDS(df['Close'].values, timeperiod=short_windows[1])
    df['BB_upper'] = upper
    df['BB_middle'] = middle
    df['BB_lower'] = lower
    df['BB_width'] = (upper - lower) / middle
    df['BB_position'] = (df['Close'] - lower) / (upper - lower)
    
    # ATR (Average True Range)
    atr_period = 14 if time_period == 'daily' else (7 if time_period == 'weekly' else 3)
    df['ATR'] = talib.ATR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=atr_period)
    
    # Stochastic
    slowk, slowd = talib.STOCH(df['High'].values, df['Low'].values, df['Close'].values,
                                fastk_period=short_windows[0],
                                slowk_period=3,
                                slowd_period=3)
    df['STOCH_K'] = slowk
    df['STOCH_D'] = slowd
    
    # ADX (Average Directional Index)
    df['ADX'] = talib.ADX(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=atr_period)
    
    # CCI (Commodity Channel Index)
    df['CCI'] = talib.CCI(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=atr_period)
    
    # Volume features
    df['volume_change'] = df['Volume'].pct_change()
    df['volume_ma_short'] = df['Volume'].rolling(window=vol_windows[0]).mean()
    df['volume_ma_long'] = df['Volume'].rolling(window=vol_windows[2]).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_ma_long']
    
    # Price momentum
    for period in mom_periods:
        df[f'momentum_{period}'] = df['Close'].pct_change(periods=period)
    
    # High-Low spread
    df['hl_spread'] = (df['High'] - df['Low']) / df['Close']
    
    # Ensure index is timezone-aware datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    
    # Time features (cyclical encoding)
    if time_period == 'daily':
        df['day_of_week'] = df.index.dayofweek
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    df['period_of_year'] = df.index.dayofyear if time_period == 'daily' else df.index.month
    max_period = 365 if time_period == 'daily' else 12
    df['year_sin'] = np.sin(2 * np.pi * df['period_of_year'] / max_period)
    df['year_cos'] = np.cos(2 * np.pi * df['period_of_year'] / max_period)
    
    return df

def create_targets(df, horizons):
    """
    Create target variables for multiple time horizons.
    
    Args:
        df: DataFrame with features
        horizons: List of horizon periods (e.g., [1, 5, 21] for daily)
    
    Returns:
        Tuple of (df with targets, dict of target column names)
    """
    targets = {}
    for h in horizons:
        # Future return
        df[f'target_{h}_return'] = df['Close'].pct_change(periods=h).shift(-h)
        
        # Binary classification: will price go up?
        df[f'target_{h}_direction'] = (df[f'target_{h}_return'] > 0).astype(int)
        
        targets[h] = {
            'return': f'target_{h}_return',
            'direction': f'target_{h}_direction'
        }
    
    return df, targets

def train_model_for_period(df, time_period, horizons, stock_name):
    """
    Train models for a specific time period (daily/weekly/monthly).
    
    Args:
        df: DataFrame with stock data
        time_period: 'daily', 'weekly', or 'monthly'
        horizons: List of prediction horizons
        stock_name: Name of the stock
    
    Returns:
        Dictionary of trained models and metadata
    """
    print(f"\n{'='*80}")
    print(f"TRAINING {time_period.upper()} MODELS FOR {stock_name}")
    print(f"{'='*80}")
    print(f"Data points: {len(df)}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Horizons: {horizons}")
    
    # Create features
    print("\nCreating features...")
    df = create_features(df, time_period)
    df, targets = create_targets(df, horizons)
    
    # Remove infinite and NaN values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Get feature columns
    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits',
                   'Stock', 'Ticker', 'Sector', 'day_of_week', 'period_of_year']
    exclude_cols += [col for col in df.columns if 'target' in col]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Drop rows with NaN
    max_horizon = max(horizons)
    df_clean = df.dropna()
    
    print(f"Clean samples: {len(df_clean)}")
    print(f"Features: {len(feature_cols)}")
    
    if len(df_clean) < 100:
        print(f"\n⚠️  Warning: Only {len(df_clean)} samples. Minimum 100 recommended.")
        return None
    
    models = {}
    
    for horizon in horizons:
        print(f"\n{'-'*60}")
        print(f"Training {horizon}-{time_period[0]} horizon")
        print(f"{'-'*60}")
        
        # Prepare data
        X = df_clean[feature_cols].values
        y_direction = df_clean[targets[horizon]['direction']].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train/test split (last 20% for testing)
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y_direction[:split_idx], y_direction[split_idx:]
        
        # Train LightGBM model
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'min_data_in_leaf': 20,
            'max_depth': 7
        }
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[test_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
        )
        
        # Evaluate
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        accuracy = np.mean(y_pred == y_test)
        
        print(f"Test Accuracy: {accuracy:.2%}")
        
        # Store model
        models[horizon] = {
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'accuracy': accuracy,
            'time_period': time_period
        }
    
    return models

def main():
    import sys
    
    print("="*80)
    print("ENHANCED MULTI-TIMEPERIOD STOCK PREDICTION")
    print("="*80)
    print("\nThis training approach matches data granularity to prediction horizon:")
    print("  • Daily data → Short-term predictions (days)")
    print("  • Weekly data → Medium-term predictions (weeks)")
    print("  • Monthly data → Long-term predictions (months)")
    print("="*80)
    
    # Check which data files exist
    data_files = {
        'daily': 'data/multi_sector_stocks.csv',
        'weekly': 'data/multi_sector_stocks_weekly.csv',
        'monthly': 'data/multi_sector_stocks_monthly.csv'
    }
    
    available_periods = {}
    for period, filepath in data_files.items():
        if os.path.exists(filepath):
            available_periods[period] = filepath
            print(f"✓ Found {period} data: {filepath}")
        else:
            print(f"✗ Missing {period} data: {filepath}")
    
    if not available_periods:
        print("\n❌ No data files found!")
        print("Please run:")
        print("  1. python download_stock_data.py (to get daily data)")
        print("  2. python aggregate_data.py (to create weekly/monthly data)")
        return
    
    # Define horizons for each time period
    horizons_config = {
        'daily': [1, 5, 21],      # 1 day, 1 week, 1 month
        'weekly': [1, 4, 12],     # 1 week, 1 month, 3 months
        'monthly': [1, 3, 6]      # 1 month, 3 months, 6 months
    }
    
    # Create models directory
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Train models for each time period
    all_models = {}
    
    for period in ['daily', 'weekly', 'monthly']:
        if period not in available_periods:
            print(f"\n⚠️  Skipping {period} - data not available")
            continue
        
        print(f"\n{'='*80}")
        print(f"PROCESSING {period.upper()} DATA")
        print(f"{'='*80}")
        
        # Load data
        df = pd.read_csv(available_periods[period], index_col=0, parse_dates=True)
        print(f"Loaded {len(df):,} records from {df.index[0]} to {df.index[-1]}")
        
        # Get unique stocks
        stocks = df.groupby(['Stock', 'Ticker', 'Sector']).size().reset_index()[['Stock', 'Ticker', 'Sector']]
        print(f"Stocks: {len(stocks)}")
        
        # Train model for each stock
        period_models = {}
        for idx, row in stocks.iterrows():
            stock_name = row['Stock']
            ticker = row['Ticker']
            
            # Get stock data
            stock_df = df[df['Ticker'] == ticker].copy()
            
            if len(stock_df) < 50:
                print(f"\n⚠️  Skipping {stock_name} - insufficient data ({len(stock_df)} records)")
                continue
            
            # Train models
            models = train_model_for_period(
                stock_df,
                period,
                horizons_config[period],
                stock_name
            )
            
            if models:
                period_models[ticker] = models
                
                # Save models
                model_file = f"{models_dir}/{ticker}_{period}_models.joblib"
                joblib.dump(models, model_file)
                print(f"✓ Saved to {model_file}")
        
        all_models[period] = period_models
    
    # Summary
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    for period, models in all_models.items():
        print(f"\n{period.upper()}:")
        print(f"  Models trained: {len(models)}")
        print(f"  Horizons: {horizons_config[period]}")
    
    print(f"\n{'='*80}")
    print("✓ TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"\nModels saved to '{models_dir}/' directory")
    print("\nYou can now use these models to make predictions at different time scales!")

if __name__ == "__main__":
    main()
