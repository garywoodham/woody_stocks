"""
Refined Model Training - Best Practices Only
Focus: Optimized LightGBM with feature selection
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

def load_sentiment_data():
    """Load sentiment data - historical if available, else static"""
    # Try historical first (time-series data)
    try:
        sentiment_df = pd.read_csv('data/sentiment_history.csv')
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        print(f"✓ Loaded HISTORICAL sentiment data")
        print(f"  Records: {len(sentiment_df)}, Dates: {sentiment_df['date'].nunique()}, Range: {sentiment_df['date'].min()} to {sentiment_df['date'].max()}")
        return sentiment_df, 'historical'
    except FileNotFoundError:
        pass
    
    # Fall back to static sentiment
    try:
        sentiment_df = pd.read_csv('sentiment_data.csv')
        print(f"⚠️  Using STATIC sentiment data (run fetch_sentiment_historical.py for time-series)")
        return sentiment_df, 'static'
    except FileNotFoundError:
        print("⚠️  No sentiment data found")
        return None, None

def create_optimized_features(df, time_period='daily', sentiment_df=None):
    """Carefully selected features - quality over quantity + sentiment"""
    df = df.copy()
    
    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    
    # Adjust parameters
    if time_period == 'daily':
        short_windows = [5, 10, 20, 50]
        vol_windows = [5, 10, 20]
        rsi_periods = [14]
        mom_periods = [1, 5, 10, 20]
    elif time_period == 'weekly':
        short_windows = [4, 8, 13, 26]
        vol_windows = [4, 8, 13]
        rsi_periods = [9]
        mom_periods = [1, 4, 8, 13]
    else:
        short_windows = [3, 6, 12, 24]
        vol_windows = [3, 6, 12]
        rsi_periods = [6]
        mom_periods = [1, 3, 6, 12]
    
    # === CORE PRICE FEATURES ===
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # === MOVING AVERAGES (KEY INDICATORS) ===
    for window in short_windows:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
        df[f'price_to_sma_{window}'] = df['Close'] / df[f'SMA_{window}']
    
    # SMA crossovers (important signals)
    df['sma_cross_5_20'] = (df['SMA_5'] > df['SMA_20']).astype(int)
    df['sma_cross_10_50'] = (df['SMA_10'] > df['SMA_50']).astype(int)
    
    # === VOLATILITY ===
    for window in vol_windows:
        df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()
    
    # === RSI (SINGLE BEST PERIOD) ===
    rsi_period = rsi_periods[0]
    df[f'RSI_{rsi_period}'] = talib.RSI(df['Close'].values, timeperiod=rsi_period)
    df['RSI_overbought'] = (df[f'RSI_{rsi_period}'] > 70).astype(int)
    df['RSI_oversold'] = (df[f'RSI_{rsi_period}'] < 30).astype(int)
    
    # === MACD ===
    if time_period == 'daily':
        macd, signal, hist = talib.MACD(df['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
    elif time_period == 'weekly':
        macd, signal, hist = talib.MACD(df['Close'].values, fastperiod=6, slowperiod=13, signalperiod=4)
    else:
        macd, signal, hist = talib.MACD(df['Close'].values, fastperiod=3, slowperiod=6, signalperiod=2)
    
    df['MACD'] = macd
    df['MACD_signal'] = signal
    df['MACD_hist'] = hist
    df['MACD_cross'] = (macd > signal).astype(int)
    
    # === BOLLINGER BANDS ===
    upper, middle, lower = talib.BBANDS(df['Close'].values, timeperiod=short_windows[1])
    df['BB_upper'] = upper
    df['BB_middle'] = middle
    df['BB_lower'] = lower
    df['BB_width'] = (upper - lower) / middle
    df['BB_position'] = (df['Close'] - lower) / (upper - lower + 1e-10)
    
    # === ATR ===
    atr_period = 14 if time_period == 'daily' else (7 if time_period == 'weekly' else 3)
    df['ATR'] = talib.ATR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=atr_period)
    df['ATR_pct'] = df['ATR'] / df['Close']
    
    # === STOCHASTIC ===
    slowk, slowd = talib.STOCH(df['High'].values, df['Low'].values, df['Close'].values,
                                fastk_period=short_windows[0], slowk_period=3, slowd_period=3)
    df['STOCH_K'] = slowk
    df['STOCH_D'] = slowd
    
    # === ADX ===
    df['ADX'] = talib.ADX(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=atr_period)
    df['ADX_strong_trend'] = (df['ADX'] > 25).astype(int)
    
    # === ADVANCED VOLUME INDICATORS ===
    # Basic volume features
    df['volume_ma_5'] = df['Volume'].rolling(window=vol_windows[0]).mean()
    df['volume_ma_10'] = df['Volume'].rolling(window=vol_windows[1]).mean()
    df['volume_ma_20'] = df['Volume'].rolling(window=vol_windows[2]).mean()
    df['volume_ratio'] = df['Volume'] / (df['volume_ma_20'] + 1)
    
    # Volume trend and momentum
    df['volume_trend'] = df['Volume'].pct_change(periods=5)
    df['volume_acceleration'] = df['volume_trend'].pct_change(periods=3)
    df['volume_spike'] = (df['Volume'] > df['volume_ma_20'] * 2).astype(int)
    
    # Volume relative to recent range
    df['volume_std_20'] = df['Volume'].rolling(window=20).std()
    df['volume_zscore'] = (df['Volume'] - df['volume_ma_20']) / (df['volume_std_20'] + 1)
    
    # OBV and variants
    df['OBV'] = talib.OBV(df['Close'].astype(float).values, df['Volume'].astype(float).values)
    df['OBV_ema'] = df['OBV'].ewm(span=short_windows[1]).mean()
    df['OBV_slope'] = df['OBV'].pct_change(periods=5)
    
    # Money Flow Index (volume-weighted RSI) - requires float64 for talib
    high_arr = df['High'].astype(np.float64).values
    low_arr = df['Low'].astype(np.float64).values
    close_arr = df['Close'].astype(np.float64).values
    volume_arr = df['Volume'].astype(np.float64).values
    
    df['MFI'] = talib.MFI(high_arr, low_arr, close_arr, volume_arr, timeperiod=14)
    
    # Accumulation/Distribution Line
    df['AD'] = talib.AD(high_arr, low_arr, close_arr, volume_arr)
    df['AD_slope'] = df['AD'].pct_change(periods=5)
    
    # Volume-Weighted Average Price (approximation)
    df['VWAP_approx'] = (df['Close'] * df['Volume']).rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()
    df['price_vs_vwap'] = (df['Close'] - df['VWAP_approx']) / df['VWAP_approx']
    
    # Chaikin Money Flow
    df['CMF'] = talib.ADOSC(high_arr, low_arr, close_arr, volume_arr, 
                            fastperiod=3, slowperiod=10)
    
    # Volume-Price Confirmation
    df['volume_price_trend'] = np.where(
        (df['Close'] > df['Close'].shift(1)) & (df['Volume'] > df['volume_ma_20']),
        1,  # Price up + High volume = bullish
        np.where(
            (df['Close'] < df['Close'].shift(1)) & (df['Volume'] > df['volume_ma_20']),
            -1,  # Price down + High volume = bearish
            0
        )
    )
    
    # === MOMENTUM ===
    for period in mom_periods:
        df[f'momentum_{period}'] = df['Close'].pct_change(periods=period)
    
    # === PRICE ACTION ===
    df['hl_spread'] = (df['High'] - df['Low']) / df['Close']
    df['oc_spread'] = (df['Close'] - df['Open']) / (df['Open'] + 1e-10)
    
    # === RANGE INDICATORS ===
    for window in [short_windows[0], short_windows[1]]:
        high_roll = df['High'].rolling(window=window).max()
        low_roll = df['Low'].rolling(window=window).min()
        df[f'position_in_range_{window}'] = (df['Close'] - low_roll) / (high_roll - low_roll + 1e-10)
    
    # === KEY INTERACTION FEATURES ===
    df['rsi_volume'] = df[f'RSI_{rsi_period}'] * df['volume_ratio']
    df['trend_momentum'] = df['ADX'] * df[f'momentum_{mom_periods[1]}']
    
    # Volume-based interactions
    df['volume_momentum'] = df['volume_ratio'] * df[f'momentum_{mom_periods[0]}']
    df['mfi_rsi_divergence'] = df['MFI'] - df[f'RSI_{rsi_period}']
    df['volume_volatility'] = df['volume_ratio'] * df['ATR_pct']
    df['obv_price_divergence'] = (df['OBV_slope'] - df['returns']) / (abs(df['returns']) + 0.01)
    
    # === TIME FEATURES ===
    if time_period == 'daily':
        df['day_of_week'] = df.index.dayofweek
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    df['month'] = df.index.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # === SENTIMENT FEATURES (IF AVAILABLE) ===
    if sentiment_df is not None:
        ticker = df['Ticker'].iloc[0] if 'Ticker' in df.columns else None
        
        if ticker:
            # Check if historical or static sentiment
            if 'date' in sentiment_df.columns:
                # HISTORICAL SENTIMENT - Merge by date
                sentiment_ticker = sentiment_df[sentiment_df['ticker'] == ticker].copy()
                
                if not sentiment_ticker.empty:
                    # Prepare for merge
                    df_temp = df.reset_index()
                    df_temp['date'] = pd.to_datetime(df_temp['Date']).dt.date if 'Date' in df_temp.columns else pd.to_datetime(df_temp.index).date
                    sentiment_ticker['date'] = pd.to_datetime(sentiment_ticker['date']).dt.date
                    
                    # Merge and forward-fill (sentiment stays same until new data)
                    df_temp = df_temp.merge(
                        sentiment_ticker[['date', 'sentiment_compound', 'sentiment_positive', 
                                         'sentiment_negative', 'news_count']],
                        on='date',
                        how='left'
                    )
                    
                    # Forward-fill missing values (weekends/holidays use last known sentiment)
                    df_temp['sentiment_compound'] = df_temp['sentiment_compound'].fillna(method='ffill').fillna(0.0)
                    df_temp['sentiment_positive'] = df_temp['sentiment_positive'].fillna(method='ffill').fillna(0.0)
                    df_temp['sentiment_negative'] = df_temp['sentiment_negative'].fillna(method='ffill').fillna(0.0)
                    df_temp['news_count'] = df_temp['news_count'].fillna(method='ffill').fillna(0)
                    
                    # Calculate sentiment momentum features
                    df_temp['sentiment_change_1d'] = df_temp['sentiment_compound'].diff(1)
                    df_temp['sentiment_ma_5'] = df_temp['sentiment_compound'].rolling(5).mean()
                    df_temp['sentiment_ma_20'] = df_temp['sentiment_compound'].rolling(20).mean()
                    df_temp['sentiment_trend'] = df_temp['sentiment_ma_5'] - df_temp['sentiment_ma_20']
                    
                    # Add to main df
                    df['sentiment_compound'] = df_temp.set_index(df.index)['sentiment_compound']
                    df['sentiment_positive'] = df_temp.set_index(df.index)['sentiment_positive']
                    df['sentiment_negative'] = df_temp.set_index(df.index)['sentiment_negative']
                    df['news_volume'] = df_temp.set_index(df.index)['news_count'] / 50.0  # Normalize
                    df['sentiment_change_1d'] = df_temp.set_index(df.index)['sentiment_change_1d'].fillna(0.0)
                    df['sentiment_trend'] = df_temp.set_index(df.index)['sentiment_trend'].fillna(0.0)
                    
                    # Sentiment interactions with technical indicators
                    df['sentiment_rsi'] = df['sentiment_compound'] * df[f'RSI_{rsi_period}']
                    df['sentiment_momentum'] = df['sentiment_compound'] * df['momentum_1']
                else:
                    # Ticker not found, use neutral
                    _add_neutral_sentiment(df, rsi_period)
            else:
                # STATIC SENTIMENT - Use same value for all dates
                sentiment_row = sentiment_df[sentiment_df['ticker'] == ticker]
                
                if not sentiment_row.empty:
                    df['sentiment_compound'] = sentiment_row['sentiment_score'].values[0]
                    df['sentiment_positive'] = sentiment_row.get('positive_ratio', [0.0]).values[0]
                    df['sentiment_negative'] = sentiment_row.get('negative_ratio', [0.0]).values[0]
                    df['news_volume'] = sentiment_row['news_count'].values[0] / 50.0
                    df['sentiment_change_1d'] = 0.0  # No history
                    df['sentiment_trend'] = 0.0  # No history
                    df['sentiment_rsi'] = df['sentiment_compound'] * df[f'RSI_{rsi_period}']
                    df['sentiment_momentum'] = df['sentiment_compound'] * df['momentum_1']
                else:
                    _add_neutral_sentiment(df, rsi_period)
        else:
            _add_neutral_sentiment(df, rsi_period)
    else:
        _add_neutral_sentiment(df, rsi_period)
    
    return df

def _add_neutral_sentiment(df, rsi_period):
    """Add neutral sentiment features when no data available"""
    df['sentiment_compound'] = 0.0
    df['sentiment_positive'] = 0.0
    df['sentiment_negative'] = 0.0
    df['news_volume'] = 0.0
    df['sentiment_change_1d'] = 0.0
    df['sentiment_trend'] = 0.0
    df['sentiment_rsi'] = 0.0
    df['sentiment_momentum'] = 0.0

def create_targets(df, horizons):
    """Create target variables"""
    targets = {}
    for h in horizons:
        df[f'target_{h}_return'] = df['Close'].pct_change(periods=h).shift(-h)
        df[f'target_{h}_direction'] = (df[f'target_{h}_return'] > 0).astype(int)
        targets[h] = {
            'return': f'target_{h}_return',
            'direction': f'target_{h}_direction'
        }
    return df, targets

def get_best_lgb_params(horizon):
    """Optimized LightGBM parameters"""
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'force_col_wise': True,
        'class_weight': 'balanced',  # Handle class imbalance
    }
    
    if horizon <= 5:
        params.update({
            'num_leaves': 20,
            'learning_rate': 0.08,
            'max_depth': 5,
            'min_data_in_leaf': 30,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'lambda_l1': 1.0,
            'lambda_l2': 1.0,
            'min_gain_to_split': 0.02,
            'scale_pos_weight': 1.0,
        })
    else:
        params.update({
            'num_leaves': 31,
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_data_in_leaf': 20,
            'feature_fraction': 0.75,
            'bagging_fraction': 0.75,
            'bagging_freq': 5,
            'lambda_l1': 0.5,
            'lambda_l2': 0.5,
            'min_gain_to_split': 0.01,
            'scale_pos_weight': 1.0,
        })
    
    return params

def train_refined_model(df, time_period, horizons, stock_name, sentiment_df=None):
    """Train refined LightGBM models with optional sentiment features"""
    
    print(f"\n{'='*80}")
    print(f"REFINED TRAINING: {time_period.upper()} - {stock_name}")
    print(f"{'='*80}")
    
    # Optimized feature engineering (with sentiment if available)
    df = create_optimized_features(df, time_period, sentiment_df)
    df, targets = create_targets(df, horizons)
    
    # Clean data
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Get feature columns
    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits',
                   'Stock', 'Ticker', 'Sector', 'day_of_week', 'month']
    exclude_cols += [col for col in df.columns if 'target' in col]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    df_clean = df.dropna()
    
    print(f"Samples: {len(df_clean)} | Features: {len(feature_cols)}")
    
    if len(df_clean) < 100:
        print(f"⚠️  Insufficient data")
        return None
    
    models = {}
    
    for horizon in horizons:
        print(f"\n{'-'*60}")
        print(f"Horizon: {horizon}-{time_period[0]}")
        
        X = df_clean[feature_cols].values
        y = df_clean[targets[horizon]['direction']].values
        
        # RobustScaler
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train/test split
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Check class balance
        train_pos = y_train.sum() / len(y_train)
        test_pos = y_test.sum() / len(y_test)
        print(f"Class balance - Train: {train_pos:.1%} UP | Test: {test_pos:.1%} UP")
        
        # Train LightGBM
        params = get_best_lgb_params(horizon)
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[test_data],
            callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(period=200)]
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
        
        print(f"Accuracy: {accuracy:.2%} | UP: {up_acc:.2%} | DOWN: {down_acc:.2%}")
        
        # Feature importance (top 10)
        importance = model.feature_importance(importance_type='gain')
        feature_imp = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"Top 5 features: {', '.join(feature_imp.head(5)['feature'].tolist())}")
        
        models[horizon] = {
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'accuracy': accuracy,
            'up_accuracy': up_acc,
            'down_accuracy': down_acc,
            'time_period': time_period,
            'feature_importance': feature_imp
        }
    
    return models

def main():
    print("="*80)
    print("REFINED MODEL TRAINING")
    print("="*80)
    print("\nOptimizations:")
    print("  ✓ LightGBM only (best individual performer)")
    print("  ✓ ~55 technical + 7 sentiment features (~62 total)")
    print("  ✓ Optimized hyperparameters per horizon")
    print("  ✓ Class balancing")
    print("  ✓ Better regularization")
    print("="*80)
    
    # Load data
    data_file = 'data/multi_sector_stocks.csv'
    if not os.path.exists(data_file):
        print(f"\n❌ {data_file} not found!")
        return
    
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    
    # Load sentiment data
    sentiment_df, sentiment_type = load_sentiment_data()
    if sentiment_df is not None:
        if sentiment_type == 'historical':
            print(f"✓ Using TIME-SERIES sentiment with momentum features")
        else:
            print(f"✓ Using STATIC sentiment (run fetch_sentiment_historical.py for better results)")
        print(f"  Stocks with sentiment: {sentiment_df['ticker'].nunique()}")
    
    # Get all unique stocks
    all_tickers = df['Ticker'].unique()
    print(f"\nTraining models for {len(all_tickers)} stocks")
    
    horizons = [1, 5, 21]
    
    results = []
    
    for ticker in all_tickers:
        stock_df = df[df['Ticker'] == ticker].copy()
        
        if len(stock_df) < 100:
            continue
        
        stock_name = stock_df['Stock'].iloc[0]
        models = train_refined_model(stock_df, 'daily', horizons, stock_name, sentiment_df)
        
        if models:
            model_file = f"models/{ticker}_daily_refined.joblib"
            joblib.dump(models, model_file)
            print(f"\n✓ Saved to {model_file}")
            
            for horizon, model_data in models.items():
                results.append({
                    'Ticker': ticker,
                    'Stock': stock_name,
                    'Horizon': f'{horizon}d',
                    'Accuracy': model_data['accuracy'],
                    'Features': len(model_data['feature_cols'])
                })
    
    # Compare
    print(f"\n{'='*80}")
    print("COMPARISON WITH BASELINE")
    print(f"{'='*80}")
    
    if results:
        results_df = pd.DataFrame(results)
        
        try:
            baseline = pd.read_csv('predictions_multiperiod.csv')
            
            print(f"\n{'Stock':<15} {'Horizon':<10} {'Baseline':<12} {'Refined':<12} {'Change':<10}")
            print("-"*70)
            
            for _, row in results_df.iterrows():
                ticker = row['Ticker']
                horizon = row['Horizon']
                refined_acc = row['Accuracy']
                
                baseline_row = baseline[baseline['Ticker'] == ticker]
                if not baseline_row.empty and f'{horizon}_Accuracy' in baseline_row.columns:
                    baseline_acc = baseline_row[f'{horizon}_Accuracy'].values[0]
                    change = refined_acc - baseline_acc
                    change_str = f"+{change:.1%}" if change > 0 else f"{change:.1%}"
                    
                    print(f"{row['Stock']:<15} {horizon:<10} {baseline_acc:<12.2%} {refined_acc:<12.2%} {change_str:<10}")
        
        except FileNotFoundError:
            print("\nRefined model results:")
            print(results_df.to_string(index=False))
        
        print(f"\nAverage accuracy: {results_df['Accuracy'].mean():.2%}")
        print(f"Features used: {results_df['Features'].iloc[0]}")
    
    print(f"\n{'='*80}")
    print("✓ COMPLETE!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
