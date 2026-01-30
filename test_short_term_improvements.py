"""
Test Short-Term Accuracy Improvements
Systematically test each improvement and compare results
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
from datetime import datetime
warnings.filterwarnings('ignore')

def load_sentiment_data():
    """Load sentiment data"""
    try:
        sentiment_df = pd.read_csv('data/sentiment_history_complete.csv')
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        return sentiment_df, 'complete'
    except:
        try:
            sentiment_df = pd.read_csv('data/sentiment_history.csv')
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
            return sentiment_df, 'historical'
        except:
            try:
                sentiment_df = pd.read_csv('sentiment_data.csv')
                return sentiment_df, 'static'
            except:
                return None, None

def add_intraday_features(df):
    """Feature Set #1: Intraday/High-Frequency Features"""
    df = df.copy()
    
    # Gap Analysis
    df['gap'] = (df['Open'] - df['Close'].shift(1)) / (df['Close'].shift(1) + 1e-10)
    df['gap_up'] = (df['gap'] > 0.01).astype(int)
    df['gap_down'] = (df['gap'] < -0.01).astype(int)
    df['gap_size'] = abs(df['gap'])
    
    # Intraday Momentum
    df['intraday_range'] = (df['High'] - df['Low']) / (df['Open'] + 1e-10)
    df['intraday_momentum'] = (df['Close'] - df['Open']) / (df['Open'] + 1e-10)
    df['intraday_range_pct'] = (df['High'] - df['Low']) / df['Close']
    
    # Price Location within Day's Range
    df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['close_near_high'] = (df['close_position'] > 0.8).astype(int)
    df['close_near_low'] = (df['close_position'] < 0.2).astype(int)
    
    # Candlestick Body & Shadows
    df['body_size'] = abs(df['Close'] - df['Open']) / (df['Open'] + 1e-10)
    df['upper_shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / (df['Open'] + 1e-10)
    df['lower_shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / (df['Open'] + 1e-10)
    df['total_shadow'] = df['upper_shadow'] + df['lower_shadow']
    df['shadow_ratio'] = df['total_shadow'] / (df['body_size'] + 1e-10)
    
    # Candlestick Patterns
    df['doji'] = ((df['body_size'] < 0.001) & (df['total_shadow'] > 0.01)).astype(int)
    df['hammer'] = ((df['lower_shadow'] > 2 * df['body_size']) & (df['close_position'] > 0.6)).astype(int)
    df['shooting_star'] = ((df['upper_shadow'] > 2 * df['body_size']) & (df['close_position'] < 0.4)).astype(int)
    
    return df

def add_order_flow_features(df):
    """Feature Set #2: Order Flow Proxies"""
    df = df.copy()
    
    # Buying/Selling Pressure
    df['buying_pressure'] = ((df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)) * df['Volume']
    df['selling_pressure'] = ((df['High'] - df['Close']) / (df['High'] - df['Low'] + 1e-10)) * df['Volume']
    df['net_pressure'] = df['buying_pressure'] - df['selling_pressure']
    
    # Pressure Momentum & Trends
    df['pressure_momentum_3'] = df['net_pressure'].pct_change(3)
    df['pressure_momentum_5'] = df['net_pressure'].pct_change(5)
    df['pressure_trend'] = df['net_pressure'].rolling(5).mean()
    df['pressure_acceleration'] = df['pressure_momentum_5'].diff()
    
    # Volume-Weighted Returns
    volume_ma_20 = df['Volume'].rolling(20).mean()
    df['volume_weighted_return'] = df['Close'].pct_change() * (df['Volume'] / (volume_ma_20 + 1))
    df['volume_weighted_return_5'] = df['volume_weighted_return'].rolling(5).mean()
    
    # Relative Buying/Selling Strength
    df['buying_ratio'] = df['buying_pressure'] / (df['buying_pressure'] + df['selling_pressure'] + 1e-10)
    df['buying_strength'] = df['buying_ratio'].rolling(5).mean()
    df['buying_momentum'] = df['buying_ratio'].diff(5)
    
    return df

def add_microstructure_features(df):
    """Feature Set #3: Microstructure Features"""
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

def add_mean_reversion_features(df):
    """Feature Set #5: Short-Term Mean Reversion"""
    df = df.copy()
    
    # Ensure gap exists (it's created in intraday features)
    if 'gap' not in df.columns:
        df['gap'] = (df['Open'] - df['Close'].shift(1)) / (df['Close'].shift(1) + 1e-10)
    
    # Short-term RSI
    df['rsi_2'] = talib.RSI(df['Close'].values, timeperiod=2)
    df['rsi_5'] = talib.RSI(df['Close'].values, timeperiod=5)
    df['rsi_extreme'] = ((df['rsi_2'] < 10) | (df['rsi_2'] > 90)).astype(int)
    
    # Z-Scores (short-term)
    for window in [3, 5, 10]:
        mean = df['Close'].rolling(window).mean()
        std = df['Close'].rolling(window).std()
        df[f'zscore_{window}'] = (df['Close'] - mean) / (std + 1e-10)
        df[f'zscore_{window}_extreme'] = (abs(df[f'zscore_{window}']) > 2).astype(int)
    
    # Bollinger Band Mean Reversion (check if BB features exist)
    if 'BB_width' in df.columns:
        df['bb_squeeze'] = (df['BB_width'] < df['BB_width'].rolling(20).quantile(0.2)).astype(int)
    if 'BB_upper' in df.columns:
        df['bb_touch_upper'] = (df['Close'] > df['BB_upper'] * 0.99).astype(int)
    if 'BB_lower' in df.columns:
        df['bb_touch_lower'] = (df['Close'] < df['BB_lower'] * 1.01).astype(int)
    
    # Gap Fill Potential
    df['gap_fill_potential'] = abs(df['gap']) * (df['Volume'] / (df['Volume'].rolling(5).mean() + 1))
    
    # Volume Exhaustion
    df['volume_exhaustion'] = ((df['Volume'] > df['Volume'].rolling(20).mean() * 1.5) & 
                                (abs(df['Close'].pct_change()) > 0.03)).astype(int)
    
    # Price Distance from Moving Averages
    for window in [5, 10, 20]:
        ma = df['Close'].rolling(window).mean()
        df[f'distance_from_ma_{window}'] = (df['Close'] - ma) / (ma + 1e-10)
        df[f'far_from_ma_{window}'] = (abs(df[f'distance_from_ma_{window}']) > 0.05).astype(int)
    
    return df

def add_time_features(df):
    """Feature Set #7: Time-of-Week Effects"""
    df = df.copy()
    
    # Day of week
    df['day_of_week'] = df.index.dayofweek
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    df['is_tuesday'] = (df['day_of_week'] == 1).astype(int)
    df['is_wednesday'] = (df['day_of_week'] == 2).astype(int)
    df['is_thursday'] = (df['day_of_week'] == 3).astype(int)
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)
    
    # Week of month
    df['day_of_month'] = df.index.day
    df['week_of_month'] = ((df['day_of_month'] - 1) // 7 + 1).clip(1, 4)
    df['is_month_start'] = (df['day_of_month'] <= 5).astype(int)
    df['is_month_end'] = (df['day_of_month'] >= 25).astype(int)
    
    # Quarter effects
    df['quarter'] = df.index.quarter
    df['is_quarter_end'] = ((df.index.month % 3 == 0) & (df['day_of_month'] >= 28)).astype(int)
    
    return df

def add_enhanced_sentiment_features(df, sentiment_df):
    """Feature Set #8: Enhanced Sentiment with News Velocity"""
    if sentiment_df is None:
        return df
    
    df = df.copy()
    
    try:
        # Ensure date column is datetime
        if 'date' in sentiment_df.columns:
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
            sentiment_df = sentiment_df.set_index('date')
        
        # Make df index timezone-naive if it's timezone-aware
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        # Make sentiment_df index timezone-naive if it's timezone-aware
        if sentiment_df.index.tz is not None:
            sentiment_df.index = sentiment_df.index.tz_localize(None)
        
        # News velocity (article count per day)
        news_velocity = sentiment_df.groupby(sentiment_df.index.date).size().reset_index()
        news_velocity.columns = ['date', 'news_velocity']
        news_velocity['date'] = pd.to_datetime(news_velocity['date'])
        news_velocity = news_velocity.set_index('date')
        
        # Merge with main df
        df = df.join(news_velocity, how='left')
        df['news_velocity'] = df['news_velocity'].fillna(0)
        
        # News velocity features
        df['news_velocity_ma_5'] = df['news_velocity'].rolling(5).mean()
        df['news_surge'] = df['news_velocity'] / (df['news_velocity_ma_5'] + 1)
        df['news_surge_extreme'] = (df['news_surge'] > 2).astype(int)
        
        # Sentiment acceleration (already in base features, enhance it)
        if 'sentiment_compound' in df.columns:
            df['sentiment_velocity'] = df['sentiment_compound'].diff()
            df['sentiment_acceleration'] = df['sentiment_velocity'].diff()
            df['sentiment_extreme'] = ((df['sentiment_compound'] > 0.8) | 
                                        (df['sentiment_compound'] < -0.8)).astype(int)
            
            # Sentiment-Volume confirmation
            volume_ma_5 = df['Volume'].rolling(5).mean()
            df['sentiment_volume_confirm'] = df['sentiment_compound'] * (df['Volume'] / (volume_ma_5 + 1))
            df['sentiment_volume_divergence'] = (abs(df['sentiment_compound']) > 0.5) & (df['Volume'] < volume_ma_5)
            df['sentiment_volume_divergence'] = df['sentiment_volume_divergence'].astype(int)
    except Exception as e:
        print(f"  Warning: Could not add enhanced sentiment features: {e}")
    
    return df

def create_baseline_features(df, sentiment_df=None):
    """Create baseline features (current implementation)"""
    from train_refined_models import create_optimized_features
    return create_optimized_features(df, 'daily', sentiment_df)

def create_enhanced_features(df, sentiment_df=None, feature_sets=None):
    """Create features with specified improvements"""
    # Start with baseline
    df = create_baseline_features(df, sentiment_df)
    
    if feature_sets is None:
        feature_sets = ['all']
    
    # Add improvements
    if 'all' in feature_sets or 'intraday' in feature_sets:
        df = add_intraday_features(df)
    
    if 'all' in feature_sets or 'order_flow' in feature_sets:
        df = add_order_flow_features(df)
    
    if 'all' in feature_sets or 'microstructure' in feature_sets:
        df = add_microstructure_features(df)
    
    if 'all' in feature_sets or 'mean_reversion' in feature_sets:
        df = add_mean_reversion_features(df)
    
    if 'all' in feature_sets or 'time' in feature_sets:
        df = add_time_features(df)
    
    if 'all' in feature_sets or 'sentiment' in feature_sets:
        df = add_enhanced_sentiment_features(df, sentiment_df)
    
    return df

def create_targets(df, horizons=[1, 5, 21]):
    """Create target variables"""
    for horizon in horizons:
        future_return = df['Close'].shift(-horizon) / df['Close'] - 1
        df[f'target_{horizon}d'] = (future_return > 0).astype(int)
    return df

def train_and_evaluate_model(ticker, df, horizon, feature_type='baseline', sentiment_df=None):
    """Train a single model and evaluate"""
    
    # Create features based on type
    if feature_type == 'baseline':
        df_features = create_baseline_features(df.copy(), sentiment_df)
    else:
        df_features = create_enhanced_features(df.copy(), sentiment_df, feature_sets=[feature_type])
    
    # Create targets
    df_features = create_targets(df_features, [horizon])
    
    # Clean data
    df_features = df_features.replace([np.inf, -np.inf], np.nan)
    df_features = df_features.dropna()
    
    if len(df_features) < 100:
        return None
    
    # Get feature columns (exclude targets, metadata)
    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 
                    'Stock', 'Ticker', 'Sector'] + [f'target_{h}d' for h in [1, 5, 21]]
    feature_cols = [col for col in df_features.columns if col not in exclude_cols]
    
    # Prepare data
    X = df_features[feature_cols].values
    y = df_features[f'target_{horizon}d'].values
    
    # Time series split
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    accuracies = []
    
    for train_idx, test_idx in tscv.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Hyperparameters optimized for short-term
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
                'max_depth': 7,
                'verbose': -1
            }
        
        # Train
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        accuracies.append(accuracy)
    
    return np.mean(accuracies)

def test_all_improvements():
    """Test all improvements systematically"""
    
    print("="*80)
    print("TESTING SHORT-TERM ACCURACY IMPROVEMENTS")
    print("="*80)
    print()
    
    # Load data
    df = pd.read_csv('data/multi_sector_stocks.csv', index_col=0, parse_dates=True)
    sentiment_df, sentiment_type = load_sentiment_data()
    
    # Get unique stocks
    stocks = df[['Stock', 'Ticker', 'Sector']].drop_duplicates()
    
    # Test on a sample of diverse stocks
    test_stocks = [
        ('AAPL', 'Apple', 'Technology'),
        ('GOOGL', 'Alphabet', 'Technology'),
        ('TLRY', 'Tilray', 'Meme'),
        ('RTX', 'Raytheon', 'Defence'),
        ('PFE', 'Pfizer', 'Pharma'),
        ('STAN.L', 'Standard Chartered', 'Banking')
    ]
    
    horizons = [1, 5, 21]
    feature_sets = ['baseline', 'intraday', 'order_flow', 'microstructure', 
                    'mean_reversion', 'time', 'all']
    
    results = []
    
    print(f"Testing {len(test_stocks)} stocks across {len(horizons)} horizons and {len(feature_sets)} feature sets")
    print()
    
    for ticker, stock_name, sector in test_stocks:
        print(f"\n{'='*80}")
        print(f"Testing: {stock_name} ({ticker})")
        print(f"{'='*80}")
        
        # Get stock data
        stock_df = df[df['Ticker'] == ticker].copy()
        
        if len(stock_df) < 200:
            print(f"  ⚠️  Insufficient data ({len(stock_df)} rows)")
            continue
        
        for horizon in horizons:
            print(f"\n  Horizon: {horizon}-day")
            print(f"  {'-'*60}")
            
            for feature_set in feature_sets:
                try:
                    accuracy = train_and_evaluate_model(ticker, stock_df, horizon, 
                                                        feature_set, sentiment_df)
                    
                    if accuracy is not None:
                        print(f"    {feature_set:20s}: {accuracy*100:6.2f}%")
                        
                        results.append({
                            'Ticker': ticker,
                            'Stock': stock_name,
                            'Sector': sector,
                            'Horizon': horizon,
                            'Feature_Set': feature_set,
                            'Accuracy': accuracy * 100
                        })
                except Exception as e:
                    print(f"    {feature_set:20s}: ERROR - {e}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('short_term_improvement_results.csv', index=False)
    
    # Analyze results
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    # Average by feature set and horizon
    summary = results_df.groupby(['Horizon', 'Feature_Set'])['Accuracy'].mean().reset_index()
    
    for horizon in horizons:
        print(f"\n{horizon}-Day Predictions:")
        print("-"*60)
        horizon_data = summary[summary['Horizon'] == horizon].sort_values('Accuracy', ascending=False)
        
        baseline_acc = horizon_data[horizon_data['Feature_Set'] == 'baseline']['Accuracy'].values[0]
        
        for _, row in horizon_data.iterrows():
            feature_set = row['Feature_Set']
            accuracy = row['Accuracy']
            improvement = accuracy - baseline_acc
            
            if feature_set == 'baseline':
                print(f"  {feature_set:20s}: {accuracy:6.2f}% (BASELINE)")
            else:
                symbol = "✅" if improvement > 0 else "❌"
                print(f"  {feature_set:20s}: {accuracy:6.2f}% ({improvement:+5.2f}%) {symbol}")
    
    # Best overall
    print("\n" + "="*80)
    print("BEST IMPROVEMENTS PER HORIZON")
    print("="*80)
    
    for horizon in horizons:
        horizon_data = summary[summary['Horizon'] == horizon]
        baseline_acc = horizon_data[horizon_data['Feature_Set'] == 'baseline']['Accuracy'].values[0]
        best = horizon_data[horizon_data['Feature_Set'] != 'baseline'].nlargest(1, 'Accuracy').iloc[0]
        
        improvement = best['Accuracy'] - baseline_acc
        print(f"\n{horizon}-Day: {best['Feature_Set']} → {best['Accuracy']:.2f}% ({improvement:+.2f}%)")
    
    print("\n" + "="*80)
    print(f"Results saved to: short_term_improvement_results.csv")
    print("="*80)
    
    return results_df

if __name__ == '__main__':
    results = test_all_improvements()
