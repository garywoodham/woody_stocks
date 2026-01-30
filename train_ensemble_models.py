"""
Ensemble Stacking Model Training
=================================
Combines LightGBM, XGBoost, and RandomForest for improved accuracy.
Uses 2-layer stacking: Base models → Meta-learner
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb
import xgboost as xgb
import talib

warnings.filterwarnings('ignore')

# Import feature engineering from existing script
import sys
sys.path.append('.')

print("=" * 80)
print("ENSEMBLE STACKING MODEL TRAINING")
print("=" * 80)
print("\nArchitecture:")
print("  Layer 1: LightGBM + XGBoost + RandomForest (base models)")
print("  Layer 2: Logistic Regression (meta-learner)")
print("  Expected: 2-5% accuracy improvement over single models")
print("=" * 80)

# Load data
print("\n📂 Loading data...")
df = pd.read_csv('data/multi_sector_stocks.csv')
df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

# Load sentiment data
try:
    sentiment_df = pd.read_csv('data/sentiment_history.csv')
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'], format='mixed', utc=True).dt.tz_localize(None)
    print(f"✓ Loaded HISTORICAL sentiment data")
    print(f"  Records: {len(sentiment_df)}, Dates: {sentiment_df['date'].nunique()}")
except FileNotFoundError:
    print("⚠️  No sentiment history - creating empty dataframe")
    sentiment_df = pd.DataFrame(columns=['date', 'ticker', 'sentiment_score', 'positive_ratio', 'negative_ratio'])

# Configuration
STOCKS = df['Ticker'].unique()
HORIZONS = {
    1: {'name': '1d', 'window': 1},
    5: {'name': '5d', 'window': 5},
    21: {'name': '21d', 'window': 21}
}

# Hyperparameters for each base model
LGBM_PARAMS = {
    1: {'num_leaves': 20, 'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 50},
    5: {'num_leaves': 25, 'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 75},
    21: {'num_leaves': 31, 'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 100}
}

XGB_PARAMS = {
    1: {'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 50, 'subsample': 0.8},
    5: {'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 75, 'subsample': 0.8},
    21: {'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 100, 'subsample': 0.8}
}

RF_PARAMS = {
    1: {'n_estimators': 50, 'max_depth': 8, 'min_samples_split': 20},
    5: {'n_estimators': 75, 'max_depth': 10, 'min_samples_split': 15},
    21: {'n_estimators': 100, 'max_depth': 12, 'min_samples_split': 10}
}


def create_features(df_stock, sentiment_df):
    """Create all 82 features (same as refined models)"""
    df = df_stock.copy()
    
    # Configuration
    short_windows = [5, 10, 20]
    long_windows = [50, 100, 200]
    atr_period = 14
    mom_periods = [5, 10, 20]
    vol_windows = [5, 10, 20]
    
    # === TECHNICAL INDICATORS ===
    
    # Returns
    df['returns'] = df['Close'].pct_change()
    
    # Moving Averages
    for window in short_windows + long_windows:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
    
    # Price to MA ratios
    for window in [5, 10, 20, 50]:
        df[f'price_to_sma_{window}'] = df['Close'] / df[f'SMA_{window}']
    
    # Bollinger Bands
    df['BB_middle'] = df['SMA_20']
    df['BB_std'] = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
    df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
    df['price_to_bb'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    # Volatility
    for window in [5, 10, 20]:
        df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(atr_period).mean()
    df['ATR_pct'] = df['ATR'] / df['Close']
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    # Stochastic
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['STOCH_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['STOCH_D'] = df['STOCH_K'].rolling(window=3).mean()
    
    # ADX
    df['ADX'] = talib.ADX(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=atr_period)
    df['ADX_strong_trend'] = (df['ADX'] > 25).astype(int)
    
    # === ADVANCED VOLUME INDICATORS ===
    df['volume_ma_5'] = df['Volume'].rolling(window=vol_windows[0]).mean()
    df['volume_ma_10'] = df['Volume'].rolling(window=vol_windows[1]).mean()
    df['volume_ma_20'] = df['Volume'].rolling(window=vol_windows[2]).mean()
    df['volume_ratio'] = df['Volume'] / (df['volume_ma_20'] + 1)
    
    df['volume_trend'] = df['Volume'].pct_change(periods=5)
    df['volume_acceleration'] = df['volume_trend'].pct_change(periods=3)
    df['volume_spike'] = (df['Volume'] > df['volume_ma_20'] * 2).astype(int)
    
    df['volume_std_20'] = df['Volume'].rolling(window=20).std()
    df['volume_zscore'] = (df['Volume'] - df['volume_ma_20']) / (df['volume_std_20'] + 1)
    
    # OBV and variants
    df['OBV'] = talib.OBV(df['Close'].astype(float).values, df['Volume'].astype(float).values)
    df['OBV_ema'] = df['OBV'].ewm(span=short_windows[1]).mean()
    df['OBV_slope'] = df['OBV'].pct_change(periods=5)
    
    # Money Flow Index, AD, CMF
    high_arr = df['High'].astype(np.float64).values
    low_arr = df['Low'].astype(np.float64).values
    close_arr = df['Close'].astype(np.float64).values
    volume_arr = df['Volume'].astype(np.float64).values
    
    df['MFI'] = talib.MFI(high_arr, low_arr, close_arr, volume_arr, timeperiod=14)
    df['AD'] = talib.AD(high_arr, low_arr, close_arr, volume_arr)
    df['AD_slope'] = df['AD'].pct_change(periods=5)
    df['VWAP_approx'] = (df['Close'] * df['Volume']).rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()
    df['price_vs_vwap'] = (df['Close'] - df['VWAP_approx']) / df['VWAP_approx']
    df['CMF'] = talib.ADOSC(high_arr, low_arr, close_arr, volume_arr, fastperiod=3, slowperiod=10)
    
    df['volume_price_trend'] = np.where(
        (df['Close'] > df['Close'].shift(1)) & (df['Volume'] > df['volume_ma_20']),
        1, np.where((df['Close'] < df['Close'].shift(1)) & (df['Volume'] > df['volume_ma_20']), -1, 0)
    )
    
    # === MOMENTUM ===
    for period in mom_periods:
        df[f'momentum_{period}'] = df['Close'].pct_change(periods=period)
    
    # === PRICE ACTION ===
    df['hl_spread'] = (df['High'] - df['Low']) / df['Close']
    df['oc_spread'] = (df['Close'] - df['Open']) / (df['Open'] + 1e-10)
    
    # === RANGE INDICATORS ===
    for window in [short_windows[0], short_windows[1]]:
        rolling_min = df['Low'].rolling(window=window).min()
        rolling_max = df['High'].rolling(window=window).max()
        df[f'position_in_range_{window}'] = (df['Close'] - rolling_min) / (rolling_max - rolling_min + 1e-10)
    
    # === PATTERN SIGNALS ===
    df['sma_cross_5_20'] = ((df['SMA_5'] > df['SMA_20']).astype(int) - 
                             (df['SMA_5'] < df['SMA_20']).astype(int))
    df['sma_cross_10_50'] = ((df['SMA_10'] > df['SMA_50']).astype(int) - 
                              (df['SMA_10'] < df['SMA_50']).astype(int))
    
    df['rsi_oversold'] = (df['RSI_14'] < 30).astype(int)
    df['rsi_overbought'] = (df['RSI_14'] > 70).astype(int)
    df['macd_cross'] = ((df['MACD'] > df['MACD_signal']).astype(int) - 
                        (df['MACD'] < df['MACD_signal']).astype(int))
    
    # === INTERACTION FEATURES ===
    df['rsi_volume'] = df['RSI_14'] * df['volume_ratio']
    df['trend_momentum'] = df['price_to_sma_20'] * df['momentum_10']
    df['volume_momentum'] = df['volume_ratio'] * df['momentum_5']
    df['mfi_rsi_divergence'] = df['MFI'] - df['RSI_14']
    df['volume_volatility'] = df['volume_ratio'] * df['ATR_pct']
    df['obv_price_divergence'] = (df['OBV_slope'] * df['momentum_5']).apply(lambda x: 1 if x < 0 else 0)
    
    # === TIME FEATURES ===
    df['day_of_week'] = pd.to_datetime(df['Date']).dt.dayofweek
    df['month'] = pd.to_datetime(df['Date']).dt.month
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        # === MARKET CONTEXT FEATURES ===
    try:
        # Load SPY (S&P 500) data
        spy_df = pd.read_csv('data/spy_data.csv', parse_dates=['Date'])
        spy_df['spy_returns'] = spy_df['Close'].pct_change()
        spy_df['spy_sma_20'] = spy_df['Close'].rolling(20).mean()
        spy_df['spy_trend'] = (spy_df['Close'] > spy_df['spy_sma_20']).astype(int)
        
        # Load VIX (volatility index) data
        vix_df = pd.read_csv('data/vix_data.csv', parse_dates=['Date'])
        vix_df = vix_df[['Date', 'Close']].rename(columns={'Close': 'vix_level'})
        vix_df['vix_regime'] = pd.cut(vix_df['vix_level'], bins=[0, 15, 25, 100], labels=[0, 1, 2])  # Low/Med/High vol
        
        # Merge market data
        df['Date_only'] = pd.to_datetime(df['Date']).dt.date
        spy_df['Date_only'] = spy_df['Date'].dt.date
        vix_df['Date_only'] = vix_df['Date'].dt.date
        
        df = df.merge(spy_df[['Date_only', 'spy_returns', 'spy_trend']], on='Date_only', how='left')
        df = df.merge(vix_df[['Date_only', 'vix_level', 'vix_regime']], on='Date_only', how='left')
        
        # Calculate correlation with SPY (20-day rolling)
        df['spy_correlation'] = df['returns'].rolling(20).corr(df['spy_returns'])
        
        # Relative performance vs market
        df['excess_return'] = df['returns'] - df['spy_returns']
        df['beta_proxy'] = df['spy_correlation'] * (df['volatility_20'] / df['spy_returns'].rolling(20).std())
        
        # Fill NaN values
        df['spy_returns'] = df['spy_returns'].fillna(0)
        df['spy_trend'] = df['spy_trend'].fillna(1)
        df['vix_level'] = df['vix_level'].fillna(df['vix_level'].mean())
        df['vix_regime'] = df['vix_regime'].fillna(1).astype(float)
        df['spy_correlation'] = df['spy_correlation'].fillna(0)
        df['excess_return'] = df['excess_return'].fillna(0)
        df['beta_proxy'] = df['beta_proxy'].fillna(1)
        
        df = df.drop('Date_only', axis=1, errors='ignore')
        
        print(f"  ✓ Added market context features (SPY correlation, VIX regime)")
    except Exception as e:
        print(f"  ⚠️ Could not load market data: {e}")
        # Add dummy features if market data unavailable
        df['spy_returns'] = 0
        df['spy_trend'] = 1
        df['vix_level'] = 20
        df['vix_regime'] = 1
        df['spy_correlation'] = 0
        df['excess_return'] = 0
        df['beta_proxy'] = 1
        # === SENTIMENT FEATURES ===
    if len(sentiment_df) > 0:
        df['Date_only'] = pd.to_datetime(df['Date']).dt.date
        sentiment_df['date_only'] = pd.to_datetime(sentiment_df['date']).dt.date
        
        # Merge sentiment data
        sentiment_merge = sentiment_df[['date_only', 'ticker', 'sentiment_score', 'positive_ratio', 'negative_ratio']].copy()
        
        df = df.merge(
            sentiment_merge,
            left_on=['Date_only', 'Ticker'],
            right_on=['date_only', 'ticker'],
            how='left'
        )
        df = df.drop(['Date_only', 'date_only'], axis=1, errors='ignore')
        if 'ticker' in df.columns:
            df = df.drop(['ticker'], axis=1)
        
        # Fill NaN with 0
        df['sentiment_score'] = df['sentiment_score'].fillna(0)
        df['positive_ratio'] = df['positive_ratio'].fillna(0)
        df['negative_ratio'] = df['negative_ratio'].fillna(0)
        
        # Sentiment derived features
        df['sentiment_ma_5'] = df.groupby('Ticker')['sentiment_score'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        df['sentiment_trend'] = df.groupby('Ticker')['sentiment_score'].transform(lambda x: x.diff(periods=3))
        df['sentiment_rsi'] = df.groupby('Ticker')['sentiment_score'].transform(
            lambda x: 100 - (100 / (1 + (x.diff().clip(lower=0).rolling(5).mean() / 
                                         (-x.diff().clip(upper=0).rolling(5).mean() + 1e-10))))
        )
    else:
        df['sentiment_score'] = 0
        df['positive_ratio'] = 0
        df['negative_ratio'] = 0
        df['sentiment_ma_5'] = 0
        df['sentiment_trend'] = 0
        df['sentiment_rsi'] = 0
    
    return df


def train_ensemble_for_stock(ticker, df_all, sentiment_df):
    """Train ensemble models for one stock"""
    print("\n" + "=" * 80)
    print(f"ENSEMBLE TRAINING: DAILY - {ticker}")
    print("=" * 80)
    
    df_stock = df_all[df_all['Ticker'] == ticker].copy()
    df_stock = create_features(df_stock, sentiment_df)
    
    # Remove NaN
    df_stock = df_stock.replace([np.inf, -np.inf], np.nan)
    initial_samples = len(df_stock)
    df_stock = df_stock.dropna()
    
    print(f"Samples: {len(df_stock)} | Features: {len(df_stock.columns) - 7}")
    
    if len(df_stock) < 100:
        print(f"⚠️  Insufficient data for {ticker}")
        return None
    
    # Feature columns
    exclude_cols = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
    feature_cols = [col for col in df_stock.columns if col not in exclude_cols]
    
    X = df_stock[feature_cols].values
    
    # Store models for all horizons
    ensemble_models = {}
    
    for horizon, config in HORIZONS.items():
        print(f"\n{'-' * 60}")
        print(f"Horizon: {config['name']}")
        
        # Create target
        df_stock['target'] = (df_stock['Close'].shift(-horizon) > df_stock['Close']).astype(int)
        y = df_stock['target'].values[:-horizon]
        X_train_full = X[:-horizon]
        
        # Train/test split (80/20)
        split_idx = int(len(X_train_full) * 0.8)
        X_train, X_test = X_train_full[:split_idx], X_train_full[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Class balance - Train: {y_train.mean()*100:.1f}% UP | Test: {y_test.mean()*100:.1f}% UP")
        
        # Calculate class weights for balancing
        pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1]) if y_train.sum() > 0 else 1.0
        print(f"Applying scale_pos_weight: {pos_weight:.2f}")
        
        # === LAYER 1: BASE MODELS ===
        
        # LightGBM
        lgbm_params = LGBM_PARAMS[horizon].copy()
        lgbm_params['scale_pos_weight'] = pos_weight
        lgbm_model = lgb.LGBMClassifier(
            **lgbm_params,
            random_state=42,
            verbose=-1
        )
        lgbm_model.fit(X_train, y_train)
        lgbm_pred_train = lgbm_model.predict_proba(X_train)[:, 1]
        lgbm_pred_test = lgbm_model.predict_proba(X_test)[:, 1]
        lgbm_acc = accuracy_score(y_test, (lgbm_pred_test > 0.5).astype(int))
        
        # XGBoost
        xgb_params = XGB_PARAMS[horizon].copy()
        xgb_params['scale_pos_weight'] = pos_weight
        xgb_model = xgb.XGBClassifier(
            **xgb_params,
            random_state=42,
            verbosity=0
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred_train = xgb_model.predict_proba(X_train)[:, 1]
        xgb_pred_test = xgb_model.predict_proba(X_test)[:, 1]
        xgb_acc = accuracy_score(y_test, (xgb_pred_test > 0.5).astype(int))
        
        # Random Forest
        rf_model = RandomForestClassifier(
            **RF_PARAMS[horizon],
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        rf_model.fit(X_train, y_train)
        rf_pred_train = rf_model.predict_proba(X_train)[:, 1]
        rf_pred_test = rf_model.predict_proba(X_test)[:, 1]
        rf_acc = accuracy_score(y_test, (rf_pred_test > 0.5).astype(int))
        
        print(f"Base Models - LightGBM: {lgbm_acc:.1%} | XGBoost: {xgb_acc:.1%} | RandomForest: {rf_acc:.1%}")
        
        # === LAYER 2: META-LEARNER (STACKING) ===
        
        # Stack predictions as new features
        stacked_train = np.column_stack([lgbm_pred_train, xgb_pred_train, rf_pred_train])
        stacked_test = np.column_stack([lgbm_pred_test, xgb_pred_test, rf_pred_test])
        
        # Logistic Regression as meta-learner
        meta_model = LogisticRegression(random_state=42, max_iter=1000)
        meta_model.fit(stacked_train, y_train)
        
        # Final predictions
        final_pred = meta_model.predict(stacked_test)
        final_proba = meta_model.predict_proba(stacked_test)[:, 1]
        ensemble_acc = accuracy_score(y_test, final_pred)
        
        # Directional metrics
        up_mask = y_test == 1
        down_mask = y_test == 0
        up_acc = accuracy_score(y_test[up_mask], final_pred[up_mask]) * 100 if up_mask.sum() > 0 else 0
        down_acc = accuracy_score(y_test[down_mask], final_pred[down_mask]) * 100 if down_mask.sum() > 0 else 0
        
        print(f"Ensemble Accuracy: {ensemble_acc:.2%} | UP: {up_acc:.2f}% | DOWN: {down_acc:.2f}%")
        
        # Meta-learner weights
        weights = meta_model.coef_[0]
        print(f"Meta Weights - LGBM: {weights[0]:.3f}, XGB: {weights[1]:.3f}, RF: {weights[2]:.3f}")
        
        # Store models
        ensemble_models[horizon] = {
            'lgbm': lgbm_model,
            'xgb': xgb_model,
            'rf': rf_model,
            'meta': meta_model,
            'feature_cols': feature_cols,
            'accuracy': ensemble_acc,
            'base_accuracies': {'lgbm': lgbm_acc, 'xgb': xgb_acc, 'rf': rf_acc}
        }
    
    return ensemble_models


# Main training loop
print(f"\nTraining ensemble models for {len(STOCKS)} stocks\n")

all_results = {}
accuracies = []

for ticker in STOCKS:
    try:
        models = train_ensemble_for_stock(ticker, df, sentiment_df)
        if models:
            # Save ensemble
            joblib.dump(models, f'models/{ticker}_daily_ensemble.joblib')
            print(f"\n✓ Saved to models/{ticker}_daily_ensemble.joblib")
            
            all_results[ticker] = models
            
            # Track accuracies
            for horizon in HORIZONS.keys():
                accuracies.append(models[horizon]['accuracy'])
        
    except Exception as e:
        print(f"❌ Error training {ticker}: {e}")
        continue

# Summary
print("\n" + "=" * 80)
print("ENSEMBLE TRAINING SUMMARY")
print("=" * 80)

if len(accuracies) > 0:
    print(f"\nAverage Ensemble Accuracy: {np.mean(accuracies):.2%}")
    print(f"Min: {np.min(accuracies):.2%} | Max: {np.max(accuracies):.2%}")
    print(f"\nModels saved: {len(all_results)} stocks × 3 horizons = {len(all_results)*3} ensembles")
else:
    print("\n⚠️  No models trained successfully")
    print("\n" + "=" * 80)
    print("=" * 80)
    sys.exit(1)

# Save summary
summary_df = []
for ticker, models in all_results.items():
    for horizon in HORIZONS.keys():
        summary_df.append({
            'ticker': ticker,
            'horizon': HORIZONS[horizon]['name'],
            'ensemble_accuracy': models[horizon]['accuracy'],
            'lgbm_accuracy': models[horizon]['base_accuracies']['lgbm'],
            'xgb_accuracy': models[horizon]['base_accuracies']['xgb'],
            'rf_accuracy': models[horizon]['base_accuracies']['rf']
        })

summary_df = pd.DataFrame(summary_df)
summary_df.to_csv('ensemble_training_summary.csv', index=False)
print("\n✓ Summary saved to ensemble_training_summary.csv")

print("\n" + "=" * 80)
print("✅ ENSEMBLE TRAINING COMPLETE!")
print("=" * 80)
