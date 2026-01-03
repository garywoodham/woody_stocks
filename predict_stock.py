import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import talib
import warnings
warnings.filterwarnings('ignore')

def create_features(df):
    """Create technical indicators and features for prediction"""
    df = df.copy()
    
    # Basic price features
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Moving averages
    for window in [5, 10, 20, 50]:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
        df[f'price_to_sma_{window}'] = df['Close'] / df[f'SMA_{window}']
    
    # Volatility
    for window in [5, 10, 20]:
        df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()
    
    # RSI
    for period in [7, 14, 21]:
        df[f'RSI_{period}'] = talib.RSI(df['Close'].values, timeperiod=period)
    
    # MACD
    macd, signal, hist = talib.MACD(df['Close'].values)
    df['MACD'] = macd
    df['MACD_signal'] = signal
    df['MACD_hist'] = hist
    
    # Bollinger Bands
    upper, middle, lower = talib.BBANDS(df['Close'].values)
    df['BB_upper'] = upper
    df['BB_middle'] = middle
    df['BB_lower'] = lower
    df['BB_width'] = (upper - lower) / middle
    df['BB_position'] = (df['Close'] - lower) / (upper - lower)
    
    # ATR (Average True Range)
    df['ATR_14'] = talib.ATR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
    
    # Stochastic
    slowk, slowd = talib.STOCH(df['High'].values, df['Low'].values, df['Close'].values)
    df['STOCH_K'] = slowk
    df['STOCH_D'] = slowd
    
    # ADX (Average Directional Index)
    df['ADX'] = talib.ADX(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
    
    # CCI (Commodity Channel Index)
    df['CCI'] = talib.CCI(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
    
    # Volume features
    df['volume_change'] = df['Volume'].pct_change()
    df['volume_ma_5'] = df['Volume'].rolling(window=5).mean()
    df['volume_ma_20'] = df['Volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_ma_20']
    
    # Price momentum
    for period in [1, 5, 10, 20]:
        df[f'momentum_{period}'] = df['Close'].pct_change(periods=period)
    
    # High-Low spread
    df['hl_spread'] = (df['High'] - df['Low']) / df['Close']
    
    # Ensure index is timezone-aware datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    
    # Day of week and month (cyclical encoding)
    df['day_of_week'] = df.index.dayofweek
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    df['day_of_month'] = df.index.day
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    
    return df

def create_targets(df, horizons=[1, 7, 30, 90]):
    """Create target variables for multiple time horizons"""
    targets = {}
    for h in horizons:
        # Future return
        df[f'target_{h}d_return'] = df['Close'].pct_change(periods=h).shift(-h)
        
        # Binary classification: will price go up?
        df[f'target_{h}d_direction'] = (df[f'target_{h}d_return'] > 0).astype(int)
        
        targets[h] = {
            'return': f'target_{h}d_return',
            'direction': f'target_{h}d_direction'
        }
    
    return df, targets

def train_models(df, horizons=[1, 7, 30, 90]):
    """Train LightGBM models for each prediction horizon"""
    
    print("Creating features...")
    df = create_features(df)
    df, targets = create_targets(df, horizons)
    
    # Remove infinite and NaN values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Get feature columns (exclude targets, basic columns, Stock, Ticker, and Sector)
    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 
                   'Stock', 'Ticker', 'Sector', 'day_of_week', 'day_of_month']
    exclude_cols += [col for col in df.columns if 'target' in col]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Drop rows with NaN (from indicators and target shifts)
    max_horizon = max(horizons)
    df_clean = df.dropna()
    
    print(f"\nTotal samples after feature engineering: {len(df_clean)}")
    print(f"Number of features: {len(feature_cols)}")
    
    models = {}
    predictions = {}
    
    for horizon in horizons:
        print(f"\n{'='*60}")
        print(f"Training model for {horizon}-day prediction")
        print(f"{'='*60}")
        
        # Prepare data
        X = df_clean[feature_cols].values
        y_direction = df_clean[targets[horizon]['direction']].values
        
        # Use time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        
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
        
        # Make predictions
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == y_test)
        print(f"\nTest Accuracy: {accuracy:.2%}")
        
        # Directional accuracy
        up_mask = y_test == 1
        down_mask = y_test == 0
        up_accuracy = np.mean(y_pred[up_mask] == y_test[up_mask]) if up_mask.sum() > 0 else 0
        down_accuracy = np.mean(y_pred[down_mask] == y_test[down_mask]) if down_mask.sum() > 0 else 0
        
        print(f"Accuracy predicting UP: {up_accuracy:.2%}")
        print(f"Accuracy predicting DOWN: {down_accuracy:.2%}")
        
        # Store model and scaler
        models[horizon] = {
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'accuracy': accuracy
        }
        
        # Make predictions on the most recent data
        latest_features = df_clean[feature_cols].iloc[-1:].values
        latest_scaled = scaler.transform(latest_features)
        latest_pred_proba = model.predict(latest_scaled)[0]
        latest_pred_direction = "UP ↑" if latest_pred_proba > 0.5 else "DOWN ↓"
        
        predictions[horizon] = {
            'probability_up': latest_pred_proba,
            'direction': latest_pred_direction,
            'confidence': abs(latest_pred_proba - 0.5) * 2  # Scale to 0-1
        }
        
        # Feature importance
        importance = model.feature_importance(importance_type='gain')
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 most important features:")
        print(feature_importance.head(10).to_string(index=False))
    
    return models, predictions, df_clean

def main():
    import joblib
    import time
    
    print("="*80)
    print("MULTI-STOCK PREDICTION MODEL TRAINING")
    print("="*80)
    
    # Load multi-sector data
    print("\nLoading multi-sector stock data...")
    df_all = pd.read_csv('data/multi_sector_stocks.csv', index_col=0, parse_dates=True)
    
    print(f"Total records: {len(df_all):,}")
    print(f"Date range: {df_all.index.min().strftime('%Y-%m-%d')} to {df_all.index.max().strftime('%Y-%m-%d')}")
    
    # Get unique stocks
    stocks = df_all.groupby(['Stock', 'Ticker', 'Sector']).size().reset_index()[['Stock', 'Ticker', 'Sector']]
    print(f"\nTotal stocks to train: {len(stocks)}")
    
    # Summary by sector
    print("\nStocks by sector:")
    for sector in stocks['Sector'].unique():
        sector_stocks = stocks[stocks['Sector'] == sector]
        print(f"  {sector}: {len(sector_stocks)} stocks")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Store all predictions
    all_predictions = []
    training_summary = []
    
    # Train models for each stock
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    for idx, row in stocks.iterrows():
        stock_name = row['Stock']
        ticker = row['Ticker']
        sector = row['Sector']
        
        print(f"\n{'='*80}")
        print(f"[{idx + 1}/{len(stocks)}] Training: {stock_name} ({ticker}) - {sector}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        # Filter data for this stock
        df_stock = df_all[df_all['Stock'] == stock_name].copy()
        print(f"Stock data: {len(df_stock)} records")
        
        try:
            # Train models for this stock
            models, predictions, df_processed = train_models(df_stock)
            
            # Save models
            safe_ticker = ticker.replace('.', '_')
            for horizon, model_data in models.items():
                model_path = f'models/{safe_ticker}_{horizon}d_model.pkl'
                joblib.dump(model_data, model_path)
            
            # Store predictions
            latest_price = df_stock['Close'].iloc[-1]
            latest_date = df_stock.index[-1]
            
            pred_summary = {
                'Stock': stock_name,
                'Ticker': ticker,
                'Sector': sector,
                'Latest_Date': latest_date.strftime('%Y-%m-%d'),
                'Latest_Price': latest_price,
            }
            
            for horizon in [1, 7, 30, 90]:
                pred = predictions[horizon]
                pred_summary[f'{horizon}d_Direction'] = pred['direction']
                pred_summary[f'{horizon}d_Prob_Up'] = pred['probability_up']
                pred_summary[f'{horizon}d_Confidence'] = pred['confidence']
                pred_summary[f'{horizon}d_Accuracy'] = models[horizon]['accuracy']
            
            all_predictions.append(pred_summary)
            
            elapsed = time.time() - start_time
            print(f"\n✓ Completed {stock_name} in {elapsed:.1f}s")
            print(f"  Latest Price: ${latest_price:.2f}")
            print(f"  Predictions: 1d={predictions[1]['direction']}, 7d={predictions[7]['direction']}, " + 
                  f"30d={predictions[30]['direction']}, 90d={predictions[90]['direction']}")
            
            training_summary.append({
                'Stock': stock_name,
                'Ticker': ticker,
                'Sector': sector,
                'Status': 'Success',
                'Time': elapsed
            })
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n✗ ERROR training {stock_name}: {str(e)}")
            training_summary.append({
                'Stock': stock_name,
                'Ticker': ticker,
                'Sector': sector,
                'Status': f'Failed: {str(e)}',
                'Time': elapsed
            })
            continue
    
    # Save predictions summary
    predictions_df = pd.DataFrame(all_predictions)
    predictions_df.to_csv('predictions_summary.csv', index=False)
    
    # Print final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*80)
    
    summary_df = pd.DataFrame(training_summary)
    successful = len(summary_df[summary_df['Status'] == 'Success'])
    failed = len(summary_df[summary_df['Status'] != 'Success'])
    total_time = summary_df['Time'].sum()
    
    print(f"\nTotal stocks processed: {len(summary_df)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total training time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    if failed > 0:
        print("\nFailed stocks:")
        failed_stocks = summary_df[summary_df['Status'] != 'Success']
        for _, row in failed_stocks.iterrows():
            print(f"  • {row['Stock']} ({row['Ticker']}): {row['Status']}")
    
    # Print predictions summary
    if len(predictions_df) > 0:
        print("\n" + "="*80)
        print("PREDICTIONS SUMMARY BY SECTOR")
        print("="*80)
        
        for sector in predictions_df['Sector'].unique():
            sector_preds = predictions_df[predictions_df['Sector'] == sector]
            print(f"\n{sector.upper()}:")
            print("-" * 80)
            
            for _, pred in sector_preds.iterrows():
                print(f"\n{pred['Stock']} ({pred['Ticker']}) - ${pred['Latest_Price']:.2f}")
                print(f"  1-Day:  {pred['1d_Direction']} ({pred['1d_Prob_Up']:.1%} prob, {pred['1d_Accuracy']:.1%} acc)")
                print(f"  7-Day:  {pred['7d_Direction']} ({pred['7d_Prob_Up']:.1%} prob, {pred['7d_Accuracy']:.1%} acc)")
                print(f"  30-Day: {pred['30d_Direction']} ({pred['30d_Prob_Up']:.1%} prob, {pred['30d_Accuracy']:.1%} acc)")
                print(f"  90-Day: {pred['90d_Direction']} ({pred['90d_Prob_Up']:.1%} prob, {pred['90d_Accuracy']:.1%} acc)")
        
        print(f"\n✓ Full predictions saved to: predictions_summary.csv")

if __name__ == "__main__":
    import os
    main()
