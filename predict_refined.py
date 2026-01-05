"""
Generate predictions using refined models with trading metrics
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from train_refined_models import create_optimized_features, create_targets, load_sentiment_data

def predict_with_refined_model(ticker, stock_name, sector, df, horizons=[1, 5, 21], sentiment_df=None):
    """Generate predictions with trading metrics and sentiment"""
    
    model_file = f'models/{ticker}_daily_refined.joblib'
    
    if not os.path.exists(model_file):
        return None
    
    try:
        models = joblib.load(model_file)
        
        # Create features (with sentiment if available)
        df = create_optimized_features(df, 'daily', sentiment_df)
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Get latest data
        df_clean = df.dropna()
        if len(df_clean) < 10:
            return None
        
        latest_price = df_clean['Close'].iloc[-1]
        latest_date = df_clean.index[-1]
        
        predictions = {
            'Stock': stock_name,
            'Ticker': ticker,
            'Sector': sector,
            'Latest_Price': latest_price,
            'Latest_Date': latest_date.strftime('%Y-%m-%d')
        }
        
        # Predict for each horizon
        for horizon in horizons:
            if horizon not in models:
                continue
            
            model_data = models[horizon]
            model = model_data['model']
            scaler = model_data['scaler']
            feature_cols = model_data['feature_cols']
            
            # Get features for latest point
            X = df_clean[feature_cols].iloc[-1:].values
            X_scaled = scaler.transform(X)
            
            # Predict
            prob_up = model.predict(X_scaled)[0]
            
            # Standard threshold
            pred_direction = "UP ↑" if prob_up > 0.5 else "DOWN ↓"
            
            # Optimal threshold (from evaluation)
            optimal_thresholds = {
                'AAPL': 0.5,
                'GOOGL': 0.5,
                'BARC.L': 0.3,  # Optimized for better balance
            }
            
            optimal_threshold = optimal_thresholds.get(ticker, 0.5)
            pred_direction_opt = "UP ↑" if prob_up > optimal_threshold else "DOWN ↓"
            
            # Confidence
            confidence = abs(prob_up - 0.5) * 2
            
            # Add to predictions
            horizon_key = f'd{horizon}'
            predictions[f'{horizon_key}_Direction'] = pred_direction_opt
            predictions[f'{horizon_key}_Prob_Up'] = prob_up
            predictions[f'{horizon_key}_Confidence'] = confidence
            predictions[f'{horizon_key}_Threshold'] = optimal_threshold
            
            # Add model metrics from training
            if 'accuracy' in model_data:
                predictions[f'{horizon_key}_Accuracy'] = model_data['accuracy']
        
        return predictions
        
    except Exception as e:
        print(f"Error predicting {ticker}: {e}")
        return None

def main():
    print("="*80)
    print("GENERATING REFINED MODEL PREDICTIONS")
    print("="*80)
    
    # Load data
    df = pd.read_csv('data/multi_sector_stocks.csv', index_col=0, parse_dates=True)
    
    # Get unique stocks
    stocks_info = df[['Stock', 'Ticker', 'Sector']].drop_duplicates()
    
    # Load sentiment data
    sentiment_df = load_sentiment_data()
    if sentiment_df is not None:
        print(f"✓ Loaded sentiment data for {len(sentiment_df)} stocks\n")
    
    all_predictions = []
    
    for _, row in stocks_info.iterrows():
        ticker = row['Ticker']
        stock_name = row['Stock']
        sector = row['Sector']
        
        print(f"Processing {stock_name} ({ticker})...", end=' ')
        
        stock_df = df[df['Ticker'] == ticker].copy()
        
        predictions = predict_with_refined_model(ticker, stock_name, sector, stock_df, sentiment_df=sentiment_df)
        
        if predictions:
            all_predictions.append(predictions)
            print("✓")
        else:
            print("✗ (no refined model)")
    
    # Create DataFrame
    if all_predictions:
        predictions_df = pd.DataFrame(all_predictions)
        
        # Sort by sector and stock name
        predictions_df = predictions_df.sort_values(['Sector', 'Stock'])
        
        # Save
        output_file = 'predictions_refined.csv'
        predictions_df.to_csv(output_file, index=False)
        
        print(f"\n{'='*80}")
        print(f"✓ Saved {len(predictions_df)} stock predictions to {output_file}")
        print(f"{'='*80}")
        
        # Summary
        print(f"\nPredictions by horizon:")
        for horizon in [1, 5, 21]:
            col = f'd{horizon}_Direction'
            if col in predictions_df.columns:
                up_count = (predictions_df[col] == 'UP ↑').sum()
                down_count = (predictions_df[col] == 'DOWN ↓').sum()
                print(f"  {horizon}d: {up_count} UP, {down_count} DOWN")
        
        print(f"\nStocks by sector:")
        print(predictions_df.groupby('Sector').size().to_string())
        
        return predictions_df
    else:
        print("\n❌ No predictions generated!")
        return None

if __name__ == "__main__":
    main()
