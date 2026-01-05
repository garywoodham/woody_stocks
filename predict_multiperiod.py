"""
Generate predictions using multi-period models (daily, weekly, monthly)
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import feature creation from training script
import sys
sys.path.insert(0, '/workspaces/woody_stocks')
from train_model_multiperiod import create_features

def load_latest_data():
    """Load the latest data for all time periods"""
    data = {}
    
    files = {
        'daily': 'data/multi_sector_stocks.csv',
        'weekly': 'data/multi_sector_stocks_weekly.csv',
        'monthly': 'data/multi_sector_stocks_monthly.csv'
    }
    
    for period, filepath in files.items():
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            data[period] = df
            print(f"✓ Loaded {period} data: {len(df)} records")
        else:
            print(f"✗ Missing {period} data: {filepath}")
    
    return data

def make_predictions_for_stock(ticker, data, horizons_config):
    """Make predictions for a single stock across all time periods"""
    predictions = {
        'Ticker': ticker,
        'predictions': {}
    }
    
    for period in ['daily', 'weekly', 'monthly']:
        if period not in data:
            continue
        
        # Load model
        model_file = f"models/{ticker}_{period}_models.joblib"
        if not os.path.exists(model_file):
            continue
        
        try:
            models = joblib.load(model_file)
        except Exception as e:
            print(f"  ⚠️  Error loading {model_file}: {e}")
            continue
        
        # Get stock data
        stock_df = data[period][data[period]['Ticker'] == ticker].copy()
        
        if len(stock_df) < 50:
            continue
        
        # Create features
        df_features = create_features(stock_df, period)
        df_features = df_features.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(df_features) == 0:
            continue
        
        # Get metadata
        latest_row = stock_df.iloc[-1]
        predictions['Stock'] = latest_row['Stock']
        predictions['Sector'] = latest_row['Sector']
        predictions['Latest_Date'] = latest_row.name
        predictions['Latest_Price'] = latest_row['Close']
        
        # Make predictions for each horizon
        for horizon in horizons_config[period]:
            if horizon not in models:
                continue
            
            model_data = models[horizon]
            model = model_data['model']
            scaler = model_data['scaler']
            feature_cols = model_data['feature_cols']
            accuracy = model_data['accuracy']
            
            # Get latest features
            try:
                latest_features = df_features[feature_cols].iloc[-1:].values
                latest_scaled = scaler.transform(latest_features)
                prob_up = model.predict(latest_scaled)[0]
                
                direction = "UP ↑" if prob_up > 0.5 else "DOWN ↓"
                confidence = abs(prob_up - 0.5) * 2
                
                # Store prediction with period prefix
                key = f"{period[0]}{horizon}"  # e.g., "d1", "w4", "m3"
                predictions['predictions'][key] = {
                    'direction': direction,
                    'prob_up': prob_up,
                    'confidence': confidence,
                    'accuracy': accuracy,
                    'period': period,
                    'horizon': horizon
                }
            except Exception as e:
                print(f"  ⚠️  Error predicting {ticker} {period} {horizon}: {e}")
                continue
    
    return predictions

def generate_all_predictions():
    """Generate predictions for all stocks"""
    print("="*80)
    print("MULTI-PERIOD STOCK PREDICTIONS")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    data = load_latest_data()
    
    if not data:
        print("\n❌ No data available!")
        return
    
    # Configuration
    horizons_config = {
        'daily': [1, 5, 21],
        'weekly': [1, 4, 12],
        'monthly': [1, 3, 6]
    }
    
    # Get all tickers
    all_tickers = set()
    for df in data.values():
        all_tickers.update(df['Ticker'].unique())
    
    print(f"\nGenerating predictions for {len(all_tickers)} stocks...")
    print(f"Time periods: {', '.join(data.keys())}")
    print("="*80)
    
    # Generate predictions
    all_predictions = []
    
    for ticker in sorted(all_tickers):
        print(f"\n{ticker}:")
        preds = make_predictions_for_stock(ticker, data, horizons_config)
        
        if preds['predictions']:
            print(f"  ✓ Generated {len(preds['predictions'])} predictions")
            all_predictions.append(preds)
        else:
            print(f"  ✗ No predictions generated")
    
    # Create summary DataFrame
    print(f"\n{'='*80}")
    print("CREATING SUMMARY")
    print(f"{'='*80}")
    
    summary_data = []
    for pred in all_predictions:
        row = {
            'Stock': pred['Stock'],
            'Ticker': pred['Ticker'],
            'Sector': pred['Sector'],
            'Latest_Date': pred['Latest_Date'],
            'Latest_Price': pred['Latest_Price']
        }
        
        # Add all predictions
        for key, p in pred['predictions'].items():
            row[f'{key}_Direction'] = p['direction']
            row[f'{key}_Prob_Up'] = p['prob_up']
            row[f'{key}_Confidence'] = p['confidence']
            row[f'{key}_Accuracy'] = p['accuracy']
        
        summary_data.append(row)
    
    df_summary = pd.DataFrame(summary_data)
    
    # Sort columns
    base_cols = ['Stock', 'Ticker', 'Sector', 'Latest_Date', 'Latest_Price']
    pred_cols = sorted([c for c in df_summary.columns if c not in base_cols])
    df_summary = df_summary[base_cols + pred_cols]
    
    # Save
    output_file = 'predictions_multiperiod.csv'
    df_summary.to_csv(output_file, index=False)
    
    print(f"\n✓ Saved predictions to: {output_file}")
    print(f"  Stocks: {len(df_summary)}")
    print(f"  Columns: {len(df_summary.columns)}")
    
    # Display summary
    print(f"\n{'='*80}")
    print("PREDICTION SUMMARY")
    print(f"{'='*80}")
    
    for period in ['daily', 'weekly', 'monthly']:
        period_prefix = period[0]
        period_cols = [c for c in df_summary.columns if c.startswith(f'{period_prefix}') and c.endswith('_Direction')]
        
        if period_cols:
            print(f"\n{period.upper()}:")
            for col in sorted(period_cols):
                up_count = (df_summary[col] == "UP ↑").sum()
                down_count = (df_summary[col] == "DOWN ↓").sum()
                total = up_count + down_count
                if total > 0:
                    print(f"  {col.replace('_Direction', '')}: {up_count} UP ({up_count/total:.1%}), {down_count} DOWN ({down_count/total:.1%})")
    
    print(f"\n{'='*80}")
    print("✓ COMPLETE!")
    print(f"{'='*80}")
    
    return df_summary

if __name__ == "__main__":
    df = generate_all_predictions()
    
    if df is not None:
        print(f"\nTop 5 stocks with highest 1-day UP probability:")
        if 'd1_Prob_Up' in df.columns:
            top5 = df.nlargest(5, 'd1_Prob_Up')[['Stock', 'Ticker', 'd1_Direction', 'd1_Prob_Up', 'd1_Confidence']]
            print(top5.to_string(index=False))
