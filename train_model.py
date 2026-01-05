import pandas as pd
import numpy as np
from feature_engineering import prepare_features, get_feature_columns
from models import StockPredictor
import warnings
warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("STOCK PRICE PREDICTION MODEL TRAINING")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv('data/barclays_daily_data.csv', index_col=0, parse_dates=True)
    print(f"Loaded {len(df)} days of data from {df.index[0]} to {df.index[-1]}")
    
    # Prepare features
    print("\nPreparing features and technical indicators...")
    df_features = prepare_features(df)
    print(f"After feature engineering: {len(df_features)} samples with {len(df_features.columns)} columns")
    
    # Get feature columns
    feature_cols = get_feature_columns(df_features)
    print(f"Using {len(feature_cols)} features for prediction")
    
    # Initialize predictor
    horizons = [1, 7, 30, 90]
    predictor = StockPredictor(horizons=horizons)
    
    # Train models
    print("\n" + "="*80)
    print("TRAINING MODELS")
    print("="*80)
    test_df = predictor.train(df_features, feature_cols)
    
    # Evaluate
    results = predictor.evaluate(test_df)
    
    # Save models
    predictor.save_models()
    
    # Feature importance (from LightGBM)
    print("\n" + "="*80)
    print("TOP 20 MOST IMPORTANT FEATURES (1-day prediction)")
    print("="*80)
    
    lgb_model = predictor.lgb_models[1]['regression']
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': lgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(20).to_string(index=False))
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print("\nModels saved to 'models/' directory")
    print("Use predict_stock.py to make predictions on new data")

if __name__ == "__main__":
    main()
