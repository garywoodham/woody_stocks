#!/usr/bin/env python3
"""
Stacked Ensemble Using Existing Refined Models
===============================================
Uses pre-trained refined models as base learners + meta-learner for final prediction.
Expected improvement: +1-2% over single refined models.
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
import sys

sys.path.insert(0, '.')
from train_refined_models import load_sentiment_data, create_optimized_features

warnings.filterwarnings('ignore')

print("=" * 80)
print("STACKED ENSEMBLE TRAINING")
print("=" * 80)
print("\nStrategy: Use existing refined models as base learners")
print("  Base Layer: Refined LightGBM models (already trained)")
print("  Meta Layer: Logistic Regression (combines base predictions)")
print("  Expected: +1-2% accuracy improvement")
print("=" * 80)

# Load data
print("\n📂 Loading data...")
df = pd.read_csv('data/multi_sector_stocks.csv')
df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

# Load auxiliary data
sentiment_df, _ = load_sentiment_data()

STOCKS = df['Ticker'].unique()
HORIZONS = [1, 5, 21]

print(f"\n✓ Data loaded: {len(STOCKS)} stocks")


def train_meta_learner(ticker, horizon, df, sentiment_df):
    """Train meta-learner using refined model predictions as features"""
    
    # Get stock data
    stock_df = df[df['Ticker'] == ticker].copy()
    stock_df = stock_df.sort_values('Date').reset_index(drop=True)
    
    # Load the refined model
    try:
        refined_model = joblib.load(f'models/{ticker}_daily_refined.joblib')
        base_model = refined_model[horizon]
    except FileNotFoundError:
        print(f"  ⚠️  No refined model found for {ticker}")
        return None
    
    # Create features (same as training - no regime/earnings in current version)
    features_df = create_optimized_features(stock_df, 'daily', sentiment_df)
    
    # Keep only numeric columns
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    features_df = features_df[numeric_cols]
    
    # Create target
    future_returns = stock_df['Close'].pct_change(horizon).shift(-horizon)
    y = (future_returns > 0).astype(int)
    
    # Reset indices to align properly
    features_df = features_df.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    # Drop rows with NaN target
    valid_mask = y.notna()
    features_df = features_df[valid_mask]
    y = y[valid_mask]
    
    # Drop inf/nan from features
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    features_df = features_df.fillna(features_df.median())
    
    if len(features_df) < 100:
        return None
    
    # Time series split
    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, test_idx = list(tscv.split(features_df))[-1]
    
    X_train = features_df.iloc[train_idx]
    X_test = features_df.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    
    # Get base model predictions (probabilities)
    feature_cols = base_model['feature_cols']
    X_train_base = X_train[feature_cols]
    X_test_base = X_test[feature_cols]
    
    # Scale
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_base)
    X_test_scaled = scaler.transform(X_test_base)
    
    # Base predictions
    base_pred_train = base_model['model'].predict_proba(X_train_scaled)[:, 1]
    base_pred_test = base_model['model'].predict_proba(X_test_scaled)[:, 1]
    base_acc = accuracy_score(y_test, (base_pred_test > 0.5).astype(int))
    
    # Create meta features (use predictions + additional context)
    # Meta features: base prediction, confidence (distance from 0.5), and trend features
    meta_train = np.column_stack([
        base_pred_train,
        np.abs(base_pred_train - 0.5),  # Confidence
        X_train['returns'].values,
        X_train['RSI_14'].values,
        X_train['volume_ratio'].values
    ])
    
    meta_test = np.column_stack([
        base_pred_test,
        np.abs(base_pred_test - 0.5),
        X_test['returns'].values,
        X_test['RSI_14'].values,
        X_test['volume_ratio'].values
    ])
    
    # Train meta-learner
    meta_model = LogisticRegression(random_state=42, max_iter=1000)
    meta_model.fit(meta_train, y_train)
    
    # Final predictions
    final_pred = meta_model.predict(meta_test)
    ensemble_acc = accuracy_score(y_test, final_pred)
    
    # Directional accuracy
    up_mask = y_test == 1
    down_mask = y_test == 0
    up_acc = accuracy_score(y_test[up_mask], final_pred[up_mask]) * 100 if up_mask.sum() > 0 else 0
    down_acc = accuracy_score(y_test[down_mask], final_pred[down_mask]) * 100 if down_mask.sum() > 0 else 0
    
    improvement = (ensemble_acc - base_acc) * 100
    
    return {
        'base_model': base_model,
        'meta_model': meta_model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'base_accuracy': base_acc,
        'ensemble_accuracy': ensemble_acc,
        'improvement': improvement,
        'up_accuracy': up_acc,
        'down_accuracy': down_acc
    }


# Train ensembles
print("\n" + "=" * 80)
print("TRAINING STACKED ENSEMBLES")
print("=" * 80)

results = []
successful = 0
failed = 0

for ticker in STOCKS:
    stock_name = df[df['Ticker'] == ticker]['Stock'].iloc[0]
    print(f"\n{'=' * 80}")
    print(f"STACKING ENSEMBLE: {stock_name}")
    print(f"{'=' * 80}")
    
    ensemble_models = {}
    
    for horizon in HORIZONS:
        print(f"\nHorizon: {horizon}-d")
        
        try:
            result = train_meta_learner(ticker, horizon, df, sentiment_df)
            
            if result:
                ensemble_models[horizon] = result
                
                print(f"  Base Model:  {result['base_accuracy']:.2%}")
                print(f"  Ensemble:    {result['ensemble_accuracy']:.2%}  "
                      f"({result['improvement']:+.2f}%)")
                print(f"  UP: {result['up_accuracy']:.2f}% | DOWN: {result['down_accuracy']:.2f}%")
                
                results.append({
                    'Ticker': ticker,
                    'Stock': stock_name,
                    'Horizon': f'{horizon}-d',
                    'Base_Accuracy': result['base_accuracy'] * 100,
                    'Ensemble_Accuracy': result['ensemble_accuracy'] * 100,
                    'Improvement': result['improvement']
                })
                
                successful += 1
            else:
                print(f"  ⚠️  Skipped (insufficient data)")
                failed += 1
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            failed += 1
            continue
    
    # Save ensemble models for this stock
    if ensemble_models:
        joblib.dump(ensemble_models, f'models/{ticker}_stacked_ensemble.joblib')
        print(f"\n✓ Saved to models/{ticker}_stacked_ensemble.joblib")

# Summary
print("\n" + "=" * 80)
print("STACKED ENSEMBLE RESULTS")
print("=" * 80)

if results:
    results_df = pd.DataFrame(results)
    
    # Overall statistics
    print(f"\n✅ Successfully trained: {successful} models")
    print(f"❌ Failed: {failed} models")
    
    print(f"\nOverall Performance:")
    print(f"  Average Base Accuracy:     {results_df['Base_Accuracy'].mean():.2f}%")
    print(f"  Average Ensemble Accuracy: {results_df['Ensemble_Accuracy'].mean():.2f}%")
    print(f"  Average Improvement:       {results_df['Improvement'].mean():+.2f}%")
    
    # By horizon
    print(f"\nBy Horizon:")
    for horizon in ['1-d', '5-d', '21-d']:
        horizon_data = results_df[results_df['Horizon'] == horizon]
        if len(horizon_data) > 0:
            print(f"  {horizon:<6} Base: {horizon_data['Base_Accuracy'].mean():.2f}%  →  "
                  f"Ensemble: {horizon_data['Ensemble_Accuracy'].mean():.2f}%  "
                  f"({horizon_data['Improvement'].mean():+.2f}%)")
    
    # Best improvements
    print(f"\nTop 5 Improvements:")
    top_5 = results_df.nlargest(5, 'Improvement')
    for _, row in top_5.iterrows():
        print(f"  {row['Stock']:<25} {row['Horizon']:<6} "
              f"{row['Base_Accuracy']:.2f}% → {row['Ensemble_Accuracy']:.2f}% "
              f"({row['Improvement']:+.2f}%)")
    
    # Save results
    results_df.to_csv('stacked_ensemble_results.csv', index=False)
    print(f"\n✓ Results saved to stacked_ensemble_results.csv")
    
else:
    print("\n⚠️  No models trained successfully")

print("\n" + "=" * 80)
print("✅ STACKED ENSEMBLE TRAINING COMPLETE!")
print("=" * 80)
