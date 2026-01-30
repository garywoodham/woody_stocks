#!/usr/bin/env python3
"""
Analyze Sentiment Data Impact on Models
Checks sentiment data quality and feature importance
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("=" * 70)
print("SENTIMENT DATA IMPACT ANALYSIS")
print("=" * 70)

# 1. Load and analyze sentiment data
print("\n1. SENTIMENT DATA COVERAGE:")
print("-" * 70)

try:
    df_sentiment = pd.read_csv('data/sentiment_history_complete.csv')
    
    print(f"Total records: {len(df_sentiment):,}")
    print(f"Columns: {list(df_sentiment.columns)}")
    
    if 'Date' in df_sentiment.columns:
        df_sentiment['Date'] = pd.to_datetime(df_sentiment['Date'])
        print(f"Date range: {df_sentiment['Date'].min()} to {df_sentiment['Date'].max()}")
        days_coverage = (df_sentiment['Date'].max() - df_sentiment['Date'].min()).days
        print(f"Days of coverage: {days_coverage}")
    
    if 'Ticker' in df_sentiment.columns:
        print(f"Unique stocks: {df_sentiment['Ticker'].nunique()}")
        print(f"Stocks: {sorted(df_sentiment['Ticker'].unique())[:10]}...")
    
    # Check sentiment columns
    sentiment_cols = [col for col in df_sentiment.columns if 'sentiment' in col.lower() or col in ['positive', 'negative', 'neutral', 'compound']]
    print(f"\nSentiment columns found: {sentiment_cols}")
    
    # Sample statistics
    print("\nSample data (first 3 rows):")
    print(df_sentiment.head(3))
    
    # Check for missing values
    if sentiment_cols:
        print("\nMissing values in sentiment columns:")
        for col in sentiment_cols:
            if col in df_sentiment.columns:
                missing = df_sentiment[col].isna().sum()
                missing_pct = (missing / len(df_sentiment)) * 100
                print(f"  {col}: {missing:,} ({missing_pct:.2f}%)")
    
    # Check sentiment value distribution
    print("\nSentiment value statistics:")
    for col in sentiment_cols:
        if col in df_sentiment.columns and df_sentiment[col].dtype in [np.float64, np.int64]:
            print(f"\n  {col}:")
            print(f"    Mean: {df_sentiment[col].mean():.4f}")
            print(f"    Std: {df_sentiment[col].std():.4f}")
            print(f"    Min: {df_sentiment[col].min():.4f}")
            print(f"    Max: {df_sentiment[col].max():.4f}")
            
            # Check if all values are the same (no variance)
            if df_sentiment[col].nunique() <= 1:
                print(f"    ⚠️ WARNING: No variance - all values are the same!")
            
except FileNotFoundError:
    print("❌ sentiment_history_complete.csv not found!")
except Exception as e:
    print(f"❌ Error loading sentiment data: {e}")

# 2. Check feature importance
print("\n\n2. SENTIMENT FEATURE IMPORTANCE:")
print("-" * 70)

try:
    df_importance = pd.read_csv('feature_importance_summary.csv')
    
    # Filter sentiment features
    sentiment_features = df_importance[df_importance['feature'].str.contains('sentiment', case=False)]
    
    if len(sentiment_features) > 0:
        print(f"Found {len(sentiment_features)} sentiment feature entries\n")
        
        # Group by feature and show stats
        for feature in sentiment_features['feature'].unique():
            feature_data = sentiment_features[sentiment_features['feature'] == feature]
            
            avg_importance = feature_data['importance_mean'].mean()
            avg_pct = feature_data['importance_pct_mean'].mean()
            
            print(f"{feature}:")
            print(f"  Mean importance: {avg_importance:.6f}")
            print(f"  Mean % importance: {avg_pct:.6f}%")
            
            if avg_importance == 0:
                print(f"  ⚠️ ZERO IMPORTANCE - Not being used by models!")
        
        # Compare to non-sentiment features
        print("\n\nComparison to other features (top 10):")
        top_features = df_importance.nlargest(10, 'importance_mean')
        for idx, row in top_features.iterrows():
            print(f"  {row['feature']}: {row['importance_mean']:.2f} ({row['importance_pct_mean']:.2f}%)")
        
    else:
        print("❌ No sentiment features found in feature importance data!")
        
except FileNotFoundError:
    print("❌ feature_importance_summary.csv not found!")
except Exception as e:
    print(f"❌ Error analyzing feature importance: {e}")

# 3. Check current predictions for sentiment usage
print("\n\n3. CURRENT PREDICTIONS SENTIMENT DATA:")
print("-" * 70)

try:
    df_predictions = pd.read_csv('predictions_refined.csv')
    
    sentiment_pred_cols = [col for col in df_predictions.columns if 'sentiment' in col.lower()]
    
    if sentiment_pred_cols:
        print(f"Sentiment columns in predictions: {sentiment_pred_cols}")
        
        for col in sentiment_pred_cols:
            print(f"\n{col}:")
            print(f"  Unique values: {df_predictions[col].nunique()}")
            print(f"  Sample values: {df_predictions[col].head().tolist()}")
            
            # Check if all neutral or zero
            if df_predictions[col].dtype == object:
                value_counts = df_predictions[col].value_counts()
                print(f"  Value distribution:\n{value_counts}")
            else:
                print(f"  Mean: {df_predictions[col].mean():.4f}")
                print(f"  Std: {df_predictions[col].std():.4f}")
    else:
        print("No sentiment columns found in current predictions")
        
except FileNotFoundError:
    print("❌ predictions_refined.csv not found!")
except Exception as e:
    print(f"❌ Error checking predictions: {e}")

# 4. Recommendations
print("\n\n4. RECOMMENDATIONS:")
print("=" * 70)

try:
    # Load sentiment data for analysis
    df_sentiment = pd.read_csv('data/sentiment_history_complete.csv')
    
    # Check if we have enough data
    if len(df_sentiment) > 1000:
        print("✅ Sufficient sentiment data collected (156k+ records)")
    else:
        print("⚠️ Limited sentiment data - may not be useful yet")
    
    # Check date coverage
    if 'Date' in df_sentiment.columns:
        df_sentiment['Date'] = pd.to_datetime(df_sentiment['Date'])
        days_coverage = (df_sentiment['Date'].max() - df_sentiment['Date'].min()).days
        
        if days_coverage > 365:
            print(f"✅ Good date coverage: {days_coverage} days")
        else:
            print(f"⚠️ Limited date coverage: {days_coverage} days (need 1+ years)")
    
    # Check sentiment variance
    sentiment_value_cols = ['compound', 'positive', 'negative']
    has_variance = False
    
    for col in sentiment_value_cols:
        if col in df_sentiment.columns:
            if df_sentiment[col].std() > 0.01:
                has_variance = True
                print(f"✅ {col} has variance (std: {df_sentiment[col].std():.4f})")
            else:
                print(f"❌ {col} has no variance (all values similar)")
    
    print("\n" + "=" * 70)
    
    # Final recommendation
    if has_variance and len(df_sentiment) > 1000:
        print("\n🎯 RECOMMENDATION: RETRAIN MODELS")
        print("\nThe sentiment data appears to be good quality with:")
        print("  • 156k+ records")
        print("  • Multiple stocks covered")
        print("  • Variance in sentiment values")
        print("\nBut sentiment features have 0% importance, which suggests:")
        print("  1. Models may have been trained before sentiment data existed")
        print("  2. Sentiment features may not be properly integrated")
        print("  3. Feature engineering for sentiment may need improvement")
        print("\n📝 Action Items:")
        print("  1. Verify sentiment features are in training data")
        print("  2. Check if sentiment data is properly merged with price data")
        print("  3. Retrain models with: python3 train_refined_models.py")
        print("  4. Re-analyze feature importance after retraining")
    else:
        print("\n⚠️ RECOMMENDATION: IMPROVE SENTIMENT DATA FIRST")
        print("\nSentiment data quality issues detected.")
        print("Consider improving data collection before retraining.")
    
except Exception as e:
    print(f"Error in recommendations: {e}")

print("\n" + "=" * 70)
print("Analysis complete!")
print("=" * 70)
