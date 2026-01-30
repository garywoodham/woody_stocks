#!/usr/bin/env python3
"""
Regenerate complete dashboard data with all merged columns.
"""
import pandas as pd
import numpy as np

print("="*80)
print("REGENERATING DASHBOARD DATA")
print("="*80)

# Load base files
print("\n📊 Loading data files...")
df_recs = pd.read_csv('stock_recommendations.csv')
df_preds = pd.read_csv('predictions_refined.csv')
df_signals = pd.read_csv('daily_signals.csv')
df_sentiment = pd.read_csv('sentiment_data.csv')

print(f"✓ Recommendations: {len(df_recs)} rows")
print(f"✓ Predictions: {len(df_preds)} rows")
print(f"✓ Signals: {len(df_signals)} rows")
print(f"✓ Sentiment: {len(df_sentiment)} rows")

# Aggregate signals by ticker
print("\n🔄 Aggregating signals...")

# Calculate average metrics first
df_signals_agg = df_signals.groupby('Ticker').agg({
    'Signal_Strength': 'mean',
    'Confidence': 'mean',
    'Probability_Up': 'mean'
}).reset_index()

# Get the signal with highest strength (most conviction) for each ticker
idx = df_signals.groupby('Ticker')['Signal_Strength'].idxmax()
df_signals_strongest = df_signals.loc[idx, ['Ticker', 'Signal']]
df_signals_agg = df_signals_agg.merge(df_signals_strongest, on='Ticker', how='left')

# If all signals for a ticker are HOLD (strength=0), keep HOLD
df_signals_agg['Signal'] = df_signals_agg['Signal'].fillna('HOLD')

df_signals_agg = df_signals_agg.rename(columns={
    'Signal_Strength': 'Strength',
    'Confidence': 'avg_confidence',
    'Probability_Up': 'avg_prob_up'
})

# Determine aggregated direction based on average probability
df_signals_agg['Predicted_Direction'] = df_signals_agg['avg_prob_up'].apply(
    lambda p: 'UP ↑' if p >= 0.5 else 'DOWN ↓'
)

# Calculate Score (positive for BUY, negative for SELL)
df_signals_agg['Score_Signal'] = df_signals_agg.apply(
    lambda row: row['Strength'] if row['Signal'] == 'BUY' 
               else -row['Strength'] if row['Signal'] == 'SELL' 
               else 0.0, 
    axis=1
)

# Consensus: combine direction and confidence
df_signals_agg['Consensus'] = df_signals_agg.apply(
    lambda row: f"{row['Predicted_Direction']} ({row['avg_confidence']:.1%})" 
               if pd.notna(row['avg_confidence']) and row['avg_confidence'] > 0
               else row['Predicted_Direction'] if pd.notna(row['Predicted_Direction'])
               else '-', 
    axis=1
)

print(f"✓ Aggregated {len(df_signals_agg)} tickers")

# Merge sentiment data
print("\n🔄 Merging sentiment data...")
# Only merge sentiment if columns don't already exist
if 'sentiment_score' not in df_recs.columns:
    df_recs = df_recs.merge(
        df_sentiment[['ticker', 'sentiment_score', 'sentiment_label']], 
        left_on='Ticker', 
        right_on='ticker', 
        how='left'
    ).drop('ticker', axis=1, errors='ignore')
    df_recs['sentiment_score'] = df_recs['sentiment_score'].fillna(0).round(4)
    df_recs['sentiment_label'] = df_recs['sentiment_label'].fillna('NEUTRAL')

if 'sentiment_score' not in df_preds.columns:
    df_preds = df_preds.merge(
        df_sentiment[['ticker', 'sentiment_score', 'sentiment_label']], 
        left_on='Ticker', 
        right_on='ticker', 
        how='left'
    ).drop('ticker', axis=1, errors='ignore')
    df_preds['sentiment_score'] = df_preds['sentiment_score'].fillna(0).round(4)
    df_preds['sentiment_label'] = df_preds['sentiment_label'].fillna('NEUTRAL')

print(f"✓ Merged sentiment")

# Merge signal data
print("\n🔄 Merging signal data...")
merge_cols = ['Ticker', 'Signal', 'Strength', 'Score_Signal', 'Consensus', 'Predicted_Direction', 'avg_confidence', 'avg_prob_up']

# Drop existing signal columns before merge to avoid conflicts
signal_cols_to_drop = ['Signal', 'Strength', 'Score_Signal', 'Consensus', 'Predicted_Direction', 'avg_confidence', 'avg_prob_up']
for col in signal_cols_to_drop:
    if col in df_recs.columns:
        df_recs = df_recs.drop(col, axis=1)
    if col in df_preds.columns:
        df_preds = df_preds.drop(col, axis=1)

df_recs = df_recs.merge(
    df_signals_agg[merge_cols], 
    on='Ticker', 
    how='left'
)

df_preds = df_preds.merge(
    df_signals_agg[merge_cols], 
    on='Ticker', 
    how='left'
)

# Fill missing signal values
for df in [df_recs, df_preds]:
    df['Signal'] = df['Signal'].fillna('HOLD')
    df['Strength'] = df['Strength'].fillna(0.0).round(4)
    df['Score_Signal'] = df['Score_Signal'].fillna(0.0).round(4)
    df['Consensus'] = df['Consensus'].fillna('-')
    df['Predicted_Direction'] = df['Predicted_Direction'].fillna('-')
    df['avg_confidence'] = df['avg_confidence'].fillna(0.0)
    df['avg_prob_up'] = df['avg_prob_up'].fillna(0.5)

print(f"✓ Merged signals")

# Format columns for display
print("\n🔄 Formatting columns...")
for df in [df_recs, df_preds]:
    if 'Strength' in df.columns:
        df['Strength'] = df['Strength'].apply(lambda x: f"{x:.4f}" if x != 0 else "0")

# Reorder columns for recommendations
recs_column_order = [
    'Stock', 'Ticker', 'Sector', 'Latest_Price',
    'sentiment_label', 'sentiment_score',
    'Signal', 'Recommendation', 'Score', 'Strength', 'Consensus'
]
existing_recs_cols = [col for col in recs_column_order if col in df_recs.columns]
other_recs_cols = [col for col in df_recs.columns if col not in existing_recs_cols]
df_recs = df_recs[existing_recs_cols + other_recs_cols]

# Reorder columns for predictions  
preds_column_order = [
    'Stock', 'Ticker', 'Sector', 'Latest_Price',
    'sentiment_label', 'sentiment_score',
    'Signal', 'Strength', 'Consensus',
    'd1_Direction', 'd1_Prob_Up', 'd1_Accuracy', 'd1_Confidence',
    'd5_Direction', 'd5_Prob_Up', 'd5_Accuracy', 'd5_Confidence',
    'd21_Direction', 'd21_Prob_Up', 'd21_Accuracy', 'd21_Confidence'
]
existing_preds_cols = [col for col in preds_column_order if col in df_preds.columns]
other_preds_cols = [col for col in df_preds.columns if col not in existing_preds_cols]
df_preds = df_preds[existing_preds_cols + other_preds_cols]

# Save updated files
print("\n💾 Saving updated files...")
df_recs.to_csv('stock_recommendations.csv', index=False)
df_preds.to_csv('predictions_refined.csv', index=False)

print(f"✓ Saved stock_recommendations.csv ({len(df_recs)} rows, {len(df_recs.columns)} columns)")
print(f"✓ Saved predictions_refined.csv ({len(df_preds)} rows, {len(df_preds.columns)} columns)")

print("\n" + "="*80)
print("✅ DATA REGENERATION COMPLETE")
print("="*80)
print("\n📊 Recommendations columns:")
print(df_recs.columns.tolist())
print("\n📊 Predictions columns:")
print(df_preds.columns.tolist())
print("\n💡 Restart dashboard to see changes:")
print("   kill $(cat dashboard.pid) && nohup python3 dashboard.py > dashboard_output.log 2>&1 & echo $! > dashboard.pid")
