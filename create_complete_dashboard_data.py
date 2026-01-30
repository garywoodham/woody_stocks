#!/usr/bin/env python3
"""
Create COMPLETE dashboard data by merging predictions with recommendations.
"""
import pandas as pd

print("="*80)
print("CREATING COMPLETE DASHBOARD DATA")
print("="*80)

# Load all files
df_recs = pd.read_csv('stock_recommendations.csv')
df_preds = pd.read_csv('predictions_refined.csv')

print(f"\n✓ Loaded recommendations: {len(df_recs)} rows, {len(df_recs.columns)} columns")
print(f"✓ Loaded predictions: {len(df_preds)} rows, {len(df_preds.columns)} columns")

# Predictions has everything except Recommendation column
# Let's merge Recommendation from recommendations into predictions
print("\n🔄 Merging Recommendation column into predictions...")

# Start with predictions (has all prediction data + sentiment + signals)
df_complete = df_preds.copy()

# Add Recommendation column from recommendations
df_complete = df_complete.merge(
    df_recs[['Ticker', 'Recommendation', 'Score']].rename(columns={'Score': 'Rec_Score'}),
    on='Ticker',
    how='left'
)

# Rename Score_Signal back to Score for consistency
if 'Score_Signal' in df_complete.columns and 'Rec_Score' in df_complete.columns:
    # Use Rec_Score as the main Score
    df_complete['Score'] = df_complete['Rec_Score']
    df_complete = df_complete.drop(['Score_Signal', 'Rec_Score'], axis=1)
elif 'Score_Signal' in df_complete.columns:
    df_complete = df_complete.rename(columns={'Score_Signal': 'Score'})

# Ensure Strength is properly formatted
if 'Strength' in df_complete.columns:
    df_complete['Strength'] = df_complete['Strength'].astype(str)

# Reorder columns for optimal display
column_order = [
    'Stock', 'Ticker', 'Sector', 'Latest_Price',
    'sentiment_label', 'sentiment_score',
    'Signal', 'Recommendation', 'Score', 'Strength', 'Consensus',
    'd1_Direction', 'd1_Prob_Up', 'd1_Accuracy',
    'd5_Direction', 'd5_Prob_Up', 'd5_Accuracy',
    'd21_Direction', 'd21_Prob_Up', 'd21_Accuracy'
]

# Keep only columns that exist
existing_cols = [col for col in column_order if col in df_complete.columns]
other_cols = [col for col in df_complete.columns if col not in existing_cols]
df_complete = df_complete[existing_cols + other_cols]

# Fill missing Recommendation values
if 'Recommendation' in df_complete.columns:
    df_complete['Recommendation'] = df_complete['Recommendation'].fillna('HOLD')

print(f"\n✓ Created complete dataset with {len(df_complete)} rows, {len(df_complete.columns)} columns")

# Save as new complete file
df_complete.to_csv('dashboard_complete.csv', index=False)
print(f"✓ Saved: dashboard_complete.csv")

# Also update recommendations to have the complete data
df_complete.to_csv('stock_recommendations.csv', index=False)
print(f"✓ Updated: stock_recommendations.csv")

print("\n" + "="*80)
print("COMPLETE DATA COLUMNS:")
print("="*80)
for i, col in enumerate(df_complete.columns, 1):
    print(f"{i:2}. {col}")

print("\n" + "="*80)
print("SAMPLE ROW (Apple):")
print("="*80)
apple = df_complete[df_complete['Ticker'] == 'AAPL'].iloc[0]
for col in ['Stock', 'sentiment_label', 'sentiment_score', 'Signal', 'Recommendation', 
            'Score', 'Strength', 'Consensus', 'd1_Direction', 'd1_Prob_Up', 'd5_Direction', 'd5_Prob_Up']:
    if col in apple.index:
        print(f"  {col:20s}: {apple[col]}")

print("\n" + "="*80)
print("✅ COMPLETE! Restart dashboard to see all data.")
print("="*80)
