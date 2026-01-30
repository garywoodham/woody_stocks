#!/usr/bin/env python3
import pandas as pd

print("Creating complete dashboard data...")

df_recs = pd.read_csv('stock_recommendations.csv')
df_preds = pd.read_csv('predictions_refined.csv')

# Start with predictions (has all the prediction columns)
df_complete = df_preds.copy()

# Add Recommendation column from recs
df_complete = df_complete.merge(
    df_recs[['Ticker', 'Recommendation', 'Score']].rename(columns={'Score': 'Rec_Score'}),
    on='Ticker',
    how='left'
)

# Use Rec_Score as Score
if 'Rec_Score' in df_complete.columns:
    df_complete['Score'] = df_complete['Rec_Score']
    df_complete = df_complete.drop(['Rec_Score', 'Score_Signal'], axis=1, errors='ignore')

# Fill missing Recommendation
df_complete['Recommendation'] = df_complete['Recommendation'].fillna('HOLD')

# Reorder columns
cols = ['Stock', 'Ticker', 'Sector', 'Latest_Price', 'sentiment_label', 'sentiment_score',
        'Signal', 'Recommendation', 'Score', 'Strength', 'Consensus',
        'd1_Direction', 'd1_Prob_Up', 'd1_Accuracy',
        'd5_Direction', 'd5_Prob_Up', 'd5_Accuracy',
        'd21_Direction', 'd21_Prob_Up', 'd21_Accuracy']
existing = [c for c in cols if c in df_complete.columns]
others = [c for c in df_complete.columns if c not in existing]
df_complete = df_complete[existing + others]

# Save
df_complete.to_csv('stock_recommendations.csv', index=False)
print(f"✅ Saved complete data: {len(df_complete)} rows, {len(df_complete.columns)} columns")
print(f"Columns: {df_complete.columns.tolist()}")
