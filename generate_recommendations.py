#!/usr/bin/env python3
"""
Generate BUY/HOLD/SELL recommendations based on combined multi-period predictions.
Combines 1-day, 5-day, and 21-day predictions with weighted scoring.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def calculate_recommendation_score(row):
    """
    Calculate a combined score from 1d, 5d, and 21d predictions.
    
    Score components:
    - Direction: UP = +1, DOWN = -1
    - Confidence: 0-1 scale showing prediction strength
    - Weights: 1d=20%, 5d=30%, 21d=50% (favor longer-term outlook)
    
    Returns score between -1 (strong sell) and +1 (strong buy)
    """
    # Extract probabilities and convert to directional scores (-1 to +1)
    d1_score = (row['d1_Prob_Up'] - 0.5) * 2  # Convert 0-1 to -1 to +1
    d5_score = (row['d5_Prob_Up'] - 0.5) * 2
    d21_score = (row['d21_Prob_Up'] - 0.5) * 2
    
    # Weight by confidence (higher confidence = more influence)
    d1_weighted = d1_score * (0.5 + row['d1_Confidence'])
    d5_weighted = d5_score * (0.5 + row['d5_Confidence'])
    d21_weighted = d21_score * (0.5 + row['d21_Confidence'])
    
    # Apply time-based weights (favor longer-term predictions)
    weights = {'d1': 0.2, 'd5': 0.3, 'd21': 0.5}
    
    combined_score = (
        d1_weighted * weights['d1'] + 
        d5_weighted * weights['d5'] + 
        d21_weighted * weights['d21']
    )
    
    return combined_score

def get_recommendation(score, strength_threshold=0.15, weak_threshold=0.05):
    """
    Convert score to BUY/HOLD/SELL recommendation.
    
    Args:
        score: Combined score (-1 to +1)
        strength_threshold: Threshold for strong buy/sell (default 0.15)
        weak_threshold: Threshold for hold zone (default 0.05)
    
    Returns:
        recommendation: BUY/HOLD/SELL
        strength: Strong/Moderate/Weak
        emoji: Visual indicator
    """
    if score >= strength_threshold:
        return 'BUY', 'Strong', '🟢'
    elif score >= weak_threshold:
        return 'BUY', 'Moderate', '🟡'
    elif score <= -strength_threshold:
        return 'SELL', 'Strong', '🔴'
    elif score <= -weak_threshold:
        return 'SELL', 'Moderate', '🟠'
    else:
        return 'HOLD', 'Neutral', '⚪'

def calculate_consensus(row):
    """Calculate how many horizons agree on direction."""
    directions = [row['d1_Direction'], row['d5_Direction'], row['d21_Direction']]
    up_count = sum(1 for d in directions if 'UP' in str(d))
    
    if up_count == 3:
        return '3/3 UP'
    elif up_count == 2:
        return '2/3 UP'
    elif up_count == 1:
        return '1/3 UP'
    else:
        return '0/3 UP'

def main():
    print("\n" + "="*80)
    print("GENERATING STOCK RECOMMENDATIONS")
    print("="*80 + "\n")
    
    # Load predictions
    df = pd.read_csv('predictions_refined.csv')
    print(f"✓ Loaded predictions for {len(df)} stocks\n")
    
    # Calculate scores and recommendations
    df['Score'] = df.apply(calculate_recommendation_score, axis=1)
    df['Recommendation'], df['Strength'], df['Signal'] = zip(*df['Score'].apply(get_recommendation))
    df['Consensus'] = df.apply(calculate_consensus, axis=1)
    
    # Calculate average confidence across horizons
    df['Avg_Confidence'] = (df['d1_Confidence'] + df['d5_Confidence'] + df['d21_Confidence']) / 3
    
    # Sort by score (highest first)
    df = df.sort_values('Score', ascending=False)
    
    # Create recommendations summary
    recommendations = df[[
        'Stock', 'Ticker', 'Sector', 'Latest_Price', 'Latest_Date',
        'Score', 'Recommendation', 'Strength', 'Signal', 'Consensus',
        'd1_Direction', 'd1_Prob_Up', 'd1_Confidence', 'd1_Accuracy',
        'd5_Direction', 'd5_Prob_Up','d5_Confidence', 'd5_Accuracy',
        'd21_Direction', 'd21_Prob_Up', 'd21_Confidence', 'd21_Accuracy',
        'Avg_Confidence'
    ]].copy()
    
    # Round numerical columns
    recommendations['Score'] = recommendations['Score'].round(4)
    recommendations['Latest_Price'] = recommendations['Latest_Price'].round(2)
    recommendations['d1_Prob_Up'] = recommendations['d1_Prob_Up'].round(4)
    recommendations['d5_Prob_Up'] = recommendations['d5_Prob_Up'].round(4)
    recommendations['d21_Prob_Up'] = recommendations['d21_Prob_Up'].round(4)
    recommendations['d1_Confidence'] = recommendations['d1_Confidence'].round(4)
    recommendations['d5_Confidence'] = recommendations['d5_Confidence'].round(4)
    recommendations['d21_Confidence'] = recommendations['d21_Confidence'].round(4)
    recommendations['d1_Accuracy'] = recommendations['d1_Accuracy'].round(4)
    recommendations['d5_Accuracy'] = recommendations['d5_Accuracy'].round(4)
    recommendations['d21_Accuracy'] = recommendations['d21_Accuracy'].round(4)
    recommendations['Avg_Confidence'] = recommendations['Avg_Confidence'].round(4)
    
    # Save to CSV
    recommendations.to_csv('stock_recommendations.csv', index=False)
    print(f"✓ Saved recommendations to stock_recommendations.csv\n")
    
    # Print summary
    print("="*80)
    print("RECOMMENDATION SUMMARY")
    print("="*80 + "\n")
    
    # Count by recommendation
    rec_counts = recommendations['Recommendation'].value_counts()
    print("Overall Recommendations:")
    for rec, count in rec_counts.items():
        emoji = recommendations[recommendations['Recommendation'] == rec]['Signal'].iloc[0]
        print(f"  {emoji} {rec}: {count} stocks")
    
    print("\nBy Strength:")
    strength_counts = recommendations.groupby(['Recommendation', 'Strength']).size()
    for (rec, strength), count in strength_counts.items():
        print(f"  {rec} ({strength}): {count} stocks")
    
    print("\n" + "="*80)
    print("TOP BUY RECOMMENDATIONS")
    print("="*80 + "\n")
    
    buy_stocks = recommendations[recommendations['Recommendation'] == 'BUY'].head(10)
    if len(buy_stocks) > 0:
        print(f"{'Stock':<25} {'Ticker':<10} {'Score':<8} {'Strength':<12} {'Consensus':<12} {'Price':<10}")
        print("-" * 80)
        for _, row in buy_stocks.iterrows():
            print(f"{row['Signal']} {row['Stock']:<22} {row['Ticker']:<10} {row['Score']:<8.4f} {row['Strength']:<12} {row['Consensus']:<12} ${row['Latest_Price']:<9.2f}")
    else:
        print("No BUY recommendations at this time.")
    
    print("\n" + "="*80)
    print("TOP SELL RECOMMENDATIONS")
    print("="*80 + "\n")
    
    sell_stocks = recommendations[recommendations['Recommendation'] == 'SELL'].tail(10).iloc[::-1]
    if len(sell_stocks) > 0:
        print(f"{'Stock':<25} {'Ticker':<10} {'Score':<8} {'Strength':<12} {'Consensus':<12} {'Price':<10}")
        print("-" * 80)
        for _, row in sell_stocks.iterrows():
            print(f"{row['Signal']} {row['Stock']:<22} {row['Ticker']:<10} {row['Score']:<8.4f} {row['Strength']:<12} {row['Consensus']:<12} ${row['Latest_Price']:<9.2f}")
    else:
        print("No SELL recommendations at this time.")
    
    print("\n" + "="*80)
    print("HOLD POSITIONS")
    print("="*80 + "\n")
    
    hold_stocks = recommendations[recommendations['Recommendation'] == 'HOLD']
    if len(hold_stocks) > 0:
        print(f"{'Stock':<25} {'Ticker':<10} {'Score':<8} {'Strength':<12} {'Consensus':<12} {'Price':<10}")
        print("-" * 80)
        for _, row in hold_stocks.iterrows():
            print(f"{row['Signal']} {row['Stock']:<22} {row['Ticker']:<10} {row['Score']:<8.4f} {row['Strength']:<12} {row['Consensus']:<12} ${row['Latest_Price']:<9.2f}")
    else:
        print("No HOLD recommendations at this time.")
    
    print("\n" + "="*80)
    print("SECTOR ANALYSIS")
    print("="*80 + "\n")
    
    sector_summary = recommendations.groupby(['Sector', 'Recommendation']).size().unstack(fill_value=0)
    print(sector_summary)
    
    print("\n" + "="*80)
    print("✓ COMPLETE!")
    print("="*80 + "\n")
    
    print("Scoring Methodology:")
    print("  • Combines 1d (20%), 5d (30%), 21d (50%) predictions")
    print("  • Weights by prediction confidence")
    print("  • Score range: -1.0 (strong sell) to +1.0 (strong buy)")
    print("  • BUY: Score > 0.05 | HOLD: -0.05 to 0.05 | SELL: Score < -0.05")
    print("  • Strong signals: |Score| > 0.15")
    print()

if __name__ == '__main__':
    main()
