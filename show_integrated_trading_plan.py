"""
Integrated Trading Decision Example
====================================
Shows how to combine stock tier filter + market regime filter for optimal trading decisions.
"""

import pandas as pd
import json
from datetime import datetime

print("=" * 80)
print("🎯 INTEGRATED TRADING DECISION SYSTEM")
print("=" * 80)

# Load filtered predictions
predictions = pd.read_csv('predictions_regime_filtered.csv')

# Load market regime
with open('current_market_regime.json', 'r') as f:
    regime = json.load(f)

# Load stock tiers
with open('stock_tiers.json', 'r') as f:
    tiers = json.load(f)

# Load earnings calendar
try:
    with open('earnings_calendar.json', 'r') as f:
        earnings = json.load(f)
    earnings_loaded = True
except FileNotFoundError:
    earnings = {}
    earnings_loaded = False

print(f"\n📊 Current Market: {regime['regime']} (Confidence: {regime['confidence']:.0f}%)")
print(f"📈 SPY Price: ${regime['spy_price']:.2f}")
print(f"😨 VIX: {regime['vix']:.2f}")
print(f"🎲 Should Trade: {'YES' if regime['should_trade'] else 'NO'}")
print(f"📊 Position Multiplier: {regime['position_multiplier']}x")

# Filter to tradeable signals only
tradeable = predictions[predictions['regime_filter'] == 'TRADE'].copy()

# Apply earnings filter
if earnings_loaded:
    earnings_filtered = []
    for idx, row in tradeable.iterrows():
        ticker = row['Ticker']
        if ticker in earnings and earnings[ticker] is not None:
            if earnings[ticker]['in_danger_zone']:
                continue  # Skip stocks near earnings
        earnings_filtered.append(idx)
    
    before_earnings_filter = len(tradeable)
    tradeable = tradeable.loc[earnings_filtered]
    filtered_by_earnings = before_earnings_filter - len(tradeable)
else:
    filtered_by_earnings = 0

print(f"\n🔢 Trading Signals:")
print(f"  Total Predictions: {len(predictions)}")
print(f"  Regime Filtered: {len(predictions) - len(tradeable) - filtered_by_earnings}")
if earnings_loaded:
    print(f"  Earnings Filtered: {filtered_by_earnings}")
print(f"  Tradeable: {len(tradeable)}")

if len(tradeable) == 0:
    print("\n⚠️  NO TRADEABLE SIGNALS - Stay in cash!")
    exit(0)

# Add tier information
def get_tier(ticker):
    if ticker in tiers['tier_1_elite']:
        return 1, 'Elite', 1.00
    elif ticker in tiers['tier_2_good']:
        return 2, 'Good', 0.65
    elif ticker in tiers['tier_3_mediocre']:
        return 3, 'Mediocre', 0.35
    else:
        return 4, 'Avoid', 0.00

tradeable[['tier', 'tier_name', 'tier_multiplier']] = tradeable['Ticker'].apply(
    lambda x: pd.Series(get_tier(x))
)

# Calculate final position size
tradeable['final_position_size'] = tradeable['tier_multiplier'] * regime['position_multiplier']

# Sort by 21-day confidence (higher = more certain prediction)
tradeable = tradeable.sort_values('d21_Confidence', ascending=False)

print("\n" + "=" * 80)
print("🚀 TOP TRADING OPPORTUNITIES")
print("=" * 80)

top_10 = tradeable.head(10)

print(f"\n{'Rank':<5} {'Ticker':<8} {'Tier':<12} {'21d Dir':<10} {'21d Conf':<10} {'Position':<10} {'Earnings':<15}")
print("-" * 95)

for idx, (rank, row) in enumerate(top_10.iterrows(), 1):
    ticker = row['Ticker']
    tier_str = f"T{row['tier']}-{row['tier_name']}"
    dir_21 = row['d21_Direction']
    conf_21 = row['d21_Confidence']
    pos_size = row['final_position_size']
    
    # Check earnings
    if earnings_loaded and ticker in earnings and earnings[ticker] is not None:
        days_until = earnings[ticker]['days_until']
        if days_until <= 14:
            earnings_str = f"📅 {days_until}d"
        else:
            earnings_str = "✓ Clear"
    else:
        earnings_str = "❓ Unknown"
    
    # Color code by tier
    if row['tier'] == 1:
        emoji = "⭐"
    elif row['tier'] == 2:
        emoji = "✅"
    else:
        emoji = "⚠️"
    
    print(f"{rank:<5} {emoji} {ticker:<6} {tier_str:<12} {dir_21:<10} {conf_21:<10.4f} {pos_size:<10.0%} {earnings_str:<15}")

# Summary statistics
print("\n" + "=" * 80)
print("📊 PORTFOLIO ALLOCATION RECOMMENDATION")
print("=" * 80)

tier_summary = tradeable.groupby('tier_name').agg({
    'Ticker': 'count',
    'final_position_size': 'first',
    'd21_Confidence': 'mean'
}).round(3)

print(f"\n{'Tier':<12} {'# Stocks':<10} {'Position Size':<15} {'Avg Confidence':<15}")
print("-" * 80)
for tier_name, row in tier_summary.iterrows():
    print(f"{tier_name:<12} {int(row['Ticker']):<10} {row['final_position_size']:<15.0%} {row['d21_Confidence']:<15.4f}")

# Trading action items
print("\n" + "=" * 80)
print("📝 ACTION ITEMS")
print("=" * 80)

if regime['regime'] == 'BULL':
    print("""
✅ BULL MARKET - Active Trading Mode

1. Review top 10 opportunities above
2. Prioritize Tier 2 stocks with 21-day predictions
3. Use full position sizes (65% for Tier 2)
4. Set stop-losses at -5%
5. Take profits at +8-12%
6. Monitor daily for regime changes

🎯 Focus Stocks Today:
""")
    for idx, row in top_10.head(5).iterrows():
        print(f"   • {row['Ticker']}: {row['tier_name']} tier, {row['final_position_size']:.0%} position, 21d {row['d21_Direction']} (conf={row['d21_Confidence']:.4f})")

elif regime['regime'] == 'SIDEWAYS':
    print("""
⚖️  SIDEWAYS MARKET - Selective Trading Mode

1. Only trade highest-confidence signals (top 5)
2. Use reduced position sizes (35-50%)
3. Take profits quickly at +3-5%
4. Set tighter stop-losses at -3%
5. Monitor for regime shift to BULL or BEAR

🎯 Focus Stocks Today:
""")
    for idx, row in top_10.head(3).iterrows():
        print(f"   • {row['Ticker']}: {row['tier_name']} tier, {row['final_position_size']:.0%} position, 21d {row['d21_Direction']} (conf={row['d21_Confidence']:.4f})")

else:  # BEAR
    print("""
❌ BEAR MARKET - Capital Preservation Mode

1. Stay 70%+ in cash
2. Only trade top 2-3 stocks if confidence > 15%
3. Use minimal position sizes (15-25%)
4. Consider inverse ETFs (SPXS, SQQQ)
5. Wait for regime to improve

⚠️  Very Limited Trading:
""")
    for idx, row in top_10.head(2).iterrows():
        print(f"   • {row['Ticker']}: {row['tier_name']} tier, {row['final_position_size']:.0%} position, 21d {row['d21_Direction']} (conf={row['d21_Confidence']:.4f})")

print("\n" + "=" * 80)
print("💡 TIP: Run this daily after generating new predictions!")
print("=" * 80)

# Save trading plan for today
today_plan = top_10[['Ticker', 'tier_name', 'd21_Direction', 'd21_Confidence', 'final_position_size']].copy()
today_plan.columns = ['Ticker', 'Tier', '21d_Direction', '21d_Confidence', 'Position_Size']
today_plan.to_csv('todays_trading_plan.csv', index=False)

print(f"\n✓ Saved today's trading plan → todays_trading_plan.csv")
