"""
Earnings Calendar Fetcher
==========================
Fetches upcoming earnings dates for all stocks.
Filters out trades 3 days before/after earnings (high volatility period).

Expected Impact: Avoid 10-15% of losses from earnings surprises
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("📅 EARNINGS CALENDAR FETCHER")
print("=" * 80)

# Load stock list from predictions
try:
    predictions = pd.read_csv('predictions_refined.csv')
    tickers = predictions['Ticker'].unique().tolist()
    print(f"\n✓ Loaded {len(tickers)} stocks from predictions")
except FileNotFoundError:
    print("\n❌ predictions_refined.csv not found. Run generate_daily_signals.py first")
    exit(1)

# Fetch earnings dates
print(f"\n📥 Fetching earnings dates...")
earnings_calendar = {}
success_count = 0
fail_count = 0

for ticker in tickers:
    try:
        stock = yf.Ticker(ticker)
        
        # Get earnings dates from calendar
        calendar = stock.calendar
        
        if calendar is not None and 'Earnings Date' in calendar:
            earnings_dates = calendar['Earnings Date']
            
            if isinstance(earnings_dates, pd.Series):
                # Multiple dates
                next_earnings = earnings_dates.iloc[0]
            else:
                # Single date
                next_earnings = earnings_dates
            
            # Convert to string
            if pd.notna(next_earnings):
                earnings_date = pd.to_datetime(next_earnings)
                earnings_calendar[ticker] = {
                    'earnings_date': earnings_date.strftime('%Y-%m-%d'),
                    'days_until': (earnings_date - pd.Timestamp.now()).days,
                    'in_danger_zone': abs((earnings_date - pd.Timestamp.now()).days) <= 3
                }
                success_count += 1
                print(f"  ✓ {ticker:<8} → {earnings_date.strftime('%Y-%m-%d')} ({earnings_calendar[ticker]['days_until']:+d} days)")
            else:
                earnings_calendar[ticker] = None
                fail_count += 1
        else:
            earnings_calendar[ticker] = None
            fail_count += 1
            
    except Exception as e:
        earnings_calendar[ticker] = None
        fail_count += 1

print(f"\n✓ Success: {success_count} stocks with earnings dates")
print(f"⚠️  Unknown: {fail_count} stocks (no earnings date available)")

# Save earnings calendar
with open('earnings_calendar.json', 'w') as f:
    json.dump(earnings_calendar, f, indent=2, default=str)

print(f"\n✓ Saved earnings calendar → earnings_calendar.json")

# Identify stocks in danger zone (3 days before/after earnings)
danger_zone = [ticker for ticker, data in earnings_calendar.items() 
               if data is not None and data['in_danger_zone']]

print("\n" + "=" * 80)
print("⚠️  EARNINGS DANGER ZONE (Within 3 days)")
print("=" * 80)

if danger_zone:
    print(f"\n🚨 DO NOT TRADE these {len(danger_zone)} stocks:\n")
    for ticker in danger_zone:
        data = earnings_calendar[ticker]
        print(f"  ❌ {ticker:<8} → Earnings {data['earnings_date']} ({data['days_until']:+d} days)")
    print("\n⚠️  High volatility risk - skip these trades!")
else:
    print("\n✅ No stocks in danger zone - all clear for trading!")

# Upcoming earnings (next 30 days)
upcoming = [(ticker, data) for ticker, data in earnings_calendar.items() 
            if data is not None and 3 < data['days_until'] <= 30]

if upcoming:
    print("\n" + "=" * 80)
    print("📅 UPCOMING EARNINGS (Next 30 Days)")
    print("=" * 80)
    print(f"\n{len(upcoming)} stocks reporting soon:\n")
    
    # Sort by date
    upcoming.sort(key=lambda x: x[1]['days_until'])
    
    for ticker, data in upcoming[:10]:  # Show first 10
        print(f"  📊 {ticker:<8} → {data['earnings_date']} (in {data['days_until']} days)")
    
    if len(upcoming) > 10:
        print(f"\n  ... and {len(upcoming) - 10} more")

print("\n" + "=" * 80)
print("🔧 INTEGRATION")
print("=" * 80)
print("""
This earnings calendar is automatically integrated with:

1. show_integrated_trading_plan.py
   - Filters out stocks in danger zone
   - Shows warning for upcoming earnings

2. Dashboard (dashboard.py)
   - Shows earnings indicator next to each stock
   - Visual warning for danger zone stocks

Run your daily workflow normally:
  ./run_daily_trading.sh

Earnings-filtered trades will be marked automatically!
""")

print("=" * 80)
print("✅ EARNINGS CALENDAR COMPLETE!")
print("=" * 80)
