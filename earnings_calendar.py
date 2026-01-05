#!/usr/bin/env python3
"""
Earnings Calendar Tracker
- Fetch earnings dates from Yahoo Finance
- Flag stocks with upcoming earnings (avoid volatility)
- Historical earnings surprise tracking
- Earnings season warnings
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EarningsCalendar:
    """Track earnings dates and provide warnings"""
    
    def __init__(self, warning_days=5):
        """
        Initialize earnings calendar.
        
        Args:
            warning_days: Days before earnings to flag (default 5)
        """
        self.warning_days = warning_days
        
    def fetch_earnings_dates(self, tickers):
        """Fetch earnings dates for list of tickers"""
        print(f"\n📅 Fetching Earnings Dates for {len(tickers)} stocks...")
        
        earnings_data = []
        today = datetime.now().date()
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                
                # Get earnings calendar (can be dict or DataFrame)
                calendar = stock.calendar
                
                earnings_date = None
                
                # Handle dict format (new yfinance API)
                if isinstance(calendar, dict):
                    if 'Earnings Date' in calendar:
                        earnings_date = calendar['Earnings Date']
                        # Handle list of dates
                        if isinstance(earnings_date, list) and len(earnings_date) > 0:
                            earnings_date = earnings_date[0]
                
                # Handle DataFrame format (old API)
                elif calendar is not None and not calendar.empty:
                    if 'Earnings Date' in calendar.index:
                        earnings_date = calendar.loc['Earnings Date']
                        if isinstance(earnings_date, pd.Series):
                            earnings_date = earnings_date.iloc[0]
                
                # Convert to date if we got something
                if earnings_date is not None and pd.notna(earnings_date):
                    if isinstance(earnings_date, str):
                        earnings_date = pd.to_datetime(earnings_date).date()
                    elif isinstance(earnings_date, pd.Timestamp):
                        earnings_date = earnings_date.date()
                    
                    days_until = (earnings_date - today).days
                    
                    earnings_data.append({
                        'Ticker': ticker,
                        'Earnings_Date': earnings_date,
                        'Days_Until': days_until,
                        'Warning': days_until <= self.warning_days and days_until >= 0
                    })
                else:
                    earnings_data.append({
                        'Ticker': ticker,
                        'Earnings_Date': None,
                        'Days_Until': 999,
                        'Warning': False
                    })
                    
            except Exception as e:
                print(f"  ⚠️  Error fetching {ticker}: {str(e)[:50]}")
                earnings_data.append({
                    'Ticker': ticker,
                    'Earnings_Date': None,
                    'Days_Until': 999,
                    'Warning': False
                })
        
        df_earnings = pd.DataFrame(earnings_data)
        
        # Count warnings
        warnings_count = df_earnings['Warning'].sum()
        if warnings_count > 0:
            print(f"   ⚠️  {warnings_count} stocks with earnings in next {self.warning_days} days")
        else:
            print(f"   ✓ No earnings in next {self.warning_days} days")
        
        return df_earnings
    
    def get_earnings_warnings(self, df_earnings, df_recommendations):
        """Generate warnings for stocks with upcoming earnings"""
        print("\n⚠️  Checking Earnings Conflicts...")
        
        # Merge with recommendations
        df_merged = df_recommendations.merge(df_earnings, on='Ticker', how='left')
        
        # Filter BUY recommendations with earnings warnings
        df_conflicts = df_merged[
            (df_merged['Recommendation'] == 'BUY') & 
            (df_merged['Warning'] == True)
        ].copy()
        
        if df_conflicts.empty:
            print("   ✓ No earnings conflicts with BUY recommendations")
            return pd.DataFrame()
        
        print(f"   ⚠️  {len(df_conflicts)} BUY recommendations have upcoming earnings:")
        
        warnings = []
        for _, row in df_conflicts.iterrows():
            warnings.append({
                'Ticker': row['Ticker'],
                'Stock': row.get('Stock', ''),
                'Recommendation': 'BUY',
                'Score': row.get('Score', 0),
                'Earnings_Date': row['Earnings_Date'],
                'Days_Until': row['Days_Until'],
                'Severity': 'HIGH' if row['Days_Until'] <= 2 else 'MEDIUM',
                'Message': f"Earnings in {row['Days_Until']} days - expect high volatility",
                'Action': 'WAIT' if row['Days_Until'] <= 2 else 'REDUCE_SIZE'
            })
        
        df_warnings = pd.DataFrame(warnings)
        
        for _, warning in df_warnings.iterrows():
            print(f"      {warning['Ticker']:6s} - {warning['Message']} → {warning['Action']}")
        
        return df_warnings
    
    def get_earnings_calendar(self, df_earnings, days_ahead=30):
        """Get calendar view of upcoming earnings"""
        today = datetime.now().date()
        end_date = today + timedelta(days=days_ahead)
        
        df_upcoming = df_earnings[
            (df_earnings['Earnings_Date'].notna()) &
            (df_earnings['Days_Until'] >= 0) &
            (df_earnings['Days_Until'] <= days_ahead)
        ].copy()
        
        df_upcoming = df_upcoming.sort_values('Days_Until')
        
        return df_upcoming
    
    def fetch_earnings_history(self, tickers, periods=4):
        """
        Fetch historical earnings surprises.
        Shows if company tends to beat or miss estimates.
        """
        print(f"\n📊 Fetching Earnings History (last {periods} quarters)...")
        
        history_data = []
        
        for ticker in tickers[:10]:  # Limit to first 10 to save time
            try:
                stock = yf.Ticker(ticker)
                earnings_history = stock.earnings_history
                
                if earnings_history is not None and not earnings_history.empty:
                    recent = earnings_history.head(periods)
                    
                    surprises = []
                    for _, row in recent.iterrows():
                        if pd.notna(row.get('Surprise', None)):
                            surprises.append(row['Surprise'])
                    
                    if surprises:
                        avg_surprise = sum(surprises) / len(surprises)
                        beat_rate = len([s for s in surprises if s > 0]) / len(surprises)
                        
                        history_data.append({
                            'Ticker': ticker,
                            'Avg_Surprise_Pct': avg_surprise * 100,
                            'Beat_Rate': beat_rate,
                            'Sample_Size': len(surprises),
                            'Earnings_Quality': 'Good' if beat_rate > 0.6 else 'Mixed' if beat_rate > 0.3 else 'Poor'
                        })
            except Exception as e:
                pass  # Skip errors silently
        
        if history_data:
            df_history = pd.DataFrame(history_data)
            print(f"   ✓ Retrieved history for {len(df_history)} stocks")
            return df_history
        else:
            print("   ⚠️  No earnings history available")
            return pd.DataFrame()

def main():
    print("\n" + "="*80)
    print("📅 EARNINGS CALENDAR SYSTEM")
    print("="*80)
    
    # Load recommendations
    try:
        df_recommendations = pd.read_csv('stock_recommendations.csv')
        tickers = df_recommendations['Ticker'].unique().tolist()
    except FileNotFoundError:
        print("❌ stock_recommendations.csv not found")
        return
    
    # Initialize earnings calendar
    ec = EarningsCalendar(warning_days=5)
    
    # 1. Fetch earnings dates
    df_earnings = ec.fetch_earnings_dates(tickers)
    df_earnings.to_csv('earnings_calendar.csv', index=False)
    print(f"✓ Saved earnings dates for {len(df_earnings)} stocks")
    
    # 2. Get earnings warnings
    df_warnings = ec.get_earnings_warnings(df_earnings, df_recommendations)
    if not df_warnings.empty:
        df_warnings.to_csv('earnings_warnings.csv', index=False)
        print(f"✓ Saved {len(df_warnings)} earnings warnings")
    
    # 3. Display upcoming earnings calendar (next 30 days)
    df_upcoming = ec.get_earnings_calendar(df_earnings, days_ahead=30)
    if not df_upcoming.empty:
        print(f"\n📅 UPCOMING EARNINGS (Next 30 Days):")
        for _, row in df_upcoming.head(10).iterrows():
            date_str = row['Earnings_Date'].strftime('%b %d, %Y')
            days = row['Days_Until']
            print(f"   {row['Ticker']:6s} - {date_str} ({days} days)")
    else:
        print("\n✓ No earnings in next 30 days")
    
    # 4. Earnings history (optional - slower)
    try:
        df_history = ec.fetch_earnings_history(tickers, periods=4)
        if not df_history.empty:
            df_history.to_csv('earnings_history.csv', index=False)
            print(f"\n📊 EARNINGS QUALITY (Historical Beats):")
            for _, row in df_history.head(5).iterrows():
                print(f"   {row['Ticker']:6s} - Beat Rate: {row['Beat_Rate']:.0%}  Avg Surprise: {row['Avg_Surprise_Pct']:+.1f}%")
    except Exception as e:
        print(f"\n⚠️  Earnings history fetch failed: {str(e)[:50]}")
    
    # Summary
    total_stocks = len(df_earnings)
    with_dates = df_earnings['Earnings_Date'].notna().sum()
    warnings_count = df_earnings['Warning'].sum()
    
    print("\n" + "="*80)
    print("✅ EARNINGS CALENDAR COMPLETE")
    print("="*80)
    print(f"\n📊 Summary:")
    print(f"   Total Stocks:        {total_stocks}")
    print(f"   With Earnings Dates: {with_dates}")
    print(f"   ⚠️  Earnings Warnings:  {warnings_count}")
    print(f"\n📁 Files created:")
    print(f"   • earnings_calendar.csv - All earnings dates")
    if not df_warnings.empty:
        print(f"   • earnings_warnings.csv - Conflict warnings")
    print()

if __name__ == '__main__':
    main()
