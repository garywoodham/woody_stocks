"""
Backfill FREE alternative data with FULL HISTORICAL ACCESS
Sources that have 2+ years of history for proper backtesting
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pytrends.request import TrendReq
import time
import os

# Stock list
STOCK_FILE = 'stocks.csv'
OUTPUT_DIR = 'alternative_data'

print("="*80)
print("BACKFILLING FREE ALTERNATIVE DATA - HISTORICAL SOURCES")
print("="*80)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load stocks
stocks_df = pd.read_csv(STOCK_FILE)
tickers = stocks_df['ticker'].tolist()
print(f"\nLoaded {len(tickers)} stocks from {STOCK_FILE}")

# ============================================================================
# 1. GOOGLE TRENDS - 5 YEARS OF WEEKLY DATA
# ============================================================================
print("\n" + "="*80)
print("1. GOOGLE TRENDS (5 years weekly data)")
print("="*80)

# Company name mapping (Google Trends uses company names, not tickers)
COMPANY_NAMES = {
    'AAPL': 'Apple',
    'GOOGL': 'Google',
    'MSFT': 'Microsoft',
    'AMZN': 'Amazon',
    'TSLA': 'Tesla',
    'META': 'Meta',
    'NVDA': 'Nvidia',
    'JPM': 'JPMorgan',
    'BAC': 'Bank of America',
    'WMT': 'Walmart',
    'PFE': 'Pfizer',
    'JNJ': 'Johnson & Johnson',
    'XOM': 'Exxon',
    'CVX': 'Chevron',
    'BA': 'Boeing',
    'CAT': 'Caterpillar',
    'GE': 'General Electric',
    'GM': 'General Motors',
    'F': 'Ford',
    'DIS': 'Disney',
    'NFLX': 'Netflix',
    'INTC': 'Intel',
    'AMD': 'AMD',
    'TLRY': 'Tilray',
    'AMC': 'AMC Theatres',
    'GME': 'GameStop',
    'PLTR': 'Palantir',
    'SOFI': 'SoFi',
    'RIVN': 'Rivian',
    'SPCE': 'Virgin Galactic',
}

pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25))

all_trends = []
success_count = 0
fail_count = 0

for ticker in tickers[:20]:  # Start with first 20 to avoid rate limits
    company_name = COMPANY_NAMES.get(ticker, ticker)
    
    try:
        print(f"\nFetching: {ticker} ({company_name})...", end=' ')
        
        # Get 5 years of weekly data
        pytrends.build_payload([company_name], timeframe='today 5-y')
        trends = pytrends.interest_over_time()
        
        if not trends.empty and company_name in trends.columns:
            trends = trends[[company_name]].copy()
            trends.columns = ['search_interest']
            trends['ticker'] = ticker
            trends['date'] = trends.index
            
            all_trends.append(trends)
            
            earliest = trends.index.min()
            latest = trends.index.max()
            print(f"✅ Got {len(trends)} weeks ({earliest.strftime('%Y-%m-%d')} to {latest.strftime('%Y-%m-%d')})")
            success_count += 1
        else:
            print(f"⚠️  No data (company name might be ambiguous)")
            fail_count += 1
        
        # Rate limit: Wait between requests
        time.sleep(2)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        fail_count += 1
        time.sleep(5)

# Combine and save
if all_trends:
    google_trends_df = pd.concat(all_trends, ignore_index=True)
    output_file = f'{OUTPUT_DIR}/google_trends_5y.csv'
    google_trends_df.to_csv(output_file, index=False)
    
    print(f"\n✅ SUCCESS: Saved {len(google_trends_df)} records to {output_file}")
    print(f"   Success: {success_count} stocks")
    print(f"   Failed: {fail_count} stocks")
else:
    print(f"\n❌ No data collected")

# ============================================================================
# 2. VIX HISTORICAL DATA (from Yahoo Finance)
# ============================================================================
print("\n" + "="*80)
print("2. VIX HISTORICAL DATA (Market regime indicator)")
print("="*80)

try:
    vix = yf.Ticker('^VIX')
    vix_hist = vix.history(period='5y')
    
    if not vix_hist.empty:
        vix_hist['date'] = vix_hist.index
        vix_hist['vix_close'] = vix_hist['Close']
        
        # Add regime classification
        vix_hist['market_regime'] = pd.cut(
            vix_hist['vix_close'],
            bins=[0, 15, 20, 30, 100],
            labels=['low_volatility', 'normal', 'high_volatility', 'extreme_fear']
        )
        
        output_file = f'{OUTPUT_DIR}/vix_history_5y.csv'
        vix_hist[['date', 'vix_close', 'market_regime']].to_csv(output_file, index=False)
        
        print(f"✅ SUCCESS: Saved {len(vix_hist)} days of VIX data")
        print(f"   Date range: {vix_hist.index.min()} to {vix_hist.index.max()}")
        print(f"   Current VIX: {vix_hist['vix_close'].iloc[-1]:.2f}")
        print(f"   Saved to: {output_file}")
    else:
        print("❌ No VIX data available")
        
except Exception as e:
    print(f"❌ Error fetching VIX: {e}")

# ============================================================================
# 3. CBOE PUT/CALL RATIO (if available)
# ============================================================================
print("\n" + "="*80)
print("3. CBOE PUT/CALL RATIO")
print("="*80)

try:
    # Try direct CSV download
    url = 'http://www.cboe.com/publish/scheduledtask/mktdata/datahouse/totalpc.csv'
    df = pd.read_csv(url, skiprows=3)
    df['DATE'] = pd.to_datetime(df['DATE'], format='%m/%d/%Y')
    
    output_file = f'{OUTPUT_DIR}/cboe_putcall_history.csv'
    df.to_csv(output_file, index=False)
    
    print(f"✅ SUCCESS: Saved {len(df)} days of put/call data")
    print(f"   Date range: {df['DATE'].min()} to {df['DATE'].max()}")
    print(f"   Saved to: {output_file}")
    
except Exception as e:
    print(f"⚠️  CBOE direct download blocked: {e}")
    print("   Workaround: Calculate from options data or skip")

# ============================================================================
# 4. S&P 500 FUTURES (Historical overnight gaps)
# ============================================================================
print("\n" + "="*80)
print("4. S&P 500 FUTURES HISTORICAL")
print("="*80)

try:
    # ES=F is S&P 500 futures
    futures = yf.Ticker('ES=F')
    futures_hist = futures.history(period='5y')
    
    if not futures_hist.empty:
        futures_hist['date'] = futures_hist.index
        futures_hist['futures_close'] = futures_hist['Close']
        futures_hist['futures_change'] = futures_hist['Close'].pct_change() * 100
        
        output_file = f'{OUTPUT_DIR}/sp500_futures_5y.csv'
        futures_hist[['date', 'futures_close', 'futures_change']].to_csv(output_file, index=False)
        
        print(f"✅ SUCCESS: Saved {len(futures_hist)} days of futures data")
        print(f"   Date range: {futures_hist.index.min()} to {futures_hist.index.max()}")
        print(f"   Saved to: {output_file}")
    else:
        print("❌ No futures data available")
        
except Exception as e:
    print(f"❌ Error fetching futures: {e}")

# ============================================================================
# 5. SECTOR ETF DATA (For sector regime detection)
# ============================================================================
print("\n" + "="*80)
print("5. SECTOR ETF HISTORICAL DATA")
print("="*80)

SECTOR_ETFS = {
    'XLF': 'Financials',
    'XLK': 'Technology', 
    'XLE': 'Energy',
    'XLV': 'Healthcare',
    'XLI': 'Industrials',
    'XLP': 'Consumer Staples',
    'XLY': 'Consumer Discretionary',
    'XLU': 'Utilities',
    'XLRE': 'Real Estate',
}

sector_data = []

for etf, sector_name in SECTOR_ETFS.items():
    try:
        ticker = yf.Ticker(etf)
        hist = ticker.history(period='5y')
        
        if not hist.empty:
            hist['date'] = hist.index
            hist['sector'] = sector_name
            hist['sector_etf'] = etf
            hist['sector_return'] = hist['Close'].pct_change() * 100
            
            sector_data.append(hist[['date', 'sector', 'sector_etf', 'Close', 'sector_return']])
            print(f"✅ {etf} ({sector_name}): {len(hist)} days")
        else:
            print(f"⚠️  {etf}: No data")
            
    except Exception as e:
        print(f"❌ {etf}: Error - {e}")

if sector_data:
    sector_df = pd.concat(sector_data, ignore_index=True)
    output_file = f'{OUTPUT_DIR}/sector_etf_5y.csv'
    sector_df.to_csv(output_file, index=False)
    print(f"\n✅ Saved {len(sector_df)} records to {output_file}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("BACKFILL COMPLETE - SUMMARY")
print("="*80)

print(f"\n📁 Data saved to: {OUTPUT_DIR}/")
print(f"\nFiles created:")

for filename in os.listdir(OUTPUT_DIR):
    filepath = os.path.join(OUTPUT_DIR, filename)
    if os.path.isfile(filepath):
        df = pd.read_csv(filepath)
        print(f"   ✅ {filename}: {len(df)} records")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("\n1. Review data quality in alternative_data/ folder")
print("2. Merge alternative data with stock price data")
print("3. Add features to train_refined_models.py")
print("4. Retrain with cross-validation")
print("5. Measure accuracy improvement")
print("\nExpected improvement: +3-5% accuracy (53% → 56-58%)")
print("\n")
