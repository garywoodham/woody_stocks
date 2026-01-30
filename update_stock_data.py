#!/usr/bin/env python3
"""
Incrementally update stock data with only the most recent days.
Only downloads data since the last available date in the CSV.
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_FILE = "data/multi_sector_stocks.csv"

print("="*80)
print("INCREMENTAL STOCK DATA UPDATE")
print("="*80)

# Load existing data
print("\n📊 Loading existing stock data...")
df_existing = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)

# Get last date in data
last_date = df_existing.index.max()
print(f"✓ Loaded {len(df_existing):,} existing records")
print(f"  Last date in data: {last_date.strftime('%Y-%m-%d')}")

# Get list of stocks
stocks_info = df_existing[['Stock', 'Ticker', 'Sector']].drop_duplicates().values
tickers = df_existing['Ticker'].unique()

print(f"  Stocks: {len(tickers)}")

# Calculate date range for update
today = pd.Timestamp.now(tz=last_date.tz)
# Add 1 day to last_date to avoid duplicates
start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
end_date = today.strftime('%Y-%m-%d')

days_to_update = (today - last_date).days

if days_to_update <= 0:
    print(f"\n✓ Data is already up to date!")
    exit(0)

print(f"\n📅 Update period:")
print(f"  From: {start_date}")
print(f"  To: {end_date}")
print(f"  Days to fetch: {days_to_update}")

def download_incremental_data(ticker, stock_name, sector):
    """Download only recent data for a stock."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, interval="1d")
        
        if df.empty:
            return None
        
        # Add metadata
        df['Stock'] = stock_name
        df['Ticker'] = ticker
        df['Sector'] = sector
        
        return df
        
    except Exception as e:
        print(f"  ✗ Error downloading {ticker}: {str(e)}")
        return None

print(f"\n🔄 Downloading updates for {len(tickers)} stocks...")

new_data = []

# Download with parallel processing
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {}
    
    for stock_name, ticker, sector in stocks_info:
        future = executor.submit(download_incremental_data, ticker, stock_name, sector)
        futures[future] = (ticker, stock_name)
    
    completed = 0
    for future in as_completed(futures):
        ticker, stock_name = futures[future]
        df = future.result()
        if df is not None and not df.empty:
            new_data.append(df)
            completed += 1
            print(f"  [{completed}/{len(tickers)}] {stock_name} ({ticker}): +{len(df)} records")

if not new_data:
    print("\n⚠️  No new data downloaded")
    exit(0)

# Combine new data
print(f"\n{'='*80}")
print("MERGING NEW DATA")
print(f"{'='*80}")

df_new = pd.concat(new_data, ignore_index=False)
df_new = df_new.sort_index()

print(f"\n✓ Downloaded {len(df_new):,} new records")
print(f"  Date range: {df_new.index.min().strftime('%Y-%m-%d')} to {df_new.index.max().strftime('%Y-%m-%d')}")

# Merge with existing data
# Create a unique key for deduplication (date + ticker)
df_existing['_merge_key'] = df_existing.index.astype(str) + '_' + df_existing['Ticker']
df_new['_merge_key'] = df_new.index.astype(str) + '_' + df_new['Ticker']

# Remove records from existing data that are in new data
df_existing_filtered = df_existing[~df_existing['_merge_key'].isin(df_new['_merge_key'])]

# Combine
df_combined = pd.concat([df_existing_filtered.drop('_merge_key', axis=1), df_new.drop('_merge_key', axis=1)])
df_combined = df_combined.sort_index()

print(f"\n💾 Saving updated data...")
df_combined.to_csv(DATA_FILE)

print(f"\n{'='*80}")
print("✅ UPDATE COMPLETE!")
print(f"{'='*80}")
print(f"Total records: {len(df_combined):,} (+{len(df_new):,})")
print(f"Date range: {df_combined.index.min().strftime('%Y-%m-%d')} to {df_combined.index.max().strftime('%Y-%m-%d')}")
print(f"Saved to: {DATA_FILE}")
print(f"{'='*80}")
