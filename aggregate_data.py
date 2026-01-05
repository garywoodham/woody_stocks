import pandas as pd
import numpy as np
import os

def aggregate_to_weekly(df):
    """
    Aggregate daily stock data to weekly intervals (ending Friday).
    
    For OHLC data:
    - Open: first value of the week
    - High: max value of the week
    - Low: min value of the week
    - Close: last value of the week
    - Volume: sum of the week
    """
    # Ensure index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    
    # Group by stock/ticker/sector and resample to weekly
    weekly_dfs = []
    
    for (stock, ticker, sector), group in df.groupby(['Stock', 'Ticker', 'Sector']):
        weekly = group.resample('W-FRI').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum',
            'Dividends': 'sum',
            'Stock Splits': 'sum'
        })
        
        # Remove weeks with no data
        weekly = weekly.dropna(subset=['Close'])
        
        # Add metadata back
        weekly['Stock'] = stock
        weekly['Ticker'] = ticker
        weekly['Sector'] = sector
        
        weekly_dfs.append(weekly)
    
    return pd.concat(weekly_dfs)

def aggregate_to_monthly(df):
    """
    Aggregate daily stock data to monthly intervals (end of month).
    
    For OHLC data:
    - Open: first value of the month
    - High: max value of the month
    - Low: min value of the month
    - Close: last value of the month
    - Volume: sum of the month
    """
    # Ensure index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    
    # Group by stock/ticker/sector and resample to monthly
    monthly_dfs = []
    
    for (stock, ticker, sector), group in df.groupby(['Stock', 'Ticker', 'Sector']):
        monthly = group.resample('ME').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum',
            'Dividends': 'sum',
            'Stock Splits': 'sum'
        })
        
        # Remove months with no data
        monthly = monthly.dropna(subset=['Close'])
        
        # Add metadata back
        monthly['Stock'] = stock
        monthly['Ticker'] = ticker
        monthly['Sector'] = sector
        
        monthly_dfs.append(monthly)
    
    return pd.concat(monthly_dfs)

def main():
    print("="*80)
    print("STOCK DATA AGGREGATION - DAILY TO WEEKLY/MONTHLY")
    print("="*80)
    
    # Load daily data
    daily_file = 'data/multi_sector_stocks.csv'
    if not os.path.exists(daily_file):
        print(f"\n❌ Error: {daily_file} not found!")
        print("Please run download_stock_data.py first to download the daily data.")
        return
    
    print(f"\nLoading daily data from {daily_file}...")
    df_daily = pd.read_csv(daily_file, index_col=0, parse_dates=True)
    
    print(f"Daily data:")
    print(f"  Records: {len(df_daily):,}")
    print(f"  Date range: {df_daily.index.min().strftime('%Y-%m-%d')} to {df_daily.index.max().strftime('%Y-%m-%d')}")
    print(f"  Stocks: {df_daily['Stock'].nunique()}")
    print(f"  Sectors: {', '.join(df_daily['Sector'].unique())}")
    
    # Aggregate to weekly
    print(f"\n{'='*80}")
    print("AGGREGATING TO WEEKLY DATA")
    print(f"{'='*80}")
    df_weekly = aggregate_to_weekly(df_daily)
    weekly_file = 'data/multi_sector_stocks_weekly.csv'
    df_weekly.to_csv(weekly_file)
    
    print(f"\n✓ Weekly data saved to: {weekly_file}")
    print(f"  Records: {len(df_weekly):,}")
    print(f"  Date range: {df_weekly.index.min().strftime('%Y-%m-%d')} to {df_weekly.index.max().strftime('%Y-%m-%d')}")
    print(f"  Average weeks per stock: {len(df_weekly) / df_weekly['Stock'].nunique():.0f}")
    
    # Aggregate to monthly
    print(f"\n{'='*80}")
    print("AGGREGATING TO MONTHLY DATA")
    print(f"{'='*80}")
    df_monthly = aggregate_to_monthly(df_daily)
    monthly_file = 'data/multi_sector_stocks_monthly.csv'
    df_monthly.to_csv(monthly_file)
    
    print(f"\n✓ Monthly data saved to: {monthly_file}")
    print(f"  Records: {len(df_monthly):,}")
    print(f"  Date range: {df_monthly.index.min().strftime('%Y-%m-%d')} to {df_monthly.index.max().strftime('%Y-%m-%d')}")
    print(f"  Average months per stock: {len(df_monthly) / df_monthly['Stock'].nunique():.0f}")
    
    # Summary by sector
    print(f"\n{'='*80}")
    print("SUMMARY BY SECTOR")
    print(f"{'='*80}")
    for sector in df_daily['Sector'].unique():
        daily_sector = df_daily[df_daily['Sector'] == sector]
        weekly_sector = df_weekly[df_weekly['Sector'] == sector]
        monthly_sector = df_monthly[df_monthly['Sector'] == sector]
        
        print(f"\n{sector}:")
        print(f"  Daily records: {len(daily_sector):,}")
        print(f"  Weekly records: {len(weekly_sector):,}")
        print(f"  Monthly records: {len(monthly_sector):,}")
    
    print(f"\n{'='*80}")
    print("✓ AGGREGATION COMPLETE!")
    print(f"{'='*80}")
    print("\nGenerated files:")
    print(f"  1. {daily_file} (daily data)")
    print(f"  2. {weekly_file} (weekly data)")
    print(f"  3. {monthly_file} (monthly data)")
    print("\nYou can now train models on different time periods using train_model.py")

if __name__ == "__main__":
    main()
