import yfinance as yf
import pandas as pd
from datetime import datetime
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define stocks by sector - Top 5 in each for growth potential
STOCKS = {
    'Defence': {
        'BA.L': 'BAE Systems',
        'LMT': 'Lockheed Martin',
        'NOC': 'Northrop Grumman',
        'RTX': 'Raytheon Technologies',
        'RR.L': 'Rolls-Royce'
    },
    'Banking': {
        'BARC.L': 'Barclays',
        'HSBA.L': 'HSBC',
        'LLOY.L': 'Lloyds Banking',
        'NWG.L': 'NatWest Group',
        'STAN.L': 'Standard Chartered'
    },
    'Pharma': {
        'AZN.L': 'AstraZeneca',
        'GSK.L': 'GSK',
        'PFE': 'Pfizer',
        'JNJ': 'Johnson & Johnson',
        'MRNA': 'Moderna'
    },
    'Technology': {
        'AAPL': 'Apple',
        'MSFT': 'Microsoft',
        'NVDA': 'NVIDIA',
        'GOOGL': 'Alphabet',
        'AMZN': 'Amazon'
    }
}

def download_stock_data(ticker, stock_name, sector):
    """
    Download historic daily stock data for a single stock.
    Returns DataFrame with stock data or None if failed.
    """
    try:
        print(f"Downloading {stock_name} ({ticker})...")
        
        # Download last 5 years of data (daily)
        stock = yf.Ticker(ticker)
        df = stock.history(period="5y", interval="1d")
        
        if df.empty:
            print(f"  ⚠️  No data found for {ticker}")
            return None
        
        # Add metadata columns
        df['Stock'] = stock_name
        df['Ticker'] = ticker
        df['Sector'] = sector
        
        print(f"  ✓ Downloaded {len(df)} records for {stock_name}")
        return df
        
    except Exception as e:
        print(f"  ✗ Error downloading {ticker}: {str(e)}")
        return None

def download_all_stocks():
    """
    Download historic daily stock data for all defined stocks across sectors.
    """
    print("="*80)
    print("DOWNLOADING MULTI-SECTOR STOCK DATA")
    print("="*80)
    print(f"Sectors: {', '.join(STOCKS.keys())}")
    print(f"Total stocks: {sum(len(stocks) for stocks in STOCKS.values())}")
    print("="*80)
    print()
    
    all_data = []
    
    # Download stocks with parallel processing for speed
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {}
        
        for sector, stocks in STOCKS.items():
            print(f"\n[{sector.upper()}]")
            for ticker, name in stocks.items():
                future = executor.submit(download_stock_data, ticker, name, sector)
                futures[future] = (ticker, name, sector)
        
        # Collect results as they complete
        for future in as_completed(futures):
            df = future.result()
            if df is not None:
                all_data.append(df)
    
    if not all_data:
        print("\n❌ No data was downloaded successfully!")
        return
    
    # Combine all data into single DataFrame
    print(f"\n{'='*80}")
    print("CONSOLIDATING DATA")
    print(f"{'='*80}")
    combined_df = pd.concat(all_data, ignore_index=False)
    combined_df = combined_df.sort_index()
    
    # Create data directory if it doesn't exist
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Save consolidated data
    filename = f"{data_dir}/multi_sector_stocks.csv"
    combined_df.to_csv(filename)
    
    print(f"\n✓ All data saved to: {filename}")
    print(f"Total records: {len(combined_df):,}")
    print(f"Date range: {combined_df.index.min().strftime('%Y-%m-%d')} to {combined_df.index.max().strftime('%Y-%m-%d')}")
    print(f"Stocks downloaded: {combined_df['Stock'].nunique()}")
    
    # Summary by sector
    print(f"\n{'='*80}")
    print("SUMMARY BY SECTOR")
    print(f"{'='*80}")
    for sector in STOCKS.keys():
        sector_data = combined_df[combined_df['Sector'] == sector]
        print(f"\n{sector}:")
        print(f"  Stocks: {sector_data['Stock'].nunique()}")
        print(f"  Records: {len(sector_data):,}")
        for stock in sector_data['Stock'].unique():
            stock_records = len(sector_data[sector_data['Stock'] == stock])
            ticker = sector_data[sector_data['Stock'] == stock]['Ticker'].iloc[0]
            latest_price = sector_data[sector_data['Stock'] == stock]['Close'].iloc[-1]
            print(f"    • {stock} ({ticker}): {stock_records} days, Latest: ${latest_price:.2f}")
    
    print(f"\n{'='*80}")
    print("✓ DOWNLOAD COMPLETE!")
    print(f"{'='*80}")

if __name__ == "__main__":
    download_all_stocks()
