#!/usr/bin/env python3
"""
Backtest the BUY/HOLD/SELL recommendation system.
Tests how recommendations would have performed historically.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RecommendationBacktester:
    """
    Backtest trading based on BUY/HOLD/SELL recommendations.
    """
    
    def __init__(self, initial_capital=10000, commission=0.001):
        """
        Initialize backtester.
        
        Args:
            initial_capital: Starting capital in dollars
            commission: Trading commission as decimal (0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        
    def backtest_stock(self, df_prices, df_recommendations, ticker):
        """
        Backtest recommendation strategy for a single stock.
        
        Args:
            df_prices: DataFrame with date index and OHLCV data
            df_recommendations: Historical recommendations with dates
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with backtest results
        """
        capital = self.initial_capital
        position = 0  # shares held
        trades = []
        equity_curve = []
        
        # Track when we last traded
        last_trade_date = None
        
        # Iterate through recommendation history
        for date, row in df_recommendations.iterrows():
            if date not in df_prices.index:
                continue
                
            current_price = df_prices.loc[date, 'Close']
            recommendation = row['Recommendation']
            score = row['Score']
            
            # Calculate current portfolio value
            portfolio_value = capital + (position * current_price if position > 0 else 0)
            
            # Trading logic based on recommendation
            if recommendation == 'BUY' and position == 0:
                # Buy signal - allocate capital based on score strength
                allocation = min(1.0, 0.5 + abs(score))  # 50-100% allocation
                shares_to_buy = int((capital * allocation) / (current_price * (1 + self.commission)))
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price * (1 + self.commission)
                    position = shares_to_buy
                    capital -= cost
                    last_trade_date = date
                    
                    trades.append({
                        'Date': date,
                        'Action': 'BUY',
                        'Price': current_price,
                        'Shares': shares_to_buy,
                        'Score': score,
                        'Value': cost
                    })
                    
            elif recommendation == 'SELL' and position > 0:
                # Sell signal - exit position
                proceeds = position * current_price * (1 - self.commission)
                profit = proceeds - (position * trades[-1]['Price'] * (1 + self.commission))
                capital += proceeds
                
                trades.append({
                    'Date': date,
                    'Action': 'SELL',
                    'Price': current_price,
                    'Shares': position,
                    'Score': score,
                    'Value': proceeds,
                    'Profit': profit
                })
                
                position = 0
                last_trade_date = date
                
            # Record equity
            equity_curve.append({
                'Date': date,
                'Portfolio_Value': portfolio_value,
                'Position': position,
                'Cash': capital
            })
        
        # Calculate final value (liquidate any remaining position)
        if position > 0:
            final_price = df_prices.iloc[-1]['Close']
            final_value = capital + (position * final_price * (1 - self.commission))
        else:
            final_value = capital
            
        # Calculate metrics
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Buy and hold comparison
        buy_hold_shares = int(self.initial_capital / (df_prices.iloc[0]['Close'] * (1 + self.commission)))
        buy_hold_value = buy_hold_shares * df_prices.iloc[-1]['Close'] * (1 - self.commission)
        buy_hold_return = (buy_hold_value - self.initial_capital) / self.initial_capital
        
        # Win rate
        profitable_trades = sum(1 for t in trades if t['Action'] == 'SELL' and t.get('Profit', 0) > 0)
        total_sell_trades = sum(1 for t in trades if t['Action'] == 'SELL')
        win_rate = profitable_trades / total_sell_trades if total_sell_trades > 0 else 0
        
        # Calculate Sharpe ratio from equity curve
        if len(equity_curve) > 1:
            df_equity = pd.DataFrame(equity_curve)
            returns = df_equity['Portfolio_Value'].pct_change().dropna()
            sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        else:
            sharpe = 0
            
        # Max drawdown
        df_equity = pd.DataFrame(equity_curve)
        rolling_max = df_equity['Portfolio_Value'].expanding().max()
        drawdowns = (df_equity['Portfolio_Value'] - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min()) if len(drawdowns) > 0 else 0
        
        return {
            'Ticker': ticker,
            'Total_Return': total_return,
            'Buy_Hold_Return': buy_hold_return,
            'Excess_Return': total_return - buy_hold_return,
            'Final_Value': final_value,
            'Total_Trades': len(trades),
            'Win_Rate': win_rate,
            'Sharpe_Ratio': sharpe,
            'Max_Drawdown': max_drawdown,
            'Trades': trades,
            'Equity_Curve': equity_curve
        }

def main():
    print("\n" + "="*80)
    print("BACKTESTING RECOMMENDATION SYSTEM")
    print("="*80 + "\n")
    
    # Load stock data
    print("Loading data...")
    df_stocks = pd.read_csv('data/multi_sector_stocks.csv', index_col=0, parse_dates=True)
    df_recommendations = pd.read_csv('stock_recommendations.csv')
    
    print(f"✓ Loaded {len(df_recommendations)} stock recommendations")
    print(f"✓ Loaded stock price data\n")
    
    # Initialize backtester
    backtester = RecommendationBacktester(initial_capital=10000)
    
    # Run backtests for each stock
    results = []
    
    print("Running backtests...")
    print("-" * 80)
    
    for _, rec in df_recommendations.iterrows():
        ticker = rec['Ticker']
        stock = rec['Stock']
        
        # Filter prices for this stock
        df_stock = df_stocks[df_stocks['Ticker'] == ticker].copy()
        
        if len(df_stock) < 100:  # Need sufficient history
            print(f"⚠️  {stock:25s} - Insufficient data (only {len(df_stock)} days)")
            continue
            
        # Create mock historical recommendations (using current recommendation for simplicity)
        # In production, this would use actual historical recommendation data
        df_rec_history = pd.DataFrame({
            'Date': df_stock.index[-100:],  # Last 100 days
            'Recommendation': [rec['Recommendation']] * 100,
            'Score': [rec['Score']] * 100
        }).set_index('Date')
        
        try:
            result = backtester.backtest_stock(df_stock, df_rec_history, ticker)
            result['Stock'] = stock
            result['Sector'] = rec['Sector']
            result['Recommendation'] = rec['Recommendation']
            results.append(result)
            
            # Print summary
            status = '✓' if result['Total_Return'] > 0 else '✗'
            print(f"{status} {stock:25s} {result['Total_Return']:>8.2%} | "
                  f"Trades: {result['Total_Trades']:3d} | "
                  f"Win Rate: {result['Win_Rate']:>5.1%} | "
                  f"Sharpe: {result['Sharpe_Ratio']:>5.2f}")
                  
        except Exception as e:
            print(f"✗ {stock:25s} - Error: {str(e)}")
            continue
    
    # Create summary DataFrame
    df_results = pd.DataFrame([{
        'Stock': r['Stock'],
        'Ticker': r['Ticker'],
        'Sector': r['Sector'],
        'Recommendation': r['Recommendation'],
        'Total_Return': r['Total_Return'],
        'Buy_Hold_Return': r['Buy_Hold_Return'],
        'Excess_Return': r['Excess_Return'],
        'Win_Rate': r['Win_Rate'],
        'Total_Trades': r['Total_Trades'],
        'Sharpe_Ratio': r['Sharpe_Ratio'],
        'Max_Drawdown': r['Max_Drawdown'],
        'Final_Value': r['Final_Value']
    } for r in results])
    
    # Save results
    df_results.to_csv('backtest_recommendations.csv', index=False)
    print("\n" + "-" * 80)
    print(f"✓ Saved results to backtest_recommendations.csv\n")
    
    # Print summary statistics
    print("=" * 80)
    print("BACKTEST SUMMARY")
    print("=" * 80 + "\n")
    
    print(f"Stocks Tested: {len(df_results)}")
    print(f"Initial Capital: ${backtester.initial_capital:,.0f}")
    print(f"\nOverall Performance:")
    print(f"  Average Return:       {df_results['Total_Return'].mean():>8.2%}")
    print(f"  Median Return:        {df_results['Total_Return'].median():>8.2%}")
    print(f"  Best Return:          {df_results['Total_Return'].max():>8.2%} ({df_results.loc[df_results['Total_Return'].idxmax(), 'Stock']})")
    print(f"  Worst Return:         {df_results['Total_Return'].min():>8.2%} ({df_results.loc[df_results['Total_Return'].idxmin(), 'Stock']})")
    print(f"\nVs Buy & Hold:")
    print(f"  Excess Return:        {df_results['Excess_Return'].mean():>8.2%}")
    print(f"  Outperformed B&H:     {(df_results['Excess_Return'] > 0).sum()} / {len(df_results)}")
    print(f"\nTrading Metrics:")
    print(f"  Average Win Rate:     {df_results['Win_Rate'].mean():>8.2%}")
    print(f"  Average Sharpe:       {df_results['Sharpe_Ratio'].mean():>8.2f}")
    print(f"  Average Max DD:       {df_results['Max_Drawdown'].mean():>8.2%}")
    print(f"  Total Trades:         {df_results['Total_Trades'].sum():>8.0f}")
    
    # Performance by recommendation type
    print(f"\nPerformance by Recommendation:")
    for rec in ['BUY', 'HOLD', 'SELL']:
        rec_data = df_results[df_results['Recommendation'] == rec]
        if len(rec_data) > 0:
            print(f"  {rec:5s} ({len(rec_data):2d} stocks):  {rec_data['Total_Return'].mean():>8.2%} avg return")
    
    # Performance by sector
    print(f"\nPerformance by Sector:")
    for sector in df_results['Sector'].unique():
        sector_data = df_results[df_results['Sector'] == sector]
        print(f"  {sector:12s}:  {sector_data['Total_Return'].mean():>8.2%} avg return")
    
    print("\n" + "=" * 80)
    print("✓ BACKTEST COMPLETE!")
    print("=" * 80 + "\n")
    
    print("Note: This is a simplified backtest using current recommendations.")
    print("For production use, implement proper walk-forward testing with historical recommendation data.")
    print()

if __name__ == '__main__':
    main()
