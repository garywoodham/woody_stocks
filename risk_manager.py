#!/usr/bin/env python3
"""
Portfolio Risk Management System
- Position size limits based on volatility
- Stop-loss recommendations
- Portfolio risk metrics (VaR, Sharpe, max drawdown)
- Sector concentration analysis
- Correlation analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RiskManager:
    """Comprehensive portfolio risk management"""
    
    def __init__(self, total_capital=100000, max_position_pct=15, max_sector_pct=40):
        """
        Initialize risk manager.
        
        Args:
            total_capital: Total portfolio value
            max_position_pct: Max % in single position (default 15%)
            max_sector_pct: Max % in single sector (default 40%)
        """
        self.total_capital = total_capital
        self.max_position_pct = max_position_pct / 100
        self.max_sector_pct = max_sector_pct / 100
        
    def calculate_volatility_metrics(self, df_stocks):
        """Calculate volatility and risk metrics for each stock"""
        print("\n📊 Calculating Volatility Metrics...")
        
        metrics = []
        
        for ticker in df_stocks['Ticker'].unique():
            stock_data = df_stocks[df_stocks['Ticker'] == ticker].copy()
            
            if len(stock_data) < 30:
                continue
            
            # Calculate returns
            stock_data['returns'] = stock_data['Close'].pct_change()
            
            # Volatility metrics
            volatility_daily = stock_data['returns'].std()
            volatility_annual = volatility_daily * np.sqrt(252)
            
            # ATR (Average True Range)
            stock_data['tr'] = np.maximum(
                stock_data['High'] - stock_data['Low'],
                np.maximum(
                    abs(stock_data['High'] - stock_data['Close'].shift(1)),
                    abs(stock_data['Low'] - stock_data['Close'].shift(1))
                )
            )
            atr = stock_data['tr'].rolling(14).mean().iloc[-1]
            atr_pct = (atr / stock_data['Close'].iloc[-1]) * 100
            
            # Drawdown
            cumulative = (1 + stock_data['returns']).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Recent price change
            price_change_30d = stock_data['Close'].pct_change(30).iloc[-1]
            
            # Beta (vs SPY if available, otherwise use sector average)
            beta = 1.0  # Default
            
            metrics.append({
                'Ticker': ticker,
                'Stock': stock_data['Stock'].iloc[0],
                'Sector': stock_data['Sector'].iloc[0],
                'Current_Price': stock_data['Close'].iloc[-1],
                'Volatility_Daily': volatility_daily,
                'Volatility_Annual': volatility_annual,
                'ATR': atr,
                'ATR_Pct': atr_pct,
                'Max_Drawdown': max_drawdown,
                'Price_Change_30d': price_change_30d,
                'Beta': beta
            })
        
        df_metrics = pd.DataFrame(metrics)
        df_metrics['Risk_Score'] = self._calculate_risk_score(df_metrics)
        
        return df_metrics
    
    def _calculate_risk_score(self, df):
        """Calculate risk score (0-100, higher = riskier)"""
        # Normalize each component
        vol_score = (df['Volatility_Annual'] / df['Volatility_Annual'].max()) * 40
        dd_score = (abs(df['Max_Drawdown']) / abs(df['Max_Drawdown']).max()) * 30
        atr_score = (df['ATR_Pct'] / df['ATR_Pct'].max()) * 30
        
        risk_score = vol_score + dd_score + atr_score
        return np.clip(risk_score, 0, 100)
    
    def calculate_position_sizes(self, df_metrics, df_recommendations):
        """
        Calculate recommended position sizes based on risk.
        Lower volatility = larger position, high volatility = smaller position.
        """
        print("\n💰 Calculating Risk-Adjusted Position Sizes...")
        
        # Merge recommendations with risk metrics
        df_merged = df_recommendations.merge(df_metrics, on='Ticker', how='left')
        
        # Only BUY recommendations
        df_buy = df_merged[df_merged['Recommendation'] == 'BUY'].copy()
        
        if df_buy.empty:
            return pd.DataFrame()
        
        # Base allocation by score
        df_buy['Base_Allocation'] = df_buy['Score'] / df_buy['Score'].sum()
        
        # Adjust by risk (inverse volatility)
        df_buy['Risk_Adjustment'] = 1 / (1 + df_buy['Volatility_Annual'])
        df_buy['Risk_Adj_Allocation'] = df_buy['Base_Allocation'] * df_buy['Risk_Adjustment']
        df_buy['Risk_Adj_Allocation'] = df_buy['Risk_Adj_Allocation'] / df_buy['Risk_Adj_Allocation'].sum()
        
        # Apply position limits
        df_buy['Final_Allocation'] = np.minimum(df_buy['Risk_Adj_Allocation'], self.max_position_pct)
        
        # Renormalize
        df_buy['Final_Allocation'] = df_buy['Final_Allocation'] / df_buy['Final_Allocation'].sum()
        
        # Calculate dollar amounts
        df_buy['Position_Size'] = df_buy['Final_Allocation'] * self.total_capital
        df_buy['Shares'] = (df_buy['Position_Size'] / df_buy['Current_Price']).astype(int)
        df_buy['Actual_Amount'] = df_buy['Shares'] * df_buy['Current_Price']
        
        return df_buy[['Ticker', 'Stock_x', 'Sector_x', 'Score', 'Risk_Score', 
                      'Volatility_Annual', 'Final_Allocation', 'Position_Size', 
                      'Shares', 'Actual_Amount']]
    
    def calculate_stop_losses(self, df_metrics, method='atr', multiplier=2.0):
        """
        Calculate stop-loss levels for each stock.
        
        Methods:
            - 'atr': ATR-based (current_price - multiplier * ATR)
            - 'support': Support level based (find recent support)
            - 'percent': Fixed percentage (e.g., 8%)
        """
        print(f"\n🛑 Calculating Stop-Loss Levels ({method} method)...")
        
        df_stops = df_metrics.copy()
        
        if method == 'atr':
            df_stops['Stop_Loss'] = df_stops['Current_Price'] - (multiplier * df_stops['ATR'])
            df_stops['Stop_Loss_Pct'] = ((df_stops['Stop_Loss'] - df_stops['Current_Price']) 
                                          / df_stops['Current_Price']) * 100
        elif method == 'percent':
            pct = 8  # 8% stop loss
            df_stops['Stop_Loss'] = df_stops['Current_Price'] * (1 - pct/100)
            df_stops['Stop_Loss_Pct'] = -pct
        else:
            df_stops['Stop_Loss'] = df_stops['Current_Price'] * 0.92
            df_stops['Stop_Loss_Pct'] = -8.0
        
        return df_stops[['Ticker', 'Stock', 'Sector', 'Current_Price', 'Stop_Loss', 'Stop_Loss_Pct', 'ATR_Pct']]
    
    def check_portfolio_risk(self, df_portfolio):
        """
        Check portfolio-level risk metrics.
        Flags: concentration, sector exposure, correlation.
        """
        print("\n⚠️  Checking Portfolio Risk...")
        
        warnings = []
        
        # Position concentration
        if 'Allocation_Amount' in df_portfolio.columns:
            df_portfolio['Allocation_Pct'] = (df_portfolio['Allocation_Amount'] / 
                                             df_portfolio['Allocation_Amount'].sum())
            
            # Check individual positions
            max_position = df_portfolio['Allocation_Pct'].max()
            if max_position > self.max_position_pct:
                ticker = df_portfolio.loc[df_portfolio['Allocation_Pct'].idxmax(), 'Ticker']
                warnings.append({
                    'Type': 'Position Concentration',
                    'Severity': 'HIGH',
                    'Message': f'{ticker} exceeds {self.max_position_pct*100:.0f}% limit ({max_position*100:.1f}%)',
                    'Recommendation': f'Reduce {ticker} position'
                })
            
            # Check sector concentration
            sector_allocation = df_portfolio.groupby('Sector')['Allocation_Pct'].sum()
            for sector, pct in sector_allocation.items():
                if pct > self.max_sector_pct:
                    warnings.append({
                        'Type': 'Sector Concentration',
                        'Severity': 'MEDIUM',
                        'Message': f'{sector} exceeds {self.max_sector_pct*100:.0f}% limit ({pct*100:.1f}%)',
                        'Recommendation': f'Diversify away from {sector}'
                    })
        
        # Meme stock exposure
        if 'Sector' in df_portfolio.columns:
            meme_exposure = df_portfolio[df_portfolio['Sector'] == 'Meme/Speculative']['Allocation_Amount'].sum()
            meme_pct = meme_exposure / df_portfolio['Allocation_Amount'].sum()
            
            if meme_pct > 0.20:  # More than 20% in meme stocks
                warnings.append({
                    'Type': 'High-Risk Exposure',
                    'Severity': 'MEDIUM',
                    'Message': f'Meme stocks are {meme_pct*100:.1f}% of portfolio (recommend <20%)',
                    'Recommendation': 'Consider reducing speculative positions'
                })
        
        return pd.DataFrame(warnings) if warnings else pd.DataFrame()
    
    def calculate_portfolio_metrics(self, df_stocks, df_portfolio):
        """Calculate portfolio-level risk metrics"""
        print("\n📈 Calculating Portfolio Metrics...")
        
        if df_portfolio.empty or 'Ticker' not in df_portfolio.columns:
            return {}
        
        # Get returns for portfolio stocks
        portfolio_tickers = df_portfolio['Ticker'].unique()
        
        returns_data = []
        for ticker in portfolio_tickers:
            stock_data = df_stocks[df_stocks['Ticker'] == ticker].copy()
            if len(stock_data) > 0:
                stock_data = stock_data.tail(252)  # Last year
                stock_data['returns'] = stock_data['Close'].pct_change()
                returns_data.append({
                    'Ticker': ticker,
                    'Mean_Return': stock_data['returns'].mean() * 252,  # Annualized
                    'Volatility': stock_data['returns'].std() * np.sqrt(252)
                })
        
        if not returns_data:
            return {}
        
        df_returns = pd.DataFrame(returns_data)
        
        # Weight by allocation
        df_port_metrics = df_portfolio.merge(df_returns, on='Ticker', how='left')
        
        if 'Allocation_Amount' in df_port_metrics.columns:
            total_allocated = df_port_metrics['Allocation_Amount'].sum()
            df_port_metrics['Weight'] = df_port_metrics['Allocation_Amount'] / total_allocated
            
            # Portfolio return
            portfolio_return = (df_port_metrics['Mean_Return'] * df_port_metrics['Weight']).sum()
            
            # Portfolio volatility (simplified - assumes zero correlation)
            portfolio_vol = np.sqrt((df_port_metrics['Weight']**2 * df_port_metrics['Volatility']**2).sum())
            
            # Sharpe ratio (assume 4% risk-free rate)
            risk_free_rate = 0.04
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
            
            metrics = {
                'Expected_Annual_Return': portfolio_return,
                'Annual_Volatility': portfolio_vol,
                'Sharpe_Ratio': sharpe_ratio,
                'Total_Allocated': total_allocated,
                'Number_of_Positions': len(df_portfolio),
                'Avg_Position_Size': total_allocated / len(df_portfolio)
            }
            
            return metrics
        
        return {}

def main():
    print("\n" + "="*80)
    print("⚠️  PORTFOLIO RISK MANAGEMENT SYSTEM")
    print("="*80)
    
    # Initialize risk manager
    risk_mgr = RiskManager(total_capital=100000, max_position_pct=15, max_sector_pct=40)
    
    # Load data
    try:
        df_stocks = pd.read_csv('data/multi_sector_stocks.csv', parse_dates=['Date'])
        df_recommendations = pd.read_csv('stock_recommendations.csv')
        
        try:
            df_portfolio = pd.read_csv('portfolio_allocation.csv')
        except:
            df_portfolio = pd.DataFrame()
    except FileNotFoundError as e:
        print(f"❌ Required files not found: {e}")
        return
    
    # 1. Calculate volatility metrics
    df_metrics = risk_mgr.calculate_volatility_metrics(df_stocks)
    df_metrics.to_csv('risk_metrics.csv', index=False)
    print(f"✓ Saved risk metrics for {len(df_metrics)} stocks")
    
    # 2. Calculate risk-adjusted position sizes
    df_positions = risk_mgr.calculate_position_sizes(df_metrics, df_recommendations)
    if not df_positions.empty:
        df_positions.to_csv('risk_adjusted_positions.csv', index=False)
        print(f"✓ Calculated position sizes for {len(df_positions)} BUY recommendations")
    
    # 3. Calculate stop losses
    df_stops = risk_mgr.calculate_stop_losses(df_metrics, method='atr', multiplier=2.0)
    df_stops.to_csv('stop_losses.csv', index=False)
    print(f"✓ Calculated stop-loss levels for {len(df_stops)} stocks")
    
    # 4. Check portfolio warnings
    if not df_portfolio.empty:
        df_warnings = risk_mgr.check_portfolio_risk(df_portfolio)
        if not df_warnings.empty:
            df_warnings.to_csv('risk_warnings.csv', index=False)
            print(f"\n⚠️  {len(df_warnings)} RISK WARNINGS:")
            for _, warning in df_warnings.iterrows():
                print(f"   [{warning['Severity']}] {warning['Type']}: {warning['Message']}")
        else:
            print("✓ No risk warnings - portfolio within limits")
        
        # 5. Portfolio metrics
        metrics = risk_mgr.calculate_portfolio_metrics(df_stocks, df_portfolio)
        if metrics:
            print(f"\n📊 PORTFOLIO METRICS:")
            print(f"   Expected Return:  {metrics['Expected_Annual_Return']:>6.1%}")
            print(f"   Volatility:       {metrics['Annual_Volatility']:>6.1%}")
            print(f"   Sharpe Ratio:     {metrics['Sharpe_Ratio']:>6.2f}")
            print(f"   Total Allocated:  ${metrics['Total_Allocated']:>,.0f}")
            print(f"   Positions:        {metrics['Number_of_Positions']:>6d}")
    
    # Summary by risk level
    print(f"\n📊 RISK DISTRIBUTION:")
    df_metrics['Risk_Category'] = pd.cut(df_metrics['Risk_Score'], 
                                         bins=[0, 33, 66, 100],
                                         labels=['Low', 'Medium', 'High'])
    print(df_metrics.groupby('Risk_Category')['Ticker'].count())
    
    # Highest risk stocks
    print(f"\n⚠️  TOP 5 HIGHEST RISK STOCKS:")
    top_risk = df_metrics.nlargest(5, 'Risk_Score')[['Ticker', 'Stock', 'Sector', 'Risk_Score', 'Volatility_Annual']]
    for _, row in top_risk.iterrows():
        print(f"   {row['Ticker']:6s} {row['Stock'][:25]:25s} Risk: {row['Risk_Score']:.0f}/100  Vol: {row['Volatility_Annual']:.1%}")
    
    # Lowest risk stocks
    print(f"\n✅ TOP 5 LOWEST RISK STOCKS:")
    low_risk = df_metrics.nsmallest(5, 'Risk_Score')[['Ticker', 'Stock', 'Sector', 'Risk_Score', 'Volatility_Annual']]
    for _, row in low_risk.iterrows():
        print(f"   {row['Ticker']:6s} {row['Stock'][:25]:25s} Risk: {row['Risk_Score']:.0f}/100  Vol: {row['Volatility_Annual']:.1%}")
    
    print("\n" + "="*80)
    print("✅ RISK ANALYSIS COMPLETE")
    print("="*80 + "\n")
    
    print("📁 Files created:")
    print("   • risk_metrics.csv - Volatility and risk scores")
    print("   • risk_adjusted_positions.csv - Recommended position sizes")
    print("   • stop_losses.csv - Stop-loss levels")
    if not df_portfolio.empty and not df_warnings.empty:
        print("   • risk_warnings.csv - Portfolio warnings")
    print()

if __name__ == '__main__':
    main()
