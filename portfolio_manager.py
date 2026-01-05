#!/usr/bin/env python3
"""
Portfolio Management System
Handles position sizing, diversification, and risk management for stock recommendations.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PortfolioManager:
    """
    Manage portfolio allocation based on stock recommendations.
    """
    
    def __init__(self, total_capital=100000, max_position_pct=0.15, 
                 max_sector_pct=0.35, min_score=0.05, cash_reserve=0.10):
        """
        Initialize portfolio manager.
        
        Args:
            total_capital: Total capital available for investment
            max_position_pct: Maximum % of capital in single stock (0.15 = 15%)
            max_sector_pct: Maximum % of capital in single sector (0.35 = 35%)
            min_score: Minimum recommendation score for allocation
            cash_reserve: Minimum cash to keep (0.10 = 10%)
        """
        self.total_capital = total_capital
        self.max_position_pct = max_position_pct
        self.max_sector_pct = max_sector_pct
        self.min_score = min_score
        self.cash_reserve = cash_reserve
        self.available_capital = total_capital * (1 - cash_reserve)
        
    def calculate_position_size(self, score, strength, consensus):
        """
        Calculate position size based on recommendation quality.
        
        Args:
            score: Recommendation score (-1 to 1)
            strength: Strength level (Strong/Moderate/Weak)
            consensus: Agreement across horizons (e.g., "3/3 UP")
            
        Returns:
            Allocation percentage (0-1)
        """
        # Base allocation on score (normalized to 0-1 range)
        base_allocation = min(1.0, abs(score) * 5)  # Score of 0.2 = 100% base
        
        # Adjust for strength
        strength_multiplier = {
            'Strong': 1.2,
            'Moderate': 1.0,
            'Weak': 0.7,
            'Neutral': 0.5
        }.get(strength, 1.0)
        
        # Adjust for consensus (extract UP count)
        try:
            up_count = int(consensus.split('/')[0])
            consensus_multiplier = {3: 1.2, 2: 1.0, 1: 0.8, 0: 0.6}.get(up_count, 1.0)
        except:
            consensus_multiplier = 1.0
        
        # Calculate final allocation
        allocation = base_allocation * strength_multiplier * consensus_multiplier
        
        # Cap at maximum position size
        return min(allocation * self.max_position_pct, self.max_position_pct)
    
    def generate_portfolio(self, df_recommendations, df_prices):
        """
        Generate optimal portfolio allocation.
        
        Args:
            df_recommendations: DataFrame with stock recommendations
            df_prices: DataFrame with current stock prices
            
        Returns:
            DataFrame with portfolio allocations
        """
        # Filter only BUY recommendations above minimum score
        buys = df_recommendations[
            (df_recommendations['Recommendation'] == 'BUY') & 
            (df_recommendations['Score'] >= self.min_score)
        ].copy()
        
        if len(buys) == 0:
            print("⚠️  No BUY recommendations meet criteria")
            return pd.DataFrame()
        
        # Calculate position sizes
        buys['Position_Pct'] = buys.apply(
            lambda row: self.calculate_position_size(
                row['Score'], row['Strength'], row['Consensus']
            ), axis=1
        )
        
        # Normalize to fit within available capital
        total_allocation = buys['Position_Pct'].sum()
        if total_allocation > (1 - self.cash_reserve):
            buys['Position_Pct'] = buys['Position_Pct'] / total_allocation * (1 - self.cash_reserve)
        
        # Check sector constraints
        sector_allocations = buys.groupby('Sector')['Position_Pct'].sum()
        over_allocated_sectors = sector_allocations[sector_allocations > self.max_sector_pct]
        
        # Adjust if any sector is over-allocated
        for sector in over_allocated_sectors.index:
            sector_mask = buys['Sector'] == sector
            sector_total = buys.loc[sector_mask, 'Position_Pct'].sum()
            scale_factor = self.max_sector_pct / sector_total
            buys.loc[sector_mask, 'Position_Pct'] *= scale_factor
        
        # Calculate dollar amounts and shares
        buys['Allocation_Amount'] = buys['Position_Pct'] * self.total_capital
        buys['Current_Price'] = buys['Latest_Price']
        buys['Shares'] = (buys['Allocation_Amount'] / buys['Current_Price']).astype(int)
        buys['Actual_Amount'] = buys['Shares'] * buys['Current_Price']
        buys['Actual_Pct'] = buys['Actual_Amount'] / self.total_capital
        
        # Sort by allocation
        buys = buys.sort_values('Actual_Pct', ascending=False)
        
        return buys[[
            'Stock', 'Ticker', 'Sector', 'Recommendation', 'Score', 'Strength', 
            'Consensus', 'Position_Pct', 'Actual_Pct', 'Allocation_Amount', 
            'Actual_Amount', 'Current_Price', 'Shares'
        ]]
    
    def calculate_portfolio_metrics(self, portfolio):
        """
        Calculate portfolio-level metrics.
        
        Args:
            portfolio: DataFrame with portfolio allocations
            
        Returns:
            Dictionary with portfolio metrics
        """
        if len(portfolio) == 0:
            return {}
        
        total_invested = portfolio['Actual_Amount'].sum()
        cash_remaining = self.total_capital - total_invested
        
        # Sector diversification
        sector_counts = portfolio['Sector'].value_counts()
        sector_allocations = portfolio.groupby('Sector')['Actual_Pct'].sum()
        
        # Risk metrics
        avg_score = portfolio['Score'].mean()
        weighted_score = (portfolio['Score'] * portfolio['Actual_Pct']).sum() / portfolio['Actual_Pct'].sum()
        
        return {
            'Total_Capital': self.total_capital,
            'Total_Invested': total_invested,
            'Investment_Rate': total_invested / self.total_capital,
            'Cash_Reserve': cash_remaining,
            'Cash_Reserve_Pct': cash_remaining / self.total_capital,
            'Number_of_Positions': len(portfolio),
            'Number_of_Sectors': len(sector_counts),
            'Largest_Position_Pct': portfolio['Actual_Pct'].max(),
            'Smallest_Position_Pct': portfolio['Actual_Pct'].min(),
            'Avg_Position_Pct': portfolio['Actual_Pct'].mean(),
            'Largest_Sector': sector_allocations.idxmax(),
            'Largest_Sector_Pct': sector_allocations.max(),
            'Avg_Score': avg_score,
            'Weighted_Avg_Score': weighted_score,
            'Sector_Distribution': sector_allocations.to_dict()
        }

def main():
    print("\n" + "="*80)
    print("PORTFOLIO MANAGEMENT SYSTEM")
    print("="*80 + "\n")
    
    # Load recommendations
    print("Loading recommendations...")
    df_recommendations = pd.read_csv('stock_recommendations.csv')
    print(f"✓ Loaded {len(df_recommendations)} stock recommendations\n")
    
    # Initialize portfolio manager
    capital = 100000  # $100k portfolio
    manager = PortfolioManager(
        total_capital=capital,
        max_position_pct=0.15,      # Max 15% per stock
        max_sector_pct=0.35,         # Max 35% per sector
        min_score=0.05,              # Only BUY with score > 0.05
        cash_reserve=0.10            # Keep 10% cash
    )
    
    print(f"Portfolio Configuration:")
    print(f"  Total Capital:        ${capital:,.0f}")
    print(f"  Max Position:         {manager.max_position_pct:.1%} per stock")
    print(f"  Max Sector:           {manager.max_sector_pct:.1%} per sector")
    print(f"  Min BUY Score:        {manager.min_score:.2f}")
    print(f"  Cash Reserve:         {manager.cash_reserve:.1%}")
    print()
    
    # Generate portfolio
    print("Generating optimal portfolio allocation...")
    print("-" * 80)
    
    portfolio = manager.generate_portfolio(df_recommendations, None)
    
    if len(portfolio) == 0:
        print("❌ No stocks meet BUY criteria")
        return
    
    # Display portfolio
    print(f"\n{'Stock':<25} {'Ticker':<8} {'Sector':<12} {'Score':>7} {'Alloc %':>8} {'Amount':>12} {'Shares':>8}")
    print("-" * 80)
    
    for _, row in portfolio.iterrows():
        print(f"{row['Stock']:<25} {row['Ticker']:<8} {row['Sector']:<12} "
              f"{row['Score']:>7.4f} {row['Actual_Pct']:>7.2%} "
              f"${row['Actual_Amount']:>11,.0f} {row['Shares']:>8,.0f}")
    
    # Calculate and display metrics
    metrics = manager.calculate_portfolio_metrics(portfolio)
    
    print("\n" + "="*80)
    print("PORTFOLIO SUMMARY")
    print("="*80 + "\n")
    
    print(f"Capital Allocation:")
    print(f"  Total Capital:        ${metrics['Total_Capital']:>12,.0f}")
    print(f"  Total Invested:       ${metrics['Total_Invested']:>12,.0f}  ({metrics['Investment_Rate']:.1%})")
    print(f"  Cash Reserve:         ${metrics['Cash_Reserve']:>12,.0f}  ({metrics['Cash_Reserve_Pct']:.1%})")
    
    print(f"\nDiversification:")
    print(f"  Number of Positions:  {metrics['Number_of_Positions']:>12}")
    print(f"  Number of Sectors:    {metrics['Number_of_Sectors']:>12}")
    print(f"  Largest Position:     {metrics['Largest_Position_Pct']:>12.2%}")
    print(f"  Average Position:     {metrics['Avg_Position_Pct']:>12.2%}")
    print(f"  Smallest Position:    {metrics['Smallest_Position_Pct']:>12.2%}")
    
    print(f"\nSector Allocation:")
    for sector, pct in sorted(metrics['Sector_Distribution'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {sector:<18}  {pct:>12.2%}")
    
    print(f"\nRisk Metrics:")
    print(f"  Average Score:        {metrics['Avg_Score']:>12.4f}")
    print(f"  Weighted Avg Score:   {metrics['Weighted_Avg_Score']:>12.4f}")
    print(f"  Largest Sector:       {metrics['Largest_Sector']:>12} ({metrics['Largest_Sector_Pct']:.1%})")
    
    # Save portfolio
    portfolio.to_csv('portfolio_allocation.csv', index=False)
    
    print("\n" + "="*80)
    print("✓ Portfolio saved to portfolio_allocation.csv")
    print("="*80 + "\n")
    
    # Generate trading instructions
    print("📋 TRADING INSTRUCTIONS")
    print("="*80 + "\n")
    
    for i, (_, row) in enumerate(portfolio.iterrows(), 1):
        print(f"{i}. BUY {row['Shares']:.0f} shares of {row['Stock']} ({row['Ticker']})")
        print(f"   → ${row['Actual_Amount']:,.0f} at ~${row['Current_Price']:.2f}/share")
        print(f"   Rationale: {row['Strength']} {row['Recommendation']} (Score: {row['Score']:.4f}, {row['Consensus']})")
        print()

if __name__ == '__main__':
    main()
