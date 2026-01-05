import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read backtest summary
try:
    df = pd.read_csv('backtest_summary.csv')
    
    print("="*80)
    print("📊 BACKTEST SUMMARY VISUALIZATION")
    print("="*80)
    
    # Overall statistics
    print(f"\nTotal Strategies Tested: {len(df)}")
    print(f"Average Return: {df['Total_Return'].mean():.2%}")
    print(f"Average Excess Return vs Buy & Hold: {df['Excess_Return'].mean():.2%}")
    print(f"Average Win Rate: {df['Win_Rate'].mean():.2%}")
    print(f"Average Sharpe Ratio: {df['Sharpe_Ratio'].mean():.2f}")
    
    # Top performers
    print("\n" + "="*80)
    print("🏆 TOP 10 BEST PERFORMING STRATEGIES")
    print("="*80)
    top_10 = df.nlargest(10, 'Total_Return')
    for i, row in top_10.iterrows():
        print(f"\n{row['Stock']} ({row['Ticker']}) - {row['Horizon']}")
        print(f"  Total Return:    {row['Total_Return']:>8.2%}")
        print(f"  Excess Return:   {row['Excess_Return']:>8.2%}")
        print(f"  Win Rate:        {row['Win_Rate']:>8.2%}")
        print(f"  Total Trades:    {row['Total_Trades']:>8.0f}")
        print(f"  Sharpe Ratio:    {row['Sharpe_Ratio']:>8.2f}")
        print(f"  Max Drawdown:    {row['Max_Drawdown']:>8.2%}")
    
    # Worst performers
    print("\n" + "="*80)
    print("⚠️  BOTTOM 5 PERFORMING STRATEGIES")
    print("="*80)
    bottom_5 = df.nsmallest(5, 'Total_Return')
    for i, row in bottom_5.iterrows():
        print(f"\n{row['Stock']} ({row['Ticker']}) - {row['Horizon']}")
        print(f"  Total Return:    {row['Total_Return']:>8.2%}")
        print(f"  Excess Return:   {row['Excess_Return']:>8.2%}")
        print(f"  Total Trades:    {row['Total_Trades']:>8.0f}")
    
    # Sector analysis
    print("\n" + "="*80)
    print("📊 PERFORMANCE BY SECTOR")
    print("="*80)
    sector_stats = df.groupby('Sector').agg({
        'Total_Return': ['mean', 'median', 'std'],
        'Excess_Return': 'mean',
        'Win_Rate': 'mean',
        'Sharpe_Ratio': 'mean',
        'Stock': 'count'
    }).round(4)
    
    for sector in df['Sector'].unique():
        sector_df = df[df['Sector'] == sector]
        print(f"\n{sector}:")
        print(f"  Avg Total Return:     {sector_df['Total_Return'].mean():>8.2%}")
        print(f"  Avg Excess Return:    {sector_df['Excess_Return'].mean():>8.2%}")
        print(f"  Avg Win Rate:         {sector_df['Win_Rate'].mean():>8.2%}")
        print(f"  Avg Sharpe Ratio:     {sector_df['Sharpe_Ratio'].mean():>8.2f}")
        print(f"  Strategies Count:     {len(sector_df):>8.0f}")
        print(f"  Best Stock:           {sector_df.nlargest(1, 'Total_Return')['Stock'].values[0]}")
        print(f"  Best Return:          {sector_df['Total_Return'].max():>8.2%}")
    
    # Strategies beating buy & hold
    beating_bh = df[df['Excess_Return'] > 0]
    print(f"\n" + "="*80)
    print(f"✅ Strategies Beating Buy & Hold: {len(beating_bh)}/{len(df)} ({len(beating_bh)/len(df)*100:.1f}%)")
    print(f"="*80)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Backtest Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Returns by Sector
    sector_returns = df.groupby('Sector')['Total_Return'].mean().sort_values(ascending=False)
    axes[0, 0].barh(sector_returns.index, sector_returns.values * 100, color='skyblue')
    axes[0, 0].set_xlabel('Average Return (%)')
    axes[0, 0].set_title('Average Return by Sector')
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # 2. Win Rate vs Return scatter
    axes[0, 1].scatter(df['Win_Rate'] * 100, df['Total_Return'] * 100, 
                      c=df['Sharpe_Ratio'], cmap='viridis', s=100, alpha=0.6)
    axes[0, 1].set_xlabel('Win Rate (%)')
    axes[0, 1].set_ylabel('Total Return (%)')
    axes[0, 1].set_title('Win Rate vs Total Return (colored by Sharpe Ratio)')
    axes[0, 1].grid(alpha=0.3)
    cbar = plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1])
    cbar.set_label('Sharpe Ratio')
    
    # 3. Excess Return distribution
    axes[1, 0].hist(df['Excess_Return'] * 100, bins=20, color='green', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Buy & Hold')
    axes[1, 0].set_xlabel('Excess Return vs Buy & Hold (%)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Excess Returns')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Risk-adjusted returns (Sharpe Ratio)
    sharpe_by_sector = df.groupby('Sector')['Sharpe_Ratio'].mean().sort_values(ascending=False)
    axes[1, 1].barh(sharpe_by_sector.index, sharpe_by_sector.values, color='coral')
    axes[1, 1].set_xlabel('Average Sharpe Ratio')
    axes[1, 1].set_title('Risk-Adjusted Returns by Sector')
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('backtest_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: backtest_analysis.png")
    
    # Summary statistics table
    print("\n" + "="*80)
    print("📈 SUMMARY STATISTICS TABLE")
    print("="*80)
    
    summary_table = pd.DataFrame({
        'Metric': [
            'Total Strategies',
            'Avg Total Return',
            'Avg Excess Return',
            'Avg Win Rate',
            'Avg Sharpe Ratio',
            'Avg Max Drawdown',
            'Best Strategy Return',
            'Worst Strategy Return',
            'Strategies Beating B&H',
            'Best Sector',
            'Worst Sector'
        ],
        'Value': [
            f"{len(df)}",
            f"{df['Total_Return'].mean():.2%}",
            f"{df['Excess_Return'].mean():.2%}",
            f"{df['Win_Rate'].mean():.2%}",
            f"{df['Sharpe_Ratio'].mean():.2f}",
            f"{df['Max_Drawdown'].mean():.2%}",
            f"{df['Total_Return'].max():.2%} ({df.loc[df['Total_Return'].idxmax(), 'Stock']})",
            f"{df['Total_Return'].min():.2%} ({df.loc[df['Total_Return'].idxmin(), 'Stock']})",
            f"{len(beating_bh)}/{len(df)} ({len(beating_bh)/len(df)*100:.1f}%)",
            f"{sector_returns.index[0]} ({sector_returns.values[0]:.2%})",
            f"{sector_returns.index[-1]} ({sector_returns.values[-1]:.2%})"
        ]
    })
    
    print(summary_table.to_string(index=False))
    
except FileNotFoundError:
    print("❌ backtest_summary.csv not found. Please run backtests first.")
except Exception as e:
    print(f"❌ Error: {str(e)}")
