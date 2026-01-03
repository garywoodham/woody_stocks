#!/usr/bin/env python3
"""
Daily Report Generator
Creates a summary report of predictions, signals, and system status.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def generate_report():
    """Generate comprehensive daily report."""
    
    report_date = datetime.now().strftime('%Y-%m-%d')
    
    print("="*80)
    print(f"📄 GENERATING DAILY REPORT - {report_date}")
    print("="*80)
    
    report_lines = []
    report_lines.append(f"# Stock Prediction System - Daily Report")
    report_lines.append(f"## {report_date}\n")
    
    # Load data
    try:
        df_predictions = pd.read_csv('predictions_summary.csv')
        df_signals = pd.read_csv('daily_signals.csv')
        df_stocks = pd.read_csv('data/multi_sector_stocks.csv', index_col=0, parse_dates=True)
        
        has_backtest = os.path.exists('backtest_summary.csv')
        if has_backtest:
            df_backtest = pd.read_csv('backtest_summary.csv')
        
    except FileNotFoundError as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # System Status
    report_lines.append("## 🔧 System Status\n")
    report_lines.append(f"- **Data Updated:** {df_stocks.index.max().strftime('%Y-%m-%d')}")
    report_lines.append(f"- **Stocks Tracked:** {df_stocks['Stock'].nunique()}")
    report_lines.append(f"- **Sectors:** {df_stocks['Sector'].nunique()}")
    report_lines.append(f"- **Active Models:** {len(df_predictions) * 4} (4 horizons per stock)")
    report_lines.append(f"- **Predictions Generated:** {len(df_predictions)}")
    report_lines.append(f"- **Signals Generated:** {len(df_signals)}\n")
    
    # Market Overview
    report_lines.append("## 📊 Market Overview\n")
    
    latest_prices = df_stocks.groupby(['Sector', 'Stock', 'Ticker']).last()
    prev_prices = df_stocks.groupby(['Sector', 'Stock', 'Ticker']).apply(lambda x: x.iloc[-2] if len(x) > 1 else x.iloc[-1])
    
    price_changes = ((latest_prices['Close'] - prev_prices['Close']) / prev_prices['Close'] * 100).reset_index()
    price_changes.columns = ['Sector', 'Stock', 'Ticker', 'Change_%']
    
    report_lines.append("### Top Gainers (Last Day)")
    top_gainers = price_changes.nlargest(5, 'Change_%')
    for _, row in top_gainers.iterrows():
        report_lines.append(f"- **{row['Stock']}** ({row['Ticker']}): +{row['Change_%']:.2f}%")
    
    report_lines.append("\n### Top Losers (Last Day)")
    top_losers = price_changes.nsmallest(5, 'Change_%')
    for _, row in top_losers.iterrows():
        report_lines.append(f"- **{row['Stock']}** ({row['Ticker']}): {row['Change_%']:.2f}%")
    
    # Trading Signals
    report_lines.append("\n## 🎯 Trading Signals\n")
    
    # 7-day signals (most actionable)
    signals_7d = df_signals[df_signals['Horizon'] == '7d']
    
    buy_signals = signals_7d[signals_7d['Signal'] == 'BUY']
    sell_signals = signals_7d[signals_7d['Signal'] == 'SELL']
    hold_signals = signals_7d[signals_7d['Signal'] == 'HOLD']
    
    report_lines.append(f"### 7-Day Horizon Signals")
    report_lines.append(f"- 🟢 **BUY:** {len(buy_signals)} signals")
    report_lines.append(f"- 🔴 **SELL:** {len(sell_signals)} signals")
    report_lines.append(f"- ⚪ **HOLD:** {len(hold_signals)} signals\n")
    
    if len(buy_signals) > 0:
        report_lines.append("### 🚀 Top 5 BUY Opportunities")
        report_lines.append("| Stock | Ticker | Price | Strength | Prob UP | Confidence |")
        report_lines.append("|-------|--------|-------|----------|---------|------------|")
        
        for _, signal in buy_signals.nlargest(5, 'Signal_Strength').iterrows():
            report_lines.append(
                f"| {signal['Stock']} | {signal['Ticker']} | "
                f"${signal['Current_Price']:.2f} | {signal['Signal_Strength']:.1%} | "
                f"{signal['Probability_Up']:.1%} | {signal['Confidence']:.1%} |"
            )
    
    if len(sell_signals) > 0:
        report_lines.append("\n### ⚠️ Top 5 SELL Signals")
        report_lines.append("| Stock | Ticker | Price | Strength | Prob DOWN | Confidence |")
        report_lines.append("|-------|--------|-------|----------|-----------|------------|")
        
        for _, signal in sell_signals.nlargest(5, 'Signal_Strength').iterrows():
            report_lines.append(
                f"| {signal['Stock']} | {signal['Ticker']} | "
                f"${signal['Current_Price']:.2f} | {signal['Signal_Strength']:.1%} | "
                f"{1-signal['Probability_Up']:.1%} | {signal['Confidence']:.1%} |"
            )
    
    # Sector Analysis
    report_lines.append("\n## 🏢 Sector Analysis\n")
    
    sector_signals = signals_7d.groupby('Sector').agg({
        'Signal_Strength': 'mean',
        'Probability_Up': 'mean',
        'Confidence': 'mean'
    }).round(3)
    
    report_lines.append("### Average Metrics by Sector (7-day)")
    report_lines.append("| Sector | Avg Strength | Avg Prob UP | Avg Confidence |")
    report_lines.append("|--------|--------------|-------------|----------------|")
    
    for sector, row in sector_signals.iterrows():
        report_lines.append(
            f"| {sector} | {row['Signal_Strength']:.1%} | "
            f"{row['Probability_Up']:.1%} | {row['Confidence']:.1%} |"
        )
    
    # Backtest Performance (if available)
    if has_backtest:
        report_lines.append("\n## 📈 Backtest Performance Summary\n")
        
        report_lines.append(f"- **Average Total Return:** {df_backtest['Total_Return'].mean():.2%}")
        report_lines.append(f"- **Average Excess Return:** {df_backtest['Excess_Return'].mean():.2%}")
        report_lines.append(f"- **Average Win Rate:** {df_backtest['Win_Rate'].mean():.2%}")
        report_lines.append(f"- **Average Sharpe Ratio:** {df_backtest['Sharpe_Ratio'].mean():.2f}")
        
        best_strategy = df_backtest.loc[df_backtest['Total_Return'].idxmax()]
        report_lines.append(f"\n### 🏆 Best Strategy")
        report_lines.append(f"- **Stock:** {best_strategy['Stock']} ({best_strategy['Ticker']})")
        report_lines.append(f"- **Horizon:** {best_strategy['Horizon']}")
        report_lines.append(f"- **Return:** {best_strategy['Total_Return']:.2%}")
        report_lines.append(f"- **Win Rate:** {best_strategy['Win_Rate']:.2%}")
    
    # Model Performance Summary
    report_lines.append("\n## 🤖 Model Performance\n")
    
    for horizon in ['1d', '7d', '30d', '90d']:
        horizon_preds = df_predictions[[f'{horizon}_Accuracy']].mean()
        report_lines.append(f"- **{horizon} Average Accuracy:** {horizon_preds.values[0]:.2%}")
    
    # Footer
    report_lines.append("\n---")
    report_lines.append(f"\n*Report generated automatically on {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}*")
    report_lines.append(f"\n*View dashboard: [http://localhost:8050](http://localhost:8050)*")
    
    # Save report
    report_content = "\n".join(report_lines)
    
    with open('DAILY_REPORT.md', 'w') as f:
        f.write(report_content)
    
    print(f"\n✅ Report saved to: DAILY_REPORT.md")
    print(f"\n{'='*80}")
    print("📊 REPORT PREVIEW")
    print(f"{'='*80}\n")
    print(report_content)
    
    return report_content


if __name__ == "__main__":
    generate_report()
