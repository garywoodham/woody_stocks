#!/usr/bin/env python3
"""
Daily Signal Generator
Generates BUY/SELL/HOLD signals based on latest predictions.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SignalGenerator:
    """Generate trading signals from predictions."""
    
    def __init__(self, 
                 min_confidence=0.15,
                 min_probability_buy=0.55,
                 min_probability_sell=0.45):
        """
        Initialize signal generator with thresholds.
        
        Args:
            min_confidence: Minimum confidence for signal (0-1)
            min_probability_buy: Minimum probability for BUY signal
            min_probability_sell: Maximum probability for SELL signal
        """
        self.min_confidence = min_confidence
        self.min_probability_buy = min_probability_buy
        self.min_probability_sell = min_probability_sell
    
    def generate_signal(self, probability, confidence, horizon):
        """
        Generate trading signal for a given horizon.
        
        Returns dict with signal, strength, and reasoning.
        """
        if confidence < self.min_confidence:
            return {
                'signal': 'HOLD',
                'strength': 0,
                'reason': 'Low confidence'
            }
        
        if probability >= self.min_probability_buy:
            strength = (probability - 0.5) * confidence * 2  # 0 to 1 scale
            return {
                'signal': 'BUY',
                'strength': strength,
                'reason': f'High upside probability ({probability:.1%})'
            }
        elif probability <= self.min_probability_sell:
            strength = (0.5 - probability) * confidence * 2
            return {
                'signal': 'SELL',
                'strength': strength,
                'reason': f'High downside probability ({1-probability:.1%})'
            }
        else:
            return {
                'signal': 'HOLD',
                'strength': 0,
                'reason': 'Neutral probability'
            }
    
    def calculate_position_size(self, strength, base_amount=1000):
        """
        Calculate recommended position size based on signal strength.
        
        Args:
            strength: Signal strength (0-1)
            base_amount: Base investment amount in dollars
            
        Returns:
            Recommended position size in dollars
        """
        # Conservative sizing: max 2x base at full strength
        return base_amount * (1 + strength)


def generate_daily_signals():
    """Generate trading signals for all stocks."""
    
    print("="*80)
    print(f"📊 GENERATING DAILY TRADING SIGNALS - {datetime.now().strftime('%Y-%m-%d')}")
    print("="*80)
    
    # Load predictions
    try:
        df_predictions = pd.read_csv('predictions_summary.csv')
    except FileNotFoundError:
        print("❌ predictions_summary.csv not found. Run predict_stock.py first.")
        return
    
    # Load latest stock data
    try:
        df_stocks = pd.read_csv('data/multi_sector_stocks.csv', index_col=0, parse_dates=True)
        latest_prices = df_stocks.groupby('Ticker')['Close'].last()
    except FileNotFoundError:
        print("❌ Stock data not found.")
        return
    
    # Initialize signal generator
    signal_gen = SignalGenerator(
        min_confidence=0.15,
        min_probability_buy=0.55,
        min_probability_sell=0.45
    )
    
    # Generate signals for each stock and horizon
    signals = []
    
    for _, row in df_predictions.iterrows():
        ticker = row['Ticker']
        stock_name = row['Stock']
        sector = row['Sector']
        
        current_price = latest_prices.get(ticker, 0)
        
        # Generate signals for each horizon
        for horizon in ['1d', '7d', '30d', '90d']:
            prob = row[f'{horizon}_Prob_Up']
            conf = row[f'{horizon}_Confidence']
            acc = row[f'{horizon}_Accuracy']
            direction = row[f'{horizon}_Direction']
            
            signal_data = signal_gen.generate_signal(prob, conf, horizon)
            
            # Calculate position size
            position_size = signal_gen.calculate_position_size(signal_data['strength'])
            
            signals.append({
                'Date': datetime.now().strftime('%Y-%m-%d'),
                'Stock': stock_name,
                'Ticker': ticker,
                'Sector': sector,
                'Current_Price': current_price,
                'Horizon': horizon,
                'Signal': signal_data['signal'],
                'Signal_Strength': signal_data['strength'],
                'Reason': signal_data['reason'],
                'Probability_Up': prob,
                'Confidence': conf,
                'Model_Accuracy': acc,
                'Predicted_Direction': direction,
                'Recommended_Position': position_size
            })
    
    # Create DataFrame
    df_signals = pd.DataFrame(signals)
    
    # Save to CSV
    df_signals.to_csv('daily_signals.csv', index=False)
    
    # Print summary
    print(f"\n✅ Generated {len(df_signals)} signals")
    print(f"\nSignal Distribution:")
    
    for horizon in ['1d', '7d', '30d', '90d']:
        horizon_signals = df_signals[df_signals['Horizon'] == horizon]
        buy_count = len(horizon_signals[horizon_signals['Signal'] == 'BUY'])
        sell_count = len(horizon_signals[horizon_signals['Signal'] == 'SELL'])
        hold_count = len(horizon_signals[horizon_signals['Signal'] == 'HOLD'])
        
        print(f"\n{horizon} Horizon:")
        print(f"  BUY:  {buy_count:2d} signals")
        print(f"  SELL: {sell_count:2d} signals")
        print(f"  HOLD: {hold_count:2d} signals")
    
    # Top BUY signals
    print(f"\n{'='*80}")
    print("🚀 TOP 5 BUY SIGNALS (7-day horizon)")
    print(f"{'='*80}")
    
    top_buys = df_signals[
        (df_signals['Signal'] == 'BUY') & 
        (df_signals['Horizon'] == '7d')
    ].nlargest(5, 'Signal_Strength')
    
    if len(top_buys) > 0:
        for _, signal in top_buys.iterrows():
            print(f"\n{signal['Stock']} ({signal['Ticker']}) - {signal['Sector']}")
            print(f"  Current Price:     ${signal['Current_Price']:.2f}")
            print(f"  Signal Strength:   {signal['Signal_Strength']:.2%}")
            print(f"  Probability UP:    {signal['Probability_Up']:.1%}")
            print(f"  Model Confidence:  {signal['Confidence']:.1%}")
            print(f"  Model Accuracy:    {signal['Model_Accuracy']:.1%}")
            print(f"  Position Size:     ${signal['Recommended_Position']:.0f}")
            print(f"  Reason:            {signal['Reason']}")
    else:
        print("  No strong BUY signals found")
    
    # Top SELL signals
    print(f"\n{'='*80}")
    print("⚠️  TOP 5 SELL SIGNALS (7-day horizon)")
    print(f"{'='*80}")
    
    top_sells = df_signals[
        (df_signals['Signal'] == 'SELL') & 
        (df_signals['Horizon'] == '7d')
    ].nlargest(5, 'Signal_Strength')
    
    if len(top_sells) > 0:
        for _, signal in top_sells.iterrows():
            print(f"\n{signal['Stock']} ({signal['Ticker']}) - {signal['Sector']}")
            print(f"  Current Price:     ${signal['Current_Price']:.2f}")
            print(f"  Signal Strength:   {signal['Signal_Strength']:.2%}")
            print(f"  Probability DOWN:  {1-signal['Probability_Up']:.1%}")
            print(f"  Model Confidence:  {signal['Confidence']:.1%}")
            print(f"  Reason:            {signal['Reason']}")
    else:
        print("  No strong SELL signals found")
    
    # Sector summary
    print(f"\n{'='*80}")
    print("📊 SIGNALS BY SECTOR (7-day horizon)")
    print(f"{'='*80}")
    
    sector_signals = df_signals[df_signals['Horizon'] == '7d'].groupby(['Sector', 'Signal']).size().unstack(fill_value=0)
    print(sector_signals)
    
    print(f"\n✅ Signals saved to: daily_signals.csv")
    print(f"{'='*80}")
    
    return df_signals


if __name__ == "__main__":
    generate_daily_signals()
