"""
Realistic Trading Strategy with Risk Management
Designed for 52-55% accuracy predictions

Strategy Philosophy:
- Accept that predictions are only slightly better than random
- Use portfolio diversification to reduce variance
- Implement strict risk management
- Focus on risk-adjusted returns, not accuracy
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class RealisticTradingStrategy:
    """
    Trading strategy designed for realistic 52-55% accuracy
    """
    
    def __init__(self, 
                 initial_capital=10000,
                 max_position_size=0.05,  # Max 5% per position
                 max_portfolio_stocks=20,  # Diversify across 20 stocks
                 confidence_threshold=0.55,  # Only trade when confidence > 55%
                 stop_loss_pct=0.03,  # 3% stop loss
                 take_profit_pct=0.06,  # 6% take profit (2:1 reward:risk)
                 max_daily_loss=0.02,  # Stop trading if down 2% in a day
                 min_accuracy=0.50):  # Don't trade stocks with <50% historical accuracy
        
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_position_size = max_position_size
        self.max_portfolio_stocks = max_portfolio_stocks
        self.confidence_threshold = confidence_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_daily_loss = max_daily_loss
        self.min_accuracy = min_accuracy
        
        self.positions = {}  # ticker: {shares, entry_price, direction, date}
        self.daily_pnl = 0
        self.trade_history = []
        
    def filter_predictions(self, predictions_df):
        """
        Filter predictions to only trade high-quality signals
        
        Criteria:
        1. Historical accuracy >= min_accuracy (50%)
        2. Confidence > confidence_threshold (55%)
        3. Not already at max positions
        4. 21-day predictions preferred (more reliable)
        """
        df = predictions_df.copy()
        
        # Focus on 21-day predictions (most reliable)
        df['score'] = (
            df['d21_Accuracy'] * 0.5 +  # 50% weight on historical accuracy
            df['d21_Confidence'] * 0.3 +  # 30% weight on current confidence
            (df['d5_Accuracy'] * 0.2)  # 20% weight on 5-day accuracy
        )
        
        # Filter criteria
        filtered = df[
            (df['d21_Accuracy'] >= self.min_accuracy) &  # Must have decent accuracy
            (df['d21_Confidence'] >= (self.confidence_threshold - 0.5))  # Adjust threshold
        ].copy()
        
        # Sort by score
        filtered = filtered.sort_values('score', ascending=False)
        
        # Sector diversification - max 3 per sector
        sector_counts = {}
        selected = []
        
        for idx, row in filtered.iterrows():
            sector = row['Sector']
            if sector_counts.get(sector, 0) >= 3:
                continue
            
            selected.append(row)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            
            if len(selected) >= self.max_portfolio_stocks:
                break
        
        return pd.DataFrame(selected)
    
    def calculate_position_size(self, price, confidence):
        """
        Calculate position size based on Kelly Criterion (modified for safety)
        
        Kelly = (p * b - q) / b
        where:
        - p = probability of win (confidence)
        - q = probability of loss (1 - confidence)
        - b = win/loss ratio (take_profit / stop_loss = 2:1)
        
        Use half-Kelly for safety
        """
        p = confidence
        q = 1 - p
        b = self.take_profit_pct / self.stop_loss_pct  # 2:1
        
        # Kelly formula
        kelly_fraction = (p * b - q) / b
        
        # Use half-Kelly for safety, cap at max_position_size
        position_fraction = min(kelly_fraction * 0.5, self.max_position_size)
        
        # Additional safety: if confidence is low, reduce size
        if confidence < 0.55:
            position_fraction *= 0.5
        
        # Ensure positive
        position_fraction = max(position_fraction, 0)
        
        # Calculate dollar amount
        position_value = self.capital * position_fraction
        shares = int(position_value / price)
        
        return shares
    
    def enter_trade(self, ticker, price, direction, confidence, date):
        """Enter a new trade"""
        if ticker in self.positions:
            return False  # Already have a position
        
        shares = self.calculate_position_size(price, confidence)
        
        if shares == 0:
            return False
        
        cost = shares * price
        
        if cost > self.capital:
            # Not enough capital
            shares = int(self.capital / price)
            cost = shares * price
        
        if shares == 0:
            return False
        
        self.positions[ticker] = {
            'shares': shares,
            'entry_price': price,
            'direction': direction,
            'date': date,
            'stop_loss': price * (1 - self.stop_loss_pct) if direction == 'UP' else price * (1 + self.stop_loss_pct),
            'take_profit': price * (1 + self.take_profit_pct) if direction == 'UP' else price * (1 - self.take_profit_pct)
        }
        
        self.capital -= cost
        
        print(f"✅ ENTER {ticker}: {shares} shares @ ${price:.2f} ({direction})")
        print(f"   Stop Loss: ${self.positions[ticker]['stop_loss']:.2f}, Take Profit: ${self.positions[ticker]['take_profit']:.2f}")
        
        return True
    
    def check_exit(self, ticker, current_price, current_date):
        """Check if we should exit a position"""
        if ticker not in self.positions:
            return None
        
        pos = self.positions[ticker]
        direction = pos['direction']
        entry_price = pos['entry_price']
        
        # Check stop loss
        if direction == 'UP ↑':
            if current_price <= pos['stop_loss']:
                return 'STOP_LOSS'
            if current_price >= pos['take_profit']:
                return 'TAKE_PROFIT'
        else:
            if current_price >= pos['stop_loss']:
                return 'STOP_LOSS'
            if current_price <= pos['take_profit']:
                return 'TAKE_PROFIT'
        
        # Check time-based exit (21 days)
        days_held = (current_date - pos['date']).days
        if days_held >= 21:
            return 'TIME_EXIT'
        
        return None
    
    def exit_trade(self, ticker, price, reason, date):
        """Exit a trade"""
        if ticker not in self.positions:
            return
        
        pos = self.positions[ticker]
        shares = pos['shares']
        entry_price = pos['entry_price']
        
        proceeds = shares * price
        self.capital += proceeds
        
        pnl = proceeds - (shares * entry_price)
        pnl_pct = (price / entry_price - 1) * 100
        
        if pos['direction'] == 'DOWN ↓':
            pnl = -pnl
            pnl_pct = -pnl_pct
        
        self.daily_pnl += pnl
        
        self.trade_history.append({
            'ticker': ticker,
            'entry_date': pos['date'],
            'exit_date': date,
            'entry_price': entry_price,
            'exit_price': price,
            'shares': shares,
            'direction': pos['direction'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason
        })
        
        print(f"❌ EXIT {ticker}: ${price:.2f} | PnL: ${pnl:+.2f} ({pnl_pct:+.1f}%) | Reason: {reason}")
        
        del self.positions[ticker]
    
    def get_portfolio_value(self, current_prices):
        """Calculate total portfolio value"""
        cash = self.capital
        holdings_value = 0
        
        for ticker, pos in self.positions.items():
            if ticker in current_prices:
                holdings_value += pos['shares'] * current_prices[ticker]
        
        return cash + holdings_value
    
    def get_stats(self):
        """Get strategy statistics"""
        if not self.trade_history:
            return None
        
        df = pd.DataFrame(self.trade_history)
        
        total_trades = len(df)
        winning_trades = len(df[df['pnl'] > 0])
        losing_trades = len(df[df['pnl'] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_win = df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df[df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        total_pnl = df['pnl'].sum()
        total_return = (total_pnl / self.initial_capital) * 100
        
        # Sharpe-like metric (simplified)
        returns = df['pnl_pct'].values
        sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0
        
        stats = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 else float('inf'),
            'total_pnl': total_pnl,
            'total_return_pct': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': self.calculate_max_drawdown(df)
        }
        
        return stats
    
    def calculate_max_drawdown(self, trades_df):
        """Calculate maximum drawdown"""
        cumulative_pnl = trades_df['pnl'].cumsum()
        running_max = cumulative_pnl.cummax()
        drawdown = cumulative_pnl - running_max
        max_drawdown = drawdown.min()
        max_drawdown_pct = (max_drawdown / self.initial_capital) * 100
        return max_drawdown_pct
    
    def print_summary(self):
        """Print strategy summary"""
        stats = self.get_stats()
        
        if stats is None:
            print("No trades executed yet")
            return
        
        print("\n" + "="*70)
        print("REALISTIC TRADING STRATEGY - PERFORMANCE SUMMARY")
        print("="*70)
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Capital: ${self.capital:,.2f}")
        print(f"Total Return: {stats['total_return_pct']:+.2f}%")
        print(f"\nTrades:")
        print(f"  Total: {stats['total_trades']}")
        print(f"  Winners: {stats['winning_trades']} ({stats['win_rate']:.1%})")
        print(f"  Losers: {stats['losing_trades']}")
        print(f"\nPnL:")
        print(f"  Avg Win: ${stats['avg_win']:,.2f}")
        print(f"  Avg Loss: ${stats['avg_loss']:,.2f}")
        print(f"  Profit Factor: {stats['profit_factor']:.2f}")
        print(f"\nRisk Metrics:")
        print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {stats['max_drawdown']:.2f}%")
        print("="*70)


def demo_strategy():
    """
    Demonstrate the strategy with current predictions
    """
    print("\n" + "="*70)
    print("REALISTIC TRADING STRATEGY DEMO")
    print("="*70)
    
    # Load predictions
    preds = pd.read_csv('predictions_refined.csv')
    
    # Initialize strategy
    strategy = RealisticTradingStrategy(
        initial_capital=10000,
        max_position_size=0.05,
        max_portfolio_stocks=20,
        confidence_threshold=0.55,
        stop_loss_pct=0.03,
        take_profit_pct=0.06
    )
    
    print(f"\nStrategy Parameters:")
    print(f"  Initial Capital: ${strategy.initial_capital:,.2f}")
    print(f"  Max Position Size: {strategy.max_position_size:.1%}")
    print(f"  Max Stocks: {strategy.max_portfolio_stocks}")
    print(f"  Stop Loss: {strategy.stop_loss_pct:.1%}")
    print(f"  Take Profit: {strategy.take_profit_pct:.1%}")
    print(f"  Min Accuracy: {strategy.min_accuracy:.1%}")
    
    # Filter predictions
    print("\n" + "-"*70)
    print("FILTERING PREDICTIONS")
    print("-"*70)
    
    selected = strategy.filter_predictions(preds)
    
    print(f"Total predictions: {len(preds)}")
    print(f"Selected for trading: {len(selected)}")
    print(f"\nTop 10 Selected Stocks:")
    print(selected[['Ticker', 'Stock', 'Sector', 'd21_Direction', 'd21_Accuracy', 'd21_Confidence']].head(10).to_string(index=False))
    
    # Simulate entries
    print("\n" + "-"*70)
    print("SIMULATED TRADES (Using current prices)")
    print("-"*70)
    
    for _, row in selected.iterrows():
        strategy.enter_trade(
            ticker=row['Ticker'],
            price=row['Latest_Price'],
            direction=row['d21_Direction'],
            confidence=row['d21_Confidence'] + 0.5,  # Add 0.5 since it's stored as deviation
            date=pd.to_datetime(row['Latest_Date'])
        )
    
    print(f"\n💰 Remaining Capital: ${strategy.capital:,.2f}")
    print(f"📊 Active Positions: {len(strategy.positions)}")
    
    # Save selected stocks for tracking
    selected.to_csv('realistic_strategy_selections.csv', index=False)
    print(f"\n✅ Saved selections to realistic_strategy_selections.csv")


if __name__ == '__main__':
    demo_strategy()
