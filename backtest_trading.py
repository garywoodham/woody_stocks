import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TradingSignalGenerator:
    """
    Generate trading signals based on model predictions with configurable thresholds.
    """
    
    def __init__(self, confidence_threshold=0.6, probability_threshold=0.55):
        """
        Initialize signal generator.
        
        Args:
            confidence_threshold: Minimum confidence level to generate signal (0-1)
            probability_threshold: Minimum probability for UP signal (0-1)
        """
        self.confidence_threshold = confidence_threshold
        self.probability_threshold = probability_threshold
        
    def generate_signal(self, probability, confidence):
        """
        Generate trading signal based on prediction probability and confidence.
        
        Returns:
            'BUY': Strong signal to buy
            'SELL': Strong signal to sell
            'HOLD': No clear signal
        """
        # Only generate signals if confidence is high enough
        if confidence < self.confidence_threshold:
            return 'HOLD'
        
        # Generate buy/sell based on probability
        if probability >= self.probability_threshold:
            return 'BUY'
        elif probability <= (1 - self.probability_threshold):
            return 'SELL'
        else:
            return 'HOLD'
    
    def calculate_position_size(self, confidence, base_position=1.0):
        """
        Calculate position size based on confidence level.
        Higher confidence = larger position (up to 2x base)
        """
        return base_position * (1 + confidence)


class BacktestEngine:
    """
    Backtest trading strategies using historical predictions.
    """
    
    def __init__(self, initial_capital=10000, commission=0.001, slippage=0.0005):
        """
        Initialize backtest engine.
        
        Args:
            initial_capital: Starting capital in dollars
            commission: Trading commission as decimal (0.001 = 0.1%)
            slippage: Price slippage as decimal (0.0005 = 0.05%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
    def run_backtest(self, df_prices, df_signals, horizon='1d'):
        """
        Run backtest simulation.
        
        Args:
            df_prices: DataFrame with Date index and Close prices
            df_signals: DataFrame with Date index and signals (BUY/SELL/HOLD)
            horizon: Trading horizon ('1d', '7d', '30d', '90d')
            
        Returns:
            Dictionary with backtest results
        """
        # Initialize tracking variables
        capital = self.initial_capital
        position = 0  # shares held
        position_value = 0
        entry_price = 0
        
        trades = []
        equity_curve = []
        
        # Get holding period from horizon
        holding_days = {'1d': 1, '7d': 7, '30d': 30, '90d': 90}[horizon]
        
        for i, (date, signal) in enumerate(df_signals.items()):
            if date not in df_prices.index:
                continue
                
            current_price = df_prices.loc[date]
            
            # Calculate current portfolio value
            portfolio_value = capital + (position * current_price if position > 0 else 0)
            equity_curve.append({
                'Date': date,
                'Portfolio_Value': portfolio_value,
                'Cash': capital,
                'Position_Value': position * current_price if position > 0 else 0
            })
            
            # Check if we should close existing position (holding period expired)
            if position > 0 and i >= holding_days:
                # Check if holding period has passed
                if len(trades) > 0:
                    last_trade_idx = df_signals.index.get_loc(trades[-1]['Entry_Date'])
                    if i - last_trade_idx >= holding_days:
                        # Close position
                        sell_price = current_price * (1 - self.slippage)
                        sell_value = position * sell_price
                        commission_cost = sell_value * self.commission
                        capital += sell_value - commission_cost
                        
                        # Record trade
                        trade_return = (sell_price - entry_price) / entry_price
                        trades[-1].update({
                            'Exit_Date': date,
                            'Exit_Price': sell_price,
                            'Exit_Value': sell_value,
                            'Exit_Commission': commission_cost,
                            'Return': trade_return,
                            'Profit_Loss': sell_value - position_value - trades[-1]['Entry_Commission'] - commission_cost
                        })
                        
                        position = 0
                        position_value = 0
            
            # Generate new signal if no position
            if position == 0 and signal == 'BUY':
                # Buy signal - enter long position
                buy_price = current_price * (1 + self.slippage)
                shares_to_buy = int(capital * 0.95 / buy_price)  # Use 95% of capital
                
                if shares_to_buy > 0:
                    position_value = shares_to_buy * buy_price
                    commission_cost = position_value * self.commission
                    
                    if capital >= position_value + commission_cost:
                        capital -= (position_value + commission_cost)
                        position = shares_to_buy
                        entry_price = buy_price
                        
                        trades.append({
                            'Entry_Date': date,
                            'Entry_Price': buy_price,
                            'Entry_Value': position_value,
                            'Entry_Commission': commission_cost,
                            'Shares': shares_to_buy,
                            'Signal': signal
                        })
        
        # Close any remaining position at the end
        if position > 0:
            final_price = df_prices.iloc[-1]
            final_value = position * final_price
            commission_cost = final_value * self.commission
            capital += final_value - commission_cost
            
            if len(trades) > 0 and 'Exit_Date' not in trades[-1]:
                trade_return = (final_price - entry_price) / entry_price
                trades[-1].update({
                    'Exit_Date': df_prices.index[-1],
                    'Exit_Price': final_price,
                    'Exit_Value': final_value,
                    'Exit_Commission': commission_cost,
                    'Return': trade_return,
                    'Profit_Loss': final_value - position_value - trades[-1]['Entry_Commission'] - commission_cost
                })
        
        # Calculate final portfolio value
        final_value = capital
        
        # Create results dictionary
        results = {
            'trades': pd.DataFrame(trades) if trades else pd.DataFrame(),
            'equity_curve': pd.DataFrame(equity_curve),
            'final_value': final_value,
            'initial_capital': self.initial_capital,
            'total_return': (final_value - self.initial_capital) / self.initial_capital,
            'total_trades': len(trades)
        }
        
        return results
    
    def calculate_metrics(self, backtest_results, df_prices):
        """
        Calculate comprehensive performance metrics.
        """
        trades_df = backtest_results['trades']
        equity_curve = backtest_results['equity_curve']
        
        metrics = {
            'Total Return': backtest_results['total_return'],
            'Final Value': backtest_results['final_value'],
            'Initial Capital': backtest_results['initial_capital'],
            'Profit/Loss': backtest_results['final_value'] - backtest_results['initial_capital'],
            'Total Trades': backtest_results['total_trades']
        }
        
        if len(trades_df) > 0:
            # Completed trades only
            completed_trades = trades_df[trades_df['Exit_Date'].notna()]
            
            if len(completed_trades) > 0:
                # Win rate
                winning_trades = completed_trades[completed_trades['Return'] > 0]
                metrics['Win Rate'] = len(winning_trades) / len(completed_trades)
                metrics['Winning Trades'] = len(winning_trades)
                metrics['Losing Trades'] = len(completed_trades) - len(winning_trades)
                
                # Average returns
                metrics['Avg Return per Trade'] = completed_trades['Return'].mean()
                metrics['Avg Winning Return'] = winning_trades['Return'].mean() if len(winning_trades) > 0 else 0
                metrics['Avg Losing Return'] = completed_trades[completed_trades['Return'] <= 0]['Return'].mean() if len(completed_trades[completed_trades['Return'] <= 0]) > 0 else 0
                
                # Profit factor
                total_wins = completed_trades[completed_trades['Profit_Loss'] > 0]['Profit_Loss'].sum()
                total_losses = abs(completed_trades[completed_trades['Profit_Loss'] < 0]['Profit_Loss'].sum())
                metrics['Profit Factor'] = total_wins / total_losses if total_losses > 0 else float('inf')
                
                # Best and worst trades
                metrics['Best Trade'] = completed_trades['Return'].max()
                metrics['Worst Trade'] = completed_trades['Return'].min()
        
        # Calculate metrics from equity curve
        if len(equity_curve) > 1:
            equity_values = equity_curve['Portfolio_Value'].values
            
            # Maximum drawdown
            peak = equity_values[0]
            max_dd = 0
            for value in equity_values:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                if dd > max_dd:
                    max_dd = dd
            metrics['Max Drawdown'] = max_dd
            
            # Calculate daily returns
            returns = np.diff(equity_values) / equity_values[:-1]
            
            if len(returns) > 0:
                # Sharpe ratio (assuming 252 trading days, 0% risk-free rate)
                if returns.std() > 0:
                    metrics['Sharpe Ratio'] = (returns.mean() / returns.std()) * np.sqrt(252)
                else:
                    metrics['Sharpe Ratio'] = 0
                
                # Sortino ratio (downside deviation)
                downside_returns = returns[returns < 0]
                if len(downside_returns) > 0 and downside_returns.std() > 0:
                    metrics['Sortino Ratio'] = (returns.mean() / downside_returns.std()) * np.sqrt(252)
                else:
                    metrics['Sortino Ratio'] = 0
        
        # Buy and hold comparison
        buy_hold_return = (df_prices.iloc[-1] - df_prices.iloc[0]) / df_prices.iloc[0]
        metrics['Buy & Hold Return'] = buy_hold_return
        metrics['Excess Return'] = metrics['Total Return'] - buy_hold_return
        
        return metrics


def run_stock_backtest(ticker, stock_name, horizon='7d', confidence_threshold=0.1, probability_threshold=0.52):
    """
    Run backtest for a single stock using trained models.
    """
    print(f"\n{'='*80}")
    print(f"Backtesting: {stock_name} ({ticker}) - {horizon} horizon")
    print(f"{'='*80}")
    
    # Load stock data
    df_all = pd.read_csv('data/multi_sector_stocks.csv', index_col=0, parse_dates=True)
    df_stock = df_all[df_all['Ticker'] == ticker].copy()
    
    if df_stock.empty:
        print(f"No data found for {ticker}")
        return None
    
    # Load trained model
    safe_ticker = ticker.replace('.', '_')
    horizon_days = horizon.replace('d', '')
    model_path = f'models/{safe_ticker}_{horizon_days}d_model.pkl'
    
    try:
        model_data = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Model not found: {model_path}")
        return None
    
    # Prepare features (same as training)
    from predict_stock import create_features, create_targets
    
    df_featured = create_features(df_stock)
    df_featured, targets = create_targets(df_featured, horizons=[int(horizon_days)])
    df_featured = df_featured.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Make predictions on all historical data
    X = df_featured[model_data['feature_cols']].values
    scaler = model_data['scaler']
    model = model_data['model']
    
    X_scaled = scaler.transform(X)
    predictions_proba = model.predict(X_scaled)
    
    # Calculate confidence (distance from 0.5)
    confidences = np.abs(predictions_proba - 0.5) * 2
    
    # Generate trading signals
    signal_gen = TradingSignalGenerator(
        confidence_threshold=confidence_threshold,
        probability_threshold=probability_threshold
    )
    
    signals = []
    for prob, conf in zip(predictions_proba, confidences):
        signal = signal_gen.generate_signal(prob, conf)
        signals.append(signal)
    
    df_signals = pd.Series(signals, index=df_featured.index)
    
    # Run backtest
    engine = BacktestEngine(initial_capital=10000, commission=0.001, slippage=0.0005)
    results = engine.run_backtest(df_featured['Close'], df_signals, horizon=horizon)
    metrics = engine.calculate_metrics(results, df_featured['Close'])
    
    # Print results
    print(f"\n📊 BACKTEST RESULTS")
    print(f"{'-'*80}")
    print(f"Period: {df_stock.index[0].strftime('%Y-%m-%d')} to {df_stock.index[-1].strftime('%Y-%m-%d')}")
    print(f"Strategy: {horizon} holding period")
    print(f"Signal Thresholds: Confidence={confidence_threshold:.1%}, Probability={probability_threshold:.1%}")
    print()
    
    print(f"💰 PERFORMANCE")
    print(f"  Initial Capital:        ${metrics['Initial Capital']:,.2f}")
    print(f"  Final Value:            ${metrics['Final Value']:,.2f}")
    print(f"  Total Return:           {metrics['Total Return']:.2%}")
    print(f"  Profit/Loss:            ${metrics['Profit/Loss']:,.2f}")
    print()
    
    print(f"📈 COMPARISON")
    print(f"  Buy & Hold Return:      {metrics['Buy & Hold Return']:.2%}")
    print(f"  Strategy Return:        {metrics['Total Return']:.2%}")
    print(f"  Excess Return:          {metrics['Excess Return']:.2%}")
    print()
    
    if metrics['Total Trades'] > 0:
        print(f"🎯 TRADE STATISTICS")
        print(f"  Total Trades:           {metrics['Total Trades']}")
        print(f"  Winning Trades:         {metrics.get('Winning Trades', 0)}")
        print(f"  Losing Trades:          {metrics.get('Losing Trades', 0)}")
        print(f"  Win Rate:               {metrics.get('Win Rate', 0):.2%}")
        print(f"  Avg Return per Trade:   {metrics.get('Avg Return per Trade', 0):.2%}")
        print(f"  Best Trade:             {metrics.get('Best Trade', 0):.2%}")
        print(f"  Worst Trade:            {metrics.get('Worst Trade', 0):.2%}")
        print(f"  Profit Factor:          {metrics.get('Profit Factor', 0):.2f}")
        print()
        
        print(f"⚠️  RISK METRICS")
        print(f"  Max Drawdown:           {metrics.get('Max Drawdown', 0):.2%}")
        print(f"  Sharpe Ratio:           {metrics.get('Sharpe Ratio', 0):.2f}")
        print(f"  Sortino Ratio:          {metrics.get('Sortino Ratio', 0):.2f}")
    else:
        print("⚠️  No trades generated with current thresholds")
    
    return {
        'ticker': ticker,
        'stock_name': stock_name,
        'horizon': horizon,
        'results': results,
        'metrics': metrics
    }


def run_all_backtests(horizons=['7d'], top_n=None):
    """
    Run backtests for all stocks and generate comparison report.
    """
    print("\n" + "="*80)
    print("🚀 RUNNING COMPREHENSIVE BACKTESTS")
    print("="*80)
    
    # Load predictions to get stock list
    df_predictions = pd.read_csv('predictions_summary.csv')
    
    if top_n:
        df_predictions = df_predictions.head(top_n)
    
    all_results = []
    
    for horizon in horizons:
        print(f"\n{'='*80}")
        print(f"Testing {horizon} Horizon")
        print(f"{'='*80}")
        
        for _, row in df_predictions.iterrows():
            result = run_stock_backtest(
                row['Ticker'], 
                row['Stock'], 
                horizon=horizon,
                confidence_threshold=0.05,  # Low threshold to get more trades
                probability_threshold=0.52
            )
            
            if result:
                result['sector'] = row['Sector']
                all_results.append(result)
    
    # Create summary comparison
    if all_results:
        print("\n" + "="*80)
        print("📊 BACKTEST SUMMARY - ALL STOCKS")
        print("="*80)
        
        summary_data = []
        for result in all_results:
            metrics = result['metrics']
            summary_data.append({
                'Stock': result['stock_name'],
                'Ticker': result['ticker'],
                'Sector': result['sector'],
                'Horizon': result['horizon'],
                'Total_Return': metrics['Total Return'],
                'Buy_Hold_Return': metrics['Buy & Hold Return'],
                'Excess_Return': metrics['Excess Return'],
                'Win_Rate': metrics.get('Win Rate', 0),
                'Total_Trades': metrics['Total Trades'],
                'Sharpe_Ratio': metrics.get('Sharpe Ratio', 0),
                'Max_Drawdown': metrics.get('Max Drawdown', 0),
                'Final_Value': metrics['Final Value']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Total_Return', ascending=False)
        
        # Save to CSV
        summary_df.to_csv('backtest_summary.csv', index=False)
        print("\n✅ Summary saved to: backtest_summary.csv")
        
        # Print top performers
        print("\n🏆 TOP 5 PERFORMING STRATEGIES:")
        print("-"*80)
        for i, row in summary_df.head(5).iterrows():
            print(f"{row['Stock']} ({row['Ticker']}) - {row['Horizon']}")
            print(f"  Return: {row['Total_Return']:.2%} | Excess: {row['Excess_Return']:.2%} | " +
                  f"Trades: {row['Total_Trades']} | Win Rate: {row['Win_Rate']:.2%}")
            print()
        
        # Print by sector
        print("\n📊 PERFORMANCE BY SECTOR:")
        print("-"*80)
        for sector in summary_df['Sector'].unique():
            sector_df = summary_df[summary_df['Sector'] == sector]
            avg_return = sector_df['Total_Return'].mean()
            avg_excess = sector_df['Excess_Return'].mean()
            print(f"{sector:15} | Avg Return: {avg_return:7.2%} | Avg Excess: {avg_excess:7.2%} | " +
                  f"Stocks: {len(sector_df)}")
        
        return summary_df
    
    return None


if __name__ == "__main__":
    import sys
    
    print("="*80)
    print("🎯 STOCK TRADING SIGNAL & BACKTEST ENGINE")
    print("="*80)
    print()
    print("This system generates trading signals from AI predictions and")
    print("validates them through comprehensive backtesting.")
    print()
    print("Options:")
    print("  1. Run single stock backtest")
    print("  2. Run all stocks backtest (7-day horizon)")
    print("  3. Run all stocks backtest (all horizons)")
    print("="*80)
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nEnter choice (1-3): ")
    
    if choice == '1':
        ticker = input("Enter ticker (e.g., BARC.L): ")
        stock_name = input("Enter stock name: ")
        horizon = input("Enter horizon (1d/7d/30d/90d) [7d]: ") or '7d'
        run_stock_backtest(ticker, stock_name, horizon)
    
    elif choice == '2':
        run_all_backtests(horizons=['7d'])
    
    elif choice == '3':
        run_all_backtests(horizons=['1d', '7d', '30d', '90d'])
    
    else:
        print("Running default: All stocks, 7-day horizon")
        run_all_backtests(horizons=['7d'])
