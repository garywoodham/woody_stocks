#!/usr/bin/env python3
"""
Track prediction accuracy by comparing predictions vs actual outcomes.
Maintains historical log of predictions and evaluates performance over time.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class PerformanceTracker:
    """Track and evaluate prediction accuracy over time"""
    
    def __init__(self):
        self.predictions_log = 'data/predictions_log.csv'
        self.performance_summary = 'data/performance_summary.csv'
        
    def log_predictions(self):
        """
        Log today's predictions for future evaluation.
        Appends current predictions with timestamp.
        """
        print("\n" + "="*80)
        print("📝 LOGGING PREDICTIONS FOR PERFORMANCE TRACKING")
        print("="*80 + "\n")
        
        # Load current predictions
        try:
            df_pred = pd.read_csv('predictions_refined.csv')
        except FileNotFoundError:
            print("❌ No predictions found. Run predict_refined.py first.")
            return
        
        # Load current prices
        try:
            df_stocks = pd.read_csv('data/multi_sector_stocks.csv', parse_dates=['Date'])
        except FileNotFoundError:
            print("❌ No stock data found.")
            return
        
        # Get latest price for each stock
        latest_prices = df_stocks.groupby('Ticker').last().reset_index()
        
        # Merge predictions with current prices
        df_log = df_pred.merge(
            latest_prices[['Ticker', 'Close', 'Date']],
            on='Ticker',
            how='left'
        )
        
        # Add prediction date
        df_log['prediction_date'] = datetime.now().strftime('%Y-%m-%d')
        df_log['entry_price'] = df_log['Close']
        
        # Calculate target prices based on predictions
        for horizon in [1, 5, 21]:
            dir_col = f'd{horizon}_Direction'
            prob_col = f'd{horizon}_Prob_Up'
            
            if dir_col in df_log.columns and prob_col in df_log.columns:
                # Expected return based on probability
                df_log[f'd{horizon}_target_return'] = np.where(
                    df_log[dir_col] == 'UP ↑',
                    df_log[prob_col] * 0.05,  # Assume 5% upside if correct
                    -(1 - df_log[prob_col]) * 0.05  # Assume 5% downside if wrong
                )
                df_log[f'd{horizon}_target_price'] = df_log['entry_price'] * (1 + df_log[f'd{horizon}_target_return'])
        
        # Select columns to log
        log_cols = ['prediction_date', 'Ticker', 'Stock', 'Sector', 'entry_price']
        
        for horizon in [1, 5, 21]:
            dir_col = f'd{horizon}_Direction'
            prob_col = f'd{horizon}_Prob_Up'
            conf_col = f'd{horizon}_Confidence'
            
            if all(col in df_log.columns for col in [dir_col, prob_col, conf_col]):
                log_cols.extend([dir_col, prob_col, conf_col, f'd{horizon}_target_price'])
        
        df_to_log = df_log[log_cols]
        
        # Append to log file
        if os.path.exists(self.predictions_log):
            df_existing = pd.read_csv(self.predictions_log)
            
            # Check if today already logged
            if datetime.now().strftime('%Y-%m-%d') in df_existing['prediction_date'].values:
                print(f"⚠️  Predictions already logged for today")
                response = input("Overwrite? (y/n): ").lower()
                if response != 'y':
                    print("Skipping log.")
                    return
                # Remove today's entries
                df_existing = df_existing[df_existing['prediction_date'] != datetime.now().strftime('%Y-%m-%d')]
            
            df_combined = pd.concat([df_existing, df_to_log], ignore_index=True)
        else:
            df_combined = df_to_log
            print("✨ Creating new predictions log")
        
        # Save
        df_combined.to_csv(self.predictions_log, index=False)
        
        print(f"✓ Logged {len(df_to_log)} predictions")
        print(f"  Total predictions in log: {len(df_combined)}")
        print(f"  Date range: {df_combined['prediction_date'].min()} to {df_combined['prediction_date'].max()}")
        print()
    
    def evaluate_predictions(self):
        """
        Evaluate past predictions against actual outcomes.
        Updates performance summary with accuracy metrics.
        """
        print("\n" + "="*80)
        print("📊 EVALUATING PREDICTION PERFORMANCE")
        print("="*80 + "\n")
        
        if not os.path.exists(self.predictions_log):
            print("❌ No predictions log found. Run log_predictions() first.")
            return
        
        # Load predictions log
        df_log = pd.read_csv(self.predictions_log)
        df_log['prediction_date'] = pd.to_datetime(df_log['prediction_date'])
        
        # Load current stock data
        df_stocks = pd.read_csv('data/multi_sector_stocks.csv', parse_dates=['Date'])
        df_stocks = df_stocks.sort_values('Date')
        
        results = []
        
        for horizon in [1, 5, 21]:
            dir_col = f'd{horizon}_Direction'
            prob_col = f'd{horizon}_Prob_Up'
            target_col = f'd{horizon}_target_price'
            
            if dir_col not in df_log.columns:
                continue
            
            print(f"\n🎯 Evaluating {horizon}-day predictions...")
            
            # Filter predictions that are old enough to evaluate
            cutoff_date = datetime.now() - timedelta(days=horizon)
            df_eval = df_log[df_log['prediction_date'] <= cutoff_date].copy()
            
            if df_eval.empty:
                print(f"   No predictions old enough to evaluate (need {horizon}+ days)")
                continue
            
            print(f"   Evaluating {len(df_eval)} predictions from {len(df_eval['prediction_date'].unique())} dates")
            
            # Get actual outcomes
            evaluated = []
            for _, pred in df_eval.iterrows():
                ticker = pred['Ticker']
                pred_date = pred['prediction_date']
                entry_price = pred['entry_price']
                
                # Find actual price after horizon days
                stock_data = df_stocks[df_stocks['Ticker'] == ticker].copy()
                stock_data = stock_data[stock_data['Date'] >= pred_date]
                
                if len(stock_data) >= horizon:
                    actual_price = stock_data.iloc[horizon]['Close']
                    actual_return = (actual_price - entry_price) / entry_price
                    actual_direction = 'UP ↑' if actual_return > 0 else 'DOWN ↓'
                    
                    predicted_direction = pred[dir_col]
                    predicted_prob = pred[prob_col]
                    
                    correct = (predicted_direction == actual_direction)
                    
                    evaluated.append({
                        'ticker': ticker,
                        'sector': pred['Sector'],
                        'prediction_date': pred_date,
                        'horizon': horizon,
                        'entry_price': entry_price,
                        'actual_price': actual_price,
                        'actual_return': actual_return,
                        'predicted_direction': predicted_direction,
                        'actual_direction': actual_direction,
                        'predicted_prob': predicted_prob,
                        'correct': correct,
                        'confidence': pred.get(f'd{horizon}_Confidence', 0)
                    })
            
            if not evaluated:
                print(f"   Could not evaluate any predictions (insufficient data)")
                continue
            
            df_results = pd.DataFrame(evaluated)
            
            # Calculate metrics
            accuracy = df_results['correct'].mean()
            up_predictions = df_results[df_results['predicted_direction'] == 'UP ↑']
            down_predictions = df_results[df_results['predicted_direction'] == 'DOWN ↓']
            
            up_accuracy = up_predictions['correct'].mean() if len(up_predictions) > 0 else 0
            down_accuracy = down_predictions['correct'].mean() if len(down_predictions) > 0 else 0
            
            avg_return = df_results['actual_return'].mean()
            avg_return_when_correct = df_results[df_results['correct']]['actual_return'].mean()
            avg_return_when_wrong = df_results[~df_results['correct']]['actual_return'].mean()
            
            # High confidence predictions
            high_conf = df_results[df_results['confidence'] > 0.15]
            high_conf_accuracy = high_conf['correct'].mean() if len(high_conf) > 0 else 0
            
            print(f"\n   📈 Results:")
            print(f"      Overall Accuracy:     {accuracy:.1%} ({df_results['correct'].sum()}/{len(df_results)})")
            print(f"      UP Accuracy:          {up_accuracy:.1%} ({len(up_predictions)} predictions)")
            print(f"      DOWN Accuracy:        {down_accuracy:.1%} ({len(down_predictions)} predictions)")
            print(f"      High Confidence Acc:  {high_conf_accuracy:.1%} ({len(high_conf)} predictions)")
            print(f"      Avg Return:           {avg_return:>+6.2%}")
            print(f"      Avg Return (correct): {avg_return_when_correct:>+6.2%}")
            print(f"      Avg Return (wrong):   {avg_return_when_wrong:>+6.2%}")
            
            # Best/worst performers
            best = df_results.nlargest(3, 'actual_return')[['ticker', 'actual_return', 'correct']]
            worst = df_results.nsmallest(3, 'actual_return')[['ticker', 'actual_return', 'correct']]
            
            print(f"\n   🏆 Best performers:")
            for _, row in best.iterrows():
                emoji = "✅" if row['correct'] else "❌"
                print(f"      {row['ticker']:6s} {row['actual_return']:>+6.2%} {emoji}")
            
            print(f"\n   📉 Worst performers:")
            for _, row in worst.iterrows():
                emoji = "✅" if row['correct'] else "❌"
                print(f"      {row['ticker']:6s} {row['actual_return']:>+6.2%} {emoji}")
            
            # By sector
            sector_performance = df_results.groupby('sector').agg({
                'correct': 'mean',
                'actual_return': 'mean',
                'ticker': 'count'
            }).round(3)
            sector_performance.columns = ['accuracy', 'avg_return', 'count']
            
            print(f"\n   🎯 By Sector:")
            for sector, row in sector_performance.iterrows():
                print(f"      {sector:20s} Acc: {row['accuracy']:.1%}, Ret: {row['avg_return']:>+6.2%} ({int(row['count'])} pred)")
            
            # Store for summary
            results.append({
                'evaluation_date': datetime.now().strftime('%Y-%m-%d'),
                'horizon': f'{horizon}d',
                'total_predictions': len(df_results),
                'accuracy': accuracy,
                'up_accuracy': up_accuracy,
                'down_accuracy': down_accuracy,
                'high_conf_accuracy': high_conf_accuracy,
                'avg_return': avg_return,
                'avg_return_correct': avg_return_when_correct,
                'avg_return_wrong': avg_return_when_wrong
            })
        
        if results:
            # Save performance summary
            df_summary = pd.DataFrame(results)
            
            if os.path.exists(self.performance_summary):
                df_existing = pd.read_csv(self.performance_summary)
                df_summary = pd.concat([df_existing, df_summary], ignore_index=True)
            
            df_summary.to_csv(self.performance_summary, index=False)
            print(f"\n✓ Saved performance summary to {self.performance_summary}")
        
        print("\n" + "="*80)
        print("✅ EVALUATION COMPLETE")
        print("="*80 + "\n")

def main():
    import sys
    
    tracker = PerformanceTracker()
    
    # Check for command line argument
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        # Auto-run both by default
        choice = '3'
    
    if choice in ['1', '3']:
        tracker.log_predictions()
    
    if choice in ['2', '3']:
        tracker.evaluate_predictions()

if __name__ == '__main__':
    main()
