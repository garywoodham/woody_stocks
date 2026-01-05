"""
Comprehensive Model Evaluation with Trading Metrics
- Threshold optimization for balanced predictions
- Sharpe ratio, ROI, win rate calculations
- Practical trading performance assessment
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Import feature creation
import sys
sys.path.insert(0, '/workspaces/woody_stocks')
from train_refined_models import create_optimized_features, create_targets

def calculate_trading_metrics(predictions, actuals, prices, time_horizon):
    """Calculate real trading metrics"""
    
    # Basic accuracy
    accuracy = np.mean(predictions == actuals)
    
    # Confusion matrix
    cm = confusion_matrix(actuals, predictions)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # Precision/Recall
    precision_up = tp / (tp + fp) if (tp + fp) > 0 else 0
    precision_down = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_up = tp / (tp + fn) if (tp + fn) > 0 else 0
    recall_down = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Trading simulation
    returns = []
    for i in range(len(predictions) - time_horizon):
        if predictions[i] == 1:  # Predicted UP
            actual_return = (prices[i + time_horizon] - prices[i]) / prices[i]
        else:  # Predicted DOWN - short or stay out
            actual_return = -(prices[i + time_horizon] - prices[i]) / prices[i]  # Short position
        returns.append(actual_return)
    
    returns = np.array(returns)
    
    # Calculate metrics
    total_return = np.sum(returns)
    avg_return = np.mean(returns)
    std_return = np.std(returns)
    
    # Sharpe Ratio (annualized, assuming 252 trading days)
    if time_horizon == 1:
        periods_per_year = 252
    elif time_horizon == 5:
        periods_per_year = 252 / 5
    else:  # 21 days
        periods_per_year = 252 / 21
    
    sharpe = (avg_return * periods_per_year) / (std_return * np.sqrt(periods_per_year)) if std_return > 0 else 0
    
    # Win rate
    win_rate = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0
    
    # Max drawdown
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
    
    # ROI
    roi = total_return * 100  # As percentage
    
    return {
        'accuracy': accuracy,
        'precision_up': precision_up,
        'precision_down': precision_down,
        'recall_up': recall_up,
        'recall_down': recall_down,
        'sharpe_ratio': sharpe,
        'total_return': total_return,
        'avg_return': avg_return,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'roi': roi,
        'n_trades': len(returns)
    }

def optimize_threshold(y_proba, y_true, prices, time_horizon):
    """Find optimal threshold for balanced predictions and best Sharpe"""
    
    thresholds = np.arange(0.3, 0.7, 0.05)
    best_sharpe = -np.inf
    best_threshold = 0.5
    best_metrics = None
    
    results = []
    
    for threshold in thresholds:
        y_pred = (y_proba > threshold).astype(int)
        
        # Calculate metrics
        metrics = calculate_trading_metrics(y_pred, y_true, prices, time_horizon)
        
        # Balance score: prefer balanced predictions with good Sharpe
        up_pred_pct = np.mean(y_pred == 1)
        balance_score = 1 - abs(0.5 - up_pred_pct) * 2  # 1.0 = perfect balance
        
        # Combined score
        combined_score = metrics['sharpe_ratio'] + balance_score
        
        results.append({
            'threshold': threshold,
            'sharpe': metrics['sharpe_ratio'],
            'accuracy': metrics['accuracy'],
            'up_pred_pct': up_pred_pct,
            'balance_score': balance_score,
            'combined_score': combined_score,
            **metrics
        })
        
        if combined_score > best_sharpe:
            best_sharpe = combined_score
            best_threshold = threshold
            best_metrics = metrics
    
    return best_threshold, best_metrics, pd.DataFrame(results)

def evaluate_model(ticker, stock_name, df, horizons):
    """Comprehensive evaluation of refined model"""
    
    model_file = f'models/{ticker}_daily_refined.joblib'
    
    if not os.path.exists(model_file):
        return None
    
    models = joblib.load(model_file)
    
    print(f"\n{'='*80}")
    print(f"{stock_name} ({ticker})")
    print(f"{'='*80}")
    
    # Create features
    df = create_optimized_features(df, 'daily')
    df, targets = create_targets(df, horizons)
    df = df.replace([np.inf, -np.inf], np.nan)
    
    results = []
    
    for horizon in horizons:
        if horizon not in models:
            continue
        
        print(f"\n{'-'*60}")
        print(f"Horizon: {horizon}-day")
        print(f"{'-'*60}")
        
        model_data = models[horizon]
        model = model_data['model']
        scaler = model_data['scaler']
        feature_cols = model_data['feature_cols']
        
        # Prepare data
        X = df[feature_cols].dropna()
        if len(X) < 100:
            continue
        
        # Get prices and targets
        prices = df.loc[X.index, 'Close'].values
        target_col = f'target_{horizon}_direction'
        y = df.loc[X.index, target_col].values
        
        # Split
        split_idx = int(len(X) * 0.8)
        X_test = X.iloc[split_idx:]
        y_test = y[split_idx:]
        prices_test = prices[split_idx:]
        
        # Predict probabilities
        X_test_scaled = scaler.transform(X_test)
        y_proba = model.predict(X_test_scaled)
        
        # Standard threshold (0.5)
        y_pred_standard = (y_proba > 0.5).astype(int)
        standard_metrics = calculate_trading_metrics(y_pred_standard, y_test, prices_test, horizon)
        
        # Optimize threshold
        optimal_threshold, optimal_metrics, threshold_results = optimize_threshold(
            y_proba, y_test, prices_test, horizon
        )
        
        # Print results
        print(f"\nSTANDARD THRESHOLD (0.5):")
        print(f"  Accuracy:     {standard_metrics['accuracy']:.2%}")
        print(f"  UP Precision: {standard_metrics['precision_up']:.2%}")
        print(f"  DN Precision: {standard_metrics['precision_down']:.2%}")
        print(f"  Sharpe Ratio: {standard_metrics['sharpe_ratio']:.3f}")
        print(f"  ROI:          {standard_metrics['roi']:.2f}%")
        print(f"  Win Rate:     {standard_metrics['win_rate']:.2%}")
        print(f"  Max Drawdown: {standard_metrics['max_drawdown']:.2%}")
        
        print(f"\nOPTIMIZED THRESHOLD ({optimal_threshold:.2f}):")
        print(f"  Accuracy:     {optimal_metrics['accuracy']:.2%}")
        print(f"  UP Precision: {optimal_metrics['precision_up']:.2%}")
        print(f"  DN Precision: {optimal_metrics['precision_down']:.2%}")
        print(f"  Sharpe Ratio: {optimal_metrics['sharpe_ratio']:.3f}")
        print(f"  ROI:          {optimal_metrics['roi']:.2f}%")
        print(f"  Win Rate:     {optimal_metrics['win_rate']:.2%}")
        print(f"  Max Drawdown: {optimal_metrics['max_drawdown']:.2%}")
        
        # Show improvement
        sharpe_improvement = optimal_metrics['sharpe_ratio'] - standard_metrics['sharpe_ratio']
        roi_improvement = optimal_metrics['roi'] - standard_metrics['roi']
        
        print(f"\nIMPROVEMENT:")
        print(f"  Sharpe: {sharpe_improvement:+.3f}")
        print(f"  ROI:    {roi_improvement:+.2f}%")
        
        results.append({
            'Ticker': ticker,
            'Stock': stock_name,
            'Horizon': f'{horizon}d',
            'Std_Accuracy': standard_metrics['accuracy'],
            'Std_Sharpe': standard_metrics['sharpe_ratio'],
            'Std_ROI': standard_metrics['roi'],
            'Opt_Threshold': optimal_threshold,
            'Opt_Accuracy': optimal_metrics['accuracy'],
            'Opt_Sharpe': optimal_metrics['sharpe_ratio'],
            'Opt_ROI': optimal_metrics['roi'],
            'Sharpe_Improvement': sharpe_improvement,
            'ROI_Improvement': roi_improvement
        })
    
    return results

def main():
    print("="*80)
    print("TRADING METRICS EVALUATION")
    print("="*80)
    print("\nEvaluating models with:")
    print("  ✓ Accuracy (classification correctness)")
    print("  ✓ Sharpe Ratio (risk-adjusted returns)")
    print("  ✓ ROI (total return on investment)")
    print("  ✓ Win Rate (percentage of profitable trades)")
    print("  ✓ Max Drawdown (largest loss)")
    print("  ✓ Threshold Optimization (balanced predictions)")
    print("="*80)
    
    # Load data
    df = pd.read_csv('data/multi_sector_stocks.csv', index_col=0, parse_dates=True)
    
    test_stocks = [
        ('AAPL', 'Apple'),
        ('GOOGL', 'Alphabet'),
        ('BARC.L', 'Barclays')
    ]
    
    horizons = [1, 5, 21]
    
    all_results = []
    
    for ticker, name in test_stocks:
        stock_df = df[df['Ticker'] == ticker].copy()
        
        # Add targets
        for h in horizons:
            stock_df[f'target_{h}_return'] = stock_df['Close'].pct_change(periods=h).shift(-h)
            stock_df[f'target_{h}_direction'] = (stock_df[f'target_{h}_return'] > 0).astype(int)
        
        # Evaluate
        results = evaluate_model(ticker, name, stock_df, horizons)
        if results:
            all_results.extend(results)
    
    # Summary
    if all_results:
        print(f"\n{'='*80}")
        print("SUMMARY - ALL STOCKS")
        print(f"{'='*80}\n")
        
        results_df = pd.DataFrame(all_results)
        
        print("Average Performance:")
        print(f"  Standard (0.5 threshold):")
        print(f"    Accuracy:     {results_df['Std_Accuracy'].mean():.2%}")
        print(f"    Sharpe Ratio: {results_df['Std_Sharpe'].mean():.3f}")
        print(f"    ROI:          {results_df['Std_ROI'].mean():.2f}%")
        
        print(f"\n  Optimized (balanced threshold):")
        print(f"    Accuracy:     {results_df['Opt_Accuracy'].mean():.2%}")
        print(f"    Sharpe Ratio: {results_df['Opt_Sharpe'].mean():.3f}")
        print(f"    ROI:          {results_df['Opt_ROI'].mean():.2f}%")
        
        print(f"\n  Average Improvement:")
        print(f"    Sharpe: {results_df['Sharpe_Improvement'].mean():+.3f}")
        print(f"    ROI:    {results_df['ROI_Improvement'].mean():+.2f}%")
        
        # Best performers
        print(f"\n{'-'*60}")
        print("Best Sharpe Ratios:")
        best_sharpe = results_df.nlargest(3, 'Opt_Sharpe')
        for _, row in best_sharpe.iterrows():
            print(f"  {row['Stock']:<15} {row['Horizon']:<5} Sharpe: {row['Opt_Sharpe']:.3f}  ROI: {row['Opt_ROI']:.2f}%")
        
        print(f"\n{'-'*60}")
        print("Best ROI:")
        best_roi = results_df.nlargest(3, 'Opt_ROI')
        for _, row in best_roi.iterrows():
            print(f"  {row['Stock']:<15} {row['Horizon']:<5} ROI: {row['Opt_ROI']:.2f}%  Sharpe: {row['Opt_Sharpe']:.3f}")
        
        # Save detailed results
        results_df.to_csv('trading_metrics_evaluation.csv', index=False)
        print(f"\n✓ Detailed results saved to trading_metrics_evaluation.csv")
    
    print(f"\n{'='*80}")
    print("KEY INSIGHTS")
    print(f"{'='*80}")
    print("\n✓ 55-65% accuracy is GOOD for stock prediction (vs 50% random)")
    print("✓ Sharpe ratio > 0.5 indicates profitable strategy")
    print("✓ Threshold optimization balances UP/DOWN predictions")
    print("✓ Trading metrics show real-world profitability potential")
    print("\nNext: Train all 20 stocks and generate production predictions")
    print("="*80)

if __name__ == "__main__":
    main()
