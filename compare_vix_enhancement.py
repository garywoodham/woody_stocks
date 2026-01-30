#!/usr/bin/env python3
"""
Compare baseline (horizon-optimized) vs VIX-enhanced model results.
"""
import re

def parse_log_file(filepath):
    """Parse training log and extract accuracy results."""
    results = {'1-d': [], '5-d': [], '21-d': []}
    current_stock = None
    current_horizon = None
    
    with open(filepath, 'r') as f:
        for line in f:
            # Extract stock name
            if 'REFINED TRAINING: DAILY -' in line:
                current_stock = line.split('DAILY -')[1].strip()
            
            # Extract horizon
            if line.startswith('Horizon:'):
                current_horizon = line.split(':')[1].strip()
            
            # Extract accuracy
            if 'Accuracy:' in line and current_horizon:
                # Parse "Accuracy: 53.48% | UP: 61.24% | DOWN: 40.74%"
                match = re.search(r'Accuracy:\s+(\d+\.\d+)%', line)
                if match:
                    accuracy = float(match.group(1))
                    results[current_horizon].append(accuracy)
    
    return results

def calculate_statistics(results):
    """Calculate mean accuracy for each horizon."""
    stats = {}
    for horizon, accuracies in results.items():
        if accuracies:
            stats[horizon] = {
                'mean': sum(accuracies) / len(accuracies),
                'count': len(accuracies)
            }
    return stats

def main():
    print("=" * 80)
    print("VIX ENHANCEMENT COMPARISON")
    print("=" * 80)
    print()
    
    # Parse both log files
    baseline_results = parse_log_file('model_training_horizon_optimized.log')
    vix_results = parse_log_file('model_training_vix_enhanced.log')
    
    # Calculate statistics
    baseline_stats = calculate_statistics(baseline_results)
    vix_stats = calculate_statistics(vix_results)
    
    # Compare results
    print("ACCURACY COMPARISON: Baseline vs VIX-Enhanced")
    print("-" * 80)
    print(f"{'Horizon':<10} {'Baseline':<20} {'VIX-Enhanced':<20} {'Change':<20}")
    print("-" * 80)
    
    total_change = 0
    for horizon in ['1-d', '5-d', '21-d']:
        baseline_acc = baseline_stats[horizon]['mean']
        vix_acc = vix_stats[horizon]['mean']
        change = vix_acc - baseline_acc
        total_change += change
        
        emoji = "🟢" if change > 0 else "🔴" if change < 0 else "⚪"
        print(f"{horizon:<10} {baseline_acc:>6.2f}%  ({baseline_stats[horizon]['count']:>2} models)  "
              f"{vix_acc:>6.2f}%  ({vix_stats[horizon]['count']:>2} models)  "
              f"{change:>+6.2f}% {emoji}")
    
    print("-" * 80)
    overall_baseline = sum(baseline_stats[h]['mean'] for h in ['1-d', '5-d', '21-d']) / 3
    overall_vix = sum(vix_stats[h]['mean'] for h in ['1-d', '5-d', '21-d']) / 3
    overall_change = overall_vix - overall_baseline
    
    emoji = "🟢" if overall_change > 0 else "🔴" if overall_change < 0 else "⚪"
    print(f"{'Overall':<10} {overall_baseline:>6.2f}%             "
          f"{overall_vix:>6.2f}%             "
          f"{overall_change:>+6.2f}% {emoji}")
    print()
    
    # Feature counts
    print("FEATURE USAGE:")
    print("-" * 80)
    print("Baseline:  1d/5d = 89 features,  21d = 99 features")
    print("VIX-Enh:   1d/5d = 95 features,  21d = 105 features")
    print("           (+6 VIX features added to all horizons)")
    print()
    
    print("VIX FEATURES ADDED:")
    print("-" * 80)
    print("1. vix_change      - VIX momentum (percent change)")
    print("2. vix_ma_5        - 5-day moving average")
    print("3. vix_trend       - Binary flag for rising fear (VIX > MA)")
    print("4. vix_spike       - Panic detector (>15% jump or VIX>30)")
    print("5. vix_rsi         - VIX regime × RSI interaction")
    print("6. vix_momentum    - VIX regime × momentum interaction")
    print("7. vix_volatility  - Combined VIX × stock volatility signal")
    print()
    
    # Analysis
    print("ANALYSIS:")
    print("-" * 80)
    if overall_change < 0:
        print("❌ VIX enhancements decreased overall accuracy")
        print(f"   Loss: {overall_change:.2f}%")
        print()
        print("LIKELY CAUSES:")
        print("  - VIX features may add noise rather than signal for these stocks")
        print("  - Banking/pharma/defense stocks may not correlate strongly with VIX")
        print("  - VIX is a broad market fear gauge, not sector-specific")
        print()
        print("RECOMMENDATION:")
        print("  ❌ Do not keep VIX enhancements")
        print("  ✅ Try next improvement: Cross-stock correlation or Ensemble stacking")
    elif overall_change > 0 and overall_change < 0.5:
        print("⚠️  VIX enhancements showed marginal improvement")
        print(f"   Gain: {overall_change:.2f}%")
        print()
        print("RECOMMENDATION:")
        print("  Consider keeping if 1-day and 5-day specifically improved")
        print("  Otherwise, try more impactful improvements:")
        print("  - Cross-stock correlation (+1-1.5% expected)")
        print("  - Ensemble stacking (+1-2% expected)")
    else:
        print("✅ VIX enhancements improved accuracy!")
        print(f"   Gain: {overall_change:.2f}%")
        print()
        print("NEXT STEPS:")
        print("  1. Keep VIX features")
        print("  2. Try next improvement: Cross-stock correlation or Ensemble stacking")

if __name__ == "__main__":
    main()
