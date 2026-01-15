#!/bin/bash
# Complete daily workflow: sentiment -> predictions -> recommendations -> portfolio -> reports

set -e  # Exit on error

echo "================================================================================"
echo "🤖 AUTOMATED STOCK PREDICTION & PORTFOLIO WORKFLOW"
echo "================================================================================"
echo ""
echo "Started: $(date)"
echo ""

# Step 0: Fetch sentiment data (using FREE RSS feeds)
echo "0️⃣  Fetching latest news sentiment..."
python collect_daily_sentiment.py
echo "   ✓ Sentiment data updated (RSS-based, no API key needed)"
echo ""

# Step 1: Generate predictions
echo "1️⃣  Generating stock predictions..."
python predict_refined.py
echo "   ✓ Predictions complete"
echo ""

# Step 2: Generate recommendations
echo "2️⃣  Generating BUY/HOLD/SELL recommendations..."
python generate_recommendations.py
echo "   ✓ Recommendations complete"
echo ""

# Step 3: Generate portfolio allocation
echo "3️⃣  Generating optimal portfolio allocation..."
python portfolio_manager.py
echo "   ✓ Portfolio allocation complete"
echo ""

# Step 4: Generate trading signals
echo "4️⃣  Generating daily trading signals..."
python generate_daily_signals.py
echo "   ✓ Trading signals complete"
echo ""

# Step 5: Run backtest (if requested)
if [ "$1" == "--backtest" ]; then
    echo "5️⃣  Running backtest on recommendations..."
    python backtest_recommendations.py
    echo "   ✓ Backtest complete"
    echo ""
fi

echo "================================================================================"
echo "✅ WORKFLOW COMPLETE!"
echo "================================================================================"
echo ""
echo "Generated files:"
echo "  📊 predictions_refined.csv        - Individual stock predictions"
echo "  🎯 stock_recommendations.csv      - BUY/HOLD/SELL recommendations"
echo "  💼 portfolio_allocation.csv       - Optimal portfolio allocation"
echo "  🚦 daily_signals.csv              - Trading signals"
if [ "$1" == "--backtest" ]; then
    echo "  📈 backtest_recommendations.csv   - Backtest results"
fi
echo ""
echo "Completed: $(date)"
echo ""
