#!/bin/bash
#
# Daily Trading System Runner
# ============================
# Automates the complete workflow: predictions → filters → trading plan
#
# Usage: ./run_daily_trading.sh
#

set -e  # Exit on any error

echo "================================================================================"
echo "🚀 DAILY TRADING SYSTEM - FULL WORKFLOW"
echo "================================================================================"
echo ""

# Step 1: Generate daily predictions
echo "📊 Step 1/5: Generating daily predictions..."
python3 generate_daily_signals.py
if [ $? -eq 0 ]; then
    echo "✅ Predictions generated successfully"
else
    echo "❌ Failed to generate predictions"
    exit 1
fi
echo ""

# Step 2: Create/update earnings calendar
echo "📅 Step 2/5: Updating earnings calendar..."
python3 create_earnings_calendar.py
if [ $? -eq 0 ]; then
    echo "✅ Earnings calendar updated"
else
    echo "❌ Failed to update earnings calendar"
    exit 1
fi
echo ""

# Step 3: Filter top stocks
echo "🔍 Step 3/5: Filtering top stocks (Tier system)..."
python3 filter_top_stocks.py
if [ $? -eq 0 ]; then
    echo "✅ Stock tiers calculated successfully"
else
    echo "❌ Failed to filter stocks"
    exit 1
fi
echo ""

# Step 4: Detect market regime
echo "🌐 Step 4/5: Detecting market regime..."
python3 detect_market_regime.py
if [ $? -eq 0 ]; then
    echo "✅ Market regime detected successfully"
else
    echo "❌ Failed to detect market regime"
    exit 1
fi
echo ""

# Step 5: Generate integrated trading plan
echo "🎯 Step 5/5: Creating integrated trading plan (with earnings filter)..."
python3 show_integrated_trading_plan.py
if [ $? -eq 0 ]; then
    echo "✅ Trading plan created successfully"
else
    echo "❌ Failed to create trading plan"
    exit 1
fi
echo ""

echo "================================================================================"
echo "✅ DAILY WORKFLOW COMPLETE!"
echo "================================================================================"
echo ""
echo "📁 Generated Files:"
echo "  • predictions_refined.csv            - All predictions"
echo "  • earnings_calendar.json             - Earnings dates (next 45 days)"
echo "  • predictions_top_stocks.csv         - Top 21 stocks only"
echo "  • predictions_regime_filtered.csv    - Regime-filtered signals"
echo "  • todays_trading_plan.csv            - 🎯 YOUR TRADING PLAN (top 10, earnings-safe)"
echo "  • stock_tiers.json                   - Stock classifications"
echo "  • current_market_regime.json         - Today's market regime"
echo ""
echo "🎯 NEXT: Review todays_trading_plan.csv for actionable trades!"
echo "💡 TIP: Run 'python3 dashboard.py' to see visual regime indicator!"
echo "================================================================================"
