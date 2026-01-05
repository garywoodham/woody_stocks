# 📘 RISK MANAGEMENT & EARNINGS CALENDAR GUIDE

## 🎯 Overview

Your portfolio now includes comprehensive risk management tools and earnings calendar tracking to protect capital and avoid surprise volatility.

---

## ⚠️ RISK MANAGEMENT SYSTEM

### What It Does

1. **Risk Scoring** - Evaluates each stock on 0-100 scale based on:
   - Annual volatility
   - Maximum drawdown
   - ATR (Average True Range)
   
2. **Position Sizing** - Recommends position sizes based on:
   - Risk-adjusted allocation
   - Volatility weighting (lower risk = larger position)
   - Position limits (15% max per stock)
   - Sector limits (40% max per sector)

3. **Stop-Loss Levels** - Calculates recommended stop-losses using:
   - 2x ATR method (adapts to volatility)
   - Protects against large losses
   - Stock-specific levels (not one-size-fits-all)

4. **Portfolio Warnings** - Alerts you to:
   - Position concentration issues
   - Sector over-exposure
   - High-risk stock allocation

### How to Use

#### Run Risk Analysis

```bash
python risk_manager.py
```

**Output Files:**
- `risk_metrics.csv` - Risk scores, volatility, drawdowns for all stocks
- `risk_adjusted_positions.csv` - Recommended position sizes for BUY signals
- `stop_losses.csv` - Stop-loss levels for all stocks
- `risk_warnings.csv` - Portfolio warnings (if any)

#### View in Dashboard

Navigate to **⚠️  Risk Management** tab to see:

1. **Summary Cards**
   - Portfolio Volatility (risk-weighted)
   - High Risk Stock Count
   - Risk Warnings
   - Earnings Warnings

2. **Risk Distribution Chart**
   - Shows Low/Medium/High risk stock counts
   - Green = Safe, Yellow = Moderate, Red = Risky

3. **Volatility by Sector**
   - Which sectors are most volatile
   - Helps with diversification decisions

4. **Top 10 Highest Risk Stocks**
   - Ranked by risk score
   - Shows volatility and max drawdown
   - Red highlighting for Risk Score > 75

5. **Stop-Loss Table**
   - Current price vs recommended stop-loss
   - Stop % shows how much buffer you have
   - ATR % shows recent volatility

### Understanding Risk Scores

| Risk Score | Category | Interpretation |
|------------|----------|----------------|
| 0-33       | **Low**  | Stable, lower volatility stocks (JNJ, MSFT, LMT) |
| 34-66      | **Medium** | Moderate risk, typical for growth stocks |
| 67-100     | **High** | Very volatile, speculative stocks (TLRY, MARA, AMC) |

### Current Portfolio Risk Status

```
Expected Return:  36.1% annually
Volatility:       14.4% annually
Sharpe Ratio:     2.24 (excellent!)
High Risk Stocks: 6/35 (17%)
Risk Warnings:    2
```

**Warnings:**
1. **PFE position exceeds 15% limit** - Consider reducing
2. **Technology sector exceeds 40% limit** - Diversify to other sectors

### Risk Management Best Practices

#### ✅ DO:
- Check risk tab daily before trading
- Respect stop-loss levels (don't remove them!)
- Rebalance when warnings appear
- Reduce high-risk positions when portfolio grows
- Use smaller positions for high-risk stocks

#### ❌ DON'T:
- Ignore risk warnings
- Concentrate >15% in single stock
- Ignore sector concentration (>40%)
- Trade without stop-losses
- Average down on losing positions

---

## 📅 EARNINGS CALENDAR SYSTEM

### What It Does

1. **Tracks Earnings Dates** - For all 35 stocks in your portfolio
2. **Flags Upcoming Earnings** - Warns if earnings in next 5 days
3. **Conflict Detection** - Alerts if BUY recommendations have imminent earnings
4. **Avoids Volatility** - Helps you avoid unpredictable earnings moves

### How to Use

#### Fetch Earnings Calendar

```bash
python earnings_calendar.py
```

**Output Files:**
- `earnings_calendar.csv` - All earnings dates for 35 stocks
- `earnings_warnings.csv` - Conflict warnings (if any)

#### View in Dashboard

The **⚠️  Risk Management** tab includes earnings data:

1. **Earnings Warnings Card** - Shows count of stocks with earnings in next 5 days
2. **Upcoming Earnings Chart** - Visual timeline of next 30 days (when available)

### Current Earnings Status

**Next 30 Days:**
- TLRY - Jan 08, 2026 (3 days) ⚠️
- JNJ  - Jan 21, 2026 (16 days)
- LMT  - Jan 27, 2026 (22 days)
- NOC  - Jan 27, 2026 (22 days)
- RTX  - Jan 27, 2026 (22 days)
- MSFT - Jan 28, 2026 (23 days)
- AAPL - Jan 29, 2026 (24 days)
- SOFI - Jan 30, 2026 (25 days)
- PLTR - Feb 02, 2026 (28 days)
- PFE  - Feb 03, 2026 (29 days)

### Earnings Trading Strategy

#### Before Earnings (5+ days out)
- ✅ **Safe to enter** new positions
- Normal risk management applies

#### Earnings Window (5 days before)
- ⚠️ **Reduce position size** by 50%
- Consider waiting until after earnings
- Tighten stop-losses

#### Earnings Day (0-2 days)
- 🛑 **Do not enter** new positions
- Close short-term positions (1d/5d trades)
- Hold long-term positions (21d+) if strong conviction
- Expect 5-15% volatility

#### After Earnings (next day)
- 📊 **Reassess** based on results
- Check if price moved to stop-loss
- Look for new entry if results were positive

### Earnings Risk Examples

**High Earnings Risk Stocks (Volatile movers):**
- TLRY, MARA, AMC, SPCE (20-40% moves)
- Avoid these completely during earnings

**Medium Earnings Risk:**
- AAPL, MSFT, NVDA, GOOGL (5-10% moves)
- Reduce position or wait

**Low Earnings Risk:**
- JNJ, PFE, LMT, GSK (2-5% moves)
- Can hold through earnings with stops

---

## 🤖 AUTOMATION

Both systems run automatically every day via GitHub Actions:

**Daily Workflow (6 PM UTC):**
1. Download latest stock data
2. Fetch sentiment
3. Generate predictions
4. Create recommendations
5. Calculate portfolio allocation
6. **→ Run risk analysis** ⚠️
7. **→ Fetch earnings calendar** 📅
8. Track performance
9. Commit & push to GitHub

This keeps your risk data and earnings calendar **always up-to-date**.

---

## 📊 INTEGRATION WITH OTHER FEATURES

### Risk + Recommendations
- Risk-adjusted position sizes use both prediction score AND volatility
- High-risk stocks get smaller allocations automatically

### Risk + Performance Tracking
- Compare if high-risk trades actually performed better
- Analyze risk-adjusted returns

### Risk + Sentiment
- High sentiment + low risk = best opportunities
- Low sentiment + high risk = avoid

### Earnings + Recommendations
- BUY signals with earnings warnings flagged
- System suggests "WAIT" or "REDUCE_SIZE" actions

---

## 📁 FILES REFERENCE

### Risk Management Files
| File | Description | Update Frequency |
|------|-------------|------------------|
| `risk_metrics.csv` | Volatility, ATR, risk scores | Daily |
| `risk_adjusted_positions.csv` | Recommended position sizes | Daily |
| `stop_losses.csv` | Stop-loss levels | Daily |
| `risk_warnings.csv` | Portfolio warnings | Daily |

### Earnings Files
| File | Description | Update Frequency |
|------|-------------|------------------|
| `earnings_calendar.csv` | All earnings dates | Daily |
| `earnings_warnings.csv` | Conflict warnings | Daily |

---

## 🎯 QUICK REFERENCE

### Daily Risk Check (2 minutes)

```bash
# 1. Check risk status
python risk_manager.py

# 2. Check earnings
python earnings_calendar.py

# 3. View in dashboard
# Navigate to: ⚠️  Risk Management tab

# 4. Check for warnings
cat risk_warnings.csv
cat earnings_warnings.csv
```

### Key Metrics to Monitor

| Metric | Target | Current | Action if Exceeded |
|--------|--------|---------|-------------------|
| Portfolio Volatility | <20% | 14.4% ✅ | Reduce high-risk positions |
| Max Position Size | <15% | 16% ⚠️ | Rebalance PFE |
| Max Sector Exposure | <40% | 46.5% ⚠️ | Reduce Tech sector |
| High Risk Stocks | <20% | 17% ✅ | Keep monitoring |
| Sharpe Ratio | >1.0 | 2.24 ✅ | Excellent! |

---

## 🚀 NEXT STEPS

Now that you have risk management and earnings calendar:

1. **Week 1** - Get familiar with risk scores
2. **Week 2** - Start using stop-losses
3. **Week 3** - Track earnings impact on your trades
4. **Week 4** - Optimize position sizes based on volatility

**Pro Tips:**
- Review risk tab every morning
- Set stop-losses immediately after entering positions
- Mark earnings dates on calendar
- Don't fight the risk warnings - they protect you!

---

## ❓ FAQ

**Q: Why is my favorite stock flagged as high risk?**
A: High risk doesn't mean bad - it means volatile. Meme stocks (AMC, GME) and crypto-related (MARA, RIOT) are inherently volatile. Just use smaller positions.

**Q: Should I always follow stop-losses?**
A: Yes! They're calculated to give stocks breathing room (2x ATR) while protecting against major losses.

**Q: What if I disagree with risk-adjusted position sizes?**
A: They're recommendations. But if you override them, stay within the 15% position limit and 40% sector limit.

**Q: Do I need to update earnings calendar manually?**
A: No! It runs automatically every day via GitHub Actions.

**Q: What if a stock I want to buy has earnings in 3 days?**
A: Wait until after earnings. Patience saves capital.

---

## 📈 SUMMARY

You now have institutional-grade risk management:

✅ **Risk Scoring** - Know what you're buying
✅ **Position Sizing** - Proper capital allocation
✅ **Stop-Losses** - Protect against big losses
✅ **Earnings Tracking** - Avoid surprise volatility
✅ **Portfolio Warnings** - Concentration alerts
✅ **Automated Updates** - Always current

**Use these tools. They will save you money.**
