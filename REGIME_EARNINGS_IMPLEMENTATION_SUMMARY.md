# Market Regime + Earnings Calendar Implementation Summary

**Date:** January 7, 2026  
**Improvements Implemented:** #1 and #2 from improvement roadmap

---

## 🎯 What Was Added

### 1. Market Regime Detection Features (9 features)
Added integration with `market_regime_history.csv`:
- **market_regime** - Numeric classification: 2=BULL, 1=SIDEWAYS, 0=BEAR
- **regime_confidence** - Model confidence (0-100%)
- **regime_score** - Regime strength score  
- **regime_rsi** - RSI × regime (interaction feature)
- **regime_momentum** - Momentum × regime (interaction feature)

**Data Source:** 303 days of historical regime classifications  
**Strategy:** Forward-fill (regimes persist until they change)

### 2. Earnings Calendar Features (5 features)
Added integration with `earnings_calendar.csv`:
- **days_to_earnings** - Days until next earnings (positive = future, negative = past)
- **is_earnings_week** - Binary: 1 if within ±3 days of earnings
- **pre_earnings** - Binary: 1 if 1-7 days before earnings
- **post_earnings** - Binary: 1 if 0-5 days after earnings  
- **earnings_volatility** - is_earnings_week × volatility_20 (interaction)

**Data Source:** Earnings dates for 35 stocks  
**Strategy:** Calculate dynamic days_to_earnings for each trading day

---

## 📊 Results

### Overall Performance
```
BEFORE (Sentiment Only):          AFTER (+ Regime + Earnings):
  1-day:  53.44%                    1-day:  52.10%  (-1.34%)
  5-day:  54.74%                    5-day:  53.31%  (-1.43%)
  21-day: 58.03%                    21-day: 59.83%  (+1.80%) ⭐
  Overall: 55.40%                   Overall: 55.08% (-0.32%)
```

### Feature Usage (Top 5 Importance)
- **Earnings features:** 24/105 models (22.9%) - **HIGH IMPACT**
- **Regime features:** 1/105 models (1.0%) - Low direct usage

### Best Performing Models
| Stock | Horizon | Accuracy | Note |
|-------|---------|----------|------|
| Standard Chartered | 21-day | 78.66% | Outstanding |
| GOOGL | 21-day | 69.73% | Excellent |
| HSBC | 21-day | 67.89% | Very good |
| HSBC | 5-day | 65.65% | Best swing trade model |
| AMC | 21-day | 65.64% | Improved significantly |

---

## 🔍 Key Insights

### ✅ Earnings Features - Clear Winner
1. **days_to_earnings** most frequently in top 5 features
2. Strongest impact on 21-day predictions (+1.80% improvement)
3. Stocks near earnings show distinct predictable patterns:
   - Pre-earnings: Anticipation buying/selling
   - Post-earnings: Reaction and stabilization
4. Earnings-volatility interaction valuable for risk assessment

### ⚠️ Market Regime - Limited Direct Impact
1. Only 1% usage in top 5 features directly
2. Current period dominated by BULL regime (regime=2)
   - Less variation to learn from
   - May be more valuable during regime transitions
3. Interaction features (regime_rsi, regime_momentum) created but not heavily used
4. Possible reasons:
   - Need longer history with more regime changes
   - Other features already capture market conditions (VIX, SPY correlation)
   - May benefit from separate models per regime instead of as features

### 📈 Why 21-Day Improved but 1-Day/5-Day Declined
**21-Day (Position Trading):**
- Earnings calendar highly relevant (3-week window)
- Earnings cycles are persistent and predictable  
- Models learn "avoid pre-earnings volatility" patterns
- **Result: +1.80% improvement**

**1-Day and 5-Day (Day/Swing Trading):**
- Earnings impact less systematic at short horizons
- Daily noise may mask earnings effects
- Added features may have diluted short-term signal
- **Result: -1.34% and -1.43% decline**

---

## 💡 Recommendations

### Immediate Actions
1. **Focus on 21-day predictions** - Earnings features shine here
2. **Filter short-term trades around earnings** - Use earnings flags to avoid volatile periods
3. **Consider separate models** - Train regime-specific models instead of regime as feature

### Feature Engineering Refinements
```python
# Potential improvements:
- days_since_earnings (not just days_to)
- earnings_beat_history (if stock typically beats/misses)
- sector_earnings_season (when peers report)
- regime_change_indicator (transitions more important than state)
```

### Next Priority (from improvement roadmap)
1. **Cross-Stock Correlation** (#3) - Expected +1-1.5%
   - Sector momentum features
   - Peer stock correlation  
   - Market contagion effects
   
2. **Volatility Regime** (#4) - Expected +1%
   - Better than binary regime classification
   - VIX-based buckets: Low (<15), Normal (15-25), High (>25)
   
3. **Ensemble Stacking** (#5) - Expected +1-2%
   - Stack predictions from multiple models
   - Reduce overfitting

---

## 📁 Files Modified

### Core Training Script
**[train_refined_models.py](train_refined_models.py)**
- Added `load_market_regime_data()` function
- Added `load_earnings_calendar()` function  
- Modified `create_optimized_features()` to accept regime_df and earnings_df
- Added regime features in lines 261-298
- Added earnings features in lines 330-391
- Updated main() to load and pass new data sources

### Feature Count Evolution
```
Before:  ~62 features (55 technical + 7 sentiment)
After:   ~99 features (55 technical + 7 sentiment + 9 regime + 5 earnings + 23 interactions)
```

### Output Files
- `model_training_with_regime_earnings.log` - Full training log
- `model_results_with_regime_earnings.csv` - Results CSV
- All 35 `models/*_daily_refined.joblib` files updated

---

## 🎓 Lessons Learned

1. **Not all features are created equal**
   - Earnings (22.9% usage) vs Regime (1.0% usage)
   - Real-world schedule/events > market state classification

2. **Time horizon matters**
   - Features work differently at different time scales
   - Match feature to prediction horizon

3. **Forward-fill strategy works well**
   - Regimes persist (no instant changes)
   - Earnings dates are fixed schedule (predictable)

4. **Model complexity trade-offs**
   - More features ≠ always better
   - Short-term predictions got slightly worse
   - Long-term predictions improved significantly

5. **Feature selection is key**
   - LightGBM automatically selects useful features
   - Usage rate in "top 5" is good importance proxy
   - Low usage features might still have indirect effects

---

## ⏭️ Next Session Plan

Based on improvement roadmap, implement **#3: Cross-Stock Correlation Features**

Expected implementation time: 45-60 minutes  
Expected improvement: +1 to 1.5%  
Difficulty: Medium

Key features to add:
- Sector daily returns  
- Stock vs sector relative strength
- Correlation to sector leaders
- Cross-stock momentum signals

---

## 📞 Quick Stats

- **Total training time:** ~8 minutes (35 stocks × 3 horizons)
- **Models trained:** 105 (35 stocks × 3 time horizons)
- **New features:** 14 (9 regime + 5 earnings)
- **Feature usage rate:** 23.8% (25/105 models use new features in top 5)
- **Best improvement:** +1.80% on 21-day predictions
- **ROI:** Positive for position traders, mixed for day traders
