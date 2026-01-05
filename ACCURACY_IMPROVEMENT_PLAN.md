# Accuracy Improvement Strategy

## Current Performance Analysis

### Accuracy Metrics
- **1-day**: 53% (barely better than random)
- **5-day**: 56%
- **21-day**: 57% (best performance)

### Key Issues
1. **Low confidence**: Mean confidence < 13%, no predictions above 30% confidence
2. **Low accuracy**: Only 3-7% better than random (50%)
3. **High variance**: 21-day predictions vary from 40% to 70% accuracy
4. **Simple model**: Using basic LightGBM with default-ish parameters

## Improvement Strategies

### 1. **Hyperparameter Optimization** ⭐ Priority
**Current problem**: Using mostly default parameters
**Solution**: Grid search or Bayesian optimization for optimal parameters

Key parameters to tune:
- `num_leaves`: 31 → Test [20, 31, 50, 100]
- `learning_rate`: 0.05 → Test [0.01, 0.03, 0.05, 0.1]
- `max_depth`: 7 → Test [5, 7, 10, -1]
- `min_data_in_leaf`: 20 → Test [10, 20, 50]
- `feature_fraction`: 0.8 → Test [0.6, 0.7, 0.8, 0.9]
- `bagging_fraction`: 0.8 → Test [0.6, 0.7, 0.8, 0.9]

Expected improvement: +5-10% accuracy

### 2. **Advanced Feature Engineering** ⭐ Priority
**Current**: ~46 basic technical indicators
**Add**:
- Relative strength across stocks
- Sector momentum
- Market regime detection (trending vs ranging)
- Fractal indicators
- Order flow imbalance proxies
- Cross-asset correlations
- Lagged features (previous predictions)
- Feature interactions (RSI * Volume ratio)

Expected improvement: +3-7% accuracy

### 3. **Ensemble Methods** ⭐⭐ High Impact
**Current**: Single LightGBM model
**Add**:
- Stacking: LightGBM + XGBoost + RandomForest
- Voting classifier with calibrated probabilities
- Time-series specific ensembles (separate models for different market regimes)

Expected improvement: +5-10% accuracy

### 4. **Class Imbalance Handling**
**Current**: No class balancing
**Add**:
- SMOTE for minority class oversampling
- Class weights in LightGBM
- Stratified time-series splits
- Focal loss instead of binary cross-entropy

Expected improvement: +2-5% accuracy

### 5. **Better Data Preprocessing**
**Current**: Simple StandardScaler
**Add**:
- RobustScaler (less sensitive to outliers)
- Feature scaling per stock/sector
- Remove extreme outliers (>3 std dev)
- Winsorization for extreme values

Expected improvement: +1-3% accuracy

### 6. **Market Regime Detection**
**Current**: Single model for all market conditions
**Add**:
- Separate models for:
  - Bull markets (uptrending)
  - Bear markets (downtrending)
  - Sideways/ranging markets
- Auto-detect regime and switch models

Expected improvement: +5-8% accuracy

### 7. **Feature Selection**
**Current**: Using all 46 features
**Add**:
- Recursive feature elimination
- Permutation importance analysis
- Remove correlated features (correlation > 0.9)
- Keep only top 20-30 most important features

Expected improvement: +2-4% accuracy (by reducing noise)

### 8. **Better Validation Strategy**
**Current**: Simple 80/20 split
**Add**:
- Walk-forward validation (expanding window)
- Multiple time-series splits
- Out-of-time testing
- Purging and embargo to prevent lookahead bias

Expected improvement: Better model selection, more realistic accuracy

### 9. **Additional Data Sources** 📊
**Current**: Only OHLCV data
**Add**:
- News sentiment (FinBERT)
- Options implied volatility
- Put/call ratios
- Insider trading data
- Economic indicators (VIX, Treasury yields, USD strength)
- Social media sentiment

Expected improvement: +5-15% accuracy (significant impact)

### 10. **Deep Learning Models** 🚀
**Current**: Traditional ML (LightGBM)
**Add**:
- LSTM/GRU for sequence modeling
- Transformer models (attention mechanism)
- Temporal Convolutional Networks (TCN)
- Hybrid models (CNN for feature extraction + LSTM)

Expected improvement: +5-15% accuracy (but requires more data and compute)

## Implementation Priority

### Phase 1: Quick Wins (1-2 days)
1. **Hyperparameter tuning** - Immediate improvement
2. **Feature engineering** - Add 20-30 new features
3. **Ensemble methods** - Combine 3 models

**Expected result**: 53% → 63-68% accuracy

### Phase 2: Advanced (3-5 days)
4. **Market regime detection** - Contextual modeling
5. **Class imbalance handling** - Better predictions
6. **Feature selection** - Noise reduction

**Expected result**: 68% → 73-78% accuracy

### Phase 3: Expert (1-2 weeks)
7. **Additional data sources** - News, sentiment, options
8. **Deep learning models** - State-of-the-art
9. **Walk-forward optimization** - Production-ready

**Expected result**: 78% → 85%+ accuracy

## Realistic Expectations

### Stock Market Prediction Limits
- **Random**: 50% (coin flip)
- **Basic TA**: 52-55%
- **Good ML**: 60-65%
- **Advanced ML**: 65-75%
- **Professional quant**: 70-80%
- **Theoretical maximum**: ~85% (market efficiency limits)

### Our Trajectory
- **Current**: 53-57% ✓
- **After Phase 1**: 63-68% (achievable)
- **After Phase 2**: 73-78% (competitive)
- **After Phase 3**: 80%+ (professional grade)

## Recommended Next Steps

### Immediate Action (Today)
```bash
# 1. Implement hyperparameter tuning
python improve_models_hyperopt.py

# 2. Add advanced features
python enhance_features.py

# 3. Train ensemble models
python train_ensemble_models.py

# Expected time: 2-4 hours
# Expected accuracy gain: +10-15%
```

### This Week
- Implement market regime detection
- Add feature selection pipeline
- Set up walk-forward validation
- Add more stocks for training diversity

### Accuracy Monitoring
Create a tracking system to measure improvements:
- Track accuracy over time
- Compare to baseline (53%)
- Monitor per-stock performance
- Validate on unseen data

## Code Structure for Improvements

```
improve_models/
├── hyperparameter_tuning.py    # Optuna/GridSearch
├── advanced_features.py         # Enhanced feature engineering
├── ensemble_models.py           # Stacking/voting classifiers
├── market_regimes.py           # Trend detection + model switching
└── validation.py               # Walk-forward validation
```

## Conclusion

**Current state**: Models are barely beating random chance (53-57% vs 50%)

**Root causes**:
1. Default hyperparameters
2. Basic features only
3. Single model (no ensemble)
4. No market context awareness

**Path forward**:
- Phase 1 (quick wins): +10-15% accuracy → 63-68%
- Phase 2 (advanced): +5-10% accuracy → 73-78%
- Phase 3 (expert): +5-7% accuracy → 80%+

**Best ROI**: Start with hyperparameter tuning + ensemble methods. These are proven techniques that will give immediate, measurable improvements with moderate effort.

Would you like me to implement Phase 1 improvements?
