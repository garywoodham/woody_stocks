"""
Quick Start: 1-Day Accuracy Improvements
Implement the easiest, highest-impact improvements first
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============================================================================
# STEP 1: PRE-MARKET DATA FETCHER
# Expected Impact: +2-3% accuracy
# ============================================================================

def fetch_premarket_signals(ticker, date=None):
    """
    Fetch pre-market price action and generate signals
    
    Returns:
        dict with pre-market features
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Get 2 days of 1-minute data with pre/post market
        hist = stock.history(period="2d", interval="1m", prepost=True)
        
        if len(hist) == 0:
            return None
        
        # Separate pre-market (4am-9:30am) from regular hours
        today = hist.index[-1].date()
        today_data = hist[hist.index.date == today]
        
        premarket = today_data[
            (today_data.index.hour < 9) | 
            ((today_data.index.hour == 9) & (today_data.index.minute < 30))
        ]
        
        # Get yesterday's close
        yesterday = hist[hist.index.date < today]
        if len(yesterday) == 0:
            return None
        
        yesterday_close = yesterday.iloc[-1]['Close']
        
        # Calculate pre-market features
        if len(premarket) > 0:
            premarket_last = premarket.iloc[-1]['Close']
            premarket_high = premarket['High'].max()
            premarket_low = premarket['Low'].min()
            premarket_volume = premarket['Volume'].sum()
            
            # Gap calculations
            gap_pct = (premarket_last / yesterday_close - 1) * 100
            gap_size = abs(gap_pct)
            
            # Price action in pre-market
            pm_range = (premarket_high - premarket_low) / premarket_last * 100
            
            # Volume analysis
            avg_premarket_volume = stock.history(period="1mo", interval="1d")['Volume'].mean() * 0.05  # ~5% of daily
            volume_ratio = premarket_volume / (avg_premarket_volume + 1)
            
            return {
                'premarket_gap_pct': gap_pct,
                'premarket_gap_size': gap_size,
                'premarket_gap_up': 1 if gap_pct > 1 else 0,
                'premarket_gap_down': 1 if gap_pct < -1 else 0,
                'premarket_range_pct': pm_range,
                'premarket_volume_ratio': volume_ratio,
                'premarket_volume_spike': 1 if volume_ratio > 2 else 0,
                'has_premarket_data': 1
            }
        else:
            return {
                'premarket_gap_pct': 0,
                'premarket_gap_size': 0,
                'premarket_gap_up': 0,
                'premarket_gap_down': 0,
                'premarket_range_pct': 0,
                'premarket_volume_ratio': 0,
                'premarket_volume_spike': 0,
                'has_premarket_data': 0
            }
    
    except Exception as e:
        print(f"Error fetching pre-market for {ticker}: {e}")
        return None


# ============================================================================
# STEP 2: MARKET REGIME DETECTOR (VIX-BASED)
# Expected Impact: +2-3% accuracy
# ============================================================================

def get_market_regime():
    """
    Detect current market regime based on VIX
    
    Returns:
        dict with regime information
    """
    try:
        vix = yf.Ticker("^VIX")
        vix_hist = vix.history(period="5d")
        
        if len(vix_hist) == 0:
            return None
        
        current_vix = vix_hist['Close'].iloc[-1]
        vix_5d_avg = vix_hist['Close'].mean()
        vix_change = (current_vix / vix_5d_avg - 1) * 100
        
        # Regime classification
        if current_vix > 30:
            regime = 'extreme_fear'
            regime_code = 4
        elif current_vix > 20:
            regime = 'high_volatility'
            regime_code = 3
        elif current_vix < 12:
            regime = 'low_volatility'
            regime_code = 1
        else:
            regime = 'normal'
            regime_code = 2
        
        # VIX trend
        vix_trend = 'rising' if vix_change > 5 else ('falling' if vix_change < -5 else 'stable')
        
        return {
            'vix_level': current_vix,
            'vix_regime': regime,
            'vix_regime_code': regime_code,
            'vix_change_pct': vix_change,
            'vix_trend': vix_trend,
            'vix_extreme_fear': 1 if current_vix > 30 else 0,
            'vix_high': 1 if 20 < current_vix <= 30 else 0,
            'vix_normal': 1 if 12 <= current_vix <= 20 else 0,
            'vix_low': 1 if current_vix < 12 else 0
        }
    
    except Exception as e:
        print(f"Error fetching VIX: {e}")
        return None


# ============================================================================
# STEP 3: EARNINGS PROXIMITY FEATURES
# Expected Impact: +1-2% accuracy
# ============================================================================

def get_earnings_features(ticker, earnings_df=None):
    """
    Get features related to earnings proximity
    
    Args:
        ticker: Stock ticker
        earnings_df: DataFrame with earnings calendar (columns: ticker, date)
    
    Returns:
        dict with earnings features
    """
    if earnings_df is None:
        try:
            earnings_df = pd.read_csv('earnings_calendar.csv')
        except:
            return None
    
    # Handle different column names
    ticker_col = 'Ticker' if 'Ticker' in earnings_df.columns else 'ticker'
    date_col = 'Earnings_Date' if 'Earnings_Date' in earnings_df.columns else ('Earnings Date' if 'Earnings Date' in earnings_df.columns else 'date')
    
    ticker_earnings = earnings_df[earnings_df[ticker_col] == ticker]
    
    if len(ticker_earnings) == 0:
        return {
            'days_to_earnings': 999,
            'near_earnings': 0,
            'avoid_earnings': 0
        }
    
    # Get next earnings date
    ticker_earnings[date_col] = pd.to_datetime(ticker_earnings[date_col])
    future_earnings = ticker_earnings[ticker_earnings[date_col] >= datetime.now()]
    
    if len(future_earnings) == 0:
        days_to_earnings = 999
    else:
        next_earnings = future_earnings.iloc[0][date_col]
        days_to_earnings = (next_earnings - datetime.now()).days
    
    return {
        'days_to_earnings': days_to_earnings,
        'near_earnings': 1 if days_to_earnings <= 7 else 0,
        'avoid_earnings': 1 if days_to_earnings <= 2 else 0  # Don't trade 0-2 days before
    }


# ============================================================================
# STEP 4: FUTURES SIGNAL
# Expected Impact: +1-2% accuracy
# ============================================================================

def get_futures_signal():
    """
    Get overnight futures movement
    
    Returns:
        dict with futures features
    """
    try:
        # S&P 500 futures
        es = yf.Ticker("ES=F")
        es_hist = es.history(period="2d", interval="1h")
        
        if len(es_hist) < 2:
            return None
        
        # Compare current to yesterday close
        yesterday_close = es_hist.iloc[-24]['Close']  # 24 hours ago
        current = es_hist.iloc[-1]['Close']
        
        futures_change_pct = (current / yesterday_close - 1) * 100
        
        return {
            'futures_change_pct': futures_change_pct,
            'futures_bullish': 1 if futures_change_pct > 0.5 else 0,
            'futures_bearish': 1 if futures_change_pct < -0.5 else 0,
            'futures_strong_bullish': 1 if futures_change_pct > 1 else 0,
            'futures_strong_bearish': 1 if futures_change_pct < -1 else 0
        }
    
    except Exception as e:
        print(f"Error fetching futures: {e}")
        return None


# ============================================================================
# STEP 5: COMBINED FEATURE GENERATOR
# ============================================================================

def generate_1day_features(ticker, earnings_df=None):
    """
    Generate all quick-win 1-day features
    
    Returns:
        dict with all features
    """
    features = {}
    
    print(f"\nGenerating 1-day features for {ticker}...")
    
    # 1. Pre-market signals
    print("  Fetching pre-market data...")
    premarket = fetch_premarket_signals(ticker)
    if premarket:
        features.update(premarket)
        print(f"    ✓ Gap: {premarket['premarket_gap_pct']:.2f}%")
    
    # 2. Market regime
    print("  Checking market regime...")
    regime = get_market_regime()
    if regime:
        features.update(regime)
        print(f"    ✓ VIX: {regime['vix_level']:.1f} ({regime['vix_regime']})")
    
    # 3. Earnings proximity
    print("  Checking earnings...")
    earnings = get_earnings_features(ticker, earnings_df)
    if earnings:
        features.update(earnings)
        print(f"    ✓ Days to earnings: {earnings['days_to_earnings']}")
    
    # 4. Futures
    print("  Checking futures...")
    futures = get_futures_signal()
    if futures:
        features.update(futures)
        print(f"    ✓ Futures: {futures['futures_change_pct']:.2f}%")
    
    return features


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("1-DAY FEATURE GENERATOR - QUICK WINS")
    print("="*70)
    
    # Test stocks
    test_tickers = ['AAPL', 'GOOGL', 'JPM', 'TLRY', 'AMC']
    
    for ticker in test_tickers:
        features = generate_1day_features(ticker)
        
        print(f"\n{ticker} Summary:")
        print(f"  Pre-market: {features.get('premarket_gap_pct', 0):.2f}% gap")
        print(f"  Market: {features.get('vix_regime', 'unknown')}")
        print(f"  Earnings: {features.get('days_to_earnings', 999)} days")
        print(f"  Futures: {features.get('futures_change_pct', 0):.2f}%")
        
        # Generate signal
        signal_strength = 0
        
        # Pre-market bullish
        if features.get('premarket_gap_up', 0) and features.get('premarket_volume_spike', 0):
            signal_strength += 1
            print("  📈 BULLISH: Large gap up with volume")
        
        # Pre-market bearish
        if features.get('premarket_gap_down', 0) and features.get('premarket_volume_spike', 0):
            signal_strength -= 1
            print("  📉 BEARISH: Large gap down with volume")
        
        # High VIX + futures up = buying opportunity
        if features.get('vix_high', 0) and features.get('futures_bullish', 0):
            signal_strength += 1
            print("  📈 BULLISH: High VIX + futures up")
        
        # Near earnings = avoid
        if features.get('avoid_earnings', 0):
            signal_strength = 0
            print("  ⚠️  AVOID: Too close to earnings")
        
        print(f"  Signal: {signal_strength:+d}")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("1. Integrate these features into train_refined_models.py")
    print("2. Add to create_optimized_features() function")
    print("3. Retrain models with new features")
    print("4. Expected improvement: +4-8% accuracy")
    print("="*70)
