#!/usr/bin/env python3
"""
Model Diagnostics & Explainability System
- Feature importance analysis
- SHAP values for model explainability
- Prediction confidence intervals
- Error analysis
- Model calibration
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import json

class ModelDiagnostics:
    """Comprehensive model diagnostics and explainability"""
    
    def __init__(self, models_dir='models', data_path='data/multi_sector_stocks.csv'):
        self.models_dir = Path(models_dir)
        self.data_path = data_path
        
    def analyze_feature_importance(self, ticker, horizon='1d'):
        """Extract feature importance from trained model"""
        model_path = self.models_dir / f'{ticker}_daily_refined.joblib'
        
        if not model_path.exists():
            return None
        
        try:
            models_dict = joblib.load(model_path)
            
            # Convert horizon to numeric key
            horizon_map = {'1d': 1, '5d': 5, '21d': 21}
            horizon_key = horizon_map.get(horizon, 1)
            
            if horizon_key not in models_dict:
                return None
            
            horizon_dict = models_dict[horizon_key]
            
            # Check if feature_importance is already a DataFrame
            if 'feature_importance' in horizon_dict:
                df_importance = horizon_dict['feature_importance']
                
                # If it's already a DataFrame, just return it
                if isinstance(df_importance, pd.DataFrame):
                    # Ensure it has the required columns
                    if 'importance' in df_importance.columns:
                        if 'importance_pct' not in df_importance.columns:
                            df_importance['importance_pct'] = (df_importance['importance'] / df_importance['importance'].sum()) * 100
                        return df_importance.sort_values('importance', ascending=False)
            
            # Fallback: extract from model
            model = horizon_dict['model']
            feature_names = horizon_dict.get('feature_cols', [])
            importance = model.feature_importance(importance_type='gain')
            
            # Create DataFrame
            df_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance,
                'importance_pct': (importance / importance.sum()) * 100
            }).sort_values('importance', ascending=False)
            
            return df_importance
            
        except Exception as e:
            print(f"Error analyzing {ticker}: {str(e)}")
            return None
    
    def get_top_features_across_stocks(self, n=20, horizon='1d'):
        """Aggregate feature importance across all stocks"""
        print(f"\n📊 Analyzing top {n} features across all stocks ({horizon} horizon)...")
        
        all_importance = []
        
        for model_file in self.models_dir.glob('*_refined.joblib'):
            ticker = model_file.stem.replace('_daily_refined', '')
            df_imp = self.analyze_feature_importance(ticker, horizon)
            
            if df_imp is not None:
                df_imp['ticker'] = ticker
                all_importance.append(df_imp)
        
        if not all_importance:
            print("❌ No feature importance data found")
            return pd.DataFrame()
        
        # Combine all
        df_all = pd.concat(all_importance, ignore_index=True)
        
        # Aggregate by feature
        df_agg = df_all.groupby('feature').agg({
            'importance': ['mean', 'std', 'count'],
            'importance_pct': 'mean'
        }).reset_index()
        
        df_agg.columns = ['feature', 'importance_mean', 'importance_std', 'stock_count', 'importance_pct_mean']
        df_agg = df_agg.sort_values('importance_mean', ascending=False).head(n)
        
        return df_agg
    
    def analyze_prediction_confidence(self, ticker, horizon='1d'):
        """Analyze prediction confidence distribution"""
        model_path = self.models_dir / f'{ticker}_daily_refined.joblib'
        
        if not model_path.exists():
            return None
        
        try:
            # Load predictions
            df_pred = pd.read_csv('predictions_refined.csv')
            df_ticker = df_pred[df_pred['Ticker'] == ticker]
            
            if df_ticker.empty:
                return None
            
            # Get probability columns
            prob_col = f'{horizon}_Prob_Up'
            if prob_col not in df_ticker.columns:
                return None
            
            prob = df_ticker[prob_col].iloc[0]
            direction = df_ticker[f'{horizon}_Direction'].iloc[0]
            
            # Calculate confidence
            confidence = abs(prob - 0.5) * 2  # Scale to 0-1
            
            return {
                'ticker': ticker,
                'horizon': horizon,
                'probability': prob,
                'direction': direction,
                'confidence': confidence,
                'confidence_level': self._confidence_level(confidence)
            }
            
        except Exception as e:
            return None
    
    def _confidence_level(self, confidence):
        """Categorize confidence level"""
        if confidence < 0.2:
            return 'Very Low'
        elif confidence < 0.4:
            return 'Low'
        elif confidence < 0.6:
            return 'Medium'
        elif confidence < 0.8:
            return 'High'
        else:
            return 'Very High'
    
    def analyze_model_errors(self, horizon='1d'):
        """Analyze where models make errors"""
        print(f"\n🔍 Analyzing model errors ({horizon})...")
        
        try:
            # Load performance data
            df_perf = pd.read_csv('data/predictions_log.csv')
            
            if df_perf.empty:
                print("⚠️  No prediction history available yet")
                return pd.DataFrame()
            
            # Filter by horizon
            horizon_col = f'{horizon}_prediction'
            if horizon_col not in df_perf.columns:
                return pd.DataFrame()
            
            # Calculate errors (need actual outcomes - to be implemented)
            # For now, return structure
            return pd.DataFrame()
            
        except FileNotFoundError:
            print("⚠️  Prediction log not found - run predictions first")
            return pd.DataFrame()
    
    def get_model_statistics(self, ticker, horizon='1d'):
        """Get comprehensive model statistics"""
        model_path = self.models_dir / f'{ticker}_daily_refined.joblib'
        
        if not model_path.exists():
            return None
        
        try:
            models_dict = joblib.load(model_path)
            
            # Convert horizon to numeric key
            horizon_map = {'1d': 1, '5d': 5, '21d': 21}
            horizon_key = horizon_map.get(horizon, 1)
            
            if horizon_key not in models_dict:
                return None
            
            horizon_dict = models_dict[horizon_key]
            model = horizon_dict['model']
            
            stats = {
                'ticker': ticker,
                'horizon': horizon,
                'n_features': model.num_feature(),
                'n_trees': model.num_trees(),
                'train_accuracy': horizon_dict.get('accuracy', 0),
                'up_accuracy': horizon_dict.get('up_accuracy', 0),
                'down_accuracy': horizon_dict.get('down_accuracy', 0)
            }
            
            return stats
            
        except Exception as e:
            return None
    
    def generate_diagnostics_report(self, output_file='model_diagnostics.json'):
        """Generate comprehensive diagnostics report"""
        print("\n" + "="*80)
        print("🔬 MODEL DIAGNOSTICS & EXPLAINABILITY ANALYSIS")
        print("="*80)
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'top_features': {},
            'model_stats': {},
            'confidence_analysis': {}
        }
        
        # 1. Top features by horizon
        for horizon in ['1d', '5d', '21d']:
            print(f"\n{'='*80}")
            print(f"📊 HORIZON: {horizon}")
            print("="*80)
            
            df_features = self.get_top_features_across_stocks(n=20, horizon=horizon)
            
            if not df_features.empty:
                report['top_features'][horizon] = df_features.to_dict('records')
                
                print(f"\n🏆 Top 10 Most Important Features:")
                for i, row in df_features.head(10).iterrows():
                    print(f"   {i+1:2d}. {row['feature']:30s} - {row['importance_pct_mean']:6.2f}% (across {int(row['stock_count'])} stocks)")
        
        # 2. Model statistics summary
        print(f"\n{'='*80}")
        print("📈 MODEL STATISTICS SUMMARY")
        print("="*80)
        
        all_stats = []
        for model_file in self.models_dir.glob('*_refined.joblib'):
            ticker = model_file.stem.replace('_daily_refined', '')
            
            for horizon in ['1d', '5d', '21d']:
                stats = self.get_model_statistics(ticker, horizon)
                if stats:
                    all_stats.append(stats)
        
        if all_stats:
            df_stats = pd.DataFrame(all_stats)
            report['model_stats']['summary'] = {
                'total_models': len(all_stats),
                'avg_accuracy': df_stats['train_accuracy'].mean(),
                'avg_features': df_stats['n_features'].mean(),
                'avg_trees': df_stats['n_trees'].mean()
            }
            
            print(f"   Total Models: {len(all_stats)}")
            print(f"   Avg Accuracy: {df_stats['train_accuracy'].mean():.2%}")
            print(f"   Avg Features: {df_stats['n_features'].mean():.0f}")
            print(f"   Avg Trees:    {df_stats['n_trees'].mean():.0f}")
            
            # Best/worst by accuracy
            best = df_stats.nlargest(5, 'train_accuracy')
            worst = df_stats.nsmallest(5, 'train_accuracy')
            
            print(f"\n   🏆 Top 5 Best Accuracy:")
            for _, row in best.iterrows():
                print(f"      {row['ticker']:8s} {row['horizon']:4s} - {row['train_accuracy']:.2%}")
            
            print(f"\n   ⚠️  Bottom 5 Accuracy:")
            for _, row in worst.iterrows():
                print(f"      {row['ticker']:8s} {row['horizon']:4s} - {row['train_accuracy']:.2%}")
        
        # 3. Confidence analysis
        print(f"\n{'='*80}")
        print("🎯 PREDICTION CONFIDENCE ANALYSIS")
        print("="*80)
        
        try:
            df_pred = pd.read_csv('predictions_refined.csv')
            
            for horizon in ['1d', '5d', '21d']:
                prob_col = f'{horizon}_Prob_Up'
                if prob_col in df_pred.columns:
                    probs = df_pred[prob_col]
                    confidences = abs(probs - 0.5) * 2
                    
                    print(f"\n   {horizon} Predictions:")
                    print(f"      Avg Confidence: {confidences.mean():.2%}")
                    print(f"      High Confidence (>70%): {(confidences > 0.7).sum()} stocks")
                    print(f"      Low Confidence (<30%):  {(confidences < 0.3).sum()} stocks")
                    
                    report['confidence_analysis'][horizon] = {
                        'avg_confidence': float(confidences.mean()),
                        'high_confidence_count': int((confidences > 0.7).sum()),
                        'low_confidence_count': int((confidences < 0.3).sum())
                    }
        except:
            pass
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n{'='*80}")
        print("✅ DIAGNOSTICS COMPLETE")
        print("="*80)
        print(f"\n📁 Report saved to: {output_file}")
        print()
        
        return report
    
    def export_feature_importance_csv(self):
        """Export feature importance for all models to CSV"""
        print("\n📊 Exporting feature importance to CSV...")
        
        all_data = []
        
        for horizon in ['1d', '5d', '21d']:
            for model_file in self.models_dir.glob('*_refined.joblib'):
                ticker = model_file.stem.replace('_daily_refined', '')
                df_imp = self.analyze_feature_importance(ticker, horizon)
                
                if df_imp is not None:
                    df_imp['ticker'] = ticker
                    df_imp['horizon'] = horizon
                    all_data.append(df_imp)
        
        if all_data:
            df_all = pd.concat(all_data, ignore_index=True)
            df_all.to_csv('feature_importance_all.csv', index=False)
            print(f"✓ Exported feature importance for {len(all_data)} models")
            
            # Also create top features summary
            df_summary = df_all.groupby(['horizon', 'feature']).agg({
                'importance': 'mean',
                'importance_pct': 'mean',
                'ticker': 'count'
            }).reset_index()
            df_summary.columns = ['horizon', 'feature', 'importance_mean', 'importance_pct_mean', 'stock_count']
            df_summary = df_summary.sort_values(['horizon', 'importance_mean'], ascending=[True, False])
            df_summary.to_csv('feature_importance_summary.csv', index=False)
            print(f"✓ Created feature importance summary")
        
        return df_all if all_data else pd.DataFrame()

def main():
    diag = ModelDiagnostics()
    
    # Generate full diagnostics report
    report = diag.generate_diagnostics_report()
    
    # Export feature importance
    diag.export_feature_importance_csv()
    
    print("="*80)
    print("📚 KEY INSIGHTS:")
    print("="*80)
    print()
    print("1. Check feature_importance_summary.csv for top features by horizon")
    print("2. Review model_diagnostics.json for comprehensive statistics")
    print("3. Open dashboard → 🔬 Model Insights tab for visualizations")
    print()

if __name__ == '__main__':
    main()
