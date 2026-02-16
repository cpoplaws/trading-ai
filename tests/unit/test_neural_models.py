#!/usr/bin/env python3
"""
Test Enhanced Neural Network Models

This script tests the new LSTM and hybrid models on our trading data.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Add project root to path
sys.path.append('/workspaces/trading-ai')

from src.modeling.enhanced_trainer import train_comprehensive_models, ModelTrainer
from src.modeling.neural_models import LSTMTradingModel, HybridTradingModel
from src.utils.logger import setup_logger

def test_neural_models():
    """Test the neural network models with our trading data."""
    
    # Setup logging
    setup_logger()
    logger = logging.getLogger(__name__)
    
    print("=" * 80)
    print("Testing Enhanced Neural Network Models for Trading AI")
    print("=" * 80)
    
    # Test data paths
    data_files = [
        '/workspaces/trading-ai/data/processed/AAPL.csv',
        '/workspaces/trading-ai/data/processed/SPY.csv',
        '/workspaces/trading-ai/data/processed/MSFT.csv'
    ]
    
    results_summary = {}
    
    for data_file in data_files:
        if not os.path.exists(data_file):
            print(f"❌ Data file not found: {data_file}")
            continue
        
        ticker = os.path.basename(data_file).split('.')[0]
        print(f"\n📊 Testing models on {ticker}...")
        
        try:
            # Load and examine data
            df = pd.read_csv(data_file, index_col=0, parse_dates=True)
            print(f"   Data shape: {df.shape}")
            print(f"   Date range: {df.index.min()} to {df.index.max()}")
            print(f"   Columns: {list(df.columns)}")
            
            # Check if we have enough data for neural networks
            if len(df) < 100:
                print(f"   ⚠️  Insufficient data for neural networks ({len(df)} samples)")
                continue
            
            # Run comprehensive training
            print(f"   🚀 Starting comprehensive model training...")
            
            results = train_comprehensive_models(
                file_path=data_file,
                save_path=f'/workspaces/trading-ai/models/{ticker}/',
                include_traditional=True,
                include_neural=True
            )
            
            if 'error' in results:
                print(f"   ❌ Training failed: {results['error']}")
                results_summary[ticker] = {'status': 'failed', 'error': results['error']}
                continue
            
            # Display results
            print(f"   ✅ Training completed successfully!")
            
            # Data info
            data_info = results.get('data_info', {})
            print(f"   📈 Processed {data_info.get('total_samples', 0)} samples")
            target_dist = data_info.get('target_distribution', {})
            if target_dist:
                print(f"   🎯 Target distribution: {target_dist}")
            
            # Model results
            models = results.get('models', {})
            print(f"   🤖 Trained {len(models)} models:")
            
            for model_name, model_results in models.items():
                if isinstance(model_results, dict) and 'error' not in model_results:
                    if 'accuracy' in model_results:
                        accuracy = model_results['accuracy']
                        print(f"      - {model_name}: {accuracy:.4f} accuracy")
                    elif 'lstm_metrics' in model_results:
                        # Hybrid model
                        lstm_acc = model_results['lstm_metrics'].get('accuracy', 0)
                        rf_acc = model_results.get('rf_accuracy', 0)
                        print(f"      - {model_name}: LSTM {lstm_acc:.4f}, RF {rf_acc:.4f}")
                else:
                    print(f"      - {model_name}: Failed ({model_results.get('error', 'Unknown error')})")
            
            # Best model
            comparison = results.get('model_comparison', {})
            if comparison and 'best_model' in comparison:
                best_model = comparison['best_model']
                best_accuracy = comparison['summary'].get('best_accuracy', 0)
                print(f"   🏆 Best model: {best_model} ({best_accuracy:.4f} accuracy)")
            
            results_summary[ticker] = {
                'status': 'success',
                'models_trained': len(models),
                'best_model': comparison.get('best_model'),
                'best_accuracy': comparison.get('summary', {}).get('best_accuracy', 0),
                'data_samples': data_info.get('total_samples', 0)
            }
            
        except Exception as e:
            print(f"   ❌ Error testing {ticker}: {str(e)}")
            results_summary[ticker] = {'status': 'error', 'error': str(e)}
    
    # Summary
    print("\n" + "=" * 80)
    print("TESTING SUMMARY")
    print("=" * 80)
    
    for ticker, result in results_summary.items():
        status = result['status']
        if status == 'success':
            best_model = result.get('best_model', 'N/A')
            best_acc = result.get('best_accuracy', 0)
            samples = result.get('data_samples', 0)
            print(f"✅ {ticker}: {best_model} ({best_acc:.4f}) - {samples} samples")
        elif status == 'failed':
            print(f"❌ {ticker}: Training failed - {result.get('error', 'Unknown')}")
        else:
            print(f"⚠️  {ticker}: Error - {result.get('error', 'Unknown')}")
    
    # Test individual neural models if we have data
    if results_summary:
        print("\n" + "=" * 80)
        print("TESTING INDIVIDUAL NEURAL MODELS")
        print("=" * 80)
        
        # Test with the first successful ticker
        successful_tickers = [t for t, r in results_summary.items() if r['status'] == 'success']
        if successful_tickers:
            test_ticker = successful_tickers[0]
            test_file = f'/workspaces/trading-ai/data/processed/{test_ticker}.csv'
            
            print(f"🧪 Testing individual neural models with {test_ticker}...")
            
            try:
                df = pd.read_csv(test_file, index_col=0, parse_dates=True)
                
                # Test LSTM model directly
                print("   Testing LSTM model...")
                lstm_model = LSTMTradingModel(sequence_length=30)
                
                # Prepare data
                df.columns = df.columns.str.lower()
                if 'target' not in df.columns:
                    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
                df = df.dropna(subset=['target'])
                
                if len(df) >= 60:
                    metrics = lstm_model.train(df, 'target', epochs=5)  # Quick test
                    if 'error' not in metrics:
                        print(f"   ✅ LSTM: {metrics.get('accuracy', 0):.4f} accuracy")
                        
                        # Test prediction
                        predictions = lstm_model.predict(df.tail(100))
                        print(f"   📈 Generated {len(predictions)} predictions")
                    else:
                        print(f"   ❌ LSTM failed: {metrics['error']}")
                else:
                    print("   ⚠️  Insufficient data for LSTM test")
                
            except Exception as e:
                print(f"   ❌ Individual test failed: {str(e)}")
    
    print("\n🎉 Neural network testing completed!")
    return results_summary


def verify_model_files():
    """Verify that model files were created correctly."""
    
    print("\n" + "=" * 80)
    print("VERIFYING SAVED MODELS")
    print("=" * 80)
    
    models_dir = '/workspaces/trading-ai/models'
    
    if not os.path.exists(models_dir):
        print("❌ Models directory not found")
        return
    
    # Check for subdirectories (one per ticker)
    subdirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    
    print(f"📁 Found {len(subdirs)} model directories: {subdirs}")
    
    for subdir in subdirs:
        subdir_path = os.path.join(models_dir, subdir)
        files = os.listdir(subdir_path)
        
        print(f"\n📂 {subdir}:")
        
        # Check for different model types
        traditional_models = [f for f in files if f.endswith('.joblib') and ('random_forest' in f or 'gradient_boosting' in f)]
        neural_models = [f for f in files if f.endswith('.h5') or 'lstm' in f or 'hybrid' in f]
        result_files = [f for f in files if 'training_results' in f and f.endswith('.json')]
        
        print(f"   🤖 Traditional models: {len(traditional_models)}")
        for model in traditional_models:
            print(f"      - {model}")
        
        print(f"   🧠 Neural models: {len(neural_models)}")
        for model in neural_models:
            print(f"      - {model}")
        
        print(f"   📊 Result files: {len(result_files)}")
        for result in result_files:
            print(f"      - {result}")


if __name__ == "__main__":
    # Run the comprehensive test
    try:
        results = test_neural_models()
        verify_model_files()
        
        print("\n🎯 Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
