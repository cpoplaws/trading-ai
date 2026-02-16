"""
Enhanced Model Training System

This module provides a unified interface for training different types of models:
- Traditional ML (Random Forest, XGBoost, etc.)
- Neural Networks (LSTM, Transformer, etc.)
- Ensemble methods

It includes advanced features like:
- Hyperparameter optimization
- Cross-validation
- Model comparison
- Automated feature selection
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime

# Traditional ML
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib

# Neural Networks
from .neural_models import LSTMTradingModel, HybridTradingModel, train_neural_model

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Unified model training system for traditional ML and neural networks.
    """
    
    def __init__(self, save_path: str = './models/'):
        """
        Initialize the model trainer.
        
        Args:
            save_path: Directory to save trained models
        """
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        
        self.models = {}
        self.model_performances = {}
        
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'target') -> Tuple[pd.DataFrame, bool]:
        """
        Prepare data for training with comprehensive preprocessing.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            
        Returns:
            Tuple of (processed_df, success)
        """
        try:
            # Standardize column names
            df.columns = df.columns.str.lower()
            
            # Create target if it doesn't exist
            if target_col not in df.columns:
                if 'close' in df.columns:
                    # Look ahead 1 period for next day prediction
                    df[target_col] = (df['close'].shift(-1) > df['close']).astype(int)
                    logger.info("Created binary target: 1 if next day close > today close, 0 otherwise")
                else:
                    raise ValueError("No 'close' column found for target creation")
            
            # Remove rows with NaN target (last row typically)
            df = df.dropna(subset=[target_col])
            
            # Ensure all numeric columns are properly converted
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Forward fill any remaining NaN values in features
            feature_cols = [col for col in df.columns if col != target_col]
            df[feature_cols] = df[feature_cols].ffill()
            
            # Drop any remaining rows with NaN
            df = df.dropna()
            
            logger.info(f"Data prepared: {len(df)} samples, target distribution: {df[target_col].value_counts().to_dict()}")
            
            return df, True
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            return df, False
    
    def train_traditional_models(self, df: pd.DataFrame, target_col: str = 'target') -> Dict[str, Any]:
        """
        Train traditional ML models with hyperparameter optimization.
        
        Args:
            df: Training data
            target_col: Target column name
            
        Returns:
            Dictionary with training results
        """
        results = {}
        
        try:
            # Define feature columns (prefer technical indicators)
            feature_priority = [
                ['sma_10', 'sma_30', 'rsi_14', 'volatility_20'],  # Primary technical indicators
                ['close', 'volume', 'high', 'low'],              # Price/volume features
            ]
            
            available_features = []
            for feature_group in feature_priority:
                group_features = [f for f in feature_group if f in df.columns]
                available_features.extend(group_features)
                if len(available_features) >= 4:  # Minimum features for good model
                    break
            
            if len(available_features) < 2:
                raise ValueError("Insufficient features for training")
            
            logger.info(f"Using features: {available_features}")
            
            # Prepare features and target
            X = df[available_features].copy()
            y = df[target_col].copy()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features for models that benefit from it
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Define models with hyperparameters
            model_configs = {
                'random_forest': {
                    'model': RandomForestClassifier(random_state=42),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [5, 10, 15, None],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    },
                    'use_scaled': False
                },
                'gradient_boosting': {
                    'model': GradientBoostingClassifier(random_state=42),
                    'params': {
                        'n_estimators': [50, 100, 150],
                        'learning_rate': [0.05, 0.1, 0.15],
                        'max_depth': [3, 5, 7],
                        'min_samples_split': [2, 5, 10]
                    },
                    'use_scaled': False
                }
            }
            
            # Train each model
            for model_name, config in model_configs.items():
                try:
                    logger.info(f"Training {model_name}...")
                    
                    # Use scaled or unscaled data
                    X_train_model = X_train_scaled if config['use_scaled'] else X_train
                    X_test_model = X_test_scaled if config['use_scaled'] else X_test
                    
                    # Grid search for best parameters
                    grid_search = GridSearchCV(
                        config['model'], 
                        config['params'], 
                        cv=3, 
                        scoring='accuracy',
                        n_jobs=-1,
                        verbose=0
                    )
                    
                    grid_search.fit(X_train_model, y_train)
                    
                    # Get best model
                    best_model = grid_search.best_estimator_
                    
                    # Evaluate
                    y_pred = best_model.predict(X_test_model)
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    # Cross-validation score
                    cv_scores = cross_val_score(best_model, X_train_model, y_train, cv=5)
                    
                    model_results = {
                        'model': best_model,
                        'scaler': scaler if config['use_scaled'] else None,
                        'features': available_features,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'best_params': grid_search.best_params_,
                        'train_samples': len(X_train),
                        'test_samples': len(X_test)
                    }
                    
                    # Save model
                    model_filename = f"{model_name}_optimized.joblib"
                    model_path = os.path.join(self.save_path, model_filename)
                    joblib.dump({
                        'model': best_model,
                        'scaler': scaler if config['use_scaled'] else None,
                        'features': available_features,
                        'metadata': model_results
                    }, model_path)
                    
                    self.models[model_name] = model_results
                    results[model_name] = model_results
                    
                    logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                    
                except Exception as e:
                    logger.error(f"Error training {model_name}: {str(e)}")
                    results[model_name] = {'error': str(e)}
            
            return results
            
        except Exception as e:
            logger.error(f"Error in traditional model training: {str(e)}")
            return {'error': str(e)}
    
    def train_neural_models(self, df: pd.DataFrame, target_col: str = 'target') -> Dict[str, Any]:
        """
        Train neural network models.
        
        Args:
            df: Training data
            target_col: Target column name
            
        Returns:
            Dictionary with training results
        """
        results = {}
        
        try:
            # LSTM Model
            logger.info("Training LSTM model...")
            lstm_success, lstm_metrics = train_neural_model(
                df=df, 
                save_path=self.save_path,
                model_type='lstm',
                sequence_length=30  # Shorter sequence for more training data
            )
            
            if lstm_success:
                results['lstm'] = lstm_metrics
                logger.info(f"LSTM trained successfully - Accuracy: {lstm_metrics.get('accuracy', 0):.4f}")
            else:
                results['lstm'] = lstm_metrics
                logger.error(f"LSTM training failed: {lstm_metrics}")
            
            # Hybrid Model
            logger.info("Training Hybrid model...")
            hybrid_success, hybrid_metrics = train_neural_model(
                df=df,
                save_path=self.save_path,
                model_type='hybrid',
                sequence_length=30
            )
            
            if hybrid_success:
                results['hybrid'] = hybrid_metrics
                logger.info(f"Hybrid model trained successfully")
            else:
                results['hybrid'] = hybrid_metrics
                logger.error(f"Hybrid training failed: {hybrid_metrics}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in neural model training: {str(e)}")
            return {'error': str(e)}
    
    def train_all_models(self, df: pd.DataFrame, target_col: str = 'target',
                        include_traditional: bool = True, include_neural: bool = True) -> Dict[str, Any]:
        """
        Train all available models and compare performance.
        
        Args:
            df: Training data
            target_col: Target column name
            include_traditional: Whether to train traditional ML models
            include_neural: Whether to train neural network models
            
        Returns:
            Comprehensive training results
        """
        try:
            # Prepare data
            df_processed, success = self.prepare_data(df, target_col)
            if not success:
                return {'error': 'Data preparation failed'}
            
            results = {
                'data_info': {
                    'total_samples': len(df_processed),
                    'features': list(df_processed.columns),
                    'target_distribution': df_processed[target_col].value_counts().to_dict()
                },
                'models': {}
            }
            
            # Train traditional models
            if include_traditional:
                logger.info("Training traditional ML models...")
                traditional_results = self.train_traditional_models(df_processed, target_col)
                results['models'].update(traditional_results)
            
            # Train neural models
            if include_neural and len(df_processed) >= 100:  # Minimum data for neural networks
                logger.info("Training neural network models...")
                neural_results = self.train_neural_models(df_processed, target_col)
                results['models'].update(neural_results)
            elif include_neural:
                logger.warning("Insufficient data for neural networks (need >= 100 samples)")
            
            # Compare models
            if results['models']:
                results['model_comparison'] = self._compare_models(results['models'])
            
            # Save results
            results_file = os.path.join(self.save_path, f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(results_file, 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                json_results = self._convert_for_json(results)
                json.dump(json_results, f, indent=2)
            
            logger.info(f"Training completed. Results saved to {results_file}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive training: {str(e)}")
            return {'error': str(e)}
    
    def _compare_models(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare model performances.
        
        Args:
            model_results: Dictionary of model results
            
        Returns:
            Model comparison summary
        """
        comparison = {
            'rankings': {},
            'best_model': None,
            'summary': {}
        }
        
        try:
            # Extract accuracy scores
            accuracies = {}
            for model_name, results in model_results.items():
                if isinstance(results, dict) and 'accuracy' in results:
                    accuracies[model_name] = results['accuracy']
                elif isinstance(results, dict) and 'lstm_metrics' in results:
                    # For hybrid models
                    accuracies[model_name] = results['lstm_metrics'].get('accuracy', 0)
            
            if accuracies:
                # Rank by accuracy
                sorted_models = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
                comparison['rankings']['by_accuracy'] = sorted_models
                comparison['best_model'] = sorted_models[0][0]
                
                # Summary statistics
                comparison['summary'] = {
                    'best_accuracy': sorted_models[0][1],
                    'worst_accuracy': sorted_models[-1][1],
                    'mean_accuracy': np.mean(list(accuracies.values())),
                    'accuracy_std': np.std(list(accuracies.values()))
                }
            
        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
        
        return comparison
    
    def _convert_for_json(self, obj: Any) -> Any:
        """
        Convert numpy types to native Python types for JSON serialization.
        Excludes non-serializable objects like trained models and scalers.
        """
        # Skip non-serializable objects
        if hasattr(obj, 'fit') and hasattr(obj, 'predict'):  # ML models
            return f"<Model: {type(obj).__name__}>"
        elif hasattr(obj, 'transform') and hasattr(obj, 'fit_transform'):  # Scalers
            return f"<Scaler: {type(obj).__name__}>"
        elif isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                # Skip model and scaler objects in dictionaries
                if key in ['model', 'scaler'] and (
                    (hasattr(value, 'fit') and hasattr(value, 'predict')) or
                    (hasattr(value, 'transform') and hasattr(value, 'fit_transform')) or
                    value is None
                ):
                    continue  # Skip these keys entirely
                result[key] = self._convert_for_json(value)
            return result
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj


def train_comprehensive_models(file_path: str, save_path: str = './models/',
                             include_traditional: bool = True, include_neural: bool = True) -> Dict[str, Any]:
    """
    Convenience function to train all models for a given dataset.
    
    Args:
        file_path: Path to the processed data CSV
        save_path: Directory to save models
        include_traditional: Whether to include traditional ML models
        include_neural: Whether to include neural network models
        
    Returns:
        Training results
    """
    try:
        # Load data
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        # Initialize trainer
        trainer = ModelTrainer(save_path=save_path)
        
        # Train all models
        results = trainer.train_all_models(
            df, 
            include_traditional=include_traditional,
            include_neural=include_neural
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Error in comprehensive training: {str(e)}")
        return {'error': str(e)}


if __name__ == "__main__":
    # Test the enhanced training system
    import sys
    sys.path.append('/workspaces/trading-ai')
    
    # Test with AAPL data
    results = train_comprehensive_models(
        file_path='/workspaces/trading-ai/data/processed/AAPL.csv',
        save_path='/workspaces/trading-ai/models/',
        include_traditional=True,
        include_neural=True
    )
    
    if 'error' not in results:
        print("Training completed successfully!")
        if 'model_comparison' in results:
            best_model = results['model_comparison'].get('best_model')
            best_accuracy = results['model_comparison']['summary'].get('best_accuracy', 0)
            print(f"Best model: {best_model} (Accuracy: {best_accuracy:.4f})")
    else:
        print(f"Training failed: {results['error']}")
