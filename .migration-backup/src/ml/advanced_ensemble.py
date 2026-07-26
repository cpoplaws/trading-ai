"""
Advanced Ensemble Models for Trading
=====================================

Implements sophisticated ensemble methods combining multiple model types
for robust predictions.

Models Included:
- Stacking Ensemble (meta-learning)
- Weighted Voting Ensemble
- Boosting Ensemble (XGBoost, LightGBM)
- Bagging Ensemble (Random Forest)
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try importing ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available. Install with: pip install lightgbm")

try:
    from sklearn.ensemble import (
        RandomForestRegressor, RandomForestClassifier,
        GradientBoostingRegressor, GradientBoostingClassifier,
        AdaBoostRegressor, AdaBoostClassifier
    )
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.svm import SVR, SVC
    from sklearn.neural_network import MLPRegressor, MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Install with: pip install scikit-learn")


class EnsembleMethod(str, Enum):
    """Ensemble methods."""
    VOTING = "voting"           # Simple/weighted voting
    STACKING = "stacking"       # Meta-learning stacking
    BOOSTING = "boosting"       # Gradient boosting
    BAGGING = "bagging"         # Bootstrap aggregating


class PredictionTask(str, Enum):
    """Type of prediction task."""
    REGRESSION = "regression"   # Price prediction
    CLASSIFICATION = "classification"  # Direction prediction


@dataclass
class EnsembleConfig:
    """Configuration for ensemble model."""
    ensemble_method: EnsembleMethod = EnsembleMethod.STACKING
    prediction_task: PredictionTask = PredictionTask.REGRESSION

    # Base model weights (for voting)
    model_weights: Optional[Dict[str, float]] = None

    # Stacking meta-learner
    meta_learner: str = "ridge"  # ridge, logistic, mlp

    # Feature engineering
    use_feature_engineering: bool = True
    lag_features: int = 10
    rolling_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50])

    # Training
    train_split: float = 0.7
    val_split: float = 0.15
    random_state: int = 42

    # XGBoost params
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'objective': 'reg:squarederror',
        'subsample': 0.8,
        'colsample_bytree': 0.8
    })

    # LightGBM params
    lgb_params: Dict[str, Any] = field(default_factory=lambda: {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    })

    # Random Forest params
    rf_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42
    })


@dataclass
class PredictionResult:
    """Ensemble prediction result."""
    predicted_value: float
    confidence: float
    individual_predictions: Dict[str, float]
    model_weights: Dict[str, float]
    features_used: int
    timestamp: datetime


class AdvancedEnsemble:
    """
    Advanced Ensemble Model combining multiple ML approaches.

    Supports:
    - Stacking (meta-learning)
    - Weighted voting
    - Multiple base models (XGBoost, LightGBM, RF, etc.)
    """

    def __init__(self, config: Optional[EnsembleConfig] = None):
        """
        Initialize ensemble model.

        Args:
            config: Ensemble configuration
        """
        self.config = config or EnsembleConfig()
        self.base_models: Dict[str, Any] = {}
        self.meta_model: Optional[Any] = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.is_trained = False
        self.feature_names: List[str] = []

        logger.info(f"Initialized AdvancedEnsemble with {self.config.ensemble_method} method")

    def _create_base_models(self) -> Dict[str, Any]:
        """Create base models for ensemble."""
        models = {}

        is_regression = self.config.prediction_task == PredictionTask.REGRESSION

        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn required but not available")
            return models

        # Random Forest
        if is_regression:
            models['random_forest'] = RandomForestRegressor(**self.config.rf_params)
        else:
            models['random_forest'] = RandomForestClassifier(**self.config.rf_params)

        # Gradient Boosting
        if is_regression:
            models['gradient_boosting'] = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.config.random_state
            )
        else:
            models['gradient_boosting'] = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.config.random_state
            )

        # XGBoost
        if XGBOOST_AVAILABLE:
            if is_regression:
                models['xgboost'] = xgb.XGBRegressor(**self.config.xgb_params)
            else:
                xgb_clf_params = self.config.xgb_params.copy()
                xgb_clf_params['objective'] = 'binary:logistic'
                models['xgboost'] = xgb.XGBClassifier(**xgb_clf_params)

        # LightGBM
        if LIGHTGBM_AVAILABLE:
            if is_regression:
                models['lightgbm'] = lgb.LGBMRegressor(**self.config.lgb_params)
            else:
                models['lightgbm'] = lgb.LGBMClassifier(**self.config.lgb_params)

        # Neural Network
        if is_regression:
            models['mlp'] = MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                max_iter=500,
                random_state=self.config.random_state,
                early_stopping=True
            )
        else:
            models['mlp'] = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                max_iter=500,
                random_state=self.config.random_state,
                early_stopping=True
            )

        logger.info(f"Created {len(models)} base models: {list(models.keys())}")
        return models

    def _create_meta_model(self) -> Any:
        """Create meta-learner for stacking."""
        if not SKLEARN_AVAILABLE:
            return None

        is_regression = self.config.prediction_task == PredictionTask.REGRESSION

        if self.config.meta_learner == "ridge" and is_regression:
            return Ridge(alpha=1.0)
        elif self.config.meta_learner == "logistic" and not is_regression:
            return LogisticRegression(max_iter=1000)
        elif self.config.meta_learner == "mlp":
            if is_regression:
                return MLPRegressor(hidden_layer_sizes=(50,), max_iter=500)
            else:
                return MLPClassifier(hidden_layer_sizes=(50,), max_iter=500)
        else:
            # Default
            return Ridge(alpha=1.0) if is_regression else LogisticRegression()

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from raw price data."""
        if not self.config.use_feature_engineering:
            return df

        df = df.copy()

        # Price-based features
        if 'close' in df.columns:
            # Returns
            df['returns'] = df['close'].pct_change()

            # Lag features
            for lag in range(1, self.config.lag_features + 1):
                df[f'close_lag_{lag}'] = df['close'].shift(lag)
                df[f'returns_lag_{lag}'] = df['returns'].shift(lag)

            # Rolling statistics
            for window in self.config.rolling_windows:
                df[f'sma_{window}'] = df['close'].rolling(window).mean()
                df[f'std_{window}'] = df['close'].rolling(window).std()
                df[f'rsi_{window}'] = self._calculate_rsi(df['close'], window)

        # Volume features
        if 'volume' in df.columns:
            for window in self.config.rolling_windows:
                df[f'volume_sma_{window}'] = df['volume'].rolling(window).mean()

        # Drop NaN rows
        df = df.dropna()

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Train ensemble model.

        Args:
            X: Training features
            y: Training targets
            X_val: Validation features
            y_val: Validation targets

        Returns:
            Training metrics
        """
        logger.info(f"Training ensemble with {X.shape[0]} samples, {X.shape[1]} features")

        # Split data if validation not provided
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=self.config.val_split / (1 - self.config.train_split),
                random_state=self.config.random_state
            )
        else:
            X_train, y_train = X, y

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Create base models
        self.base_models = self._create_base_models()

        # Train base models
        base_predictions_train = {}
        base_predictions_val = {}

        for name, model in self.base_models.items():
            logger.info(f"Training {name}...")
            try:
                model.fit(X_train_scaled, y_train)

                # Get predictions
                base_predictions_train[name] = model.predict(X_train_scaled)
                base_predictions_val[name] = model.predict(X_val_scaled)

                # Calculate score
                score = model.score(X_val_scaled, y_val)
                logger.info(f"{name} validation score: {score:.4f}")

            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                continue

        # Ensemble method
        if self.config.ensemble_method == EnsembleMethod.STACKING:
            # Stack predictions for meta-learner
            X_meta_train = np.column_stack(list(base_predictions_train.values()))
            X_meta_val = np.column_stack(list(base_predictions_val.values()))

            # Train meta-learner
            self.meta_model = self._create_meta_model()
            logger.info("Training meta-learner...")
            self.meta_model.fit(X_meta_train, y_train)

            # Evaluate
            train_score = self.meta_model.score(X_meta_train, y_train)
            val_score = self.meta_model.score(X_meta_val, y_val)

            logger.info(f"Meta-learner train score: {train_score:.4f}")
            logger.info(f"Meta-learner val score: {val_score:.4f}")

            metrics = {
                'train_score': train_score,
                'val_score': val_score,
                'num_base_models': len(self.base_models)
            }

        elif self.config.ensemble_method == EnsembleMethod.VOTING:
            # Weighted voting - use validation scores as weights if not provided
            if self.config.model_weights is None:
                weights = {}
                for name, model in self.base_models.items():
                    score = model.score(X_val_scaled, y_val)
                    weights[name] = max(score, 0)  # Ensure non-negative

                # Normalize weights
                total_weight = sum(weights.values())
                if total_weight > 0:
                    weights = {k: v / total_weight for k, v in weights.items()}
                else:
                    # Equal weights if all scores <= 0
                    weights = {k: 1.0 / len(weights) for k in weights}

                self.config.model_weights = weights

            # Calculate ensemble prediction
            ensemble_pred_val = self._weighted_average(base_predictions_val)

            # Calculate R² score for regression
            if self.config.prediction_task == PredictionTask.REGRESSION:
                ss_res = np.sum((y_val - ensemble_pred_val) ** 2)
                ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
                val_score = 1 - (ss_res / ss_tot)
            else:
                # Accuracy for classification
                val_score = np.mean((ensemble_pred_val > 0.5) == y_val)

            logger.info(f"Ensemble validation score: {val_score:.4f}")
            logger.info(f"Model weights: {self.config.model_weights}")

            metrics = {
                'val_score': val_score,
                'model_weights': self.config.model_weights
            }

        self.is_trained = True
        return metrics

    def _weighted_average(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate weighted average of predictions."""
        if self.config.model_weights is None:
            # Equal weights
            return np.mean(list(predictions.values()), axis=0)

        weighted_sum = np.zeros_like(list(predictions.values())[0])
        for name, pred in predictions.items():
            weight = self.config.model_weights.get(name, 0)
            weighted_sum += weight * pred

        return weighted_sum

    def predict(self, X: np.ndarray) -> PredictionResult:
        """
        Make prediction using ensemble.

        Args:
            X: Features (single sample or batch)

        Returns:
            Prediction result with confidence
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Ensure 2D array
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Get base model predictions
        individual_preds = {}
        for name, model in self.base_models.items():
            try:
                pred = model.predict(X_scaled)
                individual_preds[name] = float(pred[0])
            except Exception as e:
                logger.warning(f"Error predicting with {name}: {e}")

        # Ensemble prediction
        if self.config.ensemble_method == EnsembleMethod.STACKING:
            # Stack predictions for meta-learner
            X_meta = np.array(list(individual_preds.values())).reshape(1, -1)
            final_pred = self.meta_model.predict(X_meta)[0]

            # Confidence from prediction variance
            pred_std = np.std(list(individual_preds.values()))
            confidence = 1.0 / (1.0 + pred_std)

        else:  # VOTING
            # Weighted average
            preds_array = {k: np.array([v]) for k, v in individual_preds.items()}
            final_pred = self._weighted_average(preds_array)[0]

            # Confidence from agreement
            pred_range = max(individual_preds.values()) - min(individual_preds.values())
            confidence = 1.0 / (1.0 + pred_range)

        return PredictionResult(
            predicted_value=float(final_pred),
            confidence=float(confidence),
            individual_predictions=individual_preds,
            model_weights=self.config.model_weights or {},
            features_used=X.shape[1],
            timestamp=datetime.utcnow()
        )

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from base models."""
        importance = {}

        for name, model in self.base_models.items():
            if hasattr(model, 'feature_importances_'):
                importance[name] = model.feature_importances_

        # Average importance across models
        if importance:
            avg_importance = np.mean(list(importance.values()), axis=0)
            return {f'feature_{i}': float(imp) for i, imp in enumerate(avg_importance)}

        return {}

    def save(self, path: str):
        """Save ensemble model to disk."""
        import pickle

        model_data = {
            'config': self.config,
            'base_models': self.base_models,
            'meta_model': self.meta_model,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'AdvancedEnsemble':
        """Load ensemble model from disk."""
        import pickle

        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        ensemble = cls(model_data['config'])
        ensemble.base_models = model_data['base_models']
        ensemble.meta_model = model_data['meta_model']
        ensemble.scaler = model_data['scaler']
        ensemble.is_trained = model_data['is_trained']
        ensemble.feature_names = model_data['feature_names']

        logger.info(f"Model loaded from {path}")
        return ensemble


# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("Advanced Ensemble Model Example")
    print("=" * 60)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20

    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + np.random.randn(n_samples) * 0.1

    # Split data
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]

    # Create and train ensemble
    config = EnsembleConfig(
        ensemble_method=EnsembleMethod.STACKING,
        prediction_task=PredictionTask.REGRESSION
    )

    ensemble = AdvancedEnsemble(config)

    print("\nTraining ensemble...")
    metrics = ensemble.train(X_train, y_train, X_test, y_test)

    print(f"\nTraining metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # Make predictions
    print(f"\nMaking predictions...")
    for i in range(5):
        result = ensemble.predict(X_test[i])
        print(f"\nSample {i+1}:")
        print(f"  Predicted: {result.predicted_value:.4f}")
        print(f"  Actual: {y_test[i]:.4f}")
        print(f"  Confidence: {result.confidence:.4f}")
        print(f"  Individual predictions: {result.individual_predictions}")

    # Feature importance
    print(f"\nFeature Importance:")
    importance = ensemble.get_feature_importance()
    for feat, imp in list(importance.items())[:5]:
        print(f"  {feat}: {imp:.4f}")

    print("\n✅ Advanced Ensemble Model Example Complete!")
