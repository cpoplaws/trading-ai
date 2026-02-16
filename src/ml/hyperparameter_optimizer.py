"""
Hyperparameter Optimization for Trading ML Models
==================================================

Automated hyperparameter tuning using multiple optimization strategies.

Methods:
- Grid Search (exhaustive search)
- Random Search (efficient sampling)
- Bayesian Optimization (smart search)
- Optuna (state-of-the-art optimization)

Supports all trading ML models:
- Ensemble models (RandomForest, GradientBoosting, XGBoost, LightGBM)
- Neural networks (LSTM, GRU, CNN-LSTM, VAE)
- Traditional models (Linear, SVM, etc.)
"""

import logging
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

# Try importing optimization libraries
try:
    from sklearn.model_selection import (
        GridSearchCV, RandomizedSearchCV,
        TimeSeriesSplit, cross_val_score
    )
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available")

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available. Install with: pip install optuna")


class OptimizationMethod(str, Enum):
    """Hyperparameter optimization methods."""
    GRID_SEARCH = "grid_search"           # Exhaustive search
    RANDOM_SEARCH = "random_search"       # Random sampling
    BAYESIAN = "bayesian"                 # Bayesian optimization (Optuna)
    HYPERBAND = "hyperband"               # Successive halving (Optuna)


class Metric(str, Enum):
    """Evaluation metrics."""
    MSE = "mse"                          # Mean Squared Error
    RMSE = "rmse"                        # Root MSE
    MAE = "mae"                          # Mean Absolute Error
    R2 = "r2"                            # R-squared
    SHARPE = "sharpe"                    # Sharpe ratio (for trading)
    SORTINO = "sortino"                  # Sortino ratio
    PROFIT = "profit"                    # Total profit


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""
    # Method
    optimization_method: OptimizationMethod = OptimizationMethod.BAYESIAN

    # Search space (depends on model)
    param_space: Dict[str, Any] = field(default_factory=dict)

    # Evaluation
    metric: Metric = Metric.RMSE
    n_splits: int = 5                    # Time series cross-validation splits
    test_size_pct: float = 0.2           # Test set percentage per split

    # Search parameters
    n_trials: int = 50                   # Number of trials (Bayesian/Random)
    n_jobs: int = -1                     # Parallel jobs (-1 = all cores)
    random_state: int = 42

    # Early stopping
    early_stopping_rounds: int = 10      # Stop if no improvement
    min_improvement: float = 0.001       # Minimum improvement threshold

    # Time limits
    timeout_seconds: Optional[int] = None  # Max optimization time


@dataclass
class OptimizationResult:
    """Results from hyperparameter optimization."""
    best_params: Dict[str, Any]
    best_score: float
    optimization_method: OptimizationMethod
    metric: Metric
    n_trials: int
    best_trial_number: int
    optimization_time: float
    all_trials: List[Dict[str, Any]]
    cv_scores: List[float]
    timestamp: datetime = field(default_factory=datetime.now)


class HyperparameterOptimizer:
    """
    Automated hyperparameter optimization for trading ML models.

    Supports multiple optimization strategies:
    - Grid Search: Exhaustive but slow
    - Random Search: Fast and often effective
    - Bayesian Optimization: Smart and efficient (recommended)
    - Hyperband: Adaptive resource allocation
    """

    def __init__(self, config: Optional[OptimizationConfig] = None):
        """Initialize hyperparameter optimizer."""
        self.config = config or OptimizationConfig()

        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for optimization")

        logger.info(f"Initialized HyperparameterOptimizer "
                   f"(method: {self.config.optimization_method.value})")

    def optimize(
        self,
        model_class: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_space: Optional[Dict[str, Any]] = None,
        custom_scorer: Optional[Callable] = None
    ) -> OptimizationResult:
        """
        Optimize hyperparameters for a model.

        Args:
            model_class: Model class (e.g., RandomForestRegressor)
            X_train: Training features
            y_train: Training targets
            param_space: Parameter search space (optional)
            custom_scorer: Custom scoring function (optional)

        Returns:
            OptimizationResult with best parameters and scores
        """
        param_space = param_space or self.config.param_space

        if not param_space:
            raise ValueError("param_space must be provided")

        start_time = datetime.now()

        # Select optimization method
        if self.config.optimization_method == OptimizationMethod.GRID_SEARCH:
            result = self._grid_search(
                model_class, X_train, y_train, param_space, custom_scorer
            )

        elif self.config.optimization_method == OptimizationMethod.RANDOM_SEARCH:
            result = self._random_search(
                model_class, X_train, y_train, param_space, custom_scorer
            )

        elif self.config.optimization_method in [
            OptimizationMethod.BAYESIAN,
            OptimizationMethod.HYPERBAND
        ]:
            if not OPTUNA_AVAILABLE:
                logger.warning("Optuna not available, falling back to Random Search")
                result = self._random_search(
                    model_class, X_train, y_train, param_space, custom_scorer
                )
            else:
                result = self._bayesian_optimization(
                    model_class, X_train, y_train, param_space, custom_scorer
                )

        else:
            raise ValueError(f"Unknown optimization method: {self.config.optimization_method}")

        end_time = datetime.now()
        result.optimization_time = (end_time - start_time).total_seconds()

        logger.info(
            f"Optimization complete: {result.optimization_method.value}, "
            f"best {self.config.metric.value}: {result.best_score:.4f}, "
            f"time: {result.optimization_time:.1f}s"
        )

        return result

    def _grid_search(
        self,
        model_class: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_space: Dict[str, Any],
        custom_scorer: Optional[Callable]
    ) -> OptimizationResult:
        """Grid search optimization."""
        logger.info(f"Starting Grid Search with {len(param_space)} parameters")

        # Create time series cross-validator
        cv = TimeSeriesSplit(n_splits=self.config.n_splits)

        # Scoring
        scoring = custom_scorer or self._get_scorer()

        # Grid search
        grid_search = GridSearchCV(
            estimator=model_class(),
            param_grid=param_space,
            scoring=scoring,
            cv=cv,
            n_jobs=self.config.n_jobs,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        # Extract results
        cv_results = pd.DataFrame(grid_search.cv_results_)
        all_trials = [
            {
                'params': cv_results.iloc[i]['params'],
                'score': cv_results.iloc[i]['mean_test_score'],
                'std': cv_results.iloc[i]['std_test_score']
            }
            for i in range(len(cv_results))
        ]

        return OptimizationResult(
            best_params=grid_search.best_params_,
            best_score=grid_search.best_score_,
            optimization_method=OptimizationMethod.GRID_SEARCH,
            metric=self.config.metric,
            n_trials=len(all_trials),
            best_trial_number=grid_search.best_index_,
            optimization_time=0.0,  # Will be set by caller
            all_trials=all_trials,
            cv_scores=grid_search.cv_results_['mean_test_score'].tolist()
        )

    def _random_search(
        self,
        model_class: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_space: Dict[str, Any],
        custom_scorer: Optional[Callable]
    ) -> OptimizationResult:
        """Random search optimization."""
        logger.info(f"Starting Random Search with {self.config.n_trials} trials")

        # Create time series cross-validator
        cv = TimeSeriesSplit(n_splits=self.config.n_splits)

        # Scoring
        scoring = custom_scorer or self._get_scorer()

        # Random search
        random_search = RandomizedSearchCV(
            estimator=model_class(),
            param_distributions=param_space,
            n_iter=self.config.n_trials,
            scoring=scoring,
            cv=cv,
            n_jobs=self.config.n_jobs,
            random_state=self.config.random_state,
            verbose=1
        )

        random_search.fit(X_train, y_train)

        # Extract results
        cv_results = pd.DataFrame(random_search.cv_results_)
        all_trials = [
            {
                'params': cv_results.iloc[i]['params'],
                'score': cv_results.iloc[i]['mean_test_score'],
                'std': cv_results.iloc[i]['std_test_score']
            }
            for i in range(len(cv_results))
        ]

        return OptimizationResult(
            best_params=random_search.best_params_,
            best_score=random_search.best_score_,
            optimization_method=OptimizationMethod.RANDOM_SEARCH,
            metric=self.config.metric,
            n_trials=len(all_trials),
            best_trial_number=random_search.best_index_,
            optimization_time=0.0,
            all_trials=all_trials,
            cv_scores=random_search.cv_results_['mean_test_score'].tolist()
        )

    def _bayesian_optimization(
        self,
        model_class: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_space: Dict[str, Any],
        custom_scorer: Optional[Callable]
    ) -> OptimizationResult:
        """Bayesian optimization using Optuna."""
        logger.info(f"Starting Bayesian Optimization with {self.config.n_trials} trials")

        # Create time series cross-validator
        cv = TimeSeriesSplit(n_splits=self.config.n_splits)

        # Track trials
        all_trials = []

        def objective(trial):
            """Optuna objective function."""
            # Sample hyperparameters
            params = {}
            for param_name, param_config in param_space.items():
                if isinstance(param_config, dict):
                    # Optuna-style config: {'type': 'int', 'low': 1, 'high': 100}
                    param_type = param_config.get('type', 'float')

                    if param_type == 'int':
                        params[param_name] = trial.suggest_int(
                            param_name,
                            param_config['low'],
                            param_config['high'],
                            log=param_config.get('log', False)
                        )
                    elif param_type == 'float':
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_config['low'],
                            param_config['high'],
                            log=param_config.get('log', False)
                        )
                    elif param_type == 'categorical':
                        params[param_name] = trial.suggest_categorical(
                            param_name,
                            param_config['choices']
                        )
                else:
                    # List of values: try categorical
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config
                    )

            # Create model with sampled parameters
            model = model_class(**params)

            # Cross-validation
            if custom_scorer:
                scores = cross_val_score(
                    model, X_train, y_train,
                    cv=cv,
                    scoring=custom_scorer,
                    n_jobs=1  # Parallel at trial level
                )
            else:
                scores = cross_val_score(
                    model, X_train, y_train,
                    cv=cv,
                    scoring=self._get_scorer(),
                    n_jobs=1
                )

            mean_score = scores.mean()
            std_score = scores.std()

            # Track trial
            all_trials.append({
                'params': params,
                'score': mean_score,
                'std': std_score
            })

            # Optuna maximizes, so negate if we're minimizing
            if self.config.metric in [Metric.MSE, Metric.RMSE, Metric.MAE]:
                return -mean_score  # Maximize negative error (minimize error)
            else:
                return mean_score

        # Create study
        sampler = TPESampler(seed=self.config.random_state)
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler
        )

        # Optimize
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout_seconds,
            n_jobs=self.config.n_jobs,
            show_progress_bar=True
        )

        # Get best trial
        best_trial = study.best_trial

        # Adjust score sign if needed
        best_score = best_trial.value
        if self.config.metric in [Metric.MSE, Metric.RMSE, Metric.MAE]:
            best_score = -best_score

        return OptimizationResult(
            best_params=best_trial.params,
            best_score=best_score,
            optimization_method=OptimizationMethod.BAYESIAN,
            metric=self.config.metric,
            n_trials=len(study.trials),
            best_trial_number=best_trial.number,
            optimization_time=0.0,
            all_trials=all_trials,
            cv_scores=[trial.value for trial in study.trials]
        )

    def _get_scorer(self) -> str:
        """Get sklearn scorer name from metric."""
        scorer_map = {
            Metric.MSE: 'neg_mean_squared_error',
            Metric.RMSE: 'neg_root_mean_squared_error',
            Metric.MAE: 'neg_mean_absolute_error',
            Metric.R2: 'r2'
        }

        return scorer_map.get(self.config.metric, 'neg_mean_squared_error')


# Predefined parameter spaces for common models

PARAM_SPACES = {
    'random_forest': {
        'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
        'max_depth': {'type': 'int', 'low': 3, 'high': 20},
        'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
        'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
        'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', None]},
    },

    'gradient_boosting': {
        'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
        'learning_rate': {'type': 'float', 'low': 0.001, 'high': 0.3, 'log': True},
        'max_depth': {'type': 'int', 'low': 3, 'high': 10},
        'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
        'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
        'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
    },

    'xgboost': {
        'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
        'learning_rate': {'type': 'float', 'low': 0.001, 'high': 0.3, 'log': True},
        'max_depth': {'type': 'int', 'low': 3, 'high': 10},
        'min_child_weight': {'type': 'int', 'low': 1, 'high': 10},
        'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
        'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
        'gamma': {'type': 'float', 'low': 0, 'high': 5},
    },

    'lightgbm': {
        'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
        'learning_rate': {'type': 'float', 'low': 0.001, 'high': 0.3, 'log': True},
        'max_depth': {'type': 'int', 'low': 3, 'high': 10},
        'num_leaves': {'type': 'int', 'low': 20, 'high': 200},
        'min_child_samples': {'type': 'int', 'low': 5, 'high': 100},
        'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
        'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
    },

    'svr': {
        'C': {'type': 'float', 'low': 0.1, 'high': 100, 'log': True},
        'epsilon': {'type': 'float', 'low': 0.001, 'high': 1.0, 'log': True},
        'kernel': {'type': 'categorical', 'choices': ['rbf', 'linear', 'poly']},
        'gamma': {'type': 'categorical', 'choices': ['scale', 'auto']},
    },

    'mlp': {
        'hidden_layer_sizes': {'type': 'categorical', 'choices': [
            (50,), (100,), (50, 50), (100, 50), (100, 100)
        ]},
        'alpha': {'type': 'float', 'low': 0.0001, 'high': 0.1, 'log': True},
        'learning_rate_init': {'type': 'float', 'low': 0.0001, 'high': 0.01, 'log': True},
        'activation': {'type': 'categorical', 'choices': ['relu', 'tanh']},
    }
}


def optimize_model(
    model_name: str,
    model_class: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str = 'bayesian',
    n_trials: int = 50,
    n_splits: int = 5
) -> OptimizationResult:
    """
    Quick optimization for common models.

    Args:
        model_name: Model name ('random_forest', 'xgboost', etc.)
        model_class: Model class
        X_train: Training features
        y_train: Training targets
        method: Optimization method ('grid', 'random', 'bayesian')
        n_trials: Number of trials
        n_splits: CV splits

    Returns:
        OptimizationResult
    """
    # Get param space
    param_space = PARAM_SPACES.get(model_name)
    if param_space is None:
        raise ValueError(f"No predefined param space for '{model_name}'")

    # Create config
    config = OptimizationConfig(
        optimization_method=OptimizationMethod(method),
        param_space=param_space,
        n_trials=n_trials,
        n_splits=n_splits
    )

    # Optimize
    optimizer = HyperparameterOptimizer(config)
    return optimizer.optimize(model_class, X_train, y_train)
