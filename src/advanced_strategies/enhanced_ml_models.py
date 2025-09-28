"""
Enhanced ML models including Prophet, ARIMA-GARCH, ensemble methods, and feature selection.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Time series libraries (with fallbacks)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logging.warning("Prophet not available. Install with: pip install prophet")

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    logging.warning("ARCH not available. Install with: pip install arch")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.warning("Statsmodels not available. Install with: pip install statsmodels")

logger = logging.getLogger(__name__)

class EnhancedMLModels:
    """
    Enhanced ML models with time series forecasting, ensemble methods, and feature selection.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize enhanced ML models.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        
    def prepare_features(self, data: pd.DataFrame, target_col: str = 'close',
                        lookback_window: int = 20) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for ML models.
        
        Args:
            data: Price data with OHLCV columns
            target_col: Target column name
            lookback_window: Lookback window for features
            
        Returns:
            Features DataFrame and target Series
        """
        try:
            df = data.copy()
            
            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing columns: {missing_cols}")
                return pd.DataFrame(), pd.Series()
            
            features = pd.DataFrame(index=df.index)
            
            # Price-based features
            features['returns'] = df[target_col].pct_change()
            features['log_returns'] = np.log(df[target_col] / df[target_col].shift(1))
            features['price_ratio'] = df[target_col] / df[target_col].rolling(20).mean()
            
            # Technical indicators
            features['sma_5'] = df[target_col].rolling(5).mean() / df[target_col]
            features['sma_20'] = df[target_col].rolling(20).mean() / df[target_col]
            features['ema_12'] = df[target_col].ewm(span=12).mean() / df[target_col]
            features['ema_26'] = df[target_col].ewm(span=26).mean() / df[target_col]
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            sma = df[target_col].rolling(bb_period).mean()
            std = df[target_col].rolling(bb_period).std()
            features['bb_upper'] = (sma + bb_std * std) / df[target_col]
            features['bb_lower'] = (sma - bb_std * std) / df[target_col]
            features['bb_position'] = (df[target_col] - sma) / (bb_std * std)
            
            # RSI
            delta = df[target_col].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = df[target_col].ewm(span=12).mean()
            ema26 = df[target_col].ewm(span=26).mean()
            features['macd'] = (ema12 - ema26) / df[target_col]
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            features['macd_histogram'] = features['macd'] - features['macd_signal']
            
            # Volume features
            features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            features['volume_price_trend'] = (df[target_col] * df['volume']).rolling(10).sum()
            
            # Volatility features
            features['volatility'] = df[target_col].rolling(20).std() / df[target_col].rolling(20).mean()
            features['price_range'] = (df['high'] - df['low']) / df[target_col]
            
            # Lag features
            for lag in [1, 2, 3, 5, 10]:
                features[f'return_lag_{lag}'] = features['returns'].shift(lag)
                features[f'price_lag_{lag}'] = df[target_col].shift(lag) / df[target_col]
            
            # Rolling statistics
            for window in [5, 10, 20]:
                features[f'return_mean_{window}'] = features['returns'].rolling(window).mean()
                features[f'return_std_{window}'] = features['returns'].rolling(window).std()
                features[f'return_skew_{window}'] = features['returns'].rolling(window).skew()
                features[f'return_kurt_{window}'] = features['returns'].rolling(window).kurt()
            
            # Calendar features
            if isinstance(df.index, pd.DatetimeIndex):
                features['day_of_week'] = df.index.dayofweek
                features['month'] = df.index.month
                features['quarter'] = df.index.quarter
                features['is_month_end'] = df.index.is_month_end.astype(int)
                features['is_quarter_end'] = df.index.is_quarter_end.astype(int)
            
            # Target variable (future returns)
            target = df[target_col].shift(-1) / df[target_col] - 1  # Next day return
            
            # Remove rows with NaN values
            combined = pd.concat([features, target.rename('target')], axis=1).dropna()
            
            if combined.empty:
                logger.warning("No valid data after feature engineering")
                return pd.DataFrame(), pd.Series()
            
            X = combined.drop('target', axis=1)
            y = combined['target']
            
            logger.info(f"Prepared {len(X.columns)} features for {len(X)} samples")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return pd.DataFrame(), pd.Series()
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       method: str = 'rfe', n_features: int = 20) -> pd.DataFrame:
        """
        Select most important features.
        
        Args:
            X: Feature matrix
            y: Target variable
            method: Feature selection method ('rfe', 'kbest', 'correlation')
            n_features: Number of features to select
            
        Returns:
            Selected features DataFrame
        """
        try:
            if X.empty or y.empty:
                return X
            
            # Handle infinite values
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            if method == 'rfe':
                # Recursive Feature Elimination
                estimator = RandomForestRegressor(n_estimators=50, random_state=self.random_state)
                selector = RFE(estimator, n_features_to_select=min(n_features, X.shape[1]))
                selector.fit(X, y)
                selected_features = X.columns[selector.support_]
                
            elif method == 'kbest':
                # Select K best features
                selector = SelectKBest(score_func=f_regression, k=min(n_features, X.shape[1]))
                selector.fit(X, y)
                selected_features = X.columns[selector.get_support()]
                
            elif method == 'correlation':
                # Select features with highest correlation to target
                correlations = X.corrwith(y).abs().sort_values(ascending=False)
                selected_features = correlations.head(min(n_features, len(correlations))).index
                
            else:
                selected_features = X.columns[:n_features]
            
            # Store selector for future use
            self.feature_selectors[method] = selector if method in ['rfe', 'kbest'] else None
            
            logger.info(f"Selected {len(selected_features)} features using {method}")
            return X[selected_features]
            
        except Exception as e:
            logger.error(f"Error selecting features: {e}")
            return X.iloc[:, :min(n_features, X.shape[1])]
    
    def fit_prophet_model(self, data: pd.DataFrame, target_col: str = 'close') -> Dict:
        """
        Fit Prophet time series model.
        
        Args:
            data: Time series data
            target_col: Target column name
            
        Returns:
            Model results and predictions
        """
        try:
            if not PROPHET_AVAILABLE:
                return {'error': 'Prophet not available'}
            
            # Prepare data for Prophet
            prophet_data = pd.DataFrame({
                'ds': data.index,
                'y': data[target_col]
            }).reset_index(drop=True)
            
            # Handle missing values
            prophet_data = prophet_data.dropna()
            
            if len(prophet_data) < 10:
                return {'error': 'Insufficient data for Prophet'}
            
            # Initialize Prophet model with financial market parameters
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,  # Financial data often doesn't have yearly patterns
                changepoint_prior_scale=0.05,  # More flexible trend changes
                seasonality_prior_scale=10,
                holidays_prior_scale=10,
                interval_width=0.95
            )
            
            # Fit model
            model.fit(prophet_data)
            
            # Make future predictions
            future_periods = 30  # 30 days ahead
            future = model.make_future_dataframe(periods=future_periods)
            forecast = model.predict(future)
            
            # Calculate metrics on historical data
            historical_forecast = forecast[:-future_periods]
            actual = prophet_data['y'].values
            predicted = historical_forecast['yhat'].values
            
            mse = mean_squared_error(actual, predicted)
            mae = mean_absolute_error(actual, predicted)
            r2 = r2_score(actual, predicted)
            
            # Get prediction intervals
            future_forecast = forecast[-future_periods:]
            
            return {
                'model_type': 'prophet',
                'model': model,
                'forecast': forecast,
                'future_predictions': future_forecast,
                'metrics': {
                    'mse': mse,
                    'mae': mae,
                    'r2_score': r2,
                    'rmse': np.sqrt(mse)
                },
                'components': model.predict(future)
            }
            
        except Exception as e:
            logger.error(f"Error fitting Prophet model: {e}")
            return {'error': str(e)}
    
    def fit_arima_garch_model(self, data: pd.Series, order: Tuple = None) -> Dict:
        """
        Fit ARIMA-GARCH model for volatility forecasting.
        
        Args:
            data: Return series
            order: ARIMA order (p, d, q)
            
        Returns:
            Model results and predictions
        """
        try:
            if not STATSMODELS_AVAILABLE or not ARCH_AVAILABLE:
                return {'error': 'Required libraries not available'}
            
            # Convert to returns if price data
            if data.std() > 1:  # Likely price data
                returns = data.pct_change().dropna() * 100  # Percentage returns
            else:
                returns = data.dropna() * 100
            
            if len(returns) < 50:
                return {'error': 'Insufficient data for ARIMA-GARCH'}
            
            # Auto-determine ARIMA order if not provided
            if order is None:
                # Simple order selection based on data characteristics
                adf_result = adfuller(returns)
                is_stationary = adf_result[1] < 0.05
                
                if is_stationary:
                    order = (1, 0, 1)  # AR(1)-MA(1) for stationary data
                else:
                    order = (1, 1, 1)  # ARIMA(1,1,1) for non-stationary data
            
            # Fit ARIMA model first
            arima_model = ARIMA(returns, order=order)
            arima_fitted = arima_model.fit()
            
            # Get ARIMA residuals
            residuals = arima_fitted.resid
            
            # Fit GARCH model to residuals
            garch_model = arch_model(residuals, vol='GARCH', p=1, q=1)
            garch_fitted = garch_model.fit(disp='off')
            
            # Generate forecasts
            arima_forecast = arima_fitted.forecast(steps=30)
            garch_forecast = garch_fitted.forecast(horizon=30)
            
            # Calculate model metrics
            fitted_values = arima_fitted.fittedvalues
            actual_values = returns[1:]  # Skip first observation due to differencing
            
            mse = mean_squared_error(actual_values, fitted_values)
            mae = mean_absolute_error(actual_values, fitted_values)
            
            return {
                'model_type': 'arima_garch',
                'arima_model': arima_fitted,
                'garch_model': garch_fitted,
                'arima_order': order,
                'return_forecast': arima_forecast,
                'volatility_forecast': garch_forecast,
                'metrics': {
                    'arima_mse': mse,
                    'arima_mae': mae,
                    'arima_aic': arima_fitted.aic,
                    'arima_bic': arima_fitted.bic,
                    'garch_aic': garch_fitted.aic,
                    'garch_bic': garch_fitted.bic
                }
            }
            
        except Exception as e:
            logger.error(f"Error fitting ARIMA-GARCH model: {e}")
            return {'error': str(e)}
    
    def create_ensemble_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Create ensemble model with multiple base estimators.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Ensemble model results
        """
        try:
            if X.empty or y.empty:
                return {'error': 'No data provided'}
            
            # Handle infinite values
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            
            # Define base models
            base_models = [
                ('rf', RandomForestRegressor(n_estimators=100, random_state=self.random_state)),
                ('gb', GradientBoostingRegressor(n_estimators=100, random_state=self.random_state)),
                ('svr', SVR(kernel='rbf', C=1.0)),
                ('ridge', Ridge(alpha=1.0)),
                ('lasso', Lasso(alpha=0.1))
            ]
            
            # Create voting regressor
            ensemble = VotingRegressor(estimators=base_models)
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = cross_val_score(ensemble, X_scaled, y, cv=tscv, scoring='r2')
            
            # Fit the ensemble model
            ensemble.fit(X_scaled, y)
            
            # Individual model performance
            individual_scores = {}
            for name, model in base_models:
                model.fit(X_scaled, y)
                y_pred = model.predict(X_scaled)
                individual_scores[name] = {
                    'r2': r2_score(y, y_pred),
                    'mse': mean_squared_error(y, y_pred),
                    'mae': mean_absolute_error(y, y_pred)
                }
            
            # Ensemble predictions
            y_pred_ensemble = ensemble.predict(X_scaled)
            ensemble_metrics = {
                'r2': r2_score(y, y_pred_ensemble),
                'mse': mean_squared_error(y, y_pred_ensemble),
                'mae': mean_absolute_error(y, y_pred_ensemble),
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std()
            }
            
            # Store models and scaler
            self.models['ensemble'] = ensemble
            self.scalers['ensemble'] = scaler
            
            return {
                'model_type': 'ensemble',
                'model': ensemble,
                'scaler': scaler,
                'metrics': ensemble_metrics,
                'individual_metrics': individual_scores,
                'cv_scores': cv_scores,
                'feature_importance': self._get_ensemble_feature_importance(ensemble, X.columns)
            }
            
        except Exception as e:
            logger.error(f"Error creating ensemble model: {e}")
            return {'error': str(e)}
    
    def _get_ensemble_feature_importance(self, ensemble, feature_names) -> Dict:
        """Get feature importance from ensemble model."""
        try:
            importance_dict = {}
            
            for name, model in ensemble.named_estimators_.items():
                if hasattr(model, 'feature_importances_'):
                    importance_dict[name] = dict(zip(feature_names, model.feature_importances_))
                elif hasattr(model, 'coef_'):
                    # For linear models, use absolute coefficients
                    importance_dict[name] = dict(zip(feature_names, np.abs(model.coef_)))
            
            # Average importance across models
            if importance_dict:
                avg_importance = {}
                for feature in feature_names:
                    importances = [imp.get(feature, 0) for imp in importance_dict.values()]
                    avg_importance[feature] = np.mean(importances)
                
                # Sort by importance
                sorted_importance = sorted(avg_importance.items(), 
                                         key=lambda x: x[1], reverse=True)
                return dict(sorted_importance)
            
            return {}
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return {}
    
    def generate_ml_signals(self, data: pd.DataFrame, target_col: str = 'close',
                           models_to_use: List[str] = None) -> Dict:
        """
        Generate trading signals using multiple ML models.
        
        Args:
            data: Price data
            target_col: Target column
            models_to_use: List of models to use
            
        Returns:
            ML trading signals
        """
        try:
            if models_to_use is None:
                models_to_use = ['ensemble', 'prophet', 'arima_garch']
            
            logger.info("Generating ML trading signals")
            
            # Prepare features
            X, y = self.prepare_features(data, target_col)
            
            if X.empty or y.empty:
                return {'error': 'No features available'}
            
            # Select features
            X_selected = self.select_features(X, y, method='rfe', n_features=20)
            
            signals = {}
            predictions = {}
            
            # Ensemble model
            if 'ensemble' in models_to_use:
                ensemble_result = self.create_ensemble_model(X_selected, y)
                if 'error' not in ensemble_result:
                    # Get latest prediction
                    latest_features = X_selected.iloc[-1:].replace([np.inf, -np.inf], np.nan).fillna(0)
                    if 'scaler' in ensemble_result:
                        latest_scaled = ensemble_result['scaler'].transform(latest_features)
                        prediction = ensemble_result['model'].predict(latest_scaled)[0]
                    else:
                        prediction = 0
                    
                    predictions['ensemble'] = prediction
                    signals['ensemble'] = {
                        'signal': 'BUY' if prediction > 0.01 else 'SELL' if prediction < -0.01 else 'HOLD',
                        'confidence': min(abs(prediction) * 10, 1.0),
                        'expected_return': prediction,
                        'model_r2': ensemble_result['metrics']['r2']
                    }
            
            # Prophet model
            if 'prophet' in models_to_use and PROPHET_AVAILABLE:
                prophet_result = self.fit_prophet_model(data, target_col)
                if 'error' not in prophet_result:
                    # Get next day prediction
                    future_pred = prophet_result['future_predictions'].iloc[0]
                    current_price = data[target_col].iloc[-1]
                    predicted_return = (future_pred['yhat'] - current_price) / current_price
                    
                    predictions['prophet'] = predicted_return
                    signals['prophet'] = {
                        'signal': 'BUY' if predicted_return > 0.01 else 'SELL' if predicted_return < -0.01 else 'HOLD',
                        'confidence': 0.7,  # Prophet generally reliable
                        'expected_return': predicted_return,
                        'trend': future_pred['trend'] if 'trend' in future_pred else 0
                    }
            
            # ARIMA-GARCH model
            if 'arima_garch' in models_to_use and STATSMODELS_AVAILABLE and ARCH_AVAILABLE:
                returns = data[target_col].pct_change().dropna()
                arima_garch_result = self.fit_arima_garch_model(returns)
                if 'error' not in arima_garch_result:
                    return_forecast = arima_garch_result['return_forecast'].iloc[0] / 100
                    volatility_forecast = arima_garch_result['volatility_forecast']['h.1'].iloc[0]
                    
                    predictions['arima_garch'] = return_forecast
                    signals['arima_garch'] = {
                        'signal': 'BUY' if return_forecast > 0.01 else 'SELL' if return_forecast < -0.01 else 'HOLD',
                        'confidence': max(0.3, 1.0 - volatility_forecast / 100),  # Lower confidence with high volatility
                        'expected_return': return_forecast,
                        'expected_volatility': volatility_forecast
                    }
            
            # Aggregate signals
            if signals:
                # Weighted average of predictions
                weights = {
                    'ensemble': 0.5,
                    'prophet': 0.3,
                    'arima_garch': 0.2
                }
                
                weighted_prediction = sum(
                    predictions.get(model, 0) * weights.get(model, 0)
                    for model in predictions.keys()
                )
                
                avg_confidence = np.mean([s['confidence'] for s in signals.values()])
                
                # Final signal
                final_signal = 'BUY' if weighted_prediction > 0.01 else 'SELL' if weighted_prediction < -0.01 else 'HOLD'
                
                return {
                    'timestamp': datetime.now(),
                    'symbol': data.index.name if hasattr(data.index, 'name') else 'UNKNOWN',
                    'final_signal': final_signal,
                    'confidence': avg_confidence,
                    'expected_return': weighted_prediction,
                    'individual_signals': signals,
                    'model_predictions': predictions,
                    'models_used': list(signals.keys())
                }
            
            return {'error': 'No valid models produced signals'}
            
        except Exception as e:
            logger.error(f"Error generating ML signals: {e}")
            return {'error': str(e)}

# Example usage
if __name__ == "__main__":
    print("🤖 Enhanced ML Models Demo")
    print("=" * 40)
    
    # Generate sample data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    
    # Simulate stock price data
    returns = np.random.normal(0.001, 0.02, len(dates))
    price = 100
    prices = [price]
    
    for ret in returns[1:]:
        price *= (1 + ret)
        prices.append(price)
    
    # Create sample DataFrame
    sample_data = pd.DataFrame({
        'open': np.array(prices) * (1 + np.random.normal(0, 0.005, len(prices))),
        'high': np.array(prices) * (1 + np.abs(np.random.normal(0, 0.01, len(prices)))),
        'low': np.array(prices) * (1 - np.abs(np.random.normal(0, 0.01, len(prices)))),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, len(prices))
    }, index=dates)
    
    # Initialize enhanced ML models
    ml_models = EnhancedMLModels()
    
    print("📊 Sample Data Shape:", sample_data.shape)
    print("Date Range:", sample_data.index[0].date(), "to", sample_data.index[-1].date())
    
    # Generate ML signals
    signals = ml_models.generate_ml_signals(sample_data)
    
    if 'error' not in signals:
        print(f"\n🎯 ML Trading Signals:")
        print(f"Final Signal: {signals['final_signal']}")
        print(f"Confidence: {signals['confidence']:.3f}")
        print(f"Expected Return: {signals['expected_return']:.3%}")
        print(f"Models Used: {', '.join(signals['models_used'])}")
        
        print(f"\n📈 Individual Model Signals:")
        for model, signal_data in signals['individual_signals'].items():
            print(f"  {model.upper()}:")
            print(f"    Signal: {signal_data['signal']}")
            print(f"    Confidence: {signal_data['confidence']:.3f}")
            print(f"    Expected Return: {signal_data['expected_return']:.3%}")
    else:
        print(f"❌ Error: {signals['error']}")
    
    # Test individual components
    print(f"\n🔧 Testing Individual Components:")
    
    # Feature preparation
    X, y = ml_models.prepare_features(sample_data)
    print(f"Features prepared: {X.shape[1]} features for {X.shape[0]} samples")
    
    # Feature selection
    if not X.empty and not y.empty:
        X_selected = ml_models.select_features(X, y, method='rfe', n_features=10)
        print(f"Selected features: {list(X_selected.columns[:5])}...")
    
    # Ensemble model
    if not X.empty and not y.empty:
        ensemble_result = ml_models.create_ensemble_model(X_selected, y)
        if 'error' not in ensemble_result:
            print(f"Ensemble R²: {ensemble_result['metrics']['r2']:.3f}")
            print(f"CV R² (mean ± std): {ensemble_result['metrics']['cv_r2_mean']:.3f} ± {ensemble_result['metrics']['cv_r2_std']:.3f}")
    
    print("\n✅ Enhanced ML models demo completed!")