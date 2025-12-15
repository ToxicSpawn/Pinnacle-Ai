"""
Market Regime Detection
Uses LSTM to classify market regimes (bull, bear, sideways)
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. Regime detector will be disabled.")


class MarketRegimeDetector:
    """
    Market regime detector using LSTM classification.
    
    Detects:
    - Bull market (uptrend)
    - Bear market (downtrend)
    - Sideways market (range-bound)
    """
    
    def __init__(self, lookback: int = 30, model_dir: Optional[str] = None):
        """
        Initialize regime detector.
        
        Args:
            lookback: Number of periods to look back
            model_dir: Directory to save/load models
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for regime detection")
        
        self.lookback = lookback
        self.model_dir = model_dir or "models/regime"
        
        self.models: Dict[str, Sequential] = {
            'bull': self._build_model(),
            'bear': self._build_model(),
            'sideways': self._build_model()
        }
        
        self.trained = False
    
    def _build_model(self) -> Sequential:
        """Build regime classification model."""
        model = Sequential([
            LSTM(64, input_shape=(self.lookback, 5), return_sequences=True),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(3, activation='softmax')  # 3 regimes
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _preprocess(self, data: pd.DataFrame) -> np.ndarray:
        """
        Convert market data to model input format.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Preprocessed array
        """
        if len(data) < self.lookback:
            raise ValueError(f"Need at least {self.lookback} periods of data")
        
        # Calculate features
        returns = data['close'].pct_change()
        
        features = pd.DataFrame({
            'volatility': returns.rolling(30).std(),
            'momentum': data['close'].pct_change(30),
            'volume_trend': data['volume'].pct_change(30),
            'price_trend': (data['close'] - data['close'].rolling(30).mean()) / data['close'].rolling(30).mean(),
            'volatility_ratio': data['high'].rolling(30).max() / data['low'].rolling(30).min()
        })
        
        # Fill NaN values
        features = features.fillna(method='bfill').fillna(0)
        
        # Normalize
        features = (features - features.mean()) / (features.std() + 1e-8)
        
        # Get last lookback periods
        X = features.values[-self.lookback:]
        
        return X.reshape(1, self.lookback, 5)
    
    def _label_regime(self, data: pd.DataFrame) -> str:
        """
        Label regime based on price action (for training).
        
        Args:
            data: Historical data
            
        Returns:
            Regime label
        """
        # Simple heuristic: compare current price to moving averages
        sma_short = data['close'].rolling(10).mean()
        sma_long = data['close'].rolling(30).mean()
        
        current_price = data['close'].iloc[-1]
        short_ma = sma_short.iloc[-1]
        long_ma = sma_long.iloc[-1]
        
        # Calculate trend strength
        trend = (current_price - long_ma) / long_ma
        
        if trend > 0.05:  # 5% above long MA
            return 'bull'
        elif trend < -0.05:  # 5% below long MA
            return 'bear'
        else:
            return 'sideways'
    
    def train(
        self,
        historical_data: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 32
    ) -> Dict[str, Dict]:
        """
        Train regime detection models.
        
        Args:
            historical_data: Historical market data
            epochs: Training epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        logger.info("Training regime detection models...")
        
        # Prepare training data
        X_list = []
        y_list = []
        
        for i in range(self.lookback, len(historical_data)):
            window = historical_data.iloc[i - self.lookback:i]
            X = self._preprocess(window)
            regime = self._label_regime(window)
            
            X_list.append(X[0])  # Remove batch dimension
            y_list.append(regime)
        
        X = np.array(X_list)
        
        # One-hot encode labels
        y_onehot = np.zeros((len(y_list), 3))
        regime_map = {'bull': 0, 'bear': 1, 'sideways': 2}
        for i, regime in enumerate(y_list):
            y_onehot[i, regime_map[regime]] = 1
        
        # Train each model (specialized for each regime)
        histories = {}
        for regime_name, model in self.models.items():
            # Create regime-specific labels (1 if matches, 0 otherwise)
            y_regime = (y_onehot[:, regime_map[regime_name]]).reshape(-1, 1)
            
            # Train
            history = model.fit(
                X, y_regime,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=0
            )
            
            histories[regime_name] = history.history
        
        self.trained = True
        logger.info("✅ Regime detection models trained")
        
        return histories
    
    def detect_regime(self, market_data: pd.DataFrame) -> Tuple[str, float]:
        """
        Detect current market regime.
        
        Args:
            market_data: Recent market data (at least lookback periods)
            
        Returns:
            Tuple of (regime, confidence)
        """
        if not self.trained:
            logger.warning("Models not trained. Using heuristic detection.")
            return self._label_regime(market_data), 0.5
        
        try:
            X = self._preprocess(market_data)
            
            # Get predictions from all models
            predictions = {}
            for regime, model in self.models.items():
                pred = model.predict(X, verbose=0)[0]
                # Use the probability for the regime class
                regime_idx = {'bull': 0, 'bear': 1, 'sideways': 2}[regime]
                predictions[regime] = pred[regime_idx]
            
            # Return regime with highest probability
            best_regime = max(predictions.items(), key=lambda x: x[1])
            
            return best_regime[0], best_regime[1]
        
        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            return self._label_regime(market_data), 0.5
    
    def save_models(self, directory: Optional[str] = None) -> None:
        """Save models to directory."""
        import os
        from pathlib import Path
        
        dir_path = Path(directory or self.model_dir)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        for regime, model in self.models.items():
            model.save(str(dir_path / f"{regime}_regime_model.h5"))
        
        logger.info(f"✅ Saved regime models to {dir_path}")
    
    def load_models(self, directory: Optional[str] = None) -> None:
        """Load models from directory."""
        from pathlib import Path
        
        dir_path = Path(directory or self.model_dir)
        
        for regime in self.models.keys():
            model_path = dir_path / f"{regime}_regime_model.h5"
            if model_path.exists():
                self.models[regime] = load_model(str(model_path))
                self.trained = True
        
        if self.trained:
            logger.info(f"✅ Loaded regime models from {dir_path}")

