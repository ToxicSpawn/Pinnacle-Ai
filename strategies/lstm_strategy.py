"""
LSTM-based Price Prediction Strategy
Uses TensorFlow/Keras for deep learning price forecasting
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
from pathlib import Path
import os

logger = logging.getLogger(__name__)

try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. LSTM strategy will be disabled.")


class LSTMPredictor:
    """
    LSTM-based price prediction model.
    
    Features:
    - Multi-layer LSTM architecture
    - Automatic feature scaling
    - Model persistence
    - Early stopping and checkpointing
    """
    
    def __init__(
        self,
        symbol: str,
        lookback: int = 60,
        model_path: Optional[Path] = None,
        units: int = 50,
        dropout: float = 0.2
    ):
        """
        Initialize LSTM predictor.
        
        Args:
            symbol: Trading pair symbol
            lookback: Number of historical candles to use for prediction
            model_path: Path to save/load model
            units: Number of LSTM units per layer
            dropout: Dropout rate for regularization
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM strategy")
        
        self.symbol = symbol
        self.lookback = lookback
        self.units = units
        self.dropout = dropout
        self.scaler = MinMaxScaler()
        self.model: Optional[Sequential] = None
        
        if model_path:
            self.model_path = Path(model_path)
        else:
            self.model_path = Path("models") / f"lstm_{symbol.replace('/', '_')}.h5"
        
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Try to load existing model
        if self.model_path.exists():
            try:
                self.model = load_model(str(self.model_path))
                logger.info(f"✅ Loaded existing LSTM model for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}. Will create new model.")
    
    def _build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Build LSTM model architecture."""
        model = Sequential([
            LSTM(self.units, return_sequences=True, input_shape=input_shape),
            Dropout(self.dropout),
            LSTM(self.units, return_sequences=True),
            Dropout(self.dropout),
            LSTM(self.units),
            Dropout(self.dropout),
            Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from OHLCV DataFrame.
        
        Args:
            df: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
        Returns:
            X, y arrays for training
        """
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")
        
        # Use close price for prediction
        prices = df[['close']].values
        
        # Scale data
        scaled_prices = self.scaler.fit_transform(prices)
        
        # Create sequences
        X, y = [], []
        for i in range(self.lookback, len(scaled_prices)):
            X.append(scaled_prices[i - self.lookback:i, 0])
            y.append(scaled_prices[i, 0])
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape for LSTM (samples, timesteps, features)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y
    
    def train(
        self,
        df: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: int = 1
    ) -> dict:
        """
        Train the LSTM model.
        
        Args:
            df: Training data DataFrame
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction of data for validation
            verbose: Verbosity level
            
        Returns:
            Training history dictionary
        """
        X, y = self.prepare_data(df)
        
        if len(X) == 0:
            raise ValueError("Not enough data for training. Need at least lookback + 1 samples.")
        
        # Build model if not loaded
        if self.model is None:
            self.model = self._build_model((X.shape[1], 1))
        
        # Callbacks
        checkpoint = ModelCheckpoint(
            str(self.model_path),
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[checkpoint, early_stop],
            verbose=verbose
        )
        
        logger.info(f"✅ LSTM model trained for {self.symbol}")
        return history.history
    
    def predict(self, df: pd.DataFrame) -> float:
        """
        Predict next price.
        
        Args:
            df: DataFrame with at least 'lookback' rows of historical data
            
        Returns:
            Predicted price
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        if len(df) < self.lookback:
            raise ValueError(f"Need at least {self.lookback} rows of data")
        
        # Get last lookback candles
        recent_data = df[['close']].tail(self.lookback).values
        
        # Scale
        scaled_data = self.scaler.transform(recent_data)
        
        # Reshape for prediction
        X = np.reshape(scaled_data, (1, self.lookback, 1))
        
        # Predict
        prediction = self.model.predict(X, verbose=0)
        
        # Inverse transform
        predicted_price = self.scaler.inverse_transform(prediction)[0][0]
        
        return float(predicted_price)
    
    def predict_sequence(self, df: pd.DataFrame, steps: int = 5) -> List[float]:
        """
        Predict multiple steps ahead.
        
        Args:
            df: Historical data
            steps: Number of steps to predict ahead
            
        Returns:
            List of predicted prices
        """
        predictions = []
        current_data = df.copy()
        
        for _ in range(steps):
            pred = self.predict(current_data)
            predictions.append(pred)
            
            # Append prediction to data for next step
            new_row = current_data.iloc[-1].copy()
            new_row['close'] = pred
            current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
        
        return predictions
    
    def get_prediction_signal(
        self,
        current_price: float,
        predicted_price: float,
        threshold: float = 0.02
    ) -> Tuple[str, float]:
        """
        Generate trading signal from prediction.
        
        Args:
            current_price: Current market price
            predicted_price: Predicted future price
            threshold: Minimum price change to trigger signal (2% default)
            
        Returns:
            Tuple of (signal, confidence)
            signal: 'BUY', 'SELL', or 'HOLD'
            confidence: Confidence score (0-1)
        """
        price_change = (predicted_price - current_price) / current_price
        
        if price_change > threshold:
            confidence = min(abs(price_change) / (threshold * 2), 1.0)
            return 'BUY', confidence
        elif price_change < -threshold:
            confidence = min(abs(price_change) / (threshold * 2), 1.0)
            return 'SELL', confidence
        else:
            return 'HOLD', 0.0

