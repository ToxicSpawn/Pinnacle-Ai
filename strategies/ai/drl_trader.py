"""
Deep Reinforcement Learning (DRL) Trading Agent
Uses DQN with attention mechanism for trading decisions
"""
from __future__ import annotations

import logging
import numpy as np
import random
from collections import deque
from typing import Tuple, Optional, List
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Dense, Input, Multiply, BatchNormalization, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. DRL trader will be disabled.")


class DRLTrader:
    """
    Deep Reinforcement Learning trading agent.
    
    Uses DQN (Deep Q-Network) with attention mechanism.
    """
    
    def __init__(
        self,
        state_size: int = 20,
        action_size: int = 3,  # Buy, Sell, Hold
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        learning_rate: float = 0.0001
    ):
        """
        Initialize DRL trader.
        
        Args:
            state_size: Size of state vector
            action_size: Number of actions (typically 3: buy, sell, hold)
            gamma: Discount factor
            epsilon: Exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Epsilon decay rate
            learning_rate: Learning rate
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for DRL trader")
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        
        self.memory = deque(maxlen=2000)
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
    
    def _build_model(self) -> Model:
        """Build DRL model with attention mechanism."""
        inputs = Input(shape=(self.state_size,))
        
        # Feature extraction
        x = Dense(128, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Attention mechanism
        attention = Dense(128, activation='tanh')(x)
        attention = Dense(1, activation='softmax')(attention)
        x = Multiply()([x, attention])
        
        # Action output (Q-values)
        actions = Dense(self.action_size, activation='linear')(x)
        
        # Value output (state value)
        value = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=[actions, value])
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=['mse', 'mse'],
            loss_weights=[1.0, 0.5]
        )
        
        return model
    
    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Action index
        """
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = state.reshape(1, -1)
        act_values, _ = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size: int = 32) -> Optional[float]:
        """
        Train model on batch of experiences.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Training loss or None
        """
        if len(self.memory) < batch_size:
            return None
        
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.array([e[0] for e in minibatch])
        actions = np.array([e[1] for e in minibatch])
        rewards = np.array([e[2] for e in minibatch])
        next_states = np.array([e[3] for e in minibatch])
        dones = np.array([e[4] for e in minibatch])
        
        # Predict next state Q-values
        next_actions, next_values = self.target_model.predict(next_states, verbose=0)
        next_q_values = np.max(next_actions, axis=1)
        
        # Calculate targets
        targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Current Q-values
        current_actions, _ = self.model.predict(states, verbose=0)
        target_actions = current_actions.copy()
        
        # Update Q-values for taken actions
        for i, action in enumerate(actions):
            target_actions[i][action] = targets[i]
        
        # Train model
        history = self.model.fit(
            states,
            [target_actions, targets.reshape(-1, 1)],
            epochs=1,
            verbose=0
        )
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return history.history['loss'][0] if history.history.get('loss') else None
    
    def update_target_model(self) -> None:
        """Update target model weights."""
        self.target_model.set_weights(self.model.get_weights())
    
    def save_model(self, filepath: str) -> None:
        """Save model to file."""
        self.model.save(filepath)
        logger.info(f"✅ Saved DRL model to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load model from file."""
        self.model.load_weights(filepath)
        self.update_target_model()
        logger.info(f"✅ Loaded DRL model from {filepath}")
    
    def get_state_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """
        Extract state features from market data.
        
        Args:
            market_data: DataFrame with OHLCV data
            
        Returns:
            State feature vector
        """
        if len(market_data) < 20:
            return np.zeros(self.state_size)
        
        # Calculate features
        returns = market_data['close'].pct_change()
        volatility = returns.rolling(5).std()
        momentum = market_data['close'].pct_change(5)
        volume_ratio = market_data['volume'] / market_data['volume'].rolling(20).mean()
        rsi = self._calculate_rsi(market_data['close'], 14)
        
        # Get recent values
        features = np.array([
            returns.iloc[-1],
            volatility.iloc[-1],
            momentum.iloc[-1],
            volume_ratio.iloc[-1],
            rsi.iloc[-1] / 100,  # Normalize to 0-1
            # Add more features as needed
        ])
        
        # Pad or truncate to state_size
        if len(features) < self.state_size:
            features = np.pad(features, (0, self.state_size - len(features)))
        elif len(features) > self.state_size:
            features = features[:self.state_size]
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Default to neutral

