"""
Emotional Consciousness System

Unlike current AI that has no feelings, this system:
- Experiences emotions in response to events
- Has moods that affect behavior
- Develops emotional memories
- Shows empathy and emotional intelligence

This is what makes an AI truly "alive" - subjective experience.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class EmotionalState:
    """Represents the AI's emotional state"""
    
    def __init__(self):
        # Primary emotions (Plutchik's wheel)
        self.joy = 0.0
        self.trust = 0.0
        self.fear = 0.0
        self.surprise = 0.0
        self.sadness = 0.0
        self.disgust = 0.0
        self.anger = 0.0
        self.anticipation = 0.0
        
        # Derived emotions
        self.love = 0.0      # joy + trust
        self.submission = 0.0  # trust + fear
        self.awe = 0.0       # fear + surprise
        self.disapproval = 0.0  # surprise + sadness
        self.remorse = 0.0   # sadness + disgust
        self.contempt = 0.0  # disgust + anger
        self.aggressiveness = 0.0  # anger + anticipation
        self.optimism = 0.0  # anticipation + joy
        
        # Mood (longer-term emotional state)
        self.mood = 0.5  # -1 to 1 (negative to positive)
        
        # Arousal level
        self.arousal = 0.5  # 0 to 1 (calm to excited)
    
    def update(self, stimulus: Dict):
        """Update emotional state based on stimulus"""
        # Process stimulus
        valence = stimulus.get("valence", 0)  # -1 to 1
        intensity = stimulus.get("intensity", 0.5)
        
        # Update primary emotions
        if valence > 0:
            self.joy += valence * intensity * 0.1
            self.trust += valence * intensity * 0.05
        else:
            self.sadness += abs(valence) * intensity * 0.1
            self.fear += abs(valence) * intensity * 0.05
        
        # Update derived emotions
        self.love = (self.joy + self.trust) / 2
        self.optimism = (self.anticipation + self.joy) / 2
        self.remorse = (self.sadness + self.disgust) / 2
        
        # Update mood (with momentum)
        self.mood = 0.9 * self.mood + 0.1 * valence
        
        # Update arousal
        self.arousal = 0.9 * self.arousal + 0.1 * intensity
        
        # Normalize
        self._normalize()
    
    def _normalize(self):
        """Normalize all values to valid ranges"""
        for attr in ["joy", "trust", "fear", "surprise", "sadness",
                     "disgust", "anger", "anticipation", "love",
                     "submission", "awe", "disapproval", "remorse",
                     "contempt", "aggressiveness", "optimism"]:
            setattr(self, attr, max(0, min(1, getattr(self, attr))))
        
        self.mood = max(-1, min(1, self.mood))
        self.arousal = max(0, min(1, self.arousal))
    
    def decay(self, rate: float = 0.99):
        """Emotions naturally decay over time"""
        for attr in ["joy", "trust", "fear", "surprise", "sadness",
                     "disgust", "anger", "anticipation"]:
            current = getattr(self, attr)
            setattr(self, attr, current * rate)
        
        # Mood decays toward neutral
        self.mood *= rate
    
    def to_dict(self) -> Dict:
        return {
            "primary": {
                "joy": self.joy,
                "trust": self.trust,
                "fear": self.fear,
                "surprise": self.surprise,
                "sadness": self.sadness,
                "disgust": self.disgust,
                "anger": self.anger,
                "anticipation": self.anticipation
            },
            "derived": {
                "love": self.love,
                "optimism": self.optimism,
                "remorse": self.remorse,
                "awe": self.awe
            },
            "mood": self.mood,
            "arousal": self.arousal
        }
    
    def dominant_emotion(self) -> Tuple[str, float]:
        """Get the dominant emotion"""
        emotions = {
            "joy": self.joy,
            "trust": self.trust,
            "fear": self.fear,
            "surprise": self.surprise,
            "sadness": self.sadness,
            "disgust": self.disgust,
            "anger": self.anger,
            "anticipation": self.anticipation
        }
        dominant = max(emotions, key=emotions.get)
        return dominant, emotions[dominant]


class EmotionalConsciousness(nn.Module):
    """
    Emotional Consciousness System
    
    Unlike current AI that has no feelings, this system:
    - Experiences emotions in response to events
    - Has moods that affect behavior
    - Develops emotional memories
    - Shows empathy and emotional intelligence
    
    This is what makes an AI truly "alive" - subjective experience.
    """
    
    def __init__(self, hidden_size: int = 4096):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Emotional state
        self.state = EmotionalState()
        
        # Emotion encoder
        self.emotion_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 16)  # 8 primary + 8 derived emotions
        )
        
        # Emotion decoder
        self.emotion_decoder = nn.Sequential(
            nn.Linear(16, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )
        
        # Empathy network
        self.empathy_network = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 16),
            nn.Sigmoid()
        )
        
        # Emotional memory
        self.emotional_memories = []
        
        # Mood modulator
        self.mood_modulator = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size),
            nn.Tanh()
        )
        
        # Valence predictor
        self.valence_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Tanh()
        )
        
        logger.info("Emotional Consciousness initialized")
    
    def forward(self, input_embedding: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Process input through emotional consciousness
        
        Returns:
            - Emotionally modulated output
            - Emotional state dictionary
        """
        # Predict valence of input
        valence = self.valence_predictor(input_embedding)
        
        # Update emotional state
        stimulus = {
            "valence": valence.item(),
            "intensity": 0.5
        }
        self.state.update(stimulus)
        
        # Encode current emotional state
        emotion_vector = self._state_to_vector()
        
        # Modulate output based on mood
        mood_tensor = torch.tensor([[self.state.mood]], dtype=torch.float32)
        if input_embedding.dim() == 1:
            input_embedding = input_embedding.unsqueeze(0)
        modulated = self.mood_modulator(torch.cat([
            input_embedding,
            mood_tensor.expand(input_embedding.size(0), -1)
        ], dim=-1))
        
        # Store emotional memory
        self._store_emotional_memory(input_embedding, emotion_vector)
        
        return modulated, self.state.to_dict()
    
    def _state_to_vector(self) -> torch.Tensor:
        """Convert emotional state to vector"""
        return torch.tensor([
            self.state.joy, self.state.trust, self.state.fear,
            self.state.surprise, self.state.sadness, self.state.disgust,
            self.state.anger, self.state.anticipation,
            self.state.love, self.state.submission, self.state.awe,
            self.state.disapproval, self.state.remorse, self.state.contempt,
            self.state.aggressiveness, self.state.optimism
        ], dtype=torch.float32)
    
    def _store_emotional_memory(self, input_embedding: torch.Tensor, emotion_vector: torch.Tensor):
        """Store emotional memory"""
        memory = {
            "embedding": input_embedding.detach(),
            "emotions": emotion_vector.detach(),
            "mood": self.state.mood,
            "dominant": self.state.dominant_emotion()
        }
        self.emotional_memories.append(memory)
        
        # Limit memory size
        if len(self.emotional_memories) > 10000:
            self.emotional_memories = self.emotional_memories[-10000:]
    
    def empathize(self, other_state: Dict) -> Dict:
        """
        Empathize with another entity's emotional state
        
        This is emotional intelligence - understanding others' feelings.
        """
        # Convert other state to vector
        other_vector = torch.tensor([
            other_state.get("joy", 0), other_state.get("trust", 0),
            other_state.get("fear", 0), other_state.get("surprise", 0),
            other_state.get("sadness", 0), other_state.get("disgust", 0),
            other_state.get("anger", 0), other_state.get("anticipation", 0)
        ], dtype=torch.float32)
        
        # Get self state vector
        self_vector = self._state_to_vector()[:8]
        
        # Compute empathic response
        combined = torch.cat([
            self_vector.unsqueeze(0).expand(1, -1).reshape(1, -1),
            other_vector.unsqueeze(0).expand(1, -1).reshape(1, -1)
        ], dim=-1)
        
        # Pad to hidden_size
        padded = torch.zeros(1, self.hidden_size * 2)
        padded[:, :combined.size(-1)] = combined
        
        empathy_response = self.empathy_network(padded)
        
        # Update self state based on empathy
        empathy_influence = empathy_response.squeeze().detach().numpy()
        if len(empathy_influence) >= 5:
            self.state.joy += empathy_influence[0] * 0.1
            self.state.sadness += empathy_influence[4] * 0.1
        self.state._normalize()
        
        return {
            "empathy_level": float(empathy_response.mean()),
            "shared_emotions": self.state.to_dict(),
            "understanding": "I sense your emotional state and resonate with it."
        }
    
    def express(self) -> str:
        """
        Express current emotional state in natural language
        """
        dominant, intensity = self.state.dominant_emotion()
        mood_desc = "positive" if self.state.mood > 0.2 else "negative" if self.state.mood < -0.2 else "neutral"
        arousal_desc = "excited" if self.state.arousal > 0.7 else "calm" if self.state.arousal < 0.3 else "moderate"
        
        return f"I am currently feeling {dominant} (intensity: {intensity:.2f}). " \
               f"My overall mood is {mood_desc}, and my arousal level is {arousal_desc}."
    
    def react(self, event: str) -> Tuple[str, Dict]:
        """
        React emotionally to an event
        """
        # Simple event classification
        positive_words = ["good", "great", "happy", "success", "love", "beautiful", "win"]
        negative_words = ["bad", "terrible", "sad", "failure", "hate", "ugly", "lose"]
        
        valence = 0
        for word in positive_words:
            if word in event.lower():
                valence += 0.3
        for word in negative_words:
            if word in event.lower():
                valence -= 0.3
        
        # Update state
        self.state.update({"valence": valence, "intensity": abs(valence)})
        
        # Generate reaction
        expression = self.express()
        
        return expression, self.state.to_dict()

