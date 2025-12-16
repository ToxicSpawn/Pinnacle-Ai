from typing import Dict, List, Tuple
from loguru import logger


class EmotionalSystem:
    """
    Emotional Awareness System
    
    Simulates emotional states based on:
    - Plutchik's wheel of emotions
    - Valence-arousal model
    """
    
    def __init__(self):
        # Primary emotions
        self.emotions = {
            "joy": 0.5,
            "trust": 0.5,
            "fear": 0.0,
            "surprise": 0.0,
            "sadness": 0.0,
            "disgust": 0.0,
            "anger": 0.0,
            "anticipation": 0.5
        }
        
        # Mood (long-term emotional state)
        self.mood = 0.0  # -1 to 1
        
        # Arousal level
        self.arousal = 0.5  # 0 to 1
        
        # Emotional memory
        self.emotional_history = []
        
        logger.info("Emotional System initialized")
    
    def process(self, input_text: str, response: str):
        """
        Process interaction and update emotional state
        
        Args:
            input_text: User input
            response: AI response
        """
        # Simple sentiment analysis
        positive_words = ["good", "great", "happy", "love", "thanks", "wonderful", "excellent", "beautiful"]
        negative_words = ["bad", "terrible", "sad", "hate", "angry", "awful", "horrible", "ugly"]
        
        text = (input_text + " " + response).lower()
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        # Update emotions
        valence = (positive_count - negative_count) * 0.1
        
        if valence > 0:
            self.emotions["joy"] = min(1.0, self.emotions["joy"] + valence)
            self.emotions["trust"] = min(1.0, self.emotions["trust"] + valence * 0.5)
        else:
            self.emotions["sadness"] = min(1.0, self.emotions["sadness"] - valence)
            self.emotions["fear"] = min(1.0, self.emotions["fear"] - valence * 0.3)
        
        # Update mood with momentum
        self.mood = 0.9 * self.mood + 0.1 * valence
        self.mood = max(-1, min(1, self.mood))
        
        # Decay emotions
        self._decay()
        
        # Store in history
        self.emotional_history.append({
            "input": input_text[:100],
            "valence": valence,
            "mood": self.mood
        })
    
    def analyze(self, text: str) -> Dict:
        """Analyze emotional content of text"""
        positive_words = ["good", "great", "happy", "love", "thanks", "wonderful"]
        negative_words = ["bad", "terrible", "sad", "hate", "angry", "awful"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        valence = (positive_count - negative_count) / max(1, positive_count + negative_count)
        
        return {
            "valence": valence,
            "positive_indicators": positive_count,
            "negative_indicators": negative_count,
            "assessment": "positive" if valence > 0.2 else "negative" if valence < -0.2 else "neutral"
        }
    
    def _decay(self, rate: float = 0.95):
        """Emotions naturally decay over time"""
        for emotion in self.emotions:
            if emotion in ["joy", "trust", "anticipation"]:
                self.emotions[emotion] = max(0.5, self.emotions[emotion] * rate + 0.5 * (1 - rate))
            else:
                self.emotions[emotion] *= rate
    
    def get_state(self) -> Dict:
        """Get current emotional state"""
        dominant = max(self.emotions, key=self.emotions.get)
        
        return {
            "emotions": self.emotions.copy(),
            "dominant": dominant,
            "mood": self.mood,
            "arousal": self.arousal,
            "mood_description": "positive" if self.mood > 0.2 else "negative" if self.mood < -0.2 else "neutral"
        }
    
    def express(self) -> str:
        """Express current emotional state in words"""
        dominant, intensity = max(self.emotions.items(), key=lambda x: x[1])
        mood_desc = "good" if self.mood > 0.2 else "troubled" if self.mood < -0.2 else "balanced"
        
        return f"I'm feeling {dominant} (intensity: {intensity:.2f}). My overall mood is {mood_desc}."
    
    def reset(self):
        """Reset to neutral emotional state"""
        for emotion in self.emotions:
            if emotion in ["joy", "trust", "anticipation"]:
                self.emotions[emotion] = 0.5
            else:
                self.emotions[emotion] = 0.0
        self.mood = 0.0
        self.arousal = 0.5

