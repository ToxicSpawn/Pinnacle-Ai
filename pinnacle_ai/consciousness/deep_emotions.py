"""
Deep Emotional System with:
- Sentiment analysis
- Emotion classification
- Emotional memory
- Empathy modeling
- Emotional response generation
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from loguru import logger

try:
    from transformers import pipeline
    TRANSFORMERS_PIPELINE_AVAILABLE = True
except ImportError:
    TRANSFORMERS_PIPELINE_AVAILABLE = False
    logger.warning("Transformers pipeline not available. Using simple emotion detection.")


class EmotionalState:
    """Internal emotional state"""
    
    def __init__(self):
        self.valence = 0.0  # -1 to 1
        self.arousal = 0.5  # 0 to 1
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
    
    def update(self, valence: float, arousal: float):
        """Update state based on new input"""
        # Momentum-based update
        self.valence = 0.8 * self.valence + 0.2 * valence
        self.arousal = 0.8 * self.arousal + 0.2 * arousal
        
        # Update specific emotions based on valence
        if valence > 0:
            self.emotions["joy"] = min(1, self.emotions["joy"] + valence * 0.1)
            self.emotions["trust"] = min(1, self.emotions["trust"] + valence * 0.05)
        else:
            self.emotions["sadness"] = min(1, self.emotions["sadness"] - valence * 0.1)
        
        # Decay
        self._decay()
    
    def _decay(self, rate: float = 0.95):
        """Emotions decay over time"""
        for emotion in self.emotions:
            if emotion in ["joy", "trust", "anticipation"]:
                self.emotions[emotion] = max(0.5, self.emotions[emotion] * rate + 0.5 * (1 - rate))
            else:
                self.emotions[emotion] *= rate
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        dominant = max(self.emotions, key=self.emotions.get)
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "emotions": self.emotions.copy(),
            "dominant": dominant,
            "mood": self.valence
        }


class DeepEmotionalSystem:
    """
    Deep Emotional System with:
    - Sentiment analysis
    - Emotion classification
    - Emotional memory
    - Empathy modeling
    - Emotional response generation
    """
    
    def __init__(self):
        # Load sentiment analyzer
        if TRANSFORMERS_PIPELINE_AVAILABLE:
            try:
                logger.info("Loading emotion models...")
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english"
                )
            except Exception as e:
                logger.warning(f"Could not load sentiment model: {e}")
                self.sentiment_analyzer = None
        else:
            self.sentiment_analyzer = None
        
        # Emotional state
        self.state = EmotionalState()
        
        # Emotional memory
        self.emotional_memories: List[Dict] = []
        
        # Empathy model
        self.empathy_level = 0.7
        
        logger.info("Deep Emotional System initialized")
    
    def analyze(self, text: str) -> Dict:
        """
        Deep emotional analysis of text
        
        Args:
            text: Text to analyze
        
        Returns:
            Emotional analysis
        """
        # Sentiment analysis
        if self.sentiment_analyzer:
            try:
                sentiment = self.sentiment_analyzer(text[:512])[0]
                valence = sentiment["score"] if sentiment["label"] == "POSITIVE" else -sentiment["score"]
            except:
                sentiment = {"label": "NEUTRAL", "score": 0.5}
                valence = 0.0
        else:
            sentiment = {"label": "NEUTRAL", "score": 0.5}
            valence = self._simple_sentiment(text)
        
        # Extract emotional markers
        emotions = self._extract_emotions(text)
        
        # Compute valence and arousal
        arousal = self._compute_arousal(text)
        
        # Update internal state
        self.state.update(valence, arousal)
        
        # Store emotional memory
        self._store_emotional_memory(text, emotions, valence)
        
        return {
            "sentiment": sentiment,
            "emotions": emotions,
            "valence": valence,
            "arousal": arousal,
            "internal_state": self.state.to_dict()
        }
    
    def _simple_sentiment(self, text: str) -> float:
        """Simple sentiment analysis fallback"""
        positive_words = ["good", "great", "happy", "love", "wonderful", "excellent"]
        negative_words = ["bad", "terrible", "sad", "hate", "awful", "horrible"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / max(1, positive_count + negative_count)
    
    def _extract_emotions(self, text: str) -> Dict[str, float]:
        """Extract specific emotions from text"""
        text_lower = text.lower()
        
        emotion_keywords = {
            "joy": ["happy", "joy", "excited", "wonderful", "great", "love", "amazing"],
            "sadness": ["sad", "depressed", "unhappy", "miserable", "crying", "grief"],
            "anger": ["angry", "furious", "annoyed", "frustrated", "rage", "hate"],
            "fear": ["afraid", "scared", "terrified", "anxious", "worried", "nervous"],
            "surprise": ["surprised", "shocked", "amazed", "astonished", "unexpected"],
            "disgust": ["disgusted", "revolted", "gross", "nasty", "horrible"],
            "trust": ["trust", "believe", "confident", "reliable", "honest"],
            "anticipation": ["excited", "looking forward", "anticipate", "eager", "hopeful"]
        }
        
        emotions = {}
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower) / max(1, len(keywords))
            emotions[emotion] = min(1.0, score * 2)  # Scale up
        
        return emotions
    
    def _compute_arousal(self, text: str) -> float:
        """Compute emotional arousal level"""
        # Arousal indicators
        high_arousal = ["!", "?!", "OMG", "wow", "amazing", "terrible", "incredible"]
        low_arousal = ["calm", "peaceful", "quiet", "relaxed", "bored"]
        
        text_lower = text.lower()
        
        high_count = sum(1 for word in high_arousal if word.lower() in text_lower)
        low_count = sum(1 for word in low_arousal if word in text_lower)
        
        # Exclamation marks increase arousal
        high_count += text.count("!")
        
        arousal = 0.5 + (high_count - low_count) * 0.1
        return max(0, min(1, arousal))
    
    def _store_emotional_memory(self, text: str, emotions: Dict, valence: float):
        """Store emotional memory"""
        memory = {
            "text": text[:200],
            "emotions": emotions,
            "valence": valence,
            "timestamp": len(self.emotional_memories)
        }
        
        self.emotional_memories.append(memory)
        
        # Keep only recent memories
        if len(self.emotional_memories) > 1000:
            self.emotional_memories = self.emotional_memories[-1000:]
    
    def empathize(self, user_emotion: Dict) -> str:
        """
        Generate empathetic response
        
        Args:
            user_emotion: User's emotional state
        
        Returns:
            Empathetic response
        """
        emotions = user_emotion.get("emotions", {})
        if emotions:
            dominant_emotion = max(emotions, key=emotions.get)
        else:
            dominant_emotion = "neutral"
        valence = user_emotion.get("valence", 0)
        
        # Empathetic responses based on emotion
        responses = {
            "joy": "I can feel your happiness! It's wonderful to share in your joy.",
            "sadness": "I sense your sadness, and I want you to know I'm here for you.",
            "anger": "I understand your frustration. Those feelings are valid.",
            "fear": "It's okay to feel afraid. I'm here to help you through this.",
            "surprise": "Wow, that must have been quite unexpected!",
            "disgust": "That sounds really unpleasant. I can understand your reaction.",
            "trust": "Thank you for sharing with me. I value your trust.",
            "anticipation": "The excitement is palpable! I'm looking forward to hearing more."
        }
        
        base_response = responses.get(dominant_emotion, "I hear you and I'm here to help.")
        
        # Adjust based on valence
        if valence < -0.5:
            base_response += " Remember, difficult times don't last forever."
        elif valence > 0.5:
            base_response += " Your positive energy is contagious!"
        
        return base_response
    
    def get_state(self) -> Dict:
        """Get current emotional state"""
        return self.state.to_dict()
    
    def express(self) -> str:
        """Express current emotional state"""
        state = self.state.to_dict()
        dominant = state["dominant"]
        mood = state["mood"]
        
        mood_desc = "positive" if mood > 0.2 else "contemplative" if mood < -0.2 else "balanced"
        
        return f"I'm feeling {dominant} right now. My overall mood is {mood_desc}."

