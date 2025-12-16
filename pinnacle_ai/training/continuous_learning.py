"""
Continuous Learning System

Learns from:
- User interactions
- Feedback
- Self-evaluation
"""

import json
import os
from typing import Dict, List
from datetime import datetime
from loguru import logger


class ContinuousLearner:
    """
    Continuous Learning System
    
    Learns from:
    - User interactions
    - Feedback
    - Self-evaluation
    """
    
    def __init__(self, ai):
        self.ai = ai
        self.interactions = []
        self.feedback_log = []
        self.learning_data_dir = "learning_data"
        os.makedirs(self.learning_data_dir, exist_ok=True)
    
    def log_interaction(
        self,
        user_input: str,
        ai_response: str,
        feedback: float = None
    ):
        """Log an interaction for learning"""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "input": user_input,
            "response": ai_response,
            "feedback": feedback  # 1 = good, -1 = bad, None = no feedback
        }
        
        self.interactions.append(interaction)
        
        # Auto-save periodically
        if len(self.interactions) % 100 == 0:
            self._save_interactions()
    
    def receive_feedback(self, interaction_id: int, feedback: float):
        """Receive feedback on a specific interaction"""
        if interaction_id < len(self.interactions):
            self.interactions[interaction_id]["feedback"] = feedback
            self.feedback_log.append({
                "interaction_id": interaction_id,
                "feedback": feedback,
                "timestamp": datetime.now().isoformat()
            })
    
    def get_training_examples(self, min_feedback: float = 0.5) -> List[Dict]:
        """Get positive examples for training"""
        positive_examples = []
        
        for interaction in self.interactions:
            if interaction.get("feedback") and interaction["feedback"] >= min_feedback:
                positive_examples.append({
                    "input": interaction["input"],
                    "output": interaction["response"]
                })
        
        return positive_examples
    
    def analyze_performance(self) -> Dict:
        """Analyze learning performance"""
        if not self.interactions:
            return {"status": "no_data"}
        
        feedback_values = [i["feedback"] for i in self.interactions if i.get("feedback")]
        
        if not feedback_values:
            return {"status": "no_feedback"}
        
        return {
            "total_interactions": len(self.interactions),
            "interactions_with_feedback": len(feedback_values),
            "average_feedback": sum(feedback_values) / len(feedback_values),
            "positive_rate": len([f for f in feedback_values if f > 0]) / len(feedback_values),
            "negative_rate": len([f for f in feedback_values if f < 0]) / len(feedback_values)
        }
    
    def _save_interactions(self):
        """Save interactions to disk"""
        filename = f"interactions_{datetime.now().strftime('%Y%m%d')}.json"
        path = os.path.join(self.learning_data_dir, filename)
        
        with open(path, "w") as f:
            json.dump(self.interactions, f, indent=2)
        
        logger.debug(f"Saved {len(self.interactions)} interactions")
    
    def retrain(self):
        """Trigger retraining with collected data"""
        examples = self.get_training_examples()
        
        if len(examples) < 10:
            logger.warning("Not enough positive examples for retraining")
            return {"status": "insufficient_data"}
        
        logger.info(f"Retraining with {len(examples)} examples...")
        
        # Save training data
        training_file = os.path.join(self.learning_data_dir, "training_data.json")
        with open(training_file, "w") as f:
            json.dump(examples, f, indent=2)
        
        # In production, this would trigger actual retraining
        return {
            "status": "training_data_saved",
            "examples": len(examples),
            "path": training_file
        }

