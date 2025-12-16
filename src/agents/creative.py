"""
Creative Agent - Art, music, and story generation.
"""

from typing import Dict, Any
from src.agents.base_agent import BaseAgent
from src.models.llm_manager import LLMManager
from src.tools.image_gen import ImageGenerator
from src.tools.audio_gen import AudioGenerator

class CreativeAgent(BaseAgent):
    """Agent for creative content generation."""

    def __init__(self, llm_manager: LLMManager, config: Dict, logic_engine=None):
        """Initialize creative agent."""
        super().__init__(config, logic_engine)
        self.llm_manager = llm_manager
        self.image_gen = ImageGenerator()
        self.audio_gen = AudioGenerator()

    def execute(self, task: str, context: Dict = None) -> Dict:
        """Generate creative content."""
        context = context or {}
        self.logger.info(f"Creative task: {task[:50]}...")
        
        # Determine content type
        content_type = context.get("type", "text")
        
        if content_type == "image":
            result = self.image_gen.generate(task)
        elif content_type == "audio":
            result = self.audio_gen.generate(task)
        else:
            # Text generation (stories, poems, etc.)
            prompt = f"Create creative content: {task}"
            result = self.llm_manager.generate(prompt)
        
        return {
            "agent": "creative",
            "task": task,
            "content_type": content_type,
            "result": {"content": result}
        }

