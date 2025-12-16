"""
Prompt Loader - Loads prompt templates.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

class PromptLoader:
    """Loads and manages prompt templates."""

    def __init__(self, prompts_dir: str = "config/prompts"):
        """Initialize prompt loader."""
        self.prompts_dir = Path(prompts_dir)
        self.prompts = {}
        self.logger = logging.getLogger(__name__)
        self._load_prompts()

    def _load_prompts(self):
        """Load all prompt templates."""
        if not self.prompts_dir.exists():
            self.logger.warning(f"Prompts directory not found: {self.prompts_dir}")
            return
        
        for prompt_file in self.prompts_dir.glob("*.txt"):
            name = prompt_file.stem
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    self.prompts[name] = f.read()
                self.logger.debug(f"Loaded prompt: {name}")
            except Exception as e:
                self.logger.error(f"Error loading prompt {name}: {e}")

    def get_prompt(self, name: str) -> Optional[str]:
        """Get a prompt template by name."""
        return self.prompts.get(name)

    def format_prompt(self, name: str, **kwargs) -> Optional[str]:
        """Format a prompt template with variables."""
        prompt = self.get_prompt(name)
        if prompt:
            try:
                return prompt.format(**kwargs)
            except KeyError as e:
                self.logger.error(f"Missing variable in prompt {name}: {e}")
                return prompt
        return None

