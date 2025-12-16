"""
Custom exceptions for Pinnacle AI.
"""

class PinnacleAIError(Exception):
    """Base exception for Pinnacle AI."""
    pass

class AgentError(PinnacleAIError):
    """Error in agent execution."""
    pass

class ConfigError(PinnacleAIError):
    """Configuration error."""
    pass

class ExecutionError(PinnacleAIError):
    """Code execution error."""
    pass

