"""
Comprehensive Error Handler with Recovery Mechanisms
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


class ComprehensiveErrorHandler:
    """Advanced error handling system with recovery mechanisms"""

    def __init__(self):
        self.error_log = deque(maxlen=1000)
        self.recovery_strategies = {
            'llm_failure': self._handle_llm_failure,
            'memory_error': self._handle_memory_error,
            'agent_failure': self._handle_agent_failure,
            'resource_limit': self._handle_resource_limit,
            'timeout': self._handle_timeout
        }
        logger.info("Comprehensive Error Handler initialized")

    def handle_error(self, error_type: str, context: Dict, error: Exception) -> Dict:
        """Handle errors with appropriate recovery strategies"""
        error_str = str(error)
        
        self.error_log.append({
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'context': context,
            'error': error_str
        })

        logger.error(f"Error [{error_type}]: {error_str}")

        if error_type in self.recovery_strategies:
            try:
                return self.recovery_strategies[error_type](context, error)
            except Exception as e:
                logger.error(f"Recovery strategy failed: {str(e)}")
                return self._default_recovery(context, error)
        
        return self._default_recovery(context, error)

    def _handle_llm_failure(self, context: Dict, error: Exception) -> Dict:
        """Handle LLM service failures"""
        # Try alternative models
        current_provider = context.get('llm_provider', 'openai')
        
        if current_provider == 'openai':
            return {
                'action': 'switch_model',
                'new_model': 'ollama',
                'message': 'Switching to alternative LLM provider'
            }
        elif current_provider == 'ollama':
            return {
                'action': 'switch_model',
                'new_model': 'local',
                'message': 'Switching to local model'
            }
        
        return {
            'action': 'retry',
            'delay': 5,
            'message': 'Retrying with same provider'
        }

    def _handle_memory_error(self, context: Dict, error: Exception) -> Dict:
        """Handle memory system failures"""
        # Implement memory compaction
        return {
            'action': 'compact_memory',
            'priority': 'high',
            'message': 'Initiating memory compaction'
        }

    def _handle_agent_failure(self, context: Dict, error: Exception) -> Dict:
        """Handle agent execution failures"""
        # Fallback to simpler agent or meta-agent
        original_agent = context.get('agent', 'unknown')
        
        return {
            'action': 'fallback_agent',
            'original_agent': original_agent,
            'fallback_agent': 'meta_agent',
            'message': f'Falling back from {original_agent} to meta_agent'
        }

    def _handle_resource_limit(self, context: Dict, error: Exception) -> Dict:
        """Handle resource limitations"""
        # Reduce resource usage
        current_max_tokens = context.get('max_tokens', 2048)
        new_max_tokens = max(512, current_max_tokens // 2)
        
        return {
            'action': 'reduce_resources',
            'parameters': {
                'max_tokens': new_max_tokens,
                'batch_size': max(1, context.get('batch_size', 8) // 2)
            },
            'message': f'Reducing max_tokens to {new_max_tokens}'
        }

    def _handle_timeout(self, context: Dict, error: Exception) -> Dict:
        """Handle execution timeouts"""
        # Break task into smaller chunks
        return {
            'action': 'decompose_task',
            'chunk_size': 'small',
            'message': 'Decomposing task into smaller chunks'
        }

    def _default_recovery(self, context: Dict, error: Exception) -> Dict:
        """Default recovery strategy"""
        return {
            'action': 'notify_user',
            'message': f'Task failed: {str(error)}',
            'error_type': type(error).__name__
        }

    def get_error_statistics(self) -> Dict:
        """Get error statistics"""
        if not self.error_log:
            return {"status": "no_errors"}
        
        error_counts = {}
        for entry in self.error_log:
            error_type = entry['error_type']
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return {
            "total_errors": len(self.error_log),
            "error_counts": error_counts,
            "recent_errors": list(self.error_log)[-10:]
        }

