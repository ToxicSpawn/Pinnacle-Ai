"""
Advanced Performance Optimization System
"""

import time
import logging
from typing import Dict, Any, Optional
from collections import deque

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available. Resource monitoring will be limited.")

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    logging.warning("GPUtil not available. GPU monitoring will be limited.")

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """Advanced performance optimization system"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.performance_metrics = deque(maxlen=1000)
        self.resource_limits = {
            'cpu': config.get('performance', {}).get('max_cpu', 80),
            'memory': config.get('performance', {}).get('max_memory', 80),
            'gpu': config.get('performance', {}).get('max_gpu', 90)
        }

    def monitor_resources(self) -> Dict:
        """Monitor system resources"""
        resources = {}
        
        if PSUTIL_AVAILABLE:
            resources['cpu'] = psutil.cpu_percent(interval=0.1)
            resources['memory'] = psutil.virtual_memory().percent
        else:
            resources['cpu'] = 0
            resources['memory'] = 0

        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                resources['gpu'] = gpus[0].load * 100 if gpus else 0
            except:
                resources['gpu'] = 0
        else:
            resources['gpu'] = 0

        self.performance_metrics.append({
            'timestamp': time.time(),
            'cpu': resources['cpu'],
            'memory': resources['memory'],
            'gpu': resources['gpu']
        })

        return resources

    def optimize_execution(self, task: str, context: Optional[Dict] = None) -> Dict:
        """Optimize task execution based on current resources"""
        if context is None:
            context = {}

        resources = self.monitor_resources()

        # Adjust parameters based on resource availability
        if resources['cpu'] > self.resource_limits['cpu']:
            context['max_workers'] = max(1, context.get('max_workers', 4) - 1)
            self.logger.warning(f"High CPU usage ({resources['cpu']:.1f}%). Reducing workers to {context['max_workers']}")

        if resources['memory'] > self.resource_limits['memory']:
            context['batch_size'] = max(1, context.get('batch_size', 8) // 2)
            self.logger.warning(f"High memory usage ({resources['memory']:.1f}%). Reducing batch size to {context['batch_size']}")

        if resources['gpu'] > self.resource_limits['gpu']:
            context['precision'] = 'fp16'  # Switch to half precision
            self.logger.warning(f"High GPU usage ({resources['gpu']:.1f}%). Switching to fp16 precision")

        # Task-specific optimizations
        task_lower = task.lower()
        if "code" in task_lower or "script" in task_lower:
            context['execution_environment'] = 'local'  # Faster for code execution
        elif "research" in task_lower:
            context['max_search_results'] = min(3, context.get('max_search_results', 5))
        elif "creative" in task_lower:
            context['quality'] = min(0.9, context.get('quality', 1.0))

        return context

    def get_optimization_suggestions(self) -> Dict:
        """Generate performance optimization suggestions"""
        if len(self.performance_metrics) < 5:
            return {"status": "insufficient_data"}

        # Analyze recent performance
        recent = list(self.performance_metrics)[-5:]
        avg_cpu = sum(m['cpu'] for m in recent) / len(recent)
        avg_memory = sum(m['memory'] for m in recent) / len(recent)
        avg_gpu = sum(m['gpu'] for m in recent) / len(recent)

        suggestions = {}

        if avg_cpu > self.resource_limits['cpu']:
            suggestions['cpu'] = {
                'current': avg_cpu,
                'limit': self.resource_limits['cpu'],
                'suggestion': 'Reduce parallel workers or optimize CPU-bound tasks'
            }

        if avg_memory > self.resource_limits['memory']:
            suggestions['memory'] = {
                'current': avg_memory,
                'limit': self.resource_limits['memory'],
                'suggestion': 'Reduce batch size or optimize memory usage'
            }

        if avg_gpu > self.resource_limits['gpu']:
            suggestions['gpu'] = {
                'current': avg_gpu,
                'limit': self.resource_limits['gpu'],
                'suggestion': 'Switch to lower precision or reduce model size'
            }

        if not suggestions:
            suggestions['status'] = 'optimal'

        return suggestions

