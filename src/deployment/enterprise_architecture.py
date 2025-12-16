"""
Enterprise-Grade Architecture
"""

import logging
import time
import threading
import queue
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from src.core.orchestrator import OmniAIOrchestrator
from src.tools.config_loader import load_config
from src.core.error_handler import ComprehensiveErrorHandler

logger = logging.getLogger(__name__)


class EnterpriseArchitecture:
    """Enterprise-grade architecture for Pinnacle AI"""

    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = load_config(config_path)
        self.logger = logging.getLogger(__name__)
        self.error_handler = ComprehensiveErrorHandler()

        # Initialize components
        self.orchestrator = OmniAIOrchestrator(self.config)
        self.task_queue = queue.PriorityQueue()
        self.result_queue = queue.Queue()
        self.worker_pool = []
        self.monitoring = EnterpriseMonitoring(self.config)
        self.scaling_manager = AutoScalingManager(self.config)

        # Start workers
        self._initialize_workers()

        # Start monitoring
        self.monitoring.start()

    def _initialize_workers(self):
        """Initialize worker pool"""
        num_workers = self.config.get("deployment", {}).get("num_workers", 4)
        for i in range(num_workers):
            worker = TaskWorker(
                f"worker_{i}",
                self.task_queue,
                self.result_queue,
                self.orchestrator,
                self.error_handler,
                self.monitoring
            )
            worker.start()
            self.worker_pool.append(worker)

    def submit_task(self, task: str, context: Optional[Dict] = None,
                   priority: int = 3) -> str:
        """Submit a task to the enterprise system"""
        if context is None:
            context = {}

        task_id = f"task_{int(time.time())}_{self.task_queue.qsize()}"

        # Add to task queue
        self.task_queue.put((priority, {
            "id": task_id,
            "task": task,
            "context": context,
            "status": "queued",
            "submitted_at": time.time()
        }))

        # Notify monitoring
        self.monitoring.task_submitted(task_id, task, priority)

        return task_id

    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Dict:
        """Get result for a task"""
        # Check if result is already available
        result = self.monitoring.get_result(task_id)
        if result:
            return result

        # Wait for result
        try:
            result = self.result_queue.get(timeout=timeout)
            if result.get("id") == task_id:
                self.monitoring.task_completed(task_id, result)
                return result
            else:
                # Put it back and wait for our task
                self.result_queue.put(result)
                return self.get_result(task_id, timeout)
        except queue.Empty:
            return {"status": "timeout", "message": "Result not available yet"}

    def shutdown(self):
        """Shutdown the enterprise system"""
        self.logger.info("Shutting down enterprise architecture")

        # Stop accepting new tasks
        self.task_queue.put((0, None))  # Sentinel value

        # Stop workers
        for worker in self.worker_pool:
            worker.stop()

        # Stop monitoring
        self.monitoring.stop()

        self.logger.info("Enterprise architecture shutdown complete")


class TaskWorker(threading.Thread):
    """Worker thread for processing tasks"""

    def __init__(self, name: str, task_queue: queue.PriorityQueue,
                 result_queue: queue.Queue, orchestrator: OmniAIOrchestrator,
                 error_handler: ComprehensiveErrorHandler, monitoring):
        super().__init__(name=name, daemon=True)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.orchestrator = orchestrator
        self.error_handler = error_handler
        self.monitoring = monitoring
        self.running = True
        self.logger = logging.getLogger(__name__)

    def run(self):
        """Worker main loop"""
        while self.running:
            try:
                # Get next task
                priority, task = self.task_queue.get()

                # Check for shutdown
                if task is None:
                    self.task_queue.put((priority, task))  # Put it back
                    break

                # Update task status
                task["status"] = "processing"
                task["worker"] = self.name
                task["started_at"] = time.time()
                self.monitoring.task_started(task["id"], self.name)

                # Execute task
                try:
                    result = self.orchestrator.meta_agent.execute(
                        task["task"],
                        task["context"]
                    )
                    result["status"] = "completed"
                    result["completed_at"] = time.time()
                except Exception as e:
                    result = self.error_handler.handle_error(
                        "task_execution",
                        {"task": task["task"], "context": task["context"]},
                        e
                    )
                    result["status"] = "failed"
                    result["error"] = str(e)

                # Add metadata
                result.update({
                    "id": task["id"],
                    "worker": self.name,
                    "processing_time": time.time() - task["started_at"]
                })

                # Put result in queue
                self.result_queue.put(result)

                # Update monitoring
                self.monitoring.task_completed(task["id"], result)

            except Exception as e:
                self.logger.error(f"Worker {self.name} error: {str(e)}")
                self.monitoring.worker_error(self.name, str(e))

    def stop(self):
        """Stop the worker"""
        self.running = False
        self.task_queue.put((0, None))  # Wake up the worker


class EnterpriseMonitoring:
    """Enterprise monitoring system"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.task_status = {}
        self.worker_status = {}
        self.metrics = []
        self.running = False
        self.monitoring_thread = None

    def start(self):
        """Start monitoring"""
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitoring_thread.start()

    def stop(self):
        """Stop monitoring"""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join()

    def _monitor_loop(self):
        """Monitoring main loop"""
        while self.running:
            try:
                # Collect metrics
                self._collect_metrics()

                # Log status
                self.logger.info(f"System status - Tasks: {len(self.task_status)}, "
                               f"Workers: {len(self.worker_status)}")

                # Sleep for interval
                time.sleep(self.config.get("monitoring", {}).get("interval", 60))
            except Exception as e:
                self.logger.error(f"Monitoring error: {str(e)}")

    def _collect_metrics(self):
        """Collect system metrics"""
        # Get task metrics
        queued = sum(1 for status in self.task_status.values() if isinstance(status, str) and status == "queued")
        processing = sum(1 for status in self.task_status.values() if isinstance(status, str) and status == "processing")
        completed = sum(1 for status in self.task_status.values() if isinstance(status, dict) and status.get("status") == "completed")
        failed = sum(1 for status in self.task_status.values() if isinstance(status, str) and status == "failed")

        # Get worker metrics
        active_workers = sum(1 for status in self.worker_status.values() if status == "active")
        idle_workers = sum(1 for status in self.worker_status.values() if status == "idle")

        # Store metrics
        self.metrics.append({
            "timestamp": time.time(),
            "tasks": {
                "queued": queued,
                "processing": processing,
                "completed": completed,
                "failed": failed,
                "total": len(self.task_status)
            },
            "workers": {
                "active": active_workers,
                "idle": idle_workers,
                "total": len(self.worker_status)
            }
        })

        # Keep only recent metrics
        retention = self.config.get("monitoring", {}).get("retention", 100)
        if len(self.metrics) > retention:
            self.metrics = self.metrics[-retention:]

    def task_submitted(self, task_id: str, task: str, priority: int):
        """Record task submission"""
        self.task_status[task_id] = "queued"
        self.logger.info(f"Task submitted: {task_id} (Priority: {priority})")

    def task_started(self, task_id: str, worker: str):
        """Record task start"""
        self.task_status[task_id] = "processing"
        self.worker_status[worker] = "active"
        self.logger.info(f"Task started: {task_id} by {worker}")

    def task_completed(self, task_id: str, result: Dict):
        """Record task completion"""
        self.task_status[task_id] = result
        self.logger.info(f"Task completed: {task_id}")

    def worker_error(self, worker: str, error: str):
        """Record worker error"""
        self.worker_status[worker] = "error"
        self.logger.error(f"Worker error: {worker} - {error}")

    def get_result(self, task_id: str) -> Optional[Dict]:
        """Get result for a task if available"""
        status = self.task_status.get(task_id)
        if isinstance(status, dict) and status.get("status") == "completed":
            return status
        return None

    def get_system_status(self) -> Dict:
        """Get current system status"""
        return {
            "tasks": {
                "queued": sum(1 for status in self.task_status.values() if isinstance(status, str) and status == "queued"),
                "processing": sum(1 for status in self.task_status.values() if isinstance(status, str) and status == "processing"),
                "completed": sum(1 for status in self.task_status.values() if isinstance(status, dict) and status.get("status") == "completed"),
                "failed": sum(1 for status in self.task_status.values() if isinstance(status, str) and status == "failed")
            },
            "workers": {
                "active": sum(1 for status in self.worker_status.values() if status == "active"),
                "idle": sum(1 for status in self.worker_status.values() if status == "idle"),
                "error": sum(1 for status in self.worker_status.values() if status == "error")
            },
            "metrics": self.metrics[-1] if self.metrics else None
        }


class AutoScalingManager:
    """Auto-scaling manager for enterprise deployment"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.scaling_strategy = self._get_scaling_strategy()
        self.current_workers = config.get("deployment", {}).get("num_workers", 4)
        self.min_workers = config.get("deployment", {}).get("min_workers", 2)
        self.max_workers = config.get("deployment", {}).get("max_workers", 16)
        self.scaling_cooldown = config.get("deployment", {}).get("scaling_cooldown", 300)
        self.last_scaling_time = 0

    def _get_scaling_strategy(self):
        """Get scaling strategy based on configuration"""
        strategy_name = self.config.get("deployment", {}).get("scaling_strategy", "reactive")

        if strategy_name == "reactive":
            return ReactiveScalingStrategy(self.config)
        elif strategy_name == "predictive":
            return PredictiveScalingStrategy(self.config)
        elif strategy_name == "adaptive":
            return AdaptiveScalingStrategy(self.config)
        else:
            self.logger.warning(f"Unknown scaling strategy: {strategy_name}, using reactive")
            return ReactiveScalingStrategy(self.config)

    def check_scaling(self, system_status: Dict) -> Dict:
        """Check if scaling is needed and return scaling action"""
        current_time = time.time()
        if current_time - self.last_scaling_time < self.scaling_cooldown:
            return {"action": "none", "reason": "cooldown"}

        action = self.scaling_strategy.determine_action(system_status)

        if action["action"] != "none":
            self.last_scaling_time = current_time

        return action

    def apply_scaling(self, action: Dict, worker_pool: List[TaskWorker]) -> Dict:
        """Apply scaling action"""
        if action["action"] == "scale_up":
            return self._scale_up(worker_pool)
        elif action["action"] == "scale_down":
            return self._scale_down(worker_pool)
        return {"status": "no_action"}

    def _scale_up(self, worker_pool: List[TaskWorker]) -> Dict:
        """Scale up the worker pool"""
        if self.current_workers >= self.max_workers:
            return {"status": "max_capacity"}

        new_worker_count = min(
            self.current_workers + self.config.get("deployment", {}).get("scale_up_step", 2),
            self.max_workers
        )

        self.current_workers = new_worker_count
        self.logger.info(f"Scaling up to {new_worker_count} workers")

        return {
            "status": "scaling_up",
            "new_worker_count": new_worker_count,
            "workers_added": new_worker_count - len(worker_pool)
        }

    def _scale_down(self, worker_pool: List[TaskWorker]) -> Dict:
        """Scale down the worker pool"""
        if self.current_workers <= self.min_workers:
            return {"status": "min_capacity"}

        new_worker_count = max(
            self.current_workers - self.config.get("deployment", {}).get("scale_down_step", 1),
            self.min_workers
        )

        self.current_workers = new_worker_count
        self.logger.info(f"Scaling down to {new_worker_count} workers")

        return {
            "status": "scaling_down",
            "new_worker_count": new_worker_count,
            "workers_removed": len(worker_pool) - new_worker_count
        }


class ScalingStrategy(ABC):
    """Abstract base class for scaling strategies"""

    def __init__(self, config: Dict):
        self.config = config

    @abstractmethod
    def determine_action(self, system_status: Dict) -> Dict:
        """Determine scaling action based on system status"""
        pass


class ReactiveScalingStrategy(ScalingStrategy):
    """Reactive scaling based on current load"""

    def determine_action(self, system_status: Dict) -> Dict:
        """Determine scaling action reactively"""
        tasks = system_status["tasks"]
        workers = system_status["workers"]

        # Calculate load
        queued_tasks = tasks["queued"]
        processing_tasks = tasks["processing"]
        active_workers = workers["active"]
        idle_workers = workers["idle"]

        # Scale up if queue is growing
        if queued_tasks > self.config.get("deployment", {}).get("scale_up_threshold", 10):
            return {
                "action": "scale_up",
                "reason": f"High queue length: {queued_tasks}"
            }

        # Scale down if workers are idle
        if idle_workers > 0 and queued_tasks == 0 and processing_tasks < active_workers:
            return {
                "action": "scale_down",
                "reason": f"Idle workers: {idle_workers}"
            }

        return {"action": "none", "reason": "normal_load"}


class PredictiveScalingStrategy(ScalingStrategy):
    """Predictive scaling based on historical patterns"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.history = []
        self.history_size = config.get("deployment", {}).get("history_size", 100)

    def determine_action(self, system_status: Dict) -> Dict:
        """Determine scaling action predictively"""
        # Add current status to history
        self.history.append(system_status)
        if len(self.history) > self.history_size:
            self.history = self.history[-self.history_size:]

        # If we don't have enough history, use reactive
        if len(self.history) < 10:
            return ReactiveScalingStrategy(self.config).determine_action(system_status)

        # Predict future load
        predicted_load = self._predict_load()

        # Get current capacity
        current_workers = system_status["workers"]["active"] + system_status["workers"]["idle"]

        # Determine action
        if predicted_load > current_workers * 1.2:  # 20% buffer
            return {
                "action": "scale_up",
                "reason": f"Predicted load: {predicted_load}, current capacity: {current_workers}"
            }
        elif predicted_load < current_workers * 0.8:  # 20% buffer
            return {
                "action": "scale_down",
                "reason": f"Predicted load: {predicted_load}, current capacity: {current_workers}"
            }

        return {"action": "none", "reason": "predicted_load_normal"}

    def _predict_load(self) -> float:
        """Predict future load based on history"""
        # Simple moving average prediction
        recent_loads = [status["tasks"]["queued"] + status["tasks"]["processing"]
                       for status in self.history[-10:]]

        return sum(recent_loads) / len(recent_loads) if recent_loads else 0


class AdaptiveScalingStrategy(ScalingStrategy):
    """Adaptive scaling that learns optimal configurations"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.performance_history = []
        self.current_strategy = "reactive"

    def determine_action(self, system_status: Dict) -> Dict:
        """Determine scaling action adaptively"""
        # First, use current strategy
        if self.current_strategy == "reactive":
            action = ReactiveScalingStrategy(self.config).determine_action(system_status)
        else:
            action = PredictiveScalingStrategy(self.config).determine_action(system_status)

        # Record performance
        self._record_performance(system_status, action)

        # Periodically evaluate strategy
        if len(self.performance_history) % 10 == 0:
            self._evaluate_strategy()

        return action

    def _record_performance(self, system_status: Dict, action: Dict):
        """Record system performance"""
        self.performance_history.append({
            "timestamp": time.time(),
            "system_status": system_status,
            "action": action,
            "queue_length": system_status["tasks"]["queued"],
            "processing": system_status["tasks"]["processing"],
            "workers": system_status["workers"]["active"]
        })

        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

    def _evaluate_strategy(self):
        """Evaluate current strategy and switch if needed"""
        if len(self.performance_history) < 20:
            return

        # Calculate performance metrics for each strategy
        reactive_perf = self._calculate_strategy_performance("reactive")
        predictive_perf = self._calculate_strategy_performance("predictive")

        # Switch to better strategy
        if predictive_perf > reactive_perf * 1.1:  # 10% better
            self.current_strategy = "predictive"
            logger.info("Switching to predictive scaling strategy")
        elif reactive_perf > predictive_perf * 1.1:
            self.current_strategy = "reactive"
            logger.info("Switching to reactive scaling strategy")

    def _calculate_strategy_performance(self, strategy: str) -> float:
        """Calculate performance for a strategy"""
        # Filter history for this strategy
        strategy_history = [p for p in self.performance_history
                           if p["action"].get("reason", "").startswith(strategy)]

        if not strategy_history:
            return 0.0

        # Calculate performance score (lower is better)
        total_score = 0.0
        for record in strategy_history:
            # Queue length penalty
            queue_penalty = record["queue_length"] * 0.1

            # Worker utilization
            workers = record["workers"]
            processing = record["processing"]
            utilization = processing / workers if workers > 0 else 0
            utilization_penalty = abs(utilization - 0.8) * 10  # Target 80% utilization

            total_score += queue_penalty + utilization_penalty

        return total_score / len(strategy_history) if strategy_history else 0.0

