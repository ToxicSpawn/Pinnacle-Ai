"""
Self-Healing System
Automatic error recovery and system health monitoring
"""
from __future__ import annotations

import logging
import time
import traceback
import os
import sys
import signal
from typing import Dict, List, Optional, Callable, Any
from functools import wraps
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available. Some health checks will be disabled.")


class SelfHealingSystem:
    """
    Self-healing system for automatic error recovery.
    
    Features:
    - Automatic retry with exponential backoff
    - Health monitoring
    - Resource usage tracking
    - Graceful degradation
    - Automatic restart
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize self-healing system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 5)
        self.health_check_interval = self.config.get('health_check_interval', 60)
        self.max_cpu = self.config.get('max_cpu', 80)
        self.max_memory = self.config.get('max_memory', 80)
        self.max_restarts = self.config.get('max_restarts', 5)
        
        self.restart_count = 0
        self.last_health_check = 0
        self.error_history: List[Dict] = []
        self.health_status = 'healthy'
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
    
    def handle_errors(self, func: Callable) -> Callable:
        """
        Decorator for automatic error handling and recovery.
        
        Args:
            func: Function to wrap
            
        Returns:
            Wrapped function
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(self.max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    self._record_error(func.__name__, e, attempt)
                    
                    logger.error(
                        f"Attempt {attempt + 1}/{self.max_retries} failed in {func.__name__}: {e}"
                    )
                    
                    # Check if we should restart
                    if self._should_restart():
                        logger.warning("System health critical. Restarting...")
                        self._restart_process()
                        return None
                    
                    # Exponential backoff
                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay * (2 ** attempt)
                        logger.info(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
            
            # All retries failed
            logger.error(f"All {self.max_retries} attempts failed for {func.__name__}")
            self._handle_critical_failure(func, args, kwargs, last_exception)
            
            if self.config.get('raise_on_failure', False):
                raise last_exception
            
            return None
        
        return wrapper
    
    def _record_error(self, func_name: str, exception: Exception, attempt: int):
        """Record error in history."""
        self.error_history.append({
            'function': func_name,
            'error': str(exception),
            'attempt': attempt,
            'timestamp': datetime.now().isoformat(),
            'traceback': traceback.format_exc()
        })
        
        # Keep only recent errors
        if len(self.error_history) > 100:
            self.error_history.pop(0)
    
    def _should_restart(self) -> bool:
        """Determine if process should be restarted."""
        # Check restart count
        if self.restart_count >= self.max_restarts:
            logger.warning(f"Max restarts ({self.max_restarts}) exceeded")
            return False
        
        # Check system health
        if not self._check_system_health():
            return True
        
        # Check for crash loop
        if self._is_crash_loop():
            logger.warning("Crash loop detected. Not restarting.")
            return False
        
        return False  # Default: don't restart
    
    def _check_system_health(self) -> bool:
        """Check system health metrics."""
        if not PSUTIL_AVAILABLE:
            return True  # Assume healthy if can't check
        
        try:
            # Check CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            if cpu_usage > self.max_cpu:
                logger.warning(f"High CPU usage: {cpu_usage:.1f}%")
                self.health_status = 'degraded'
                return False
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > self.max_memory:
                logger.warning(f"High memory usage: {memory.percent:.1f}%")
                self.health_status = 'degraded'
                return False
            
            # Check disk usage
            disk = psutil.disk_usage('/')
            if disk.percent > 90:
                logger.warning(f"High disk usage: {disk.percent:.1f}%")
                self.health_status = 'degraded'
                return False
            
            self.health_status = 'healthy'
            return True
        
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return True  # Assume healthy if check fails
    
    def _is_crash_loop(self) -> bool:
        """Check if we're in a crash loop."""
        if not PSUTIL_AVAILABLE:
            return False
        
        try:
            # Get process start time
            process = psutil.Process(os.getpid())
            start_time = process.create_time()
            
            # If process has been running for less than 1 minute and we've had multiple restarts
            if time.time() - start_time < 60 and self.restart_count > 3:
                return True
            
            # Check error rate
            recent_errors = [
                e for e in self.error_history
                if time.time() - datetime.fromisoformat(e['timestamp']).timestamp() < 60
            ]
            
            if len(recent_errors) > 10:
                return True
            
            return False
        
        except Exception:
            return False
    
    def _restart_process(self):
        """Restart the current process."""
        logger.info("Restarting process...")
        self.restart_count += 1
        
        # Save state
        self._save_state()
        
        # Restart
        try:
            os.execl(sys.executable, sys.executable, *sys.argv)
        except Exception as e:
            logger.error(f"Failed to restart: {e}")
            self._graceful_shutdown()
    
    def _handle_critical_failure(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        exception: Exception
    ):
        """Handle critical failures."""
        logger.critical(f"Critical failure in {func.__name__}")
        
        # Send critical alert
        self._send_critical_alert(func, args, kwargs, exception)
        
        # Attempt graceful shutdown if configured
        if self.config.get('graceful_shutdown_on_critical', False):
            self._graceful_shutdown()
    
    def _send_critical_alert(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        exception: Exception
    ):
        """Send critical alert."""
        alert_message = (
            f"🚨 CRITICAL FAILURE\n"
            f"Function: {func.__name__}\n"
            f"Error: {str(exception)}\n"
            f"Args: {args}\n"
            f"Kwargs: {kwargs}\n"
            f"Traceback:\n{traceback.format_exc()}"
        )
        
        logger.critical(alert_message)
        
        # In production, this would send to alerting system
        # await self.telegram.send_error_alert(alert_message)
    
    def _graceful_shutdown(self):
        """Attempt graceful shutdown."""
        logger.info("Attempting graceful shutdown...")
        
        # Close connections
        self._close_connections()
        
        # Save state
        self._save_state()
        
        # Exit
        os._exit(1)
    
    def _close_connections(self):
        """Close all open connections."""
        # In production, this would close:
        # - Database connections
        # - Exchange connections
        # - WebSocket connections
        # - File handles
        pass
    
    def _save_state(self):
        """Save current state."""
        # In production, this would save:
        # - Current positions
        # - Active orders
        # - Strategy state
        # - Configuration
        pass
    
    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}. Shutting down gracefully...")
        self._graceful_shutdown()
    
    def start_health_monitoring(self):
        """Start health monitoring in background."""
        import threading
        
        monitor_thread = threading.Thread(
            target=self._health_monitoring_loop,
            daemon=True
        )
        monitor_thread.start()
        
        logger.info("Health monitoring started")
    
    def _health_monitoring_loop(self):
        """Health monitoring loop."""
        while True:
            try:
                self._check_health()
                time.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Health monitoring failed: {e}")
                time.sleep(self.health_check_interval)
    
    def _check_health(self):
        """Perform health check."""
        # Check system resources
        if not self._check_system_health():
            logger.warning("System health check failed")
            if self._should_restart():
                self._restart_process()
        
        # Check component health
        self._check_component_health()
    
    def _check_component_health(self):
        """Check health of individual components."""
        # In production, this would check:
        # - Database connection
        # - Exchange connections
        # - WebSocket connections
        # - Strategy performance
        # - Order execution quality
        pass
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        return {
            'status': self.health_status,
            'restart_count': self.restart_count,
            'recent_errors': len([
                e for e in self.error_history
                if time.time() - datetime.fromisoformat(e['timestamp']).timestamp() < 3600
            ]),
            'system_health': self._check_system_health() if PSUTIL_AVAILABLE else 'unknown'
        }

