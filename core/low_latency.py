"""
Low-Latency Infrastructure Optimizations
Kernel-level and hardware optimizations for HFT
"""
from __future__ import annotations

import os
import sys
import time
import logging
import platform
from typing import Optional, Dict
import subprocess

logger = logging.getLogger(__name__)


class LowLatencyEngine:
    """
    Low-latency engine for HFT optimizations.
    
    Features:
    - Network kernel optimizations
    - CPU pinning
    - Memory optimizations
    - Latency measurement
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize low-latency engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.is_linux = platform.system() == 'Linux'
        self.is_root = os.geteuid() == 0 if self.is_linux else False
        
        if not self.is_linux:
            logger.warning("Low-latency optimizations are Linux-specific")
        
        if self.is_linux and not self.is_root:
            logger.warning("Some optimizations require root privileges")
    
    def optimize_network(self) -> bool:
        """
        Apply kernel-level network optimizations.
        
        Returns:
            True if optimizations applied successfully
        """
        if not self.is_linux:
            return False
        
        optimizations = {
            # Disable Nagle's algorithm for low latency
            'net.ipv4.tcp_no_metrics_save': '1',
            # Increase TCP buffer sizes
            'net.core.rmem_max': '16777216',
            'net.core.wmem_max': '16777216',
            'net.core.rmem_default': '16777216',
            'net.core.wmem_default': '16777216',
            # TCP keepalive settings
            'net.ipv4.tcp_keepalive_time': '60',
            'net.ipv4.tcp_keepalive_probes': '3',
            'net.ipv4.tcp_keepalive_intvl': '10',
            # TCP congestion control (BBR for better performance)
            'net.core.default_qdisc': 'fq',
            'net.ipv4.tcp_congestion_control': 'bbr',
            # Disable TCP slow start
            'net.ipv4.tcp_slow_start_after_idle': '0',
            # Increase connection tracking
            'net.netfilter.nf_conntrack_max': '1000000',
        }
        
        applied = 0
        for key, value in optimizations.items():
            try:
                if self.is_root:
                    subprocess.run(
                        ['sysctl', '-w', f'{key}={value}'],
                        check=True,
                        capture_output=True
                    )
                    applied += 1
                else:
                    logger.warning(f"Cannot set {key} without root privileges")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                logger.warning(f"Failed to set {key}: {e}")
        
        if applied > 0:
            logger.info(f"✅ Applied {applied} network optimizations")
        
        return applied > 0
    
    def optimize_hardware(self, cpu_cores: Optional[list] = None) -> bool:
        """
        Apply hardware optimizations.
        
        Args:
            cpu_cores: List of CPU cores to pin to (e.g., [0, 1, 2, 3])
            
        Returns:
            True if optimizations applied successfully
        """
        if not self.is_linux:
            return False
        
        applied = 0
        
        # CPU pinning
        if cpu_cores:
            try:
                pid = os.getpid()
                cores_str = ','.join(map(str, cpu_cores))
                if self.is_root:
                    subprocess.run(
                        ['taskset', '-cp', cores_str, str(pid)],
                        check=True,
                        capture_output=True
                    )
                    logger.info(f"✅ Pinned process to CPU cores: {cores_str}")
                    applied += 1
                else:
                    logger.warning("CPU pinning requires root privileges")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                logger.warning(f"Failed to pin CPU: {e}")
        
        # Huge pages for better memory performance
        try:
            if self.is_root:
                subprocess.run(
                    ['sysctl', '-w', 'vm.nr_hugepages=128'],
                    check=True,
                    capture_output=True
                )
                logger.info("✅ Configured huge pages")
                applied += 1
            else:
                logger.warning("Huge pages configuration requires root privileges")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"Failed to configure huge pages: {e}")
        
        # Set process priority
        try:
            os.nice(-10)  # Higher priority (requires root)
            logger.info("✅ Set process priority")
            applied += 1
        except (OSError, PermissionError):
            logger.warning("Process priority setting requires root privileges")
        
        return applied > 0
    
    def measure_latency(
        self,
        exchange,
        symbol: str = 'BTC/USDT',
        iterations: int = 10
    ) -> Dict[str, float]:
        """
        Measure round-trip latency to exchange.
        
        Args:
            exchange: Exchange instance
            symbol: Trading symbol to test
            iterations: Number of iterations
            
        Returns:
            Dictionary with latency statistics
        """
        latencies = []
        
        for i in range(iterations):
            start = time.perf_counter_ns()
            try:
                exchange.fetch_ticker(symbol)
            except Exception as e:
                logger.warning(f"Latency measurement failed: {e}")
                continue
            end = time.perf_counter_ns()
            
            latency_ms = (end - start) / 1_000_000
            latencies.append(latency_ms)
        
        if not latencies:
            return {'error': 'No successful measurements'}
        
        return {
            'min': min(latencies),
            'max': max(latencies),
            'mean': sum(latencies) / len(latencies),
            'median': sorted(latencies)[len(latencies) // 2],
            'std': (
                sum((x - sum(latencies) / len(latencies))**2 for x in latencies) / len(latencies)
            ) ** 0.5,
            'p95': sorted(latencies)[int(len(latencies) * 0.95)],
            'p99': sorted(latencies)[int(len(latencies) * 0.99)],
        }
    
    def optimize_all(self, cpu_cores: Optional[list] = None) -> Dict[str, bool]:
        """
        Apply all optimizations.
        
        Args:
            cpu_cores: CPU cores to pin to
            
        Returns:
            Dictionary with optimization results
        """
        results = {
            'network': self.optimize_network(),
            'hardware': self.optimize_hardware(cpu_cores),
        }
        
        logger.info(f"Optimization results: {results}")
        return results


class FPGAOrderExecutor:
    """
    FPGA-based order executor (framework).
    
    Note: Actual FPGA implementation requires hardware and specialized libraries.
    This is a framework that can be extended with actual FPGA integration.
    """
    
    def __init__(self, fpga_ip: Optional[str] = None):
        """
        Initialize FPGA executor.
        
        Args:
            fpga_ip: FPGA IP address (if using network-attached FPGA)
        """
        self.fpga_ip = fpga_ip
        self.fpga_available = False
        
        # Check if FPGA libraries are available
        try:
            # This would be replaced with actual FPGA library
            # import fpga_library
            # self.fpga = fpga_library.connect(fpga_ip)
            self.fpga_available = False  # Set to True when FPGA is available
            logger.info("FPGA executor initialized (framework only)")
        except ImportError:
            logger.warning("FPGA libraries not available. Using software fallback.")
    
    def _load_bitstream(self, bitstream_path: str) -> bool:
        """
        Load FPGA bitstream.
        
        Args:
            bitstream_path: Path to bitstream file
            
        Returns:
            True if loaded successfully
        """
        if not self.fpga_available:
            logger.warning("FPGA not available. Cannot load bitstream.")
            return False
        
        # Actual implementation would load bitstream here
        logger.info(f"Would load bitstream: {bitstream_path}")
        return True
    
    def execute_order(self, order: Dict) -> Optional[Dict]:
        """
        Execute order with FPGA acceleration.
        
        Args:
            order: Order dictionary
            
        Returns:
            Execution result or None
        """
        if not self.fpga_available:
            logger.warning("FPGA not available. Using software execution.")
            return None
        
        # Convert order to FPGA-compatible format
        fpga_order = {
            'symbol': order['symbol'].encode(),
            'side': 0 if order['side'] == 'buy' else 1,
            'price': int(order.get('price', 0) * 1e8),  # Fixed-point
            'amount': int(order.get('amount', 0) * 1e8),
            'type': 0 if order.get('type', 'market') == 'limit' else 1
        }
        
        # Send to FPGA (actual implementation)
        # result = self.fpga.send(fpga_order)
        # return self.fpga.receive()
        
        logger.warning("FPGA execution not implemented. Use software executor.")
        return None

