"""
Quantum-Ready Infrastructure
Quantum computing framework for portfolio optimization
"""
from __future__ import annotations

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from qiskit import QuantumCircuit, Aer, execute
    from qiskit.algorithms.optimizers import COBYLA
    from qiskit.circuit.library import RealAmplitudes
    from qiskit.opflow import PauliSumOp
    from qiskit.utils import QuantumInstance
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    logger.warning("Qiskit not available. Quantum features will be disabled.")


class QuantumTradingInfrastructure:
    """
    Quantum computing infrastructure for trading optimization.
    
    Features:
    - Quantum portfolio optimization
    - Quantum state to portfolio weights conversion
    - Quantum circuit definition for trading
    """
    
    def __init__(self, backend: str = 'qasm_simulator'):
        """
        Initialize quantum infrastructure.
        
        Args:
            backend: Quantum backend ('qasm_simulator' or actual quantum device)
        """
        if not QISKIT_AVAILABLE:
            logger.warning("Qiskit not available. Quantum features disabled.")
            self.enabled = False
            return
        
        self.enabled = True
        self.backend = Aer.get_backend(backend)
        self.quantum_instance = QuantumInstance(self.backend, shots=1024)
        self.optimizer = COBYLA(maxiter=100)
    
    def quantum_optimize(
        self,
        portfolio: List[str],
        market_data: Dict[str, Dict]
    ) -> Dict[str, float]:
        """
        Optimize portfolio using quantum computing.
        
        Args:
            portfolio: List of asset symbols
            market_data: Market data dictionary with expected returns and covariance
            
        Returns:
            Dictionary of asset weights
        """
        if not self.enabled:
            logger.warning("Quantum optimization disabled. Using classical optimization.")
            return self._classical_optimization(portfolio, market_data)
        
        try:
            num_assets = len(portfolio)
            
            # Define quantum circuit
            qc = QuantumCircuit(num_assets)
            qc.h(range(num_assets))  # Apply Hadamard gates
            qc.barrier()
            
            # Add parameterized rotations
            theta_params = []
            for i in range(num_assets):
                qc.ry(i, i)  # Parameterized Y rotation
                theta_params.append(i)
            qc.barrier()
            
            # Add entanglement
            for i in range(num_assets - 1):
                qc.cx(i, i + 1)  # CNOT gates for entanglement
            qc.barrier()
            
            # Measure all qubits
            qc.measure_all()
            
            # Define objective function
            def objective(params):
                """Objective function for optimization."""
                # Set parameters
                bound_qc = qc.bind_parameters({i: params[i] for i in range(num_assets)})
                
                # Execute circuit
                result = execute(bound_qc, self.backend, shots=1024).result()
                counts = result.get_counts()
                
                # Calculate expected return and risk
                expected_return = self._calculate_expected_return(
                    counts, portfolio, market_data
                )
                risk = self._calculate_risk(counts, portfolio, market_data)
                
                # Return negative Sharpe ratio (to minimize)
                if risk > 0:
                    return -expected_return / risk
                return 0
            
            # Optimize
            initial_params = np.random.random(num_assets)
            result = self.optimizer.minimize(objective, initial_params)
            
            # Convert to weights
            weights = self._counts_to_weights(result.x, portfolio)
            
            logger.info(f"Quantum optimization completed. Best Sharpe: {-result.fun:.4f}")
            return weights
            
        except Exception as e:
            logger.error(f"Quantum optimization failed: {e}")
            return self._classical_optimization(portfolio, market_data)
    
    def _calculate_expected_return(
        self,
        counts: Dict[str, int],
        portfolio: List[str],
        market_data: Dict[str, Dict]
    ) -> float:
        """Calculate expected return from quantum results."""
        total = sum(counts.values())
        if total == 0:
            return 0.0
        
        expected_return = 0.0
        
        for state, count in counts.items():
            weights = self._state_to_weights(state, portfolio)
            portfolio_return = sum(
                weights[i] * market_data[asset].get('expected_return', 0.0)
                for i, asset in enumerate(portfolio)
            )
            expected_return += portfolio_return * count / total
        
        return expected_return
    
    def _calculate_risk(
        self,
        counts: Dict[str, int],
        portfolio: List[str],
        market_data: Dict[str, Dict]
    ) -> float:
        """Calculate portfolio risk from quantum results."""
        total = sum(counts.values())
        if total == 0:
            return 1.0
        
        portfolio_variance = 0.0
        
        for state, count in counts.items():
            weights = self._state_to_weights(state, portfolio)
            variance = 0.0
            
            # Calculate portfolio variance
            for i, asset1 in enumerate(portfolio):
                for j, asset2 in enumerate(portfolio):
                    covariance = market_data[asset1].get('covariance', {}).get(asset2, 0.0)
                    variance += weights[i] * weights[j] * covariance
            
            portfolio_variance += variance * count / total
        
        return np.sqrt(max(portfolio_variance, 0.0))
    
    def _state_to_weights(self, state: str, portfolio: List[str]) -> np.ndarray:
        """Convert quantum state to portfolio weights."""
        weights = np.zeros(len(portfolio))
        
        # Reverse state string to match qubit order
        state_reversed = state[::-1]
        
        for i, bit in enumerate(state_reversed):
            if i < len(weights) and bit == '1':
                weights[i] = 1.0
        
        # Normalize
        total = np.sum(weights)
        if total > 0:
            weights = weights / total
        
        return weights
    
    def _counts_to_weights(
        self,
        params: np.ndarray,
        portfolio: List[str]
    ) -> Dict[str, float]:
        """Convert optimization parameters to portfolio weights."""
        # Apply softmax to get weights
        exp_params = np.exp(params - np.max(params))
        weights = exp_params / np.sum(exp_params)
        
        # Ensure weights sum to 1
        weights = weights / np.sum(weights)
        
        return {asset: float(weight) for asset, weight in zip(portfolio, weights)}
    
    def _classical_optimization(
        self,
        portfolio: List[str],
        market_data: Dict[str, Dict]
    ) -> Dict[str, float]:
        """Fallback classical optimization."""
        # Simple equal weights
        num_assets = len(portfolio)
        weight = 1.0 / num_assets if num_assets > 0 else 0.0
        
        return {asset: weight for asset in portfolio}

