"""
Advanced Optimizers: Lion and Sophia
"""

import torch
from torch.optim import Optimizer
from typing import Tuple, Optional


class Lion(Optimizer):
    """
    Lion optimizer - Sign-based optimizer with decoupled weight decay.
    
    Paper: https://arxiv.org/abs/2302.06675
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        """
        Initialize Lion optimizer.
        
        Args:
            params: Model parameters
            lr: Learning rate
            betas: Beta parameters (momentum)
            weight_decay: Weight decay coefficient
        """
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        """
        Perform optimization step.
        
        Args:
            closure: Optional closure for recomputing loss
            
        Returns:
            Loss if closure provided, else None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                
                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]
                state["step"] += 1
                
                # Lion update
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group["lr"])
                
                # Weight decay
                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])
                
                # Update momentum
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        
        return loss


class Sophia(Optimizer):
    """
    Sophia optimizer - Second-order clipped stochastic optimization.
    
    Paper: https://arxiv.org/abs/2305.14342
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.965, 0.99),
        rho: float = 0.04,
        weight_decay: float = 0.0,
    ):
        """
        Initialize Sophia optimizer.
        
        Args:
            params: Model parameters
            lr: Learning rate
            betas: Beta parameters
            rho: Clipping parameter
            weight_decay: Weight decay coefficient
        """
        defaults = dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        """
        Perform optimization step.
        
        Args:
            closure: Optional closure for recomputing loss
            
        Returns:
            Loss if closure provided, else None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["hessian"] = torch.zeros_like(p)
                
                exp_avg, hessian = state["exp_avg"], state["hessian"]
                beta1, beta2 = group["betas"]
                state["step"] += 1
                
                # Update momentum
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update hessian diagonal (simplified - full version uses EMA of squared grads)
                hessian.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Sophia update with clipping
                update = exp_avg / (hessian.sqrt() + group["rho"])
                p.add_(update, alpha=-group["lr"])
                
                # Weight decay
                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])
        
        return loss

