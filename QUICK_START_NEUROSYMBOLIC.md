# 🚀 Quick Start: Neurosymbolic AI

## Installation

```bash
# Install dependencies (PyKE is optional)
pip install torch transformers

# Note: PyKE is not available via pip
# The system works without it using fallback mode
```

## Basic Usage

### 1. Logic Engine

```python
from pinnacle_ai.core.neurosymbolic import LogicEngine

engine = LogicEngine()
proof = engine.prove("irrational(sqrt(2))")
print(proof)
```

### 2. Neurosymbolic Model

```python
from pinnacle_ai.core.neurosymbolic import NeurosymbolicMistral
from pinnacle_ai.core.models.mistral import MistralConfig

config = MistralConfig()
model = NeurosymbolicMistral(config)

result = model.generate_with_reasoning(
    "Prove that the square root of 2 is irrational"
)
print(result)
```

### 3. Research Agent

```python
from pinnacle_ai.agents.research_agent import ResearchAgent

agent = ResearchAgent(model)
hypothesis = agent.generate_hypothesis("neurosymbolic AI")
print(hypothesis)

agent.self_improve("AI improvements", num_iterations=2)
```

### 4. API Server

```bash
# Start server
uvicorn pinnacle_ai.api.neurosymbolic_api:app --reload

# Test
curl http://localhost:8000/health
```

## Expected Output

When proving √2 is irrational:

```
Proof that √2 is irrational:

1. Assume √2 is rational → √2 = a/b (reduced fraction, gcd(a,b) = 1)
2. Then 2 = a²/b² → a² = 2b²
3. Thus a² is even → a is even → a = 2k for some integer k
4. Substituting: (2k)² = 2b² → 4k² = 2b² → b² = 2k²
5. Thus b² is even → b is even
6. Contradiction: a and b share factor 2, but we assumed gcd(a,b) = 1
7. Therefore √2 is irrational. QED.
```

## Files Created

- ✅ `pinnacle_ai/core/neurosymbolic/logic_engine.py`
- ✅ `pinnacle_ai/core/neurosymbolic/neural_adapter.py`
- ✅ `pinnacle_ai/agents/research_agent.py`
- ✅ `pinnacle_ai/api/neurosymbolic_api.py`
- ✅ `tests/test_math.py`
- ✅ `examples/test_neurosymbolic.py`

## Status

✅ **All components implemented and ready to use!**

