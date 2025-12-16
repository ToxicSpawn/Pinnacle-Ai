# API Reference

## PinnacleAI Class

### `__init__(config_path: str = "config/settings.yaml")`

Initialize Pinnacle AI with configuration.

**Parameters:**
- `config_path`: Path to configuration file

### `execute_task(task: str, context: Optional[Dict] = None) -> Dict`

Execute a task using the meta-agent.

**Parameters:**
- `task`: Task description
- `context`: Optional context dictionary

**Returns:**
- Dictionary with execution results

### `interactive_mode()`

Run in interactive mode.

### `benchmark(tasks: List[Dict]) -> Dict`

Run a benchmark on a set of tasks.

**Parameters:**
- `tasks`: List of task dictionaries with "description" key

**Returns:**
- Dictionary with benchmark results

## OmniAIOrchestrator Class

### `__init__(config: Dict)`

Initialize the orchestrator.

**Parameters:**
- `config`: Configuration dictionary

### `improve_system() -> Dict[str, Any]`

Improve the entire AI system.

**Returns:**
- Dictionary with improvement results

## Agent Classes

All agents inherit from `BaseAgent` and implement:

### `execute(task: str, context: Dict = None) -> Dict`

Execute a task.

**Parameters:**
- `task`: Task description
- `context`: Context dictionary

**Returns:**
- Dictionary with execution results

### `improve() -> Dict`

Improve the agent.

**Returns:**
- Dictionary with improvement status

## Core Components

### LogicEngine

- `add_rule(rule: str)`
- `add_constraint(constraint: str)`
- `reason(premises: List[str]) -> List[str]`
- `verify(statement: str) -> bool`

### CausalGraph

- `add_node(concept: str, properties: Dict = None)`
- `add_edge(cause: str, effect: str, strength: float = 1.0)`
- `find_causes(effect: str) -> List[str]`
- `find_effects(cause: str) -> List[str]`

### UnifiedEncoder

- `encode(data: Any, modality: str) -> List[float]`
- `decode(embedding: List[float], target_modality: str) -> Any`
- `cross_modal_translate(source: Any, source_modality: str, target_modality: str) -> Any`

### EntangledMemory

- `store(key: str, value: Any, associations: List[str] = None)`
- `retrieve(key: str) -> Optional[Any]`
- `associate(key: str, related_keys: List[str])`
- `recall_by_association(key: str) -> List[Any]`

