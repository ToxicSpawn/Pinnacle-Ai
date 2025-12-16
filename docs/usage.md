# Usage Guide

## Basic Usage

### Interactive Mode

Start Pinnacle AI in interactive mode:

```bash
python main.py --interactive
```

Commands:
- `exit` - Quit the interactive mode
- `improve` - Trigger self-improvement cycle
- `help` - Show help information
- Any other text - Execute as a task

### Single Task Mode

Execute a single task:

```bash
python main.py "Write a Python script for data analysis"
```

### Benchmark Mode

Run benchmark tests:

```bash
python main.py --benchmark
```

## Task Examples

### Programming Tasks

```bash
python main.py "Write a Python web application using FastAPI"
```

### Research Tasks

```bash
python main.py "Research the latest advancements in neurosymbolic AI"
```

### Creative Tasks

```bash
python main.py "Create a fantasy world with unique creatures and cultures"
```

### Scientific Tasks

```bash
python main.py "Design a scientific study to investigate social media effects"
```

### Philosophical Tasks

```bash
python main.py "Analyze the philosophical implications of artificial general intelligence"
```

## Configuration

### Custom Configuration

Use a custom configuration file:

```bash
python main.py --config config/custom.yaml "Your task"
```

### Debug Mode

Enable debug logging:

```bash
python main.py --debug "Your task"
```

## Advanced Usage

### Programmatic Usage

```python
from main import PinnacleAI

# Initialize
pinnacle = PinnacleAI("config/settings.yaml")

# Execute task
result = pinnacle.execute_task("Your task here")

# Run benchmark
tasks = [
    {"description": "Task 1"},
    {"description": "Task 2"}
]
results = pinnacle.benchmark(tasks)
```

## Best Practices

1. **Start Simple**: Begin with simple tasks to understand the system
2. **Use Interactive Mode**: Best for exploration and experimentation
3. **Provide Context**: Include relevant context in your tasks
4. **Monitor Performance**: Use benchmark mode to track improvements
5. **Regular Updates**: Keep dependencies and configuration up to date

