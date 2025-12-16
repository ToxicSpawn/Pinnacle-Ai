# Quick Start Guide - Pinnacle AI

Get up and running with Pinnacle AI in minutes!

## Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) Git for cloning

## Installation

### Option 1: Automated Setup (Recommended)

**Windows (PowerShell):**
```powershell
.\scripts\setup_environment.ps1
```

**Linux/Mac:**
```bash
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh
```

### Option 2: Manual Setup

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Optional, for development
   ```

3. **Configure:**
   ```bash
   cp config/settings.yaml.example config/settings.yaml
   # Edit config/settings.yaml with your API keys
   ```

## First Run

### Interactive Mode (Recommended for beginners)

```bash
python main.py --interactive
```

Then try:
```
Pinnacle AI> Write a Python function to calculate fibonacci numbers
Pinnacle AI> Research quantum computing
Pinnacle AI> Create a short story about space exploration
Pinnacle AI> exit
```

### Single Task Mode

```bash
python main.py "Write a Python script for data analysis"
```

### Benchmark Mode

```bash
python main.py --benchmark
```

Or use the benchmark script:
```bash
python scripts/benchmark.py
```

## Configuration

Edit `config/settings.yaml` to configure:

1. **LLM Provider** - Set your API keys:
   ```yaml
   llm:
     provider: "openai"
     api_key: "your-api-key-here"
   ```

2. **Agents** - Enable/disable agents:
   ```yaml
   agents:
     available_agents:
       - planner
       - researcher
       - coder
   ```

3. **Tools** - Configure tools:
   ```yaml
   tools:
     web_search:
       enabled: true
       api_key: "your-api-key"
   ```

## Docker Deployment

### Build and Run

```bash
docker-compose -f docker-compose.pinnacle.yml up --build
```

### Run in Background

```bash
docker-compose -f docker-compose.pinnacle.yml up -d
```

## Testing

Run all tests:
```bash
pytest tests/
```

Run specific test categories:
```bash
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
pytest tests/e2e/           # End-to-end tests
```

## Common Tasks

### Research Task
```bash
python main.py "Research the latest AI developments"
```

### Coding Task
```bash
python main.py "Write a REST API in Python using FastAPI"
```

### Creative Task
```bash
python main.py "Write a poem about artificial intelligence"
```

### Planning Task
```bash
python main.py "Plan a software project with multiple phases"
```

## Troubleshooting

### Import Errors
- Make sure you're in the project root directory
- Activate the virtual environment
- Install dependencies: `pip install -r requirements.txt`

### Configuration Errors
- Check that `config/settings.yaml` exists
- Verify YAML syntax is correct
- Ensure API keys are set (if using external services)

### Module Not Found
- Verify you're using Python 3.9+
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

## Next Steps

1. Read the [Architecture Documentation](docs/architecture.md)
2. Explore [Example Tasks](docs/examples.md)
3. Check the [API Reference](docs/api_reference.md)
4. Review [Agent Documentation](docs/agents.md)

## Getting Help

- Check the [Documentation](docs/)
- Open an [Issue](https://github.com/ToxicSpawn/Pinnacle-AI/issues)
- Review [Examples](docs/examples.md)

Happy exploring! 🚀

