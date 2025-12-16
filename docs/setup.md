# Setup Instructions

## Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) Virtual environment

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ToxicSpawn/Pinnacle-AI.git
cd Pinnacle-AI
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development
```

### 4. Configuration

Copy the example configuration:

```bash
cp config/settings.yaml.example config/settings.yaml
```

Edit `config/settings.yaml` and add your API keys:
- LLM API keys (OpenAI, Anthropic, etc.)
- Web search API keys (if using)
- Image generation API keys (if using)
- Audio generation API keys (if using)

### 5. Verify Installation

```bash
python main.py --help
```

## Quick Start

### Interactive Mode

```bash
python main.py --interactive
```

### Single Task

```bash
python main.py "Your task here"
```

### Benchmark

```bash
python main.py --benchmark
```

## Troubleshooting

### Import Errors

Make sure you're in the project root directory and have installed all dependencies.

### Configuration Errors

Check that `config/settings.yaml` exists and has valid YAML syntax.

### API Key Errors

Ensure all required API keys are set in the configuration file.

