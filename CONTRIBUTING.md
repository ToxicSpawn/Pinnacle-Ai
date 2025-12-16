# Contributing to Pinnacle AI

Thank you for your interest in contributing to Pinnacle AI! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect different viewpoints and experiences

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in the issues
2. If not, create a new issue using the bug report template
3. Provide as much detail as possible:
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment information
   - Error messages or logs

### Suggesting Features

1. Check if the feature has already been suggested
2. Create a new issue using the feature request template
3. Clearly describe the feature and its use case
4. Explain why it would be valuable

### Submitting Code

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
   - Follow the code style guidelines
   - Write tests for new functionality
   - Update documentation as needed
4. **Commit your changes**
   ```bash
   git commit -m "Add: description of your changes"
   ```
5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```
6. **Open a Pull Request**
   - Provide a clear description
   - Reference any related issues
   - Ensure all tests pass

## Code Style

- Follow PEP 8 for Python code
- Use type hints where appropriate (e.g., `def func(x: int) -> str:`)
- Write docstrings for all functions and classes
- Keep functions focused and small
- Use meaningful variable and function names
- Format code with `black` before committing
- Run `ruff` for linting

### Pre-commit Setup

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install
```

## Commit Message Format

Use conventional commits:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

Example:
```bash
git commit -m "feat: add new planner agent capabilities"
git commit -m "fix: resolve memory leak in orchestrator"
git commit -m "docs: update API reference"
```

## Testing

- Write tests for all new functionality
- Ensure all existing tests pass
- Aim for good test coverage
- Run tests before submitting:
  ```bash
  pytest tests/
  ```

## Documentation

- Update relevant documentation when adding features
- Keep docstrings up to date
- Add examples for new features
- Update README if needed

## Development Setup

1. Clone your fork
2. Create a virtual environment
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```
4. Run tests to verify setup:
   ```bash
   pytest tests/
   ```

## Pull Request Process

1. Ensure your code follows the style guidelines
2. Update documentation as needed
3. Add tests for new functionality
4. Ensure all tests pass
5. Update CHANGELOG.md if applicable
6. Request review from maintainers

## Issue Labels

We use the following labels to categorize issues:

- `good first issue` → Beginner-friendly tasks
- `enhancement` → New features or improvements
- `bug` → Bug fixes
- `documentation` → Documentation improvements
- `question` → Questions or discussions
- `help wanted` → Extra attention needed

## Development Workflow

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Pinnacle-AI.git
   cd Pinnacle-AI
   ```

2. **Create Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**
   - Write code following style guidelines
   - Add tests for new functionality
   - Update documentation

4. **Test**
   ```bash
   pytest tests/
   ```

5. **Commit**
   ```bash
   git add .
   git commit -m "feat: your feature description"
   ```

6. **Push and PR**
   ```bash
   git push origin feature/your-feature-name
   # Then open a Pull Request on GitHub
   ```

## Code Review Process

1. All PRs require at least one review
2. Address review comments promptly
3. Keep PRs focused and reasonably sized
4. Update CHANGELOG.md for user-facing changes

## Questions?

Feel free to open an issue with the question template or contact the maintainers.

Thank you for contributing to Pinnacle AI! 🚀

