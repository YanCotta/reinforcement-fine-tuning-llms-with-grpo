# Contributing to RL4LMS

Thank you for your interest in contributing to RL4LMS! We welcome contributions from everyone, regardless of experience level. This document provides guidelines for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in the [Issues](https://github.com/yourusername/reinforcement-fine-tuning-llms-with-grpo/issues) section.
2. If not, create a new issue with a clear title and description.
3. Include steps to reproduce the bug, expected behavior, and actual behavior.
4. Add any relevant logs or screenshots.

### Suggesting Enhancements

1. Check if the enhancement has already been suggested.
2. Create a new issue describing the enhancement.
3. Explain why this enhancement would be useful.
4. Include any relevant examples or references.

### Making Code Contributions

1. Fork the repository.
2. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes and ensure tests pass.
4. Format your code using Black and ensure it passes all linters.
5. Commit your changes with a descriptive commit message.
6. Push your branch and create a pull request.

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/reinforcement-fine-tuning-llms-with-grpo.git
   cd reinforcement-fine-tuning-llms-with-grpo
   ```

2. Install the package in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide.
- Use [Black](https://github.com/psf/black) for code formatting.
- Use [isort](https://pycqa.github.io/isort/) for import sorting.
- Use [mypy](http://mypy-lang.org/) for static type checking.
- Keep lines under 88 characters.

## Testing

- Write tests for new features and bug fixes.
- Run tests using `pytest`.
- Ensure all tests pass before submitting a pull request.

## Documentation

- Update documentation when adding new features or changing existing ones.
- Follow [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

## Pull Request Process

1. Ensure tests pass and coverage remains high.
2. Update the README.md if needed.
3. Ensure your code is properly documented.
4. Request a review from one of the maintainers.
5. Once approved, your PR will be merged.

Thank you for contributing to RL4LMS!
