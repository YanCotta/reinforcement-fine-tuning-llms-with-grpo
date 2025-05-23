# RL4LMS: Reinforcement Learning for Language Model Supervision

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub stars](https://img.shields.io/github/stars/YanCotta/reinforcement-fine-tuning-llms-with-grpo?style=social)](https://github.com/YanCotta/reinforcement-fine-tuning-llms-with-grpo/stargazers)

RL4LMS is a powerful and flexible library designed for fine-tuning large language models (LLMs) using reinforcement learning, with a primary focus on the GRPO (Generalized Reinforcement Policy Optimization) algorithm. This library provides researchers and practitioners with a robust framework for implementing custom reward functions, environments, and training loops to optimize language models for specific tasks.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Custom Reward Functions](#custom-reward-functions)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

## Features

RL4LMS comes packed with powerful features designed to streamline the process of fine-tuning language models:

- **🔄 Flexible Reward Function API**: Intuitive interface for defining custom reward functions tailored to your specific task
- **🤗 HuggingFace Integration**: Seamless compatibility with all HuggingFace Transformers models
- **⚡ Efficient Training**: Optimized for both single and multi-GPU training with minimal setup
- **🧩 Extensible Architecture**: Modular design that makes it easy to add new components and environments
- **📊 Built-in Evaluation**: Comprehensive tools for monitoring and evaluating model performance
- **🎮 Wordle Environment**: Built-in Wordle game environment for RL training and experimentation

## Installation

RL4LMS can be installed with just a few simple steps:

1. **Clone the repository**
   ```bash
   git clone https://github.com/YanCotta/reinforcement-fine-tuning-llms-with-grpo.git
   cd reinforcement-fine-tuning-llms-with-grpo
   ```

2. **Set up a virtual environment** (recommended):
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install the package in development mode**
   ```bash
   pip install -e .
   ```

4. **Install additional dependencies**

   ```bash
   pip install -r requirements.txt
   ```

### Optional: Install with development dependencies

For contributing to the project or running tests:

```bash
pip install -e ".[dev]"
```

## Quick Start

### Fine-tuning on Wordle

RL4LMS includes a ready-to-use implementation for fine-tuning language models on the Wordle game. Here's how to get started:

1. **Prepare your environment** as described in the Installation section
2. **Run the example script**:

   ```bash
   python examples/wordle_finetuning.py
   ```

### Basic Usage Example

Here's a minimal example showing how to use RL4LMS to fine-tune a model:

```python
from rl4lms.trainer import GRPOTrainer
from rl4lms.reward_functions.wordle import WordleRewardFunction
from rl4lms.envs.wordle_env import WordleEnv

# Initialize components
model = AutoModelForCausalLM.from_pretrained("gpt2")
ref_model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
reward_fn = WordleRewardFunction()

# Create trainer and start training
trainer = GRPOTrainer(
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    reward_fn=reward_fn,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    batch_size=8,
    num_epochs=3,
    learning_rate=1e-5,
    output_dir="./wordle_grpo_output"
)

trainer.train()
```

## Project Structure

```text
rl4lms/
├── envs/                  # Environment implementations
│   ├── __init__.py
│   └── wordle_env.py      # Wordle game environment
├── losses/
│   ├── __init__.py
│   └── grpo_loss.py       # GRPO loss implementation
├── models/                # Model architectures
│   └── __init__.py
├── reward_functions/      # Reward function implementations
│   ├── __init__.py
│   ├── base.py           # Base reward function class
│   └── wordle.py         # Wordle-specific reward functions
├── trainer/
│   ├── __init__.py
│   └── grpo_trainer.py   # Training loop implementation
└── utils/                 # Utility functions
    └── __init__.py

examples/                # Example scripts
├── wordle_finetuning.py  # Wordle fine-tuning example

tests/                   # Unit tests
└── test_reward_functions.py
```

## Custom Reward Functions

To create a custom reward function, inherit from the `RewardFunction` base class and implement the `__call__` method:

```python
from rl4lms.reward_functions import RewardFunction
import torch

class MyRewardFunction(RewardFunction):
    def __init__(self, **kwargs):
        super().__init__()
        # Initialize any parameters
        
    def __call__(self, prompt_texts, generated_texts, **kwargs):
        """
        Calculate rewards for generated text.
        
        Args:
            prompt_texts: List of input prompts
            generated_texts: List of generated texts to score
            **kwargs: Additional metadata
            
        Returns:
            torch.Tensor: Tensor of rewards for each generated text
        """
        # Calculate rewards here
        rewards = torch.ones(len(generated_texts))  # Example: return 1 for each text
        return rewards
```

## Documentation

For detailed documentation, including API references, advanced usage examples, and tutorials, please visit our [documentation site](https://yancotta.github.io/reinforcement-fine-tuning-llms-with-grpo/).

## Contributing

We welcome contributions from the community! Whether you're fixing bugs, adding new features, or improving documentation, your help is greatly appreciated.

### How to Contribute

1. **Fork** the repository on GitHub
2. **Clone** your fork locally
3. Create a new **branch** for your changes
4. **Commit** your changes with clear, descriptive messages
5. **Push** your changes to your fork
6. Open a **Pull Request** with a clear description of your changes

### Development Setup

1. **Install development dependencies**:

   ```bash
   pip install -e ".[dev]"
   ```

2. **Run tests**:

   ```bash
   pytest tests/
   ```

3. **Format your code**:

   ```bash
   black .
   isort .
   ```

4. **Check for code style issues**:

   ```bash
   flake8 src tests
   mypy src
   ```

## Contact

For questions, suggestions, or support, please reach out:

- **Email**: [yanpcotta@gmail.com](mailto:yanpcotta@gmail.com)
- **GitHub**: [@YanCotta](https://github.com/YanCotta)
- **Issues**: [Open an issue](https://github.com/YanCotta/reinforcement-fine-tuning-llms-with-grpo/issues)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project was inspired by the course "Reinforcement Fine-Tuning LLMs With GRPO".
- Built with ❤️ using [PyTorch](https://pytorch.org/) and [HuggingFace Transformers](https://huggingface.co/transformers/).
