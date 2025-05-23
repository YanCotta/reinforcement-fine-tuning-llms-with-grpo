#!/usr/bin/env python
# coding: utf-8

"""
Example script for fine-tuning a language model on the Wordle task using GRPO.

This script demonstrates how to use the GRPOTrainer to fine-tune a language model
to play Wordle using reinforcement learning.
"""

import os
import logging
import random
from typing import List, Dict, Any

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)

# Add parent directory to path to allow imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rl4lms.trainer import GRPOTrainer
from src.rl4lms.reward_functions.wordle import WordleRewardFunction
from src.rl4lms.envs.wordle_env import WordleEnv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_word_list(file_path: str) -> List[str]:
    """Load a list of words from a file.
    
    Args:
        file_path: Path to the file containing words (one per line)
        
    Returns:
        List of words
    """
    with open(file_path, 'r') as f:
        words = [line.strip().lower() for line in f if line.strip()]
    return words


def create_wordle_dataset(
    word_list: List[str],
    num_examples: int = 1000,
    max_length: int = 20,
) -> Dataset:
    """Create a dataset of Wordle game states.
    
    Args:
        word_list: List of valid words for the game
        num_examples: Number of examples to generate
        max_length: Maximum length of the prompt
        
    Returns:
        Dataset containing prompts and target words
    """
    data = []
    
    for _ in range(num_examples):
        # Select a random target word
        target_word = random.choice(word_list)
        
        # Create a prompt with the target word
        prompt = f"Guess the 5-letter word: "
        
        data.append({
            "prompt": prompt,
            "target_word": target_word,
        })
    
    return Dataset.from_list(data)


def main():
    # Set random seed for reproducibility
    seed = 42
    set_seed(seed)
    random.seed(seed)
    
    # Configuration
    model_name = "gpt2"  # Use a small model for demonstration
    output_dir = "./wordle_grpo_output"
    num_epochs = 3
    batch_size = 8
    learning_rate = 1e-5
    
    # Load word list (you'll need to provide your own word list file)
    word_list = ["apple", "table", "chair", "house", "mouse", "grape", "tiger", "zebra",
                 "light", "water", "earth", "heart", "smile", "happy", "cloud", "beach"]
    
    # Create datasets
    train_dataset = create_wordle_dataset(word_list, num_examples=1000)
    eval_dataset = create_wordle_dataset(word_list, num_examples=100)
    
    # Initialize models and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Initialize reward function
    reward_fn = WordleRewardFunction(partial_credit=True)
    
    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        output_dir=output_dir,
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Training complete!")
    logger.info(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
