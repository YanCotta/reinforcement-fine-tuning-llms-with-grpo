"""
Wordle-specific reward functions for reinforcement learning.

This module contains reward functions specifically designed for the Wordle game,
including format checking, semantic evaluation, and other custom reward signals.
"""

from typing import List, Dict, Any, Optional
import torch
import numpy as np
from enum import Enum

from .base import RewardFunction


class LetterFeedback(str, Enum):
    """Feedback for each letter in a Wordle guess."""
    CORRECT = "✓"      # Correct letter in correct position
    WRONG_POS = "-"     # Correct letter in wrong position
    WRONG_LETTER = "x"  # Letter not in word


class GuessWithFeedback:
    """Container for a Wordle guess and its feedback."""
    
    def __init__(self, guess: str, feedback: List[LetterFeedback]):
        """Initialize with guess and feedback.
        
        Args:
            guess: The guessed word
            feedback: List of LetterFeedback enums for each letter position
        """
        self.guess = guess
        self.feedback = feedback
    
    def __repr__(self) -> str:
        """Return a string representation of the guess with feedback."""
        return f"{self.guess}\n{''.join(f.value for f in self.feedback)}"


def get_feedback(guess: str, secret_word: str) -> List[LetterFeedback]:
    """Generate feedback for a Wordle guess.
    
    Args:
        guess: The guessed word
        secret_word: The target word to guess
        
    Returns:
        List of LetterFeedback enums for each letter position
    """
    feedback = []
    secret_letters = list(secret_word)
    
    # First pass: Check for correct letters in correct position
    for i, (g, s) in enumerate(zip(guess, secret_word)):
        if g == s:
            feedback.append(LetterFeedback.CORRECT)
            secret_letters.remove(g)  # Remove matched letters from pool
        else:
            feedback.append(None)  # Placeholder for second pass
    
    # Second pass: Check for correct letters in wrong position
    for i, (g, fb) in enumerate(zip(guess, feedback)):
        if fb is None:  # Only check positions not already marked as correct
            if g in secret_letters:
                feedback[i] = LetterFeedback.WRONG_POS
                secret_letters.remove(g)  # Remove matched letters from pool
            else:
                feedback[i] = LetterFeedback.WRONG_LETTER
    
    return feedback


def wordle_reward(guess: str, secret_word: str) -> float:
    """Calculate a simple reward for a Wordle guess.
    
    Args:
        guess: The guessed word
        secret_word: The target word
        
    Returns:
        Reward value (1.0 for correct guess, 0.0 otherwise)
    """
    return 1.0 if guess == secret_word else 0.0


def wordle_reward_partial_credit(guess: str, secret_word: str) -> float:
    """Calculate a reward with partial credit for Wordle.
    
    Gives partial credit based on:
    - Number of correct letters in correct position (2 points each)
    - Number of correct letters in wrong position (1 point each)
    
    Args:
        guess: The guessed word
        secret_word: The target word
        
    Returns:
        Reward value between 0.0 and 1.0
    """
    if guess == secret_word:
        return 1.0
    
    feedback = get_feedback(guess, secret_word)
    score = 0.0
    
    for fb in feedback:
        if fb == LetterFeedback.CORRECT:
            score += 2
        elif fb == LetterFeedback.WRONG_POS:
            score += 1
    
    # Normalize to [0, 1] range (max possible is 10 for 5 letters * 2 points)
    return min(score / 10.0, 1.0)


class WordleRewardFunction(RewardFunction):
    """Reward function for Wordle game."""
    
    def __init__(self, secret_word: str, partial_credit: bool = True):
        """Initialize with target word and reward type.
        
        Args:
            secret_word: The target word to guess
            partial_credit: Whether to use partial credit scoring (default: True)
        """
        self.secret_word = secret_word.lower()
        self.partial_credit = partial_credit
    
    def __call__(
        self, 
        prompt_texts: List[str], 
        generated_texts: List[str],
        meta_info: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        """Calculate rewards for generated Wordle guesses.
        
        Args:
            prompt_texts: List of prompt texts (unused)
            generated_texts: List of generated guesses
            meta_info: Optional list of metadata (unused)
            
        Returns:
            Tensor of reward values
        """
        rewards = []
        for guess in generated_texts:
            guess = guess.lower().strip()
            if self.partial_credit:
                reward = wordle_reward_partial_credit(guess, self.secret_word)
            else:
                reward = wordle_reward(guess, self.secret_word)
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32)
