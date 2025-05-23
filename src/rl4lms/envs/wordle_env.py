"""
Wordle environment for reinforcement learning.

This module implements the Wordle game environment that can be used for
reinforcement learning with language models.
"""

import random
from typing import List, Dict, Optional, Tuple, NamedTuple
from enum import Enum, auto


class LetterFeedback(str, Enum):
    """Feedback for each letter in a Wordle guess."""
    CORRECT = "✓"      # Correct letter in correct position
    WRONG_POS = "-"     # Correct letter in wrong position
    WRONG_LETTER = "x"  # Letter not in word


class GuessWithFeedback(NamedTuple):
    """Container for a Wordle guess and its feedback."""
    guess: str
    feedback: List[LetterFeedback]
    
    def __str__(self) -> str:
        """Return a string representation of the guess with feedback."""
        return f"{self.guess}\n{''.join(f.value for f in self.feedback)}"


class WordleEnv:
    """Wordle game environment for reinforcement learning."""
    
    def __init__(self, word_list: List[str], max_turns: int = 6):
        """Initialize the Wordle environment.
        
        Args:
            word_list: List of valid words that can be used as secret words
            max_turns: Maximum number of turns (guesses) allowed
        """
        self.word_list = [w.lower() for w in word_list]
        self.word_length = len(word_list[0]) if word_list else 5
        self.max_turns = max_turns
        self.reset()
    
    def reset(self, secret_word: Optional[str] = None) -> None:
        """Reset the environment with a new secret word.
        
        Args:
            secret_word: Optional secret word to use. If None, a random word is selected.
        """
        if secret_word is None:
            self.secret_word = random.choice(self.word_list)
        else:
            if len(secret_word) != self.word_length:
                raise ValueError(f"Word must be {self.word_length} letters long")
            self.secret_word = secret_word.lower()
        
        self.turn = 0
        self.guesses: List[GuessWithFeedback] = []
        self.game_over = False
        self.won = False
    
    def submit_guess(self, guess: str) -> GuessWithFeedback:
        """Submit a guess and get feedback.
        
        Args:
            guess: The guessed word
            
        Returns:
            GuessWithFeedback containing the guess and feedback
            
        Raises:
            ValueError: If the game is already over or the guess is invalid
        """
        if self.game_over:
            raise ValueError("Game is already over")
            
        guess = guess.lower().strip()
        if len(guess) != self.word_length:
            raise ValueError(f"Guess must be {self.word_length} letters long")
        
        # Generate feedback
        feedback = self._get_feedback(guess)
        guess_with_feedback = GuessWithFeedback(guess=guess, feedback=feedback)
        self.guesses.append(guess_with_feedback)
        
        # Check win condition
        if guess == self.secret_word:
            self.game_over = True
            self.won = True
        
        # Check lose condition
        self.turn += 1
        if self.turn >= self.max_turns and not self.won:
            self.game_over = True
        
        return guess_with_feedback
    
    def _get_feedback(self, guess: str) -> List[LetterFeedback]:
        """Generate feedback for a guess.
        
        Args:
            guess: The guessed word
            
        Returns:
            List of LetterFeedback enums for each letter position
        """
        feedback = []
        secret_letters = list(self.secret_word)
        
        # First pass: Check for correct letters in correct position
        for i, (g, s) in enumerate(zip(guess, self.secret_word)):
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
    
    def render(self) -> str:
        """Render the current game state as a string.
        
        Returns:
            String representation of the current game state
        """
        lines = []
        for guess in self.guesses:
            lines.append(str(guess))
        
        # Add empty lines for remaining turns
        for _ in range(self.max_turns - len(self.guesses)):
            lines.append("_" * self.word_length)
        
        # Add game status
        if self.game_over:
            if self.won:
                lines.append(f"\nYou won in {len(self.guesses)} turns!")
            else:
                lines.append(f"\nGame over! The word was: {self.secret_word}")
        
        return "\n".join(lines)
    
    def get_state(self) -> Dict:
        """Get the current game state.
        
        Returns:
            Dictionary containing the current game state
        """
        return {
            'secret_word': self.secret_word,
            'guesses': [g.guess for g in self.guesses],
            'feedbacks': [g.feedback for g in self.guesses],
            'turn': self.turn,
            'game_over': self.game_over,
            'won': self.won,
        }
