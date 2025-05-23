from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch


class RewardFunction(ABC):
    """
    Abstract base class for reward functions used in RL fine-tuning of language models.
    
    All custom reward functions should inherit from this class and implement the __call__ method.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize the reward function.
        
        Args:
            name: Optional name for the reward function. If None, the class name will be used.
        """
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def __call__(
        self,
        prompt_texts: list[str],
        generated_texts: list[str],
        meta_info: Optional[list[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        """
        Compute rewards for the given generated texts.
        
        Args:
            prompt_texts: List of input prompt texts.
            generated_texts: List of generated texts to be evaluated.
            meta_info: Optional list of dictionaries containing additional information
                     that might be needed for reward computation.
                     
        Returns:
            A torch.Tensor containing the reward for each generated text.
        """
        pass
    
    def reset(self):
        """
        Reset any internal state of the reward function.
        
        This method is called at the beginning of each episode during training.
        """
        pass
