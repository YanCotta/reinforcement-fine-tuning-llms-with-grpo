import pytest
import torch
from rl4lms.reward_functions import RewardFunction


def test_base_reward_function():
    """Test that the base RewardFunction is abstract and can't be instantiated."""
    with pytest.raises(TypeError):
        RewardFunction()


class TestRewardFunction(RewardFunction):
    """Concrete implementation for testing."""
    def __call__(self, prompt_texts, generated_texts, meta_info=None):
        return torch.ones(len(generated_texts))


def test_concrete_reward_function():
    """Test that a concrete implementation works as expected."""
    reward_fn = TestRewardFunction()
    rewards = reward_fn("test prompt", "test generation")
    assert isinstance(rewards, torch.Tensor)
    assert rewards.shape == (1,)
    assert rewards[0] == 1.0


def test_reward_function_with_metadata():
    """Test that reward function works with metadata."""
    reward_fn = TestRewardFunction()
    meta_info = [{"key": "value"}]
    rewards = reward_fn("test", "test", meta_info)
    assert rewards[0] == 1.0
