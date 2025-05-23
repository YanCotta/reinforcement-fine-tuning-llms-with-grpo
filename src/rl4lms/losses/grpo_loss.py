"""
Generalized Reinforcement Policy Optimization (GRPO) loss functions.

This module implements the GRPO loss function, which is a variant of PPO (Proximal Policy Optimization)
with additional features for language model fine-tuning.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


def compute_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute log probabilities of token sequences using the model.
    
    Args:
        model: The language model
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        
    Returns:
        Tuple of (logits, log_probs) where:
        - logits: Model outputs [batch_size, seq_len, vocab_size]
        - log_probs: Log probabilities of input tokens [batch_size, seq_len]
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
    
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    
    # Calculate log probs
    log_probs = F.log_softmax(shift_logits, dim=-1)
    log_probs = torch.gather(log_probs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
    
    return logits, log_probs


def grpo_loss(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    advantages: Optional[torch.Tensor] = None,
    epsilon: float = 0.2,
    beta: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """Compute the GRPO loss for language model fine-tuning.
    
    Args:
        model: The current policy model
        ref_model: The reference model (usually the initial model before fine-tuning)
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        advantages: Advantage estimates [batch_size]
        epsilon: Clipping parameter for policy ratio
        beta: KL penalty coefficient
        
    Returns:
        Dictionary containing loss components:
        - 'loss': Total loss
        - 'policy_loss': Policy gradient loss
        - 'value_loss': Value function loss (if applicable)
        - 'kl_penalty': KL divergence penalty
    """
    if advantages is None:
        advantages = torch.ones(input_ids.size(0), device=input_ids.device)
    
    # Get log probs from current and reference models
    _, log_probs = compute_log_probs(model, input_ids, attention_mask)
    with torch.no_grad():
        _, ref_log_probs = compute_log_probs(ref_model, input_ids, attention_mask)
    
    # Compute KL divergence
    kl_div = log_probs - ref_log_probs
    kl_penalty = beta * kl_div.mean()
    
    # Compute policy ratio
    ratio = torch.exp(log_probs - ref_log_probs)
    
    # Compute policy loss
    policy_loss1 = -advantages * ratio
    policy_loss2 = -advantages * torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    policy_loss = torch.max(policy_loss1, policy_loss2).mean()
    
    # Total loss
    total_loss = policy_loss + kl_penalty
    
    return {
        'loss': total_loss,
        'policy_loss': policy_loss,
        'kl_penalty': kl_penalty,
    }
