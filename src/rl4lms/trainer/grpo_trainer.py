"""
GRPO (Generalized Reinforcement Policy Optimization) trainer for language models.

This module implements the training loop for fine-tuning language models using
the GRPO algorithm, which is a variant of PPO (Proximal Policy Optimization)
with additional features for language model fine-tuning.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
)

from ..losses.grpo_loss import grpo_loss, compute_log_probs
from ..reward_functions import RewardFunction

logger = logging.getLogger(__name__)


class GRPOTrainer:
    """Trainer for fine-tuning language models using GRPO."""
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        ref_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        reward_fn: RewardFunction,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        batch_size: int = 8,
        num_epochs: int = 3,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        epsilon: float = 0.2,
        beta: float = 0.1,
        gamma: float = 0.99,
        lam: float = 0.95,
        num_warmup_steps: int = 0,
        num_eval_steps: Optional[int] = None,
        output_dir: str = "./grpo_output",
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        """Initialize the GRPO trainer.
        
        Args:
            model: The language model to fine-tune
            ref_model: Reference model (usually the initial model before fine-tuning)
            tokenizer: Tokenizer for the model
            reward_fn: Reward function to evaluate generated text
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            learning_rate: Learning rate for the optimizer
            weight_decay: Weight decay for the optimizer
            max_grad_norm: Maximum gradient norm for gradient clipping
            epsilon: Clipping parameter for PPO
            beta: KL penalty coefficient
            gamma: Discount factor for rewards
            lam: GAE (Generalized Advantage Estimation) lambda parameter
            num_warmup_steps: Number of warmup steps for learning rate scheduling
            num_eval_steps: Number of steps between evaluations (default: eval every epoch)
            output_dir: Directory to save checkpoints and logs
            device: Device to run training on (default: cuda if available, else cpu)
        """
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.epsilon = epsilon
        self.beta = beta
        self.gamma = gamma
        self.lam = lam
        self.num_warmup_steps = num_warmup_steps
        self.num_eval_steps = num_eval_steps
        self.output_dir = output_dir
        
        # Set device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.ref_model = self.ref_model.to(self.device)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        # Initialize data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )
        
        if self.eval_dataset is not None:
            self.eval_loader = DataLoader(
                self.eval_dataset,
                batch_size=self.batch_size,
                collate_fn=self._collate_fn,
            )
        else:
            self.eval_loader = None
        
        # Calculate total training steps
        self.num_training_steps = len(self.train_loader) * self.num_epochs
        if self.num_eval_steps is None:
            self.num_eval_steps = len(self.train_loader)  # Eval once per epoch by default
        
        # Initialize learning rate scheduler
        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps,
        )
        
        # Training state
        self.global_step = 0
        self.best_eval_loss = float("inf")
    
    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate function for the data loader."""
        prompts = [item["prompt"] for item in batch]
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "prompts": prompts,
        }
    
    def train(self) -> None:
        """Run the training loop."""
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num Epochs = {self.num_epochs}")
        logger.info(f"  Batch size = {self.batch_size}")
        logger.info(f"  Total optimization steps = {self.num_training_steps}")
        
        # Training loop
        self.model.train()
        self.ref_model.eval()
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            epoch_reward = 0.0
            
            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                leave=False,
            )
            
            for step, batch in enumerate(progress_bar):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                prompts = batch["prompts"]
                
                # Generate completions
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=100,
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                
                # Decode completions
                completions = self.tokenizer.batch_decode(
                    outputs[:, input_ids.size(1):], skip_special_tokens=True
                )
                
                # Calculate rewards
                rewards = self.reward_fn(prompts, completions).to(self.device)
                
                # Calculate advantages using GAE
                with torch.no_grad():
                    _, log_probs = compute_log_probs(
                        self.model, input_ids, attention_mask
                    )
                    _, ref_log_probs = compute_log_probs(
                        self.ref_model, input_ids, attention_mask
                    )
                    
                    # Calculate advantages using GAE
                    # For simplicity, we use a simple advantage calculation here
                    # In practice, you might want to implement a more sophisticated advantage estimator
                    advantages = rewards - ref_log_probs.exp()
                
                # Calculate GRPO loss
                loss_dict = grpo_loss(
                    model=self.model,
                    ref_model=self.ref_model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    advantages=advantages,
                    epsilon=self.epsilon,
                    beta=self.beta,
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss_dict["loss"].backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                
                # Update weights
                self.optimizer.step()
                self.lr_scheduler.step()
                
                # Update progress bar
                epoch_loss += loss_dict["loss"].item()
                epoch_reward += rewards.mean().item()
                
                progress_bar.set_postfix(
                    loss=loss_dict["loss"].item(),
                    reward=rewards.mean().item(),
                    lr=self.optimizer.param_groups[0]["lr"],
                )
                
                # Evaluation
                self.global_step += 1
                if self.eval_loader is not None and self.global_step % self.num_eval_steps == 0:
                    eval_metrics = self.evaluate()
                    logger.info(f"Step {self.global_step}: {eval_metrics}")
                    
                    # Save best model
                    if eval_metrics["eval_loss"] < self.best_eval_loss:
                        self.best_eval_loss = eval_metrics["eval_loss"]
                        self.save_model(os.path.join(self.output_dir, "best_model"))
            
            # Log epoch metrics
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            avg_epoch_reward = epoch_reward / len(self.train_loader)
            logger.info(
                f"Epoch {epoch + 1}:"
                f" Loss: {avg_epoch_loss:.4f}"
                f" Reward: {avg_epoch_reward:.4f}"
            )
            
            # Save checkpoint
            self.save_model(os.path.join(self.output_dir, f"checkpoint-{epoch + 1}"))
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on the evaluation set."""
        if self.eval_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        total_reward = 0.0
        num_batches = 0
        
        for batch in tqdm(self.eval_loader, desc="Evaluating", leave=False):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            prompts = batch["prompts"]
            
            # Generate completions
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=100,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            
            # Decode completions
            completions = self.tokenizer.batch_decode(
                outputs[:, input_ids.size(1):], skip_special_tokens=True
            )
            
            # Calculate rewards
            rewards = self.reward_fn(prompts, completions).to(self.device)
            
            # Calculate loss
            loss_dict = grpo_loss(
                model=self.model,
                ref_model=self.ref_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                advantages=rewards,  # For evaluation, we just use rewards as advantages
                epsilon=self.epsilon,
                beta=self.beta,
            )
            
            total_loss += loss_dict["loss"].item()
            total_reward += rewards.mean().item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_reward = total_reward / num_batches
        
        self.model.train()
        
        return {
            "eval_loss": avg_loss,
            "eval_reward": avg_reward,
        }
    
    def save_model(self, output_dir: str) -> None:
        """Save the model and tokenizer."""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")
