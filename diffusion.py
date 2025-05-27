"""
Inspired and partly copied from https://github.com/google-research/google-research/blob/ad2d81983e4c717f477a232f625d0da2808b15aa/d3pm/text/diffusion.py
"""

import os
import torch
import abc
import fastNLP
import numpy as np
from typing import Any, List, Optional, Sequence, Union
import utils
from transformers import AutoTokenizer
from transformers.generation import LogitsProcessor
from dataclasses import dataclass
import losses
import time

class DiffusionSchedule(abc.ABC):
    """Base class for a diffusion schedule."""

    @abc.abstractmethod
    def get_qt(self, t: torch.Tensor) -> torch.Tensor:
        """Get transition matrices for time steps t."""
        pass

    @abc.abstractmethod
    def get_qt_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Get cumulative transition matrices for time steps t."""
        pass

    @abc.abstractmethod
    def get_qt_bar_log(self, t: torch.Tensor) -> torch.Tensor:
        """Get log cumulative transition matrices for time steps t."""
        pass

    @abc.abstractmethod
    def get_qt_log(self, t: torch.Tensor) -> torch.Tensor:
        """Get log transition matrices for time steps t."""
        pass

class UniformDiffusion(DiffusionSchedule):
    """Discrete diffusion with uniform transition probabilities."""

    def __init__(self, num_timesteps: int, vocab_size: int):
        """Initialize the diffusion schedule.

        Args:
            num_timesteps: The number of diffusion steps.
            vocab_size: The size of the vocabulary.
        """
        self.num_timesteps = num_timesteps
        self.vocab_size = vocab_size

        # Beta schedule
        betas = torch.linspace(0, 1, num_timesteps)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        # Store log versions
        self.log_alpha_bars = torch.log(alpha_bars)
        self.log_betas = torch.log(betas)
        self.log_alphas = torch.log(alphas)

    def get_qt(self, t: torch.Tensor) -> torch.Tensor:
        """Get transition matrices for time steps t."""
        log_qt = self.get_qt_log(t)
        return torch.exp(log_qt)

    def get_qt_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Get cumulative transition matrices for time steps t."""
        log_qt_bar = self.get_qt_bar_log(t)
        return torch.exp(log_qt_bar)

    def get_qt_bar_log(self, t: torch.Tensor) -> torch.Tensor:
        """Get log cumulative transition matrices for time steps t."""
        # Shape checking
        t = torch.as_tensor(t)
        if t.dim() == 0:
            t = t.unsqueeze(0)

        # Compute transition probabilities
        log_alpha_bars = self.log_alpha_bars[t]
        log_one_minus_alpha_bars = torch.log1p(-torch.exp(log_alpha_bars))

        # Expand dims for broadcasting
        log_alpha_bars = log_alpha_bars.unsqueeze(-1)
        log_one_minus_alpha_bars = log_one_minus_alpha_bars.unsqueeze(-1)

        # Compute matrix
        log_qt_bar = torch.zeros(
            t.shape[0], self.vocab_size, self.vocab_size, device=t.device
        )
        log_qt_bar = log_qt_bar + torch.eye(self.vocab_size, device=t.device) * log_alpha_bars
        log_qt_bar = log_qt_bar + (1 - torch.eye(self.vocab_size, device=t.device)) * (
            log_one_minus_alpha_bars - np.log(self.vocab_size - 1)
        )

        return log_qt_bar

    def get_qt_log(self, t: torch.Tensor) -> torch.Tensor:
        """Get log transition matrices for time steps t."""
        # Shape checking
        t = torch.as_tensor(t)
        if t.dim() == 0:
            t = t.unsqueeze(0)

        # Compute transition probabilities
        log_alphas = self.log_alphas[t]
        log_betas = self.log_betas[t]

        # Expand dims for broadcasting
        log_alphas = log_alphas.unsqueeze(-1)
        log_betas = log_betas.unsqueeze(-1)

        # Compute matrix
        log_qt = torch.zeros(
            t.shape[0], self.vocab_size, self.vocab_size, device=t.device
        )
        log_qt = log_qt + torch.eye(self.vocab_size, device=t.device) * log_alphas
        log_qt = log_qt + (1 - torch.eye(self.vocab_size, device=t.device)) * (
            log_betas - np.log(self.vocab_size - 1)
        )

        return log_qt

def create_discrete_diffusion_schedule(schedule_name: str, num_steps: int = 2048) -> DiffusionSchedule:
    """Create a diffusion schedule."""
    if schedule_name == "uniform":
        return UniformDiffusion(num_steps, vocab_size=30522)  # BERT vocab size
    else:
        raise ValueError(f"Unknown schedule: {schedule_name}")

class MaskDiffusion:
    """Diffusion model for masked language modeling."""

    def __init__(self, dim: int, schedule: DiffusionSchedule, tokenizer, sample_cls, word_freq_lambda: float = 0.3, device=None):
        """Initialize the diffusion model.

        Args:
            dim: The size of the vocabulary.
            schedule: The diffusion schedule.
            tokenizer: The tokenizer.
            sample_cls: The sampling class.
            word_freq_lambda: The weight for word frequency loss.
            device: The device to use.
        """
        self.dim = dim
        self.schedule = schedule
        self.num_steps = schedule.num_timesteps
        self.tokenizer = tokenizer
        self.sample_cls = sample_cls
        self.word_freq_lambda = word_freq_lambda
        self.device = device

    def sample_t(self, batch_size: int = None) -> torch.Tensor:
        """Sample timesteps."""
        if batch_size is None:
            return torch.randint(0, self.num_steps, ()).to(self.device)
        return torch.randint(0, self.num_steps, (batch_size,)).to(self.device)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample from q(x_t | x_0)."""
        qt_bar = self.schedule.get_qt_bar(t)
        qt_bar = qt_bar.to(x_0.device)

        # Sample from categorical distribution
        probs = qt_bar.gather(-1, x_0.unsqueeze(-1)).squeeze(-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def compute_loss(self, x_0: torch.Tensor, t: torch.Tensor, model_output: torch.Tensor, word_freq_logits: torch.Tensor = None) -> torch.Tensor:
        """Compute the loss for training."""
        # Get transition matrices
        qt_bar = self.schedule.get_qt_bar(t)
        qt_bar = qt_bar.to(x_0.device)

        # Compute cross entropy loss
        ce_loss = torch.nn.functional.cross_entropy(
            model_output.view(-1, self.dim),
            x_0.view(-1),
            reduction="none"
        )

        # Add word frequency loss if provided
        if word_freq_logits is not None and self.word_freq_lambda > 0:
            word_freq_loss = torch.nn.functional.mse_loss(
                torch.softmax(model_output, dim=-1),
                word_freq_logits.unsqueeze(-1),
                reduction="none"
            )
            loss = ce_loss + self.word_freq_lambda * word_freq_loss.mean(-1)
        else:
            loss = ce_loss

        return loss.mean()

def compute_kl_reverse_process(x_0: torch.Tensor, t: torch.Tensor, denoise_fn, diffusion, target_mask=None, hybrid_lambda=0.01, predict_x0=True, word_freq_logits=None):
    """Compute KL divergence for the reverse process.

    Args:
        x_0: Input tokens.
        t: Timesteps.
        denoise_fn: Function that predicts x_0 or noise.
        diffusion: The diffusion model.
        target_mask: Mask for valid tokens.
        hybrid_lambda: Weight for hybrid loss.
        predict_x0: Whether to predict x_0 directly.
        word_freq_logits: Word frequency logits for additional loss.

    Returns:
        Dictionary containing loss and metrics.
    """
    # Sample from q(x_t | x_0)
    x_t = diffusion.q_sample(x_0, t)

    # Get model predictions
    model_output = denoise_fn(x_t, t, target_mask)

    # Compute loss
    loss = diffusion.compute_loss(x_0, t, model_output, word_freq_logits)

    # Add hybrid loss if needed
    if hybrid_lambda > 0:
        hybrid_loss = torch.nn.functional.cross_entropy(
            model_output.view(-1, diffusion.dim),
            x_t.view(-1),
            reduction="mean"
        )
        loss = loss + hybrid_lambda * hybrid_loss

    return {"loss": loss}

def discrete_diffusion_elbo(x_0: torch.Tensor, denoise_fn, diffusion, target_mask=None, normalize_without_padding=True, eval_step_size=4, word_freq_logits=None, device=None):
    """Compute ELBO for discrete diffusion.

    Args:
        x_0: Input tokens.
        denoise_fn: Function that predicts x_0 or noise.
        diffusion: The diffusion model.
        target_mask: Mask for valid tokens.
        normalize_without_padding: Whether to normalize without padding.
        eval_step_size: Step size for evaluation.
        word_freq_logits: Word frequency logits for additional loss.
        device: The device to use.

    Returns:
        Dictionary containing ELBO and related metrics.
    """
    # Initialize metrics
    metrics = {
        'elbo': torch.tensor(0., device=device),
        'elbo_in_bits_per_dim': torch.tensor(0., device=device),
    }

    # Sample timesteps
    t = torch.arange(0, diffusion.num_steps, eval_step_size, device=device)
    if t[-1] != diffusion.num_steps - 1:
        t = torch.cat([t, torch.tensor([diffusion.num_steps - 1], device=device)])

    # Compute ELBO
    for i in range(len(t)):
        # Sample from q(x_t | x_0)
        x_t = diffusion.q_sample(x_0, t[i])

        # Get model predictions
        model_output = denoise_fn(x_t, t[i], target_mask)

        # Compute loss
        loss = diffusion.compute_loss(x_0, t[i], model_output, word_freq_logits)

        # Update metrics
        metrics['elbo'] += loss
        metrics['elbo_in_bits_per_dim'] += loss / np.log(2)

    # Normalize metrics
    if normalize_without_padding and target_mask is not None:
        num_tokens = target_mask.sum()
        metrics['elbo'] = metrics['elbo'] / num_tokens
        metrics['elbo_in_bits_per_dim'] = metrics['elbo_in_bits_per_dim'] / num_tokens
    else:
        metrics['elbo'] = metrics['elbo'] / x_0.numel()
        metrics['elbo_in_bits_per_dim'] = metrics['elbo_in_bits_per_dim'] / x_0.numel()

    return metrics 