# Inspired and partly copied from https://github.com/google-research/google-research/blob/ad2d81983e4c717f477a232f625d0da2808e15aa/d3pm/text/diffusion.py

import os
import torch
import abc
import numpy as np
from typing import Any, List, Optional, Sequence, Union
#import utils.utils as utils
from transformers import AutoTokenizer
from dataclasses import dataclass
import torch.nn.functional as F
import math
import losses
import utils


import torch
import torch.nn.functional as F

def top_k_top_p_filtering(
    logits, top_k=100, top_p=0.95, filter_value=-float("Inf"), min_tokens_to_keep=5, vocab_size=30522
):
    """Custom top-k and top-p filtering to avoid transformers issues."""
    # Validate shape
    if logits.shape[-1] != vocab_size:
        print(f"top_k_top_p_filtering: WARNING: logits.shape={logits.shape}, expected last dim={vocab_size}")
    
    # Check for NaN or Inf values in logits
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        print("top_k_top_p_filtering: WARNING: NaN or inf detected, using uniform distribution")
        logits = torch.zeros_like(logits)
    
    # Clamp logits to prevent extreme values
    logits = logits.clamp(-1e4, 1e4)
    
    if top_k > 0:
        # Ensure top_k is within valid range
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        # Find the kth largest logit value
        threshold = torch.topk(logits, top_k)[0][..., -1, None]
        # Mask logits below threshold
        indices_to_remove = logits < threshold
        logits = logits.masked_fill(indices_to_remove, filter_value)
    
    if top_p < 1.0:
        # Sort logits descending and compute softmax cumulative probs
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        
        # Debug log for sorted indices range
        print(f"top_k_top_p_filtering: sorted_indices min={sorted_indices.min().item()}, max={sorted_indices.max().item()}")
        
        # Check indices validity
        if sorted_indices.min().item() < 0 or sorted_indices.max().item() >= vocab_size:
            print("top_k_top_p_filtering: WARNING: Invalid sorted_indices, using uniform distribution")
            return torch.zeros_like(logits)
        
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Create mask for tokens to remove (cumulative prob > top_p)
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Always keep at least min_tokens_to_keep tokens
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = False
        
        # Scatter the mask back to original logits order
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)
        
        logits = logits.masked_fill(indices_to_remove, filter_value)
    
    # Check if all logits are filtered (all -inf)
    if torch.isinf(logits).all(dim=-1).any():
        print(f"top_k_top_p_filtering: WARNING: All logits filtered out, using uniform distribution")
        logits = torch.zeros_like(logits)
    
    return logits


class DiffusionSchedule:
    def __init__(self, schedule_fn, num_steps, is_constant=False, device=None):
        self._schedule_fn = schedule_fn
        self.num_steps = num_steps
        self.is_constant = is_constant
        self.device = device

    def __call__(self, step):
        if isinstance(step, torch.Tensor) and self.device is not None:
            step = step.to(self.device)
        result = self._schedule_fn(step)
        if isinstance(result, torch.Tensor) and self.device is not None:
            result = result.to(self.device)
        return result

    def __repr__(self):
        return f"DiffusionSchedule(steps={self.num_steps}, is_constant={self.is_constant}, device={self.device})"

class DiscreteDiffusionBase(abc.ABC):
    num_steps: int
    dim: int
    tokenizer: Any
    device: Any

    @abc.abstractmethod
    def stationary_probs(self, shape):
        pass

    @abc.abstractmethod
    def sample_stationary(self, shape):
        pass

    def sample_t(self, size=(1,)):
        if self.device is None:
            raise ValueError("Device not set in DiscreteDiffusionBase")
        return torch.randint(low=0, high=self.num_steps, size=size, device=self.device)

    def supports_efficient_get(self):
        return False

    def supports_efficient_inference(self):
        return False

    @abc.abstractmethod
    def get_qt_given_q0(self,
                        q0,
                        t,
                        return_logits=False,
                        make_one_hot=False,
                        word_freq_logits=None,
                        epsilon=1e-20):
        pass

    @abc.abstractmethod
    def sample_and_compute_posterior_q(
        self,
        x_0,
        t,
        samples=None,
        transition_probs=None,
        return_logits=True,
        return_transition_probs=True,
        transition_probs_in_logits=True,
        make_one_hot=True,
        epsilon=1e-20,
        step_size=1,
        word_freq_logits=None
    ):
        pass

class DiscreteDiffusionMatrixBase(DiscreteDiffusionBase):
    def get(self, t):
        raise NotImplementedError

    def custom_product_fn(self, t):
        raise NotImplementedError

    def qt_reverse(self,
                   qt_plus_1,
                   t,
                   return_logits=False,
                   make_one_hot=False,
                   epsilon=1e-20):
        raise NotImplementedError

    def get_qt_matrix(self, t):
        if self.supports_efficient_inference():
            return self.custom_product_fn(t)

        print("WARNING: Using inefficient matrix product.")

        def product_fn(i, state):
            return torch.matmul(self.get(i), state)

        final_product = utils.fori_loop(0, t, product_fn, torch.eye(self.dim, device=self.device))
        return final_product

    def get_qt_given_q0(self,
                        q0,
                            t,
                            return_logits=False,
                            make_one_hot=False,
                            word_freq_logits=None,
                            epsilon=1e-20):
        if self.device is None:
            raise ValueError("Device not set in DiscreteDiffusionMatrixBase")
        if make_one_hot:
            q0 = torch.nn.functional.one_hot(q0, num_classes=self.dim).to(self.device)
        if self.supports_efficient_inference():
            prob_at_time_t = torch.einsum("ij,...j->...i", self.get_qt_matrix(t), q0).to(self.device)
            if return_logits:
                return torch.log(prob_at_time_t + epsilon)
            return prob_at_time_t
        raise NotImplementedError

    def sample_and_compute_posterior_q(
            self,
            x_0,
            t,
            samples=None,
            transition_probs=None,
            return_logits=True,
            return_transition_probs=False,
            transition_probs_in_logits=True,
            make_one_hot=True,
            epsilon=1e-20,
            step_size=1,
            word_freq_logits=None):
        if self.device is None:
            raise ValueError("Device not set in DiscreteDiffusionMatrixBase")
        x_0 = x_0.to(self.device)
        if make_one_hot:
            x_0 = torch.nn.functional.one_hot(x_0, self.dim).reshape(x_0.shape + (self.dim,)).to(self.device)

        prob_at_time_t = self.get_qt_given_q0(q0=x_0, t=t, word_freq_logits=word_freq_logits).to(self.device)
        if self.supports_efficient_get():
            if step_size > 1:
                transition_matrix = torch.eye(self.dim, device=self.device)
                for i in range(step_size):
                    transition_matrix = self.get(t + i).to(self.device) @ transition_matrix
            else:
                transition_matrix = self.get(t).to(self.device)
            prob_at_time_t_plus_one = torch.einsum("ij,...j->...i", transition_matrix, prob_at_time_t).to(self.device)
        else:
            prob_at_time_t_plus_one = self.get_qt_given_q0(q0=x_0, t=t + step_size, word_freq_logits=word_freq_logits).to(self.device)

        if samples is None and transition_probs is not None:
            raise ValueError("samples were not provided but transition_probs were.")

        if samples is None:
            logits = torch.log(prob_at_time_t_plus_one + epsilon)
            samples = self.sample_cls.sample(logits, x_0).to(self.device)

        if transition_probs is None:
            if self.supports_efficient_get():
                transition_probs = transition_matrix[samples].to(self.device)
            else:
                if step_size > 1:
                    transition_probs = torch.nn.functional.one_hot(samples, self.dim).to(self.device)
                    for i in range(step_size):
                        transition_probs = self.qt_reverse(qt_plus_1=transition_probs, make_one_hot=False, t=t + step_size - 1 - i).to(self.device)
                else:
                    transition_probs = self.qt_reverse(qt_plus_1=samples, make_one_hot=True, t=t).to(self.device)

        if not transition_probs_in_logits and not return_logits:
            raise ValueError("Cannot exclude transition probs from logits if return_logits is false.")

        if return_logits:
            posterior_logits = torch.log(prob_at_time_t + epsilon).to(self.device)
            if transition_probs_in_logits:
                posterior_logits = posterior_logits + torch.log(transition_probs + epsilon)
            if return_transition_probs:
                return posterior_logits, samples, transition_probs
            return posterior_logits, samples
        posterior = transition_probs * prob_at_time_t
        denominator = posterior.sum(dim=-1, keepdims=True)
        posterior = posterior / denominator
        if return_transition_probs:
            return posterior, samples, transition_probs
        return posterior, samples

class MaskDiffusion(DiscreteDiffusionMatrixBase):
    def __init__(self,
                 dim,
                 schedule,
                 tokenizer,
                 use_fast_inference=True,
                 sample_cls=None,
                 word_freq=None,
                 word_freq_lambda=0.1,  # Reduced for stability
                 device=None):
        self.num_steps = schedule.num_steps
        self.sample_cls = sample_cls
        self.schedule = schedule
        self.use_fast_inference = use_fast_inference
        self.dim = dim
        self.tokenizer = tokenizer
        self.device = device
        if self.device is None:
            raise ValueError("Device must be specified for MaskDiffusion")
        self.mask = torch.nn.functional.one_hot(
            torch.tensor(self.tokenizer.mask_token_id, device=self.device), num_classes=self.dim
        ).unsqueeze(1).repeat(1, self.dim).float()
        self.state = self._create_state()
        self.word_freq = word_freq.to(self.device) if word_freq is not None else None
        self.word_freq_lambda = word_freq_lambda * torch.sin(
            torch.arange(schedule.num_steps + 1, device=self.device) / schedule.num_steps * math.pi
        )

    def _create_state(self):
        betas = torch.cat(
            (torch.tensor([0.0], device=self.device), self.schedule(torch.arange(self.num_steps, device=self.device)))
        ).double()
        alphas = 1 - betas
        state = torch.cumprod(alphas, dim=0)
        state[-1] = 0.0
        return state.to(torch.float32)

    def noise_fn(self, q0, t, word_freq_logits):
        print(f"noise_fn: q0.shape = {q0.shape}")
        if q0.shape[-1] != self.dim:
            raise ValueError(f"Expected q0 last dim to be {self.dim} but got {q0.shape[-1]}")
        
        # Get state probability for time t
        p = self.state[t]
        
        # Initialize word frequency influence
        if word_freq_logits is None and self.word_freq is not None:
            # Expand word_freq to match batch dimensions
            word_freq = self.word_freq.to(q0.device)
            word_freq = word_freq.unsqueeze(0).expand(q0.size(0), -1)  # [B, V]
            
            # Get the indices of the most likely tokens
            token_indices = q0.argmax(dim=-1)  # [B, L]
            
            # Gather word frequencies for the selected tokens
            word_freq_for_tokens = word_freq.gather(1, token_indices)  # [B, L]
            word_freq_for_tokens = word_freq_for_tokens - word_freq_for_tokens.mean(dim=1, keepdim=True)
            
            # Scale the word frequency influence
            word_freq_influence = self.word_freq_lambda[t] * word_freq_for_tokens.unsqueeze(-1)  # [B, L, 1]
        else:
            word_freq_influence = 0.0

        # Apply noise transformation
        non_mask_prob = p * q0
        
        # Add word frequency influence if available
        if isinstance(word_freq_influence, torch.Tensor):
            non_mask_prob = torch.clamp(non_mask_prob + word_freq_influence, 0., 0.999)
        
        # Calculate mask probability
        mask_prob = 1 - non_mask_prob.sum(-1, keepdim=True) + non_mask_prob[..., self.tokenizer.mask_token_id].unsqueeze(-1)
        
        # Concatenate probabilities
        prob_at_time_t = torch.cat(
            (
                non_mask_prob[..., :self.tokenizer.mask_token_id],
                mask_prob,
                non_mask_prob[..., self.tokenizer.mask_token_id + 1:]
            ),
            dim=-1
        )
        
        print(f"noise_fn: t={t}, prob_at_time_t min={prob_at_time_t.min().item()}, max={prob_at_time_t.max().item()}")
        return prob_at_time_t

    def supports_efficient_inference(self):
        return self.use_fast_inference

    def stationary_probs(self, size):
        stationary = torch.zeros(size=size + (self.dim,), device=self.device)
        stationary[..., self.tokenizer.mask_token_id] = 1
        return stationary

    def sample_stationary(self, size):
        return torch.full(size=size, fill_value=self.tokenizer.mask_token_id, device=self.device)

    def custom_product_fn(self, t):
        dim = self.dim
        if self.schedule.is_constant:
            beta = self.schedule(0)
            one_minus_beta_t_sq = (1 - beta) ** t
            return one_minus_beta_t_sq * torch.eye(dim, device=self.device) + (1 - one_minus_beta_t_sq) * self._get_mask()
        p = self.state[t]
        return p * torch.eye(dim, device=self.device) + (1 - p) * self._get_mask()

    def _get_mask(self):
        return self.mask

    def get(self, t):
        beta = self.schedule(t)
        return (1 - beta) * torch.eye(self.dim, device=self.device) + beta * self._get_mask()

    def qt_reverse(self,
                   qt_plus_1,
                   t,
                   return_logits=False,
                   make_one_hot=False,
                   epsilon=1e-20):
        if make_one_hot:
            assert qt_plus_1.dtype == torch.int64
            qt_plus_1 = torch.nn.functional.one_hot(qt_plus_1, num_classes=self.dim).to(self.device)
        beta = self.schedule(t)
        qtpls1_at_mask = qt_plus_1[..., self.tokenizer.mask_token_id: self.tokenizer.mask_token_id + 1]
        non_mask_prob0 = (1 - beta) * qt_plus_1[..., :self.tokenizer.mask_token_id] + beta * qtpls1_at_mask
        non_mask_prob1 = (1 - beta) * qt_plus_1[..., self.tokenizer.mask_token_id + 1:] + beta * qtpls1_at_mask
        prob_at_time_t = torch.cat((non_mask_prob0, qtpls1_at_mask, non_mask_prob1), dim=-1)
        if return_logits:
            return torch.log(prob_at_time_t + epsilon)
        return prob_at_time_t

    def get_qt_given_q0(self,
                        q0,
                        t,
                        return_logits=False,
                        make_one_hot=False,
                        epsilon=1e-20,
                        word_freq_logits=None):
        if not self.supports_efficient_inference():
            return super().get_qt_given_q0(q0, t, return_logits=return_logits, epsilon=epsilon)
        if make_one_hot:
            assert q0.dtype == torch.int64
            q0 = torch.nn.functional.one_hot(q0, num_classes=self.dim).to(self.device)
        prob_at_time_t = q0 if t == 0 else self.noise_fn(q0, t, word_freq_logits)
        if return_logits:
            return torch.log(prob_at_time_t + epsilon)
        return prob_at_time_t

    def supports_efficient_get(self):
        return not self.use_fast_inference

def create_discrete_diffusion_schedule(
        kind="mutual",
        beta_min=1e-3,
        beta_max=1e-1,
        num_steps=512,
        scale=1.0,
        s=0.008,
        device=None):
    assert beta_min <= beta_max
    assert num_steps > 0
    assert scale >= 1

    if kind == "mutual":
        print(f"using standard schedule with num_steps: {num_steps}.")

        def schedule_fn(step):
            if isinstance(step, torch.Tensor):
                step = step.to(device)
            return torch.tensor(1 / (num_steps - step), device=device, dtype=torch.float32)

        return DiffusionSchedule(schedule_fn, num_steps, is_constant=False, device=device)
    elif kind == "linear":
        print(f"using provided beta_min {beta_min} and beta_max {beta_max}.")
        is_constant = beta_min == beta_max
        schedule_fn = utils.create_learning_rate_scheduler(
            "constant * linear_warmup_from",
            warmup_steps=num_steps,
            min_learning_rate=beta_min,
            base_learning_rate=beta_max
        )
        return DiffusionSchedule(schedule_fn, num_steps, is_constant=is_constant, device=device)
    elif kind == "cosine":
        print("using cosine schedule inspired by OpenAI I-DDPM paper.")

        def cosine_fn(step):
            if isinstance(step, torch.Tensor):
                step = step.to(device)
            return torch.cos((step / num_steps + s) / (1 + s) * np.pi / 2)

        def schedule_fn(step):
            return torch.clip(1 - (cosine_fn(step + 1) / cosine_fn(step)), 0, 0.999)

        return DiffusionSchedule(schedule_fn, num_steps, is_constant=False, device=device)
    else:
        raise ValueError(f"kind {kind} is not supported.")

def p_forward(
        denoise_fn,
        target_mask,
        x_t,
        t,
        diffusion,
        predict_x0=True,
        return_x0=False,
        return_logits=False,
        special_case_x0=False,
        transition_probs=None,
        transition_probs_in_logits=True,
        maximum_likelihood=False,
        epsilon=1e-20,
        step_size=1,
        word_freq_logits=None):
    x_t = x_t.to(diffusion.device)
    target_mask = target_mask.to(diffusion.device)
    if isinstance(t, torch.Tensor):
        t = t.to(diffusion.device)
    logits = denoise_fn(targets=x_t, timestep=t, attention_mask=target_mask)
    # Debug: Log logits stats
    print(f"p_forward: t={t}, logits shape={logits.shape}, min={logits.min().item()}, max={logits.max().item()}")
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        print(f"p_forward: t={t}, WARNING: NaN or inf in logits, clamping")
        logits = torch.where(torch.isnan(logits) | torch.isinf(logits), torch.zeros_like(logits), logits)
    probs = torch.nn.Softmax(dim=-1)(logits)
    if not predict_x0:
        retval = logits if return_logits else probs
        if return_x0:
            return retval, None
        return retval
    if maximum_likelihood:
        probs = probs.argmax(-1)
    qt_probs, _ = diffusion.sample_and_compute_posterior_q(
        x_0=probs,
        t=t - step_size,
        return_logits=return_logits,
        make_one_hot=maximum_likelihood,
        transition_probs_in_logits=transition_probs_in_logits,
        transition_probs=transition_probs,
        samples=x_t,
        epsilon=epsilon,
        step_size=step_size,
        word_freq_logits=word_freq_logits
    )
    retval_x0 = logits if return_logits else probs
    retval = qt_probs
    mask = ((t == step_size) & special_case_x0).long().to(diffusion.device)
    retval = mask * retval_x0 + (1 - mask) * retval
    if return_x0:
        return retval, retval_x0
    return retval

def compute_prior_kl(x_start, diffusion, target_mask=None, word_freq_logits=None):
    x_start = x_start.to(diffusion.device)
    if target_mask is not None:
        target_mask = target_mask.to(diffusion.device)
    num_steps = diffusion.num_steps
    q_probs = diffusion.get_qt_given_q0(q0=x_start, t=num_steps, return_logits=False, make_one_hot=True, word_freq_logits=word_freq_logits)
    p_probs = diffusion.stationary_probs(q_probs.shape[:-1])
    loss = losses.kl_divergence_with_probs(q_probs, p_probs)
    if target_mask is not None:
        loss = (loss * target_mask).sum()
    else:
        loss = loss.sum()
    return loss, 1

def compute_kl_reverse_process(x_start,
                               t,
                               *,
                               diffusion,
                               denoise_fn,
                               predict_x0=True,
                               log_space=False,
                               hybrid_lambda=0.0,
                               use_cached_transition=True,
                               target_mask=None,
                               word_freq_logits=None,
                               step_size=1,
                               device=None):
    x_start = x_start.to(diffusion.device)
    if isinstance(t, torch.Tensor):
        t = t.to(diffusion.device)
    if target_mask is not None:
        target_mask = target_mask.to(diffusion.device)
    if step_size > 1 and not predict_x0:
        raise ValueError("cannot skip steps when not predicting x0.")
    q_t, x_t_plus_1, transition_probs = diffusion.sample_and_compute_posterior_q(
        x_start,
        t,
        return_logits=log_space,
        return_transition_probs=True,
        step_size=step_size,
        word_freq_logits=word_freq_logits
    )
    transition_probs = transition_probs if use_cached_transition else None
    p_t = p_forward(
        denoise_fn,
        target_mask,
        x_t_plus_1,
        t + step_size,
        diffusion,
        predict_x0=predict_x0,
        return_x0=predict_x0 and hybrid_lambda > 0.0,
        return_logits=log_space,
        transition_probs=transition_probs,
        step_size=step_size,
        word_freq_logits=word_freq_logits
    )
    if predict_x0 and hybrid_lambda > 0.0:
        p_t, p_0 = p_t
        if log_space:
            cross_entropy = losses.cross_entropy_with_logits(logits=p_0, targets=x_start)
        else:
            cross_entropy = losses.cross_entropy_with_probs(probs=p_0, targets=x_start)
        hybrid_loss = hybrid_lambda * cross_entropy
    else:
        hybrid_loss = torch.tensor([0.], device=diffusion.device)
    if log_space:
        kl = losses.kl_divergence_with_logits(q_t, p_t)
        cross_entropy = losses.cross_entropy_with_logits(logits=p_t, targets=x_start)
    else:
        kl = losses.kl_divergence_with_probs(q_t, p_t)
        cross_entropy = losses.cross_entropy_with_probs(probs=p_t, targets=x_start)
    if target_mask is not None:
        kl = (kl * target_mask).sum()
        cross_entropy = (cross_entropy * target_mask).sum()
        hybrid_loss = (hybrid_loss * target_mask).sum()
    else:
        kl = kl.sum()
        cross_entropy = cross_entropy.sum()
        hybrid_loss = hybrid_loss.sum()
    mask = (t == 0).long().to(diffusion.device)
    base_loss = mask * cross_entropy + (1 - mask) * kl
    loss = base_loss + hybrid_loss
    denominator = 1
    metrics_dict = {
        "loss": loss,
        "denominator": denominator,
        "hybrid_loss": hybrid_loss,
        "base_loss": base_loss,
        "cross_entropy_loss": cross_entropy,
        "t0_loss": mask * cross_entropy,
        "kl_loss": kl,
    }
    return metrics_dict

def discrete_diffusion_elbo(
        x_start,
        *,
        denoise_fn,
        diffusion,
        target_mask,
        word_freq_logits,
        predict_x0=True,
        length_probs=None,
        normalize_without_padding=True,
        eval_step_size=1,
        device=None):
    x_start = x_start.to(diffusion.device)
    if target_mask is not None:
        target_mask = target_mask.to(diffusion.device)
    assert diffusion.num_steps % eval_step_size == 0
    assert diffusion.num_steps > eval_step_size

    @dataclass
    class State:
        t: Any
        log_likelihood: Any

    def elbo_body_fn(state, _):
        metrics_dict = compute_kl_reverse_process(
            x_start,
            state.t,
            denoise_fn=denoise_fn,
            diffusion=diffusion,
            predict_x0=predict_x0,
            target_mask=target_mask,
            hybrid_lambda=0.0,
            step_size=eval_step_size,
            word_freq_logits=word_freq_logits,
            device=diffusion.device
        )
        log_likelihood = metrics_dict["base_loss"] / metrics_dict["denominator"]
        return State(
            t=state.t - eval_step_size,
            log_likelihood=state.log_likelihood + log_likelihood,
        ), None

    init_state = State(
        t=torch.tensor([diffusion.num_steps - eval_step_size], device=diffusion.device),
        log_likelihood=torch.tensor(0.0, device=diffusion.device),
    )
    num_steps = diffusion.num_steps // eval_step_size
    final_state, _ = utils.scan(elbo_body_fn, init_state, None, num_steps)
    log_likelihood = final_state.log_likelihood
    prior, denominator = compute_prior_kl(x_start, diffusion, target_mask=target_mask, word_freq_logits=word_freq_logits)
    if target_mask is not None:
        target_length = torch.count_nonzero(target_mask)
    else:
        target_length = None
    if length_probs is not None:
        length_probs = torch.tensor(length_probs, device=diffusion.device)
        length_log_likelihood = -torch.log(length_probs[target_length])
    else:
        length_log_likelihood = torch.tensor(0.0, device=diffusion.device)
    elbo = log_likelihood + length_log_likelihood + prior / denominator
    elbo_length = target_length if normalize_without_padding else x_start.size(-1)
    return {
        "elbo": elbo,
        "elbo_in_bits_per_dim": elbo / (np.log(2) * elbo_length),
        "likelihood": log_likelihood,
        "prior": prior,
        "length_likelihood": length_log_likelihood,
        "nn/num_steps": num_steps,
    }

def discrete_diffusion_predict_fn(
        shape,
        denoise_fn,
        diffusion,
        target_mask=None,
        predict_x0=False,
        use_maximum_likelihood_decoding=False,
        step_size=1,
        topk=0,
        topp=-1.0,
        context_fn=None,
        sample_cls=None,
        show_process=False,
        temperature=1.0):
    if diffusion.device is None:
        raise ValueError("Device not set in diffusion object")
    if show_process:
        tk = AutoTokenizer.from_pretrained('bert-base-uncased')
    num_steps = diffusion.num_steps
    assert num_steps % step_size == 0
    assert step_size < num_steps

    @dataclass
    class SamplingState:
        x: torch.Tensor
        x0: Any
        t: int

    length = shape[-1]

    def sampling_step(step, state):
        del step
        t = state.t
        logits, x0_logits = p_forward(
            denoise_fn,
            target_mask,
            x_t=state.x,
            t=t,
            diffusion=diffusion,
            predict_x0=predict_x0,
            return_x0=True,
            return_logits=True,
            maximum_likelihood=use_maximum_likelihood_decoding,
            step_size=step_size
        )
        x0 = x0_logits.argmax(-1) if x0_logits is not None else None
        # Validate logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"sampling_step: t={t}, WARNING: NaN or inf in logits, clamping")
            logits = torch.where(torch.isnan(logits) | torch.isinf(logits), torch.zeros_like(logits), logits)
        logits = logits.clamp(-1e4, 1e4)
        # Debug: Log logits stats
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()
        print(f"sampling_step: t={t}, logits entropy={entropy.item()}, min={logits.min().item()}, max={logits.max().item()}")
        logits = logits / temperature
        print("got here!!!!! %f"%diffusion.dim)
        logits = top_k_top_p_filtering(logits, top_k=topk, top_p=topp, vocab_size=diffusion.dim)
        # Check filtered logits
        if logits.sum(dim=-1).eq(0).any():
            print(f"sampling_step: t={t}, WARNING: All logits filtered out, using uniform distribution")
            logits = torch.zeros_like(logits) / logits.shape[-1]
        sample = torch.distributions.categorical.Categorical(logits=logits).sample().to(diffusion.device)
        # Debug: Log sample diversity
        unique_samples = torch.unique(sample, dim=0).shape[0]
        print(f"sampling_step: t={t}, unique_samples={unique_samples}/{sample.shape[0]}")
        if show_process:
            print(tk.batch_decode(x0, clean_up_tokenization_spaces=False))
        return SamplingState(x=sample, x0=x0, t=t - step_size)

    x = diffusion.sample_stationary(shape)
    if context_fn is not None:
        x = context_fn(x)
    init_state = SamplingState(
        x=x.to(diffusion.device),
        x0=x.to(diffusion.device) if predict_x0 else None,
        t=torch.tensor([num_steps], device=diffusion.device)
    )
    total_steps = num_steps // step_size
    final_state = utils.fori_loop(0, total_steps, sampling_step, init_state)
    predictions = {
        "final_state": final_state.x,
        "initial_state": init_state.x,
        "scalar/num_steps": num_steps,
        "scalar/length": length,
        "scalar/total_steps": total_steps,
    }
    return predictions