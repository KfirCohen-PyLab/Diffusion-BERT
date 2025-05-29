import torch
import numpy as np
from tqdm import tqdm

def create_discrete_diffusion_schedule(schedule_name, num_steps):
    if schedule_name == 'mutual':
        betas = torch.linspace(0.0001, 0.02, num_steps)
    else:
        raise NotImplementedError
    
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_prod': alphas_prod,
        'alphas_prod_p': alphas_prod_p,
        'alphas_bar_sqrt': alphas_bar_sqrt,
        'one_minus_alphas_bar_log': one_minus_alphas_bar_log,
        'one_minus_alphas_bar_sqrt': one_minus_alphas_bar_sqrt,
    }

class MaskDiffusion:
    def __init__(self, dim, schedule, tokenizer, sample_cls, word_freq=None, word_freq_lambda=0.3, device='cuda'):
        self.dim = dim
        self.schedule = schedule
        self.num_steps = len(schedule['betas'])
        self.tokenizer = tokenizer
        self.sample_cls = sample_cls
        self.word_freq = word_freq
        self.word_freq_lambda = word_freq_lambda
        self.device = device

        for k, v in schedule.items():
            self.register_buffer(k, v.to(device))

    def register_buffer(self, name, val):
        setattr(self, name, val.to(self.device))

    def q_sample(self, x_0, t):
        # Compute q(x_t | x_0)
        noise_level = self.alphas_bar_sqrt[t]
        noise = torch.randn_like(noise_level)
        return x_0 * noise_level + noise * self.one_minus_alphas_bar_sqrt[t]

    def compute_loss(self, x_0, t, model_output, word_freq_logits=None):
        # Word frequency weighted loss
        if word_freq_logits is not None and self.word_freq_lambda > 0:
            freq_weight = torch.exp(self.word_freq_lambda * word_freq_logits)
            freq_weight = freq_weight / freq_weight.mean()
        else:
            freq_weight = 1.0

        # Compute the loss
        noise_level = self.alphas_bar_sqrt[t]
        noise = torch.randn_like(noise_level)
        x_t = x_0 * noise_level + noise * self.one_minus_alphas_bar_sqrt[t]
        
        pred = model_output
        target = x_0
        
        loss = torch.nn.functional.cross_entropy(
            pred.view(-1, self.dim),
            target.view(-1),
            reduction='none'
        )
        
        loss = loss * freq_weight.view(-1)
        return loss.mean()

    def p_sample(self, x_t, t, denoise_fn):
        # Sample from p(x_{t-1} | x_t)
        model_output = denoise_fn(x_t, t)
        return self.sample_cls.sample(model_output, x_t) 