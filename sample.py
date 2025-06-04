import torch
import abc


class SampleClassBase(abc.ABC):
    def sample(self, logits, x_0):
        raise NotImplementedError

    def post_process_sample_in_prediction(self, sample, x_0):
        return sample


class Categorical(SampleClassBase):
    def __init__(self, device=None):
        self.device = device

    def sample(self, logits, x_0):
        logits = logits.to(self.device) if self.device is not None else logits
        x_0 = x_0.to(self.device) if self.device is not None else x_0
        # Validate logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"Categorical.sample: WARNING: NaN or inf in logits, using uniform distribution")
            logits = torch.ones_like(logits) / logits.shape[-1]
        # Debug: Log logits stats
        print(f"Categorical.sample: logits min={logits.min().item()}, max={logits.max().item()}")
        try:
            sample = torch.distributions.categorical.Categorical(logits=logits).sample()
        except RuntimeError as e:
            print(f"Categorical.sample: ERROR: {str(e)}, using uniform distribution")
            sample = torch.randint(0, logits.shape[-1], logits.shape[:-1], device=logits.device)
        # Debug: Log sample diversity
        unique_samples = torch.unique(sample, dim=0).shape[0]
        print(f"Categorical.sample: unique_samples={unique_samples}/{sample.shape[0]}")
        return sample


class WholeWordMasking(SampleClassBase):
    def __init__(self, tokenizer, device=None):
        self.dim = tokenizer.vocab_size
        self.mask_id = tokenizer.mask_token_id
        self.device = device
        self.post_tokens = torch.zeros(size=(tokenizer.vocab_size,), device=self.device, dtype=torch.long)
        for token, id in tokenizer.vocab.items():
            if token.startswith('##'):
                self.post_tokens[id] = 1

    def sample(self, logits, x_0):
        logits = logits.to(self.device) if self.device is not None else logits
        x_0 = x_0.to(self.device) if self.device is not None else x_0
        is_post = (self.post_tokens.to(x_0.device) * x_0).sum(-1).nonzero()
        # Validate logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"WholeWordMasking.sample: WARNING: NaN or inf in logits, using uniform distribution")
            logits = torch.ones_like(logits) / logits.shape[-1]
        # Debug: Log logits stats
        print(f"WholeWordMasking.sample: logits min={logits.min().item()}, max={logits.max().item()}")
        try:
            samp = torch.distributions.categorical.Categorical(logits=logits).sample()
        except RuntimeError as e:
            print(f"WholeWordMasking.sample: ERROR: {str(e)}, using uniform distribution")
            samp = torch.randint(0, logits.shape[-1], logits.shape[:-1], device=logits.device)
        for index in is_post:
            samp[index[0], index[1]] = self.mask_id if samp[index[0], index[1] - 1] == self.mask_id else x_0[index[0], index[1]].argmax()
        # Debug: Log sample diversity
        unique_samples = torch.unique(samp, dim=0).shape[0]
        print(f"WholeWordMasking.sample: unique_samples={unique_samples}/{samp.shape[0]}")
        return samp.to(self.device) if self.device is not None else samp

    def post_process_sample_in_prediction(self, sample, x_0):
        sample = sample.to(self.device) if self.device is not None else sample
        x_0 = x_0.to(self.device) if self.device is not None else x_0
        x_0_one_hot = torch.nn.functional.one_hot(x_0, num_classes=self.dim).to(self.device)
        is_post = (self.post_tokens.to(x_0.device) * x_0_one_hot).sum(-1).nonzero()
        for index in is_post:
            sample[index[0], index[1]] = self.mask_id if sample[index[0], index[1] - 1] == self.mask_id else x_0_one_hot[index[0], index[1]].argmax()
        return sample.to(self.device) if self.device is not None else sample