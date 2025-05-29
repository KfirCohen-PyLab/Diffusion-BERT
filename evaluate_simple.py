# Add current directory to Python path
import sys
sys.path.append('.')

# Basic imports
import os
import torch
from transformers import BertTokenizer, BertConfig
from models.modeling_diffusion_bert_checkpoint import DiffusionBertForMaskedLM
from sample import Categorical
from tqdm import tqdm
from dataloader import DiffusionLoader
from torch.nn.utils.rnn import pad_sequence

# Check GPU availability
print("Is CUDA available:", torch.cuda.is_available())
print("GPU Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Configuration
model_ckpt_path = './checkpoints/diffusion_bert_lm1b_final.pt'
model_name = 'bert-base-uncased'
predict_x0 = True
sample_strategy = 'Categorical'
num_steps = 2048
kind = 'word_freq'
word_freq_lambda = 0.3
schedule = 'mutual'
eval_step_size = 16

# Create necessary directories
os.makedirs('word_freq', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

# Model setup
tokenizer = BertTokenizer.from_pretrained(model_name)
sample_cls = Categorical()

# Diffusion setup
import diffusion_word_freq as diffusion
word_freq = torch.load(f'./word_freq/{model_name}_lm1b.pt', map_location=device)

def word_freq_preprocess_fn(wf):
    wf = wf + 1
    wf = wf.log()
    wf = wf / wf.max()
    return wf

word_freq = word_freq_preprocess_fn(word_freq)
diffusion_schedule = diffusion.create_discrete_diffusion_schedule(schedule, num_steps=num_steps)
diffusion_instance = diffusion.MaskDiffusion(
    dim=tokenizer.vocab_size,
    schedule=diffusion_schedule,
    tokenizer=tokenizer,
    sample_cls=sample_cls,
    word_freq=word_freq,
    word_freq_lambda=word_freq_lambda,
    device=device
)

# Model initialization
cfg = BertConfig.from_pretrained(model_name)
cfg.overall_timestep = diffusion_instance.num_steps

model = DiffusionBertForMaskedLM(cfg)
# Load state dict with CPU mapping first, then move to GPU
state_dict = torch.load(model_ckpt_path, map_location='cpu')
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# Create tensors on the correct device
cls = torch.full((1, 1), fill_value=tokenizer.cls_token_id, device=device)
sep = torch.full((1, 1), fill_value=tokenizer.sep_token_id, device=device)

# Denoise function setup
att_ones = torch.ones((1, 1), device=device)
att_zeros = torch.zeros((1, 1), device=device)

def denoise_fn(targets, timestep, attention_mask):
    assert len(targets.size()) == 2  # bsz * seqlen
    bsz = targets.size(0)
    targets = torch.cat((cls.repeat(bsz, 1), targets, sep.repeat(bsz, 1)), dim=1)
    attention_mask = torch.cat((att_ones.repeat(bsz, 1), attention_mask, att_zeros.repeat(bsz, 1)), dim=1)
    return model(
        input_ids=targets,
        attention_mask=attention_mask,
        timestep=timestep
    )['logits'][:, 1:-1, :]

# Data processing functions
def process_fn_in_collate(wf):
    return wf - wf.mean()

def collate_fn(batch_input):
    input_ids = [torch.tensor(d['input_ids'], device=device) for d in batch_input]
    attention_mask = [torch.tensor(d['attention_mask'], device=device) for d in batch_input]
    word_freq_logits = [process_fn_in_collate(word_freq.gather(0, torch.tensor(d['input_ids'], device=device))) for d in batch_input]
    input_ids = pad_sequence(input_ids, batch_first=True)
    attention_mask = pad_sequence(attention_mask, batch_first=True)
    word_freq_logits = pad_sequence(word_freq_logits, batch_first=True)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'word_freq_logits': word_freq_logits
    }

# Evaluation
elbo = 0.
count = 0

test_data = DiffusionLoader(tokenizer=tokenizer).my_load(task_name='lm1b', splits=['test'])[0]
_, test_data = test_data.train_test_split(test_size=5e-2).values()
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, collate_fn=collate_fn, num_workers=2, pin_memory=True)

with torch.no_grad():
    for batch in tqdm(test_loader):
        batch_dev_metrics = diffusion.discrete_diffusion_elbo(
            batch['input_ids'],  # Already on GPU from collate_fn
            denoise_fn=denoise_fn,
            diffusion=diffusion_instance,
            target_mask=batch['attention_mask'],  # Already on GPU from collate_fn
            word_freq_logits=batch['word_freq_logits'],  # Already on GPU from collate_fn
            normalize_without_padding=True,
            eval_step_size=eval_step_size,
            device=device
        )

        if not torch.isnan(batch_dev_metrics['elbo']):
            print(f"ELBO: {batch_dev_metrics['elbo']}")
            elbo += batch_dev_metrics['elbo']
            count += 1

print(f"Final ELBO: {elbo / (64. * count)}") 