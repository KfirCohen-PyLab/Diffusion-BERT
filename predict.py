import torch
import os
import time
from tqdm import tqdm
from transformers import BertTokenizer, BertConfig
from sample import Categorical, WholeWordMasking
from compute_metric import self_bleu, get_bleu 
import argparse
import torch.nn.functional as F
import warnings

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--topk", default=100, type=int)  # Increased
parser.add_argument("--topp", default=0.95, type=float)  # Increased
parser.add_argument("--step_size", default=2, type=int)
parser.add_argument("--name", default='D3PM', type=str)
parser.add_argument("--temperature", default=1.5, type=float)  # Increased
args = parser.parse_args()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters
step_size = args.step_size
model_name = 'bert-base-uncased'
predict_x0 = True
sample_strategy = 'Categorical'
num_steps = 512
schedule = 'mutual'
topk = args.topk
topp = args.topp
iteration = 2
shape = torch.Size([16, 32])
name = args.name
temperature = args.temperature
vocab_size = 30522  # bert-base-uncased vocab size
if num_steps % step_size != 0:
    raise ValueError(f"step_size={step_size} must divide num_steps={num_steps} evenly")
if step_size > 10:
    warnings.warn(f"Large step_size={step_size} may cause numerical instability...")
print(f"Running generation with model: {name}, temperature: {temperature}, topk: {topk}, topp: {topp}, step_size: {step_size}")

# Model checkpoint paths
model_path_dict = {
    'D3PM': (r'G:\ML_Project_Sem6\Diffusion-BERT-main\model_name_bert-base-uncased_lr_0.0005_seed_42_numsteps_512_sample_Categorical_schedule_mutual_hybridlambda_0.01_wordfreqlambda_0.3_fromscratch_True_timestep_none_ckpts\epoch_4.pt', 'token'),
    'custom': (r'G:\ML_Project_Sem6\Diffusion-BERT-main\model_name_bert-base-uncased_lr_3e-05_seed_42_numsteps_300_sample_Categorical_schedule_mutual_hybridlambda_0.05_wordfreqlambda_0.3_fromscratch_True_timestep_none_ckpts\epoch_5.pt', 'token'),
}
try:
    model_ckpt_path, timestep = model_path_dict[name]
except KeyError:
    raise ValueError(f"Model name '{name}' not found in model_path_dict")

# Import model class based on timestep strategy
if timestep in ['none', 'token']:
    from models.modeling_bert import BertForMaskedLM
elif timestep in ['embedding', 'layerwise']:
    from models.modeling_bert_new_timestep import BertForMaskedLM
else:
    raise NotImplementedError(f"Timestep strategy '{timestep}' not supported")

# Tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)

# Sample strategy
if sample_strategy == 'Categorical':
    sample_cls = Categorical(device=device)
elif sample_strategy == 'wwm':
    sample_cls = WholeWordMasking(tokenizer, device=device)
else:
    raise ValueError(f"Unknown sample strategy '{sample_strategy}'")

# Determine if using word frequency variant
kind = 'word_freq' if name.startswith('word_freq') or name == 'D3PM' else 'base'

# Load diffusion schedule and instance
import diffusion_word_freq as diffusion
diffusion_schedule = diffusion.create_discrete_diffusion_schedule(schedule, num_steps=num_steps, device=device)
if isinstance(diffusion_schedule, torch.Tensor):
    diffusion_schedule = diffusion_schedule.to(device)
elif isinstance(diffusion_schedule, dict):
    diffusion_schedule = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in diffusion_schedule.items()}

if kind == 'word_freq':
    word_freq_path = r'G:\ML_Project_Sem6\Diffusion-BERT-main\word_freq\bert-base-uncased.pt'
    try:
        word_freq = torch.load(word_freq_path).to(device)
        print(f"Loaded word_freq: shape={word_freq.shape}, device={word_freq.device}")
        word_freq = (word_freq + 1).log() / (word_freq + 1).log().max()
    except FileNotFoundError:
        raise FileNotFoundError(f"Word frequency file not found at {word_freq_path}...")
    except Exception as e:
        raise RuntimeError(f"Failed to load word frequency file {word_freq_path}: {str(e)}")
    diffusion_instance = diffusion.MaskDiffusion(
        dim=tokenizer.vocab_size,
        schedule=diffusion_schedule,
        tokenizer=tokenizer,
        sample_cls=sample_cls,
        word_freq=word_freq,
        word_freq_lambda=0.3,
        device=device
    )
else:
    diffusion_instance = diffusion.MaskDiffusion(
        dim=tokenizer.vocab_size,
        schedule=diffusion_schedule,
        tokenizer=tokenizer,
        sample_cls=sample_cls,
        device=device
    )

# Load model config and model
cfg = BertConfig.from_pretrained(model_name)
cfg.overall_timestep = diffusion_instance.num_steps
model = BertForMaskedLM(cfg)
try:
    ckpt = torch.load(model_ckpt_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    model.load_state_dict(state_dict)
except Exception as e:
    raise RuntimeError(f"Failed to load checkpoint from {model_ckpt_path}: {str(e)}")
model = model.to(device)
model.eval()

# Special tokens
cls = torch.full((1, 1), tokenizer.cls_token_id, device=device)
sep = torch.full((1, 1), tokenizer.sep_token_id, device=device)

# Denoise function
def denoise_fn(targets, timestep, attention_mask):
    device = next(model.parameters()).device
    bsz = targets.size(0)
    targets = targets.to(device)
    attention_mask = attention_mask.to(device)
    
    if isinstance(timestep, torch.Tensor):
        timestep = timestep.to(device)
        timestep_val = timestep.item()
    else:
        timestep_val = timestep

    if timestep_val == 'none':
        input_ids = torch.cat((cls.repeat(bsz, 1), targets, sep.repeat(bsz, 1)), dim=1)
        attention_mask = torch.cat(
            (torch.ones(bsz, 1, device=device), attention_mask, torch.ones(bsz, 1, device=device)), dim=1
        )
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = outputs.logits[:, 1:-1, :]
    elif isinstance(timestep_val, (int, float)):
        token_id = torch.full((bsz, 1), fill_value=int(timestep_val) + 110, device=device)
        input_ids = torch.cat((cls.repeat(bsz, 1), token_id, targets, sep.repeat(bsz, 1)), dim=1)
        attention_mask = torch.cat(
            (torch.ones(bsz, 2, device=device), attention_mask, torch.ones(bsz, 1, device=device)), dim=1
        )
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, timestep=timestep - 1, return_dict=True)
        logits = outputs.logits[:, 2:-1, :]
    else:
        raise NotImplementedError(f"Timestep handling for '{timestep_val}' not implemented")

    # Validate logits
    if logits.shape[-1] != vocab_size:
        print(f"denoise_fn: timestep={timestep_val}, WARNING: logits.shape={logits.shape}, expected last dim={vocab_size}")
        raise ValueError(f"Invalid logits shape: {logits.shape}")
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        print(f"denoise_fn: timestep={timestep_val}, WARNING: NaN or inf detected in logits")
        logits = torch.where(torch.isnan(logits) | torch.isinf(logits), torch.zeros_like(logits), logits)
    # Debug: Log logits stats
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()
    print(f"denoise_fn: timestep={timestep_val}, logits entropy={entropy.item()}, min={logits.min().item()}, max={logits.max().item()}")
    return logits

# Generation loop
os.makedirs('./generation_results', exist_ok=True)
output_path = f'./generation_results/{name}_step_curve.txt'
temp_path = './temp.txt'

try:
    with open(temp_path, 'w', encoding='utf-8') as fdata, open(output_path, 'a+', encoding='utf-8') as fcurve:
        sentences = []
        with torch.no_grad():
            for i in tqdm(range(iteration), desc="Generating..."):
                start = time.time()
                state = diffusion.discrete_diffusion_predict_fn(
                    shape=shape,
                    denoise_fn=denoise_fn,
                    diffusion=diffusion_instance,
                    predict_x0=predict_x0,
                    sample_cls=sample_cls,
                    step_size=step_size,
                    topk=topk,
                    topp=topp,
                    target_mask=torch.ones(shape, device=device),
                    show_process=True,
                    temperature=temperature
                )['final_state']

                duration = time.time() - start
                print(duration, file=fcurve)

                sentence_batch = tokenizer.batch_decode(state, skip_special_tokens=True)
                print(f"Iteration {i}: {sentence_batch[:5]}")
                sentences.extend(sentence_batch)
                for s in sentence_batch:
                    print(s, file=fdata, flush=True)
except Exception as e:
    print(f"Error during generation: {str(e)}")
    raise

# Optional: Compute BLEU scores
if sentences:
    bleu_scores = self_bleu(sentences)
    print(f"Self-BLEU: {bleu_scores}")