import torch
import os
import glob
import re
from tqdm import tqdm
from transformers import BertTokenizer, BertConfig
import argparse
import torch.nn.functional as F
from compute_metric import self_bleu, get_bleu
import warnings


# Argument parsing with more options
parser = argparse.ArgumentParser()
parser.add_argument("--topk", default=100, type=int)
parser.add_argument("--topp", default=0.95, type=float)
parser.add_argument("--step_size", default=2, type=int)
parser.add_argument("--temperature", default=1.5, type=float)
parser.add_argument("--t_start", default=128, type=int)
#parser.add_argument("--checkpoint_dir", default="G:\\ML_Project_Sem6\\Diffusion-BERT-main\\model_name_bert-base-uncased_lr_0.0005_seed_42_numsteps_512_sample_Categorical_schedule_mutual_hybridlambda_0.01_wordfreqlambda_0.3_fromscratch_True_timestep_none_ckpts", type=str)
parser.add_argument("--checkpoint_dir", default="G:\ML_Project_Sem6\Diffusion-BERT-main\model_name_bert-base-uncased_lr_3e-05_seed_42_numsteps_300_sample_Categorical_schedule_mutual_hybridlambda_0.05_wordfreqlambda_0.3_fromscratch_True_timestep_none_ckpts", type=str)
args = parser.parse_args()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model configuration
model_name = 'bert-base-uncased'
predict_x0 = True
sample_strategy = 'Categorical'
num_steps = 512
schedule = 'mutual'

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)

# Initialize components
from models.modeling_bert import BertForMaskedLM
from sample import Categorical
import diffusion_word_freq as diffusion

# Initialize diffusion
diffusion_schedule = diffusion.create_discrete_diffusion_schedule(schedule, num_steps=num_steps, device=device)
sample_cls = Categorical(device=device)

# Load word frequencies
word_freq_path = r'G:\ML_Project_Sem6\Diffusion-BERT-main\word_freq\bert-base-uncased.pt'
word_freq = torch.load(word_freq_path).to(device)
word_freq = (word_freq + 1).log() / (word_freq + 1).log().max()

# Create diffusion instance
diffusion_instance = diffusion.MaskDiffusion(
    dim=tokenizer.vocab_size,
    schedule=diffusion_schedule,
    tokenizer=tokenizer,
    sample_cls=sample_cls,
    word_freq=word_freq,
    word_freq_lambda=0.3,
    device=device
)

# Test texts
test_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world."
]

def evaluate_checkpoint(checkpoint_path, params):
    """Evaluate a single checkpoint with given parameters"""
    try:
        # Load model
        cfg = BertConfig.from_pretrained(model_name)
        cfg.overall_timestep = diffusion_instance.num_steps
        model = BertForMaskedLM(cfg)
        
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt.get('model_state_dict', ckpt.get('model', ckpt)))
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"Failed to load {checkpoint_path}: {str(e)}")
        return None


def generate_texts(texts, device, params):

    model.eval()
    generated_texts = []
    masked_positions_log = []  # To track masked tokens
    
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding='max_length',
                         truncation=True, max_length=12, 
                         add_special_tokens=True)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        
        # Log original input
        original_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        print(f"\nInput: {original_text}")
        
        # Initialize with 50% masks, 50% random tokens
        masked = torch.full_like(input_ids, tokenizer.mask_token_id)
        random = torch.randint(0, tokenizer.vocab_size, input_ids.size(), device=device)
        current_q = torch.where(torch.rand_like(input_ids.float()) < 0.5, masked, random)
        
        # Track masked positions
        masked_positions = (current_q == tokenizer.mask_token_id).nonzero(as_tuple=True)[1].tolist()
        masked_words = [tokenizer.decode([input_ids[0][pos].item()]) for pos in masked_positions]
        print(f"Initially Masked Words: {masked_words}")
        
        for current_t in range(params['t_start'], 0, -params['step_size']):
            t_tensor = torch.full((input_ids.size(0),), current_t, device=device)
            
            with torch.no_grad():
                outputs = model(input_ids=current_q,
                              attention_mask=attention_mask,
                              timestep=t_tensor)
                logits = outputs.logits / max(params['temperature'], 0.1)
                
                # Apply top-k and top-p filtering
                if params['topk'] > 0:
                    top_k = min(params['topk'], logits.size(-1))
                    indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
                    logits[indices_to_remove] = -float('Inf')
                
                if params['topp'] > 0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > params['topp']
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('Inf')
                
                probs = F.softmax(logits, dim=-1)
                samples = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(*current_q.shape)
                current_q = samples

        # Decode and clean generated text
        generated = tokenizer.decode(current_q[0], skip_special_tokens=True)
        generated = re.sub(r'\[unused\d+\]', '', generated)
        generated = re.sub(r'[^\w\s]', '', generated)
        generated = re.sub(r'\s+', ' ', generated).strip()
        generated_texts.append(generated)
        
        print(f"Generated: {generated}")
    
    return generated_texts

    # Generate texts
    generated_texts = generate_texts(test_texts, device, params)
    
    # Calculate BLEU scores
    bleu_scores = []
    for ref, gen in zip(test_texts, generated_texts):
        bleu = get_bleu(ref, gen)
        bleu_scores.append(bleu)
    
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    self_bleu_score = self_bleu(generated_texts) if len(generated_texts) > 1 else 0.0
    
    # Only return results if BLEU > 0
    if avg_bleu > 0:
        print(f"\n✅ Non-Zero BLEU Found: {avg_bleu:.4f}")
        return {
            'checkpoint': os.path.basename(checkpoint_path),
            'params': params,
            'generated': generated_texts,
            'bleu_scores': bleu_scores,
            'avg_bleu': avg_bleu,
            'self_bleu': self_bleu_score
        }
    else:
        return None

def evaluate_all_checkpoints():
    """Evaluate all checkpoints in the directory with different parameters"""
    # Parameter combinations to test
    param_combinations = [
        # Baseline
        {'topk': 100, 'topp': 0.95, 'temperature': 1.0, 't_start': 128, 'step_size': 2},
        
        # Top-k variations
        {'topk': 50, 'topp': 0.95, 'temperature': 1.0, 't_start': 128, 'step_size': 2},
        {'topk': 150, 'topp': 0.95, 'temperature': 1.0, 't_start': 128, 'step_size': 2},
        {'topk': 200, 'topp': 0.95, 'temperature': 1.0, 't_start': 128, 'step_size': 2},
        
        # Top-p variations
        {'topk': 100, 'topp': 0.85, 'temperature': 1.0, 't_start': 128, 'step_size': 2},
        {'topk': 100, 'topp': 0.90, 'temperature': 1.0, 't_start': 128, 'step_size': 2},
        {'topk': 100, 'topp': 0.98, 'temperature': 1.0, 't_start': 128, 'step_size': 2},
        
        # Temperature variations
        {'topk': 100, 'topp': 0.95, 'temperature': 0.7, 't_start': 128, 'step_size': 2},
        {'topk': 100, 'topp': 0.95, 'temperature': 0.9, 't_start': 128, 'step_size': 2},
        {'topk': 100, 'topp': 0.95, 'temperature': 1.2, 't_start': 128, 'step_size': 2},
        
        # Diffusion steps variations
        {'topk': 100, 'topp': 0.95, 'temperature': 1.0, 't_start': 64, 'step_size': 2},
        {'topk': 100, 'topp': 0.95, 'temperature': 1.0, 't_start': 96, 'step_size': 2},
        {'topk': 100, 'topp': 0.95, 'temperature': 1.0, 't_start': 160, 'step_size': 2},
        
        # Step size variations
        {'topk': 100, 'topp': 0.95, 'temperature': 1.0, 't_start': 128, 'step_size': 1},
        {'topk': 100, 'topp': 0.95, 'temperature': 1.0, 't_start': 128, 'step_size': 4},
        
        # Combined variations
        {'topk': 50, 'topp': 0.90, 'temperature': 0.9, 't_start': 96, 'step_size': 2},
        {'topk': 150, 'topp': 0.98, 'temperature': 1.2, 't_start': 64, 'step_size': 1},
        {'topk': 200, 'topp': 0.85, 'temperature': 0.7, 't_start': 160, 'step_size': 4}
    ]
    
    # Find all checkpoint files
    checkpoint_files = sorted(glob.glob(os.path.join(args.checkpoint_dir, '*.pt')))
    if not checkpoint_files:
        raise ValueError(f"No checkpoint files found in {args.checkpoint_dir}")
    
    results = []
    best_result = None
    best_bleu = 0
    
    # Create results directory
    os.makedirs('./generation_results', exist_ok=True)
    
    for checkpoint_path in tqdm(checkpoint_files, desc="Evaluating checkpoints"):
        for params in param_combinations:
            try:
                result = evaluate_checkpoint(checkpoint_path, params)
                if result:
                    results.append(result)
                    
                    # Print current best
                    if result['avg_bleu'] > best_bleu:
                        best_bleu = result['avg_bleu']
                        best_result = result
                        print(f"\n🔥 NEW BEST: BLEU={best_bleu:.4f}")
                        print(f"Checkpoint: {result['checkpoint']}")
                        print(f"Parameters: {params}")
                        for ref, gen, score in zip(test_texts, result['generated'], result['bleu_scores']):
                            print(f"\nReference: {ref}")
                            print(f"Generated: {gen}")
                            print(f"BLEU: {score:.4f}")
                    
                    # Save individual results
                    with open(f"./generation_results/{result['checkpoint']}_topk{params['topk']}_topp{params['topp']}_temp{params['temperature']}.txt", 'w', encoding='utf-8') as f:
                        f.write(f"Parameters: {params}\n")
                        f.write(f"Avg BLEU: {result['avg_bleu']:.4f}\n")
                        f.write(f"Self-BLEU: {result['self_bleu']:.4f}\n\n")
                        for ref, gen, score in zip(test_texts, result['generated'], result['bleu_scores']):
                            f.write(f"Reference: {ref}\n")
                            f.write(f"Generated: {gen}\n")
                            f.write(f"BLEU: {score:.4f}\n\n")
            except Exception as e:
                print(f"Error evaluating {checkpoint_path} with {params}: {str(e)}")
                continue
    
    # Save summary of all results
    with open('./generation_results/summary.csv', 'w', encoding='utf-8') as f:
        f.write("Checkpoint,TopK,TopP,Temperature,T_Start,Step_Size,Avg_BLEU,Self_BLEU\n")
        for result in results:
            f.write(f"{result['checkpoint']},")
            f.write(f"{result['params']['topk']},")
            f.write(f"{result['params']['topp']},")
            f.write(f"{result['params']['temperature']},")
            f.write(f"{result['params']['t_start']},")
            f.write(f"{result['params']['step_size']},")
            f.write(f"{result['avg_bleu']:.4f},")
            f.write(f"{result['self_bleu']:.4f}\n")
    
    # Save best result
    if best_result:
        with open('./generation_results/best_result.txt', 'w', encoding='utf-8') as f:
            f.write(f"Best Checkpoint: {best_result['checkpoint']}\n")
            f.write(f"Parameters: {best_result['params']}\n")
            f.write(f"Avg BLEU: {best_result['avg_bleu']:.4f}\n")
            f.write(f"Self-BLEU: {best_result['self_bleu']:.4f}\n\n")
            for ref, gen, score in zip(test_texts, best_result['generated'], best_result['bleu_scores']):
                f.write(f"Reference: {ref}\n")
                f.write(f"Generated: {gen}\n")
                f.write(f"BLEU: {score:.4f}\n\n")
    
    return best_result

if __name__ == "__main__":
    best_result = evaluate_all_checkpoints()
    if best_result:
        print("\n=== FINAL BEST RESULT ===")
        print(f"Checkpoint: {best_result['checkpoint']}")
        print(f"Avg BLEU: {best_result['avg_bleu']:.4f}")
        print(f"Results saved to ./generation_results/")
        
    import matplotlib.pyplot as plt

    # Your data
    data = [
        1.2374117374420166,
        0.973400354385376,
        20.636457681655884,
        20.080617904663086,
        1.3640563488006592,
        1.45560622215271
    ]

    # Plot step curve
    plt.step(range(len(data)), data, where='mid', label='Step Curve')
    plt.title("Step Curve Visualization")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    import numpy as np

    data_array = np.array(data)
    print("Mean:", np.mean(data_array))
    print("Max:", np.max(data_array))
    print("Min:", np.min(data_array))
    
    # Save to file
    with open('step_curve.txt', 'w') as f:
        for value in data:
            f.write(f"{value}\n")

    # Load from file
    with open('step_curve.txt', 'r') as f:
        loaded_data = [float(line.strip()) for line in f]
        
    plt.plot(data, label='Training Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Progress")
    plt.show()
    
    changes = np.diff(data)
    big_jumps = np.where(np.abs(changes) > 5)[0]  # Threshold=5
    print("Big jumps at steps:", big_jumps)
    
    from scipy.ndimage import gaussian_filter1d

    smoothed = gaussian_filter1d(data, sigma=1)
    plt.plot(data, label='Original')
    plt.plot(smoothed, label='Smoothed')
    plt.legend()
    plt.show()
    
    import pandas as pd

    df = pd.DataFrame({'Step': range(len(data)), 'Value': data})
    print(df.describe())  # Statistics