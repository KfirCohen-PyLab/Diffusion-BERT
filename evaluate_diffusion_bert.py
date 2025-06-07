import torch
import os
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig
from sample import Categorical, WholeWordMasking
import argparse
import re
import functools
import time
import warnings

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, 
                       default=r"G:\ML_Project_Sem6\Diffusion-BERT-main\model_name_bert-base-uncased_lr_0.0005_seed_42_numsteps_512_sample_Categorical_schedule_mutual_hybridlambda_0.01_wordfreqlambda_0.3_fromscratch_True_timestep_none_ckpts",
                       help="Path to checkpoint directory")
    parser.add_argument("--load_step", type=str, default="epoch_4.pt",
                       help="Specific checkpoint step to load")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Evaluation batch size")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of text samples to generate")
    parser.add_argument("--max_length", type=int, default=32,
                       help="Max length for generated samples")
    parser.add_argument("--step_size", type=int, default=4,
                       help="Step size for generation")
    parser.add_argument("--topk", type=int, default=50,
                       help="Top-k filtering")
    parser.add_argument("--topp", type=float, default=0.9,
                       help="Top-p filtering")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                       help="Model name")
    return parser.parse_args()

def load_model_and_diffusion(args):
    """Load model, tokenizer, and diffusion components"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Import model and diffusion
    from models.modeling_bert import BertForMaskedLM
    import diffusion_word_freq as diffusion
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    
    # Initialize sample class
    sample_cls = Categorical(device=device)
    
    # Create diffusion schedule
    diffusion_schedule = diffusion.create_discrete_diffusion_schedule(
        'mutual', num_steps=512, device=device
    )
    
    # Load word frequency
    word_freq_path = r'G:\ML_Project_Sem6\Diffusion-BERT-main\word_freq\bert-base-uncased_lm1b.pt'
    try:
        word_freq = torch.load(word_freq_path).to(device)
        print(f"Loaded word_freq: shape={word_freq.shape}, device={word_freq.device}")
        word_freq = (word_freq + 1).log() / (word_freq + 1).log().max()
    except FileNotFoundError:
        print("Warning: Word frequency file not found, using None")
        word_freq = None
    
    # Initialize diffusion
    diffusion_instance = diffusion.MaskDiffusion(
        dim=tokenizer.vocab_size,
        schedule=diffusion_schedule,
        tokenizer=tokenizer,
        sample_cls=sample_cls,
        word_freq=word_freq,
        word_freq_lambda=0.3,
        device=device
    )
    
    # Load model
    cfg = BertConfig.from_pretrained(args.model_name)
    cfg.overall_timestep = diffusion_instance.num_steps
    model = BertForMaskedLM(cfg)
    
    # Load checkpoint
    ckpt_path = os.path.join(args.checkpoint_path, args.load_step)
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
        model.load_state_dict(state_dict)
        print(f"Successfully loaded checkpoint: {ckpt_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {ckpt_path}: {str(e)}")
    
    model = model.to(device)
    model.eval()
    
    return model, tokenizer, diffusion_instance, diffusion, device

def create_denoise_fn(model, tokenizer, device):
    """Create the denoise function for the diffusion process"""
    cls = torch.full((1, 1), tokenizer.cls_token_id, device=device)
    sep = torch.full((1, 1), tokenizer.sep_token_id, device=device)
    vocab_size = tokenizer.vocab_size
    
    def denoise_fn(targets, timestep, attention_mask):
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
            print(f"denoise_fn: WARNING: logits.shape={logits.shape}, expected last dim={vocab_size}")
            raise ValueError(f"Invalid logits shape: {logits.shape}")
        
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"denoise_fn: WARNING: NaN or inf detected in logits")
            logits = torch.where(torch.isnan(logits) | torch.isinf(logits), torch.zeros_like(logits), logits)
        
        return logits
    
    return denoise_fn

def generate_samples(denoise_fn, diffusion_instance, diffusion, args, device):
    """Generate text samples using the diffusion model"""
    samples = []
    shape = torch.Size([args.batch_size, args.max_length])
    
    print(f"Generating {args.num_samples} samples...")
    print(f"Vocab size: {diffusion_instance.tokenizer.vocab_size}")
    print(f"Special tokens: CLS={diffusion_instance.tokenizer.cls_token_id}, SEP={diffusion_instance.tokenizer.sep_token_id}, MASK={diffusion_instance.tokenizer.mask_token_id}")
    
    for i in tqdm(range(args.num_samples // args.batch_size + 1), desc="Generating batches"):
        if i * args.batch_size >= args.num_samples:
            break
            
        try:
            with torch.no_grad():
                state = diffusion.discrete_diffusion_predict_fn(
                    shape=shape,
                    denoise_fn=denoise_fn,
                    diffusion=diffusion_instance,
                    predict_x0=True,
                    step_size=args.step_size,
                    topk=args.topk,
                    topp=args.topp,
                    target_mask=torch.ones(shape, device=device),
                    show_process=False,
                    temperature=args.temperature
                )['final_state']
                
                # Debug: Print token IDs for first sample
                if i == 0:
                    print(f"\nDEBUG: First sample token IDs: {state[0][:10].cpu().tolist()}")
                    print(f"DEBUG: Token ID range: min={state.min().item()}, max={state.max().item()}")
                    
                    # Check for problematic tokens
                    problematic_tokens = []
                    for token_id in state[0][:10].cpu().tolist():
                        if token_id >= diffusion_instance.tokenizer.vocab_size:
                            problematic_tokens.append(token_id)
                    if problematic_tokens:
                        print(f"DEBUG: Found problematic token IDs: {problematic_tokens}")
                
                # Clamp token IDs to valid range
                state = torch.clamp(state, 0, diffusion_instance.tokenizer.vocab_size - 1)

                # Decode samples
                batch_samples = diffusion_instance.tokenizer.batch_decode(state, skip_special_tokens=True)
                
                # Alternative decoding for debugging
                if i == 0:
                    print("\nDEBUG: Raw token decoding:")
                    raw_tokens = [diffusion_instance.tokenizer.decode([token_id]) for token_id in state[0][:10].cpu().tolist()]
                    print(f"First 10 raw tokens: {raw_tokens}")
                
                samples.extend(batch_samples)
                
                # Print some samples for monitoring
                if i == 0:
                    print("Sample generations:")
                    for j, sample in enumerate(batch_samples[:3]):
                        print(f"  {j+1}: {sample}")
                        
        except Exception as e:
            print(f"Error generating batch {i}: {str(e)}")
            # Add placeholder samples for failed batches
            for _ in range(min(args.batch_size, args.num_samples - len(samples))):
                samples.append(f"[Generation failed: {str(e)}]")
    
    return samples[:args.num_samples]

def evaluate_model_quality(samples):
    """Evaluate the quality of generated samples"""
    results = {
        'total_samples': len(samples),
        'failed_samples': 0,
        'empty_samples': 0,
        'mask_only_samples': 0,
        'valid_samples': 0,
        'avg_length': 0,
        'unique_samples': 0
    }
    
    valid_samples = []
    lengths = []
    
    for sample in samples:
        if "[Generation failed" in sample:
            results['failed_samples'] += 1
        elif len(sample.strip()) == 0:
            results['empty_samples'] += 1
        elif sample.count('[MASK]') == len(sample.split()):
            results['mask_only_samples'] += 1
        else:
            results['valid_samples'] += 1
            valid_samples.append(sample)
            lengths.append(len(sample.split()))
    
    if lengths:
        results['avg_length'] = np.mean(lengths)
    
    results['unique_samples'] = len(set(valid_samples))
    
    return results, valid_samples

def main():
    args = parse_args()
    print(f"Arguments: {args}")
    
    # Load model and components
    try:
        model, tokenizer, diffusion_instance, diffusion, device = load_model_and_diffusion(args)
        print("Model and diffusion loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create denoise function
    denoise_fn = create_denoise_fn(model, tokenizer, device)
    
    # Generate samples
    try:
        start_time = time.time()
        samples = generate_samples(denoise_fn, diffusion_instance, diffusion, args, device)
        generation_time = time.time() - start_time
        
        print(f"\nGeneration completed in {generation_time:.2f} seconds")
        print(f"Average time per sample: {generation_time/len(samples):.2f} seconds")
        
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Evaluate sample quality
    results, valid_samples = evaluate_model_quality(samples)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total samples: {results['total_samples']}")
    print(f"Valid samples: {results['valid_samples']}")
    print(f"Failed samples: {results['failed_samples']}")
    print(f"Empty samples: {results['empty_samples']}")
    print(f"Mask-only samples: {results['mask_only_samples']}")
    print(f"Unique samples: {results['unique_samples']}")
    print(f"Average length: {results['avg_length']:.2f} words")
    print(f"Success rate: {results['valid_samples']/results['total_samples']*100:.1f}%")
    
    # Print all samples
    print("\n" + "="*50)
    print("ALL GENERATED SAMPLES")
    print("="*50)
    for i, sample in enumerate(samples, 1):
        print(f"{i:2d}. {sample}")
    
    # Save results
    os.makedirs('./evaluation_results', exist_ok=True)
    
    with open('./evaluation_results/generation_results.txt', 'w', encoding='utf-8') as f:
        f.write("EVALUATION RESULTS\n")
        f.write("="*50 + "\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
        f.write(f"Success rate: {results['valid_samples']/results['total_samples']*100:.1f}%\n")
        f.write("\nALL GENERATED SAMPLES\n")
        f.write("="*50 + "\n")
        for i, sample in enumerate(samples, 1):
            f.write(f"{i:2d}. {sample}\n")
    
    print(f"\nResults saved to ./evaluation_results/generation_results.txt")

if __name__ == '__main__':
    main()