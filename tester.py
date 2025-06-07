import torch
import os
import time
from transformers import BertTokenizer, BertConfig
from sample import Categorical, WholeWordMasking

def load_model_and_diffusion(model_path, device):
    """Load model, tokenizer, and diffusion components"""
    # Import model and diffusion
    from models.modeling_bert import BertForMaskedLM
    import diffusion_word_freq as diffusion
    
    # Model configuration
    model_name = 'bert-base-uncased'
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
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
    cfg = BertConfig.from_pretrained(model_name)
    cfg.overall_timestep = diffusion_instance.num_steps
    model = BertForMaskedLM(cfg)
    
    # Load checkpoint
    try:
        ckpt = torch.load(model_path, map_location=device)
        state_dict = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
        model.load_state_dict(state_dict)
        print(f"Successfully loaded checkpoint: {model_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {model_path}: {str(e)}")
    
    model = model.to(device)
    model.eval()
    
    return model, tokenizer, diffusion_instance, diffusion

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

def generate_texts(model, tokenizer, diffusion_instance, diffusion, denoise_fn, texts, device, 
                  step_size=2, topk=100, topp=0.95, temperature=1.5, max_length=32):
    """
    Generate texts using the diffusion model.
    
    Args:
        model: loaded BERT model
        tokenizer: BERT tokenizer
        diffusion_instance: MaskDiffusion instance
        diffusion: diffusion module
        denoise_fn: denoise function
        texts: list of input strings (not used for unconditional generation)
        device: cuda or cpu
        step_size: sampling step size
        topk: top-k filtering
        topp: top-p filtering  
        temperature: sampling temperature
        max_length: maximum sequence length
        
    Returns:
        List of generated text strings
    """
    batch_size = len(texts)
    shape = torch.Size([batch_size, max_length])
    
    print(f"Generating {batch_size} samples with shape {shape}")
    
    with torch.no_grad():
        # Use the discrete diffusion prediction function
        state = diffusion.discrete_diffusion_predict_fn(
            shape=shape,
            denoise_fn=denoise_fn,
            diffusion=diffusion_instance,
            predict_x0=True,
            step_size=step_size,
            topk=topk,
            topp=topp,
            target_mask=torch.ones(shape, device=device),
            show_process=False,
            temperature=temperature
        )['final_state']

        # Decode generated samples
        generated_texts = tokenizer.batch_decode(state, skip_special_tokens=True)
    
    return generated_texts

def test_conditional_generation(model, tokenizer, diffusion_instance, diffusion, denoise_fn, texts, device):
    """
    Test conditional generation by starting from input texts.
    This is more experimental and may not work as well.
    """
    print("Testing conditional generation (experimental)...")
    
    # Tokenize input texts
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=32)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    
    batch_size, seq_len = input_ids.shape
    
    # Add some noise to the input by masking some tokens
    noise_ratio = 0.3  # Mask 30% of tokens
    mask_positions = torch.rand(batch_size, seq_len) < noise_ratio
    noisy_input_ids = input_ids.clone()
    noisy_input_ids[mask_positions] = tokenizer.mask_token_id
    
    print("Noisy inputs:")
    for i, noisy_text in enumerate(tokenizer.batch_decode(noisy_input_ids, skip_special_tokens=True)):
        print(f"  {i+1}: {noisy_text}")
    
    # Use diffusion to denoise (simplified approach)
    with torch.no_grad():
        # Run a few denoising steps
        current_state = noisy_input_ids
        num_denoise_steps = 5
        
        for step in range(num_denoise_steps):
            # Get logits from model
            logits = denoise_fn(
                targets=current_state, 
                timestep=torch.tensor([100 - step * 20], device=device),  # Decreasing timestep
                attention_mask=attention_mask
            )
            
            # Sample from logits with temperature
            probs = torch.softmax(logits / 1.2, dim=-1)
            current_state = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(current_state.shape)
    
    # Decode final results
    denoised_texts = tokenizer.batch_decode(current_state, skip_special_tokens=True)
    
    return denoised_texts

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Model path - adjust this to your trained model
    model_path = r'G:\ML_Project_Sem6\Diffusion-BERT-main\model_name_bert-base-uncased_lr_0.0005_seed_42_numsteps_512_sample_Categorical_schedule_mutual_hybridlambda_0.01_wordfreqlambda_0.3_fromscratch_True_timestep_none_ckpts\epoch_4.pt'
    
    # Example input texts (for conditional generation experiments)
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world."
    ]

    try:
        # Load model and components
        print("Loading model and diffusion components...")
        model, tokenizer, diffusion_instance, diffusion = load_model_and_diffusion(model_path, device)
        
        # Create denoise function
        denoise_fn = create_denoise_fn(model, tokenizer, device)
        
        print("Model loaded successfully!")
        
        # Test 1: Unconditional generation
        print("\n" + "="*60)
        print("TEST 1: UNCONDITIONAL GENERATION")
        print("="*60)
        
        start_time = time.time()
        generated = generate_texts(
            model, tokenizer, diffusion_instance, diffusion, denoise_fn, 
            test_texts, device, 
            step_size=4, topk=50, topp=0.9, temperature=1.0, max_length=16
        )
        generation_time = time.time() - start_time
        
        print(f"Generation completed in {generation_time:.2f} seconds")
        print("Generated texts:")
        for i, text in enumerate(generated):
            print(f"  {i+1}: {text}")
        
        # Test 2: Conditional generation (experimental)
        print("\n" + "="*60)
        print("TEST 2: CONDITIONAL GENERATION (EXPERIMENTAL)")
        print("="*60)
        
        try:
            denoised = test_conditional_generation(
                model, tokenizer, diffusion_instance, diffusion, denoise_fn, test_texts, device
            )
            
            print("Results:")
            for i, (original, denoised_text) in enumerate(zip(test_texts, denoised)):
                print(f"  Original {i+1}: {original}")
                print(f"  Denoised {i+1}: {denoised_text}")
                print()
        except Exception as e:
            print(f"Conditional generation failed: {e}")
        
        # Test 3: Quality evaluation
        print("\n" + "="*60)
        print("TEST 3: QUALITY EVALUATION")
        print("="*60)
        
        # Generate more samples for evaluation
        larger_sample = generate_texts(
            model, tokenizer, diffusion_instance, diffusion, denoise_fn, 
            [""] * 5, device,  # Generate 5 samples
            step_size=4, topk=50, topp=0.9, temperature=1.0, max_length=20
        )
        
        print("Sample quality analysis:")
        mask_count = sum(1 for text in larger_sample if '[MASK]' in text)
        empty_count = sum(1 for text in larger_sample if len(text.strip()) == 0)
        valid_count = len(larger_sample) - mask_count - empty_count
        
        print(f"  Total samples: {len(larger_sample)}")
        print(f"  Valid samples: {valid_count}")
        print(f"  Samples with [MASK]: {mask_count}")
        print(f"  Empty samples: {empty_count}")
        print(f"  Success rate: {valid_count/len(larger_sample)*100:.1f}%")
        
        print("\nAll samples:")
        for i, text in enumerate(larger_sample, 1):
            print(f"  {i}: {text}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()