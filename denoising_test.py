import torch
import numpy as np
from transformers import BertTokenizer, BertConfig
from sample import Categorical

def load_model_and_diffusion():
    """Load model, tokenizer, and diffusion components"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Import model and diffusion
    from models.modeling_bert import BertForMaskedLM
    import diffusion_word_freq as diffusion
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    sample_cls = Categorical(device=device)
    
    # Create diffusion schedule
    diffusion_schedule = diffusion.create_discrete_diffusion_schedule(
        'mutual', num_steps=512, device=device
    )
    
    # Load word frequency
    word_freq_path = r'G:\ML_Project_Sem6\Diffusion-BERT-main\word_freq\bert-base-uncased_lm1b.pt'
    try:
        word_freq = torch.load(word_freq_path).to(device)
        word_freq = (word_freq + 1).log() / (word_freq + 1).log().max()
        print(f"Loaded word frequency: {word_freq.shape}")
    except FileNotFoundError:
        print("Warning: Word frequency file not found")
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
    cfg = BertConfig.from_pretrained('bert-base-uncased')
    cfg.overall_timestep = diffusion_instance.num_steps
    model = BertForMaskedLM(cfg)
    
    # Load checkpoint
    model_path = r'G:\ML_Project_Sem6\Diffusion-BERT-main\model_name_bert-base-uncased_lr_0.0005_seed_42_numsteps_512_sample_Categorical_schedule_mutual_hybridlambda_0.01_wordfreqlambda_0.3_fromscratch_True_timestep_none_ckpts\epoch_4.pt'
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model, tokenizer, diffusion_instance, device

def create_denoise_fn(model, tokenizer, device):
    """Create the denoise function"""
    cls = torch.full((1, 1), tokenizer.cls_token_id, device=device)
    sep = torch.full((1, 1), tokenizer.sep_token_id, device=device)
    
    def denoise_fn(targets, timestep, attention_mask):
        bsz = targets.size(0)
        targets = targets.to(device)
        attention_mask = attention_mask.to(device)
        
        if isinstance(timestep, torch.Tensor):
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
        
        return logits
    
    return denoise_fn

def test_denoising():
    """Test the denoising capability of the model"""
    print("Loading model...")
    model, tokenizer, diffusion_instance, device = load_model_and_diffusion()
    denoise_fn = create_denoise_fn(model, tokenizer, device)
    
    # Test sentences
    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Python is a powerful programming language.",
        "The weather today is sunny and warm."
    ]
    
    print("\n" + "="*80)
    print("DENOISING TEST")
    print("="*80)
    
    for i, sentence in enumerate(test_sentences):
        print(f"\nTest {i+1}: {sentence}")
        print("-" * 60)
        
        # Tokenize the sentence
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=20)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        
        print(f"Original tokens: {tokenizer.convert_ids_to_tokens(input_ids[0])}")
        
        # Create different noise levels
        noise_levels = [0.2, 0.4, 0.6]
        
        for noise_ratio in noise_levels:
            # Create noisy version by masking tokens
            noisy_input_ids = input_ids.clone()
            seq_len = attention_mask.sum().item()  # Get actual sequence length
            
            # Randomly mask tokens (except CLS and SEP)
            mask_positions = torch.rand(1, seq_len-2) < noise_ratio
            mask_indices = torch.where(mask_positions)[1] + 1  # +1 to skip CLS token
            
            if len(mask_indices) > 0:
                noisy_input_ids[0, mask_indices] = tokenizer.mask_token_id
            
            noisy_text = tokenizer.decode(noisy_input_ids[0], skip_special_tokens=True)
            print(f"\nNoise ratio {noise_ratio:.1f}: {noisy_text}")
            
            # Try simple model-based denoising (not full diffusion)
            try:
                with torch.no_grad():
                    # Get model predictions for masked tokens
                    logits = denoise_fn(
                        targets=noisy_input_ids,
                        timestep='none',  # Use direct model inference
                        attention_mask=attention_mask
                    )
                    
                    # Replace only the masked tokens with predictions
                    predicted_ids = logits.argmax(dim=-1)
                    denoised_ids = noisy_input_ids.clone()
                    
                    # Only replace [MASK] tokens
                    mask_token_positions = (noisy_input_ids == tokenizer.mask_token_id)
                    denoised_ids[mask_token_positions] = predicted_ids[mask_token_positions]
                    
                    # Clamp to valid token range
                    denoised_ids = torch.clamp(denoised_ids, 0, tokenizer.vocab_size - 1)
                    
                    denoised_text = tokenizer.decode(denoised_ids[0], skip_special_tokens=True)
                    print(f"  -> Denoised: {denoised_text}")
                    
                    # Check if denoising was successful
                    original_words = sentence.lower().split()
                    denoised_words = denoised_text.lower().split()
                    
                    # Simple word overlap metric
                    common_words = set(original_words) & set(denoised_words)
                    success_rate = len(common_words) / len(original_words) if original_words else 0
                    print(f"  -> Word overlap: {success_rate:.2f} ({len(common_words)}/{len(original_words)} words)")
                    
            except Exception as e:
                print(f"  -> Error: {str(e)}")

def test_unconditional_generation():
    """Test unconditional generation to see token quality"""
    print("\n" + "="*80)
    print("UNCONDITIONAL GENERATION TEST")
    print("="*80)
    
    model, tokenizer, diffusion_instance, device = load_model_and_diffusion()
    
    # Generate a few tokens using simple sampling
    with torch.no_grad():
        # Start with random tokens
        seq_len = 10
        random_ids = torch.randint(1000, 5000, (1, seq_len), device=device)  # Sample from common token range
        
        print(f"Random token IDs: {random_ids[0].cpu().tolist()}")
        random_text = tokenizer.decode(random_ids[0], skip_special_tokens=True)
        print(f"Random text: {random_text}")
        
        # Try to generate tokens within vocabulary range
        vocab_size = tokenizer.vocab_size
        print(f"Tokenizer vocab size: {vocab_size}")
        
        # Check some token ranges
        print("\nToken sampling from different ranges:")
        for start_range in [0, 1000, 2000, 3000, 5000]:
            end_range = min(start_range + 100, vocab_size - 1)
            sample_ids = torch.randint(start_range, end_range, (10,))
            sample_tokens = [tokenizer.decode([token_id]) for token_id in sample_ids]
            print(f"Range {start_range}-{end_range}: {sample_tokens[:5]}")

if __name__ == "__main__":
    try:
        test_denoising()
        test_unconditional_generation()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc() 