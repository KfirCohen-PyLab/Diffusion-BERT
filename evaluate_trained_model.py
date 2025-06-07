"""
Comprehensive Model Evaluation Script
Tests your trained Diffusion BERT model on various tasks
"""

import torch
import functools
from transformers import BertTokenizer, BertConfig
from models.modeling_bert import BertForMaskedLM
import diffusion_condition
from sample import Categorical
import os
import argparse

def load_trained_model(checkpoint_path, device):
    """Load a trained model from checkpoint"""
    print(f"🔄 Loading model from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize components
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    cfg = BertConfig.from_pretrained('bert-base-uncased')
    cfg.overall_timestep = 32  # Default from training
    
    model = BertForMaskedLM(cfg).to(device)
    model.load_state_dict(checkpoint['model'])
    
    # Setup diffusion
    word_freq = torch.zeros(tokenizer.vocab_size)
    word_freq = (word_freq + 1).log() / (word_freq + 1).log().max()
    
    diffusion_schedule = diffusion_condition.create_discrete_diffusion_schedule('mutual', num_steps=32)
    diffusion_instance = diffusion_condition.MaskDiffusion(
        dim=tokenizer.vocab_size,
        schedule=diffusion_schedule,
        tokenizer=tokenizer,
        sample_cls=Categorical(),
        word_freq=word_freq,
        word_freq_lambda=0.1,
        device=device
    )
    
    print("✅ Model loaded successfully!")
    return model, tokenizer, diffusion_instance

def test_custom_examples(model, tokenizer, diffusion_instance, device):
    """Test model on custom examples"""
    print("\n🧪 TESTING CUSTOM EXAMPLES")
    print("="*50)
    
    # Custom test examples (you can modify these)
    test_examples = [
        "How can I improve my English speaking skills?",
        "What is the best way to learn programming?", 
        "Can you recommend a good restaurant in New York?",
        "How do I fix my broken laptop?",
        "What are the benefits of regular exercise?",
        "How can I save money for vacation?",
        "What is artificial intelligence?",
        "How do I cook a perfect steak?"
    ]
    
    model.eval()
    
    def denoise_fn(input_ids, corrupted_input_ids, timestep, attention_mask, target_mask):
        new_input_ids = torch.where(target_mask.bool(), corrupted_input_ids, input_ids)
        return model(input_ids=new_input_ids, attention_mask=attention_mask)['logits']
    
    with torch.no_grad():
        for i, text in enumerate(test_examples):
            print(f"\n📝 Example {i+1}: {text}")
            print("-" * 40)
            
            # Tokenize
            tokens = tokenizer.encode(text, return_tensors='pt', padding=True, truncation=True, max_length=64)
            tokens = tokens.to(device)
            attention_mask = (tokens != tokenizer.pad_token_id).long()
            
            # Create different corruption scenarios
            for corruption_ratio in [0.2, 0.5, 0.8]:
                try:
                    # Randomly select tokens to corrupt
                    seq_len = tokens.size(1)
                    n_corrupt = int(corruption_ratio * seq_len)
                    
                    # Create target mask (what to regenerate)
                    target_mask = torch.zeros_like(tokens)
                    if n_corrupt > 0:
                        corrupt_indices = torch.randperm(seq_len)[:n_corrupt]
                        target_mask[0, corrupt_indices] = 1
                    
                    # Create corrupted version
                    corrupted_tokens = tokens.clone()
                    corrupted_tokens[target_mask.bool()] = tokenizer.mask_token_id
                    
                    # Mix source and corrupted parts for conditional input
                    conditional_input = torch.where(target_mask.bool(), corrupted_tokens, tokens)
                    
                    # Decode corrupted version
                    corrupted_text = tokenizer.decode(conditional_input[0], skip_special_tokens=True)
                    
                    # Try to denoise
                    logits = denoise_fn(tokens, corrupted_tokens, torch.tensor([16], device=device), attention_mask, target_mask)
                    predicted_tokens = logits.argmax(dim=-1)
                    denoised_tokens = torch.where(target_mask.bool(), predicted_tokens, tokens)
                    denoised_text = tokenizer.decode(denoised_tokens[0], skip_special_tokens=True)
                    
                    # Calculate similarity
                    original_words = set(text.lower().split())
                    denoised_words = set(denoised_text.lower().split())
                    similarity = len(original_words & denoised_words) / max(len(original_words), 1)
                    
                    print(f"  Corruption {corruption_ratio:.0%}: {corrupted_text}")
                    print(f"  Reconstructed: {denoised_text}")
                    print(f"  Similarity: {similarity:.1%} {'✅' if similarity > 0.7 else '⚠️' if similarity > 0.4 else '❌'}")
                    
                except Exception as e:
                    print(f"  ❌ Error with {corruption_ratio:.0%} corruption: {e}")

def test_paraphrase_generation(model, tokenizer, diffusion_instance, device):
    """Test paraphrase generation capabilities"""
    print("\n🔄 TESTING PARAPHRASE GENERATION")
    print("="*50)
    
    # Test sentences for paraphrasing
    sentences = [
        "The weather is really nice today.",
        "I need to go to the store.",
        "This movie was very entertaining.",
        "Learning new skills is important.",
        "Technology is changing rapidly."
    ]
    
    model.eval()
    
    def generate_paraphrase(text, corruption_strength=0.6):
        """Generate a paraphrase by corrupting and regenerating parts of the text"""
        tokens = tokenizer.encode(text, return_tensors='pt', max_length=32, truncation=True)
        tokens = tokens.to(device)
        
        # Create random corruption mask
        seq_len = tokens.size(1)
        n_corrupt = int(corruption_strength * seq_len)
        
        target_mask = torch.zeros_like(tokens)
        if n_corrupt > 0:
            # Focus corruption on content words (avoid corrupting first/last tokens)
            middle_indices = torch.arange(1, seq_len-1)
            if len(middle_indices) > 0:
                corrupt_indices = middle_indices[torch.randperm(len(middle_indices))[:min(n_corrupt, len(middle_indices))]]
                target_mask[0, corrupt_indices] = 1
        
        # Apply diffusion process
        try:
            t = torch.tensor([diffusion_instance.num_steps // 2], device=device)  # Mid-level noise
            
            # Sample corruption
            posterior_logits, corrupted_ids = diffusion_instance.sample_and_compute_posterior_q(
                tokens, t, return_logits=True, return_transition_probs=False
            )
            
            # Create conditional input
            conditional_input = torch.where(target_mask.bool(), corrupted_ids, tokens)
            attention_mask = (tokens != tokenizer.pad_token_id).long()
            
            # Denoise
            with torch.no_grad():
                new_input_ids = torch.where(target_mask.bool(), corrupted_ids, tokens)
                logits = model(input_ids=new_input_ids, attention_mask=attention_mask)['logits']
                predicted_ids = logits.argmax(dim=-1)
                result = torch.where(target_mask.bool(), predicted_ids, tokens)
                
            return tokenizer.decode(result[0], skip_special_tokens=True)
            
        except Exception as e:
            return f"Error: {e}"
    
    for sentence in sentences:
        print(f"\n📝 Original: {sentence}")
        
        # Generate multiple paraphrases
        for i in range(3):
            paraphrase = generate_paraphrase(sentence, corruption_strength=0.4 + i * 0.2)
            print(f"  Paraphrase {i+1}: {paraphrase}")

def benchmark_model_quality(model, tokenizer, diffusion_instance, device):
    """Benchmark model quality with standardized tests"""
    print("\n📊 BENCHMARKING MODEL QUALITY")
    print("="*50)
    
    # Standard test cases
    test_cases = [
        {"text": "The quick brown fox jumps over the lazy dog", "expected_type": "complete_sentence"},
        {"text": "Artificial intelligence is transforming modern technology", "expected_type": "technical_content"},
        {"text": "I love eating pizza with my friends", "expected_type": "casual_content"},
        {"text": "The research shows significant improvements in accuracy", "expected_type": "academic_content"},
        {"text": "Please call me when you arrive at the airport", "expected_type": "instruction"}
    ]
    
    total_score = 0
    total_tests = 0
    
    model.eval()
    
    for test_case in test_cases:
        text = test_case["text"]
        print(f"\n🔍 Testing: {text}")
        
        try:
            # Tokenize
            tokens = tokenizer.encode(text, return_tensors='pt', max_length=32, truncation=True)
            tokens = tokens.to(device)
            
            # Test different corruption levels
            corruption_scores = []
            
            for corruption_ratio in [0.3, 0.5, 0.7]:
                # Create corruption
                seq_len = tokens.size(1)
                n_corrupt = int(corruption_ratio * seq_len)
                
                target_mask = torch.zeros_like(tokens)
                if n_corrupt > 0:
                    corrupt_indices = torch.randperm(seq_len)[:n_corrupt]
                    target_mask[0, corrupt_indices] = 1
                
                corrupted_tokens = tokens.clone()
                corrupted_tokens[target_mask.bool()] = tokenizer.mask_token_id
                
                # Reconstruct
                attention_mask = (tokens != tokenizer.pad_token_id).long()
                with torch.no_grad():
                    new_input_ids = torch.where(target_mask.bool(), corrupted_tokens, tokens)
                    logits = model(input_ids=new_input_ids, attention_mask=attention_mask)['logits']
                    predicted_tokens = logits.argmax(dim=-1)
                    result = torch.where(target_mask.bool(), predicted_tokens, tokens)
                
                # Calculate score
                original_text = text.lower()
                reconstructed_text = tokenizer.decode(result[0], skip_special_tokens=True).lower()
                
                # Word-level similarity
                original_words = set(original_text.split())
                reconstructed_words = set(reconstructed_text.split())
                word_similarity = len(original_words & reconstructed_words) / max(len(original_words), 1)
                
                corruption_scores.append(word_similarity)
                print(f"  {corruption_ratio:.0%} corruption: {word_similarity:.1%} similarity")
            
            # Overall score for this test case
            avg_score = sum(corruption_scores) / len(corruption_scores)
            total_score += avg_score
            total_tests += 1
            
            print(f"  📈 Average score: {avg_score:.1%}")
            
        except Exception as e:
            print(f"  ❌ Test failed: {e}")
    
    # Final benchmark score
    if total_tests > 0:
        final_score = total_score / total_tests
        print(f"\n🏆 OVERALL BENCHMARK SCORE: {final_score:.1%}")
        
        if final_score > 0.8:
            print("🎉 Excellent! Your model performs very well.")
        elif final_score > 0.6:
            print("👍 Good! Your model shows solid performance.")
        elif final_score > 0.4:
            print("⚠️ Fair. Your model needs more training.")
        else:
            print("❌ Poor. Consider retraining with more data/epochs.")
    
    return final_score if total_tests > 0 else 0

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Diffusion BERT model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    try:
        model, tokenizer, diffusion_instance = load_trained_model(args.checkpoint, device)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Run tests
    print("\n🎯 STARTING COMPREHENSIVE MODEL EVALUATION")
    print("="*60)
    
    # Test 1: Custom examples
    test_custom_examples(model, tokenizer, diffusion_instance, device)
    
    # Test 2: Paraphrase generation
    test_paraphrase_generation(model, tokenizer, diffusion_instance, device)
    
    # Test 3: Benchmark quality
    benchmark_score = benchmark_model_quality(model, tokenizer, diffusion_instance, device)
    
    print(f"\n🎉 EVALUATION COMPLETE!")
    print(f"📊 Final Benchmark Score: {benchmark_score:.1%}")

if __name__ == "__main__":
    main() 