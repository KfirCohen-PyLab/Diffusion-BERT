"""
Interactive Text Generation with Trained Diffusion BERT
Use your trained model for real-time text generation and paraphrasing
"""

import torch
import argparse
from transformers import BertTokenizer, BertConfig
from models.modeling_bert import BertForMaskedLM
import diffusion_condition
from sample import Categorical
import os
import random

class DiffusionTextGenerator:
    def __init__(self, checkpoint_path, device):
        """Initialize the text generator with trained model"""
        self.device = device
        self.load_model(checkpoint_path)
        
    def load_model(self, checkpoint_path):
        """Load trained model from checkpoint"""
        print(f"🔄 Loading model from {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Initialize components
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        cfg = BertConfig.from_pretrained('bert-base-uncased')
        cfg.overall_timestep = 32
        
        self.model = BertForMaskedLM(cfg).to(self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        
        # Setup diffusion
        word_freq = torch.zeros(self.tokenizer.vocab_size)
        word_freq = (word_freq + 1).log() / (word_freq + 1).log().max()
        
        diffusion_schedule = diffusion_condition.create_discrete_diffusion_schedule('mutual', num_steps=32)
        self.diffusion = diffusion_condition.MaskDiffusion(
            dim=self.tokenizer.vocab_size,
            schedule=diffusion_schedule,
            tokenizer=self.tokenizer,
            sample_cls=Categorical(),
            word_freq=word_freq,
            word_freq_lambda=0.1,
            device=self.device
        )
        
        print("✅ Model loaded and ready for generation!")
    
    def fill_masks(self, text, mask_ratio=0.3, num_samples=3):
        """Fill masked parts of text using diffusion model"""
        print(f"🔧 Filling masks in: '{text}'")
        
        results = []
        
        # If no [MASK] tokens, create some
        if '[MASK]' not in text:
            tokens = self.tokenizer.encode(text, return_tensors='pt', max_length=64, truncation=True)
            tokens = tokens.to(self.device)
            
            # Randomly mask some tokens
            seq_len = tokens.size(1)
            n_mask = max(1, int(mask_ratio * seq_len))
            
            for sample_idx in range(num_samples):
                masked_tokens = tokens.clone()
                
                # Select random positions to mask (avoid special tokens)
                maskable_positions = list(range(1, seq_len - 1))  # Skip [CLS] and [SEP]
                if len(maskable_positions) >= n_mask:
                    mask_positions = random.sample(maskable_positions, n_mask)
                    for pos in mask_positions:
                        masked_tokens[0, pos] = self.tokenizer.mask_token_id
                
                # Decode to show what was masked
                masked_text = self.tokenizer.decode(masked_tokens[0], skip_special_tokens=False)
                print(f"  Sample {sample_idx + 1} masked: {masked_text}")
                
                # Generate
                filled_text = self._generate_from_masked(masked_tokens)
                results.append(filled_text)
                print(f"  Sample {sample_idx + 1} filled: {filled_text}")
        else:
            # Text already has [MASK] tokens
            for sample_idx in range(num_samples):
                tokens = self.tokenizer.encode(text, return_tensors='pt', max_length=64, truncation=True)
                tokens = tokens.to(self.device)
                
                filled_text = self._generate_from_masked(tokens)
                results.append(filled_text)
                print(f"  Sample {sample_idx + 1}: {filled_text}")
        
        return results
    
    def _generate_from_masked(self, masked_tokens):
        """Generate text from masked input using the diffusion model"""
        with torch.no_grad():
            # Create attention mask
            attention_mask = (masked_tokens != self.tokenizer.pad_token_id).long()
            
            # Find masked positions
            mask_positions = (masked_tokens == self.tokenizer.mask_token_id)
            
            if not mask_positions.any():
                return self.tokenizer.decode(masked_tokens[0], skip_special_tokens=True)
            
            # Use model to predict masked tokens
            outputs = self.model(input_ids=masked_tokens, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Sample from predictions (add some randomness)
            predicted_tokens = masked_tokens.clone()
            
            for pos in mask_positions.nonzero():
                token_logits = logits[pos[0], pos[1]]
                
                # Apply temperature sampling for variety
                temperature = 0.8
                token_probs = torch.softmax(token_logits / temperature, dim=-1)
                
                # Sample top-k tokens
                top_k = 10
                top_k_probs, top_k_indices = torch.topk(token_probs, top_k)
                
                # Sample from top-k
                sampled_idx = torch.multinomial(top_k_probs, 1)
                predicted_token = top_k_indices[sampled_idx]
                
                predicted_tokens[pos[0], pos[1]] = predicted_token
            
            return self.tokenizer.decode(predicted_tokens[0], skip_special_tokens=True)
    
    def paraphrase_text(self, text, strength=0.5, num_paraphrases=3):
        """Generate paraphrases by selectively corrupting and regenerating text"""
        print(f"🔄 Generating paraphrases for: '{text}'")
        
        paraphrases = []
        
        for i in range(num_paraphrases):
            try:
                # Tokenize
                tokens = self.tokenizer.encode(text, return_tensors='pt', max_length=64, truncation=True)
                tokens = tokens.to(self.device)
                
                # Create strategic corruption (focus on content words)
                seq_len = tokens.size(1)
                n_corrupt = max(1, int(strength * seq_len))
                
                # Apply diffusion corruption
                t = torch.tensor([self.diffusion.num_steps // 2], device=self.device)
                
                # Sample corruption using diffusion
                posterior_logits, corrupted_ids = self.diffusion.sample_and_compute_posterior_q(
                    tokens, t, return_logits=True, return_transition_probs=False
                )
                
                # Select which tokens to regenerate (strategic selection)
                target_mask = torch.zeros_like(tokens)
                
                # Focus on middle tokens (content words)
                if seq_len > 4:
                    start_idx = max(1, seq_len // 4)
                    end_idx = min(seq_len - 1, 3 * seq_len // 4)
                    
                    candidate_positions = list(range(start_idx, end_idx))
                    if len(candidate_positions) >= n_corrupt:
                        corrupt_positions = random.sample(candidate_positions, n_corrupt)
                        for pos in corrupt_positions:
                            target_mask[0, pos] = 1
                
                # Create mixed input (keep some original, corrupt some)
                mixed_input = torch.where(target_mask.bool(), corrupted_ids, tokens)
                
                # Generate paraphrase
                attention_mask = (tokens != self.tokenizer.pad_token_id).long()
                
                with torch.no_grad():
                    outputs = self.model(input_ids=mixed_input, attention_mask=attention_mask)
                    logits = outputs.logits
                    
                    # Apply temperature sampling for diversity
                    temperature = 0.9
                    probs = torch.softmax(logits / temperature, dim=-1)
                    
                    # Sample tokens where we have masks
                    result_tokens = mixed_input.clone()
                    for pos in target_mask.nonzero():
                        token_probs = probs[pos[0], pos[1]]
                        
                        # Top-p sampling for quality
                        top_p = 0.9
                        sorted_probs, sorted_indices = torch.sort(token_probs, descending=True)
                        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                        
                        # Find cutoff for top-p
                        cutoff_idx = (cumsum_probs <= top_p).sum().item()
                        cutoff_idx = max(1, cutoff_idx)  # At least 1 token
                        
                        # Sample from top-p
                        top_p_probs = sorted_probs[:cutoff_idx]
                        top_p_indices = sorted_indices[:cutoff_idx]
                        
                        sampled_idx = torch.multinomial(top_p_probs, 1)
                        result_tokens[pos[0], pos[1]] = top_p_indices[sampled_idx]
                
                paraphrase = self.tokenizer.decode(result_tokens[0], skip_special_tokens=True)
                paraphrases.append(paraphrase)
                print(f"  Paraphrase {i + 1}: {paraphrase}")
                
            except Exception as e:
                print(f"  ❌ Error generating paraphrase {i + 1}: {e}")
                paraphrases.append(f"Error: {e}")
        
        return paraphrases
    
    def complete_text(self, text_start, max_length=50, num_completions=3):
        """Complete partial text using the model"""
        print(f"✏️ Completing text: '{text_start}'...")
        
        completions = []
        
        for i in range(num_completions):
            try:
                # Add [MASK] tokens to the end for completion
                extended_text = text_start + " [MASK]" * (max_length // 4)
                
                tokens = self.tokenizer.encode(extended_text, return_tensors='pt', max_length=max_length, truncation=True)
                tokens = tokens.to(self.device)
                
                # Only generate for the [MASK] tokens (the completion part)
                completion = self._generate_from_masked(tokens)
                
                # Extract just the new part
                if completion.startswith(text_start):
                    new_part = completion[len(text_start):].strip()
                    full_completion = text_start + " " + new_part
                else:
                    full_completion = completion
                
                completions.append(full_completion)
                print(f"  Completion {i + 1}: {full_completion}")
                
            except Exception as e:
                print(f"  ❌ Error generating completion {i + 1}: {e}")
                completions.append(f"Error: {e}")
        
        return completions

def find_model_checkpoint():
    """Find available model checkpoints"""
    current_dir = "."
    model_dirs = [d for d in os.listdir(current_dir) if d.startswith("model_name_") and os.path.isdir(d)]
    
    if not model_dirs:
        return None
    
    print("📁 Available model directories:")
    for i, dir_name in enumerate(model_dirs, 1):
        print(f"  {i}. {dir_name}")
    
    while True:
        try:
            choice = int(input(f"Select model directory (1-{len(model_dirs)}): ")) - 1
            if 0 <= choice < len(model_dirs):
                selected_dir = model_dirs[choice]
                
                # Look for best checkpoint
                checkpoint_files = ['best_model.pt', 'final_model.pt']
                for checkpoint_file in checkpoint_files:
                    checkpoint_path = os.path.join(selected_dir, checkpoint_file)
                    if os.path.exists(checkpoint_path):
                        return checkpoint_path
                
                # Look for epoch checkpoints
                epoch_files = [f for f in os.listdir(selected_dir) if f.startswith('epoch_') and f.endswith('.pt')]
                if epoch_files:
                    latest_epoch = max([int(f.split('_')[1].split('.')[0]) for f in epoch_files])
                    return os.path.join(selected_dir, f'epoch_{latest_epoch}.pt')
                
                print(f"❌ No valid checkpoints found in {selected_dir}")
                return None
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number.")

def interactive_mode(generator):
    """Interactive text generation mode"""
    print("\n🎮 INTERACTIVE GENERATION MODE")
    print("="*50)
    print("Commands:")
    print("  'fill <text>'      - Fill randomly masked parts of text")
    print("  'para <text>'      - Generate paraphrases")
    print("  'complete <text>'  - Complete partial text")
    print("  'mask <text>'      - Fill [MASK] tokens in text")
    print("  'help'            - Show this help")
    print("  'quit'            - Exit")
    print()
    
    while True:
        try:
            user_input = input("🤖 Enter command: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'help':
                print("\nCommands:")
                print("  fill <text>      - Fill randomly masked parts")
                print("  para <text>      - Generate paraphrases") 
                print("  complete <text>  - Complete partial text")
                print("  mask <text>      - Fill existing [MASK] tokens")
                print("  quit            - Exit")
                continue
            
            # Parse command
            parts = user_input.split(' ', 1)
            if len(parts) != 2:
                print("❌ Invalid command format. Use: <command> <text>")
                continue
            
            command, text = parts
            command = command.lower()
            
            print(f"\n🎯 Processing: {command} '{text}'")
            print("-" * 40)
            
            if command == 'fill':
                generator.fill_masks(text, mask_ratio=0.3, num_samples=3)
                
            elif command == 'para':
                generator.paraphrase_text(text, strength=0.5, num_paraphrases=3)
                
            elif command == 'complete':
                generator.complete_text(text, max_length=64, num_completions=3)
                
            elif command == 'mask':
                if '[MASK]' not in text:
                    print("❌ No [MASK] tokens found in text")
                else:
                    generator.fill_masks(text, num_samples=3)
                    
            else:
                print(f"❌ Unknown command: {command}")
                print("Use 'help' to see available commands")
            
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Interactive text generation with trained Diffusion BERT")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    
    args = parser.parse_args()
    
    print("🎭 DIFFUSION BERT TEXT GENERATOR")
    print("="*50)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = find_model_checkpoint()
        
    if not checkpoint_path:
        print("❌ No checkpoint found. Please specify --checkpoint path")
        return
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # Initialize generator
    try:
        generator = DiffusionTextGenerator(checkpoint_path, device)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Quick demo
    print("\n🎯 QUICK DEMO")
    print("-" * 30)
    demo_text = "The weather is really nice today"
    print(f"Demo text: '{demo_text}'")
    generator.fill_masks(demo_text, mask_ratio=0.4, num_samples=2)
    
    # Interactive mode
    interactive_mode(generator)

if __name__ == "__main__":
    main() 