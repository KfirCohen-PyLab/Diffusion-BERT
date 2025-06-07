"""
Diffusion Noise + BERT Denoise Demo
Exactly what you want: Use diffusion to add noise, then BERT to reconstruct
"""

import torch
import argparse
from transformers import BertTokenizer, BertConfig
from models.modeling_bert import BertForMaskedLM
import diffusion_condition
from sample import Categorical
import os
import random

class DiffusionBERTDenoiser:
    def __init__(self, checkpoint_path, device):
        """Initialize the denoiser with trained model"""
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
        
        # Setup diffusion (for noise generation)
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
        
        print("✅ Model loaded and ready for diffusion noise + BERT denoise!")
    
    def diffusion_noise_bert_denoise(self, text, timestep=16, num_samples=3, show_process=True):
        """
        Your exact workflow: 
        1. Input clean sentence
        2. Add noise using DIFFUSION method
        3. Use BERT to denoise/reconstruct
        4. Return reconstructed sentences
        """
        print(f"🎯 DIFFUSION NOISE → BERT DENOISE")
        print(f"Original text: '{text}'")
        print(f"Noise timestep: {timestep}/{self.diffusion.num_steps}")
        print("-" * 50)
        
        # Tokenize input
        tokens = self.tokenizer.encode(text, return_tensors='pt', max_length=64, truncation=True)
        tokens = tokens.to(self.device)
        
        results = []
        
        for sample_idx in range(num_samples):
            print(f"\n📋 Sample {sample_idx + 1}:")
            
            # STEP 1: Apply DIFFUSION NOISE
            t = torch.tensor([timestep], device=self.device)
            
            # Use diffusion to corrupt tokens
            posterior_logits, corrupted_tokens = self.diffusion.sample_and_compute_posterior_q(
                tokens, t, return_logits=True, return_transition_probs=False
            )
            
            # Show corrupted version
            corrupted_text = self.tokenizer.decode(corrupted_tokens[0], skip_special_tokens=True)
            if show_process:
                print(f"  🔀 After diffusion noise: '{corrupted_text}'")
            
            # STEP 2: Use BERT to DENOISE
            reconstructed_tokens = self._bert_denoise(corrupted_tokens)
            reconstructed_text = self.tokenizer.decode(reconstructed_tokens[0], skip_special_tokens=True)
            
            if show_process:
                print(f"  🔧 After BERT denoise: '{reconstructed_text}'")
                
                # Calculate similarity
                original_words = set(text.lower().split())
                reconstructed_words = set(reconstructed_text.lower().split())
                similarity = len(original_words & reconstructed_words) / max(len(original_words), 1)
                print(f"  📊 Similarity: {similarity:.1%}")
            
            results.append({
                'original': text,
                'corrupted': corrupted_text, 
                'reconstructed': reconstructed_text,
                'timestep': timestep
            })
        
        return results
    
    def _bert_denoise(self, corrupted_tokens):
        """Use BERT model to denoise corrupted tokens"""
        with torch.no_grad():
            # Create attention mask
            attention_mask = (corrupted_tokens != self.tokenizer.pad_token_id).long()
            
            # BERT forward pass for reconstruction
            outputs = self.model(input_ids=corrupted_tokens, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Get most likely tokens (you can add sampling here)
            predicted_tokens = logits.argmax(dim=-1)
            
            return predicted_tokens
    
    def compare_noise_levels(self, text, timesteps=[4, 8, 16, 24, 31]):
        """Compare reconstruction quality at different noise levels"""
        print(f"\n🔬 NOISE LEVEL COMPARISON")
        print(f"Original: '{text}'")
        print("=" * 60)
        
        for timestep in timesteps:
            print(f"\n⏰ Timestep {timestep}/{self.diffusion.num_steps}")
            
            # Single sample for comparison
            tokens = self.tokenizer.encode(text, return_tensors='pt', max_length=64, truncation=True)
            tokens = tokens.to(self.device)
            
            # Apply diffusion noise
            t = torch.tensor([timestep], device=self.device)
            posterior_logits, corrupted_tokens = self.diffusion.sample_and_compute_posterior_q(
                tokens, t, return_logits=True, return_transition_probs=False
            )
            
            # BERT denoise
            reconstructed_tokens = self._bert_denoise(corrupted_tokens)
            
            # Decode results
            corrupted_text = self.tokenizer.decode(corrupted_tokens[0], skip_special_tokens=True)
            reconstructed_text = self.tokenizer.decode(reconstructed_tokens[0], skip_special_tokens=True)
            
            # Calculate metrics
            original_words = set(text.lower().split())
            reconstructed_words = set(reconstructed_text.lower().split())
            similarity = len(original_words & reconstructed_words) / max(len(original_words), 1)
            
            print(f"  Corrupted:     '{corrupted_text}'")
            print(f"  Reconstructed: '{reconstructed_text}'")
            print(f"  Similarity:    {similarity:.1%} {'✅' if similarity > 0.7 else '⚠️' if similarity > 0.4 else '❌'}")
    
    def batch_denoise(self, sentences, timestep=16):
        """Denoise multiple sentences at once"""
        print(f"\n📦 BATCH DIFFUSION NOISE → BERT DENOISE")
        print(f"Processing {len(sentences)} sentences with timestep {timestep}")
        print("-" * 60)
        
        all_results = []
        
        for i, text in enumerate(sentences):
            print(f"\n{i+1}. Processing: '{text}'")
            
            # Apply your workflow
            results = self.diffusion_noise_bert_denoise(text, timestep=timestep, num_samples=1, show_process=False)
            
            result = results[0]
            similarity = self._calculate_similarity(result['original'], result['reconstructed'])
            
            print(f"   Reconstructed: '{result['reconstructed']}'")
            print(f"   Quality: {similarity:.1%} {'✅' if similarity > 0.7 else '⚠️' if similarity > 0.4 else '❌'}")
            
            all_results.append(result)
        
        return all_results
    
    def _calculate_similarity(self, text1, text2):
        """Calculate word-level similarity between two texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        return len(words1 & words2) / max(len(words1), 1)

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
                
                print(f"❌ No valid checkpoints found in {selected_dir}")
                return None
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number.")

def interactive_demo(denoiser):
    """Interactive demo of diffusion noise + BERT denoise"""
    print("\n🎮 INTERACTIVE DIFFUSION NOISE + BERT DENOISE")
    print("=" * 50)
    print("Commands:")
    print("  'denoise <text>'           - Apply diffusion noise then BERT denoise")
    print("  'compare <text>'           - Compare different noise levels")
    print("  'batch <text1> | <text2>'  - Process multiple sentences")
    print("  'help'                     - Show this help")
    print("  'quit'                     - Exit")
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
                print("  denoise <text>           - Your exact workflow")
                print("  compare <text>           - Test different noise levels")
                print("  batch <text1> | <text2> - Multiple sentences")
                print("  quit                     - Exit")
                continue
            
            # Parse command
            parts = user_input.split(' ', 1)
            if len(parts) != 2:
                print("❌ Invalid format. Use: <command> <text>")
                continue
                
            command, text = parts
            command = command.lower()
            
            if command == 'denoise':
                # Your exact workflow
                denoiser.diffusion_noise_bert_denoise(text, timestep=16, num_samples=3)
                
            elif command == 'compare':
                # Compare noise levels
                denoiser.compare_noise_levels(text)
                
            elif command == 'batch':
                # Multiple sentences
                sentences = [s.strip() for s in text.split('|')]
                denoiser.batch_denoise(sentences)
                
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
    parser = argparse.ArgumentParser(description="Diffusion Noise + BERT Denoise Demo")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    
    args = parser.parse_args()
    
    print("🎯 DIFFUSION NOISE + BERT DENOISE DEMO")
    print("=" * 50)
    print("Your exact workflow: Clean Text → Diffusion Noise → BERT Denoise")
    print()
    
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
    
    # Initialize denoiser
    try:
        denoiser = DiffusionBERTDenoiser(checkpoint_path, device)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Quick demo of your exact workflow
    print("\n🎯 QUICK DEMO OF YOUR WORKFLOW")
    print("-" * 40)
    demo_text = "The weather is really nice today"
    denoiser.diffusion_noise_bert_denoise(demo_text, timestep=16, num_samples=2)
    
    # Compare different noise levels
    print("\n🔬 NOISE LEVEL COMPARISON")
    print("-" * 40)
    denoiser.compare_noise_levels(demo_text, timesteps=[8, 16, 24])
    
    # Interactive mode
    interactive_demo(denoiser)

if __name__ == "__main__":
    main() 