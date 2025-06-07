"""
Simple Mask + BERT Denoise Demo
Random masking + excellent BERT reconstruction
Focus on denoising quality, not complex noise methods
"""

import torch
import argparse
from transformers import BertTokenizer, BertConfig
from models.modeling_bert import BertForMaskedLM
import os
import random
import numpy as np

class SimpleMaskDenoiser:
    def __init__(self, checkpoint_path, device):
        """Initialize the denoiser with trained model"""
        self.device = device
        self.load_model(checkpoint_path)
        
    def load_model(self, checkpoint_path):
        """Load trained model from checkpoint"""
        print(f"🔄 Loading model from {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Determine model type from checkpoint path
        if 'roberta' in checkpoint_path.lower():
            from transformers import RobertaTokenizer, RobertaConfig
            from models.modeling_roberta import RobertaForMaskedLM
            
            print("🤖 Detected RoBERTa model")
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            cfg = RobertaConfig.from_pretrained('roberta-base')
            cfg.overall_timestep = 32
            self.model = RobertaForMaskedLM(cfg).to(self.device)
        else:
            print("🤖 Detected BERT model")
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            cfg = BertConfig.from_pretrained('bert-base-uncased')
            cfg.overall_timestep = 32
            self.model = BertForMaskedLM(cfg).to(self.device)
        
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        
        print("✅ Model loaded and ready for mask + denoise!")
    
    def mask_and_denoise(self, text, mask_ratio=0.3, num_samples=3, show_process=True):
        """
        Simple workflow:
        1. Input clean sentence
        2. Randomly mask tokens
        3. Use BERT to reconstruct
        4. Return reconstructed sentences
        """
        if show_process:
            print(f"🎯 RANDOM MASK → BERT DENOISE")
            print(f"Original text: '{text}'")
            print(f"Mask ratio: {mask_ratio:.0%}")
            print("-" * 50)
        
        # Tokenize input
        tokens = self.tokenizer.encode(text, return_tensors='pt', max_length=64, truncation=True)
        tokens = tokens.to(self.device)
        
        results = []
        
        for sample_idx in range(num_samples):
            if show_process:
                print(f"\n📋 Sample {sample_idx + 1}:")
            
            # STEP 1: Random masking
            masked_tokens = self._apply_random_mask(tokens.clone(), mask_ratio)
            
            # Show masked version
            masked_text = self.tokenizer.decode(masked_tokens[0], skip_special_tokens=True)
            if show_process:
                print(f"  🎭 Masked: '{masked_text}'")
            
            # STEP 2: BERT reconstruction
            reconstructed_tokens = self._bert_denoise(masked_tokens)
            reconstructed_text = self.tokenizer.decode(reconstructed_tokens[0], skip_special_tokens=True)
            
            if show_process:
                print(f"  🔧 Reconstructed: '{reconstructed_text}'")
                
                # Calculate similarity
                similarity = self._calculate_similarity(text, reconstructed_text)
                print(f"  📊 Similarity: {similarity:.1%} {'✅' if similarity > 0.7 else '⚠️' if similarity > 0.4 else '❌'}")
            
            results.append({
                'original': text,
                'masked': masked_text,
                'reconstructed': reconstructed_text,
                'similarity': self._calculate_similarity(text, reconstructed_text)
            })
        
        return results
    
    def _apply_random_mask(self, tokens, mask_ratio):
        """Apply random masking to tokens"""
        seq_len = tokens.size(1)
        n_mask = max(1, int(mask_ratio * seq_len))
        
        # Don't mask special tokens (works for both BERT and RoBERTa)
        maskable_positions = list(range(1, seq_len - 1))
        
        if len(maskable_positions) >= n_mask:
            mask_positions = random.sample(maskable_positions, n_mask)
            for pos in mask_positions:
                tokens[0, pos] = self.tokenizer.mask_token_id
        
        return tokens
    
    def _bert_denoise(self, masked_tokens, use_sampling=True, temperature=0.8):
        """Use BERT model to denoise masked tokens with better sampling"""
        with torch.no_grad():
            # Create attention mask
            attention_mask = (masked_tokens != self.tokenizer.pad_token_id).long()
            
            # BERT forward pass
            outputs = self.model(input_ids=masked_tokens, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Find masked positions
            mask_positions = (masked_tokens == self.tokenizer.mask_token_id)
            
            if not mask_positions.any():
                return masked_tokens
            
            # Reconstruct tokens
            reconstructed_tokens = masked_tokens.clone()
            
            if use_sampling:
                # Use temperature sampling for better results
                for pos in mask_positions.nonzero():
                    token_logits = logits[pos[0], pos[1]]
                    
                    # Apply temperature
                    scaled_logits = token_logits / temperature
                    probs = torch.softmax(scaled_logits, dim=-1)
                    
                    # Top-k sampling for quality
                    top_k = 10
                    top_k_probs, top_k_indices = torch.topk(probs, top_k)
                    
                    # Sample from top-k
                    sampled_idx = torch.multinomial(top_k_probs, 1)
                    predicted_token = top_k_indices[sampled_idx]
                    
                    reconstructed_tokens[pos[0], pos[1]] = predicted_token
            else:
                # Simple argmax
                predicted_tokens = logits.argmax(dim=-1)
                reconstructed_tokens = torch.where(mask_positions, predicted_tokens, masked_tokens)
            
            return reconstructed_tokens
    
    def compare_mask_ratios(self, text, mask_ratios=[0.1, 0.2, 0.3, 0.5, 0.7]):
        """Compare reconstruction quality at different mask ratios"""
        print(f"\n🔬 MASK RATIO COMPARISON")
        print(f"Original: '{text}'")
        print("=" * 60)
        
        for mask_ratio in mask_ratios:
            print(f"\n🎭 Mask Ratio: {mask_ratio:.0%}")
            
            # Get average performance over 3 samples
            results = self.mask_and_denoise(text, mask_ratio=mask_ratio, num_samples=3, show_process=False)
            
            avg_similarity = np.mean([r['similarity'] for r in results])
            best_result = max(results, key=lambda x: x['similarity'])
            
            print(f"  Best reconstruction: '{best_result['reconstructed']}'")
            print(f"  Average similarity: {avg_similarity:.1%} {'✅' if avg_similarity > 0.7 else '⚠️' if avg_similarity > 0.4 else '❌'}")
    
    def batch_denoise(self, sentences, mask_ratio=0.3):
        """Denoise multiple sentences"""
        print(f"\n📦 BATCH MASK + DENOISE")
        print(f"Processing {len(sentences)} sentences with {mask_ratio:.0%} masking")
        print("-" * 60)
        
        all_results = []
        total_similarity = 0
        
        for i, text in enumerate(sentences):
            print(f"\n{i+1}. Original: '{text}'")
            
            # Get best result from 3 attempts
            results = self.mask_and_denoise(text, mask_ratio=mask_ratio, num_samples=3, show_process=False)
            best_result = max(results, key=lambda x: x['similarity'])
            
            print(f"   Reconstructed: '{best_result['reconstructed']}'")
            print(f"   Quality: {best_result['similarity']:.1%} {'✅' if best_result['similarity'] > 0.7 else '⚠️' if best_result['similarity'] > 0.4 else '❌'}")
            
            all_results.append(best_result)
            total_similarity += best_result['similarity']
        
        avg_similarity = total_similarity / len(sentences)
        print(f"\n📊 BATCH SUMMARY:")
        print(f"   Average quality: {avg_similarity:.1%}")
        print(f"   Success rate: {sum(1 for r in all_results if r['similarity'] > 0.7) / len(all_results):.1%}")
        
        return all_results
    
    def stress_test(self, text, target_similarity=0.8, max_attempts=10):
        """Keep trying until we get good reconstruction"""
        print(f"\n💪 STRESS TEST")
        print(f"Target: '{text}'")
        print(f"Goal: {target_similarity:.0%} similarity")
        print("-" * 40)
        
        best_result = None
        best_similarity = 0
        
        for attempt in range(max_attempts):
            results = self.mask_and_denoise(text, mask_ratio=0.3, num_samples=1, show_process=False)
            result = results[0]
            
            print(f"Attempt {attempt+1}: '{result['reconstructed']}' ({result['similarity']:.1%})")
            
            if result['similarity'] > best_similarity:
                best_result = result
                best_similarity = result['similarity']
            
            if result['similarity'] >= target_similarity:
                print(f"🎉 SUCCESS! Achieved {result['similarity']:.1%} similarity in {attempt+1} attempts")
                return result
        
        print(f"⚠️ Best achieved: {best_similarity:.1%} similarity")
        return best_result
    
    def _calculate_similarity(self, text1, text2):
        """Calculate word-level similarity"""
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
                
                # Look for best checkpoint (both .pt and .th)
                checkpoint_files = ['best_model.pt', 'final_model.pt', 'best.th']
                for checkpoint_file in checkpoint_files:
                    checkpoint_path = os.path.join(selected_dir, checkpoint_file)
                    if os.path.exists(checkpoint_path):
                        return checkpoint_path
                
                # Look for best() pattern
                for file in os.listdir(selected_dir):
                    if file.startswith('best(') and (file.endswith('.pt') or file.endswith('.th')):
                        return os.path.join(selected_dir, file)
                
                # Look for epoch checkpoints (both .pt and .th)
                epoch_files = [f for f in os.listdir(selected_dir) if f.startswith('epoch_') and (f.endswith('.pt') or f.endswith('.th'))]
                if epoch_files:
                    # Get the highest epoch number
                    highest_epoch = 0
                    latest_file = None
                    for f in epoch_files:
                        try:
                            epoch_num = int(f.split('_')[1].split('.')[0])
                            if epoch_num > highest_epoch:
                                highest_epoch = epoch_num
                                latest_file = f
                        except:
                            continue
                    if latest_file:
                        return os.path.join(selected_dir, latest_file)
                
                # Look for final.th
                final_th = os.path.join(selected_dir, 'final.th')
                if os.path.exists(final_th):
                    return final_th
                
                print(f"❌ No valid checkpoints found in {selected_dir}")
                return None
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number.")

def interactive_demo(denoiser):
    """Interactive demo of mask + denoise"""
    print("\n🎮 INTERACTIVE MASK + DENOISE")
    print("=" * 50)
    print("Commands:")
    print("  'denoise <text>'           - Mask and reconstruct text")
    print("  'compare <text>'           - Compare different mask ratios")
    print("  'batch <text1> | <text2>'  - Process multiple sentences")
    print("  'stress <text>'            - Keep trying until good result")
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
                print("  denoise <text>           - Random mask + BERT denoise")
                print("  compare <text>           - Test different mask ratios")
                print("  batch <text1> | <text2> - Multiple sentences")
                print("  stress <text>            - Keep trying for best result")
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
                denoiser.mask_and_denoise(text, mask_ratio=0.3, num_samples=3)
                
            elif command == 'compare':
                denoiser.compare_mask_ratios(text)
                
            elif command == 'batch':
                sentences = [s.strip() for s in text.split('|')]
                denoiser.batch_denoise(sentences)
                
            elif command == 'stress':
                denoiser.stress_test(text, target_similarity=0.8, max_attempts=10)
                
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
    parser = argparse.ArgumentParser(description="Simple Mask + BERT Denoise Demo")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    
    args = parser.parse_args()
    
    print("🎯 SIMPLE MASK + BERT DENOISE DEMO")
    print("=" * 50)
    print("Random masking + excellent BERT reconstruction")
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
        denoiser = SimpleMaskDenoiser(checkpoint_path, device)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Quick demo
    print("\n🎯 QUICK DEMO")
    print("-" * 30)
    demo_text = "The weather is really nice today"
    denoiser.mask_and_denoise(demo_text, mask_ratio=0.3, num_samples=2)
    
    # Test different mask ratios
    print("\n🔬 MASK RATIO COMPARISON")
    print("-" * 30)
    denoiser.compare_mask_ratios(demo_text, mask_ratios=[0.2, 0.4, 0.6])
    
    # Interactive mode
    interactive_demo(denoiser)

if __name__ == "__main__":
    main() 