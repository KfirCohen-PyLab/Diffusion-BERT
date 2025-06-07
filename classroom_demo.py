"""
Classroom Demonstration Script for Diffusion BERT
Optimized for presentations with smooth flow and impressive examples
"""

import torch
import argparse
from transformers import BertTokenizer, BertConfig
from models.modeling_bert import BertForMaskedLM
import os
import random
import time

class ClassroomDemo:
    def __init__(self, checkpoint_path, device):
        """Initialize demo with trained model"""
        self.device = device
        self.load_model(checkpoint_path)
        
    def load_model(self, checkpoint_path):
        """Load trained model from checkpoint"""
        print("🎓 Loading Diffusion BERT model for demonstration...")
        print(f"📂 Checkpoint: {os.path.basename(checkpoint_path)}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Initialize model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        cfg = BertConfig.from_pretrained('bert-base-uncased')
        cfg.overall_timestep = 32
        
        self.model = BertForMaskedLM(cfg).to(self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        
        print("✅ Model loaded successfully!\n")
    
    def demonstrate_capability(self, text, mask_ratio=0.3, show_process=True):
        """Demonstrate single example with clear explanation"""
        if show_process:
            print(f"🎯 DEMONSTRATION: Mask & Reconstruct")
            print(f"📝 Original sentence: '{text}'")
            print(f"🎭 Masking {mask_ratio:.0%} of tokens randomly...")
            time.sleep(1)  # Pause for audience
        
        # Tokenize input
        tokens = self.tokenizer.encode(text, return_tensors='pt', max_length=64, truncation=True)
        tokens = tokens.to(self.device)
        
        # Apply random masking
        masked_tokens = self._apply_random_mask(tokens.clone(), mask_ratio)
        masked_text = self.tokenizer.decode(masked_tokens[0], skip_special_tokens=True)
        
        if show_process:
            print(f"🔍 After masking: '{masked_text}'")
            print("🤖 Running BERT reconstruction...")
            time.sleep(1)
        
        # BERT reconstruction
        reconstructed_tokens = self._bert_denoise(masked_tokens)
        reconstructed_text = self.tokenizer.decode(reconstructed_tokens[0], skip_special_tokens=True)
        
        # Calculate similarity
        similarity = self._calculate_similarity(text, reconstructed_text)
        
        if show_process:
            print(f"✨ Reconstructed: '{reconstructed_text}'")
            print(f"📊 Accuracy: {similarity:.1%} {'🎉 Excellent!' if similarity > 0.8 else '👍 Good!' if similarity > 0.6 else '⚠️ Fair'}")
            print("-" * 60)
        
        return {
            'original': text,
            'masked': masked_text,
            'reconstructed': reconstructed_text,
            'accuracy': similarity
        }
    
    def run_planned_demonstration(self):
        """Run a pre-planned demonstration with impressive examples"""
        print("🎓 DIFFUSION BERT - CLASSROOM DEMONSTRATION")
        print("=" * 60)
        print("📚 What we're demonstrating:")
        print("   • Randomly mask words in sentences")
        print("   • Use trained BERT to reconstruct missing words")
        print("   • Show how AI understands language context")
        print("=" * 60)
        print()
        
        # Impressive demo examples
        demo_examples = [
            "Artificial intelligence is transforming modern technology",
            "Students learn best through hands-on practice and examples", 
            "Machine learning models require large amounts of training data",
            "Natural language processing helps computers understand human text",
            "Deep learning networks can solve complex pattern recognition problems"
        ]
        
        total_accuracy = 0
        
        for i, example in enumerate(demo_examples, 1):
            print(f"📋 EXAMPLE {i}/5")
            result = self.demonstrate_capability(example, mask_ratio=0.35)
            total_accuracy += result['accuracy']
            
            if i < len(demo_examples):
                input("\n⏸️  Press Enter to continue to next example...")
                print()
        
        # Summary
        avg_accuracy = total_accuracy / len(demo_examples)
        print(f"🏆 DEMONSTRATION SUMMARY")
        print(f"   📊 Average Reconstruction Accuracy: {avg_accuracy:.1%}")
        print(f"   🎯 Model Performance: {'Excellent' if avg_accuracy > 0.8 else 'Good' if avg_accuracy > 0.6 else 'Needs Improvement'}")
        print(f"   💡 This shows the model has learned language patterns!")
    
    def interactive_demo(self):
        """Interactive demo for audience participation"""
        print("\n🎮 INTERACTIVE DEMONSTRATION")
        print("=" * 40)
        print("Now let's try some examples from the audience!")
        print("(Teachers/students can suggest sentences to test)")
        print()
        
        while True:
            try:
                user_input = input("📝 Enter a sentence to test (or 'done' to finish): ").strip()
                
                if user_input.lower() in ['done', 'quit', 'exit', 'stop']:
                    print("🎉 Thank you! End of interactive demonstration.")
                    break
                
                if not user_input:
                    continue
                
                print()
                self.demonstrate_capability(user_input)
                print()
                
            except KeyboardInterrupt:
                print("\n🎉 Thank you! End of demonstration.")
                break
    
    def quick_comparison_demo(self):
        """Show different mask ratios for educational purposes"""
        print("\n🔬 EDUCATIONAL DEMO: Effect of Different Corruption Levels")
        print("=" * 60)
        
        example = "Machine learning algorithms can process vast amounts of data"
        print(f"📝 Test sentence: '{example}'")
        print()
        
        mask_ratios = [0.2, 0.4, 0.6]
        
        for ratio in mask_ratios:
            print(f"🎭 Masking {ratio:.0%} of words:")
            result = self.demonstrate_capability(example, mask_ratio=ratio, show_process=False)
            print(f"   Masked: '{result['masked']}'")
            print(f"   Result: '{result['reconstructed']}'")
            print(f"   Accuracy: {result['accuracy']:.1%}")
            print()
            time.sleep(1)
    
    def _apply_random_mask(self, tokens, mask_ratio):
        """Apply random masking to tokens"""
        seq_len = tokens.size(1)
        n_mask = max(1, int(mask_ratio * seq_len))
        
        maskable_positions = list(range(1, seq_len - 1))
        
        if len(maskable_positions) >= n_mask:
            mask_positions = random.sample(maskable_positions, n_mask)
            for pos in mask_positions:
                tokens[0, pos] = self.tokenizer.mask_token_id
        
        return tokens
    
    def _bert_denoise(self, masked_tokens):
        """Use BERT model to denoise masked tokens"""
        with torch.no_grad():
            attention_mask = (masked_tokens != self.tokenizer.pad_token_id).long()
            outputs = self.model(input_ids=masked_tokens, attention_mask=attention_mask)
            logits = outputs.logits
            
            mask_positions = (masked_tokens == self.tokenizer.mask_token_id)
            
            if not mask_positions.any():
                return masked_tokens
            
            reconstructed_tokens = masked_tokens.clone()
            
            # Use temperature sampling for better results
            for pos in mask_positions.nonzero():
                token_logits = logits[pos[0], pos[1]]
                temperature = 0.8
                scaled_logits = token_logits / temperature
                probs = torch.softmax(scaled_logits, dim=-1)
                
                # Top-k sampling
                top_k = 10
                top_k_probs, top_k_indices = torch.topk(probs, top_k)
                sampled_idx = torch.multinomial(top_k_probs, 1)
                predicted_token = top_k_indices[sampled_idx]
                
                reconstructed_tokens[pos[0], pos[1]] = predicted_token
            
            return reconstructed_tokens
    
    def _calculate_similarity(self, text1, text2):
        """Calculate word-level similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        return len(words1 & words2) / max(len(words1), 1)

def main():
    parser = argparse.ArgumentParser(description="Classroom Demonstration of Diffusion BERT")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "planned", "interactive", "comparison"], 
                       help="Demo mode: full (all), planned (scripted), interactive (audience), comparison (educational)")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    print()
    
    # Initialize demo
    try:
        demo = ClassroomDemo(args.checkpoint, device)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Run demonstration based on mode
    if args.mode == "full":
        demo.run_planned_demonstration()
        demo.quick_comparison_demo()
        demo.interactive_demo()
    elif args.mode == "planned":
        demo.run_planned_demonstration()
    elif args.mode == "interactive":
        demo.interactive_demo()
    elif args.mode == "comparison":
        demo.quick_comparison_demo()

if __name__ == "__main__":
    main() 