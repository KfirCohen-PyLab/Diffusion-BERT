"""
Comprehensive Diffusion BERT Training Script with Data Splitting and Demonstrations
Supports conditional training with proper 70% train / 20% eval / 10% test split
"""

import functools
import os
import sys
import random
import numpy as np
import argparse
import torch
import time
import json
from dataloader import QQPLoader, QTLoader, WikiLoader, CCLoader
from transformers import BertTokenizer, BertConfig, RobertaTokenizer, RobertaConfig
from models.modeling_bert import BertForMaskedLM
from models.modeling_roberta import RobertaForMaskedLM
import diffusion_condition
from torch.optim import AdamW
from tqdm import tqdm
from sample import Categorical, WholeWordMasking
import datetime
from torch.utils.data import random_split, Subset
import matplotlib.pyplot as plt

def set_seed(args):
    """Set random seeds for reproducibility"""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Diffusion BERT with data splitting and demonstrations")
    
    # Model and training parameters
    parser.add_argument("--epochs", default=3, type=int, help="Number of training epochs")
    parser.add_argument("--model_name_or_path", default='bert-base-uncased', type=str, help="Model name or path")
    parser.add_argument("--task_name", default='qqp', type=str, choices=['qqp', 'QT', 'wiki_alignment', 'CC'], help="Task name")
    parser.add_argument("--lr", default=3e-5, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size")
    parser.add_argument("--word_freq_lambda", default=0.1, type=float, help="Word frequency lambda")
    parser.add_argument("--num_steps", default=32, type=int, help="Number of diffusion steps")
    parser.add_argument("--eval_step_size", default=4, type=int, help="Evaluation step size")
    parser.add_argument("--accumulation_steps", default=2, type=int, help="Gradient accumulation steps")
    parser.add_argument("--hybrid_lambda", default=3e-4, type=float, help="Hybrid lambda")
    
    # Data splitting parameters
    parser.add_argument("--train_ratio", default=0.7, type=float, help="Training data ratio")
    parser.add_argument("--eval_ratio", default=0.2, type=float, help="Evaluation data ratio") 
    parser.add_argument("--test_ratio", default=0.1, type=float, help="Test data ratio")
    parser.add_argument("--max_samples", default=1000, type=int, help="Maximum samples to use (for demo)")
    
    # Evaluation and logging
    parser.add_argument("--eval_steps", default=50, type=int, help="Evaluation frequency")
    parser.add_argument("--logging_steps", default=10, type=int, help="Logging frequency")
    parser.add_argument("--save_steps", default=100, type=int, help="Save frequency")
    parser.add_argument("--demonstration_steps", default=25, type=int, help="Demonstration frequency")
    
    # Other parameters
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--device", default='cuda:0', type=str, help="Device")
    parser.add_argument('--predict_x0', default=True, type=bool, help="Predict x0")
    parser.add_argument("--load_step", default=-1, type=int, help="Load step")
    parser.add_argument("--sample_strategy", default='Categorical', type=str, help="Sample strategy")
    parser.add_argument("--schedule", default='mutual', type=str, help="Schedule")
    parser.add_argument("--from_scratch", default=True, type=bool, help="Train from scratch")
    parser.add_argument("--timestep", default='none', type=str, help="Timestep strategy")
    
    return parser.parse_args()

def load_and_split_data(args, tokenizer):
    """Load data and create 70%-20%-10% splits"""
    print("="*60)
    print("LOADING AND SPLITTING DATA")
    print("="*60)
    
    # Select appropriate data loader
    Dataloaders = {
        'qqp': QQPLoader,
        'QT': QTLoader,
        'wiki_alignment': WikiLoader,
        'CC': CCLoader,
    }
    
    Loader = Dataloaders[args.task_name]
    loader = Loader(tokenizer=tokenizer)
    
    # Load full dataset
    try:
        full_train_data, validation_data = loader.my_load(splits=['train', 'validation'])
        print(f"✅ Loaded {args.task_name} dataset")
        print(f"   Original train size: {len(full_train_data)}")
        print(f"   Original validation size: {len(validation_data)}")
    except Exception as e:
        print(f"❌ Error loading {args.task_name} dataset: {e}")
        return None, None, None, None
    
    # Limit dataset size for demonstration
    total_size = min(len(full_train_data), args.max_samples)
    indices = list(range(total_size))
    random.shuffle(indices)
    
    # Calculate split sizes
    train_size = int(total_size * args.train_ratio)
    eval_size = int(total_size * args.eval_ratio)
    test_size = total_size - train_size - eval_size
    
    print(f"\nData split (total: {total_size} samples):")
    print(f"  📚 Train: {train_size} samples ({args.train_ratio*100:.1f}%)")
    print(f"  📊 Eval:  {eval_size} samples ({args.eval_ratio*100:.1f}%)")
    print(f"  🧪 Test:  {test_size} samples ({args.test_ratio*100:.1f}%)")
    
    # Create splits
    train_indices = indices[:train_size]
    eval_indices = indices[train_size:train_size + eval_size]
    test_indices = indices[train_size + eval_size:train_size + eval_size + test_size]
    
    train_data = Subset(full_train_data, train_indices)
    eval_data = Subset(full_train_data, eval_indices)
    test_data = Subset(full_train_data, test_indices)
    
    # Show examples from each split
    print(f"\n📚 TRAIN DATA EXAMPLES:")
    for i in range(min(2, len(train_data))):
        example = train_data[i]
        print(f"  Example {i+1}:")
        print(f"    Source: {example['src']}")
        print(f"    Target: {example['trg']}")
    
    print(f"\n📊 EVAL DATA EXAMPLES:")
    for i in range(min(2, len(eval_data))):
        example = eval_data[i]
        print(f"  Example {i+1}:")
        print(f"    Source: {example['src']}")
        print(f"    Target: {example['trg']}")
    
    print(f"\n🧪 TEST DATA EXAMPLES:")
    for i in range(min(2, len(test_data))):
        example = test_data[i]
        print(f"  Example {i+1}:")
        print(f"    Source: {example['src']}")
        print(f"    Target: {example['trg']}")
    
    return train_data, eval_data, test_data, Loader

def create_data_loaders(train_data, eval_data, test_data, Loader, tokenizer, args):
    """Create data loaders for train, eval, and test"""
    print("\n📦 Creating data loaders...")
    
    collate_fn = functools.partial(Loader.collate_fn, tokenizer=tokenizer)
    
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=0,  # Windows compatibility
        pin_memory=True,
        shuffle=True
    )
    
    eval_loader = torch.utils.data.DataLoader(
        eval_data,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"✅ Created data loaders:")
    print(f"   Train: {len(train_loader)} batches")
    print(f"   Eval:  {len(eval_loader)} batches")
    print(f"   Test:  {len(test_loader)} batches")
    
    return train_loader, eval_loader, test_loader

def demonstrate_diffusion_process(model, diffusion_instance, tokenizer, device, step, batch):
    """Demonstrate the diffusion process with examples"""
    print(f"\n🎭 DIFFUSION DEMONSTRATION (Step {step})")
    print("="*50)
    
    model.eval()
    with torch.no_grad():
        # Take first example from batch
        input_ids = batch['input_ids'][0:1].to(device)  # Single example
        attention_mask = batch['attention_mask'][0:1].to(device)
        target_mask = batch['target_mask'][0:1].to(device)
        
        # Decode original
        original_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        print(f"📝 Original: {original_text}")
        
        # Show what parts will be corrupted
        source_tokens = []
        target_tokens = []
        for i, (token_id, is_target) in enumerate(zip(input_ids[0], target_mask[0])):
            token = tokenizer.decode([token_id])
            if is_target.item():
                target_tokens.append(token)
            else:
                source_tokens.append(token)
        
        print(f"🎯 Source (kept): {''.join(source_tokens)}")
        print(f"🎯 Target (to generate): {''.join(target_tokens)}")
        
        # Apply corruption at different timesteps
        timesteps_to_show = [1, diffusion_instance.num_steps // 2, diffusion_instance.num_steps]
        
        for t in timesteps_to_show:
            print(f"\n⏰ Timestep {t}/{diffusion_instance.num_steps}:")
            
            # Sample corruption
            t_tensor = torch.tensor([t], device=device)
            posterior_logits, corrupted_ids = diffusion_instance.sample_and_compute_posterior_q(
                input_ids, t_tensor, return_logits=True, return_transition_probs=False
            )
            
            # Create conditional input (mix source and corrupted target)
            conditional_ids = torch.where(target_mask.bool(), corrupted_ids, input_ids)
            corrupted_text = tokenizer.decode(conditional_ids[0], skip_special_tokens=True)
            print(f"   🔀 Corrupted: {corrupted_text}")
            
            # Try to denoise
            try:
                def denoise_fn(input_ids, corrupted_input_ids, timestep, attention_mask, target_mask):
                    new_input_ids = torch.where(target_mask.bool(), corrupted_input_ids, input_ids)
                    return model(input_ids=new_input_ids, attention_mask=attention_mask)['logits']
                
                logits = denoise_fn(input_ids, corrupted_ids, t_tensor, attention_mask, target_mask)
                predicted_ids = logits.argmax(dim=-1)
                denoised_conditional = torch.where(target_mask.bool(), predicted_ids, input_ids)
                denoised_text = tokenizer.decode(denoised_conditional[0], skip_special_tokens=True)
                print(f"   🔧 Denoised: {denoised_text}")
                
            except Exception as e:
                print(f"   ❌ Denoising failed: {e}")
    
    model.train()

def evaluate_model(model, eval_loader, diffusion_instance, tokenizer, device, args):
    """Evaluate model performance"""
    print("\n📊 EVALUATING MODEL")
    print("="*40)
    
    model.eval()
    total_loss = 0.0
    total_samples = 0
    nan_count = 0
    
    def denoise_fn(input_ids, corrupted_input_ids, timestep, attention_mask, target_mask):
        new_input_ids = torch.where(target_mask.bool(), corrupted_input_ids, input_ids)
        return model(input_ids=new_input_ids, attention_mask=attention_mask)['logits']
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Evaluating")):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            try:
                # Sample timestep
                t = diffusion_instance.sample_t().to(device)
                
                # Compute metrics
                metrics = diffusion_condition.compute_kl_reverse_process(
                    batch['input_ids'],
                    t,
                    denoise_fn=functools.partial(
                        denoise_fn,
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        target_mask=batch['target_mask']
                    ),
                    diffusion=diffusion_instance,
                    target_mask=batch['target_mask'],
                    hybrid_lambda=args.hybrid_lambda,
                    predict_x0=args.predict_x0,
                    word_freq_logits=torch.zeros_like(batch['input_ids'])
                )
                
                if not torch.isnan(metrics['loss']):
                    total_loss += metrics['loss'].item()
                    total_samples += batch['input_ids'].size(0)
                else:
                    nan_count += 1
                    
            except Exception as e:
                print(f"Evaluation error in batch {batch_idx}: {e}")
                nan_count += 1
    
    avg_loss = total_loss / max(1, total_samples)
    success_rate = max(0, (len(eval_loader) - nan_count) / len(eval_loader))
    
    print(f"📈 Evaluation Results:")
    print(f"   Average Loss: {avg_loss:.4f}")
    print(f"   Success Rate: {success_rate:.2%}")
    print(f"   Total Samples: {total_samples}")
    print(f"   Failed Batches: {nan_count}")
    
    model.train()
    return avg_loss, success_rate

def test_model(model, test_loader, diffusion_instance, tokenizer, device, args):
    """Test model and show detailed results"""
    print("\n🧪 TESTING MODEL")
    print("="*40)
    
    model.eval()
    test_results = []
    
    def denoise_fn(input_ids, corrupted_input_ids, timestep, attention_mask, target_mask):
        new_input_ids = torch.where(target_mask.bool(), corrupted_input_ids, input_ids)
        return model(input_ids=new_input_ids, attention_mask=attention_mask)['logits']
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            if batch_idx >= 3:  # Show only first 3 batches for demo
                break
                
            batch = {k: v.to(device) for k, v in batch.items()}
            
            print(f"\n🔬 Test Batch {batch_idx + 1}:")
            print("-" * 30)
            
            # Show original examples
            for i in range(min(2, batch['input_ids'].size(0))):
                original_text = tokenizer.decode(batch['input_ids'][i], skip_special_tokens=True)
                print(f"  Original {i+1}: {original_text}")
                
                # Test generation at different corruption levels
                for corruption_ratio in [0.3, 0.6]:
                    try:
                        # Create corrupted version
                        target_positions = batch['target_mask'][i:i+1].bool()
                        input_ids_single = batch['input_ids'][i:i+1]
                        
                        # Random corruption
                        corrupted_ids = input_ids_single.clone()
                        n_target_tokens = target_positions.sum().item()
                        n_to_corrupt = int(corruption_ratio * n_target_tokens)
                        
                        if n_to_corrupt > 0:
                            target_indices = target_positions.nonzero()[:, 1]
                            if len(target_indices) > 0:
                                corrupt_indices = target_indices[torch.randperm(len(target_indices))[:n_to_corrupt]]
                                corrupted_ids[0, corrupt_indices] = tokenizer.mask_token_id
                        
                        # Create conditional input
                        conditional_input = torch.where(target_positions, corrupted_ids, input_ids_single)
                        corrupted_text = tokenizer.decode(conditional_input[0], skip_special_tokens=True)
                        
                        # Try to denoise
                        logits = denoise_fn(
                            input_ids_single, 
                            corrupted_ids, 
                            torch.tensor([16], device=device),
                            batch['attention_mask'][i:i+1], 
                            batch['target_mask'][i:i+1]
                        )
                        
                        predicted_ids = logits.argmax(dim=-1)
                        denoised_input = torch.where(target_positions, predicted_ids, input_ids_single)
                        denoised_text = tokenizer.decode(denoised_input[0], skip_special_tokens=True)
                        
                        print(f"    Corrupted ({corruption_ratio:.0%}): {corrupted_text}")
                        print(f"    Denoised: {denoised_text}")
                        
                        # Calculate similarity (simple word overlap)
                        original_words = set(original_text.lower().split())
                        denoised_words = set(denoised_text.lower().split())
                        similarity = len(original_words & denoised_words) / max(len(original_words), 1)
                        print(f"    Similarity: {similarity:.2%}")
                        
                        test_results.append({
                            'original': original_text,
                            'corrupted': corrupted_text,
                            'denoised': denoised_text,
                            'similarity': similarity,
                            'corruption_ratio': corruption_ratio
                        })
                        
                    except Exception as e:
                        print(f"    ❌ Test failed: {e}")
    
    # Summary statistics
    if test_results:
        avg_similarity = np.mean([r['similarity'] for r in test_results])
        print(f"\n📋 Test Summary:")
        print(f"   Samples tested: {len(test_results)}")
        print(f"   Average similarity: {avg_similarity:.2%}")
        
        # Show best and worst examples
        best_result = max(test_results, key=lambda x: x['similarity'])
        worst_result = min(test_results, key=lambda x: x['similarity'])
        
        print(f"\n🏆 Best reconstruction (similarity: {best_result['similarity']:.2%}):")
        print(f"   Original: {best_result['original']}")
        print(f"   Denoised: {best_result['denoised']}")
        
        print(f"\n💔 Worst reconstruction (similarity: {worst_result['similarity']:.2%}):")
        print(f"   Original: {worst_result['original']}")
        print(f"   Denoised: {worst_result['denoised']}")
    
    model.train()
    return test_results

def save_training_results(args, training_history, test_results, save_path):
    """Save training results and metrics"""
    results = {
        'args': vars(args),
        'training_history': training_history,
        'test_results': test_results,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    results_file = os.path.join(save_path, 'training_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Saved training results to {results_file}")

def main():
    """Main training function"""
    print("🚀 STARTING DIFFUSION BERT TRAINING")
    print("="*60)
    
    # Parse arguments and setup
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    set_seed(args)
    
    # Create save directory
    save_path = f'./model_name_{args.model_name_or_path}_taskname_{args.task_name}_lr_{args.lr}_seed_{args.seed}_numsteps_{args.num_steps}_sample_{args.sample_strategy}_schedule_{args.schedule}_hybridlambda_{args.hybrid_lambda}_wordfreqlambda_{args.word_freq_lambda}_fromscratch_{args.from_scratch}_timestep_{args.timestep}_ckpts'
    
    os.makedirs(save_path, exist_ok=True)
    print(f"💾 Save path: {save_path}")
    
    # Initialize model components
    if args.model_name_or_path in ['bert-base-uncased', 'bert-large-uncased']:
        model_cls = BertForMaskedLM
        cfg_cls = BertConfig
        tok_cls = BertTokenizer
    elif args.model_name_or_path in ['roberta-base']:
        model_cls = RobertaForMaskedLM
        cfg_cls = RobertaConfig
        tok_cls = RobertaTokenizer
    else:
        raise NotImplementedError(f"Model {args.model_name_or_path} not supported")
    
    tokenizer = tok_cls.from_pretrained(args.model_name_or_path)
    
    # Setup word frequency (simplified for conditional tasks)
    word_freq = torch.zeros(tokenizer.vocab_size)
    word_freq = (word_freq + 1).log() / (word_freq + 1).log().max()
    
    # Initialize diffusion
    if args.sample_strategy == 'Categorical':
        sample_cls = Categorical()
    elif args.sample_strategy == 'wwm':
        sample_cls = WholeWordMasking(tokenizer)
    else:
        raise ValueError(f"Unknown sample strategy: {args.sample_strategy}")
    
    diffusion_schedule = diffusion_condition.create_discrete_diffusion_schedule(
        args.schedule, num_steps=args.num_steps
    )
    diffusion_instance = diffusion_condition.MaskDiffusion(
        dim=tokenizer.vocab_size,
        schedule=diffusion_schedule,
        tokenizer=tokenizer,
        sample_cls=sample_cls,
        word_freq=word_freq,
        word_freq_lambda=args.word_freq_lambda,
        device=device
    )
    
    # Initialize model
    cfg = cfg_cls.from_pretrained(args.model_name_or_path)
    cfg.overall_timestep = diffusion_instance.num_steps
    
    if args.from_scratch:
        model = model_cls(cfg).to(device)
        print("🔨 Training from scratch")
    else:
        model = model_cls.from_pretrained(args.model_name_or_path, config=cfg).to(device)
        print("🔄 Fine-tuning pretrained model")
    
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=1000)
    
    # Load and split data
    train_data, eval_data, test_data, Loader = load_and_split_data(args, tokenizer)
    if train_data is None:
        print("❌ Failed to load data. Exiting.")
        return
    
    # Create data loaders
    train_loader, eval_loader, test_loader = create_data_loaders(
        train_data, eval_data, test_data, Loader, tokenizer, args
    )
    
    # Define denoise function
    def denoise_fn(input_ids, corrupted_input_ids, timestep, attention_mask, target_mask):
        new_input_ids = torch.where(target_mask.bool(), corrupted_input_ids, input_ids)
        return model(input_ids=new_input_ids, attention_mask=attention_mask)['logits']
    
    # Training loop
    print(f"\n🎯 STARTING TRAINING ({args.epochs} epochs)")
    print("="*50)
    
    training_history = []
    best_eval_loss = float('inf')
    global_step = 0
    
    for epoch in range(args.epochs):
        print(f"\n📚 Epoch {epoch + 1}/{args.epochs}")
        print("-" * 40)
        
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            global_step += 1
            
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            try:
                t = diffusion_instance.sample_t().to(device)
                metrics = diffusion_condition.compute_kl_reverse_process(
                    batch['input_ids'],
                    t,
                    denoise_fn=functools.partial(
                        denoise_fn,
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        target_mask=batch['target_mask']
                    ),
                    diffusion=diffusion_instance,
                    target_mask=batch['target_mask'],
                    hybrid_lambda=args.hybrid_lambda,
                    predict_x0=args.predict_x0,
                    word_freq_logits=torch.zeros_like(batch['input_ids'])
                )
                
                loss = metrics['loss'] / args.accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Step optimizer
                if (batch_idx + 1) % args.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_value_(model.parameters(), 5)
                    optimizer.step()
                    optimizer.zero_grad()
                    warmup_scheduler.step()
                
                # Track loss
                if not torch.isnan(metrics['loss']):
                    epoch_loss += metrics['loss'].item()
                    epoch_steps += 1
                
                # Logging
                if global_step % args.logging_steps == 0:
                    avg_loss = epoch_loss / max(1, epoch_steps)
                    print(f"Step {global_step}: Loss = {avg_loss:.4f}")
                
                # Demonstration
                if global_step % args.demonstration_steps == 0:
                    demonstrate_diffusion_process(
                        model, diffusion_instance, tokenizer, device, global_step, batch
                    )
                
                # Evaluation
                if global_step % args.eval_steps == 0:
                    eval_loss, success_rate = evaluate_model(
                        model, eval_loader, diffusion_instance, tokenizer, device, args
                    )
                    
                    training_history.append({
                        'step': global_step,
                        'epoch': epoch + 1,
                        'train_loss': epoch_loss / max(1, epoch_steps),
                        'eval_loss': eval_loss,
                        'success_rate': success_rate
                    })
                    
                    # Save best model
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        torch.save({
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'warmup_scheduler': warmup_scheduler.state_dict(),
                            'eval_loss': eval_loss,
                            'step': global_step
                        }, os.path.join(save_path, 'best_model.pt'))
                        print(f"💎 New best model saved! Eval loss: {eval_loss:.4f}")
                
            except Exception as e:
                print(f"❌ Error in training step {global_step}: {e}")
                continue
        
        # End of epoch
        avg_epoch_loss = epoch_loss / max(1, epoch_steps)
        print(f"📊 Epoch {epoch + 1} complete. Average loss: {avg_epoch_loss:.4f}")
        
        # Save epoch checkpoint
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'warmup_scheduler': warmup_scheduler.state_dict(),
            'epoch': epoch + 1,
            'loss': avg_epoch_loss
        }, os.path.join(save_path, f'epoch_{epoch + 1}.pt'))
    
    # Final testing
    print(f"\n🧪 FINAL TESTING")
    print("="*40)
    test_results = test_model(model, test_loader, diffusion_instance, tokenizer, device, args)
    
    # Save final results
    save_training_results(args, training_history, test_results, save_path)
    
    # Final checkpoint
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'warmup_scheduler': warmup_scheduler.state_dict(),
        'training_complete': True
    }, os.path.join(save_path, 'final_model.pt'))
    
    print(f"\n🎉 TRAINING COMPLETE!")
    print(f"📁 All results saved to: {save_path}")
    print(f"💎 Best evaluation loss: {best_eval_loss:.4f}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc() 