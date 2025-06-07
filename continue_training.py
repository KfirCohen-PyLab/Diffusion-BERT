"""
Continue Training Script
Loads your trained model and continues training with enhanced settings
"""

import argparse
import torch
import os
from train_evaluate_test_conditional import main as train_main
import sys

def find_latest_checkpoint(model_dir):
    """Find the latest checkpoint in model directory"""
    print(f"🔍 Looking for checkpoints in {model_dir}")
    
    if not os.path.exists(model_dir):
        print(f"❌ Directory {model_dir} not found")
        return None
    
    # Look for checkpoints
    checkpoints = []
    for file in os.listdir(model_dir):
        if file.endswith('.pt'):
            checkpoints.append(file)
    
    if not checkpoints:
        print("❌ No checkpoints found")
        return None
    
    # Find best checkpoint
    if 'best_model.pt' in checkpoints:
        best_path = os.path.join(model_dir, 'best_model.pt')
        print(f"✅ Found best model: {best_path}")
        return best_path
    
    # Or latest epoch
    epoch_checkpoints = [f for f in checkpoints if f.startswith('epoch_')]
    if epoch_checkpoints:
        latest_epoch = max([int(f.split('_')[1].split('.')[0]) for f in epoch_checkpoints])
        latest_path = os.path.join(model_dir, f'epoch_{latest_epoch}.pt')
        print(f"✅ Found latest epoch: {latest_path}")
        return latest_path
    
    # Or final model
    if 'final_model.pt' in checkpoints:
        final_path = os.path.join(model_dir, 'final_model.pt')
        print(f"✅ Found final model: {final_path}")
        return final_path
    
    print("❌ No suitable checkpoint found")
    return None

def suggest_enhanced_settings(current_performance=None):
    """Suggest enhanced training settings based on current performance"""
    print("🎯 ENHANCED TRAINING SUGGESTIONS")
    print("="*50)
    
    configurations = {
        "1": {
            "name": "Quality Boost",
            "description": "More epochs + higher diffusion steps for better quality",
            "settings": {
                "--epochs": "5",
                "--num_steps": "64", 
                "--batch_size": "6",
                "--lr": "2e-5",
                "--max_samples": "1500",
                "--word_freq_lambda": "0.2",
                "--hybrid_lambda": "5e-4"
            }
        },
        "2": {
            "name": "Deep Training",
            "description": "Extended training with advanced parameters",
            "settings": {
                "--epochs": "8",
                "--num_steps": "128",
                "--batch_size": "4", 
                "--lr": "1e-5",
                "--max_samples": "2000",
                "--word_freq_lambda": "0.3",
                "--hybrid_lambda": "1e-3",
                "--accumulation_steps": "4"
            }
        },
        "3": {
            "name": "Fine-tuning",
            "description": "Conservative improvements on existing model",
            "settings": {
                "--epochs": "3",
                "--num_steps": "48",
                "--batch_size": "8",
                "--lr": "1e-5",
                "--max_samples": "1000", 
                "--word_freq_lambda": "0.15"
            }
        },
        "4": {
            "name": "Custom",
            "description": "Set your own enhanced parameters",
            "settings": {}
        }
    }
    
    print("Available enhancement options:")
    for key, config in configurations.items():
        print(f"{key}. {config['name']}: {config['description']}")
    
    return configurations

def create_enhanced_args(base_checkpoint, enhancement_config, task_name="qqp"):
    """Create command line arguments for enhanced training"""
    
    # Base arguments to continue from checkpoint
    args = [
        "--task_name", task_name,
        "--from_scratch", "False",  # Continue from existing model
        "--seed", "42"
    ]
    
    # Add enhancement settings
    for key, value in enhancement_config["settings"].items():
        args.extend([key, value])
    
    # Add checkpoint loading (we'll handle this in the training script)
    checkpoint_dir = os.path.dirname(base_checkpoint)
    args.extend(["--checkpoint_dir", checkpoint_dir])
    
    return args

def main():
    parser = argparse.ArgumentParser(description="Continue training from checkpoint")
    parser.add_argument("--model_dir", type=str, help="Directory containing model checkpoints")
    parser.add_argument("--auto", action="store_true", help="Auto-select best enhancement")
    
    args = parser.parse_args()
    
    print("🚀 CONTINUE TRAINING SCRIPT")
    print("="*50)
    
    # Find model directory
    if args.model_dir:
        model_dir = args.model_dir
    else:
        # Look for recent model directories
        current_dir = "."
        model_dirs = [d for d in os.listdir(current_dir) if d.startswith("model_name_") and os.path.isdir(d)]
        
        if not model_dirs:
            print("❌ No model directories found in current directory")
            print("💡 Run this script with --model_dir /path/to/your/model/directory")
            return
        
        if len(model_dirs) == 1:
            model_dir = model_dirs[0]
            print(f"📁 Found model directory: {model_dir}")
        else:
            print("📁 Multiple model directories found:")
            for i, dir_name in enumerate(model_dirs, 1):
                print(f"  {i}. {dir_name}")
            
            while True:
                try:
                    choice = int(input(f"Select directory (1-{len(model_dirs)}): ")) - 1
                    if 0 <= choice < len(model_dirs):
                        model_dir = model_dirs[choice]
                        break
                    else:
                        print("Invalid choice. Please try again.")
                except ValueError:
                    print("Please enter a number.")
    
    # Find checkpoint
    checkpoint_path = find_latest_checkpoint(model_dir)
    if not checkpoint_path:
        return
    
    # Get enhancement configuration
    configurations = suggest_enhanced_settings()
    
    if args.auto:
        print("🤖 Auto-selecting Quality Boost configuration")
        selected_config = configurations["1"]
    else:
        print(f"\nSelect enhancement configuration (1-{len(configurations)}): ", end="")
        choice = input().strip()
        
        if choice in configurations:
            selected_config = configurations[choice]
            
            if choice == "4":  # Custom configuration
                print("\n🛠️ Custom Enhancement Settings:")
                custom_settings = {}
                
                settings_prompts = {
                    "--epochs": "Number of additional epochs (default 5): ",
                    "--num_steps": "Diffusion steps (default 64): ",
                    "--batch_size": "Batch size (default 6): ",
                    "--lr": "Learning rate (default 1e-5): ",
                    "--max_samples": "Max samples (default 1500): "
                }
                
                for key, prompt in settings_prompts.items():
                    value = input(prompt).strip()
                    if value:
                        custom_settings[key] = value
                
                selected_config = {
                    "name": "Custom",
                    "settings": custom_settings
                }
        else:
            print("❌ Invalid choice. Using Quality Boost.")
            selected_config = configurations["1"]
    
    print(f"\n📋 Selected: {selected_config['name']}")
    print("Settings:")
    for key, value in selected_config["settings"].items():
        print(f"  {key}: {value}")
    
    # Confirm
    confirm = input(f"\n▶️ Continue training with {selected_config['name']}? (y/n): ").lower()
    if confirm != 'y':
        print("❌ Training cancelled.")
        return
    
    # Create enhanced training arguments 
    enhanced_args = create_enhanced_args(checkpoint_path, selected_config)
    
    print(f"\n🚀 STARTING ENHANCED TRAINING")
    print("="*50)
    print("Command arguments:")
    for i in range(0, len(enhanced_args), 2):
        if i + 1 < len(enhanced_args):
            print(f"  {enhanced_args[i]}: {enhanced_args[i+1]}")
    
    # Save current args and run training
    original_argv = sys.argv
    try:
        sys.argv = ['train_evaluate_test_conditional.py'] + enhanced_args
        train_main()
    except Exception as e:
        print(f"❌ Enhanced training failed: {e}")
    finally:
        sys.argv = original_argv
    
    print(f"\n🎉 Enhanced training completed!")
    print(f"💡 Check the new model directory for improved checkpoints")

if __name__ == "__main__":
    main() 