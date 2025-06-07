"""
Launcher script for Diffusion BERT Conditional Training
Demonstrates different training configurations with examples
"""

import subprocess
import sys
import os
import time

def run_training(config_name, args):
    """Run training with specific configuration"""
    print(f"\n{'='*60}")
    print(f"🏃 STARTING TRAINING: {config_name}")
    print(f"{'='*60}")
    
    # Build command
    cmd = [sys.executable, "train_evaluate_test_conditional.py"] + args
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Working directory: {os.getcwd()}")
    
    try:
        # Run the training
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            print(f"✅ {config_name} completed successfully!")
        else:
            print(f"❌ {config_name} failed with return code {result.returncode}")
            
    except Exception as e:
        print(f"❌ Error running {config_name}: {e}")

def main():
    """Main function to run different training configurations"""
    
    print("🚀 DIFFUSION BERT CONDITIONAL TRAINING LAUNCHER")
    print("="*60)
    print("This script will run different training configurations")
    print("Each configuration demonstrates different aspects of the model")
    print()
    
    # Configuration 1: Quick Demo (QQP task)
    print("📋 Available configurations:")
    print("1. Quick Demo - QQP paraphrase task (small scale)")
    print("2. Fast Training - QQP with moderate settings")
    print("3. Deep Training - QQP with more steps and epochs")
    print("4. Custom - Run with your own parameters")
    print()
    
    choice = input("Select configuration (1-4): ").strip()
    
    if choice == "1":
        # Quick demo configuration
        config_name = "Quick Demo (QQP)"
        args = [
            "--task_name", "qqp",
            "--epochs", "2",
            "--batch_size", "4",
            "--max_samples", "100",
            "--num_steps", "16",
            "--lr", "5e-5",
            "--eval_steps", "10",
            "--logging_steps", "5",
            "--demonstration_steps", "10",
            "--from_scratch", "False",
            "--device", "cuda:0" if input("Use GPU? (y/n): ").lower() == 'y' else "cpu"
        ]
        
    elif choice == "2":
        # Fast training configuration
        config_name = "Fast Training (QQP)"
        args = [
            "--task_name", "qqp", 
            "--epochs", "3",
            "--batch_size", "8",
            "--max_samples", "500",
            "--num_steps", "32",
            "--lr", "3e-5",
            "--eval_steps", "25",
            "--logging_steps", "10",
            "--demonstration_steps", "25",
            "--from_scratch", "True",
            "--device", "cuda:0" if input("Use GPU? (y/n): ").lower() == 'y' else "cpu"
        ]
        
    elif choice == "3":
        # Deep training configuration
        config_name = "Deep Training (QQP)"
        args = [
            "--task_name", "qqp",
            "--epochs", "5", 
            "--batch_size", "16",
            "--max_samples", "1000",
            "--num_steps", "64",
            "--lr", "2e-5",
            "--eval_steps", "50",
            "--logging_steps", "10",
            "--demonstration_steps", "25",
            "--word_freq_lambda", "0.2",
            "--hybrid_lambda", "5e-4",
            "--from_scratch", "True",
            "--device", "cuda:0" if input("Use GPU? (y/n): ").lower() == 'y' else "cpu"
        ]
        
    elif choice == "4":
        # Custom configuration
        config_name = "Custom Configuration"
        
        print("\n🛠️ Custom Configuration:")
        task = input("Task name (qqp/QT): ").strip() or "qqp"
        epochs = input("Number of epochs (default 3): ").strip() or "3"
        batch_size = input("Batch size (default 8): ").strip() or "8"
        max_samples = input("Max samples (default 500): ").strip() or "500"
        num_steps = input("Diffusion steps (default 32): ").strip() or "32"
        lr = input("Learning rate (default 3e-5): ").strip() or "3e-5"
        device = "cuda:0" if input("Use GPU? (y/n): ").lower() == 'y' else "cpu"
        from_scratch = "True" if input("Train from scratch? (y/n): ").lower() == 'y' else "False"
        
        args = [
            "--task_name", task,
            "--epochs", epochs,
            "--batch_size", batch_size,
            "--max_samples", max_samples,
            "--num_steps", num_steps,
            "--lr", lr,
            "--device", device,
            "--from_scratch", from_scratch,
            "--eval_steps", "25",
            "--logging_steps", "10", 
            "--demonstration_steps", "25"
        ]
        
    else:
        print("❌ Invalid choice. Exiting.")
        return
    
    # Show configuration summary
    print(f"\n📋 Configuration Summary for {config_name}:")
    print("-" * 50)
    for i in range(0, len(args), 2):
        if i + 1 < len(args):
            print(f"  {args[i]}: {args[i+1]}")
    
    # Confirm before running
    confirm = input(f"\n▶️ Run {config_name}? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Training cancelled.")
        return
    
    # Run the training
    start_time = time.time()
    run_training(config_name, args)
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"\n⏱️ Training completed in {duration:.1f} seconds ({duration/60:.1f} minutes)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Launcher interrupted by user")
    except Exception as e:
        print(f"\n❌ Launcher error: {e}")
        import traceback
        traceback.print_exc() 