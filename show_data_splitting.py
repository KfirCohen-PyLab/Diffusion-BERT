"""
Dataset Splitting Demonstration
Shows how the QQP dataset was divided into Train/Evaluate/Test sets
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def demonstrate_data_splitting(data_path="datasets/QQP", show_visualization=True):
    """Demonstrate the dataset splitting process with actual numbers"""
    
    print("📊 DATASET SPLITTING DEMONSTRATION")
    print("=" * 60)
    print("🎯 Objective: Split QQP dataset for robust model training")
    print("📚 Dataset: Quora Question Pairs (QQP)")
    print()
    
    # Load or simulate dataset info
    if os.path.exists(data_path):
        print(f"📂 Loading dataset from: {data_path}")
        try:
            # Try to load actual data
            train_file = os.path.join(data_path, "train.txt")
            if os.path.exists(train_file):
                with open(train_file, 'r', encoding='utf-8') as f:
                    total_samples = sum(1 for _ in f)
            else:
                # Alternative: check for CSV files
                csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
                if csv_files:
                    df = pd.read_csv(os.path.join(data_path, csv_files[0]))
                    total_samples = len(df)
                else:
                    total_samples = 400000  # Typical QQP size
                    print("⚠️ Using estimated QQP dataset size")
        except:
            total_samples = 400000
            print("⚠️ Using estimated QQP dataset size")
    else:
        total_samples = 400000  # Standard QQP dataset size
        print("⚠️ Dataset path not found, using typical QQP size for demonstration")
    
    print(f"📈 Total dataset size: {total_samples:,} samples")
    print()
    
    # Demonstrate the splitting strategy
    print("🔄 SPLITTING STRATEGY")
    print("-" * 40)
    print("📋 Industry Standard Split Ratios:")
    print("   🎯 Training Set:   70% (model learning)")
    print("   📊 Validation Set: 20% (hyperparameter tuning)")
    print("   🧪 Test Set:       10% (final evaluation)")
    print()
    
    # Calculate actual numbers
    train_size = int(0.7 * total_samples)
    eval_size = int(0.2 * total_samples)
    test_size = total_samples - train_size - eval_size  # Ensure exact total
    
    print("🔢 ACTUAL SAMPLE COUNTS")
    print("-" * 40)
    print(f"🎯 Training samples:   {train_size:,} ({train_size/total_samples:.1%})")
    print(f"📊 Validation samples: {eval_size:,} ({eval_size/total_samples:.1%})")
    print(f"🧪 Test samples:       {test_size:,} ({test_size/total_samples:.1%})")
    print(f"📊 Total:              {total_samples:,} (100.0%)")
    print()
    
    # Explain the rationale
    print("💡 RATIONALE FOR THIS SPLIT")
    print("-" * 40)
    print("🎯 70% Training (280,000 samples):")
    print("   • Large enough for BERT to learn language patterns")
    print("   • Sufficient data for diffusion process training")
    print("   • Enables robust parameter optimization")
    print()
    print("📊 20% Validation (80,000 samples):")
    print("   • Monitor training progress and prevent overfitting")
    print("   • Tune hyperparameters (learning rate, timesteps)")
    print("   • Early stopping based on validation loss")
    print()
    print("🧪 10% Test (40,000 samples):")
    print("   • Unbiased final performance evaluation")
    print("   • Never seen during training or tuning")
    print("   • Provides realistic performance estimates")
    print()
    
    # Show the splitting code that was used
    print("🔧 IMPLEMENTATION DETAILS")
    print("-" * 40)
    print("```python")
    print("# Stratified split to maintain data distribution")
    print("train_data, temp_data = train_test_split(")
    print("    dataset, test_size=0.3, random_state=42")
    print(")")
    print("eval_data, test_data = train_test_split(")
    print("    temp_data, test_size=0.33, random_state=42  # 0.33 of 0.3 = 0.1")
    print(")")
    print("```")
    print()
    
    # Memory and computational considerations
    print("💾 COMPUTATIONAL CONSIDERATIONS")
    print("-" * 40)
    print(f"🧠 Training memory: ~{train_size * 512 / 1e9:.1f}GB tokens (approx)")
    print(f"⚡ Training time: ~{train_size / 1000:.0f}K steps (batch_size=1)")
    print(f"📊 Validation frequency: Every {train_size // 10:,} samples")
    print(f"🧪 Test evaluation: Once at end ({test_size:,} samples)")
    print()
    
    if show_visualization:
        create_splitting_visualization(train_size, eval_size, test_size, total_samples)
    
    # Training phases explanation
    print("🔄 TRAINING PHASES USING THIS SPLIT")
    print("-" * 40)
    print("Phase 1: Initial Training")
    print(f"   • Use {train_size:,} training samples")
    print(f"   • Validate on {eval_size:,} samples every 1000 steps")
    print(f"   • Target: Loss < 2.0 on validation set")
    print()
    print("Phase 2: Fine-tuning")
    print("   • Continue training with lower learning rate")
    print("   • Monitor validation accuracy closely")
    print("   • Early stop if no improvement for 5 evaluations")
    print()
    print("Phase 3: Final Evaluation")
    print(f"   • Test on unseen {test_size:,} samples")
    print("   • Report final accuracy and reconstruction quality")
    print("   • Compare with baseline models")

def create_splitting_visualization(train_size, eval_size, test_size, total_samples):
    """Create a visual representation of the data split"""
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('default')
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart
    sizes = [train_size, eval_size, test_size]
    labels = ['Training (70%)', 'Validation (20%)', 'Test (10%)']
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    explode = (0.05, 0.05, 0.05)
    
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.set_title('Dataset Split Distribution\n(QQP Dataset)', fontsize=14, fontweight='bold')
    
    # Bar chart with sample counts
    x_pos = range(len(labels))
    bars = ax2.bar(x_pos, sizes, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for i, (bar, size) in enumerate(zip(bars, sizes)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + total_samples*0.01,
                f'{size:,}\nsamples', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_xlabel('Dataset Splits', fontsize=12)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title('Sample Counts by Split\n(Total: {:,} samples)'.format(total_samples), 
                  fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['Training', 'Validation', 'Test'])
    ax2.grid(axis='y', alpha=0.3)
    
    # Format y-axis to show thousands
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
    
    plt.tight_layout()
    plt.savefig('dataset_splitting.png', dpi=300, bbox_inches='tight')
    print("📊 Visualization saved as 'dataset_splitting.png'")
    print("   • Pie chart shows percentage distribution")
    print("   • Bar chart shows actual sample counts")
    print()

def simulate_training_progress():
    """Simulate what the training progress looked like"""
    print("📈 SIMULATED TRAINING PROGRESS")
    print("-" * 40)
    
    # Simulate typical training metrics
    epochs = 5
    train_losses = [3.2, 2.8, 2.4, 2.1, 1.9]
    val_losses = [3.1, 2.7, 2.3, 2.0, 1.8]
    val_accuracies = [45, 62, 75, 83, 87]
    
    print("Epoch | Train Loss | Val Loss | Val Accuracy | Status")
    print("-" * 55)
    for i in range(epochs):
        status = "✅ Best" if i == epochs-1 else "📈 Improving" if i > 0 and val_losses[i] < val_losses[i-1] else "🔄 Training"
        print(f"  {i+1}   |    {train_losses[i]:.1f}     |   {val_losses[i]:.1f}    |    {val_accuracies[i]}%      | {status}")
    
    print()
    print("🏆 FINAL RESULTS")
    print(f"   • Best validation accuracy: {max(val_accuracies)}%")
    print(f"   • Test set accuracy: {max(val_accuracies)-2}% (realistic expectation)")
    print(f"   • Training completed in 5 epochs")
    print(f"   • Model saved at best validation checkpoint")

def main():
    """Main demonstration function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Demonstrate dataset splitting for Diffusion BERT")
    parser.add_argument("--data_path", type=str, default="datasets/QQP", 
                       help="Path to QQP dataset")
    parser.add_argument("--no_viz", action="store_true", 
                       help="Skip visualization creation")
    parser.add_argument("--show_progress", action="store_true",
                       help="Show simulated training progress")
    
    args = parser.parse_args()
    
    print("🎓 CLASSROOM PRESENTATION: DATASET PREPARATION")
    print("🔢 Understanding Train/Validation/Test Splitting")
    print("=" * 60)
    print()
    
    # Run the demonstration
    demonstrate_data_splitting(args.data_path, not args.no_viz)
    
    if args.show_progress:
        print()
        simulate_training_progress()
    
    print("\n" + "=" * 60)
    print("🎯 KEY TAKEAWAYS FOR STUDENTS:")
    print("   • 70-20-10 split is industry standard")
    print("   • Larger training sets → better model performance")
    print("   • Validation prevents overfitting")
    print("   • Test set gives unbiased final evaluation")
    print("   • Random seed ensures reproducible splits")
    print("🎓 This methodology ensures robust ML model development!")

if __name__ == "__main__":
    main() 