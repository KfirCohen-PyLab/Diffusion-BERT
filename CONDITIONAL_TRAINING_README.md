# Diffusion BERT Conditional Training

This directory contains comprehensive scripts for training Diffusion BERT models using conditional training with proper data splitting and demonstrations.

## 🚀 Quick Start

### Option 1: Windows Batch File (Recommended for Windows)
```bash
run_training.bat
```

### Option 2: Python Launcher
```bash
python run_conditional_training.py
```

### Option 3: Direct Training Script
```bash
python train_evaluate_test_conditional.py --task_name qqp --epochs 3
```

## 📁 Files Overview

- **`train_evaluate_test_conditional.py`** - Main training script with 70%-20%-10% data splitting
- **`run_conditional_training.py`** - Interactive launcher with preset configurations
- **`run_training.bat`** - Windows batch file for easy execution
- **`CONDITIONAL_TRAINING_README.md`** - This documentation

## 🎯 Key Features

### ✨ What This Training Does

1. **Conditional Text Generation**: The model learns to generate target text conditioned on source text
2. **Proper Data Splitting**: 70% training, 20% evaluation, 10% testing
3. **Real-time Demonstrations**: Shows diffusion process during training
4. **Comprehensive Evaluation**: Detailed metrics and example outputs
5. **Robust Error Handling**: Handles CUDA errors and training issues gracefully

### 📊 Data Split Strategy

```
📚 Training (70%): Used for model parameter updates
📊 Evaluation (20%): Used for hyperparameter tuning and early stopping  
🧪 Testing (10%): Used for final performance assessment
```

### 🎭 Training Demonstrations

During training, you'll see:
- **Original Text**: The input sentence
- **Source/Target Split**: What parts are kept vs. generated
- **Corruption Process**: How diffusion adds noise at different timesteps
- **Denoising Process**: How the model reconstructs text
- **Progress Metrics**: Loss curves and success rates

## 🛠️ Configuration Options

### Quick Demo (Recommended First Run)
- 2 epochs, 100 samples
- Small batch size for fast testing
- Shows all key features

### Fast Training
- 3 epochs, 500 samples  
- Moderate settings for balance
- Good for initial experiments

### Deep Training
- 5 epochs, 1000 samples
- Higher diffusion steps
- Better final quality

### Custom Configuration
- Set your own parameters
- Full control over training

## 📋 Command Line Arguments

### Essential Parameters
```bash
--task_name         # Dataset: qqp, QT, wiki_alignment, CC
--epochs           # Number of training epochs (default: 3)
--batch_size       # Batch size (default: 8)  
--max_samples      # Limit dataset size (default: 1000)
--lr               # Learning rate (default: 3e-5)
```

### Diffusion Parameters
```bash
--num_steps        # Diffusion timesteps (default: 32)
--schedule         # Noise schedule: mutual, linear, cosine
--sample_strategy  # Sampling: Categorical, wwm
--hybrid_lambda    # Hybrid loss weight (default: 3e-4)
--word_freq_lambda # Word frequency regularization (default: 0.1)
```

### Data Splitting
```bash
--train_ratio      # Training data ratio (default: 0.7)
--eval_ratio       # Evaluation data ratio (default: 0.2)  
--test_ratio       # Test data ratio (default: 0.1)
```

### Logging & Evaluation
```bash
--eval_steps       # Evaluation frequency (default: 50)
--logging_steps    # Logging frequency (default: 10)
--demonstration_steps # Demo frequency (default: 25)
```

## 🎮 Example Usage

### Basic QQP Paraphrase Training
```bash
python train_evaluate_test_conditional.py \
    --task_name qqp \
    --epochs 3 \
    --batch_size 8 \
    --max_samples 500 \
    --num_steps 32
```

### Advanced Configuration
```bash
python train_evaluate_test_conditional.py \
    --task_name qqp \
    --epochs 5 \
    --batch_size 16 \
    --max_samples 1000 \
    --num_steps 64 \
    --lr 2e-5 \
    --word_freq_lambda 0.2 \
    --hybrid_lambda 5e-4 \
    --from_scratch True
```

## 📈 What You'll See During Training

### 1. Data Loading Phase
```
============================================================
LOADING AND SPLITTING DATA
============================================================
✅ Loaded qqp dataset
   Original train size: 363846
   Original validation size: 40430

Data split (total: 1000 samples):
  📚 Train: 700 samples (70.0%)
  📊 Eval:  200 samples (20.0%)
  🧪 Test:  100 samples (10.0%)

📚 TRAIN DATA EXAMPLES:
  Example 1:
    Source: How do I create a new Yahoomail account?
    Target: How can I create a new Yahoo account?
```

### 2. Training Progress
```
🎯 STARTING TRAINING (3 epochs)
==================================================

📚 Epoch 1/3
----------------------------------------
Step 10: Loss = 8.2431
Step 20: Loss = 7.9876
```

### 3. Diffusion Demonstrations
```
🎭 DIFFUSION DEMONSTRATION (Step 25)
==================================================
📝 Original: How do I create a new Yahoo account?
🎯 Source (kept): How do I create a new 
🎯 Target (to generate): Yahoo account?

⏰ Timestep 1/32:
   🔀 Corrupted: How do I create a new [MASK] [MASK]?
   🔧 Denoised: How do I create a new email account?

⏰ Timestep 16/32:  
   🔀 Corrupted: How do I create a new [MASK] [MASK]?
   🔧 Denoised: How do I create a new Yahoo account?
```

### 4. Evaluation Results
```
📊 EVALUATING MODEL
========================================
📈 Evaluation Results:
   Average Loss: 6.7432
   Success Rate: 94.50%
   Total Samples: 189
   Failed Batches: 1

💎 New best model saved! Eval loss: 6.7432
```

### 5. Final Testing
```
🧪 TESTING MODEL
========================================

🔬 Test Batch 1:
------------------------------
  Original 1: How can I improve my English speaking skills?
    Corrupted (30%): How can I improve my [MASK] speaking skills?
    Denoised: How can I improve my English speaking skills?
    Similarity: 100%

📋 Test Summary:
   Samples tested: 12
   Average similarity: 87%

🏆 Best reconstruction (similarity: 100%):
   Original: How can I improve my English speaking skills?
   Denoised: How can I improve my English speaking skills?
```

## 💾 Output Files

Training creates a timestamped directory with:

```
model_name_bert-base-uncased_taskname_qqp_[...]/
├── best_model.pt           # Best model checkpoint
├── epoch_1.pt             # Epoch checkpoints  
├── epoch_2.pt
├── epoch_3.pt
├── final_model.pt         # Final model
└── training_results.json  # Complete training log
```

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
--batch_size 4

# Reduce max samples  
--max_samples 200

# Use CPU
--device cpu
```

**2. Data Loading Errors**
```bash
# Check if dataset exists
# Try different task: --task_name QT
# Reduce samples: --max_samples 100
```

**3. Training Slow**
```bash
# Reduce diffusion steps
--num_steps 16

# Smaller dataset
--max_samples 500

# Fewer epochs
--epochs 2
```

### Performance Tips

1. **Start Small**: Use Quick Demo first
2. **Monitor Memory**: Watch GPU memory usage
3. **Check Demonstrations**: Ensure model is learning
4. **Save Frequently**: Models save automatically
5. **Use GPU**: Much faster than CPU

## 🎯 Expected Results

### Good Training Signs
- ✅ Loss decreasing steadily
- ✅ High evaluation success rate (>90%)
- ✅ Demonstrations show improving quality
- ✅ Test similarity scores >70%

### Warning Signs
- ❌ Loss not decreasing after many steps
- ❌ Low success rate (<50%)
- ❌ Demonstrations show poor quality
- ❌ Test similarity scores <30%

## 🎉 Next Steps

After successful training:

1. **Load Best Model**: Use `best_model.pt` for inference
2. **Fine-tune Further**: Continue training with different data
3. **Experiment**: Try different tasks (QT, wiki_alignment)
4. **Scale Up**: Increase epochs/samples for better quality

## 📚 Additional Resources

- Original Diffusion-BERT paper
- Hugging Face Transformers documentation  
- PyTorch diffusion tutorials
- Conditional text generation guides

---

🎯 **Ready to start?** Run `python run_conditional_training.py` and select your configuration! 