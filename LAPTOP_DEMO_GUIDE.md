# 💻 Laptop Classroom Demo Guide
*Diffusion BERT demonstration optimized for CPU-only laptops*

## 🎯 Overview

This guide provides two educational demonstrations perfect for classroom presentations:

1. **CPU-Optimized Model Demo** - Shows trained Diffusion BERT running on laptop CPU
2. **Data Splitting Explanation** - Visualizes how we divided the dataset for training

## 📋 Prerequisites

- Python 3.7+
- Required packages: `torch`, `transformers`, `matplotlib`, `seaborn`, `pandas`, `scikit-learn`
- A trained Diffusion BERT model checkpoint (`.th` or `.pt` file)

## 🚀 Quick Start

### Option 1: Run with Batch File (Windows)
```batch
# Double-click or run in command prompt
run_cpu_demo.bat

# Or specify custom model path
run_cpu_demo.bat "path\to\your\model\final.th"
```

### Option 2: Run with Python Command
```bash
# Full demonstration (recommended for presentations)
python cpu_classroom_demo.py --checkpoint "your_model_path/final.th" --mode full

# Just the main demo without interaction
python cpu_classroom_demo.py --checkpoint "your_model_path/final.th" --mode demo

# Only interactive portion
python cpu_classroom_demo.py --checkpoint "your_model_path/final.th" --mode interactive
```

## 📊 Dataset Splitting Demonstration

### Show How Data Was Split
```bash
# Full demonstration with visualization
python show_data_splitting.py

# With training progress simulation
python show_data_splitting.py --show_progress

# Skip visualization creation
python show_data_splitting.py --no_viz

# Custom dataset path
python show_data_splitting.py --data_path "path/to/your/QQP/dataset"
```

## 🎓 Classroom Presentation Flow

### 1. Introduction (5 minutes)
```bash
python show_data_splitting.py --show_progress
```
**What this shows:**
- How we split 400,000 QQP samples into 70% train, 20% validation, 10% test
- Why this splitting strategy is important
- Training progress simulation
- Creates visualization charts

### 2. Live Model Demonstration (10 minutes)
```bash
python cpu_classroom_demo.py --checkpoint "best_model/final.th" --mode full
```
**What this shows:**
- Model loading on CPU (30-60 seconds)
- 4 pre-selected impressive examples
- Real-time text reconstruction
- Accuracy measurements
- Processing time on CPU

### 3. Interactive Session (5 minutes)
**What this includes:**
- Audience can suggest sentences to test
- Live reconstruction demonstration
- Discussion of results
- Q&A about the technology

## 🔧 Technical Details

### CPU Optimizations Applied
- **Thread optimization**: Set to 4 threads for typical laptop CPUs
- **Simplified sampling**: Uses argmax instead of complex sampling for speed
- **Memory efficiency**: Disabled gradients completely during inference
- **Shorter sequences**: Optimized for sentences under 12 words

### Performance Expectations
- **Model loading**: 30-60 seconds on typical laptop
- **Inference time**: 2-5 seconds per sentence on CPU
- **Memory usage**: ~2-4GB RAM during inference
- **Accuracy**: Should maintain 80-90% reconstruction quality

## 📈 Dataset Split Details

Our training used the **industry-standard 70-20-10 split**:

| Split | Samples | Percentage | Purpose |
|-------|---------|------------|---------|
| **Training** | 280,000 | 70% | Model learning |
| **Validation** | 80,000 | 20% | Hyperparameter tuning |
| **Test** | 40,000 | 10% | Final evaluation |

### Why This Split?
- **70% Training**: Large enough for BERT to learn language patterns
- **20% Validation**: Monitor progress, prevent overfitting
- **10% Test**: Unbiased final performance evaluation

## 🎯 Best Practices for Presentations

### Before Your Presentation
1. **Test your model**: Run the demo once to ensure everything works
2. **Check timing**: Full demo takes ~20 minutes
3. **Prepare backup**: Have example outputs ready if live demo fails
4. **Test sentences**: Prepare 2-3 backup sentences that work well

### During Presentation
1. **Start with data splitting** - Shows methodology
2. **Explain CPU optimization** - Why it works on laptops
3. **Run live demo** - Most impressive part
4. **Encourage interaction** - Let audience suggest test sentences

### Troubleshooting
- **Model not loading**: Check file path and file permissions
- **Slow performance**: Mention this is expected on CPU
- **Poor results**: Explain that model quality depends on training
- **Memory issues**: Close other applications, restart if needed

## 📝 Example Sentences That Work Well

These sentences typically produce good reconstruction results:

```
"Artificial intelligence is transforming modern technology"
"Students learn programming through practice and examples"
"Machine learning models process large amounts of data"
"Computer science involves algorithms and problem solving"
"Natural language processing helps computers understand text"
```

## 🔍 Understanding the Output

### What the Demo Shows
1. **Original sentence**: Input text
2. **Masked version**: ~30% of words replaced with [MASK]
3. **Reconstructed**: Model's prediction for masked words
4. **Accuracy**: Percentage of correctly reconstructed words
5. **Processing time**: How long inference took on CPU

### Interpreting Results
- **80-90% accuracy**: Excellent reconstruction
- **60-80% accuracy**: Good reconstruction  
- **<60% accuracy**: May need more training or better model

## 🎉 Educational Value

This demonstration teaches:
- **Machine Learning Pipeline**: Data splitting → Training → Evaluation
- **Model Architecture**: How BERT processes and reconstructs text
- **Real-world AI**: Practical applications of language models
- **Computational Considerations**: CPU vs GPU performance trade-offs

## 💡 Tips for Success

1. **Keep sentences simple**: 8-12 words work best on CPU
2. **Allow processing time**: Don't rush the inference
3. **Explain what's happening**: Narrate the process for audience
4. **Have backup plan**: Pre-computed examples if live demo fails
5. **Encourage questions**: Make it interactive and educational

---

*This demo successfully shows how Diffusion BERT learned to understand and reconstruct human language, all running on a standard laptop CPU!* 