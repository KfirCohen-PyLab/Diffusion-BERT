# DiffusionBERT

This is an implementation of DiffusionBERT: Improving Generative Masked Language Models with Diffusion Models.

## Quick Start with Google Colab

1. Open the `DiffusionBERT.ipynb` notebook in Google Colab:
   - Go to [Google Colab](https://colab.research.google.com)
   - Click File -> Upload Notebook
   - Select the `DiffusionBERT.ipynb` file from this repository

2. Make sure you're using a GPU runtime:
   - Click Runtime -> Change runtime type
   - Select "GPU" from the Hardware accelerator dropdown
   - Click Save

3. Run the cells in order:
   - The first cell clones this repository
   - The second cell installs required dependencies
   - The third cell downloads and prepares the LM1B dataset
   - The fourth cell calculates word frequencies
   - The fifth cell trains the model
   - The sixth cell generates text samples

## Local Setup

If you prefer to run the code locally:

1. Clone the repository:
```bash
git clone https://github.com/KfirCohen-PyLab/Diffusion-BERT.git
cd Diffusion-BERT
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download and prepare the dataset:
```bash
python -c "from datasets import load_dataset; dataset = load_dataset('lm1b', split='train[:50000]')"
```

4. Calculate word frequencies:
```bash
python word_freq.py
```

5. Train the model:
```bash
python main.py \
    --train_data_dir "./conditional_data" \
    --vocab_size 30522 \
    --block_size 128 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --num_train_epochs 100 \
    --gradient_accumulation_steps 2 \
    --model_type "bert-base-uncased" \
    --diffusion_steps 2000 \
    --noise_schedule "cosine" \
    --spindle_schedule True \
    --word_freq_file "word_freq.json" \
    --output_dir "./diffusion_models" \
    --num_workers 4 \
    --fp16 True
```

6. Generate text:
```bash
python predict.py \
    --checkpoint_path "./diffusion_models/checkpoint-10.pt" \
    --model_type "bert-base-uncased" \
    --vocab_size 30522 \
    --block_size 128 \
    --batch_size 4 \
    --diffusion_steps 2000 \
    --output_file "generated_texts.txt"
```

## Model Architecture

DiffusionBERT combines BERT with discrete diffusion models to improve text generation. Key features:

- Uses BERT as the base model for denoising
- Implements a discrete diffusion process with an absorbing state
- Includes a spindle schedule for noise addition based on token information
- Supports both unconditional and conditional text generation

## Citation

```bibtex
@article{he2022diffusionbert,
  title={DiffusionBERT: Improving Generative Masked Language Models with Diffusion Models},
  author={He, Zhengfu and Sun, Tianxiang and Wang, Kuanning and Huang, Xuanjing and Qiu, Xipeng},
  journal={arXiv preprint arXiv:2211.15029},
  year={2022}
}
```