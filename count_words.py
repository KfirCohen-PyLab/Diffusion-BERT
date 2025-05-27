import os
import torch
from transformers import BertTokenizer
import datasets
from tqdm import tqdm

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load dataset
print("Loading dataset...")
dataset = datasets.load_dataset('lm1b', split='train[:50000]')

# Initialize word frequency counter
word_freq = torch.zeros((tokenizer.vocab_size,), dtype=torch.int64)

# Process each example
print("Processing examples...")
for example in tqdm(dataset):
    # Tokenize text
    tokens = tokenizer.encode(example['text'], max_length=128, truncation=True, add_special_tokens=False)
    # Count frequencies
    for token_id in tokens:
        word_freq[token_id] += 1

# Create output directory if it doesn't exist
if not os.path.exists('./word_freq'):
    os.mkdir('word_freq')

# Save word frequencies
print("Saving word frequencies...")
torch.save(word_freq, './word_freq/bert-base-uncased_lm1b.pt')
print("Done!") 