import json
import os
from collections import Counter
from transformers import BertTokenizer
import torch
import numpy as np
from tqdm import tqdm
import requests
import gzip
from io import StringIO

def download_text_file():
    """Download a sample text file for word frequency calculation"""
    print("Downloading text data...")
    
    # Using the raw text file from LM1B (Google Billion Words)
    url = "http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz"
    
    try:
        # Create data directory if it doesn't exist
        os.makedirs('./data', exist_ok=True)
        
        # Download the file
        local_file = './data/lm1b.txt'
        
        if not os.path.exists(local_file):
            print(f"Downloading from {url}...")
            response = requests.get(url)
            response.raise_for_status()
            
            # Save the raw content
            with open(local_file, 'wb') as f:
                f.write(response.content)
            print("Download complete!")
        else:
            print("Using cached text file.")
        
        # Read the content
        with open(local_file, 'r', encoding='utf-8') as f:
            content = f.readlines()
        
        # Take first 100,000 lines or all if less
        content = content[:min(len(content), 100000)]
        return content
        
    except Exception as e:
        print(f"Error downloading/reading file: {str(e)}")
        # Fallback to a small sample text if download fails
        return [
            "The model learns to predict the next word in the sequence.",
            "Neural networks process information through layers.",
            "Transformers use self-attention mechanisms.",
            "Deep learning models require training data.",
            "BERT is a bidirectional encoder representation.",
            "Language models can understand text.",
            "Machine learning algorithms improve with examples.",
            "The attention mechanism helps focus on input.",
            "Tokenization breaks text into smaller units.",
            "Word embeddings represent words as vectors."
        ]

def main():
    print("Starting word frequency generation...")
    
    # Initialize tokenizer
    print("Loading BERT tokenizer...")
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # Create word_freq directory if it doesn't exist
    os.makedirs('./word_freq', exist_ok=True)
    
    # Get text data
    texts = download_text_file()
    print(f"Processing {len(texts)} lines of text")
    
    # Count word frequencies
    print("Counting word frequencies...")
    word_counter = Counter()
    
    # Process each line of text
    for text in tqdm(texts, desc="Processing texts"):
        if isinstance(text, str) and text.strip():
            # Tokenize the text
            tokens = tokenizer.encode(text.strip(), add_special_tokens=False)
            word_counter.update(tokens)
    
    # Create frequency tensor
    print("Creating frequency tensor...")
    vocab_size = tokenizer.vocab_size
    freq_tensor = torch.zeros(vocab_size)
    
    # Fill the tensor with counts
    for token_id, count in word_counter.items():
        if isinstance(token_id, int) and 0 <= token_id < vocab_size:
            freq_tensor[token_id] = count
    
    # Normalize to get frequencies
    total_tokens = freq_tensor.sum()
    if total_tokens > 0:
        freq_tensor = freq_tensor / total_tokens
    
    # Add smoothing to avoid zeros
    freq_tensor = freq_tensor + 1e-10
    freq_tensor = freq_tensor / freq_tensor.sum()
    
    # Save tensor
    save_path = f'./word_freq/{model_name}_lm1b.pt'
    print(f"Saving frequency tensor to {save_path}...")
    torch.save(freq_tensor, save_path)
    
    print("\nWord frequency generation complete!")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Total tokens processed: {int(total_tokens)}")
    print(f"File saved: {save_path}")
    
    # Print some statistics
    print("\nTop 10 most frequent tokens:")
    top_k = 10
    values, indices = torch.topk(freq_tensor, top_k)
    for i, (idx, val) in enumerate(zip(indices, values)):
        token = tokenizer.convert_ids_to_tokens([idx])[0]
        print(f"{i+1}. Token: {token}, Frequency: {val:.6f}")

if __name__ == "__main__":
    main() 