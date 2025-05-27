import json
import os
from collections import Counter
from tqdm import tqdm
from transformers import BertTokenizer
import torch

def load_jsonl(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in {file_path}: {e}")
        return []
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []

def main():
    # Create output directory if it doesn't exist
    os.makedirs('word_freqs', exist_ok=True)
    
    # Initialize tokenizer
    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return
    
    # Load all data
    all_texts = []
    files_found = False
    
    print("Loading data files...")
    for i in tqdm(range(50000)):  # Adjust range based on your dataset size
        try:
            data = load_jsonl(f'train_{i}.jsonl')
            if data:
                all_texts.extend([item['text'] for item in data])
                files_found = True
        except FileNotFoundError:
            continue
    
    if not files_found:
        print("No valid data files found. Please check the data directory.")
        return
    
    if not all_texts:
        print("No texts loaded. Please check the data files.")
        return
    
    print(f"\nProcessing {len(all_texts)} texts...")
    
    # Tokenize all texts
    word_counter = Counter()
    for text in tqdm(all_texts, desc="Tokenizing texts"):
        try:
            tokens = tokenizer.tokenize(text)
            word_counter.update(tokens)
        except Exception as e:
            print(f"Error tokenizing text: {e}")
            continue
    
    if not word_counter:
        print("No tokens found. Please check the texts and tokenizer.")
        return
    
    # Save word frequencies
    total_words = sum(word_counter.values())
    word_freqs = {word: count/total_words for word, count in word_counter.items()}
    
    # Save as JSON
    json_path = 'word_freqs/word_freq.json'
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(word_freqs, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving word frequencies to JSON: {e}")
        return
    
    # Save as PyTorch tensor
    pt_path = 'word_freqs/word_freq.pt'
    try:
        # Create tensor of word frequencies in vocab order
        vocab_size = tokenizer.vocab_size
        freq_tensor = torch.zeros(vocab_size)
        for word, freq in word_freqs.items():
            if word in tokenizer.vocab:
                freq_tensor[tokenizer.vocab[word]] = freq
        torch.save(freq_tensor, pt_path)
    except Exception as e:
        print(f"Error saving word frequencies to tensor: {e}")
        return
    
    print(f"\nProcessed {len(all_texts)} texts")
    print(f"Found {len(word_freqs)} unique tokens")
    print(f"Word frequencies saved to {json_path} and {pt_path}")

if __name__ == "__main__":
    main() 