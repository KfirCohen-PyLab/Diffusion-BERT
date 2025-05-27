import json
from collections import Counter
from tqdm import tqdm
from transformers import BertTokenizer

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def main():
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Load all data
    all_texts = []
    for i in range(50000):  # Adjust range based on your dataset size
        try:
            data = load_jsonl(f'train_{i}.jsonl')
            all_texts.extend([item['text'] for item in data])
        except FileNotFoundError:
            continue
    
    # Tokenize all texts
    word_counter = Counter()
    for text in tqdm(all_texts, desc="Processing texts"):
        tokens = tokenizer.tokenize(text)
        word_counter.update(tokens)
    
    # Save word frequencies
    total_words = sum(word_counter.values())
    word_freqs = {word: count/total_words for word, count in word_counter.items()}
    
    with open('word_freq.json', 'w') as f:
        json.dump(word_freqs, f)
    
    print(f"Processed {len(all_texts)} texts")
    print(f"Found {len(word_freqs)} unique tokens")
    print("Word frequencies saved to word_freq.json")

if __name__ == "__main__":

import json
from collections import Counter
from tqdm import tqdm
from transformers import BertTokenizer

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def main():
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Load all data
    all_texts = []
    for i in range(50000):  # Adjust range based on your dataset size
        try:
            data = load_jsonl(f'train_{i}.jsonl')
            all_texts.extend([item['text'] for item in data])
        except FileNotFoundError:
            continue
    
    # Tokenize all texts
    word_counter = Counter()
    for text in tqdm(all_texts, desc="Processing texts"):
        tokens = tokenizer.tokenize(text)
        word_counter.update(tokens)
    
    # Save word frequencies
    total_words = sum(word_counter.values())
    word_freqs = {word: count/total_words for word, count in word_counter.items()}
    
    with open('word_freq.json', 'w') as f:
        json.dump(word_freqs, f)
    
    print(f"Processed {len(all_texts)} texts")
    print(f"Found {len(word_freqs)} unique tokens")
    print("Word frequencies saved to word_freq.json")

if __name__ == "__main__":
    main() 