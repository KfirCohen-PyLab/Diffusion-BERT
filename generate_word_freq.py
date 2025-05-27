import json
import os
from collections import Counter
from transformers import BertTokenizer
import torch

def create_sample_text():
    """Create sample text for word frequency calculation"""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
        "Actions speak louder than words.",
        "Where there's a will there's a way.",
        "Knowledge is power, but wisdom is divine.",
        "Time and tide wait for no man.",
        "Practice makes perfect.",
        "Better late than never.",
        # Add more common English sentences
        "Life is what happens while you're busy making other plans.",
        "Every cloud has a silver lining.",
        "Rome wasn't built in a day.",
        "Two wrongs don't make a right.",
        "When in Rome, do as the Romans do.",
        "The pen is mightier than the sword.",
        "Fortune favors the bold.",
        "A picture is worth a thousand words.",
        "Beauty is in the eye of the beholder.",
        "Necessity is the mother of invention."
    ]

def main():
    print("Starting word frequency generation...")
    
    # Initialize tokenizer
    print("Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Get sample text
    texts = create_sample_text()
    print(f"Created {len(texts)} sample texts")
    
    # Tokenize and count words
    print("Tokenizing texts and counting words...")
    word_counter = Counter()
    for text in texts:
        tokens = tokenizer.tokenize(text)
        word_counter.update(tokens)
    
    # Calculate frequencies
    print("Calculating word frequencies...")
    total_words = sum(word_counter.values())
    word_freqs = {word: count/total_words for word, count in word_counter.items()}
    
    # Save as JSON
    print("Saving word frequencies to JSON...")
    with open('word_freq.json', 'w', encoding='utf-8') as f:
        json.dump(word_freqs, f, ensure_ascii=False, indent=2)
    
    # Create and save tensor
    print("Creating frequency tensor...")
    vocab_size = tokenizer.vocab_size
    freq_tensor = torch.zeros(vocab_size)
    for word, freq in word_freqs.items():
        if word in tokenizer.vocab:
            freq_tensor[tokenizer.vocab[word]] = freq
    
    # Save tensor
    print("Saving frequency tensor...")
    torch.save(freq_tensor, 'word_freq.pt')
    
    print("\nWord frequency generation complete!")
    print(f"Found {len(word_freqs)} unique tokens")
    print("Files saved:")
    print("- word_freq.json")
    print("- word_freq.pt")

if __name__ == "__main__":
    main() 