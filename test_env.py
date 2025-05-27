import torch
import transformers
from transformers import BertTokenizer

print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")

# Test tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
test_text = "Hello world! This is a test of the tokenizer."
tokens = tokenizer.tokenize(test_text)
print(f"\nTest text: {test_text}")

import torch
import transformers
from transformers import BertTokenizer

print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")

# Test tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
test_text = "Hello world! This is a test of the tokenizer."
tokens = tokenizer.tokenize(test_text)
print(f"\nTest text: {test_text}")
 
print(f"Tokenized: {tokens}") 