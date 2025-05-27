import datasets
import os
from functools import partial
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DiffusionLoader:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def _load(self, task_name: str, split: str) -> datasets.Dataset:
        try:
            dataset = datasets.load_dataset('lm1b', split=split)
            logger.info(f'Example in {split} set:')
            logger.info(dataset[0])
            
            dataset = dataset.map(
                partial(self.convert_to_features, tokenizer=self.tokenizer),
                batched=True,
                remove_columns='text',
                desc=f"Processing {split} split"
            )
            return dataset
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def my_load(self, task_name: str, splits: List[str]) -> List[datasets.Dataset]:
        return [self._load(task_name, name) for name in splits]

    @staticmethod
    def convert_to_features(example_batch: Dict[str, List[str]], tokenizer: PreTrainedTokenizer) -> Dict[str, List[List[int]]]:
        try:
            input_encodings = tokenizer.batch_encode_plus(
                example_batch['text'],
                max_length=128,
                truncation=True,
                add_special_tokens=False,
                return_tensors=None
            )
            
            return {
                'input_ids': input_encodings['input_ids'],
                'attention_mask': input_encodings['attention_mask'],
            }
        except Exception as e:
            logger.error(f"Error converting features: {e}")
            raise

def collate_fn(batch_input: List[Dict[str, torch.Tensor]], word_freq: torch.Tensor, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    try:
        input_ids = [torch.tensor(d['input_ids'], device=device) for d in batch_input]
        attention_mask = [torch.tensor(d['attention_mask'], device=device) for d in batch_input]
        
        # Process word frequencies
        if word_freq is not None:
            word_freq = word_freq.to(device) if device is not None else word_freq
            word_freq_logits = [word_freq.gather(0, torch.tensor(d['input_ids'], device=device)) for d in batch_input]
            word_freq_logits = pad_sequence(word_freq_logits, batch_first=True)
        else:
            word_freq_logits = None
        
        # Pad sequences
        input_ids = pad_sequence(input_ids, batch_first=True)
        attention_mask = pad_sequence(attention_mask, batch_first=True)
        
        output = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        
        if word_freq_logits is not None:
            output['word_freq_logits'] = word_freq_logits
            
        return output
    except Exception as e:
        logger.error(f"Error in collate function: {e}")
        raise

def load_word_frequencies(path: str, device: Optional[torch.device] = None) -> torch.Tensor:
    try:
        if path.endswith('.pt'):
            freq = torch.load(path, map_location=device)
        elif path.endswith('.json'):
            import json
            with open(path, 'r') as f:
                freq_dict = json.load(f)
            freq = torch.tensor([freq_dict.get(str(i), 0.0) for i in range(len(freq_dict))], device=device)
        else:
            raise ValueError(f"Unsupported file format for word frequencies: {path}")
        
        return freq
    except Exception as e:
        logger.error(f"Error loading word frequencies: {e}")
        raise

class ConditionalLoader:
    def __init__(self, tokenizer, return_source_length=False):
        self.tokenizer = tokenizer
        self.return_source_length = return_source_length
        self.data_dir = './conditional_data'

    @staticmethod
    def _convert_to_features_original(example_batch, tokenizer):
        q1 = tokenizer.batch_encode_plus(example_batch['src'], max_length=128, truncation=True, add_special_tokens=False)
        q2 = tokenizer.batch_encode_plus(example_batch['trg'], max_length=128, truncation=True, add_special_tokens=False)
        return {
            'source': q1['input_ids'],
            'target': q2['input_ids'],
        }

    def load_original(self, split):
        dataset = datasets.load_dataset(os.path.join(self.data_dir, self.task_name, f'{self.task_name}.py'), split=split)
        dataset = dataset.map(partial(self._convert_to_features_original, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        print(f'Example in {split} set:')
        print(dataset[0])
        return dataset

    def _load(self, split):
        dataset = datasets.load_dataset(os.path.join(self.data_dir, self.task_name, f'{self.task_name}.py'), split=split)
        if self.return_source_length:
            dataset = dataset.map(partial(self.add_original_src_length, tokenizer=self.tokenizer))
        dataset = dataset.map(self.add_prompt)
        dataset = dataset.map(partial(self.convert_to_features, tokenizer=self.tokenizer), batched=True)
        print(f'Example in {split} set:')
        print(dataset[0])
        return dataset

    def add_original_src_length(self, example, tokenizer):
        return {
            'original_src_length': len(tokenizer.encode(example['src'], max_length=128, truncation=True, add_special_tokens=False))
        }

    def my_load(self, splits):
        return [self._load(name) for name in splits]

    @staticmethod
    def convert_to_features(example_batch, tokenizer):
        q1 = tokenizer.batch_encode_plus(example_batch['src'], max_length=128, truncation=True, add_special_tokens=False)
        q2 = tokenizer.batch_encode_plus(example_batch['trg'], max_length=128, truncation=True, add_special_tokens=False)
        encodings = {
            'source': q1['input_ids'],
            'target': q2['input_ids'],
        }

        return encodings

    @staticmethod
    def collate_fn(batch_input, tokenizer):
        input_ids = pad_sequence([torch.tensor(
            [tokenizer.cls_token_id] + d['source'] + d['target'] + [tokenizer.sep_token_id]
        ) for d in batch_input], batch_first=True)

        attention_mask = torch.ones_like(input_ids)

        target_mask = torch.stack([torch.cat([
            torch.zeros(len(d['source']) + 1), torch.ones(input_ids.size(1) - len(d['source']) - 1)
        ]) for d in batch_input])

        assert input_ids.size() == attention_mask.size() == target_mask.size()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask,
        }

class QQPLoader(ConditionalLoader):
    def __init__(self, tokenizer, return_source_length=False):
        super(QQPLoader, self).__init__(tokenizer, return_source_length)
        self.task_name = 'qqp'

    @staticmethod
    def add_prompt(example):
        example['src'] = '"' + example['src'] + '" is equal to "'
        example['trg'] = example['trg']
        return example



class QTLoader(ConditionalLoader):
    def __init__(self, tokenizer, return_source_length=False):
        super(QTLoader, self).__init__(tokenizer, return_source_length)
        self.task_name = 'Q-T'

    @staticmethod
    def add_prompt(example):
        example['src'] = ' Answer: ' + example['src'] + ' Question: '
        example['trg'] = example['trg']
        return example


class WikiLoader(ConditionalLoader):
    def __init__(self, tokenizer, return_source_length=False):
        super(WikiLoader, self).__init__(tokenizer, return_source_length)
        self.task_name = 'wiki_alignment'

    @staticmethod
    def add_prompt(example):
        example['src'] = '"' + example['src'] + '" can be summarized as: '
        example['trg'] = example['trg']
        return example

class CCLoader(ConditionalLoader):
    def __init__(self, tokenizer, return_source_length=False):
        super(CCLoader, self).__init__(tokenizer, return_source_length)
        self.task_name = 'CC'

    @staticmethod
    def add_prompt(example):
        example['src'] = example['src'] + ' - '
        example['trg'] = example['trg']
        return example


class DiffusionLoaderWithElectra(DiffusionLoader):
    def __init__(self, model_tokenizer, electra_tokenizer, electra_model):
        super().__init__(model_tokenizer)
        self.electra_tokenizer = electra_tokenizer
        self.electra_model = electra_model

    def _load(self, task_name, split):
        dataset = datasets.load_dataset(f'./dataloaders/{task_name}.py', split=split)
        print(f'Example in {split} set:')
        print(dataset[0])
        dataset = dataset.map(partial(self.new_convert_to_features, model_tokenizer=self.tokenizer, electra_tokenizer=self.electra_tokenizer, electra_model=self.electra_model), batched=True, remove_columns='text')
        return dataset

    @staticmethod
    def new_convert_to_features(example_batch, model_tokenizer, electra_tokenizer, electra_model):
        input_encodings = model_tokenizer.batch_encode_plus(example_batch['text'], max_length=256, truncation=True, add_special_tokens=False)
        electra_encodings = electra_tokenizer.batch_encode_plus(example_batch['text'], max_length=256, truncation=True, padding=True, return_tensors='pt', add_special_tokens=False)
        for k in electra_encodings.keys():
            electra_encodings[k] = electra_encodings[k].cuda()
        position = electra_encodings['attention_mask'].count_nonzero(1)
        with torch.no_grad():
            logits = electra_model(**electra_encodings)


        encodings = {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'electra_logits': [logits[i][:position[i]] for i in range(position.size(0))]
        }

        return encodings



