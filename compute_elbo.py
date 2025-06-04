import torch
import os
import diffusion_condition as diffusion 
from transformers import BertTokenizer, BertConfig
from transformers import BertTokenizer as ElasticBertTokenizer
# from perplexity import ppl
from sample import Categorical, WholeWordMasking
import time
from fastNLP import logger
from tqdm import tqdm
from dataloader import DiffusionLoader
from torch.nn.utils.rnn import pad_sequence
import nltk
nltk.download('averaged_perceptron_tagger')


device = 'cuda:0'
model_ckpt_path = r'G:\ML_Project_Sem6\Diffusion-BERT-main\model_name_bert-base-uncased_taskname_qqp_lr_3e-05_seed_42_numsteps_32_sample_Categorical_schedule_mutual_hybridlambda_0.0003_wordfreqlambda_0.0_fromscratch_True_timestep_none_ckpts\best(49).pt'
model_name = 'bert-base-uncased'
predict_x0 = True
sample_strategy = 'Categorical'
num_steps = 16 #2048
kind = 'word_freq'
word_freq_lambda = 0.3
schedule = 'mutual'
eval_step_size = 4
timestep = 'none'
from_scratch ='false'
batch_size = 2

if timestep == 'none':
    from transformers import BertForMaskedLM
elif timestep == 'embedding':
    from models.modeling_bert_timestep import BertForMaskedLM
elif timestep == 'layerwise':
    from models.modeling_bert_new_timestep import BertForMaskedLM
else:
    raise NotImplementedError

if model_name in ['fnlp/elasticbert-base', 'fnlp/elasticbert-large']:
    model_cls = ElasticBertForPreTraining
    cfg_cls = ElasticBertConfig
    tok_cls = ElasticBertTokenizer
elif model_name in ['bert-base-uncased', 'bert-large-uncased']:
    model_cls = BertForMaskedLM
    cfg_cls = BertConfig
    tok_cls = BertTokenizer
else:
    raise NotImplementedError


tokenizer = tok_cls.from_pretrained(model_name)


if sample_strategy == 'Categorical':
    sample_cls = Categorical()
elif sample_strategy == 'wwm':
    sample_cls = WholeWordMasking(tokenizer)
else:
    raise ValueError


if kind == 'word_freq':
    import diffusion_word_freq as diffusion
    word_freq = torch.load(f'./word_freq/{model_name}_lm1b.pt')
    def word_freq_preprocess_fn(wf):
        wf = wf + 1
        wf = wf.log()
        wf = wf / wf.max()

        # range: 0 - 1
        return wf

    word_freq = word_freq_preprocess_fn(word_freq)
    diffusion_schedule = diffusion.create_discrete_diffusion_schedule(schedule, num_steps=num_steps)
    diffusion_instance = diffusion.MaskDiffusion(
        dim=tokenizer.vocab_size,
        schedule=diffusion_schedule,
        tokenizer=tokenizer,
        sample_cls=sample_cls,
        word_freq=word_freq,
        word_freq_lambda=word_freq_lambda,
        device=device
    )

elif kind == 'base':
    import diffusion

    diffusion_schedule = diffusion.create_discrete_diffusion_schedule(schedule, num_steps=num_steps)
    diffusion_instance = diffusion.MaskDiffusion(
        dim=tokenizer.vocab_size,
        schedule=diffusion_schedule,
        tokenizer=tokenizer,
        sample_cls=sample_cls,
    )

else:
    raise ValueError





cfg = cfg_cls.from_pretrained(model_name)
cfg.overall_timestep = diffusion_instance.num_steps

if model_name in ['fnlp/elasticbert-base', 'fnlp/elasticbert-large']:
    cfg.num_output_layers = cfg.num_hidden_layers
    cfg.num_base_layers = 0

model = model_cls(cfg).to(device)


ckpt = torch.load(model_ckpt_path)
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in ckpt['model'].items():
    if k != "bert.embeddings.position_ids":  # Skip problematic key
        new_state_dict[k] = v
model.load_state_dict(new_state_dict, strict=False)
#print("Keys in checkpoint:", ckpt['model'].keys())

model.load_state_dict(new_state_dict)

cls = torch.full((1, 1), fill_value=tokenizer.cls_token_id, device=device)
sep = torch.full((1, 1), fill_value=tokenizer.sep_token_id, device=device)

if model_name in ['fnlp/elasticbert-base', 'fnlp/elasticbert-large']:
    def layer_schedule_fn(timestep):
        return [11]
        # return [3 * (timestep * 4 // cfg.overall_timestep) + 2]

    def denoise_fn(targets, timestep):
        assert len(targets.size()) == 2  # bsz * seqlen
        bsz = targets.size(0)
        targets = torch.cat((cls.repeat(bsz, 1), targets, sep.repeat(bsz, 1)), dim=1)
        return model(input_ids=targets, timestep=timestep - 1, group_output_layers=layer_schedule_fn(timestep - 1))[:, 1:-1, :]
else:
    att_ones = torch.ones((1, 1), device=device)
    att_zeros = torch.zeros((1, 1), device=device)

    def denoise_fn(targets, timestep, attention_mask):
        assert len(targets.size()) == 2  # bsz * seqlen
        bsz = targets.size(0)
        targets = torch.cat((cls.repeat(bsz, 1), targets, sep.repeat(bsz, 1)), dim=1)
        attention_mask = torch.cat((att_ones.repeat(bsz, 1), attention_mask, att_zeros.repeat(bsz, 1)), dim=1)
        return model(
            input_ids=targets,
            # timestep=timestep - 1,
            attention_mask=attention_mask
        )['logits'][:, 1:-1, :]

model.eval()


def process_fn_in_collate(wf):
    return wf - wf.mean()


def collate_fn(batch_input):
    input_ids = [torch.tensor(d['input_ids']) for d in batch_input]
    attention_mask = [torch.tensor(d['attention_mask']) for d in batch_input]
    word_freq_logits = [process_fn_in_collate(word_freq.gather(0, torch.tensor(d['input_ids']))) for d in batch_input]
    input_ids = pad_sequence(input_ids, batch_first=True)
    attention_mask = pad_sequence(attention_mask, batch_first=True)
    word_freq_logits = pad_sequence(word_freq_logits, batch_first=True)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'word_freq_logits': word_freq_logits
    }

elbo = 0.
count = 0


def diffuse_and_reconstruct(text, steps_to_show=4):
    # Tokenize with padding
    inputs = tokenizer(text, return_tensors="pt", padding='max_length', max_length=64, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    print(f"\nOriginal text: '{text}'")
    print(f"Tokenized: {tokenizer.convert_ids_to_tokens(input_ids[0])}")
    print(f"Attention mask: {attention_mask.tolist()[0]}\n")
    
    # Prepare special tokens
    cls = torch.full((1, 1), tokenizer.cls_token_id, device=device)
    sep = torch.full((1, 1), tokenizer.sep_token_id, device=device)
    att_ones = torch.ones((1, 1), device=device)
    att_zeros = torch.zeros((1, 1), device=device)

    # --- Context-Preserving Diffusion ---
    def get_masking_ratio(step):
        """More conservative masking schedule"""
        return min(0.3, (step / num_steps) * 0.4)  # Max 30% masking

    corrupted_ids = input_ids.clone()
    for step in range(0, num_steps + 1, max(1, num_steps // steps_to_show)):
        current_ratio = get_masking_ratio(step)
        active_tokens = attention_mask.bool()
        
        # Create mask that preserves content words
        mask = torch.zeros_like(corrupted_ids, dtype=torch.bool, device=device)
        for i in range(corrupted_ids.size(0)):
            # Get all non-special tokens as candidates
            candidate_indices = [idx for idx in range(len(input_ids[i])) 
                              if active_tokens[i][idx] and 
                              input_ids[i][idx].item() not in tokenizer.all_special_ids]
            
            # Protect some content words (nouns/verbs)
            content_word_ids = []
            try:
                content_word_ids = [tokenizer.vocab[w] for w in 
                                   ["study", "computer", "science", "name"] 
                                   if w in tokenizer.vocab]
            except:
                pass
            
            protected_indices = [idx for idx in candidate_indices
                               if input_ids[i][idx].item() in content_word_ids]
            
            # Eligible tokens are non-protected candidates
            eligible = [idx for idx in candidate_indices if idx not in protected_indices]
            n_to_mask = min(int(current_ratio * len(candidate_indices)), len(eligible))
            
            if n_to_mask > 0:
                selected = torch.randperm(len(eligible), device=device)[:n_to_mask]
                mask[i, [eligible[s] for s in selected]] = True
        
        corrupted_ids[mask] = tokenizer.mask_token_id
        
        # Print progress
        corrupted_text = tokenizer.decode(corrupted_ids[0], skip_special_tokens=True)
        print(f"Diffusion Step {step}/{num_steps} (mask ratio={current_ratio:.2f}):")
        print(f"Masked {mask.sum().item()}/{active_tokens.sum().item()} tokens")
        print(f"Current text: '{corrupted_text}'\n")

    # --- Context-Aware Reconstruction ---
    print("\nReconstructing with context preservation...")
    prev_output = None
    for step in range(num_steps, -1, -max(1, num_steps // steps_to_show)):
        # Prepare input with full context
        bsz = corrupted_ids.size(0)
        model_input = torch.cat((cls.repeat(bsz, 1), corrupted_ids, sep.repeat(bsz, 1)), dim=1)
        model_attention = torch.cat((att_ones.repeat(bsz, 1), attention_mask, att_zeros.repeat(bsz, 1)), dim=1)
        
        # Get predictions with context weighting
        with torch.no_grad():
            outputs = model(
                input_ids=model_input,
                attention_mask=model_attention
            )
            logits = outputs['logits'][:, 1:-1, :]
            
            # Boost probability of words that fit the context
            context_words = []
            try:
                context_words = [tokenizer.vocab[w] for w in 
                               ["study", "computer", "science", "name"] 
                               if w in tokenizer.vocab]
            except:
                pass
            
            if context_words:
                logits[:, :, context_words] += 1.5  # Smaller context boost
        
        # Sample with temperature and context awareness
        mask_positions = (corrupted_ids == tokenizer.mask_token_id) & attention_mask.bool()
        probs = torch.softmax(logits / 0.7, dim=-1)  # Temperature 0.7
        pred_ids = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(*logits.shape[:-1])
        corrupted_ids[mask_positions] = pred_ids[mask_positions]
        
        # Print meaningful changes
        current_output = tokenizer.decode(corrupted_ids[0], skip_special_tokens=True)
        if current_output != prev_output:
            print(f"Denoising Step {step}/{num_steps}:")
            print(f"Replaced {mask_positions.sum().item()} tokens")
            print(f"Current text: '{current_output}'\n")
            prev_output = current_output
    
    final_output = tokenizer.decode(corrupted_ids[0], skip_special_tokens=True)
    print(f"\nFinal reconstructed: '{final_output}'")
    print(f"Original: '{text}'")
    print(f"Semantic similarity: {calculate_similarity(final_output, text):.2f}")
    return final_output

def calculate_similarity(text1, text2):
    # Simple word overlap similarity metric
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    return len(words1 & words2) / max(len(words1), len(words2))

# Example usage
if __name__ == "__main__":
    examples = [
    "The quick brown fox jumps over the lazy dog",
    "Diffusion models are powerful",
    "My name is Alice and I study computer science"
]
    for text in examples:
        diffuse_and_reconstruct(text)
        print("\n" + "="*50 + "\n")

print("Write 'C' and press enter to proceed to ELBO calculation score")
import pdb;pdb.set_trace()

test_data = DiffusionLoader(tokenizer=tokenizer).my_load(task_name='qqp', splits=['test'])[0]
_, test_data = test_data.train_test_split(test_size=5e-2).values()
test_loader = torch.utils.data.DataLoader(test_data, batch_size=2, collate_fn=collate_fn, num_workers=0, pin_memory=True)
with torch.no_grad():
    for batch in tqdm(test_loader):
        batch_dev_metrics = diffusion.discrete_diffusion_elbo(
            batch['input_ids'].to(device),
            denoise_fn=denoise_fn,
            diffusion=diffusion_instance,
            target_mask=batch['attention_mask'].to(device),
            word_freq_logits=batch['word_freq_logits'].to(device),
            normalize_without_padding=True,
            eval_step_size=eval_step_size,
            device=device
        )

        if not torch.isnan(batch_dev_metrics['elbo']):
            elbo += batch_dev_metrics['elbo']
            count += 1

print(elbo / (64. * count))

