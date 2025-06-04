import torch
import os
import diffusion_word_freq as diffusion
from transformers import BertTokenizer, BertConfig
from transformers import BertTokenizer as ElasticBertTokenizer
#from models.modeling_elasticbert import ElasticBertForPreTraining
#from models.configuration_elasticbert import ElasticBertConfig
# from perplexity import ppl
from sample import Categorical, WholeWordMasking
import time
from fastNLP import logger
from tqdm import tqdm
from dataloader import DiffusionLoader
from torch.nn.utils.rnn import pad_sequence

from datasets import load_dataset

# Load LM1B dataset
dataset = load_dataset("lm1b")

# Split into train/test (example)
train_data = dataset["train"]
test_data = dataset["test"]


device = 'cuda:0'
model_ckpt_path = 'G:\ML_Project_Sem6\Diffusion-BERT-main\model_name_roberta-base_taskname_qqp_lr_3e-05_seed_42_numsteps_32_sample_Categorical_schedule_mutual_hybridlambda_0.0003_wordfreqlambda_0.0_fromscratch_True_timestep_token_ckpts\best(1999).th'
model_name = 'bert-base-uncased'
predict_x0 = True
sample_strategy = 'Categorical'
num_steps = 512 #2048
kind = 'word_freq'
word_freq_lambda = 0.3
schedule = 'mutual'
eval_step_size = 16
timestep = 'none'

if timestep == 'none':
    from transformers import BertForMaskedLM
elif timestep == 'embedding':
    from models.modeling_bert_timestep import BertForMaskedLM
elif timestep == 'layerwise':
    from models.modeling_bert_new_timestep import BertForMaskedLM
else:
    raise NotImplementedError

try:
    if model_name in ['bert-base-uncased', 'bert-large-uncased']:
        model_cls = BertForMaskedLM
        cfg_cls = BertConfig
        tok_cls = BertTokenizer
    else:
        raise NotImplementedError
except:
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
    word_freq = torch.load(r'G:\ML_Project_Sem6\Diffusion-BERT-main\word_freq\bert-base-uncased_lm1b.pt')
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

print(cfg)
print("\n")


model = model_cls(cfg).to(device)

print(model)
print("\n")

ckpt = torch.load(model_ckpt_path, map_location=device)
state_dict = ckpt.get('model_state_dict', ckpt.get('model', ckpt))


#model.load_state_dict(ckpt['model'])
model.load_state_dict(state_dict,strict=False)

cls = torch.full((1, 1), fill_value=tokenizer.cls_token_id, device=device)
sep = torch.full((1, 1), fill_value=tokenizer.sep_token_id, device=device)


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

# test_data = DiffusionLoader(tokenizer=tokenizer).my_load(task_name='lm1b', splits=['test'])[0]
# _, test_data = test_data.train_test_split(test_size=5e-2).values()

from sklearn.model_selection import train_test_split

test_data = DiffusionLoader(tokenizer=tokenizer).my_load(task_name='lm1b', splits=['test'])[0]
# Convert to list of examples first if needed
_, test_data = train_test_split(test_data, test_size=0.05, random_state=42)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, collate_fn=collate_fn, num_workers=4, pin_memory=True)
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
            logger.info(batch_dev_metrics['elbo'])
            elbo += batch_dev_metrics['elbo']
            count += 1

print(elbo / (64. * count))