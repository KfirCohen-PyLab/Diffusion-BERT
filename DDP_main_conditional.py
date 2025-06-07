import functools
import os
import sys
import random
import numpy as np
import argparse
import torch
from dataloader import QQPLoader, QTLoader
from transformers import BertTokenizer, BertConfig, RobertaTokenizer, RobertaConfig
from models.modeling_bert import BertForMaskedLM
from models.modeling_roberta import RobertaForMaskedLM
import diffusion_condition
from torch.optim import AdamW
import fastNLP
from tqdm import tqdm
from sample import Categorical, WholeWordMasking
import datetime

scaler = torch.cuda.amp.GradScaler()

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=5, type=int, required=False)
    parser.add_argument("--model_name_or_path", default='bert-base-uncased', type=str, required=False)
    parser.add_argument("--task_name", default='qqp', type=str, required=False)
    parser.add_argument("--lr", default=3e-5, type=float, required=False)
    parser.add_argument("--batch_size", default=16, type=int, required=False)
    parser.add_argument("--word_freq_lambda", default=0.0, type=float, required=False)
    parser.add_argument("--num_steps", default=16, type=int, required=False)
    parser.add_argument("--eval_step_size", default=4, type=int, required=False)
    parser.add_argument("--accumulation_steps", default=2, type=int, required=False)
    parser.add_argument("--hybrid_lambda", default=3e-4, type=float, required=False)
    parser.add_argument("--eval_steps", default=200, type=int, required=False)
    parser.add_argument("--seed", default=42, type=int, required=False)
    parser.add_argument("--device", default='cuda:0', type=str, required=False)
    parser.add_argument("--logging_steps", default=100, type=int, required=False)
    parser.add_argument("--save_steps", default=500, type=int, required=False)
    parser.add_argument('--predict_x0', default=True, type=bool, required=False)
    parser.add_argument("--load_step", default=-1, type=int, required=False)
    parser.add_argument("--sample_strategy", default='Categorical', type=str, required=False)
    parser.add_argument("--schedule", default='mutual', type=str, required=False)
    parser.add_argument("--from_scratch", default=False, type=bool, required=False)
    parser.add_argument("--timestep", default='none', type=str, required=False)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args)

    Dataloaders = {
        'qqp': QQPLoader,
        'QT': QTLoader,
    }

    Loader = Dataloaders[args.task_name]

    save_path = f'./model_name_{args.model_name_or_path}_taskname_{args.task_name}_lr_{args.lr}_seed_{args.seed}_numsteps_{args.num_steps}_sample_{args.sample_strategy}_schedule_{args.schedule}_hybridlambda_{args.hybrid_lambda}_wordfreqlambda_{args.word_freq_lambda}_fromscratch_{args.from_scratch}_timestep_{args.timestep}_ckpts'
    
    if args.model_name_or_path in ['bert-base-uncased', 'bert-large-uncased']:
        model_cls = BertForMaskedLM
        cfg_cls = BertConfig
        tok_cls = BertTokenizer
    elif args.model_name_or_path in ['roberta-base']:
        model_cls = RobertaForMaskedLM
        cfg_cls = RobertaConfig
        tok_cls = RobertaTokenizer
    else:
        raise NotImplementedError

    tokenizer = tok_cls.from_pretrained(args.model_name_or_path)
    word_freq = torch.zeros(tokenizer.vocab_size)
    assert word_freq.size(0) == tokenizer.vocab_size

    def word_freq_preprocess_fn(wf):
        wf = wf + 1
        wf = wf.log()
        wf = wf / wf.max()
        return wf

    word_freq = word_freq_preprocess_fn(word_freq)

    if args.sample_strategy == 'Categorical':
        sample_cls = Categorical()
    elif args.sample_strategy == 'wwm':
        sample_cls = WholeWordMasking(tokenizer)
    else:
        raise ValueError

    diffusion_schedule = diffusion_condition.create_discrete_diffusion_schedule(args.schedule, num_steps=args.num_steps)
    diffusion_instance = diffusion_condition.MaskDiffusion(
        dim=tokenizer.vocab_size,
        schedule=diffusion_schedule,
        tokenizer=tokenizer,
        sample_cls=sample_cls,
        word_freq=word_freq,
        word_freq_lambda=args.word_freq_lambda,
        device=device
    )

    if args.load_step > 0:
        ckpt = torch.load(os.path.join(save_path, f'{args.load_step}.pt'))
    cfg = cfg_cls.from_pretrained(args.model_name_or_path)
    cfg.overall_timestep = diffusion_instance.num_steps

    if args.from_scratch:
        model = model_cls(cfg).to(device)
    elif args.load_step <= 0:
        model = model_cls.from_pretrained(args.model_name_or_path, config=cfg).to(device)
    else:
        model = model_cls(cfg).to(device)
        model.load_state_dict(ckpt['model'])

    optimizer = AdamW(model.parameters(), lr=args.lr)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=10000)

    train_data, dev_data = Loader(tokenizer=tokenizer).my_load(splits=['train', 'validation'])

    train_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=args.batch_size, 
        collate_fn=functools.partial(Loader.collate_fn, tokenizer=tokenizer),
        num_workers=4, 
        pin_memory=True,
        shuffle=True
    )
    
    dev_loader = torch.utils.data.DataLoader(
        dev_data, 
        batch_size=args.batch_size * 2, 
        collate_fn=functools.partial(Loader.collate_fn, tokenizer=tokenizer),
        num_workers=4, 
        pin_memory=True
    )

    if args.load_step > 0:
        optimizer.load_state_dict(ckpt['optimizer'])
        warmup_scheduler.load_state_dict(ckpt['warmup_scheduler'])
    model.train()

    def denoise_fn(input_ids, corrupted_input_ids, timestep, attention_mask, target_mask):
        # input_ids 'I am from China. I am Chinese. [PAD][PAD]'
        # corrupted_input_ids  '[MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK]'
        # target mask  '0 0 0 0 0 1 1 1 1 1 0 0'
        # new_input_ids  'I am from China. [MASK] [MASK] [MASK] [MASK] [PAD] [PAD]'

        # input_ids 'I am from China. I am Chinese. [PAD][PAD]'
        # corrupted_input_ids  '[MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK]'
        # target mask  '0 0 0 0 0 1 1 1 1 1 1 1'
        # new_input_ids  'I am from China. [MASK] [MASK] [MASK] [MASK] [MASK] [MASK]'
        new_input_ids = torch.where(target_mask.bool(), corrupted_input_ids, input_ids)
        
        #print("input_ids: %s"%str(input_ids))
        #print("corrupted_input_ids: %s"%str(corrupted_input_ids))
        #print("target mask: %s"%str(target_mask))
        #print("new input ids: %s"%str(new_input_ids))
        return model(
            input_ids=new_input_ids,
            attention_mask=attention_mask,
        )['logits']

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    best_dev_elbo = float('inf')

    train_loss = 0.0
    nan_count = 0
    i = -1
    print(args.epochs)
    #import pdb;pdb.set_trace()
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        for i, batch in enumerate(tqdm(train_loader)):
            # Memory monitoring
            print(f"Memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
            print(f"Memory cached: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
            torch.cuda.empty_cache()
            
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                t = diffusion_instance.sample_t().to(device)
                metrics = diffusion_condition.compute_kl_reverse_process(
                    batch['input_ids'],
                    t,
                    denoise_fn=functools.partial(
                        denoise_fn,
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        target_mask=batch['target_mask']
                    ),
                    diffusion=diffusion_instance,
                    target_mask=batch['target_mask'],
                    hybrid_lambda=args.hybrid_lambda,
                    predict_x0=args.predict_x0,
                    word_freq_logits=torch.zeros_like(batch['input_ids'])
                )
                
                # Normalize loss by accumulation steps
                loss = metrics['loss'] / args.accumulation_steps
            
            # Backward pass
            scaler.scale(loss).backward()  # REMOVED THE DUPLICATE backward() CALL
            
            # Gradient clipping
            torch.nn.utils.clip_grad_value_(model.parameters(), 5)
            
            # Step only when accumulation is complete
            if (i + 1) % args.accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                warmup_scheduler.step()
            
            # Track training loss
            if not torch.isnan(metrics['loss']):
                train_loss += metrics['loss'].item()
            else:
                nan_count += 1
            
            # Evaluation
            if i % args.eval_steps == args.eval_steps - 1:
                model.eval()
                dev_metrics = {
                    'elbo': 0.0,
                    'elbo_in_bits_per_dim': 0.0,
                }
                nan_count_in_dev = 0
                
                with torch.no_grad():
                    for dev_batch in dev_loader:
                        dev_batch = {k: v.to(device) for k, v in dev_batch.items()}
                        
                        batch_dev_metrics = diffusion_condition.discrete_diffusion_elbo(
                            dev_batch['input_ids'],
                            denoise_fn=functools.partial(
                                denoise_fn,
                                input_ids=dev_batch['input_ids'],
                                attention_mask=dev_batch['attention_mask'],
                                target_mask=dev_batch['target_mask']
                            ),
                            diffusion=diffusion_instance,
                            target_mask=dev_batch['target_mask'],
                            normalize_without_padding=True,
                            eval_step_size=args.eval_step_size,
                            word_freq_logits=torch.zeros_like(dev_batch['input_ids'])
                        )
                        
                        for name in dev_metrics.keys():
                            val = batch_dev_metrics[name].squeeze()
                            if not torch.isnan(val):
                                dev_metrics[name] += val * dev_batch['input_ids'].size(0)
                            else:
                                nan_count_in_dev += 1
                
                # Normalize dev metrics
                for name in dev_metrics.keys():
                    dev_metrics[name] /= max(1, (len(dev_data) - nan_count_in_dev * args.batch_size * 2))
                
                # Save best model
                if dev_metrics['elbo_in_bits_per_dim'] <= best_dev_elbo:
                    best_dev_elbo = dev_metrics['elbo_in_bits_per_dim']
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'warmup_scheduler': warmup_scheduler.state_dict(),
                    }, f'./{save_path}/best({i}).pt')
                
                model.train()

            # Uncomment if you want periodic saves
            if i % args.save_steps == args.save_steps - 1:
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'warmup_scheduler': warmup_scheduler.state_dict(),
                }, f'{save_path}/{i}.pt')