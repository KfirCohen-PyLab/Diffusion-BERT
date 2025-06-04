import os
import sys
import random
import numpy as np
import argparse
os.environ['TORCH_USE_LIBUV'] = '0'
import torch
import fitlog
from dataloader import DiffusionLoader, collate_fn
from transformers import BertTokenizer, BertConfig, RobertaTokenizer, RobertaConfig
from models.modeling_roberta import RobertaForMaskedLM
import diffusion_word_freq
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Subset
import fastNLP
from tqdm import tqdm
from sample import Categorical, WholeWordMasking
import math
import datetime
import traceback
import platform
from functools import partial

# Global flag for using fitlog
USE_FITLOG = False  # Set to False to disable fitlog

def init_fitlog():
    """Initialize fitlog if enabled"""
    global USE_FITLOG
    if USE_FITLOG:
        try:
            log_dir = './logs'
            os.makedirs(log_dir, exist_ok=True)
            fitlog.set_log_dir(log_dir)
            # Initialize a new fitlog project if needed
            if not os.path.exists(os.path.join(log_dir, '.fitconfig')):
                with open(os.path.join(log_dir, '.fitconfig'), 'w') as f:
                    f.write('name: Diffusion-BERT\n')
            fitlog.commit(__file__)
            return True
        except Exception as e:
            print(f"Warning: Failed to initialize fitlog: {str(e)}")
            print("Continuing without fitlog logging...")
            USE_FITLOG = False
    return False

def log_metric(value, name, step=None):
    """Wrapper for fitlog logging"""
    global USE_FITLOG
    if USE_FITLOG:
        try:
            if step is not None:
                fitlog.add_metric(value, name=name, step=step)
            else:
                fitlog.add_metric(value, name=name)
        except Exception as e:
            print(f"Warning: Failed to log metric: {str(e)}")

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default='bert-base-uncased', type=str, required=False)
    parser.add_argument("--task_name", default='lm1b', type=str, required=False)
    parser.add_argument("--lr", default=5e-4, type=float, required=False)
    parser.add_argument("--epochs", default=1, type=int, required=False)
    parser.add_argument("--batch_size", default=32, type=int, required=False)
    parser.add_argument("--word_freq_lambda", default=0.3, type=float, required=False)
    parser.add_argument("--num_steps", default=512, type=int, required=False)
    parser.add_argument("--eval_step_size", default=2, type=int, required=False)
    parser.add_argument("--dev_size", default=5e-4, type=float, required=False)
    parser.add_argument("--hybrid_lambda", default=1e-2, type=float, required=False)
    parser.add_argument("--eval_steps", default=50, type=int, required=False)
    parser.add_argument("--seed", default=42, type=int, required=False)
    parser.add_argument("--logging_steps", default=10, type=int, required=False)
    parser.add_argument('--predict_x0', default=True, type=bool, required=False)
    parser.add_argument("--load_step", default=-1, type=int, required=False)
    parser.add_argument("--sample_strategy", default='Categorical', type=str, required=False)
    parser.add_argument("--schedule", default='mutual', type=str, required=False)
    parser.add_argument("--from_scratch", default=False, type=bool, required=False)
    parser.add_argument("--timestep", default='none', type=str, choices=['none', 'token', 'layerwise'], help='Timestep type: none, token, or layerwise', required=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    try:
        print("Starting DDP_main.py in single GPU mode")
        args = parse_args()
        
        # Use CUDA if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Using device: {device}")

        set_seed(args)

        if args.timestep in ['none', 'token']:
            from models.modeling_bert import BertForMaskedLM
        elif args.timestep == 'layerwise':
            from models.modeling_bert_new_timestep import BertForMaskedLM
        else:
            raise NotImplementedError

        # Initialize fitlog
        init_fitlog()

        save_path = f'./model_name_{args.model_name_or_path}_lr_{args.lr}_seed_{args.seed}_numsteps_{args.num_steps}_sample_{args.sample_strategy}_schedule_{args.schedule}_hybridlambda_{args.hybrid_lambda}_wordfreqlambda_{args.word_freq_lambda}_fromscratch_{args.from_scratch}_timestep_{args.timestep}_ckpts'

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
        word_freq = torch.load(f'./word_freq/{args.model_name_or_path}_{args.task_name}.pt')
        assert word_freq.size(0) == tokenizer.vocab_size

        def word_freq_preprocess_fn(wf):
            wf = wf + 1
            wf = wf.log()
            wf = wf / wf.max()
            return wf

        def process_fn_in_collate(wf):
            return wf - wf.mean()

        word_freq = word_freq_preprocess_fn(word_freq)
        word_freq[tokenizer.pad_token_id] = 0.  # Stable training

        if args.sample_strategy == 'Categorical':
            sample_cls = Categorical()
        elif args.sample_strategy == 'wwm':
            sample_cls = WholeWordMasking(tokenizer)
        else:
            raise ValueError

        diffusion_schedule = diffusion_word_freq.create_discrete_diffusion_schedule(args.schedule, num_steps=args.num_steps)
        diffusion_instance = diffusion_word_freq.MaskDiffusion(
            dim=tokenizer.vocab_size,
            schedule=diffusion_schedule,
            tokenizer=tokenizer,
            sample_cls=sample_cls,
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
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda n: n / 10000. + 1e-3 if n < 10000 else 100. / math.sqrt(n))

        train_data, test_data = DiffusionLoader(tokenizer=tokenizer).my_load(task_name='lm1b', splits=['train', 'test'])
        
        # Take only a small subset of training data for quick training
        total_samples = min(len(train_data), 1000)  # Use at most 1000 samples total
        indices = list(range(len(train_data)))
        random.shuffle(indices)
        
        # Split indices for train and dev
        train_size = int(total_samples * 0.95)  # 95% for training
        train_indices = indices[:train_size]
        dev_indices = indices[train_size:total_samples]
        
        # Create train and dev datasets from original data
        train_data = Subset(train_data, train_indices)
        dev_data = Subset(train_data, dev_indices)
        
        print(f'Original dataset size: {len(indices)}')
        print(f'# of train data: {len(train_data)}')
        if len(train_data) > 0:
            print('Train example:')
            print(train_data[0])
        
        print(f'\n# of dev data: {len(dev_data)}')
        if len(dev_data) > 0:
            print('Dev example:')
            print(dev_data[0])
        
        print(f'\n# of test data: {len(test_data)}')
        if len(test_data) > 0:
            print('Test example:')
            print(test_data[0])

        # Determine number of workers based on OS
        num_workers = 0 if platform.system() == 'Windows' else 4
        
        # Create partial collate function with word_freq
        collate_with_freq = partial(collate_fn, word_freq=word_freq)

        # Regular DataLoader without distributed sampler
        train_loader = torch.utils.data.DataLoader(
            train_data, 
            batch_size=args.batch_size, 
            collate_fn=collate_with_freq, 
            num_workers=num_workers, 
            pin_memory=True,
            shuffle=True
        )
        
        dev_loader = torch.utils.data.DataLoader(
            dev_data, 
            batch_size=args.batch_size * 2, 
            collate_fn=collate_with_freq, 
            num_workers=num_workers, 
            pin_memory=True
        )

        model.train()

        cls = torch.full((1, 1), fill_value=tokenizer.cls_token_id, device=device)
        sep = torch.full((1, 1), fill_value=tokenizer.sep_token_id, device=device)
        att_ones = torch.ones((1, 1), device=device)
        att_zeros = torch.zeros((1, 1), device=device)

        if args.timestep == 'none':
            def denoise_fn(targets, timestep, attention_mask):
                assert len(targets.size()) == 2  # bsz * seqlen
                bsz = targets.size(0)
                targets = torch.cat((cls.repeat(bsz, 1), targets, sep.repeat(bsz, 1)), dim=1)
                attention_mask = torch.cat((att_ones.repeat(bsz, 1), attention_mask, att_zeros.repeat(bsz, 1)), dim=1)
                return model(input_ids=targets, timestep=timestep - 1, attention_mask=attention_mask)['logits'][:, 1:-1, :]
        elif args.timestep == 'token':
            def denoise_fn(targets, timestep, attention_mask):
                assert len(targets.size()) == 2  # bsz * seqlen
                bsz = targets.size(0)
                targets = torch.cat((
                    cls.repeat(bsz, 1),
                    torch.full((bsz, 1), fill_value=timestep.item() + 110, device=device),
                    targets,
                    sep.repeat(bsz, 1)
                ), dim=1)
                attention_mask = torch.cat((att_ones.repeat(bsz, 2), attention_mask, att_zeros.repeat(bsz, 1)), dim=1)
                return model(input_ids=targets, timestep=timestep - 1, attention_mask=attention_mask)['logits'][:, 2:-1, :]
        elif args.timestep == 'layerwise':
            def denoise_fn(targets, timestep, attention_mask):
                assert len(targets.size()) == 2  # bsz * seqlen
                bsz = targets.size(0)
                targets = torch.cat((
                    cls.repeat(bsz, 1),
                    targets,
                    sep.repeat(bsz, 1)
                ), dim=1)
                attention_mask = torch.cat((att_ones.repeat(bsz, 1), attention_mask, att_zeros.repeat(bsz, 1)), dim=1)
                return model(input_ids=targets, timestep=timestep - 1, attention_mask=attention_mask)['logits'][:, 1:-1, :]
        else:
            raise NotImplementedError

        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        best_dev_elbo = float('inf')
        
        # Save initial checkpoint
        print(f"\nSaving initial checkpoint...")
        initial_save_path = f'./{save_path}/initial.pt'
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'warmup_scheduler': warmup_scheduler.state_dict(),
        }, initial_save_path)
        print(f"Saved initial checkpoint to {initial_save_path}")

        train_loss = .0
        for epoch in range(args.epochs):
            print(f"\nStarting epoch {epoch + 1}/{args.epochs}")
            for i, batch in enumerate(tqdm(train_loader), args.load_step + 1):
                metrics = diffusion_word_freq.compute_kl_reverse_process(
                    batch['input_ids'].to(device),
                    diffusion_instance.sample_t(),
                    denoise_fn=denoise_fn,
                    diffusion=diffusion_instance,
                    target_mask=batch['attention_mask'].to(device),
                    hybrid_lambda=args.hybrid_lambda,
                    predict_x0=args.predict_x0,
                    word_freq_logits=batch['word_freq_logits'].to(device)
                )

                loss = metrics['loss'] / args.batch_size
                if torch.isnan(loss):
                    print(f'Warning: NaN loss encountered at step {i}')
                    continue
                    
                train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), 5)
                optimizer.step()
                model.zero_grad()
                optimizer.zero_grad()
                warmup_scheduler.step()

                if i % args.logging_steps == args.logging_steps - 1:
                    print(f'Loss at step {i} is {train_loss / args.logging_steps}')
                    log_metric(train_loss / args.logging_steps, name='train_loss', step=i)
                    train_loss = .0

                # Save checkpoint every eval_steps
                if i % args.eval_steps == args.eval_steps - 1:
                    print(f"\nEvaluating and saving checkpoint at step {i}...")
                    model.eval()
                    dev_metrics = {
                        'elbo': .0,
                        'elbo_in_bits_per_dim': .0,
                    }

                    with torch.no_grad():
                        for dev_batch in dev_loader:
                            batch_dev_metrics = diffusion_word_freq.discrete_diffusion_elbo(
                                dev_batch['input_ids'].to(device),
                                denoise_fn=denoise_fn,
                                diffusion=diffusion_instance,
                                target_mask=dev_batch['attention_mask'].to(device),
                                normalize_without_padding=True,
                                eval_step_size=args.eval_step_size,
                                word_freq_logits=dev_batch['word_freq_logits'].to(device),
                                device=device
                            )

                            for name in dev_metrics.keys():
                                if not torch.isnan(batch_dev_metrics[name]):
                                    dev_metrics[name] += batch_dev_metrics[name].item()

                        for name in dev_metrics.keys():
                            dev_metrics[name] /= len(dev_data)
                            log_metric(dev_metrics[name], name=name, step=i)

                        # Always save a checkpoint at evaluation
                        save_file = f'./{save_path}/step_{i}.pt'
                        print(f"Saving checkpoint to {save_file}")
                        torch.save({
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'warmup_scheduler': warmup_scheduler.state_dict(),
                        }, save_file)
                        print("Checkpoint saved successfully!")

                        # Save best model if it's the best so far
                        if dev_metrics['elbo_in_bits_per_dim'] <= best_dev_elbo:
                            best_dev_elbo = dev_metrics['elbo_in_bits_per_dim']
                            if USE_FITLOG:
                                fitlog.add_best_metric(dev_metrics['elbo_in_bits_per_dim'], name='dev_elbo_in_bits_per_dim')
                            best_save_file = f'./{save_path}/best({i}).pt'
                            print(f"New best model! Saving to {best_save_file}")
                            torch.save({
                                'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'warmup_scheduler': warmup_scheduler.state_dict(),
                            }, best_save_file)
                            print("Best model checkpoint saved successfully!")
                    model.train()

            # Save checkpoint at end of each epoch
            print(f"\nSaving checkpoint at end of epoch {epoch + 1}...")
            epoch_save_file = f'./{save_path}/epoch_{epoch + 1}.pt'
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'warmup_scheduler': warmup_scheduler.state_dict(),
            }, epoch_save_file)
            print(f"Saved epoch checkpoint to {epoch_save_file}")

        # Save final checkpoint
        print("\nSaving final checkpoint...")
        final_save_file = f'./{save_path}/final.pt'
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'warmup_scheduler': warmup_scheduler.state_dict(),
        }, final_save_file)
        print(f"Saved final checkpoint to {final_save_file}")

    except Exception as e:
        print(f"Error in main: {str(e)}")
        traceback.print_exc()
        raise