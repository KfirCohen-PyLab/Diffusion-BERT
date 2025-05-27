import os
import sys
import random
import numpy as np
import argparse
import torch
import fitlog
from dataloader import DiffusionLoader, load_word_frequencies, collate_fn
from transformers import BertTokenizer, BertConfig, RobertaTokenizer, RobertaConfig
from models.modeling_roberta import RobertaForMaskedLM
import diffusion
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
import fastNLP
from tqdm import tqdm
from sample import Categorical, WholeWordMasking
import math
import json
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str, required=False)
    parser.add_argument("--task_name", default="lm1b", type=str, required=False)
    parser.add_argument("--lr", default=5e-4, type=float, required=False)
    parser.add_argument("--epochs", default=3, type=int, required=False)
    parser.add_argument("--batch_size", default=64, type=int, required=False)
    parser.add_argument("--word_freq_lambda", default=0.3, type=float, required=False)
    parser.add_argument("--num_steps", default=2048, type=int, required=False)
    parser.add_argument("--eval_step_size", default=4, type=int, required=False)
    parser.add_argument("--dev_size", default=5e-4, type=float, required=False)
    parser.add_argument("--hybrid_lambda", default=1e-2, type=float, required=False)
    parser.add_argument("--eval_steps", default=15000, type=int, required=False)
    parser.add_argument("--seed", default=42, type=int, required=False)
    parser.add_argument("--logging_steps", default=1000, type=int, required=False)
    parser.add_argument("--predict_x0", default=True, type=bool, required=False)
    parser.add_argument("--load_step", default=-1, type=int, required=False)
    parser.add_argument("--sample_strategy", default="Categorical", type=str, required=False)
    parser.add_argument("--schedule", default="mutual", type=str, required=False)
    parser.add_argument("--from_scratch", default=False, type=bool, required=False)
    parser.add_argument("--timestep", default="none", type=str, required=False)
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, default=30522)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--model_type", type=str, default="bert-base-uncased")
    parser.add_argument("--diffusion_steps", type=int, default=2000)
    parser.add_argument("--noise_schedule", type=str, default="cosine")
    parser.add_argument("--spindle_schedule", type=bool, default=True)
    parser.add_argument("--word_freq_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--fp16", type=bool, default=True)
    return parser.parse_args()


class TextDataset(Dataset):
    def __init__(self, data_dir, tokenizer, block_size):
        self.examples = []
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # Load all jsonl files
        for filename in os.listdir(data_dir):
            if filename.endswith('.jsonl'):
                with open(os.path.join(data_dir, filename), 'r') as f:
                    for line in f:
                        item = json.loads(line)
                        tokens = tokenizer.encode(item['text'], 
                                               max_length=block_size,
                                               truncation=True,
                                               padding='max_length')
                        self.examples.append(tokens)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)

class DiffusionBERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(config)
        self.projection = nn.Linear(config.hidden_size, config.vocab_size)
        
    def forward(self, x, timesteps):
        outputs = self.bert(x)
        hidden_states = outputs.last_hidden_state
        logits = self.projection(hidden_states)
        return logits

def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(output_dir, "train.log"))
        ]
    )

def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device

def main():
    args = parse_args()
    setup_logging(args.output_dir)
    device = setup_device()

    set_seed(args)
    if args.timestep in ["none", "token"]:
        from models.modeling_bert import BertForMaskedLM
    elif args.timestep == "layerwise":
        from models.modeling_bert_new_timestep import BertForMaskedLM
    else:
        raise NotImplementedError

    log_dir = "./logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fitlog.set_log_dir(log_dir)
    fitlog.commit(__file__)
    fitlog.add_hyper(args)
    fitlog.add_hyper_in_file(__file__)

    save_path = f"./model_name_{args.model_name_or_path}_lr_{args.lr}_seed_{args.seed}_numsteps_{args.num_steps}_sample_{args.sample_strategy}_schedule_{args.schedule}_hybridlambda_{args.hybrid_lambda}_wordfreqlambda_{args.word_freq_lambda}_fromscratch_{args.from_scratch}_timestep_{args.timestep}_ckpts"
    if args.model_name_or_path in ["bert-base-uncased", "bert-large-uncased"]:
        model_cls = BertForMaskedLM
        cfg_cls = BertConfig
        tok_cls = BertTokenizer
    elif args.model_name_or_path in ["roberta-base"]:
        model_cls = RobertaForMaskedLM
        cfg_cls = RobertaConfig
        tok_cls = RobertaTokenizer
    else:
        raise NotImplementedError

    tokenizer = tok_cls.from_pretrained(args.model_name_or_path)
    word_freq = torch.load(f"./word_freq/{args.model_name_or_path}_{args.task_name}.pt")
    assert word_freq.size(0) == tokenizer.vocab_size

    def word_freq_preprocess_fn(wf):
        wf = wf + 1
        wf = wf.log()
        wf = wf / wf.max()
        return wf

    def process_fn_in_collate(wf):
        return wf - wf.mean()

    word_freq = word_freq_preprocess_fn(word_freq)
    word_freq[tokenizer.pad_token_id] = 0.  # stable training

    if args.sample_strategy == "Categorical":
        sample_cls = Categorical()
    elif args.sample_strategy == "wwm":
        sample_cls = WholeWordMasking(tokenizer)
    else:
        raise ValueError

    diffusion_schedule = diffusion.create_discrete_diffusion_schedule(args.schedule, num_steps=args.num_steps)
    diffusion_instance = diffusion.MaskDiffusion(
        dim=tokenizer.vocab_size,
        schedule=diffusion_schedule,
        tokenizer=tokenizer,
        sample_cls=sample_cls,
        word_freq_lambda=args.word_freq_lambda,
        device=device
    )

    if args.load_step > 0:
        ckpt = torch.load(os.path.join(save_path, f"{args.load_step}.th"))
    cfg = cfg_cls.from_pretrained(args.model_name_or_path)
    cfg.overall_timestep = diffusion_instance.num_steps

    if args.from_scratch:
        model = model_cls(cfg).to(device)
    elif args.load_step <= 0:
        model = model_cls.from_pretrained(args.model_name_or_path, config=cfg).to(device)
    else:
        model = model_cls(cfg).to(device)
        model.load_state_dict(ckpt["model"])

    optimizer = AdamW(model.parameters(), lr=args.lr)
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda n: n / 10000. + 1e-3 if n < 10000 else 100. / math.sqrt(n))

    train_data, test_data = DiffusionLoader(tokenizer=tokenizer).my_load(task_name="lm1b", splits=["train", "test"])
    train_data, dev_data = train_data.train_test_split(test_size=args.dev_size).values()

    logger.info("# of train data: {}".format(len(train_data)))
    logger.info("Example:")
    logger.info(train_data[0])
    logger.info("\n# of dev data: {}".format(len(dev_data)))
    logger.info("Example:")
    logger.info(dev_data[0])
    logger.info("\n# of test data: {}".format(len(test_data)))
    logger.info("Example:")
    logger.info(test_data[0])

    def collate_fn(batch_input):
        input_ids = [torch.tensor(d["input_ids"]) for d in batch_input]
        attention_mask = [torch.tensor(d["attention_mask"]) for d in batch_input]
        word_freq_logits = [process_fn_in_collate(word_freq.gather(0, torch.tensor(d["input_ids"]))) for d in batch_input]
        input_ids = pad_sequence(input_ids, batch_first=True)
        attention_mask = pad_sequence(attention_mask, batch_first=True)
        word_freq_logits = pad_sequence(word_freq_logits, batch_first=True)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_freq_logits": word_freq_logits
        }

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=True, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=args.batch_size * 2, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    model.train()

    cls = torch.full((1, 1), fill_value=tokenizer.cls_token_id, device=device)
    sep = torch.full((1, 1), fill_value=tokenizer.sep_token_id, device=device)

    att_ones = torch.ones((1, 1), device=device)
    att_zeros = torch.zeros((1, 1), device=device)

    if args.timestep == "none":
        def denoise_fn(targets, timestep, attention_mask):
            assert len(targets.size()) == 2  # bsz * seqlen
            bsz = targets.size(0)
            targets = torch.cat((cls.repeat(bsz, 1), targets, sep.repeat(bsz, 1)), dim=1)
            attention_mask = torch.cat((att_ones.repeat(bsz, 1), attention_mask, att_zeros.repeat(bsz, 1)), dim=1)
            return model(input_ids=targets, timestep=timestep - 1, attention_mask=attention_mask)["logits"][:, 1:-1, :]
    elif args.timestep == "token":
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
            return model(input_ids=targets, timestep=timestep - 1, attention_mask=attention_mask)["logits"][:, 2:-1, :]
    elif args.timestep == "layerwise":
        def denoise_fn(targets, timestep, attention_mask):
            assert len(targets.size()) == 2  # bsz * seqlen
            bsz = targets.size(0)
            targets = torch.cat((
                cls.repeat(bsz, 1),
                targets,
                sep.repeat(bsz, 1)
            ), dim=1)
            attention_mask = torch.cat((att_ones.repeat(bsz, 1), attention_mask, att_zeros.repeat(bsz, 1)), dim=1)
            return model(input_ids=targets, timestep=timestep - 1, attention_mask=attention_mask)["logits"][:, 1:-1, :]
    else:
        raise NotImplementedError

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    best_dev_elbo = float("inf")

    train_loss = .0
    nan_count = 0
    for epoch in range(args.epochs):
        for i, batch in enumerate(tqdm(train_loader), args.load_step + 1):
            metrics = diffusion.compute_kl_reverse_process(
                batch["input_ids"].to(device),
                diffusion_instance.sample_t(),
                denoise_fn=denoise_fn,
                diffusion=diffusion_instance,
                target_mask=batch["attention_mask"].to(device),
                hybrid_lambda=args.hybrid_lambda,
                predict_x0=args.predict_x0,
                word_freq_logits=batch["word_freq_logits"].to(device)
            )

            loss = metrics["loss"] / args.batch_size
            if torch.isnan(loss):
                nan_count += 1
                logger.warning(f"NaN encountered {nan_count} times")
                continue
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 5)
            optimizer.step()
            model.zero_grad()
            optimizer.zero_grad()
            warmup_scheduler.step()

            if i % args.logging_steps == args.logging_steps - 1:
                logger.info(f"Loss at step {i} is {train_loss / args.logging_steps}")
                fitlog.add_loss(train_loss / args.logging_steps, name="train_loss", step=i)
                train_loss = .0

            if i % args.eval_steps == args.eval_steps - 1:
                nan_count_in_dev = 0
                model.eval()
                dev_metrics = {
                    "elbo": .0,
                    "elbo_in_bits_per_dim": .0,
                }

                with torch.no_grad():
                    for dev_batch in dev_loader:
                        batch_dev_metrics = diffusion.discrete_diffusion_elbo(
                            dev_batch["input_ids"].to(device),
                            denoise_fn=denoise_fn,
                            diffusion=diffusion_instance,
                            target_mask=dev_batch["attention_mask"].to(device),
                            normalize_without_padding=True,
                            eval_step_size=args.eval_step_size,
                            word_freq_logits=dev_batch["word_freq_logits"].to(device),
                            device=device
                        )

                        for name in dev_metrics.keys():
                            if not torch.isnan(batch_dev_metrics[name]):
                                dev_metrics[name] += batch_dev_metrics[name]
                            else:
                                nan_count_in_dev += 1
                                logger.warning(f"NaN encountered {nan_count_in_dev} times in dev")

                    for name in dev_metrics.keys():
                        dev_metrics[name] /= len(dev_data)
                        fitlog.add_metric(dev_metrics[name], name=name, step=i)

                    if dev_metrics["elbo_in_bits_per_dim"] <= best_dev_elbo:
                        best_dev_elbo = dev_metrics["elbo_in_bits_per_dim"]
                        fitlog.add_best_metric(dev_metrics["elbo_in_bits_per_dim"], name="dev_elbo_in_bits_per_dim")
                        torch.save({
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "warmup_scheduler": warmup_scheduler.state_dict(),
                        }, f"./{save_path}/best({i}).th")
                model.train()

    # Prepare dataset
    dataset = TextDataset(args.train_data_dir, tokenizer, args.block_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                          shuffle=True, num_workers=args.num_workers)
    
    # Training setup
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    if args.fp16:
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
    
    # Training loop
    model.train()
    for epoch in range(args.num_train_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            batch = batch.to(device)
            
            # Generate random timesteps
            timesteps = torch.randint(0, args.diffusion_steps, (batch.size(0),), device=device)
            
            with autocast(enabled=args.fp16):
                logits = model(batch, timesteps)
                loss = nn.CrossEntropyLoss()(logits.view(-1, cfg.vocab_size), batch.view(-1))
                loss = loss / args.gradient_accumulation_steps
            
            if args.fp16:
                scaler.scale(loss).backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": total_loss / (step + 1)})
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, os.path.join(args.output_dir, f"checkpoint-{epoch+1}.pt"))

if __name__ == "__main__":
    main() 