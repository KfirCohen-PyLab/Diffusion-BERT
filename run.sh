#!/bin/bash

python main.py \
    --train_data_dir "./conditional_data" \
    --vocab_size 30522 \
    --block_size 128 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --num_train_epochs 100 \
    --gradient_accumulation_steps 2 \
    --model_type "bert-base-uncased" \
    --diffusion_steps 2000 \
    --noise_schedule "cosine" \
    --spindle_schedule True \
    --word_freq_file "word_freq.json" \
    --output_dir "./diffusion_models" \
    --num_workers 4 \
    --fp16 True