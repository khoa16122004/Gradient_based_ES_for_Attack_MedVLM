#!/bin/bash

python main_attack.py \
    --dataset_name rsna \
    --model_name ViT-B-32 \
    --attacker_name ES_1_Lambda \
    --epsilon 0.1 \
    --norm linf \
    --max_evaluation 10000 \
    --lamda 50 \
    --out_dir attack_results/rsna_ES_1_Lambda \
    --start_idx 0 \
    --end_idx 1000 \
