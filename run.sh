#!/bin/bash

python main_attack.py \
    --dataset_name rsna \
    --model_name ViT-B-32 \
    --attacker_name ES_1_Lambda \
    --epsilon 0.05 \
    --norm linf \
    --max_evaluation 10000 \
    --lamda 50 \
    --out_dir test_visualization \
    --start_idx 0 \
    --end_idx 1 \
    --index_path "evaluate_result\intersection.txt"

# python main_attack.py \
#     --dataset_name rsna \
#     --model_name ViT-B-32 \
#     --attacker_name PGD \
#     --epsilon 0.03 \
#     --norm linf \
#     --PGD_steps 100 \
#     --alpha 0.01 \
#     --out_dir attack_results/RSNA_PGD \
#     --start_idx 0 \
#     --end_idx 1000 \
#     --index_path "evaluate_result\union_data.txt"

# python main_attack.py \
#     --dataset_name rsna \
#     --model_name ViT-B-16 \
#     --attacker_name PGD \
#     --epsilon 0.03 \
#     --norm linf \
#     --PGD_steps 100 \
#     --alpha 0.01 \
#     --out_dir attack_results/RSNA_PGD \
#     --start_idx 0 \
#     --end_idx 1000 \
#     --index_path "evaluate_result\union_data.txt"
    
# python main_attack.py \
#     --dataset_name rsna \
#     --model_name ViT-L-14 \
#     --attacker_name PGD \
#     --epsilon 0.03 \
#     --norm linf \
#     --PGD_steps 100 \
#     --alpha 0.01 \
#     --out_dir attack_results/RSNA_PGD \
#     --start_idx 0 \
#     --end_idx 1000 \
# #     --index_path "evaluate_result\union_data.txt"

# python main_attack.py \
#     --dataset_name rsna \
#     --model_name medclip \
#     --attacker_name PGD \
#     --epsilon 0.03 \
#     --norm linf \
#     --PGD_steps 100 \
#     --alpha 0.01 \
#     --out_dir attack_results/RSNA_PGD \
#     --start_idx 0 \
#     --end_idx 1000 \
#     --index_path "evaluate_result\union_data.txt"

# python transfer_attack.py \
#     --dataset_name rsna \
#     --model_name ViT-B-16 \
#     --transfer_dir "attack_results\rsna_ES_1_Lambda\ViT-B-32\rsna\attack_name=ES_1_Lambda_epsilon=0.03_lamda=50_norm=linf_seed=42" \
#     --start_idx 0 \
#     --end_idx 1000 \
#     --index_path "evaluate_result\union_data.txt"



