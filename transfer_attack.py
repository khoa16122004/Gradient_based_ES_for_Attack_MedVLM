from modules.dataset.factory import DatasetFactory
from modules.utils.constants import MODEL_TRANSFORMS, DEFAULT_TEMPLATES, RSNA_CLASS_PROMPTS, RSNA_CLASS_PROMPTS, SIZE_TRANSFORM, DATA_ROOT
from modules.models.factory import ModelFactory
from tqdm import tqdm
import numpy as np
import torch
import json
from modules.attack.attack import ES_1_Lambda, PGDAttack
from modules.attack.evaluator import EvaluatePerturbation
from modules.attack.util import seed_everything 
from modules.utils.helpers import _extract_label, load_open_clip_model
import os
from torchvision import transforms
import yaml
import pandas as pd
from PIL import Image
from collections import OrderedDict
_toTensor = transforms.ToTensor()


def main(args):
    
    # ================================ Take dataset ================================ 
    dataset = DatasetFactory.create_dataset(
        dataset_name=args.dataset_name,
        model_type='medclip',
        data_root=DATA_ROOT,
        transform=None
    )

    
   # ========= class_prompt_based ========= #
    class_prompts = RSNA_CLASS_PROMPTS
    num_classes = len(class_prompts)

    # ========= class_prompt_based ========= #
    class_prompts = RSNA_CLASS_PROMPTS
    num_classes = len(class_prompts)
    if args.model_name in ['medclip', 'biomedclip']:
        model = ModelFactory.create_model(
            model_type=args.model_name,
            variant='base',
            pretrained=True
        )
    elif args.model_name in ['ViT-B-32', 'ViT-B-16', 'ViT-L-14']:
        model = ModelFactory.create_model(
            model_type='ViT',
            variant=args.model_name,
        )
    else:
        raise NotImplementedError(f"Model {args.model_name} not implemented.")


    # ================================ Load selected indices ================================
    with open(args.index_path, "r") as f:
        # indxs = [int(line.strip()) for line in f.readlines()]
        indxs = [int(sample_id) for sample_id in os.listdir(args.transfer_dir)]

    if not args.end_idx:
        indxs = indxs[args.start_idx:]
    else:
        indxs = indxs[args.start_idx:args.end_idx]

    print("Len attack: ", len(indxs))

    
        
    class_features = []
    for class_name, item in class_prompts.items():
        text_feats = model.encode_text(item)
        mean_feats = text_feats.mean(dim=0)
        class_features.append(mean_feats) 
    class_features = torch.stack(class_features) #  NUM_ClASS x D

    
    asr = 0
    # --------------------------- Main LOOP ------------------ 
    for index in tqdm(indxs):
        img_transfer_path = os.path.join(args.transfer_dir, str(index), "adv_img.png")
        adv_img = Image.open(img_transfer_path)
        img, label_dict = dataset[index]
        label_id = _extract_label(label_dict)
        
        # clean_preds
        img_tensor = _toTensor(img.convert('RGB')).unsqueeze(0).cuda()
        img_feats_clean = model.encode_pretransform_image(img_tensor)
        sims = img_feats_clean @ class_features.T                     # (B, NUM_CLASS)
        clean_preds = sims.argmax(dim=-1).item()                    # (B,)

        img_attack_tensor = _toTensor(adv_img).unsqueeze(0).cuda()
        img_feats = model.encode_posttransform_image(img_attack_tensor)      
        sims = img_feats @ class_features.T                     # (B, NUM_CLASS)
        adv_preds = sims.argmax(dim=-1).item()                    # (B,)
        # if clean_preds != label_id:
        #     print(index, '\n' ,  clean_preds, '\n', label_id )
        #     raise
        if adv_preds != clean_preds or adv_preds != label_id:
            asr += 1
            
        
    print("Asr: ", asr * 100 / len(indxs))  
        
        
        
        
    

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Adversarial Attack Runner")

    # Dataset & model
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Name of dataset (e.g., rsna, chestxray, etc.)")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model architecture (e.g., clip, biomedclip, etc.)")
    

    # transfer_path
    parser.add_argument("--transfer_dir", type=str)

    # Files
    parser.add_argument("--index_path", type=str, required=True,
                        help="Path to txt file containing selected indices (one per line)")
    

 
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)


    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    
    
    # using decoder
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    main(args)
    
    
# CUDA_VISIBLE_DEVICES=4 python main_atttack.py --dataset_name rsna --model_name medclip --index_path evaluate_result/selected_indices_covid_medclip.txt --prompt_path evaluate_result/model_name\=medclip_dataset\=rsna_prompt.json --attacker_name ES_1_1 --epsilon 0.03 --norm linf --mode pre_transform --seed 22520691
