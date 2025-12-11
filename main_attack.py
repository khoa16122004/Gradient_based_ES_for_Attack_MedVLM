from modules.dataset.factory import DatasetFactory
from modules.utils.constants import MODEL_TRANSFORMS, DEFAULT_TEMPLATES, RSNA_CLASS_PROMPTS, RSNA_CLASS_PROMPTS, SIZE_TRANSFORM, DATA_ROOT
from modules.models.factory import ModelFactory
from tqdm import tqdm
import numpy as np
import torch
import json
from modules.attack.attack import ES_1_Lambda
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
    # ========= Dataset ========= #
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

    # ========= Model ========= #
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
    

    # ========= Read selected indices ========= #
    if args.index_path:
        with open(args.index_path, "r") as f:
            indxs = [int(line.strip()) for line in f.readlines()]

        if not args.end_idx:
            indxs = indxs[args.start_idx:]
        else:
            indxs = indxs[args.start_idx:args.end_idx]
    else:
        indxs = list(range(len(dataset)))
    print("Len attack: ", len(indxs))
    
    
    # ========================== Evaluator ==========================
    evaluator = EvaluatePerturbation(
        model=model,
        class_prompts=class_prompts,
        eps=args.epsilon,
        norm=args.norm
    )
    
    # path dir save
    save_dir = os.path.join(args.out_dir, args.model_name, args.dataset_name, "attack_name={args.attacker_name}_epsilon={args.epsilon}_norm={args.norm}_mode={args.mode}_seed={args.seed}")
    os.makedirs(save_dir, exist_ok=True)
    
    
    # ========================== Attacker ==========================
    if args.attacker_name == "ES_1_Lambda": # number of evalation = ierations * lambda
        attacker = ES_1_Lambda(
            evaluator=evaluator,
            eps=args.epsilon,
            norm=args.norm,
            max_evaluation=args.max_evaluation,
            lam=args.lamda
        )
        
    

    # --------------------------- Main LOOP ------------------ 
    for index in tqdm(indxs):
        img, label_dict = dataset[index]
        label_id = _extract_label(label_dict)


        img_attack = img.convert("RGB")
        img_attack_tensor = _toTensor(img_attack).unsqueeze(0).cuda()
        img_feats = model.encode_pretransform_image(img_attack_tensor)
      
        # re-evaluation
        sims = img_feats @ evaluator.class_text_feats.T                     # (B, NUM_CLASS)
        clean_preds = sims.argmax(dim=-1).item()                    # (B,)


        # main attack
        attacker.evaluator.set_data(
            image=img_attack,
            clean_pred_id=clean_preds
        )
        
        result = attacker.run()
        delta = result['best_delta']
        adv_imgs, pil_adv_imgs = evaluator.take_adv_img(delta)            
        img_feats = model.encode_pretransform_image(adv_imgs)  # (B, D)
        
        sims = img_feats @ evaluator.class_text_feats.T                     # (B, NUM_CLASS)
        adv_preds = sims.argmax(dim=-1).item()                    # (B,)
        # print("Adv preds: ", preds)
        
        # save_dir
        index_dir = os.path.join(save_dir, str(index))
        os.makedirs(index_dir, exist_ok=True)
        pil_adv_imgs[0].save(os.path.join(index_dir, f'adv_img.png'))
        img_attack.save(os.path.join(index_dir, "clean_img.png"))
        
        info = {
            'clean_pred': clean_preds,
            'adv_pred': adv_preds,
            'gt': label_id,
            'success_iterations': result['num_evaluation']
        }
        with open(os.path.join(index_dir, "info.json"), "w") as f:
            json.dump(info, f, indent=4)
            
            
        
        
        
        
        
        
    

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Adversarial Attack Runner")

    # Dataset & model
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Name of dataset (e.g., rsna, chestxray, etc.)")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model architecture (e.g., clip, biomedclip, etc.)")
    
    # Files
    parser.add_argument("--index_path", type=str, default=None,
                        help="Path to txt file containing selected indices (one per line)")
    
    # Attack configuration
    parser.add_argument("--attacker_name", type=str, required=True,
                        choices=[ "ES_1_Lambda", 'PGD'],
                        help="Name of attacker algorithm")
    parser.add_argument("--epsilon", type=float, default=8/255,
                        help="Maximum perturbation magnitude (default: 8/255)")
    parser.add_argument("--norm", type=str, default="linf",
                        choices=["linf", "l2"],
                        help="Norm constraint type")
    parser.add_argument("--max_evaluation", type=int, default=10000)
    parser.add_argument("--PGD_steps", type=int, default=100)
    parser.add_argument("--lamda", type=int, default=50)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)


    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    
    # outdir
    parser.add_argument("--out_dir", type=str, default="attack_new")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    main(args)
    
    
# CUDA_VISIBLE_DEVICES=4 python main_atttack.py --dataset_name rsna --model_name medclip --index_path evaluate_result/selected_indices_covid_medclip.txt --prompt_path evaluate_result/model_name\=medclip_dataset\=rsna_prompt.json --attacker_name ES_1_1 --epsilon 0.03 --norm linf --mode pre_transform --seed 22520691
