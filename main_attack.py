from modules.dataset.factory import DatasetFactory
from modules.utils.constants import MODEL_TRANSFORMS, DEFAULT_TEMPLATES, RSNA_CLASS_PROMPTS, RSNA_CLASS_PROMPTS, SIZE_TRANSFORM, DATA_ROOT, ENTREP_CLASS_PROMPTS
from modules.models.factory import ModelFactory
from tqdm import tqdm
import numpy as np
import torch
import json
from modules.attack.attack import ES_1_Lambda, PGDAttack, ES_1_Lambda_Gradient, CEM_Attack, ESGD_Attack, NES_Attack, GridES_1_Lambda
from modules.attack.evaluator import EvaluatePerturbation
from modules.attack.util import seed_everything 
from modules.utils.helpers import _extract_label, load_open_clip_model
import os
from torchvision import transforms
import yaml
import pandas as pd
from PIL import Image
from collections import OrderedDict
import pickle as pkl

_toTensor = transforms.ToTensor()

def main(args):
    # ========= Dataset ========= #        
    dataset = DatasetFactory.create_dataset(
        dataset_name=args.dataset_name,
        model_type=args.model_name,
        data_root=DATA_ROOT,
        transform=None
    )


    # ============ size transform ===========
    size_transform = SIZE_TRANSFORM[args.model_name]



   # ========= class_prompt_based ========= #
    if args.model_name in ['medclip', 'biomedclip', 'ViT-B-32', 'ViT-B-16', "ViT-L-14"]:
        class_prompts = RSNA_CLASS_PROMPTS
    elif args.model_name == "entrep":
        class_prompts = ENTREP_CLASS_PROMPTS    
    num_classes = len(class_prompts)

    # ========= class_prompt_based ========= #
    class_prompts = RSNA_CLASS_PROMPTS
    num_classes = len(class_prompts)

    # ========= Model ========= #
    if args.model_name in ['medclip', 'biomedclip']:
        model = ModelFactory.create_model(
            model_type=args.model_name,
            variant='base',
            pretrained=True,
            mode_pretrained=args.mode_pretrained
        )

    elif args.model_name == 'entrep':
        config_path = "configs/entrep_contrastive.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        model_config = config.get('model', {})
        model = ModelFactory.create_model(
            model_type="entrep",
            variant='base',
            checkpoint=None,
            pretrained=False,
            **{k: v for k, v in model_config.items() if k != 'model_type' and k != "pretrained" and k != "checkpoint"},
            mode_pretrained=args.mode_pretrained
            )            
    elif args.model_name in ['ViT-B-32', 'ViT-B-16', 'ViT-L-14']:
        model = ModelFactory.create_model(
            model_type='ViT',
            variant=args.model_name,
        )

    elif args.model_name == "rmedclip":
        model = ModelFactory.create_model(
            model_type='rmedclip',
            variant='base',
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
    if args.target_image:
        target_image = Image.open(args.target_image)
        target_image = size_transform(target_image).convert("RGB")
        target_image_feat = model.encode_posttransform_image(_toTensor(target_image).unsqueeze(0).cuda())
    else:
        target_image_feat = None
    
    
    if args.target_text:
        target_text = args.target_text
        target_text_feat = model.encode_text([target_text]) 
    else:
        target_text_feat = None

    
    # ============= evaluatỏ =================

    evaluator = EvaluatePerturbation(
        model=model,
        class_prompts=class_prompts,
        eps=args.epsilon,
        norm=args.norm,
        target_image_feat=target_image_feat,
        target_text_feat=target_text_feat,
        mode=args.mode
    )
    img, label_dict = dataset[0]
    if args.mode == "post_transform": # knowing transform
        img_attack = size_transform(img).convert("RGB")
    elif args.mode == "pre_transform": # w/o knoiwng transform
        img_attack = img.convert("RGB")

    evaluator.set_data( # setting size
        image=img_attack,
        clean_pred_id=None
    )
          

    
    # path dir save
    if args.attacker_name == "ES_1_Lambda":
        # ko có lkambda mặt định là 50
        save_dir = os.path.join(args.out_dir, args.model_name, args.mode_pretrained, args.dataset_name, f"attack_name={args.attacker_name}_mode={args.mode}_epsilon={args.epsilon}_pattern={args.pattern}_lamda={args.lamda}_norm={args.norm}_seed={args.seed}")
    elif args.attacker_name == "NES":
        save_dir = os.path.join(args.out_dir, args.model_name, args.mode_pretrained, args.dataset_name, f"attack_name={args.attacker_name}_mode={args.mode}_epsilon={args.epsilon}_q={args.q}_alpha={args.alpha}_norm={args.norm}_seed={args.seed}")

    
    elif args.attacker_name == "PGD":
        save_dir = os.path.join(args.out_dir, args.model_name, args.mode_pretrained, args.dataset_name, f"attack_name={args.attacker_name}_mode={args.mode}_epsilon={args.epsilon}_steps={args.PGD_steps}_alpha={args.alpha}_norm={args.norm}_seed={args.seed}")
    elif args.attacker_name == "ES_1_Lambda_Gradient":
        save_dir = os.path.join(args.out_dir, args.model_name, args.dataset_name, f"attack_name={args.attacker_name}_mode={args.mode}_epsilon={args.epsilon}_theta={args.theta}_lamda={args.lamda}_norm={args.norm}_seed={args.seed}")
    elif args.attacker_name == "CEM":
        save_dir = os.path.join(args.out_dir, args.model_name, args.dataset_name, f"attack_name={args.attacker_name}_mode={args.mode}_epsilon={args.epsilon}_lamda={args.lamda}_mu={args.mu}_norm={args.norm}_seed={args.seed}")
    elif args.attacker_name == "ESGD":
        save_dir = os.path.join(args.out_dir, args.model_name, args.dataset_name, f"attack_name={args.attacker_name}_mode={args.mode}_epsilon={args.epsilon}_lamda={args.lamda}_mu={args.mu}_norm={args.norm}_seed={args.seed}")
    elif args.attacker_name == "GridES_1_Lambda":
        save_dir = os.path.join(args.out_dir, args.model_name, args.dataset_name, f"attack_name={args.attacker_name}_mode={args.mode}_epsilon={args.epsilon}_patch_size={args.patch_size}_lamda={args.lamda}_norm={args.norm}_seed={args.seed}")

    os.makedirs(save_dir, exist_ok=True)
    
    
    # ========================== Attacker ==========================
    if args.attacker_name == "ES_1_Lambda": # number of evalation = ierations * lambda
        attacker = ES_1_Lambda(
            evaluator=evaluator,
            pattern=args.pattern,
            eps=args.epsilon,
            norm=args.norm,
            max_evaluation=args.max_evaluation,
            lam=args.lamda,
        )
    elif args.attacker_name == "PGD":
        attacker = PGDAttack(
            eps=args.epsilon,
            alpha=args.alpha,
            norm=args.norm,
            steps=args.PGD_steps,
            evaluator=evaluator
        )

    elif args.attacker_name == "ES_1_Lambda_Gradient":
        attacker = ES_1_Lambda_Gradient(
            evaluator=evaluator,
            eps=args.epsilon,
            norm=args.norm,
            theta=args.theta,
            max_evaluation=args.max_evaluation,
            lam=args.lamda
        )
    elif args.attacker_name == "CEM":
        attacker = CEM_Attack(
            evaluator=evaluator,
            eps=args.epsilon,
            norm=args.norm,
            max_evaluation=args.max_evaluation,
            N=args.lamda,
            Ne=args.mu
        )

    elif args.attacker_name == "ESGD":
        attacker = ESGD_Attack(
            evaluator=evaluator,
            eps=args.epsilon,
            norm=args.norm,
            max_evaluation=args.max_evaluation,
            lam=args.lamda,
            mu=args.mu
        )
    
    elif args.attacker_name == "NES":
        attacker = NES_Attack(
            evaluator=evaluator,
            eps=args.epsilon,
            norm=args.norm,
            max_evaluation=args.max_evaluation,
            q=args.q, # random vector query,
            batch_q=args.batch_q, # batch for estimation,
            alpha=args.alpha
        )

    elif args.attacker_name == "GridES_1_Lambda":
        attacker = GridES_1_Lambda(
            evaluator=evaluator,
            patch_size=args.patch_size,
            eps=args.epsilon,
            max_evaluation=args.max_evaluation,
            lam=args.lamda,
            local_steps=args.local_steps
        )
    
    



    # --------------------------- Main LOOP ------------------ 
    for index in tqdm(indxs):
        img, label_dict = dataset[index]
        label_id = _extract_label(label_dict)


        if args.mode == "post_transform": # knowing transform
            img_attack = size_transform(img).convert("RGB")
            img_attack_tensor = _toTensor(img_attack).unsqueeze(0).cuda()
            img_feats = model.encode_posttransform_image(img_attack_tensor)
        
        elif args.mode == "pre_transform": # w/o knoiwng transform
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
        if args.mode == "post_transform": # knowing transform
            img_feats = model.encode_posttransform_image(adv_imgs) # (B, NUM_CLASS)
            
        elif args.mode == "pre_transform":
            img_feats = model.encode_pretransform_image(adv_imgs)  # (B, D)
                    
        sims = img_feats @ evaluator.class_text_feats.T                     # (B, NUM_CLASS)
        adv_preds = sims.argmax(dim=-1).item()                    # (B,)
        # print("Adv preds: ", preds)
        
        # save_dir
        index_dir = os.path.join(save_dir, str(index))
        os.makedirs(index_dir, exist_ok=True)
        # pil_adv_imgs[0].save(os.path.join(index_dir, f'adv_img.png'))

        # img_attack.save(os.path.join(index_dir, "clean_img.png"))
        
        info = {
            'clean_pred': clean_preds,
            'adv_pred': adv_preds,
            'gt': label_id,
            'success_evaluation': result['success_evaluation'],
            'l2': result['l2']
        }
        with open(os.path.join(index_dir, "info.json"), "w") as f:
            json.dump(info, f, indent=4)

        with open(os.path.join(index_dir, "history.txt"), "w") as f:
            for (n_eval, score) in result['history']:
                f.write(f"{str(n_eval)},{str(score)}\n") 
        
        with open(os.path.join(index_dir, "adv_img.pkl"), "wb") as f:
            pkl.dump(adv_imgs.cpu(), f)
            
            
        
        
        
        
        
        
    

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
                        choices=[ "ES_1_Lambda", "ES_1_Lambda_Gradient", 'PGD', "CEM", "ESGD", "NES", "GridES_1_Lambda"],
                        help="Name of attacker algorithm")
    parser.add_argument("--epsilon", type=float, default=8/255,
                        help="Maximum perturbation magnitude (default: 8/255)")
    parser.add_argument("--norm", type=str, default="linf",
                        choices=["linf", "l2"],
                        help="Norm constraint type")
    parser.add_argument("--theta", type=float, default=0.001)
    parser.add_argument("--max_evaluation", type=int, default=10000)
    parser.add_argument("--PGD_steps", type=int, default=100)
    parser.add_argument("--lamda", type=int, default=50)
    parser.add_argument("--mu", type=int, default=8)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--target_image", type=str, default=None)
    parser.add_argument("--target_text", type=str, default=None)
    parser.add_argument("--mode", type=str,)
    parser.add_argument("--patch_size", type=int)
    parser.add_argument("--local_steps", type=int)
    parser.add_argument("--pattern", type=str, default='')
    parser.add_argument("--mode_pretrained", type=str, default='scratch')
    # NES
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--batch_q", type=int)
    parser.add_argument("--q", type=int)

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
