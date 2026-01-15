from modules.dataset.factory import DatasetFactory
from modules.utils.constants import MODEL_TRANSFORMS, DEFAULT_TEMPLATES, RSNA_CLASS_PROMPTS, RSNA_CLASS_PROMPTS, SIZE_TRANSFORM, DATA_ROOT, ENTREP_CLASS_PROMPTS
from modules.models.factory import ModelFactory
from modules.utils.helpers import setup_seed, _extract_label, load_open_clip_model
from tqdm import tqdm
import numpy as np
import torch
import json
from modules.attack.attack import ES_1_Lambda
from modules.attack.evaluator import EvaluatePerturbation
from modules.attack.util import seed_everything 
import os
from torchvision import transforms
import yaml
import pandas as pd
from PIL import Image
import argparse
import torch.nn as nn

_toTensor = transforms.ToTensor()


class Denoiser(nn.Module):
    def __init__(self, channels=3, num_of_layers=17):
        super().__init__()
        kernel_size = 3
        padding = 1
        features = 64

        layers = [
            nn.Conv2d(channels, features, kernel_size, padding=padding, bias=False),
            nn.ReLU(inplace=True)
        ]

        for _ in range(num_of_layers - 2):
            layers += [
                nn.Conv2d(features, features, kernel_size, padding=padding, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True)
            ]

        layers.append(
            nn.Conv2d(features, channels, kernel_size, padding=padding, bias=False)
        )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  

def main(args):
    # ========= Dataset ========= #
    dataset = DatasetFactory.create_dataset(
        dataset_name=args.dataset_name,
        model_type='medclip',
        data_root=DATA_ROOT,
        transform=None
    )

    if args.pretrained_denoiser:

        denoiser = Denoiser().cuda()
        
        ckpt = torch.load(args.pretrained_denoiser, map_location="cuda")
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

        print(denoiser.load_state_dict(state_dict, strict=False))
    

    # ========= class_prompt_based ========= #
    if args.dataset_name == "rsna":
        class_prompts = RSNA_CLASS_PROMPTS

    if args.model_name == "entrep":
        class_prompts = ENTREP_CLASS_PROMPTS
    num_classes = len(class_prompts)

    # ========= Model ========= #
    if args.model_name in ['medclip', 'biomedclip']:
        model = ModelFactory.create_model(
            model_type=args.model_name,
            variant='base',
            pretrained=True,
            mode_pretrained=args.mode_pretrained
        )

    elif args.model_name == "entrep":
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
    else:
        raise NotImplementedError(f"Model {args.model_name} not implemented.")

    # ========= compute class features ========= #
    class_features = []
    for class_name, item in class_prompts.items():
        text_feats = model.encode_text(item)
        class_features.append(text_feats.mean(dim=0))
    class_features = torch.stack(class_features)

    # ========= Track performance ========= #
    total = 0
    correct = 0
    correct_samples = []

    class_total = [0] * num_classes
    class_correct = [0] * num_classes

    # ========= Prepare output folder ========= #
    os.makedirs("evaluate_result", exist_ok=True)

    # ========= Evaluation loop ========= #
    for i in tqdm(range(0, len(dataset), args.batch_size)):
        images_batch = []
        labels_batch = []
        ids_batch = []

        for j in range(i, min(i + args.batch_size, len(dataset))):
            image, label_dict = dataset[j]
            image = image.convert("RGB")
            image_tensor = _toTensor(image)
            images_batch.append(image_tensor)

            label_id = _extract_label(label_dict)
            labels_batch.append(label_id)
            ids_batch.append(j)

        images_batch = torch.stack(images_batch).cuda()
        if args.pretrained_denoiser:
            noises = torch.randn_like(images_batch).cuda()
            images_batch = (images_batch + args.epsilon * noises).clamp(0, 1)
            images_batch = denoiser(images_batch)
        labels_batch = torch.tensor(labels_batch).cuda()



        with torch.no_grad():
            image_feats = model.encode_pretransform_image(images_batch)
            sims = image_feats @ class_features.T
            pred_id = sims.argmax(dim=1)

        # ===== overall accuracy ===== #
        total += labels_batch.size(0)
        correct_mask = (pred_id == labels_batch)
        correct += correct_mask.sum().item()

        # ===== per-class accuracy ===== #
        for gt, pred in zip(labels_batch, pred_id):
            class_total[int(gt)] += 1
            if gt == pred:
                class_correct[int(gt)] += 1

        # ===== save correct samples ===== #
        for idx, ok in enumerate(correct_mask):
            if ok:
                correct_samples.append({
                    "id": ids_batch[idx],
                    "gt_id": int(labels_batch[idx].item()),
                    "pred_id": int(pred_id[idx].item())
                })

    # ========= Compute overall accuracy ========= #
    acc = correct / total
    print(f"\nOverall Accuracy: {acc * 100:.2f}%\n")

    # ========= Print class-wise performance ========= #
    print("===== Class-wise Performance =====")
    for c in range(num_classes):
        if class_total[c] > 0:
            acc_c = class_correct[c] / class_total[c]
            print(f"Class {c}: {acc_c * 100:.2f}%  ({class_correct[c]}/{class_total[c]})")
        else:
            print(f"Class {c}: No samples")

    # ========= Write JSON ========= #
    with open(args.json_path, "w") as f:
        json.dump(correct_samples, f, indent=4)

    print(f"\nSaved correct samples to {args.json_path}")


def get_args():
    parser = argparse.ArgumentParser(description="Clean Performance Evaluation")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument("--pretrained_denoiser", type=str, default=None)
    parser.add_argument("--mode_pretrained", type=str, default="scratch")
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--json_path", type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
