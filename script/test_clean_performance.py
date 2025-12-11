from modules.dataset.factory import DatasetFactory
from modules.utils.constants import MODEL_TRANSFORMS, DEFAULT_TEMPLATES, RSNA_CLASS_PROMPTS, RSNA_CLASS_PROMPTS, SIZE_TRANSFORM, DATA_ROOT
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
    json_path = f"evaluate_result/{args.model_name}.json"

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
    with open(json_path, "w") as f:
        json.dump(correct_samples, f, indent=4)

    print(f"\nSaved correct samples to {json_path}")


def get_args():
    parser = argparse.ArgumentParser(description="Clean Performance Evaluation")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
