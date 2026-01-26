import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional
import json
import os
import argparse
from tqdm import tqdm
import sys
import time
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import torch.nn.functional as F

sys.path.append('.')
from dataset import get_dataloader, DATA_COLLECTIONS
from cls_to_names import *
from utils import _MODELS
from models import (
    ClipZeroShot,
    # MedclipZeroShot,
    BioMedClipZeroShot,
    # UniMedClipZeroShot,
    RobustMedClip,
)

# Define available models

DATASETS = {
    "medmnist": ('../MedMNIST-C', ["bloodmnist", "retinamnist", "breastmnist", "octmnist", "pneumoniamnist"]),
    "medimeta": ('../MediMeta-C', ["aml", "fundus", "mammo_calc", "mammo_mass", "pneumonia", "oct", "pbc"])
}

AVAILABLE_CORRUPTIONS = [
    'clean', 'gaussian_noise', 'impulse_noise', 'motion_blur', 'zoom_blur', 'pixelate',
      'snow', 'frost', 'fog', 'brightness', 'contrast' , 'shot_noise' , 'defocus_blur', 'glass_blur', 
]

SEVERITY_LEVELS = [1, 2, 3, 4, 5]

PROMPT_TEMPLATES = [
    "a medical image of {}",
    "an image of {}",
    "a picture of {}",
    "a photo of a {}",
    "a photograph of {}",
    "a medical image showing {}",
]

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="Evaluate CLIP models on medical datasets")
    parser.add_argument("--model", type=str, required=True, 
                        help="Model to evaluate")
    parser.add_argument("--backbone", type=str, required=True, 
                        help="Backbone to evaluate")
    parser.add_argument("--pretrained_path", type=str, required=False,
                        help="Path to pretrained model weights")
    parser.add_argument("--gpu", type=int, default=0, 
                        help="GPU ID to use")
    parser.add_argument("--corruptions", type=str, default="clean", 
                        help="Comma-separated list of corruptions to evaluate")
    parser.add_argument("--overwrite", action="store_true", 
                        help="Overwrite existing results")
    parser.add_argument("--prompt_template", type=str, default="a medical image of a {} belonging to {}",
                        help="Prompt template for text features")
    
    return parser.parse_args()

def load_model(args, device="cuda"):
    """
    Load a model once to be used for multiple dataset evaluations
    """
    # Initialize the model
    model_class = eval(_MODELS[args.model]['class_name'])
    model = model_class(
        vision_cls=args.backbone,
        device=device,
        lora_rank=16,
        load_pretrained=False,
    )
    # print(model)
    if args.pretrained_path:
        print(f"Loading model from {args.pretrained_path}")
        model.load(args.pretrained_path)
    model.to(device)
    return model

def main():
    args = parse_args()
    
    # Validate model and backbone
    if args.model not in _MODELS:
        print(f"Error: Unknown model {args.model}")
        sys.exit(1)
        
    if args.backbone not in _MODELS[args.model]['backbones']:
        print(f"Error: Unknown backbone {args.backbone} for model {args.model}")
        sys.exit(1)
    
    args.lora_rank = 16
    model = load_model(args, 'cuda')

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
