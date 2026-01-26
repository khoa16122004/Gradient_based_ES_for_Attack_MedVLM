import sys
import os
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Union
from PIL import Image
from ..utils import constants
from collections import OrderedDict
from huggingface_hub import hf_hub_download, snapshot_download
import torch.nn.functional as F

# Add RobustMedCLIP to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../RobustMedCLIP'))
try:
    from utils import _MODELS
    from models import (
        ClipZeroShot,
        BioMedClipZeroShot,
        RobustMedClip,
    )
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current sys.path: {sys.path}")
    raise

class RMedCLIP(nn.Module):
    def __init__(self):
        super().__init__()

        model_class = eval(_MODELS['rmedclip']['class_name'])

        # ===== Download from HuggingFace =====

        local_dir = snapshot_download(
            repo_id="razaimam45/RobustMedCLIP",
            allow_patterns=[
                "exp-rank-16/vit/fewshot_100_percent/checkpoints/best_model/*"
            ]
        )

        ckpt_dir = os.path.join(
            local_dir,
            "exp-rank-16",
            "vit",
            "fewshot_100_percent",
            "checkpoints",
            "best_model"
        )

        pretrained_path = os.path.join(ckpt_dir, "model.pth")

        print("Loaded checkpoint from:", pretrained_path)


        self.model = model_class(
            vision_cls='vit',
            device='cuda',
            # lora_rank=16,
            lora_rank=8,
            load_pretrained=False,
        )

        self.model.load(pretrained_path)
        self.model = self.model.cuda()

        self.normalize_transform = constants.TENSOR_NORMALIZE_TRANSFORM['biomedclip']



    def encode_text(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode text inputs to embeddings.
        
        Args:
            texts: Single text string or list of text strings
            normalize: Whether to normalize the embeddings
            
        Returns:
            Text embeddings tensor
        """
        if isinstance(texts, str):
            texts = [texts]
        
        text_features = self.model.text_features(texts)
        
        if normalize:
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features

    def encode_posttransform_image( # truyền voad image tensor dwuodjc scale
        self,
        images: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode image inputs to embeddings.
        
        Args:
            images: Image tensor, PIL Image, or list of PIL Images
            normalize: Whether to normalize the embeddings
            
        Returns:
            Image embeddings tensor
        """
        # Handle different input types
        image_tensors = self.normalize_transform(images)
        
        image_features = self.model.model.encode_image(image_tensors)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features    
            
    def encode_pretransform_image( # truyền voad image tensor dwuodjc scale
        self,
        images: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode image inputs to embeddings.
        
        Args:
            images: Image tensor, PIL Image, or list of PIL Images
            normalize: Whether to normalize the embeddings
            
        Returns:
            Image embeddings tensor
        """
        # Handle different input types
        # images_ = torch.round(images * 255.0).clamp(0, 255)
        images_ = (images * 255.0).clamp(0, 255)

        # Resize to model input size
        image_tensors = F.interpolate(images_, size=(224, 224), mode="bilinear", align_corners=False)
        image_tensors = image_tensors / 255.0
        image_tensors = self.normalize_transform(image_tensors)
        


        
        image_features = self.model.model.encode_image(image_tensors)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features  


if __name__ == "__main__":
    model = RMedCLIP(
        pretrained_path=r'D:\Gradient_based_ES_for_Attack_MedVLM\RobustMedCLIP\model.pth'
    )

    # ===== TEXT TEST =====
    feature_text = model.encode_text(['pneumonia'])
    print("Text feature shape:", feature_text.shape)

    # ===== IMAGE TEST =====

    # Fake image: batch=1, RGB=3, H=W=512
    img = torch.randint(
        low=0,
        high=256,
        size=(1, 3, 224, 224),
        dtype=torch.uint8
    ).float()

    # Convert to float + scale to [0,1]
    # img = img.float() / 255.0

    # Move to GPU
    img = img.cuda()

    with torch.no_grad():
        feature_img = model.encode_posttransform_image(img)

    print("Image feature shape:", feature_img.shape)
