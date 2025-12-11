import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from typing import Optional
from pathlib import Path
from transformers import AutoModelForMaskedLM
from functools import partial
from PIL import Image
# Import base classes
from .base import TextEncoder, VisionEncoder
from ..utils import constants
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import open_clip
from collections import OrderedDict
from ..utils.logging_config import get_logger
logger = get_logger(__name__)

class ViTModel(nn.Module):
    def __init__(self, model_name: str):
        super(ViTModel, self).__init__()
        import open_clip
        pretrained_path = self.download_weights(model_name)
        self.model, self.preprocess, self.tokenizer = self.load_open_clip_model(model_name, pretrained_path)
        self.normalize_transform = constants.TENSOR_NORMALIZE_TRANSFORM[model_name]

    def _strip_prefix_from_state_dict(self, sd, prefixes=('visual.')):

        if isinstance(sd, dict) and 'state_dict' in sd and isinstance(sd['state_dict'], dict):
            sd = sd['state_dict']

        new_sd = OrderedDict()
        for k, v in sd.items():
            nk = k
            # Bỏ lần lượt các prefix nếu có
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p):]
            new_sd[nk] = v
        return new_sd

    def load_open_clip_model(self, model_name, pretrained_path):
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained="openai",
         )
        model = model.cuda()
        
        logger.info(f"Loading pretrained weights from {pretrained_path}")

        ckpt = torch.load(pretrained_path, map_location="cpu")['state_dict']
        # state_dict = self._strip_prefix_from_state_dict(ckpt)
        missing, unexpected = model.load_state_dict(ckpt, strict=True)

        logger.info(f"Missing keys: {missing}")

        model.eval()
        tokenizer = open_clip.get_tokenizer(model_name)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model {model_name} has {total_params/1e6:.2f}M parameters.")


        return model, preprocess, tokenizer

    def download_weights(self, model_name):
        repo_id = "Woffy/ViT_finetuning_OpenCLIP_MiMic"
        if model_name == "ViT-B-32":
            file_name = "openclip_vit_B_32_sratch_openai.pt"
        elif model_name == "ViT-B-16":
            file_name = "openclip_vit_B_16_sratch_openai.pt"
        elif model_name == "ViT-L-14":
            file_name = "openclip_vit_B_L_14_scratch_openai.pt"

        logger.info(f"Downloading weights for {model_name}:{file_name} from Huggingface/{repo_id}.")
        checkpoint_path = hf_hub_download(repo_id=repo_id, filename=file_name)
        return checkpoint_path
    
    def encode_text(
        self,
        texts: str,
        normalize: bool = True
    ):
    
        texts = self.tokenizer(texts).cuda()
        text_features = self.model.encode_text(texts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features

    
    def encode_pretransform_image( # truyền voad image tensor dwuodjc scale
        self,
        images: torch.Tensor
    ) -> torch.Tensor:
        image_tensors = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        center_crop = T.CenterCrop(224)
        image_tensors = center_crop(image_tensors)
        image_tensors = self.normalize_transform(image_tensors)
        
        image_features = self.model.encode_image(image_tensors)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features 

    def encode_posttransform_image( 
        self,
        images: torch.Tensor
    ) -> torch.Tensor:

        image_tensors = self.normalize_transform(images)
        
        image_features = self.model.encode_image(image_tensors)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features 