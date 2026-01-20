import torch
import torch.nn as nn
from typing import List
from .util import pil_to_tensor, tensor_to_pillow, project_delta
from time import time
import math


class EvaluatePerturbation:
    def __init__(
        self,
        model: nn.Module,
        class_prompts: List[str], # (NUM_CLASSES x D)
        eps: float=0.03,
        norm: str='linf',
        target_image_feat: str = None,
        target_text_feat: str = None,
        mode: str = "post_transform"
    ):
        self.model = model
        self.class_text_feats = self.extract_centroid_vector(class_prompts)
        self.eps = eps
        self.norm = norm
        self.target_image_feat = target_image_feat
        self.target_text_feat = target_text_feat
        self.mode = mode
        
    def set_data(self, image, clean_pred_id):
        self.img = image
        self.img_tensor = pil_to_tensor([image]).cuda()
        self.clean_pred_id = clean_pred_id
        
    
    @torch.no_grad() 
    def extract_centroid_vector(self, class_prompts): 
        class_features = [] 
        for class_name, item in class_prompts.items(): 
            text_feats = self.model.encode_text(item) 
            mean_feats = text_feats.mean(dim=0)
            class_features.append(mean_feats) 
            
        class_features = torch.stack(class_features) # NUM_ClASS x D 
        return class_features
            
    
    @torch.no_grad()
    def cal_l2(self, perturbations: torch.Tensor) -> torch.Tensor:
        return perturbations.view(perturbations.size(0), -1).norm(p=2, dim=1)
    
    @torch.no_grad()
    def evaluate_blackbox(self, perturbations: torch.Tensor):
        perturbations_ = perturbations.clone()
        adv_imgs = self.img_tensor + perturbations_
        adv_imgs = torch.clamp(adv_imgs, 0, 1)

        if self.mode == "post_transform":
            adv_feats = self.model.encode_posttransform_image(adv_imgs)  # (B, D)
        
        elif self.mode == "pre_transform":
            adv_feats = self.model.encode_pretransform_image(adv_imgs)  # (B, D)
        
        sims = adv_feats @ self.class_text_feats.T     # (B, NUM_CLASSES)
        correct_sim = sims[:, self.clean_pred_id].unsqueeze(-1)

        # Max of other classes
        mask = torch.ones_like(sims, dtype=bool)
        mask[:, self.clean_pred_id] = False
        other_max_sim = sims[mask].view(sims.size(0), -1).max(dim=1, keepdim=True).values  # (B, 1)
        # margin = correct_sim - other_max_sim
        margin = correct_sim - other_max_sim + 0.1

        if self.target_image_feat is not None:
            target_sim = adv_feats @ self.target_image_feat.T
            margin = margin - target_sim

        elif self.target_text_feat is not None:
            target_sim = adv_feats @ self.target_text_feat.T
            margin = margin - target_sim


        l2 = self.cal_l2(perturbations_)
        return margin, l2

    def evaluate_whitebox(self, perturbations: torch.Tensor):
        perturbations_ = perturbations.clone()
        adv_imgs = self.img_tensor + perturbations_
        adv_imgs = torch.clamp(adv_imgs, 0, 1)
                
        if self.mode == "post_transform":
            adv_feats = self.model.encode_posttransform_image(adv_imgs)  # (B, D)
        
        elif self.mode == "pre_transform":
            adv_feats = self.model.encode_pretransform_image(adv_imgs)  # (B, D)
                    
        sims = adv_feats @ self.class_text_feats.T     # (B, NUM_CLASSES)
        # Correct class similarity
        correct_sim = sims[:, self.clean_pred_id].unsqueeze(-1)

        # Max of other classes
        mask = torch.ones_like(sims, dtype=bool)
        mask[:, self.clean_pred_id] = False
        other_max_sim = sims[mask].view(sims.size(0), -1).max(dim=1, keepdim=True).values  # (B, 1)
        margin = correct_sim - other_max_sim + 0.1

        
        # l2
        l2 = self.cal_l2(perturbations_)
        return margin, l2
    
  
    
    def take_adv_img(self, perturbation):
        adv_imgs = self.img_tensor + perturbation
        adv_imgs = torch.clamp(adv_imgs, 0, 1)
        pil_adv_imgs = tensor_to_pillow(adv_imgs) # pillow image
        return adv_imgs, pil_adv_imgs
    

    

    
    