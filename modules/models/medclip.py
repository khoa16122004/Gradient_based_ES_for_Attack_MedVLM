import pdb
import os
import copy
from collections import defaultdict
import requests
from typing import Optional, Dict, List, Union
from PIL import Image
from torchvision.utils import save_image

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
import numpy as np
import torchvision
import torch.nn.functional as F

from ..utils import constants
from .base import VisionLanguageModel
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from huggingface_hub import hf_hub_download
from collections import OrderedDict
class MedCLIPTextModel(nn.Module):
    def __init__(self,
        bert_type=constants.BERT_TYPE,
        proj_dim = 512,
        proj_bias = False) -> None:
        super().__init__()
        self.bert_type = bert_type
        self.last_n_layer = 4
        self.model = AutoModel.from_pretrained(self.bert_type, output_hidden_states=True)
        # this tokenizer is actually not used
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
        self.projection_head = nn.Linear(768, proj_dim, bias=proj_bias)

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2], output['hidden_states'][-1]]) # n_layer, batch, seqlen, emb_dim
        embed = last_hidden_states.permute(1,0,2,3).mean(2).mean(1) # pooling
        embed = self.projection_head(embed)
        return embed

class MedCLIPVisionModel(nn.Module):
    '''
    take resnet50 as backbone.
    '''
    def __init__(self, checkpoint=None, medclip_checkpoint=None):
        super().__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        num_fts = self.model.fc.in_features
        self.model.fc = nn.Linear(num_fts, 512, bias=False) # projection head
        if checkpoint is not None:
            state_dict = torch.load(os.path.join(checkpoint, constants.WEIGHTS_NAME), map_location='cpu')
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            print('missing keys:', missing_keys)
            print('unexpected keys:', unexpected_keys)
            print('load model weight from:', checkpoint)
        if medclip_checkpoint is not None:
            self.load_from_medclip(medclip_checkpoint)

    def load_from_medclip(self, checkpoint):
        '''handle key mismatch of medclip and the vision encoder.
        '''
        state_dict = torch.load(os.path.join(checkpoint, constants.WEIGHTS_NAME), map_location='cpu')
        new_state_dict = {}
        for key in state_dict.keys():
            if 'vision_model' in key:
                new_state_dict[key.replace('vision_model.','')] = state_dict[key]
        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
        print('missing keys:', missing_keys)
        print('unexpected keys:', unexpected_keys)
        print('load model weight from:', checkpoint)

    def forward(self, pixel_values, **kwargs):
        '''args:
        pixel_values: tensor with shape [bs, 3, img_size, img_size]
        '''
        if pixel_values.shape[1] == 1: pixel_values = pixel_values.repeat((1,3,1,1))
        img_embeds = self.model(pixel_values)
        return img_embeds

class MedCLIPVisionModelViT(nn.Module):
    '''take an VIT model as the backbone.
    '''
    def __init__(self, checkpoint=None, medclip_checkpoint=None) -> None:
        '''args:
        checkpoint: load from the vision encoder checkpoint
        medclip_checkpoint: load from the vision-text dual encoders checkpoint
        '''
        super().__init__()
        self.vit_type = constants.VIT_TYPE
        self.model = AutoModel.from_pretrained(self.vit_type)
        self.projection_head = nn.Linear(768, 512, bias=False)
        if checkpoint is not None:
            state_dict = torch.load(os.path.join(checkpoint, constants.WEIGHTS_NAME), map_location='cpu')
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            print('missing keys:', missing_keys)
            print('unexpected keys:', unexpected_keys)
            print('load model weight from:', checkpoint)
        if medclip_checkpoint is not None:
            self.load_from_medclip(medclip_checkpoint)

    def load_from_medclip(self, checkpoint):
        '''handle key mismatch of medclip and the vision encoder.
        '''
        state_dict = torch.load(os.path.join(checkpoint, constants.WEIGHTS_NAME), map_location='cpu')
        new_state_dict = {}
        for key in state_dict.keys():
            if 'vision_model' in key:
                new_state_dict[key.replace('vision_model.','')] = state_dict[key]
        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
        print('missing keys:', missing_keys)
        print('unexpected keys:', unexpected_keys)
        print('load model weight from:', checkpoint)


    def forward(self, pixel_values, project=True):
        '''args:
        pixel_values: tensor with shape [bs, 3, img_size, img_size]
        '''
        if pixel_values.shape[1] == 1: pixel_values = pixel_values.repeat((1,3,1,1))
        output = self.model(pixel_values)
        img_embeds = output['pooler_output']
        if project:
            img_embeds = self.projection_head(img_embeds)
        return img_embeds

class MedCLIPModel(VisionLanguageModel):
    def __init__(self,
        text_encoder_type='bert',
        vision_encoder_type='vit',
        checkpoint=None,
        vision_checkpoint=None,
        vision_pretrained=None,  # New parameter for pretrained vision weights
        text_pretrained=None,     # New parameter for pretrained text weights
        logit_scale_init_value=0.07,
        mode_pretrained='scratch',
        **kwargs
        ) -> None:
        super().__init__()
        
        # Store encoder types
        self.text_encoder_type = text_encoder_type
        self.vision_encoder_type = vision_encoder_type
        
        # New flexible mode - use encoder types
        self.vision_model = self._create_vision_encoder(
            vision_encoder_type, vision_checkpoint
        )
        
        # Create text encoder
        self.text_model = self._create_text_encoder(text_encoder_type)
        
        # Create tokenizer for text processing
        self.tokenizer = AutoTokenizer.from_pretrained(constants.BERT_TYPE)

        # learnable temperature for contrastive loss
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/logit_scale_init_value)))

        self.mode_pretrained = mode_pretrained
        # Load full model checkpoint if provided
        if checkpoint is not None:
            self.load_checkpoint(checkpoint)
        else:
            repo_id = "Woffy/Thesis_Pretrained_Medical_Moddel"
            if self.mode_pretrained == "scratch":    
                file_name = "medclip.pt"
            elif self.mode_pretrained == "ssl":
                file_name = "medclip_ssl_finetuning.pth"
            elif self.mode_pretrained == "at":
                file_name = "medclip_AT.pth"

            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=file_name,
                local_dir=".",          # lưu ngay thư mục hiện tại
                local_dir_use_symlinks=False  # QUAN TRỌNG: copy file thật, không tạo symlink
            )   
            ckpt = torch.load(local_path)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model_state_dict = ckpt["model_state_dict"]
            else:
                model_state_dict = ckpt     
            # model_state_dict = self._strip_prefix_from_state_dict(model_state_dict, 'model.')

            incompatible_keys = self.load_state_dict(model_state_dict, strict=True)
            print(f"Incompatible keys when load {local_path}: {incompatible_keys}")
        # Load vision-specific pretrained weights if provided
        if vision_pretrained is not None:
            self.load_vision_pretrained(vision_pretrained)
        
        # Load text-specific pretrained weights if provided
        if text_pretrained is not None:
            self.load_text_pretrained(text_pretrained)
    
        # proccessor (khoatn fix)
        self.preprocess = constants.MODEL_TRANSFORMS['medclip']
        self.normalize_transform = constants.TENSOR_NORMALIZE_TRANSFORM['medclip']
        self.pil_to_tensor_norm = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),                    # -> [0,1] float32, shape (C,H,W)
            self.normalize_transform,                 # Normalize(mean,std)
        ])
        
        # Move model to device
        self.to(self.device)
    
    def _create_text_encoder(self, encoder_type: str):
        """Create text encoder based on type"""
        if encoder_type == 'bert':
            return MedCLIPTextModel(proj_bias=False)
        else:
            raise ValueError(f"Unsupported text encoder type: {encoder_type}")
   
    def _strip_prefix_from_state_dict(self, sd, prefixes=('visual.', 'module.', 'model.')):

        if isinstance(sd, dict) and 'state_dict' in sd and isinstance(sd['state_dict'], dict):
            sd = sd['state_dict']

        new_sd = OrderedDict()
        for k, v in sd.items():
            nk = k
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p):]
            new_sd[nk] = v
        return new_sd
    def _create_vision_encoder(self, encoder_type: str, checkpoint=None):
        """Create vision encoder based on type"""
        if encoder_type == 'resnet':
            return MedCLIPVisionModel(checkpoint=checkpoint)
        elif encoder_type == 'vit':
            return MedCLIPVisionModelViT(checkpoint=checkpoint)
        else:
            raise ValueError(f"Unsupported vision encoder type: {encoder_type}. Use 'resnet' or 'vit'")
    
    def get_encoder_info(self) -> dict:
        """Get information about current encoders"""
        return {
            'text_encoder': self.text_encoder_type,
            'vision_encoder': self.vision_encoder_type,
            'vision_model_type': type(self.vision_model).__name__,
            'text_model_type': type(self.text_model).__name__
        }
    
    def load_checkpoint(self, checkpoint_path: str, strict: bool = False):
        """
        Load full model weights from checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file or directory
            strict: Whether to strictly enforce that the keys in state_dict match
        """
        if os.path.isdir(checkpoint_path):
            checkpoint_file = os.path.join(checkpoint_path, constants.WEIGHTS_NAME)
        else:
            checkpoint_file = checkpoint_path
            
        if os.path.exists(checkpoint_file):
            state_dict = torch.load(checkpoint_file, map_location='cpu')
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=strict)
            
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
                
            print(f'✅ Loaded model weights from: {checkpoint_file}')
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_file}")
    
    def load_vision_pretrained(self, vision_pretrained_path: str, strict: bool = False):
        """
        Load pretrained vision encoder weights.
        
        Args:
            vision_pretrained_path: Path to the pretrained vision weights
            strict: Whether to strictly enforce that the keys match
        """
        if os.path.exists(vision_pretrained_path):
            state_dict = torch.load(vision_pretrained_path, map_location='cpu')
            
            # Try to load directly first
            try:
                missing_keys, unexpected_keys = self.vision_model.load_state_dict(state_dict, strict=strict)
            except RuntimeError:
                # If direct loading fails, try to extract vision-specific weights
                vision_state_dict = {}
                for key, value in state_dict.items():
                    if 'vision_model' in key:
                        # Remove 'vision_model.' prefix if present
                        new_key = key.replace('vision_model.', '')
                        vision_state_dict[new_key] = value
                    elif 'visual' in key or 'image' in key:
                        vision_state_dict[key] = value
                    else:
                        # Try to load as-is for simple vision model weights
                        vision_state_dict[key] = value
                
                missing_keys, unexpected_keys = self.vision_model.load_state_dict(vision_state_dict, strict=strict)
            
            if missing_keys:
                print(f"Missing keys in vision model: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys in vision model: {unexpected_keys}")
                
            print(f'✅ Loaded vision pretrained weights from: {vision_pretrained_path}')
        else:
            raise FileNotFoundError(f"Vision pretrained weights not found at {vision_pretrained_path}")
    
    def load_text_pretrained(self, text_pretrained_path: str, strict: bool = False):
        """
        Load pretrained text encoder weights.
        
        Args:
            text_pretrained_path: Path to the pretrained text weights
            strict: Whether to strictly enforce that the keys match
        """
        if os.path.exists(text_pretrained_path):
            state_dict = torch.load(text_pretrained_path, map_location='cpu')
            
            # Try to load directly first
            try:
                missing_keys, unexpected_keys = self.text_model.load_state_dict(state_dict, strict=strict)
            except RuntimeError:
                # If direct loading fails, try to extract text-specific weights
                text_state_dict = {}
                for key, value in state_dict.items():
                    if 'text_model' in key:
                        # Remove 'text_model.' prefix if present
                        new_key = key.replace('text_model.', '')
                        text_state_dict[new_key] = value
                    elif 'text' in key or 'bert' in key:
                        text_state_dict[key] = value
                    else:
                        # Try to load as-is for simple text model weights
                        text_state_dict[key] = value
                
                missing_keys, unexpected_keys = self.text_model.load_state_dict(text_state_dict, strict=strict)
            
            if missing_keys:
                print(f"Missing keys in text model: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys in text model: {unexpected_keys}")
                
            print(f'✅ Loaded text pretrained weights from: {text_pretrained_path}')
        else:
            raise FileNotFoundError(f"Text pretrained weights not found at {text_pretrained_path}")
    
    def save_pretrained(self, save_path: str):
        """
        Save model weights to a file or directory.
        
        Args:
            save_path: Path to save the model weights
        """
        if os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)
            save_file = os.path.join(save_path, constants.WEIGHTS_NAME)
        else:
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            save_file = save_path
        
        torch.save(self.state_dict(), save_file)
        print(f'✅ Saved model weights to: {save_file}')
    
    def freeze_vision_encoder(self):
        """Freeze vision encoder parameters."""
        for param in self.vision_model.parameters():
            param.requires_grad = False
        print("Vision encoder frozen")
    
    def unfreeze_vision_encoder(self):
        """Unfreeze vision encoder parameters."""
        for param in self.vision_model.parameters():
            param.requires_grad = True
        print("Vision encoder unfrozen")
    
    def freeze_text_encoder(self):
        """Freeze text encoder parameters."""
        for param in self.text_model.parameters():
            param.requires_grad = False
        print("Text encoder frozen")
    
    def unfreeze_text_encoder(self):
        """Unfreeze text encoder parameters."""
        for param in self.text_model.parameters():
            param.requires_grad = True
        print("Text encoder unfrozen")
    
    def get_trainable_parameters(self) -> dict:
        """Get information about trainable parameters."""
        vision_trainable = sum(p.numel() for p in self.vision_model.parameters() if p.requires_grad)
        vision_total = sum(p.numel() for p in self.vision_model.parameters())
        text_trainable = sum(p.numel() for p in self.text_model.parameters() if p.requires_grad)
        text_total = sum(p.numel() for p in self.text_model.parameters())
        
        return {
            'vision_trainable': vision_trainable,
            'vision_total': vision_total,
            'vision_frozen': vision_total - vision_trainable,
            'text_trainable': text_trainable,
            'text_total': text_total,
            'text_frozen': text_total - text_trainable,
            'total_trainable': vision_trainable + text_trainable,
            'total_parameters': vision_total + text_total
        }

    def from_pretrained(self, input_dir=None):
        '''
        If input_dir is None, download pretrained weight from google cloud and load.
        '''
        import wget
        import zipfile
        pretrained_url = None
        
        # Determine pretrained model based on vision encoder type
        if self.vision_encoder_type == 'resnet' or isinstance(self.vision_model, MedCLIPVisionModel):
            # resnet
            pretrained_url = constants.PRETRAINED_URL_MEDCLIP_RESNET
            if input_dir is None:
                input_dir = '../pretrained/medclip-resnet'
        elif self.vision_encoder_type == 'vit' or isinstance(self.vision_model, MedCLIPVisionModelViT):
            # ViT
            pretrained_url = constants.PRETRAINED_URL_MEDCLIP_VIT
            if input_dir is None:
                input_dir = '../pretrained/medclip-vit'
        else:
            raise ValueError(f'We only have pretrained weight for MedCLIP-ViT or MedCLIP-ResNet, got vision_encoder_type={self.vision_encoder_type}')

        if not os.path.exists(input_dir):
            os.makedirs(input_dir)

            # download url link
            pretrained_url = requests.get(pretrained_url).text
            filename = wget.download(pretrained_url, input_dir)

            # unzip
            zipf = zipfile.ZipFile(filename)
            zipf.extractall(input_dir)
            zipf.close()
            print('\n Download pretrained model from:', pretrained_url)
        
        state_dict = torch.load(os.path.join(input_dir, constants.WEIGHTS_NAME), map_location='cpu')
        self.load_state_dict(state_dict, strict=False)
        print('load model weight from:', input_dir)


    def encode_text(self, texts=None, input_ids=None, attention_mask=None, normalize=True):
        """
        Encode text inputs to embeddings
        
        Args:
            texts: Text string or list of text strings (optional)
            input_ids: Tokenized text input (optional)
            attention_mask: Attention mask for input (optional)
            normalize: Whether to normalize the embeddings
            
        Returns:
            Text embeddings tensor
        """
        # If texts are provided as strings, tokenize them
        if texts is not None:
            if isinstance(texts, str):
                texts = [texts]
            
            # Tokenize texts
            encoded = self.tokenizer(texts, padding=True, truncation=True, 
                                   return_tensors='pt', max_length=77)
            input_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']
        
        # Move to device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
            
        text_embeds = self.text_model(input_ids, attention_mask)
        
        if normalize:
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        return text_embeds
    
        
    
    def encode_image(
        self,
        images: Union[torch.Tensor, List[Image.Image], Image.Image],
        normalize=True
    ) -> torch.Tensor:
        """
        Encode image inputs to embeddings.
        
        Args:
            images: Image tensor, PIL Image, or list of PIL Images
            normalize: Whether to normalize the embeddings
            
        Returns:
            Image embeddings tensor
        """
        if isinstance(images, Image.Image):
            images = [images]
        
        if isinstance(images, list):
            # Process PIL images
            image_tensors = torch.stack([self.preprocess(img) for img in images])
            image_tensors = image_tensors.to(self.device)
        else:
            # Assume tensor input
            image_tensors = images.to(self.device) 
            
    
                   
        
        image_features = self.vision_model(pixel_values=image_tensors)
        if normalize:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features    

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
        
        image_features = self.vision_model(pixel_values=image_tensors)
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
        


        
        image_features = self.vision_model(pixel_values=image_tensors)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features    

    def forward(self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        texts=None,
        return_loss=None,
        **kwargs,
        ):
        # Move tensors to device
        if input_ids is not None:
            input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device)

        img_embeds = self.encode_image(pixel_values)
        
        # Use new encode_text method that can handle both tokenized and string inputs
        if texts is not None:
            text_embeds = self.encode_text(texts=texts)
        else:
            text_embeds = self.encode_text(input_ids=input_ids, attention_mask=attention_mask)

        logits_per_image = self.compute_logits(img_embeds, text_embeds)
        logits_per_text = logits_per_image.t()

        if return_loss:
            loss = self.clip_loss(logits_per_text)
        else:
            loss = None

        return {'img_embeds':img_embeds, 'text_embeds':text_embeds,
            'logits':logits_per_image, 'loss_value':loss, 'logits_per_text':logits_per_text}

    def compute_logits(self, img_emb, text_emb):
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_emb, img_emb.t()) * logit_scale
        return logits_per_text.t()

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.T)
        return (caption_loss + image_loss) / 2.0

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

class PromptClassifier(nn.Module):
    '''take MedCLIP model with prompts for zero-shot classification
    '''
    def __init__(self, medclip_model, ensemble=False, **kwargs) -> None:
        super().__init__()
        self.model = medclip_model
        self.ensemble = ensemble

    def forward(self, pixel_values=None, prompt_inputs=None, **kwargs):
        '''take image pixel values (after transform) and prompt_inputs
        (a dict of {'class1':{'input_ids':...,'attention_mask':,...}), 'class2':...}
        '''
        pixel_values = pixel_values.cuda()
        class_similarities = []
        class_names = []
        for cls_name, cls_text in prompt_inputs.items():
            inputs = {'pixel_values':pixel_values}
            for k in cls_text.keys(): inputs[k] = cls_text[k].cuda()

            # TODO:
            # take soft mask over class_prompts to reach the similarities to classes
            medclip_outputs = self.model(**inputs)
            logits = medclip_outputs['logits']

            # take logits max as the class similarity
            # cls_sim = torch.max(logits, 1)[0] # equivalent use only one prompt
            if self.ensemble:
                cls_sim = torch.mean(logits, 1) # equivalent to prompt ensembling
            else:
                cls_sim = torch.max(logits, 1)[0]
            class_similarities.append(cls_sim)
            class_names.append(cls_name)

        class_similarities = torch.stack(class_similarities, 1)
        outputs = {
            'logits': class_similarities,
            'class_names': class_names,
        }
        return outputs

class SuperviseClassifier(nn.Module):
    '''take MedCLIP model with linear heads for supervised classification on images.
    '''
    def __init__(self,
        vision_model,
        num_class=14,
        input_dim=768,
        mode=None,
        **kwargs) -> None:
        '''args:
        vision_model: the medclip vision model that encodes input images into embeddings.
        num_class: number of classes to predict
        input_dim: the embedding dim before the linear output layer
        mode: multilabel, multiclass, or binary
        '''
        super().__init__()
        self.model = vision_model
        self.num_class = num_class
        assert mode.lower() in ['multiclass','multilabel','binary']
        self.mode = mode.lower()
        if num_class > 2:
            if mode == 'multiclass':
                self.loss_fn = nn.CrossEntropyLoss()
            else:
                self.loss_fn = nn.BCEWithLogitsLoss()

            self.fc = nn.Linear(input_dim, num_class)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()
            self.fc = nn.Linear(input_dim, 1)

    def forward(self,
        pixel_values,
        labels=None,
        return_loss=True,
        **kwargs,
        ):
        outputs = defaultdict()
        pixel_values = pixel_values.cuda()
        # take embeddings before the projection head
        img_embeds = self.model(pixel_values, project=False)
        logits = self.fc(img_embeds)
        outputs['embedding'] = img_embeds
        outputs['logits'] = logits
        if labels is not None and return_loss:
            labels = labels.cuda().float()
            if len(labels.shape) == 1: labels = labels.view(-1,1)
            if self.mode == 'multiclass': labels = labels.flatten().long()
            loss = self.loss_fn(logits, labels)
            outputs['loss_value'] = loss
        return outputs


class PartiallyFixedEmbedding(nn.Module):
    def __init__(self, fixed_weights, num_to_learn):
        super().__init__()
        print(f'{num_to_learn} new tokens added to the embedding layer.')
        self.num_fixed = fixed_weights.size(0)
        self.num_to_learn = num_to_learn
        weight = torch.empty(self.num_fixed+num_to_learn, fixed_weights.size(1))
        weight[:self.num_fixed] = fixed_weights
        self.trainable_weight = nn.Parameter(torch.empty(num_to_learn, fixed_weights.size(1)))
        nn.init.kaiming_uniform_(self.trainable_weight)
        weight[self.num_fixed:] = self.trainable_weight
        self.register_buffer('weight', weight)

    def forward(self, inp):
        self.weight.detach_()
        self.weight[self.num_fixed:] = self.trainable_weight
        return nn.functional.embedding(input=inp,
                                       weight=self.weight,
                                       padding_idx=None,
                                       max_norm=None,
                                       norm_type=2.0,
                                       scale_grad_by_freq=False,
                                       sparse=False)


class PromptTuningClassifier(nn.Module):
    '''take MedCLIP model with prompt tuning
    '''
    def __init__(self, medclip_model, n_context, class_specific_context, num_class, mode, ensemble=True,
                 joint_train_emb=False, **kwargs) -> None:
        super().__init__()
        self.model = medclip_model
        self.ensemble = ensemble
        self.n_context = n_context
        self.class_specific_context = class_specific_context
        self.num_class = num_class
        self.mode = mode
        # calculate number of new context tokens
        if class_specific_context:
            self.n_new_tokens = n_context * num_class
        else:
            self.n_new_tokens = n_context
        # add embeddings for new tokens
        self.prev_n_tokens = self.model.text_model.model.embeddings.word_embeddings.num_embeddings
        self.prev_embeddings = copy.deepcopy(self.model.text_model.model.embeddings.word_embeddings.weight.data)
        if not joint_train_emb:
            self.model.text_model.model.embeddings.word_embeddings = PartiallyFixedEmbedding(
                fixed_weights=self.prev_embeddings,
                num_to_learn=self.n_new_tokens
            )
        else:
            num_old = self.prev_n_tokens
            num_new = self.n_new_tokens
            dim = self.prev_embeddings.shape[1]
            self.model.text_model.model.embeddings.word_embeddings = nn.Embedding(num_old + num_new, dim)
            self.model.text_model.model.embeddings.word_embeddings.weight.data[:num_old] = self.prev_embeddings

        # set loss function
        assert mode.lower() in ['multiclass', 'multilabel', 'binary']
        if mode == 'multilabel':
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()
        return

    def forward(self, pixel_values=None, prompt_inputs=None, labels=None, return_loss=True, **kwargs):
        '''take image pixel values (after transform) and prompt_inputs
        (a dict of {'class1':{'input_ids':...,'attention_mask':,...}), 'class2':...}
        '''
        pixel_values = pixel_values.cuda()
        class_similarities = []
        class_names = []
        for cls_name, cls_text in prompt_inputs.items():
            inputs = {'pixel_values':pixel_values}
            for k in cls_text.keys(): inputs[k] = cls_text[k].cuda()

            # TODO:
            # take soft mask over class_prompts to reach the similarities to classes
            medclip_outputs = self.model(**inputs)
            logits = medclip_outputs['logits']

            # take logits max as the class similarity
            # cls_sim = torch.max(logits, 1)[0] # equivalent use only one prompt
            if self.ensemble:
                cls_sim = torch.mean(logits, 1) # equivalent to prompt ensembling
            else:
                cls_sim = torch.max(logits, 1)[0]
            class_similarities.append(cls_sim)
            class_names.append(cls_name)

        class_similarities = torch.stack(class_similarities, 1)
        outputs = {
            'logits': class_similarities,
            'class_names': class_names,
        }

        if labels is not None and return_loss:
            labels = labels.cuda().float()
            if len(labels.shape) == 1: labels = labels.view(-1,1)
            if self.mode in ['multiclass', 'binary']: labels = labels.flatten().long()
            loss = self.loss_fn(class_similarities, labels)
            outputs['loss_value'] = loss

        return outputs