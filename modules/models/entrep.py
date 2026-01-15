import torch
import torch.nn as nn
import torch.nn.functional as F
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


__all__ = [
    'CLIPTextEncoder',
    'DinoV2Model',
    'EntVitModel',
    'ENTRepModel',
]

# For logging
from ..utils.logging_config import get_logger
logger = get_logger(__name__)

class CLIPTextEncoder(TextEncoder):
    """CLIP Text Encoder"""
    
    def __init__(self, model_name: str = "medicalai/ClinicalBERT", 
                 feature_dim: int = 768, dropout_rate: float = 0.3, ckp_path: Optional[str] = None,
                 pretrained: bool = True):
        super().__init__(feature_dim)
        self.feature_dim = feature_dim
        
        if pretrained:
            try:
                self.text_model = AutoModelForMaskedLM.from_pretrained("medicalai/ClinicalBERT", 
                                             use_safetensors=False, 
                                            #  local_files_only=True,
                                            #  trust_remote_code=True
                                             )
            except Exception as e:
                # Fallback if loading fails
                logger.warning(f"Failed to load ClinicalBERT with safetensors=False: {e}")
                logger.info("Trying alternative loading method...")
                self.text_model = AutoModelForMaskedLM.from_pretrained("medicalai/ClinicalBERT")
        else:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained("medicalai/ClinicalBERT")
            self.text_model = AutoModelForMaskedLM.from_config(config)
            
        # Layer normalization and dropout
        self.ln = nn.LayerNorm(self.get_feature_dim())
        self.dropout = nn.Dropout(dropout_rate)
        
        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(self.get_feature_dim(), self.feature_dim),
            nn.LayerNorm(self.feature_dim)
        )

        if ckp_path is not None:
            self.load_pretrained(ckp_path)

    def load_pretrained(self, model_path: str):
        """Load pretrained model from path"""
        self.text_model.load_state_dict(torch.load(model_path))

    def get_feature_dim(self) -> int:
        return self.text_model.config.hidden_size
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        text_outputs = self.text_model(
        input_ids=input_ids, 
        attention_mask=attention_mask,
        output_hidden_states=True  # Get hidden states
    )
    
        # Get the last hidden layer
        hidden_states = text_outputs.hidden_states[-1]  # Shape: [batch, seq_len, hidden_dim]
        
        # Pooling: take [CLS] token (first token) or mean pooling
        embeddings = hidden_states[:, 0, :]
        embeddings = self.ln(embeddings)
        embeddings = self.dropout(embeddings)
        embeddings = self.projection(embeddings)
        text_features = embeddings / embeddings.norm(dim=-1, keepdim=True)
        if return_features:
            return embeddings
        return text_features

class DinoV2Backbone(nn.Module):
    """DinoV2 backbone for feature extraction"""
    
    def __init__(self, model_name: str = 'dinov2_vitb14'):
        super().__init__()
        
        # Load pre-trained DinoV2 model
        try:
            print(f"📥 Loading pretrained {model_name} from torch.hub...")
            self.backbone = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
            print(f"✅ Successfully loaded pretrained {model_name}")
        except Exception as e:
            print(f"⚠️ Could not load pretrained {model_name}: {e}")
            print("💡 Falling back to torchvision ViT with ImageNet pretrained weights")
            # Fallback to ViT with pretrained weights
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            self.backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            self.backbone.heads = nn.Identity()
            print("✅ Using fallback ViT model with pretrained weights")
        
        # Get output dimension
        self.feature_dim = self._get_feature_dim()
        
    def _get_feature_dim(self) -> int:
        """Get feature dimension"""
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = self.backbone(dummy_input)
        return features.shape[-1]
    
    def forward(self, x):
        """Forward pass"""
        return self.backbone(x)
class DinoV2Head(nn.Module):
    """Classification/Feature head for DinoV2"""
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int = 768,
                 num_classes: int = 1000,
                 dropout: float = 0.1):
        super().__init__()
        
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x, return_features=False):
        """Forward pass"""
        features = self.feature_projection(x)
        
        if return_features:
            return features
        
        logits = self.classifier(features)
        return logits
class DinoV2Core(nn.Module):
    """Complete DinoV2 model for image retrieval"""
    
    def __init__(self, 
                 model_name: str = 'dinov2_vitb14',
                 feature_dim: int = 768,
                 num_classes: int = 1000,
                 dropout: float = 0.1,
                 freeze_backbone: bool = False):
        super().__init__()
        
        # Initialize backbone
        self.backbone = DinoV2Backbone(model_name)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("🔒 Frozen backbone parameters")
        
        # Initialize head
        self.head = DinoV2Head(
            input_dim=self.backbone.feature_dim,
            hidden_dim=feature_dim,
            num_classes=num_classes,
            dropout=dropout
        )
        
        # Store config
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
    def forward(self, x, return_features=False):
        """Forward pass"""
        # Extract backbone features
        backbone_features = self.backbone(x)
        
        # Pass through head
        output = self.head(backbone_features, return_features=return_features)
        
        return output
    
    def get_features(self, x):
        """Get feature embeddings"""
        return self.forward(x, return_features=True)
    
    def get_backbone_features(self, x):
        """Get raw backbone features"""
        return self.backbone(x)
class DinoV2Model(nn.Module):
    """DinoV2 Model for Image Retrieval - Wrapper around core DinoV2Model"""
    
    def __init__(self, 
                 model_name: str = 'dinov2_vitb14',
                 feature_dim: int = 768,
                 num_classes: int = 1000,
                 dropout: float = 0.1,
                 freeze_backbone: bool = False):
        super().__init__()
        
        # Use the core DinoV2Model from models module
        self.model = DinoV2Core(
            model_name=model_name,
            feature_dim=feature_dim,
            num_classes=num_classes,
            dropout=dropout,
            freeze_backbone=freeze_backbone
        )
        
        # Store dimensions for compatibility
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.model_name = model_name
        
    def forward(self, x, return_features=False):
        """Forward pass"""
        return self.model.forward(x, return_features=return_features)
    
    def get_features(self, x):
        """Get feature embeddings"""
        return self.model.get_features(x)

class EntVitBackbone(nn.Module):
    """EntVit backbone for feature extraction"""
    
    def __init__(self, model_name: str = 'egeozsoy/EndoViT'):
        super().__init__()
        
        # First try to load EndoViT from HuggingFace
        try:
            print("📥 Loading pretrained EndoViT from HuggingFace...")
            from huggingface_hub import snapshot_download
            from timm.models.vision_transformer import VisionTransformer
            from functools import partial
            
            # Download model files
            model_path = snapshot_download(repo_id=model_name, revision="main")
            model_weights_path = Path(model_path) / "pytorch_model.bin"
            
            if model_weights_path.exists():
                # Define the EndoViT model architecture (based on ViT-B/16)
                self.backbone = VisionTransformer(
                    patch_size=16, 
                    embed_dim=768, 
                    depth=12, 
                    num_heads=12, 
                    mlp_ratio=4, 
                    qkv_bias=True, 
                    norm_layer=partial(nn.LayerNorm, eps=1e-6)
                ).eval()
                
                # Load the pretrained weights
                model_weights = torch.load(model_weights_path, map_location='cpu', weights_only=False)
                if 'model' in model_weights:
                    model_weights = model_weights['model']
                    
                loading_info = self.backbone.load_state_dict(model_weights, strict=False)
                self.feature_dim = 768
                print(f"✅ Successfully loaded pretrained EndoViT: {loading_info}")
            else:
                raise FileNotFoundError("EndoViT weights not found")
                
        except Exception as e:
            print(f"⚠️ Could not load EndoViT from HuggingFace: {e}")
            print("💡 Falling back to standard ViT with ImageNet pretrained weights")
            # Fallback to standard pretrained ViT
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            self.backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            self.backbone.heads = nn.Identity()
            self.feature_dim = 768
            print("✅ Using fallback ViT model with pretrained weights")
        
    def forward(self, x):
        """Forward pass"""
        try:
            # For EndoViT (timm ViT) - use forward_features method
            if hasattr(self.backbone, 'forward_features'):
                features = self.backbone.forward_features(x)
                # Take the CLS token (first token) from the sequence
                if len(features.shape) == 3:  # [batch, seq_len, embed_dim]
                    return features[:, 0]  # CLS token
                else:
                    return features
            # For torchvision ViT
            else:
                return self.backbone(x)
        except Exception as e:
            print(f"⚠️ Error in EntVit forward pass: {e}")
            # Fallback to regular forward
            return self.backbone(x)
class EntVitHead(nn.Module):
    """Classification/Feature head for EntVit"""
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int = 768,
                 num_classes: int = 1000,
                 dropout: float = 0.1):
        super().__init__()
        
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x, return_features=False):
        """Forward pass"""
        features = self.feature_projection(x)
        
        if return_features:
            return features
        
        logits = self.classifier(features)
        return logits
class EntVitCore(nn.Module):
    """Complete EntVit model for image retrieval"""
    
    def __init__(self, 
                 model_name: str = 'egeozsoy/EndoViT',
                 feature_dim: int = 768,
                 num_classes: int = 1000,
                 dropout: float = 0.1,
                 freeze_backbone: bool = False):
        super().__init__()
        
        # Initialize backbone with EndoViT
        self.backbone = EntVitBackbone(model_name)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("🔒 Frozen backbone parameters")
        
        # Initialize head
        self.head = EntVitHead(
            input_dim=self.backbone.feature_dim,
            hidden_dim=feature_dim,
            num_classes=num_classes,
            dropout=dropout
        )
        
        # Store config
        self.feature_dim = feature_dim
        self.num_classes = num_classes
    
    def forward(self, x, return_features=False):
        """Forward pass"""
        # Extract backbone features
        backbone_features = self.backbone(x)
        
        # Pass through head
        output = self.head(backbone_features, return_features=return_features)
        
        return output
    
    def get_features(self, x):
        """Get feature embeddings"""
        return self.forward(x, return_features=True)
    
    def get_backbone_features(self, x):
        """Get raw backbone features"""
        return self.backbone(x)
class EntVitModel(nn.Module):
    """EndoViT Model for Image Retrieval - Wrapper around core EntVitModel"""
    
    def __init__(self, 
                 model_name = 'egeozsoy/EndoViT',
                 feature_dim: int = 768,
                 num_classes: int = 1000,
                 dropout: float = 0.1,
                 freeze_backbone: bool = False):
        super().__init__()
        
        # Use the core EntVitModel from models module
        self.model = EntVitCore(
            model_name='egeozsoy/EndoViT',  # Use actual EndoViT from HuggingFace
            feature_dim=feature_dim,
            num_classes=num_classes,
            dropout=dropout,
            freeze_backbone=freeze_backbone
        )
        
        # Store dimensions for compatibility
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
    def forward(self, x, return_features=False):
        """Forward pass"""
        return self.model.forward(x, return_features=return_features)
    
    def get_features(self, x):
        """Get feature embeddings"""
        return self.model.get_features(x)

class ENTRepModel(nn.Module):
    """
    ENTRep Vision-Language Model
    - Text encoder: CLIPTextEncoder (optional)
    - Vision encoder: DinoV2Model or EntVitModel wrapper to load checkpoint correctly
    """
    
    def __init__(self,
                 vision_encoder_type: str = 'dinov2',
                 text_encoder_type: str = 'clip',
                 text_checkpoint: Optional[str] = None,
                 model_name: str = 'dinov2_vitb14',
                 feature_dim: int = 768,
                 num_classes: int = 7,
                 dropout: float = 0.1,
                 dropout_rate: float = 0.3,
                 freeze_backbone: bool = False,
                 vision_checkpoint: Optional[str] = None,
                 checkpoint: Optional[str] = None,
                 logit_scale_init_value: float = 0.07,
                 pretrained: bool = True,
                 mode_pretrained: str = "scratch"
        ):
        """
        Initialize ENTRep Model
        
        Args:
            vision_encoder_type: 'dinov2' or 'endovit'
            text_encoder_type: 'clip' or 'none'
            text_checkpoint: Path to text encoder checkpoint (riêng)
            model_name: Model variant name
            feature_dim: Feature dimension
            num_classes: Number of classes
            dropout: Dropout rate for vision encoder
            dropout_rate: Dropout rate for text encoder
            freeze_backbone: Whether to freeze backbone
            vision_checkpoint: Path to vision encoder checkpoint (riêng)
            checkpoint: Path to full ENTRep checkpoint (toàn bộ model - ưu tiên cao nhất)
            logit_scale_init_value: Initial value for logit scale
            pretrained: Whether to use pretrained weights
            
        Note:
            Checkpoint priority:
            1. checkpoint: Load full (vision + text + logit_scale)
                2. vision_checkpoint + text_checkpoint: load separately
                3. No checkpoint: Init from pretrained or random
        """
        super().__init__()
        
        self.vision_encoder_type = vision_encoder_type
        self.text_encoder_type = text_encoder_type
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.preprocess = constants.MODEL_TRANSFORMS['entrep']
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.normalize_transform = constants.TENSOR_NORMALIZE_TRANSFORM['entrep']
        self.mode_pretrained = mode_pretrained
        # Create text encoder (optional)
        # Không load text_checkpoint nếu có checkpoint (sẽ load sau)
        if text_encoder_type == 'clip':
            logger.info(f"🏗️ Creating CLIP text encoder...")
            self.text_model = CLIPTextEncoder(
                feature_dim=feature_dim,
                dropout_rate=dropout_rate,
                ckp_path=text_checkpoint if not checkpoint else None,
                pretrained=pretrained
            )
        elif text_encoder_type is None or text_encoder_type == 'none':
            self.text_model = None
            logger.info("⚠️ No text encoder (vision-only mode)")
        else:
            raise ValueError(f"Unknown text encoder type: {text_encoder_type}")
        
        if vision_encoder_type == 'dinov2':
            logger.info(f"🏗️ Creating DinoV2Model wrapper...")
            self.vision_model = DinoV2Model(
                model_name=model_name,
                feature_dim=feature_dim,
                num_classes=num_classes,
                dropout=dropout,
                freeze_backbone=freeze_backbone
            )
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/logit_scale_init_value)))
        
      
        ckp = self.download_checkpoint()
        print(ckp)
        self._load_full_checkpoint(ckp)
            
            
    def download_checkpoint(self):
        try:
            repo_id = "Woffy/Thesis_Pretrained_Medical_Moddel"
            if self.mode_pretrained == "scratch":
                file_name = "entrep.pt"
            elif self.mode_pretrained == "ssl":
                file_name = "entrep_ssl_finetuning.pt"
            elif self.mode_pretrained == "at":
                file_name = "entrep_AT.pth"
            
            print(file_name)
            local_path = hf_hub_download(
                    repo_id=repo_id,  # sửa repo của bạn
                    filename=file_name,
                    local_dir=".",                 # tải đúng vào thư mục bạn muốn
                    local_dir_use_symlinks=False  # QUAN TRỌNG: copy file thật, không tạo symlink
                )   
            return local_path
        except Exception as e:
            logger.error(f"Failed to download ENTREP checkpoint: {e}")
            return None
    def _load_full_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint cho toàn bộ ENTRep model (vision + text + logit_scale)
        Tự động filter ra classifier keys nếu num_classes khác để tránh size mismatch
        """
        logger.info(f"📥 Loading full ENTRep checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Filter ra classifier keys để tránh size mismatch khi num_classes khác
        filtered_state_dict = {}
        skipped_keys = []
        
        # Get current model state để check size compatibility
        current_state = self.state_dict()
        
        for key, value in state_dict.items():
            # Skip nếu là classifier key
            if 'classifier' in key:
                skipped_keys.append(key)
                continue
            
            # Skip nếu key có size mismatch
            if key in current_state:
                if value.shape != current_state[key].shape:
                    logger.warning(f"   ⚠️ Size mismatch for {key}: {value.shape} vs {current_state[key].shape}, skipping")
                    skipped_keys.append(key)
                    continue
            
            filtered_state_dict[key] = value
        
        logger.info(f"   Loading {len(filtered_state_dict)} keys, skipping {len(skipped_keys)} keys")
        
        # Load với strict=False
        missing_keys, unexpected_keys = self.load_state_dict(filtered_state_dict, strict=False)
        
        if missing_keys:
            logger.info(f"✅ Missing keys (expected - classifier + new params): {len(missing_keys)}")
            logger.debug(f"Missing: {missing_keys[:10]}")  # First 10
        
        if unexpected_keys:
            logger.warning(f"⚠️ Unexpected keys: {len(unexpected_keys)}")
            logger.debug(f"Unexpected: {unexpected_keys[:10]}")
        
        if not missing_keys and not unexpected_keys:
            logger.info("✅ Full ENTRep checkpoint loaded perfectly!")
        else:
            logger.info("✅ Full ENTRep checkpoint loaded (backbone + feature_projection loaded, classifier re-initialized)")
        
        # Log thông tin từ checkpoint nếu có
        if 'epoch' in checkpoint:
            logger.info(f"   📊 Checkpoint epoch: {checkpoint['epoch']}")
        if 'best_metric' in checkpoint:
            logger.info(f"   📊 Best metric: {checkpoint['best_metric']}")
        if 'optimizer_state_dict' in checkpoint:
            logger.info(f"   ✅ Optimizer state available (for resume training)")
    
    # def encode_text(self, input_ids=None, attention_mask=None):
    #     """Encode text inputs"""
    #     if self.text_model is None:
    #         raise NotImplementedError("Text encoding not supported (text_model is None)")
        
    #     # Get text embeddings
    #     text_embeds = self.text_model(input_ids, attention_mask, return_features=False)
    #     # Already normalized in CLIPTextEncoder
    #     return text_embeds
    
    # def encode_image(self, pixel_values):
    #     """Encode image inputs"""
    #     # Get image features from wrapper model
    #     features = self.vision_model.get_features(pixel_values)
    #     # Normalize
    #     img_embeds = F.normalize(features, dim=-1)
    #     return img_embeds
    
    def encode_text(
        self,
        texts: str,
        normalize: bool = True
    ):
    
        text_inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        )
        text_inputs = {k: v.cuda() for k, v in text_inputs.items()}
        text_features = self.text_model(
            text_inputs['input_ids'], 
            text_inputs['attention_mask'],
            return_features=False
        )
        
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def encode_text_from_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        normalize: bool = True
    ):
        """
        Encode text from pre-tokenized input_ids
        Used for training with collated batches
        """
        text_features = self.text_model(
            input_ids, 
            attention_mask,
            return_features=False
        )
        
        if normalize:
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
        
    def encode_image(
        self,
        images,
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
            image_tensors = image_tensors.cuda()
            # Assume tensor input
        else:
            image_tensors = images.cuda()
            
        image_features = self.vision_model.get_features(image_tensors)
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
        
        image_features = self.vision_model.get_features(image_tensors)
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
        images_ = torch.round(images * 255.0).clamp(0, 255)
        # Resize to model input size
        image_tensors = F.interpolate(images_, size=(224, 224), mode="bilinear", align_corners=False)
        image_tensors = image_tensors / 255.0
        image_tensors = self.normalize_transform(image_tensors)
        
        image_features = self.vision_model.get_features(image_tensors)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features   
    
    
    def forward(self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
                return_loss=False,
                **kwargs):
        """
        Forward pass for VLM
        
        Args:
            input_ids: Text input ids
            pixel_values: Image tensors
            attention_mask: Text attention mask
            return_loss: Whether to compute contrastive loss
            
        Returns:
            Dictionary with embeddings, logits, and optionally loss
        """
        # Encode images
        img_embeds = self.encode_image(pixel_values)
        
        # Encode text if available
        if self.text_model is not None and input_ids is not None:
            # Use encode_text_from_tokens for pre-tokenized inputs
            text_embeds = self.encode_text_from_tokens(input_ids, attention_mask)
        else:
            text_embeds = None

        # Compute logits if both modalities present
        if text_embeds is not None:
            logits_per_image = self.compute_logits(img_embeds, text_embeds)
            logits_per_text = logits_per_image.t()
        else:
            logits_per_image = None
            logits_per_text = None

        # Compute loss
        if return_loss and logits_per_text is not None:
            loss = self.clip_loss(logits_per_text)
        else:
            loss = None

        return {
            'img_embeds': img_embeds,
            'text_embeds': text_embeds,
            'logits': logits_per_image,
            'logits_per_text': logits_per_text,
            'loss_value': loss
        }

    def compute_logits(self, img_emb, text_emb):
        """Compute logits with temperature scaling"""
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_emb, img_emb.t()) * logit_scale
        return logits_per_text.t()

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        """CLIP contrastive loss"""
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.T)
        return (caption_loss + image_loss) / 2.0

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Contrastive loss function"""
        return nn.functional.cross_entropy(
            logits, 
            torch.arange(len(logits), device=logits.device)
        )
    
    def get_features(self, x):
        """Get vision feature embeddings (for vision-only usage)"""
        return self.vision_model.get_features(x)
    
    def get_encoder_info(self):
        """Get encoder information"""
        return {
            'text_encoder': self.text_encoder_type if self.text_model else 'none',
            'vision_encoder': self.vision_encoder_type,
            'vision_model_type': type(self.vision_model).__name__,
            'text_model_type': type(self.text_model).__name__ if self.text_model else 'None',
            'feature_dim': self.feature_dim,
            'num_classes': self.num_classes
        }