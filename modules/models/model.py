"""
Base classes for medical vision-language models
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional, Any
import torch
import torch.nn as nn
from PIL import Image


class BaseVisionLanguageModel(nn.Module, ABC):
    """
    Abstract base class for all vision-language models
    Provides unified interface for MedCLIP, BioMedCLIP, and future models
    """
    
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        checkpoint: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize base vision-language model
        
        Args:
            model_name: Name of the model ('medclip', 'biomedclip', etc.)
            device: Device to load model on ('cuda' or 'cpu')
            checkpoint: Optional checkpoint path to load model weights
        """
        super().__init__()
        self.model_name = model_name
        
        # Set device
        if device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = torch.device(device)
    
    @abstractmethod
    def encode_text(
        self,
        texts: Union[str, List[str], torch.Tensor],
        normalize: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        Encode text inputs to embeddings
        
        Args:
            texts: Text inputs (strings, list of strings, or tokenized tensors)
            normalize: Whether to normalize the embeddings
            
        Returns:
            Text embeddings tensor
        """
        pass
    
    @abstractmethod
    def encode_image(
        self,
        images: Union[torch.Tensor, List[Image.Image], Image.Image],
        normalize: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        Encode image inputs to embeddings
        
        Args:
            images: Image inputs (tensors or PIL images)
            normalize: Whether to normalize the embeddings
            
        Returns:
            Image embeddings tensor
        """
        pass
    
    @abstractmethod
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass for the model
        
        Args:
            pixel_values: Preprocessed image tensors
            input_ids: Tokenized text input ids
            attention_mask: Attention mask for text inputs
            return_loss: Whether to compute and return contrastive loss
            
        Returns:
            Dictionary containing embeddings, logits, and optionally loss
        """
        pass
    
    def compute_similarity(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute similarity between image and text features
        
        Args:
            image_features: Normalized image embeddings
            text_features: Normalized text embeddings
            logit_scale: Optional temperature scaling factor
            
        Returns:
            Similarity matrix
        """
        if logit_scale is not None:
            return logit_scale * image_features @ text_features.t()
        return image_features @ text_features.t()
    
    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        """
        Compute CLIP-style contrastive loss
        
        Args:
            similarity: Similarity matrix between images and texts
            
        Returns:
            Contrastive loss value
        """
        caption_loss = nn.functional.cross_entropy(
            similarity, torch.arange(len(similarity), device=similarity.device)
        )
        image_loss = nn.functional.cross_entropy(
            similarity.t(), torch.arange(len(similarity), device=similarity.device)
        )
        return (caption_loss + image_loss) / 2.0
    
    def save_pretrained(self, save_directory: str):
        """
        Save model weights to directory
        
        Args:
            save_directory: Directory to save model weights
        """
        os.makedirs(save_directory, exist_ok=True)
        save_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    
    def load_pretrained(self, checkpoint_path: str, strict: bool = True):
        """
        Load model weights from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file or directory
            strict: Whether to strictly enforce that the keys match
        """
        if os.path.isdir(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_path, 'pytorch_model.bin')
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=strict)
        
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        
        print(f"Loaded model weights from {checkpoint_path}")
        return missing_keys, unexpected_keys


class BaseClassifier(nn.Module, ABC):
    """
    Abstract base class for classifiers built on vision-language models
    """
    
    def __init__(
        self,
        base_model: BaseVisionLanguageModel,
        num_classes: Optional[int] = None,
        mode: str = 'zeroshot',
        **kwargs
    ):
        """
        Initialize classifier
        
        Args:
            base_model: Base vision-language model
            num_classes: Number of classes (for supervised mode)
            mode: 'zeroshot' or 'supervised'
        """
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.mode = mode
        
        if mode not in ['zeroshot', 'supervised']:
            raise ValueError(f"Mode must be 'zeroshot' or 'supervised', got {mode}")
    
    @abstractmethod
    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass for classification
        
        Args:
            pixel_values: Image tensors
            labels: Ground truth labels (for supervised mode)
            
        Returns:
            Dictionary containing logits and optionally loss
        """
        pass
    
    def get_device(self) -> torch.device:
        """Get the device of the model"""
        return self.base_model.device


class BaseZeroShotClassifier(BaseClassifier):
    """
    Base class for zero-shot classifiers
    """
    
    def __init__(
        self,
        base_model: BaseVisionLanguageModel,
        class_names: List[str],
        templates: Optional[List[str]] = None,
        ensemble: bool = False,
        **kwargs
    ):
        """
        Initialize zero-shot classifier
        
        Args:
            base_model: Base vision-language model
            class_names: List of class names
            templates: Text templates for prompts
            ensemble: Whether to use prompt ensembling
        """
        super().__init__(base_model, num_classes=len(class_names), mode='zeroshot')
        self.class_names = class_names
        self.templates = templates if templates else ["a photo of {}"]
        self.ensemble = ensemble
    
    def create_text_prompts(
        self,
        class_names: Optional[List[str]] = None
    ) -> List[str]:
        """
        Create text prompts for classes
        
        Args:
            class_names: Optional list of class names to override defaults
            
        Returns:
            List of text prompts
        """
        if class_names is None:
            class_names = self.class_names
        
        prompts = []
        for class_name in class_names:
            for template in self.templates:
                prompts.append(template.format(class_name))
        
        return prompts
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        prompt_inputs: Optional[Dict[str, Any]] = None,
        class_names: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass for zero-shot classification
        
        Args:
            pixel_values: Image tensors
            prompt_inputs: Optional pre-computed prompt inputs
            class_names: Optional class names to override defaults
            
        Returns:
            Dictionary containing logits and class names
        """
        if class_names is None:
            class_names = self.class_names
        
        # Encode images
        image_features = self.base_model.encode_image(pixel_values, normalize=True)
        
        # Create and encode text prompts if not provided
        if prompt_inputs is None:
            prompts = self.create_text_prompts(class_names)
            text_features = self.base_model.encode_text(prompts, normalize=True)
        else:
            # Use provided prompt inputs
            text_features = self._process_prompt_inputs(prompt_inputs)
        
        # Compute similarities
        similarities = self.base_model.compute_similarity(image_features, text_features)
        
        # Aggregate similarities per class
        if self.ensemble and len(self.templates) > 1:
            # Reshape to (batch_size, num_classes, num_templates)
            similarities = similarities.view(
                similarities.size(0), 
                len(class_names), 
                len(self.templates)
            )
            # Average over templates
            logits = similarities.mean(dim=-1)
        else:
            logits = similarities
        
        return {
            'logits': logits,
            'class_names': class_names,
            'image_features': image_features,
            'text_features': text_features
        }
    
    def _process_prompt_inputs(
        self,
        prompt_inputs: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Process pre-computed prompt inputs
        
        Args:
            prompt_inputs: Dictionary of prompt inputs per class
            
        Returns:
            Text features tensor
        """
        text_features_list = []
        
        for class_name, class_prompts in prompt_inputs.items():
            if isinstance(class_prompts, dict):
                # Tokenized inputs
                outputs = self.base_model.forward(
                    input_ids=class_prompts.get('input_ids'),
                    attention_mask=class_prompts.get('attention_mask')
                )
                text_features = outputs['text_embeds']
            else:
                # Raw text
                text_features = self.base_model.encode_text(class_prompts, normalize=True)
            
            text_features_list.append(text_features)
        
        return torch.cat(text_features_list, dim=0)


class BaseSupervisedClassifier(BaseClassifier):
    """
    Base class for supervised classifiers with learnable classification head
    """
    
    def __init__(
        self,
        base_model: BaseVisionLanguageModel,
        num_classes: int,
        feature_dim: Optional[int] = None,
        freeze_encoder: bool = True,
        dropout: float = 0.5,
        task_type: str = 'multiclass',
        **kwargs
    ):
        """
        Initialize supervised classifier
        
        Args:
            base_model: Base vision-language model
            num_classes: Number of output classes
            feature_dim: Feature dimension (will be auto-detected if None)
            freeze_encoder: Whether to freeze encoder weights
            dropout: Dropout rate
            task_type: 'binary', 'multiclass', or 'multilabel'
        """
        super().__init__(base_model, num_classes=num_classes, mode='supervised')
        
        self.task_type = task_type
        self.freeze_encoder = freeze_encoder
        
        # Validate task type
        if task_type not in ['binary', 'multiclass', 'multilabel']:
            raise ValueError(f"task_type must be 'binary', 'multiclass', or 'multilabel', got {task_type}")
        
        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Auto-detect feature dimension if not provided
        if feature_dim is None:
            with torch.no_grad():
                dummy_image = torch.randn(1, 3, 224, 224).to(self.base_model.device)
                features = self.base_model.encode_image(dummy_image, normalize=False)
                feature_dim = features.shape[-1]
        
        # Create classification head
        if task_type == 'binary':
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feature_dim, 1)
            )
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feature_dim, num_classes)
            )
            if task_type == 'multiclass':
                self.loss_fn = nn.CrossEntropyLoss()
            else:  # multilabel
                self.loss_fn = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_loss: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass for supervised classification
        
        Args:
            pixel_values: Image tensors
            labels: Ground truth labels
            return_loss: Whether to compute and return loss
            
        Returns:
            Dictionary containing logits, embeddings, and optionally loss
        """
        # Extract features
        with torch.no_grad() if self.freeze_encoder else torch.enable_grad():
            image_features = self.base_model.encode_image(pixel_values, normalize=False)
        
        # Classification
        logits = self.classifier(image_features)
        
        outputs = {
            'logits': logits,
            'embeddings': image_features
        }
        
        # Compute loss if labels provided
        if labels is not None and return_loss:
            labels = labels.to(self.base_model.device)
            
            if self.task_type == 'binary':
                labels = labels.float().view(-1, 1)
            elif self.task_type == 'multiclass':
                labels = labels.long()
            else:  # multilabel
                labels = labels.float()
            
            loss = self.loss_fn(logits, labels)
            outputs['loss_value'] = loss
        
        return outputs
    
    def unfreeze_encoder(self, num_layers: Optional[int] = None):
        """
        Unfreeze encoder layers for fine-tuning
        
        Args:
            num_layers: Number of layers to unfreeze (from the end). If None, unfreeze all.
        """
        if num_layers is None:
            # Unfreeze all
            for param in self.base_model.parameters():
                param.requires_grad = True
            print("Unfroze all encoder layers")
        else:
            # Unfreeze last n layers
            # This is model-specific and would need to be implemented per model
            print(f"Unfreezing last {num_layers} layers - implement model-specific logic")


class BasePromptLearner(nn.Module):
    """
    Base class for prompt learning methods (CoOp, CoCoOp, etc.)
    """
    
    def __init__(
        self,
        base_model: BaseVisionLanguageModel,
        n_context: int = 4,
        class_specific: bool = False,
        class_names: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize prompt learner
        
        Args:
            base_model: Base vision-language model
            n_context: Number of learnable context tokens
            class_specific: Whether to learn class-specific prompts
            class_names: List of class names
        """
        super().__init__()
        self.base_model = base_model
        self.n_context = n_context
        self.class_specific = class_specific
        self.class_names = class_names if class_names else []
        
        # Initialize learnable prompts
        self._init_prompts()
    
    @abstractmethod
    def _init_prompts(self):
        """Initialize learnable prompt embeddings"""
        pass
    
    @abstractmethod
    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Forward pass with learnable prompts"""
        pass
