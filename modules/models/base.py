"""
Base classes for vision and text encoders
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Union, List
from PIL import Image


class TextEncoder(nn.Module, ABC):
    """Abstract base class for text encoders"""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
    
    @abstractmethod
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                return_features: bool = False) -> torch.Tensor:
        """
        Encode text inputs
        
        Args:
            input_ids: Tokenized text input
            attention_mask: Attention mask for input
            return_features: Whether to return raw features or normalized embeddings
            
        Returns:
            Text embeddings
        """
        pass
    
    @abstractmethod
    def get_feature_dim(self) -> int:
        """Get the feature dimension of the encoder"""
        pass
    
    def load_pretrained(self, model_path: str):
        """Load pretrained model weights - default implementation"""
        state_dict = torch.load(model_path, map_location='cpu')
        self.load_state_dict(state_dict, strict=False)


class VisionEncoder(nn.Module, ABC):
    """Abstract base class for vision encoders"""
    
    def __init__(self, feature_dim: int = 768):
        super().__init__()
        self.feature_dim = feature_dim
    
    @abstractmethod
    def forward(self, images: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Encode image inputs
        
        Args:
            images: Input image tensor
            return_features: Whether to return raw features or normalized embeddings
            
        Returns:
            Image embeddings or logits
        """
        pass
    
    @abstractmethod
    def get_feature_dim(self) -> int:
        """Get the feature dimension of the encoder"""
        pass
    
    def load_pretrained(self, model_path: str):
        """Load pretrained model weights - default implementation"""
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        self.load_state_dict(state_dict, strict=False)
        print("✅ Pretrained weights loaded successfully")
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get normalized feature embeddings"""
        return self.forward(x, return_features=True)


class VisionLanguageModel(nn.Module, ABC):
    """Abstract base class for vision-language models"""
    
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @abstractmethod
    def encode_text(self, texts: Union[str, List[str]], normalize: bool = True) -> torch.Tensor:
        """
        Encode text inputs to embeddings
        
        Args:
            texts: Text string or list of text strings
            normalize: Whether to normalize the embeddings
            
        Returns:
            Text embeddings tensor
        """
        pass
    
    @abstractmethod
    def encode_image(self, images: Union[torch.Tensor, List[Image.Image], Image.Image], 
                    normalize: bool = True) -> torch.Tensor:
        """
        Encode image inputs to embeddings
        
        Args:
            images: Image tensor, PIL Image, or list of PIL Images
            normalize: Whether to normalize the embeddings
            
        Returns:
            Image embeddings tensor
        """
        pass
    
    @abstractmethod
    def forward(self, **kwargs):
        """
        Forward pass of the model
        
        Args:
            **kwargs: Model-specific arguments
            
        Returns:
            Model outputs dictionary
        """
        pass
    
    def load_checkpoint(self, checkpoint_path: str, strict: bool = False):
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            strict: Whether to strictly enforce key matching
        """
        if not checkpoint_path:
            print("No checkpoint path provided")
            return
            
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=strict)
            
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
                
            print(f"✅ Checkpoint loaded from: {checkpoint_path}")
            
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
    
    def to_device(self, device: Optional[torch.device] = None):
        """Move model to specified device"""
        if device is None:
            device = self.device
        self.device = device
        return super().to(device)
    
    def get_model_info(self) -> dict:
        """
        Get information about the model
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_type': self.__class__.__name__,
            'device': str(self.device),
            'parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
