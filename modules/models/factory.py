"""
Complete testing
Factory for creating vision-language models and classifiers
"""

from typing import Dict, List, Optional, Union, Any
import torch
import torch.nn as nn

# Import base classes
from .model import (
    BaseVisionLanguageModel,
    BaseClassifier,
    BaseZeroShotClassifier,
    BaseSupervisedClassifier,
    BasePromptLearner
)

# Import concrete implementations
from .medclip import (
    MedCLIPModel,
    PromptClassifier,
    SuperviseClassifier,
    PromptTuningClassifier
)
from .biomedclip import (
    BioMedCLIPModel,
    BioMedCLIPClassifier
)
from .entrep import (
    ENTRepModel
)

from .vit import (
    ViTModel
)

from .robustmedclip import (
    RMedCLIP
)


# Import constants
from ..utils.constants import SUPPORTED_MODELS, DEFAULT_TEMPLATES
from ..utils import logging_config

logger = logging_config.get_logger(__name__)

class ModelFactory:
    """
    Factory class for creating vision-language models and classifiers
    """
    
    # Registry of base models
    MODEL_REGISTRY = {
        'medclip': {
            'base': MedCLIPModel,
        },
        'biomedclip': {
            'base': BioMedCLIPModel,
        },
        'entrep': {
            'base': ENTRepModel,
        },
        'ViT': {
            'base': ViTModel,
        },
        'rmedclip': {
            'base': RMedCLIP
        }
    }
    
    # Registry of classifiers
    CLASSIFIER_REGISTRY = {
        'medclip': {
            'zeroshot': PromptClassifier
        },
        'biomedclip': {
            'zeroshot': BioMedCLIPClassifier
        },
        'entrep': {
            'zeroshot': ENTRepModel
        }
    }
    
    # Default configurations
    DEFAULT_CONFIGS = {
        'medclip': {
            'text_encoder_type': 'bert',
            'vision_encoder_type': 'vit',  # 'resnet' or 'vit'
            'logit_scale_init_value': 0.07,
            'checkpoint': None,
        },
        'biomedclip': {
            'model_name': 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
            'context_length': 256,
            'checkpoint': None,
        },
        'entrep': {
            'text_encoder_type': 'clip',
            'vision_encoder_type': 'dinov2',
            'feature_dim': 768,
            'dropout': 0.1,
            'num_classes': 7,
            'freeze_backbone': False,
            'vision_checkpoint': None,
            'text_checkpoint': None,
            'logit_scale_init_value': 0.07,
        }
    }
    
    @classmethod
    def _validate_model_type(cls, model_type: str, variant: str):
        """
        Validate model type and variant
        
        Args:
            model_type: Type of model
            variant: Model variant
            
        Raises:
            ValueError: If model type or variant is invalid
        """
        if model_type not in cls.MODEL_REGISTRY:
            available_types = list(cls.MODEL_REGISTRY.keys())
            raise ValueError(
                f"Unknown model type: '{model_type}'. "
                f"Available types: {available_types}"
            )
        
        model_variants = cls.MODEL_REGISTRY[model_type]
        if variant not in model_variants:
            available_variants = list(model_variants.keys())
            raise ValueError(
                f"Unknown variant '{variant}' for model type '{model_type}'. "
                f"Available variants: {available_variants}"
            )
    
    @classmethod
    def _load_checkpoint(cls, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load checkpoint from file
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dictionary containing checkpoint data
        """
        logger.info(f"📥 Loading checkpoint from: {checkpoint_path}")
        
        if not checkpoint_path or not isinstance(checkpoint_path, str):
            raise ValueError(f"Invalid checkpoint path: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            logger.info(f"✅ Checkpoint loaded successfully")
            return checkpoint
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            raise
    
    @classmethod
    def _prepare_model_config(cls, model_type: str, checkpoint: Optional[str], **kwargs) -> Dict[str, Any]:
        """
        Prepare model configuration by merging defaults with user-provided kwargs
        
        Args:
            model_type: Type of model
            checkpoint: Optional checkpoint path
            **kwargs: Additional model-specific arguments
            
        Returns:
            Merged configuration dictionary
        """
        # Start with default config for this model type
        config = cls.DEFAULT_CONFIGS.get(model_type, {}).copy()
        
        # Update with user-provided kwargs
        config.update(kwargs)
        
        # Set checkpoint path if provided
        if checkpoint:
            config['checkpoint'] = checkpoint
        
        return config
    
    @classmethod
    def _instantiate_model(cls, model_type: str, model_class, config: Dict[str, Any], mode_pretrained: str) -> BaseVisionLanguageModel:
        """
        Instantiate model based on type and configuration
        
        Args:
            model_type: Type of model
            model_class: Model class to instantiate
            config: Model configuration
            
        Returns:
            Model instance
        """
        logger.info(f"🏗️ Creating {model_type} model...")
        
        try:

            model = model_class(**config, mode_pretrained=mode_pretrained)
            logger.info(f"✅ {model_type} model created successfully")
            return model
        except Exception as e:
            logger.error(f"❌ Failed to create {model_type} model: {e}")
            raise
    
    @classmethod
    def _load_pretrained_weights(cls, model: BaseVisionLanguageModel, model_type: str, checkpoint: Optional[str], pretrained: bool):
        """
        Load pretrained weights into model
        
        Args:
            model: Model instance
            model_type: Type of model
            checkpoint: Optional checkpoint path
            pretrained: Whether to load pretrained weights
        """
        # Load from local checkpoint file
        if checkpoint:
            checkpoint_data = cls._load_checkpoint(checkpoint)
            
            # Extract model state dict
            if 'model_state_dict' in checkpoint_data:
                state_dict = checkpoint_data['model_state_dict']
            elif 'state_dict' in checkpoint_data:
                state_dict = checkpoint_data['state_dict']
            else:
                # Assume entire checkpoint is the state dict
                state_dict = checkpoint_data
            
            # Load state dict into model
            try:
                model.load_state_dict(state_dict, strict=True)
                logger.info(f"✅ Loaded weights from checkpoint: {checkpoint}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load checkpoint with strict=True, trying strict=False: {e}")
                model.load_state_dict(state_dict, strict=False)
                logger.info(f"✅ Partially loaded weights from checkpoint: {checkpoint}")
        
        # Load default pretrained weights (for models that support it)
        elif pretrained and model_type == 'medclip':
            if hasattr(model, 'from_pretrained'):
                logger.info(f"📥 Loading pretrained weights for {model_type}...")
                model.from_pretrained()
                logger.info(f"✅ Pretrained weights loaded")
    
    @classmethod
    def create_model(
        cls,
        model_type: str = 'medclip',
        variant: str = 'base',
        checkpoint: Optional[str] = None,
        pretrained: bool = True,
        mode_pretrained: str = "scratch",
        device: Optional[str] = None,
        **kwargs
    ) -> BaseVisionLanguageModel:
        """
        Create a vision-language model with optional checkpoint loading
        
        Args:
            model_type: Model type - 'medclip', 'biomedclip', or 'entrep'
            variant: Model variant (default: 'base')
            checkpoint: Path to local checkpoint file for loading pretrained weights
            pretrained: Whether to load default pretrained weights (only if checkpoint is None)
            device: Target device ('cuda', 'cpu', or None for auto-detection)
            **kwargs: Additional model-specific arguments
            
        Returns:
            Model instance
            
        Examples:
            >>> # Create MedCLIP model with default pretrained weights
            >>> model = ModelFactory.create_model('medclip', pretrained=True)
            
            >>> # Create ENTRep model from local checkpoint
            >>> model = ModelFactory.create_model(
            ...     'entrep',
            ...     checkpoint='checkpoints/entrep_best.pt',
            ...     vision_encoder_type='dinov2'
            ... )
            
            >>> # Create BioMedCLIP model
            >>> model = ModelFactory.create_model('biomedclip')
        """
        if model_type == 'ViT':
            model_name = variant # ViT-B-32, ViT-B-16, ViT-L-14
            model_class = cls.MODEL_REGISTRY['ViT']['base']
            model = model_class(model_name)
        elif model_type == 'rmedclip':
            model_class = cls.MODEL_REGISTRY[model_type][variant]
            model = model_class()
            
        else:
            # 1. Validate inputs
            cls._validate_model_type(model_type, variant)
            
            # 2. Get model class
            model_class = cls.MODEL_REGISTRY[model_type][variant]
            
            # 3. Prepare configuration
            config = cls._prepare_model_config(model_type, checkpoint, **kwargs)
            
            # IMPORTANT: Add pretrained flag to config for ENTRep
            if model_type == 'entrep':
                config['pretrained'] = pretrained
            
            # 4. Instantiate model
            model = cls._instantiate_model(model_type, model_class, config, mode_pretrained)
            
            # 5. Load pretrained weights
            # cls._load_pretrained_weights(model, model_type, checkpoint, pretrained)
            
            # 6. Move to device
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        model = model.cuda()
        # logger.info(f"🔧 Model moved to device: {cu}")
        
        return model
    
    @classmethod
    def get_available_models(cls) -> Dict[str, List[str]]:
        """
        Get list of available models and variants
        
        Returns:
            Dictionary {model_type: [available_variants]}
        """
        return {
            model_type: list(variants.keys())
            for model_type, variants in cls.MODEL_REGISTRY.items()
        }
    
    @classmethod
    def get_available_classifiers(cls) -> Dict[str, List[str]]:
        """
        Get list of available classifiers
        
        Returns:
            Dictionary {model_type: [available_task_types]}
        """
        return {
            model_type: list(task_types.keys())
            for model_type, task_types in cls.CLASSIFIER_REGISTRY.items()
        }
    
    @classmethod
    def print_registry(cls):
        """logger.info information about available models and classifiers"""
        logger.info("🏭 Model Factory Registry")
        logger.info("=" * 50)
        
        logger.info("🤖 Available Models:")
        for model_type, variants in cls.MODEL_REGISTRY.items():
            logger.info(f"  {model_type}: {list(variants.keys())}")
        
        logger.info("🎯 Available Classifiers:")
        for model_type, task_types in cls.CLASSIFIER_REGISTRY.items():
            logger.info(f"  {model_type}: {list(task_types.keys())}")
        
        logger.info(f"📚 Supported Model Types: {SUPPORTED_MODELS}")


# Convenience functions
def create_medclip(
    text_encoder: str = 'bert',
    vision_encoder: str = 'vit',
    pretrained: bool = True,
    checkpoint: Optional[str] = None,
    **kwargs
) -> BaseVisionLanguageModel:
    """
    Create MedCLIP model
    
    Args:
        text_encoder: 'bert' (currently only BERT supported)
        vision_encoder: 'resnet' or 'vit'
        pretrained: Whether to load pretrained weights
        checkpoint: Optional checkpoint path
        **kwargs: Additional arguments
        
    Returns:
        MedCLIP model instance
    """
    return ModelFactory.create_model(
        model_type='medclip',
        variant='base',
        text_encoder_type=text_encoder,
        vision_encoder_type=vision_encoder,
        pretrained=pretrained,
        checkpoint=checkpoint,
        **kwargs
    )


def create_biomedclip(
    checkpoint: Optional[str] = None,
    **kwargs
) -> BaseVisionLanguageModel:
    """
    Create BioMedCLIP model
    
    Args:
        checkpoint: Optional checkpoint path
        **kwargs: Additional arguments
        
    Returns:
        BioMedCLIP model instance
    """
    return ModelFactory.create_model(
        model_type='biomedclip',
        checkpoint=checkpoint,
        **kwargs
    )

# Wrapper functions for easier import
def create_model(model_type: str = 'medclip', **kwargs):
    """Create a model using ModelFactory"""
    return ModelFactory.create_model(model_type=model_type, **kwargs)

def create_entrep(
    text_encoder: str = 'clip',
    vision_encoder: str = 'dinov2',
    checkpoint: Optional[str] = None,
    vision_checkpoint: Optional[str] = None,
    **kwargs
) -> BaseVisionLanguageModel:
    """
    Create ENTRep model
    
    Args:
        text_encoder: 'clip' or 'none'
        vision_encoder: 'clip', 'endovit', or 'dinov2'
        checkpoint: Path to full model checkpoint (contains both text and vision)
        vision_checkpoint: Path to vision encoder checkpoint only (deprecated, use checkpoint)
        **kwargs: Additional arguments
        
    Returns:
        ENTRep model instance
        
    Examples:
        >>> # Create ENTRep with full checkpoint
        >>> model = create_entrep(
        ...     checkpoint='checkpoints/entrep_best.pt',
        ...     vision_encoder='dinov2'
        ... )
        
        >>> # Create ENTRep without checkpoint
        >>> model = create_entrep(
        ...     text_encoder='clip',
        ...     vision_encoder='dinov2'
        ... )
    """
    return ModelFactory.create_model(
        model_type='entrep',
        variant='base',
        checkpoint=checkpoint,
        text_encoder_type=text_encoder,
        vision_encoder_type=vision_encoder,
        vision_checkpoint=vision_checkpoint,
        **kwargs
    )


if __name__ == "__main__":
    demo_factory()
