"""
Vision-Language Models for Medical Image Analysis

This module provides implementations of various vision-language models
for medical image analysis, including MedCLIP and BioMedCLIP.
"""

# Base classes
from .model import (
    BaseVisionLanguageModel,
    BaseClassifier,
    BaseZeroShotClassifier,
    BaseSupervisedClassifier,
    BasePromptLearner
)

# MedCLIP models
from .medclip import (
    MedCLIPModel,
    MedCLIPTextModel,
    MedCLIPVisionModel,
    MedCLIPVisionModelViT,
    PromptClassifier,
    SuperviseClassifier,
    PromptTuningClassifier,
    PartiallyFixedEmbedding
)

# BioMedCLIP models
from .biomedclip import (
    BioMedCLIPModel,
    BioMedCLIPClassifier
)

# ENTRep models
from .entrep import (
    ENTRepModel,
    DinoV2Model,
    EntVitModel
)

# Factory
from .factory import (
    ModelFactory,
    create_model,
    create_medclip,
    create_biomedclip,
    create_entrep
)

# Version
__version__ = "0.1.0"

# Define what should be imported with "from models import *"
__all__ = [
    # Base classes
    'BaseVisionLanguageModel',
    'BaseClassifier',
    'BaseZeroShotClassifier',
    'BaseSupervisedClassifier',
    'BasePromptLearner',
    
    # MedCLIP
    'MedCLIPModel',
    'MedCLIPTextModel',
    'MedCLIPVisionModel',
    'MedCLIPVisionModelViT',
    'PromptClassifier',
    'SuperviseClassifier',
    'PromptTuningClassifier',
    
    # BioMedCLIP
    'BioMedCLIPModel',
    'BioMedCLIPClassifier',
    'BioMedCLIPFeatureExtractor',
    
    # ENTRep
    'ENTRepModel',
    'DinoV2Model',
    'EntVitModel',
    
    # Factory
    'ModelFactory',
    'create_model',
    'create_medclip',
    'create_biomedclip',
    'create_entrep',
]
