"""
Medical Vision-Language Models Framework

This package provides a comprehensive framework for medical vision-language models
including datasets, models, evaluation, and training utilities.
"""

__version__ = "1.0.0"
__author__ = "Medical AI Team"

# Import main modules
from . import dataset
from . import models
from . import utils

__all__ = [
    'dataset',
    'models', 
    'utils'
]
