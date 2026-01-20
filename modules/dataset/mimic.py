"""
Completed testing
MIMIC-CXR only used for text-to-image retrieval tasks
"""

import os
import random
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict
from torchvision import transforms
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from open_clip import get_tokenizer
from datasets import load_from_disk
from PIL import Image

from .base import BaseContrastiveDataset, BaseClassificationDataset, BaseCollator
from ..utils.constants import (
    DATASET_CONFIGS,
    BERT_TYPE, BIOMEDCLIP_MODEL, DEFAULT_TEMPLATES
)
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class MIMICContrastiveDataset(BaseContrastiveDataset):
    """
    MIMIC dataset for contrastive learning (pretraining)
    Simple format: image-findings pairs
    """
    
    def __init__(
        self,
        data_root: str = '/data/elo/data/mimic',
        split: str = 'train',
        model_type: str = 'medclip',
        transform: Optional[transforms.Compose] = None,
        **kwargs
    ):
        """
        Args:
            data_root: Root directory containing MIMIC dataset
            split: Data split ('train')
            model_type: 'medclip' or 'biomedclip'
        """
        
        super().__init__(
            data_root=data_root,
            split=split,
            model_type=model_type,
            transform=transform,
            **kwargs
        )
        
    def _load_data(self) -> pd.DataFrame:
        """Load MIMIC data from HuggingFace dataset format"""
        try:
            if not os.path.exists(self.data_root):
                raise FileNotFoundError(f"MIMIC data not found in root {self.data_root}")
            logger.info(f"Loading MIMIC data from {self.data_root}")

            # Get train split
            if self.split == 'train':
                data = pd.read_csv(os.path.join(self.data_root, 'train.csv'))

            elif self.split == 'val':
                data = pd.read_csv(os.path.join(self.data_root, 'val.csv'))
            else:
                raise ValueError(f"Invalid split: {self.split}")
            # Convert to DataFrame
            df = pd.DataFrame({
                'image_path': data['file_path'].tolist(),
                'text': data['caption'].tolist()
            })
            # Filter out empty findings
            df = df[df['text'].notna() & (df['text'] != '')]
            df = df.reset_index(drop=True)
            
            logger.info(f"Loaded {len(df)} image-text pairs")
            return df
            
        except Exception as e:
            logger.error(f"Error loading MIMIC data: {e}")
            raise FileNotFoundError(f"Could not load MIMIC data from {self.data_root}")
        
    def _setup_dataset(self):
        """Setup MIMIC-specific configurations"""
        self.class_names = [] 
        
    def get_class_names(self) -> List[str]:
        """Return empty list"""
        return []
        
    def get_class_prompts(self) -> Dict[str, List[str]]:
        """Return empty dict"""
        return {}

class MIMICContrastiveCollator(BaseCollator):
    """
    Collator for MIMIC contrastive learning
    """
    
    def __init__(
        self,
        model_type: str = 'medclip',
        **kwargs
    ):
        """
        Args:
            model_type: 'medclip' or 'biomedclip'
        """
        super().__init__(model_type=model_type, mode='contrastive', **kwargs)
        
        # Initialize tokenizer
        if model_type == 'medclip':
            self.tokenizer = AutoTokenizer.from_pretrained(BERT_TYPE)
            self.tokenizer.model_max_length = 77
        elif model_type == 'biomedclip':
            self.tokenizer = get_tokenizer(BIOMEDCLIP_MODEL)
            self.context_length = 256
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
    def __call__(self, batch: List[Tuple]) -> Dict[str, Any]:
        """
        Process batch for contrastive learning
        
        Args:
            batch: List of (img_tensor, findings_text)
            
        Returns:
            Dict containing processed batch data
        """
        inputs = defaultdict(list)
        
        for data in batch:
            img_tensor, findings = data
            inputs['pixel_values'].append(img_tensor)
            inputs['text'].append(findings)
            
        # Process images
        inputs['pixel_values'] = self._process_images(inputs['pixel_values'])
        
        # Tokenize text
        if self.model_type == 'medclip':
            text_inputs = self.tokenizer(
                inputs['text'],
                truncation=True,
                padding=True,
                return_tensors='pt'
            )
            inputs['input_ids'] = text_inputs['input_ids']
            inputs['attention_mask'] = text_inputs['attention_mask']
            
        elif self.model_type == 'biomedclip':
            text_tokens = self.tokenizer(inputs['text'], context_length=self.context_length)
            inputs['text_tokens'] = text_tokens
            
        return inputs


# Convenience functions
def create_mimic_contrastive_dataloader(
    data_root: str = '/data/elo/data/mimic',
    model_type: str = 'medclip',
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
) -> torch.utils.data.DataLoader:
    """
    Create DataLoader for MIMIC contrastive learning
    """
    from torch.utils.data import DataLoader
    
    # Create dataset
    dataset = MIMICContrastiveDataset(
        data_root=data_root,
        model_type=model_type,
        **kwargs
    )
    
    # Create collator
    collator = MIMICContrastiveCollator(model_type=model_type)
    
    # Create DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available()
    )


def demo_mimic_dataset():
    """
    Demo using MIMIC dataset
    """
    logger.info("🏥 MIMIC Dataset Demo")
    
    # Test contrastive dataset
    logger.info("📋 Testing MIMIC Contrastive Dataset:")
    try:
        dataset = MIMICContrastiveDataset(
            data_root='/data/elo/data/mimic',
            model_type='medclip'
        )
        logger.info(f"  Dataset size: {len(dataset)}")
        
        # Test one sample
        if len(dataset) > 0:
            img, findings = dataset[0]
            logger.info(f"  Sample image shape: {img.shape}")
            logger.info(f"  Sample findings: {findings[:100]}...")
            
    except Exception as e:
        logger.error(f"  Error: {e}")
        
    # Test DataLoader
    logger.info("📋 Testing MIMIC DataLoader:")
    try:
        dataloader = create_mimic_contrastive_dataloader(
            data_root='/data/elo/data/mimic',
            model_type='medclip',
            batch_size=2
        )
        
        for batch in dataloader:
            logger.info(f"  Batch pixel_values shape: {batch['pixel_values'].shape}")
            logger.info(f"  Batch text length: {len(batch['text'])}")
            if 'input_ids' in batch:
                logger.info(f"  Batch input_ids shape: {batch['input_ids'].shape}")
            break
            
    except Exception as e:
        logger.error(f"  Error: {e}")
        
    logger.info("✅ MIMIC Dataset Demo completed!")


if __name__ == "__main__":
    demo_mimic_dataset()