"""
Simple base classes for medical datasets
"""

import os
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Union, Optional, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pydicom
from torchvision import transforms

from ..utils.constants import (
    IMG_SIZE, IMG_MEAN, IMG_STD, DEFAULT_DATA_ROOT,
    SUPPORTED_MODELS, DEFAULT_TEMPLATES
)
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class BaseMedicalDataset(Dataset, ABC):
    """
    Abstract base class for all medical datasets
    Simple format: image-text pairs
    """
    
    def __init__(
        self,
        data_root: str = DEFAULT_DATA_ROOT,
        split: str = 'train',
        model_type: str = 'medclip',
        transform: Optional[transforms.Compose] = None,
        **kwargs
    ):
        """
        Args:
            data_root: Root directory containing data files
            split: Data split ('train', 'test', 'val', etc.)
            model_type: 'medclip' or 'biomedclip'
            transform: Custom image transforms
        """
        super().__init__()
        
        if model_type not in SUPPORTED_MODELS:
            raise ValueError(f"Model type {model_type} not supported. Choose from {SUPPORTED_MODELS}")
            
        self.data_root = data_root
        self.split = split
        self.model_type = model_type


        # khoa'fix
        # # Set default transforms
        # if transform is None:
        #     self.transform = self._get_default_transform() # sửa lại cáinày
        # else:
        #     self.transform = transform
        
        self.transform = transform
        
            
        # Load data
        self.df = self._load_data()
        
        # Dataset-specific setup
        self._setup_dataset()
        
    @abstractmethod
    def _load_data(self) -> pd.DataFrame:
        """Load data from files and return DataFrame"""
        pass
        
    @abstractmethod
    def _setup_dataset(self):
        """Setup dataset-specific configurations"""
        pass
        
    @abstractmethod
    def get_class_names(self) -> List[str]:
        """Return list of class names"""
        pass
        
    @abstractmethod
    def get_class_prompts(self) -> Dict[str, List[str]]:
        """Return class prompts dictionary"""
        pass
        
    def _get_default_transform(self) -> transforms.Compose:
        """Get default image transforms"""
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[IMG_MEAN], std=[IMG_STD])
        ])
        
    def _pad_img(self, img: Image.Image, min_size: int = 224, fill_color: int = 0) -> Image.Image:
        """Pad image to square"""
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im
        
    def _load_image(self, img_path: str) -> Image.Image:
        """Load image from path, supports both DICOM and standard formats"""
        if img_path.lower().endswith('.dcm'):
            # Load DICOM file
            dicom_data = pydicom.dcmread(img_path)
            img_array = dicom_data.pixel_array
            # Normalize pixel values to 0-255 range
            img_array = img_array.astype(np.float32)
            img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255
            img_array = img_array.astype(np.uint8)
            
            # Convert to PIL Image
            img = Image.fromarray(img_array, mode='L')
        else:
            img = Image.open(img_path).convert('L')
            
        return self._pad_img(img)
        
    def __len__(self) -> int:
        return len(self.df)
        
    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        """Get item at index"""
        pass


class BaseContrastiveDataset(BaseMedicalDataset):
    """
    Base class for contrastive learning datasets
    Simple image-text pairs format
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        """
        Return image and text (simple format)
        
        Returns:
            img: Preprocessed image tensor
            text: Text description
        """
        row = self.df.iloc[index]
        
        # Load and preprocess image
        if 'image_path' in row:
            img_path = row['image_path']
            img = self._load_image(img_path)
        else:
            img = row['image']
        img_tensor = self.transform(img)
        
        # Add channel dimension
        if img_tensor.dim() == 2:
            img_tensor = img_tensor.unsqueeze(0)
            
        # Get text
        text = row.get('findings', row.get('text', ''))
        if pd.isna(text) or text == '':
            text = "No acute findings."
            
        return img_tensor, text


class BaseClassificationDataset(BaseMedicalDataset):
    """
    Base class for classification datasets
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        Return image and labels
        
        Returns:
            image: Preprocessed image tensor
            labels: Dictionary {class_name: label_value}
        """
        row = self.df.iloc[index]
        
        
            # Get labels
        class_names = self.get_class_names()
        labels = {class_name: int(row.get(class_name, 0)) for class_name in class_names}
        
        # Load and preprocess image
        img_path = row['imgpath'] if 'imgpath' in row else row['image_path']
        img = self._load_image(img_path)

        # khoa fix
        if self.transform:
            img_tensor = self.transform(img)
            
            # Add channel dimension
            if img_tensor.dim() == 2:
                img_tensor = img_tensor.unsqueeze(0)
            elif img_tensor.shape[0] == 1 and self.model_type == 'biomedclip':
                # BiomedCLIP expects RGB
                img_tensor = img_tensor.repeat(3, 1, 1)
            
            return img_tensor, labels

        else:
            return img, labels
        
  
        


class BaseCollator(ABC):
    """
    Abstract base class for data collators
    """
    
    def __init__(
        self,
        model_type: str = 'medclip',
        mode: str = 'multiclass',
        **kwargs
    ):
        """
        Args:
            model_type: 'medclip' or 'biomedclip'
            mode: 'multiclass', 'multilabel', 'binary', 'contrastive'
        """
        if model_type not in SUPPORTED_MODELS:
            raise ValueError(f"Model type {model_type} not supported. Choose from {SUPPORTED_MODELS}")
            
        valid_modes = ['multiclass', 'multilabel', 'binary', 'contrastive']
        if mode not in valid_modes:
            raise ValueError(f"Mode must be one of {valid_modes}")
            
        self.model_type = model_type
        self.mode = mode
        
    @abstractmethod
    def __call__(self, batch: List[Any]) -> Dict[str, Any]:
        """Process a batch of data"""
        pass
        
    def _process_images(self, images: List[torch.Tensor]) -> torch.Tensor:
        """Process batch of images"""
        pixel_values = torch.stack(images, dim=0)
        
        # Ensure correct format for each model type
        if self.model_type == 'biomedclip':
            # BiomedCLIP expects RGB
            if pixel_values.shape[1] == 1:
                pixel_values = pixel_values.repeat(1, 3, 1, 1)
        elif self.model_type == 'medclip':
            # MedCLIP can handle grayscale, but usually convert to RGB
            if pixel_values.shape[1] == 1:
                pixel_values = pixel_values.repeat(1, 3, 1, 1)
                
        return pixel_values
        
    def _process_labels(
        self, 
        labels: List[Dict[str, int]], 
        class_names: List[str]
    ) -> torch.Tensor:
        """Process batch of labels"""
        # Convert to matrix
        label_matrix = []
        for label_dict in labels:
            label_row = [label_dict.get(class_name, 0) for class_name in class_names]
            label_matrix.append(label_row)
            
        label_matrix = np.array(label_matrix, dtype=float)
        
        # Convert according to mode
        if self.mode in ['multiclass', 'binary']:
            # Convert to class indices
            labels_tensor = torch.tensor(label_matrix.argmax(axis=1), dtype=torch.long)
        else:  # multilabel
            # Keep as one-hot
            labels_tensor = torch.tensor(label_matrix, dtype=torch.float)
            
        return labels_tensor


def create_dataloader(
    dataset: BaseMedicalDataset,
    collator: BaseCollator,
    batch_size: int = 16,
    shuffle: bool = False,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    """
    Create DataLoader with dataset and collator
    
    Args:
        dataset: Dataset instance
        collator: Collator instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of workers
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
        **kwargs
    )