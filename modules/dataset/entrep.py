"""
ENTRep dataset for endoscopic image classification tasks
"""

import os
import random
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict
import zipfile
import gdown
import numpy as np
import pandas as pd
import torch
from torchvision import transforms

from transformers import AutoTokenizer
from open_clip import get_tokenizer
from datasets import load_from_disk
from PIL import Image

from .base import BaseContrastiveDataset, BaseClassificationDataset, BaseCollator
from ..utils.constants import (
    DATASET_CONFIGS,
    ENTREP_TASKS, DEFAULT_TEMPLATES, RANDOM_STATE, TRAIN_RATIO, TEST_RATIO, VAL_RATIO
)
from ..utils.logging_config import get_logger
from huggingface_hub import hf_hub_download

logger = get_logger(__name__)

class ENTREPDataset(BaseContrastiveDataset):
    def __init__(
        self,
        data_root: str = 'local_data/entrep',
        split: str = 'train',
        model_type: str = 'entrep',
        transform: Optional[transforms.Compose] = None,
        **kwargs
    ):
        super().__init__(
            data_root=data_root,
            split=split,
            model_type=model_type,
            transform=transform,
            **kwargs
        )
        
        self.df = self._load_data()
    def create_csv(self) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(self.data_root, 'entrep-data.csv'))
        for index, row in df.iterrows():
            row['image_path'] = os.path.join(self.data_root, 'images', row['image_path'])
            df.loc[index, 'image_path'] = row['image_path']
        df.to_csv(os.path.join(self.data_root, 'entrep-data.csv'), index=False)

        nose_df = df[df['nose'] == 1]
        nose_df = nose_df.sample(frac=1, random_state=RANDOM_STATE)

        vocal_throat_df = df[df['vocal-throat'] == 1]
        vocal_throat_df = vocal_throat_df.sample(frac=1, random_state=RANDOM_STATE)

        ear_df = df[df['ear'] == 1]
        ear_df = ear_df.sample(frac=1, random_state=RANDOM_STATE)

        throat_df = df[df['throat'] == 1]
        throat_df = throat_df.sample(frac=1, random_state=RANDOM_STATE)

        # Split into train, test, and validation sets
        nose_train = nose_df.iloc[:int(len(nose_df) * TRAIN_RATIO)]
        nose_test = nose_df.iloc[int(len(nose_df) * TRAIN_RATIO):int(len(nose_df) * (TRAIN_RATIO + TEST_RATIO))]
        nose_val = nose_df.iloc[int(len(nose_df) * (TRAIN_RATIO + TEST_RATIO)):]

        vocal_throat_train = vocal_throat_df.iloc[:int(len(vocal_throat_df) * TRAIN_RATIO)]
        vocal_throat_test = vocal_throat_df.iloc[int(len(vocal_throat_df) * TRAIN_RATIO):int(len(vocal_throat_df) * (TRAIN_RATIO + TEST_RATIO))]
        vocal_throat_val = vocal_throat_df.iloc[int(len(vocal_throat_df) * (TRAIN_RATIO + TEST_RATIO)):]

        ear_train = ear_df.iloc[:int(len(ear_df) * TRAIN_RATIO)]
        ear_test = ear_df.iloc[int(len(ear_df) * TRAIN_RATIO):int(len(ear_df) * (TRAIN_RATIO + TEST_RATIO))]
        ear_val = ear_df.iloc[int(len(ear_df) * (TRAIN_RATIO + TEST_RATIO)):]

        throat_train = throat_df.iloc[:int(len(throat_df) * TRAIN_RATIO)]
        throat_test = throat_df.iloc[int(len(throat_df) * TRAIN_RATIO):int(len(throat_df) * (TRAIN_RATIO + TEST_RATIO))]
        throat_val = throat_df.iloc[int(len(throat_df) * (TRAIN_RATIO + TEST_RATIO)):]

        # Create train, test, and validation sets
        train_df = pd.concat([nose_train, vocal_throat_train, ear_train, throat_train])
        test_df = pd.concat([nose_test, vocal_throat_test, ear_test, throat_test])
        val_df = pd.concat([nose_val, vocal_throat_val, ear_val, throat_val])

        # Save to CSV
        train_df = train_df.drop('Unnamed: 0', axis=1, errors='ignore').reset_index(drop=True)
        test_df = test_df.drop('Unnamed: 0', axis=1, errors='ignore').reset_index(drop=True)
        val_df = val_df.drop('Unnamed: 0', axis=1, errors='ignore').reset_index(drop=True)

        train_df.to_csv(os.path.join(self.data_root, 'entrep', 'entrep-train-meta.csv'), index=True)
        test_df.to_csv(os.path.join(self.data_root, 'entrep',  'entrep-test-meta.csv'), index=True)
        val_df.to_csv(os.path.join(self.data_root, 'entrep',  'entrep-val-meta.csv'), index=True)
    
    def _load_data(self) -> pd.DataFrame:
        """Load ENTREP data from CSV file"""
        def download_entrep_dataset():    
            if gdown is None:
                logger.error("gdown not installed. Please install with: pip install gdown")
                return False
                
            url_id = "1oJwG1T18ghT8xeHVIUOV3Ao5oDBHKYnR"
            entrep_dir= os.path.join(self.data_root, 'entrep')
            os.makedirs(entrep_dir, exist_ok=True)
            entrep_output= os.path.join(entrep_dir, "entrep.zip")
            logger.info("Downloading ENTREP dataset from Google Drive...")
            
            try:
                gdown.download(id=url_id, output=entrep_output, quiet=False)
                # hf_hub_download(
                #     repo_id="Woffy/ENTREP_CLIP",  # sửa repo của bạn
                #     filename="entrep.zip",
                #     local_dir="local_data",                 # tải đúng vào thư mục bạn muốn
                #     force_download=False,                    # không tải lại nếu đã có sẵn
                # )
                with zipfile.ZipFile(entrep_output, 'r') as zip_ref:
                    zip_ref.extractall(entrep_dir)
                os.remove(entrep_output) 
                self.create_csv()
                return True
            except Exception as e:
                logger.error(f"Failed to download ENTREP dataset: {e}")
                return False
        
        os.makedirs(self.data_root, exist_ok=True)
        entrep_data_path = os.path.join(self.data_root, 'entrep')
        # print("Entrep data path: ", entrep_data_path)
        # input()
        
        # Check if required files exist
        data_csv_path = os.path.join(entrep_data_path, "entrep_data.csv")
        train_csv_path = os.path.join(entrep_data_path, "entrep-train-meta.csv")
        test_csv_path = os.path.join(entrep_data_path, "entrep-test-meta.csv")
        val_csv_path = os.path.join(entrep_data_path, "entrep-val-meta.csv")
        if not os.path.exists(train_csv_path) or not os.path.exists(test_csv_path) or not os.path.exists(val_csv_path):
            logger.info(f"ENTREP data not found in {self.data_root}, preparing data")
            if not download_entrep_dataset():
                logger.warning("Failed to download ENTREP data")
        
        # Load appropriate split
        # if self.split == 'train':
        #     csv_path = train_csv_path
        # elif self.split == 'test':
        #     csv_path = test_csv_path
        # elif self.split == 'val':
        #     csv_path = val_csv_path
        # else:
        #     raise ValueError(f"Invalid split: {self.split}")
        csv_path = data_csv_path    
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found: {csv_path}")
            
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} samples from {csv_path}")
        
        return df
    
    def _setup_dataset(self):
        """Additional setup after loading data"""
        # Create label mapping if needed
        self.class_names = ['nose', 'vocal-throat', 'ear', 'throat']
        self.num_classes = len(self.class_names)
    
    def get_class_names(self) -> List[str]:
        """Return list of class names"""
        return self.class_names if hasattr(self, 'class_names') else ['nose', 'vocal-throat', 'ear', 'throat']
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, index: int):
        """
        Return image and text for contrastive learning
        
        Returns:
            img: Preprocessed image tensor
            text: Text description
        """
        row = self.df.iloc[index]
        
        # Load image
        img_path = row['image_path']

        # img = self._load_image(img_path)
        img = Image.open(img_path)
        labels = {
            'vocal-throat': int(row['vocal-throat']),
            'nose': int(row['nose']),
            'ear': int(row['ear']),
            'throat': int(row['throat']),
        }
        
        # Apply transforms
        # if self.transform:
        #     img_tensor = self.transform(img)
        # else:
        #     img_tensor = transforms.ToTensor()(img)
            
        # # Add channel dimension if needed
        # if img_tensor.dim() == 2:
        #     img_tensor = img_tensor.unsqueeze(0)
            
        # Get text description
        # Ưu tiên sử dụng description từ CSV nếu có
        if 'description' in row and pd.notna(row['description']):
            text = str(row['description'])
        elif 'text' in row and pd.notna(row['text']):
            text = str(row['text'])
        else:
            # Fallback: tạo text từ labels nếu không có description
            text_parts = []
            for class_name in self.class_names:
                if row.get(class_name, 0) == 1:
                    text_parts.append(f"{class_name} endoscopy")
                    
            if text_parts:
                text = "Endoscopic image showing: " + ", ".join(text_parts)
            else:
                text = "Endoscopic image"
            
        # return img_tensor, text
        return img, labels
    
    def get_class_prompts(self) -> Dict[str, List[str]]:
        """Return class prompts for zero-shot classification"""
        return {
            'vocal-throat': ['vocal throat endoscopy', 'vocal cord endoscopy', 'laryngoscopy'],
            'nose': ['nose endoscopy', 'nasal endoscopy', 'endoscopic image of nose'],
            'ear': ['ear endoscopy', 'otoscopy', 'endoscopic image of ear'],
            'throat': ['throat endoscopy', 'pharyngoscopy', 'endoscopic image of throat']
        }


class ENTREPCollator(BaseCollator):
    """
    Collator for ENTREP contrastive learning
    """
    
    def __init__(
        self,
        model_type: str = 'entrep',
        tokenizer_name: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            model_type: Model type (entrep, medclip, biomedclip)
            tokenizer_name: Tokenizer name to use
        """
        super().__init__(model_type=model_type, mode='contrastive', **kwargs)
        
        # Initialize tokenizer based on model type
        if tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            # Default tokenizers
            if model_type == 'medclip':
                self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
                self.tokenizer.model_max_length = 77
            elif model_type == 'biomedclip':
                self.tokenizer = get_tokenizer("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
                self.context_length = 256
            elif model_type == 'entrep':
                self.tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
                self.tokenizer.model_max_length = 77
            else:
                # Default to CLIP tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
                self.tokenizer.model_max_length = 77
                
    def __call__(self, batch: List[Tuple]) -> Dict[str, Any]:
        """
        Process batch for contrastive learning
        
        Args:
            batch: List of (img_tensor, text)
            
        Returns:
            Dict containing processed batch data
        """
        inputs = defaultdict(list)
        
        for img_tensor, text in batch:
            inputs['pixel_values'].append(img_tensor)
            inputs['text'].append(text)
            
        # Process images
        inputs['pixel_values'] = self._process_images(inputs['pixel_values'])
        
        # Tokenize text
        if hasattr(self.tokenizer, 'model_max_length'):
            # HuggingFace tokenizer
            text_inputs = self.tokenizer(
                inputs['text'],
                truncation=True,
                padding=True,
                max_length=getattr(self.tokenizer, 'model_max_length', 77),
                return_tensors='pt'
            )
            inputs['input_ids'] = text_inputs['input_ids']
            inputs['attention_mask'] = text_inputs['attention_mask']
        else:
            # OpenCLIP 
            text_tokens = self.tokenizer(inputs['text'], context_length=self.context_length)
            inputs['text_tokens'] = text_tokens
            
        # Remove temporary text list
        del inputs['text']
        
        return inputs

def create_entrep_dataloader(
    data_root: str = 'local_data/entrep',
    split: str = 'train',
    model_type: str = 'entrep',
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    transform: Optional[transforms.Compose] = None,
    tokenizer_name: Optional[str] = None,
    **kwargs
) -> torch.utils.data.DataLoader:
    """
    Create ENTREP dataloader for contrastive learning
    """
    # Use default transform if not provided
    if transform is None:
        from ..utils.constants import MODEL_TRANSFORMS
        if model_type in MODEL_TRANSFORMS:
            transform = MODEL_TRANSFORMS[model_type]
        else:
            logger.warning(f"No default transform for model_type={model_type}, using basic transform")
            transform = transforms.Compose([
                transforms.Lambda(lambda x: x.convert("RGB")),
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
    
    # Create dataset
    dataset = ENTREPDataset(
        data_root=data_root,
        split=split,
        model_type=model_type,
        transform=transform,
        **kwargs
    )
    
    # Create collator
    collator = ENTREPCollator(
        model_type=model_type,
        tokenizer_name=tokenizer_name,
        **kwargs
    )
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    return dataloader