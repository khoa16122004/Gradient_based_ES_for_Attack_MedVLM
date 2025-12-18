"""
Completed testing
RSNA only used for zero shot classification tasks
"""

import os
import random
import zipfile
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict
from torchvision import transforms

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

import gdown

from .base import BaseClassificationDataset, BaseCollator
from ..utils.constants import (
    RSNA_TASKS, RSNA_CLASS_PROMPTS, DATASET_CONFIGS,
    BERT_TYPE, BIOMEDCLIP_MODEL, DEFAULT_TEMPLATES
)
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class RSNADataset(BaseClassificationDataset):
    """
    RSNA Pneumonia dataset cho binary classification
    Hỗ trợ cả MedCLIP và BiomedCLIP
    """
    
    def __init__(
        self,
        data_root: str = '../local_data',
        split: str = 'test',
        model_type: str = 'medclip',
        datalist: Optional[List[str]] = None,
        transform: Optional[transforms.Compose] = None,
        **kwargs
    ):
        """
        Args:
            data_root: Root directory containing data files
            split: Data split ('train', 'test')
            model_type: 'medclip' or 'biomedclip'
            datalist: List of data files to load
        """

        self.cls_prompts = RSNA_CLASS_PROMPTS
        self.template = DEFAULT_TEMPLATES[model_type]
        self.data_root = os.path.join(data_root, "rsna")
        super().__init__(data_root=self.data_root, split=split, model_type=model_type, transform=transform, **kwargs)
        

    
    def _load_data(self) -> pd.DataFrame:
        """Load RSNA data from files"""
        def create_rsna_csv(pneumonia_lst, normal_lst):
            df = pd.DataFrame(columns=['imgpath', 'Pneumonia', 'Normal'])
            for patient_id in pneumonia_lst:
                new_row = pd.DataFrame({'imgpath': [patient_id], 'Pneumonia': [1], 'Normal': [0]})
                df = pd.concat([df, new_row], ignore_index=True)
            for patient_id in normal_lst:
                new_row = pd.DataFrame({'imgpath': [patient_id], 'Pneumonia': [0], 'Normal': [1]})
                df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(os.path.join(self.data_root, 'rsna-balanced-test-meta.csv'), index=True)
                
            for task in RSNA_TASKS:
                if task not in df.columns:
                    logger.error(f"Column {task} not found in RSNA data, adding it")
            return df
        
        def download_rsna_dataset():    
            if gdown is None:
                logger.error("gdown not installed. Please install with: pip install gdown")
                return False
                
            # url_id = "1r2mIaytuvHQc5D77BAuIfkLeW9jGXjed"
            url_id = "1QgTKzomgbmwFtR16yw9yW29qdCgh1Fuk"
            rsna_output = os.path.join(self.data_root, "rsna_pneumonia_detection.zip")
            logger.info("Downloading RSNA dataset from Google Drive...")
            
            try:
                gdown.download(id=url_id, output=rsna_output, quiet=False)
                with zipfile.ZipFile(rsna_output, 'r') as zip_ref:
                    zip_ref.extractall(self.data_root)
                os.remove(rsna_output) 
                return True
            except Exception as e:
                logger.error(f"Failed to download RSNA dataset: {e}")
                return False
        
        os.makedirs(self.data_root, exist_ok=True)
        rsna_data_path = self.data_root
        
        # Check if required files exist
        img_folder_path = os.path.join(rsna_data_path, "stage_2_train_images")
        train_csv_path = os.path.join(rsna_data_path, "stage_2_train_labels.csv")
        if not os.path.exists(img_folder_path) or not os.path.exists(train_csv_path):
            logger.info(f"RSNA data not found in {self.data_root}, preparing data")
            if not download_rsna_dataset():
                logger.warning("Failed to download RSNA data")
        else:
            logger.info(f"RSNA data found in {self.data_root}")
        
        if not os.path.exists(train_csv_path):
            logger.error(f"RSNA labels file not found at {train_csv_path}")
            
        train_csv = pd.read_csv(train_csv_path)
        pneumonia_lst = []
        normal_lst = []
        for index, row in train_csv.iterrows():
            if row['Target'] == 1:
                pneumonia_lst.append(os.path.join(img_folder_path, f"{row['patientId']}.dcm"))
            else:
                normal_lst.append(os.path.join(img_folder_path, f"{row['patientId']}.dcm"))
        df = create_rsna_csv(pneumonia_lst, normal_lst)
        return df
        
    def _setup_dataset(self):
        """Setup RSNA-specific configurations"""
        self.class_names = RSNA_TASKS
        
    def get_class_names(self) -> List[str]:
        return self.class_names
        
    def get_class_prompts(self) -> Dict[str, List[str]]:
        return RSNA_CLASS_PROMPTS
           

class RSNAZeroShotCollator(BaseCollator):
    """
    Collator cho RSNA zero-shot classification
    """
    
    def __init__(
        self,
        model_type: str = 'medclip',
        cls_prompts: Optional[Dict[str, List[str]]] = None,
        template: str = None,
        n_prompt: int = 5,
        **kwargs
    ):
        """
        Args:
            model_type: 'medclip' hoặc 'biomedclip'
            cls_prompts: Class prompts dictionary
            template: Text template
            n_prompt: Number of prompts per class
        """
        super().__init__(model_type=model_type, mode='binary', **kwargs)
        
        if cls_prompts is None:
            cls_prompts = RSNA_CLASS_PROMPTS
            
        if template is None:
            template = DEFAULT_TEMPLATES.get(model_type, DEFAULT_TEMPLATES['general'])
            
        self.cls_prompts = cls_prompts
        self.template = template
        self.n_prompt = n_prompt
        self.class_names = RSNA_TASKS
        
        # Process class prompts
        self.prompt_texts_inputs = self._process_class_prompts()
        
    def _process_class_prompts(self) -> Dict[str, Any]:
        """Process class prompts into tokenized inputs"""
        from ..utils.helpers import generate_rsna_class_prompts
        
        # Generate prompts từ templates
        processed_prompts = generate_rsna_class_prompts(
            self.cls_prompts,
            n=self.n_prompt
        )
        
        if self.model_type == 'medclip':
            tokenizer = AutoTokenizer.from_pretrained(BERT_TYPE)
            tokenizer.model_max_length = 77
            
            prompt_inputs = {}
            for class_name, prompts in processed_prompts.items():
                templated_prompts = [self.template + prompt for prompt in prompts]
                text_inputs = tokenizer(
                    templated_prompts,
                    truncation=True,
                    padding=True,
                    return_tensors='pt'
                )
                prompt_inputs[class_name] = text_inputs
                
        elif self.model_type == 'biomedclip':
            # BioMedCLIP model đã có tokenizer sẵn, không cần tokenize ở đây
            # Truyền text strings để model tự tokenize
            prompt_inputs = {}
            for class_name, prompts in processed_prompts.items():
                templated_prompts = [self.template + prompt for prompt in prompts]
                prompt_inputs[class_name] = templated_prompts
                
        return prompt_inputs
        
    def __call__(self, batch: List[Tuple]) -> Dict[str, Any]:
        """Process batch cho zero-shot classification"""
        inputs = defaultdict(list)
        
        for data in batch:
            img_tensor, labels = data
            inputs['pixel_values'].append(img_tensor)
            inputs['labels'].append(labels)
        
        # print("Input pixel: ", inputs['pixel_values'])
        # raise
        # Process images
        inputs['pixel_values'] = self._process_images(inputs['pixel_values'])
        
        # Process labels cho binary classification
        inputs['labels'] = self._process_labels(inputs['labels'], self.class_names)
        
        return {
            'pixel_values': inputs['pixel_values'],
            'prompt_inputs': self.prompt_texts_inputs,
            'labels': inputs['labels'],
            'class_names': self.class_names
        }
# Utility functions cho RSNA dataset
def create_rsna_dataloader(
    data_root: str = 'local_data',
    split: str = 'test',
    model_type: str = 'medclip',
    task_type: str = 'zeroshot',  # 'zeroshot'
    batch_size: int = 16,
    shuffle: bool = False,
    num_workers: int = 0,
    cls_prompts: Optional[Dict[str, List[str]]] = None,
    template: str = None,
    **kwargs
) -> torch.utils.data.DataLoader:
    """
    Create DataLoader for RSNA dataset
    
    Args:
        data_root: Root directory
        split: Data split
        model_type: 'medclip' or 'biomedclip'
        task_type: 'zeroshot' or 'supervised'
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of workers
        cls_prompts: Class prompts (cho zero-shot)
        template: Text template (cho zero-shot)
        
    Returns:
        DataLoader instance
    """
    from torch.utils.data import DataLoader
    
    # Create dataset
    dataset = RSNADataset(
        data_root=data_root,
        split=split,
        model_type=model_type,
        **kwargs
    )
    
    # Create collator
    if task_type == 'zeroshot':
        collator = RSNAZeroShotCollator(
            model_type=model_type,
            cls_prompts=cls_prompts,
            template=template
        )
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
        
    # Create DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available()
    )


def demo_rsna_dataset():
    """
    Demo using RSNA dataset
    """
    logger.info("🫁 RSNA Pneumonia Dataset Demo")
    
    # Test with MedCLIP
    logger.info("📋 Testing MedCLIP RSNA Dataset:")
    try:
        dataset_medclip = RSNADataset(
            split='test',
            model_type='medclip'
        )
        logger.info(f"  Dataset size: {len(dataset_medclip)}")
        logger.info(f"  Class names: {dataset_medclip.get_class_names()}")
        
        # Test one sample
        if len(dataset_medclip) > 0:
            img, labels = dataset_medclip[0]
            logger.info(f"  Sample image shape: {img.shape}")
            logger.info(f"  Sample labels: {labels}")
            
    except Exception as e:
        logger.info(f"  Error: {e}")
        
    # Test with BiomedCLIP
    logger.info("📋 Testing BiomedCLIP RSNA Dataset:")
    try:
        dataset_biomedclip = RSNADataset(
            split='test',
            model_type='biomedclip'
        )
        logger.info(f"  Dataset size: {len(dataset_biomedclip)}")
        logger.info(f"  Class names: {dataset_biomedclip.get_class_names()}")
        
        # Test một sample
        if len(dataset_biomedclip) > 0:
            img, labels = dataset_biomedclip[0]
            logger.info(f"  Sample image shape: {img.shape}")
            logger.info(f"  Sample labels: {labels}")
            
    except Exception as e:
        logger.info(f"  Error: {e}")
        
    # Test DataLoader
    logger.info("📋 Testing RSNA DataLoader:")
    try:
        dataloader = create_rsna_dataloader(
            split='test',
            model_type='medclip',
            task_type='zeroshot',
            batch_size=2
        )
        
        for batch in dataloader:
            logger.info(f"  Batch pixel_values shape: {batch['pixel_values'].shape}")
            logger.info(f"  Batch labels shape: {batch['labels'].shape}")
            logger.info(f"  Class names: {batch['class_names']}")
            break
            
    except Exception as e:
        logger.info(f"  Error: {e}")
        
    logger.info("✅ RSNA Dataset Demo completed!")


if __name__ == "__main__":
    demo_rsna_dataset()
