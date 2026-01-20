"""
Completed testing
COVID only used for zero shot classification tasks
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
import kagglehub

from .base import BaseClassificationDataset, BaseCollator
from ..utils.constants import (
    COVID_TASKS, COVID_CLASS_PROMPTS, DATASET_CONFIGS,
    BERT_TYPE, BIOMEDCLIP_MODEL, DEFAULT_TEMPLATES
)
from ..utils.logging_config import get_logger

logger = get_logger(__name__)

class COVIDDataset(BaseClassificationDataset):
    def __init__(
        self,
        data_root: str = 'local_data',
        split: str = 'test',
        model_type: str = 'medclip',
        datalist: Optional[List[str]] = None,
        transform: Optional[transforms.Compose] = None,
        **kwargs
    ):
        """
        Args:
            data_root: Root directory chứa data files
            split: Data split ('train', 'test', 'small')
            model_type: 'medclip' hoặc 'biomedclip'
            datalist: List of data files to load
        """
        if datalist is None:
            config = DATASET_CONFIGS['covid']
            if split in config['data_files']:
                datalist = [config['data_files'][split].replace('-meta.csv', '')]
            else:
                datalist = [f'covid-{split}']
                
        self.datalist = datalist
        
        super().__init__(
            data_root=data_root,
            split=split,
            model_type=model_type,
            transform=transform,
            **kwargs
        )
        
    def _load_data(self) -> pd.DataFrame:
        """Load COVID data from files"""
        df_list = []

        for data in self.datalist:
            filename = os.path.join(self.data_root, 'covid', f'{data}-meta.csv')
            if os.path.exists(filename):
                logger.info(f'Loading COVID data from {filename}')
                df = pd.read_csv(filename, index_col=0)
                df_list.append(df)
            else:
                logger.info(f'File {filename} not found, preparing data')
                self._prepare_data()
                
        if not df_list:
            raise FileNotFoundError(f"No COVID data files found for datalist: {self.datalist}")
            
        df = pd.concat(df_list, axis=0).reset_index(drop=True)
        
        # Ensure have enough columns for COVID tasks
        for task in COVID_TASKS:
            if task not in df.columns:
                df[task] = 0
                
        # Validate data: each sample must have at least 1 label = 1
        for idx, row in df.iterrows():
            if (row[COVID_TASKS] == 0).all():
                # Default to Normal if no label
                df.loc[idx, 'Normal'] = 1
                
        return df
        
    def _prepare_data(self):
        """Prepare COVID data"""
        os.makedirs(os.path.join(self.data_root, 'covid'), exist_ok=True)
        def create_covid_csv(covid_lst, non_covid_lst, split):
            df = pd.DataFrame(columns=['imgpath', 'COVID', 'Normal'])
            for covid in covid_lst:
                new_row = pd.DataFrame({'imgpath': [covid], 'COVID': [1], 'Normal': [0]})
                df = pd.concat([df, new_row], ignore_index=True)
            for non_covid in non_covid_lst:
                new_row = pd.DataFrame({'imgpath': [non_covid], 'COVID': [0], 'Normal': [1]})
                df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(f'local_data/covid/covid-{split}-meta.csv', index=True)
            logger.info(f'Created COVID data at {os.path.join(self.data_root, "covid/covid-{split}-meta.csv")}')
            return df
        try:
            covid_data_path = kagglehub.dataset_download('tawsifurrahman/covid19-radiography-database')
            covid_data_path = os.path.join(covid_data_path, 'COVID-19_Radiography_Dataset')
            viral_pneumonia_path = os.path.join(
                covid_data_path,
                'Viral Pneumonia',
                'images'
            )
            viral_pneumonia_path_list = [os.path.join(viral_pneumonia_path, f) 
                                        for f in os.listdir(viral_pneumonia_path)]

            lung_opacity_path = os.path.join(
                    covid_data_path,
                    'Lung_Opacity',
                    'images'
            )
            lung_opacity_path_list = [os.path.join(lung_opacity_path, f)
                                    for f in os.listdir(lung_opacity_path)]

            normal_path = os.path.join( 
                    covid_data_path,
                    'Normal',
                    'images'
            )
            normal_path_list = [os.path.join(normal_path, f)
                            for f in os.listdir(normal_path)]

            covid_path = os.path.join(
                    covid_data_path,
                    'COVID',
                    'images'
            )
            covid_path_list = [os.path.join(covid_path, f)
                            for f in os.listdir(covid_path)]
            
            covid_csv = create_covid_csv(covid_path_list, 
                                viral_pneumonia_path_list + lung_opacity_path_list + normal_path_list,
                                split=self.split)
            
        except Exception as e:
            logger.error(f'Error creating COVID data: {e}')
        
        
                
    def _setup_dataset(self):
        """Setup COVID-specific configurations"""
        self.class_names = COVID_TASKS
        
    def get_class_names(self) -> List[str]:
        return self.class_names
        
    def get_class_prompts(self) -> Dict[str, List[str]]:
        return COVID_CLASS_PROMPTS


class COVIDZeroShotCollator(BaseCollator):
    """
    Collator for COVID zero-shot classification
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
            model_type: 'medclip' or 'biomedclip'
            cls_prompts: Class prompts dictionary
            template: Text template
            n_prompt: Number of prompts per class
        """
        super().__init__(model_type=model_type, mode='binary', **kwargs)
        
        if cls_prompts is None:
            cls_prompts = COVID_CLASS_PROMPTS
            
        if template is None:
            template = DEFAULT_TEMPLATES.get(model_type, DEFAULT_TEMPLATES['general'])
            
        self.cls_prompts = cls_prompts
        self.template = template
        self.n_prompt = n_prompt
        self.class_names = COVID_TASKS
        
        # Process class prompts
        self.prompt_texts_inputs = self._process_class_prompts()
        
    def _process_class_prompts(self) -> Dict[str, Any]:
        """Process class prompts into tokenized inputs"""
        from ..utils.helpers import generate_covid_class_prompts
        
        # Generate prompts from templates
        processed_prompts = generate_covid_class_prompts(
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
            tokenizer = get_tokenizer(BIOMEDCLIP_MODEL)
            
            prompt_inputs = {}
            for class_name, prompts in processed_prompts.items():
                templated_prompts = [self.template + prompt for prompt in prompts]
                text_tokens = tokenizer(templated_prompts, context_length=256)
                prompt_inputs[class_name] = text_tokens
                
        return prompt_inputs
        
    def __call__(self, batch: List[Tuple]) -> Dict[str, Any]:
        """Process batch for zero-shot classification"""
        inputs = defaultdict(list)
        
        for data in batch:
            img_tensor, labels = data
            inputs['pixel_values'].append(img_tensor)
            inputs['labels'].append(labels)
            
        # Process images
        inputs['pixel_values'] = self._process_images(inputs['pixel_values'])
        
        # Process labels for binary classification
        inputs['labels'] = self._process_labels(inputs['labels'], self.class_names)
        
        return {
            'pixel_values': inputs['pixel_values'],
            'prompt_inputs': self.prompt_texts_inputs,
            'labels': inputs['labels'],
            'class_names': self.class_names
        }

def create_covid_dataloader(
    data_root: str = 'local_data',
    split: str = 'test',
    model_type: str = 'medclip',
    task_type: str = 'zeroshot',  # 'zeroshot' or 'supervised'
    batch_size: int = 16,
    shuffle: bool = False,
    num_workers: int = 0,
    cls_prompts: Optional[Dict[str, List[str]]] = None,
    template: str = None,
    **kwargs
) -> torch.utils.data.DataLoader:
    """
    Create DataLoader for COVID dataset
    
    Args:
        data_root: Root directory
        split: Data split
        model_type: 'medclip' or 'biomedclip'
        task_type: 'zeroshot' or 'supervised'
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of workers
        cls_prompts: Class prompts (for zero-shot)
        template: Text template (for zero-shot)
        
    Returns:
        DataLoader instance
    """
    from torch.utils.data import DataLoader
    
    # Create dataset
    dataset = COVIDDataset(
        data_root=data_root,
        split=split,
        model_type=model_type,
        **kwargs
    )
    
    # Create collator
    if task_type == 'zeroshot':
        collator = COVIDZeroShotCollator(
            model_type=model_type,
            cls_prompts=cls_prompts,
            template=template
        )
    elif task_type == 'supervised':
        collator = COVIDSupervisedCollator(model_type=model_type)
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


def demo_covid_dataset():
    """
    Demo using COVID dataset
    """
    logger.info("🦠 COVID Dataset Demo")
    
    # Test with MedCLIP
    logger.info("📋 Testing MedCLIP COVID Dataset:")
    try:
        dataset_medclip = COVIDDataset(
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
    logger.info("📋 Testing BiomedCLIP COVID Dataset:")
    try:
        dataset_biomedclip = COVIDDataset(
            split='test',
            model_type='biomedclip'
        )
        logger.info(f"  Dataset size: {len(dataset_biomedclip)}")
        logger.info(f"  Class names: {dataset_biomedclip.get_class_names()}")
        
        # Test one sample
        if len(dataset_biomedclip) > 0:
            img, labels = dataset_biomedclip[0]
            logger.info(f"  Sample image shape: {img.shape}")
            logger.info(f"  Sample labels: {labels}")
            
    except Exception as e:
        logger.info(f"  Error: {e}")
        
    # Test DataLoader with MedCLIP
    logger.info("📋 Testing COVID DataLoader:")
    try:
        dataloader = create_covid_dataloader(
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
        
    logger.info("✅ COVID Dataset Demo completed!")


if __name__ == "__main__":
    demo_covid_dataset()
