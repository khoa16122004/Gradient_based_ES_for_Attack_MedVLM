from typing import Dict, List, Optional, Union, Any
import torch
from torch.utils.data import DataLoader

from .base import BaseMedicalDataset, BaseCollator
from .mimic import (
    MIMICContrastiveDataset,
    MIMICContrastiveCollator
)
from .covid import COVIDDataset, COVIDZeroShotCollator
from .rsna import RSNADataset, RSNAZeroShotCollator
from .entrep import ENTREPDataset, ENTREPCollator
from ..utils.constants import DATASET_CONFIGS, SUPPORTED_MODELS
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class DatasetFactory:
    # Registry of available datasets
    DATASET_REGISTRY = {
        'mimic': {
            'contrastive': MIMICContrastiveDataset
        },
        'covid': {
            'classification': COVIDDataset
        },
        'rsna': {
            'classification': RSNADataset
        },
        'entrep': {
            'contrastive': ENTREPDataset,
            'classification': ENTREPDataset,
        }
    }
    
    # Registry of available collators
    COLLATOR_REGISTRY = {
        'mimic': {
            'contrastive': MIMICContrastiveCollator
        },
        'covid': {
            'zeroshot': COVIDZeroShotCollator,
        },
        'rsna': {
            'zeroshot': RSNAZeroShotCollator,
            # 'supervised': RSNASupervisedCollator
        },
        'entrep': {
            'contrastive': ENTREPCollator,
        }
    }
    
    @classmethod
    def create_dataset(
        cls,
        dataset_name: str,
        dataset_type: str = 'classification',
        model_type: str = 'medclip',
        split: str = 'test',
        data_root: str = './local_data',
        **kwargs
    ) -> BaseMedicalDataset:
        """
        Create dataset instance
        
        Args:
            dataset_name: 'mimic', 'covid', 'rsna'
            dataset_type: 'classification', 'contrastive'
            model_type: 'medclip', 'biomedclip'
            split: Data split
            data_root: Root directory
            **kwargs: Additional arguments
            
        Returns:
            Dataset instance
        """
        # Validate inputs
        if dataset_name not in cls.DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(cls.DATASET_REGISTRY.keys())}")
            
        if model_type not in SUPPORTED_MODELS:
            raise ValueError(f"Unknown model type: {model_type}. Available: {SUPPORTED_MODELS}")
            
        dataset_classes = cls.DATASET_REGISTRY[dataset_name]
   
        if dataset_type not in dataset_classes:
            raise ValueError(f"Dataset type '{dataset_type}' not available for {dataset_name}. Available: {list(dataset_classes.keys())}")
            
        # Get dataset class
        dataset_class = dataset_classes[dataset_type]
        
        # Create dataset
        dataset = dataset_class(
            data_root=data_root,
            split=split,
            model_type=model_type,
            **kwargs
        )
        
        return dataset
        
    @classmethod
    def create_collator(
        cls,
        dataset_name: str,
        task_type: str = 'zeroshot',
        model_type: str = 'medclip',
        **kwargs
    ) -> BaseCollator:
        """
        Tạo collator instance
        
        Args:
            dataset_name: 'mimic', 'covid', 'rsna'
            task_type: 'contrastive', 'zeroshot', 'supervised'
            model_type: 'medclip', 'biomedclip'
            **kwargs: Additional arguments
            
        Returns:
            Collator instance
        """
        # Validate inputs
        if dataset_name not in cls.COLLATOR_REGISTRY:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(cls.COLLATOR_REGISTRY.keys())}")
            
        if model_type not in SUPPORTED_MODELS:
            raise ValueError(f"Unknown model type: {model_type}. Available: {SUPPORTED_MODELS}")
            
        collator_classes = cls.COLLATOR_REGISTRY[dataset_name]
        
        if task_type not in collator_classes:
            raise ValueError(f"Task type '{task_type}' not available for {dataset_name}. Available: {list(collator_classes.keys())}")
            
        # Get collator class
        collator_class = collator_classes[task_type]
        
        # Create collator
        collator = collator_class(
            model_type=model_type,
            **kwargs
        )
        
        return collator
        
    @classmethod
    def create_dataloader(
        cls,
        dataset_name: str,
        dataset_type: str = 'classification',
        task_type: str = 'zeroshot',
        model_type: str = 'medclip',
        split: str = 'test',
        data_root: str = '../local_data',
        batch_size: int = 16,
        shuffle: bool = False,
        num_workers: int = 0,
        **kwargs
    ) -> DataLoader:
        """
        Create DataLoader with corresponding dataset and collator
        
        Args:
            dataset_name: 'mimic', 'covid', 'rsna'
            dataset_type: 'classification', 'contrastive'
            task_type: 'contrastive', 'zeroshot', 'supervised'
            model_type: 'medclip', 'biomedclip'
            split: Data split
            data_root: Root directory
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of workers
            **kwargs: Additional arguments
            
        Returns:
            DataLoader instance
        """
        # Create dataset
        dataset = cls.create_dataset(
            dataset_name=dataset_name,
            dataset_type=dataset_type,
            model_type=model_type,
            split=split,
            data_root=data_root,
            **kwargs
        )
        
        # Create collator
        collator = cls.create_collator(
            dataset_name=dataset_name,
            task_type=task_type,
            model_type=model_type,
            **kwargs
        )
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=torch.cuda.is_available()
        )
        
        return dataloader
        
    @classmethod
    def get_available_datasets(cls) -> Dict[str, List[str]]:
        """
        Get list of available datasets and types
        
        Returns:
            Dictionary {dataset_name: [available_types]}
        """
        return {
            dataset_name: list(dataset_types.keys())
            for dataset_name, dataset_types in cls.DATASET_REGISTRY.items()
        }
        
    @classmethod
    def get_available_collators(cls) -> Dict[str, List[str]]:
        """
        Get list of available collators
        
        Returns:
            Dictionary {dataset_name: [available_task_types]}
        """
        return {
            dataset_name: list(collator_types.keys())
            for dataset_name, collator_types in cls.COLLATOR_REGISTRY.items()
        }
        
    @classmethod
    def print_registry(cls):
        """
        Print information about registry
        """
        print("🏭 Dataset Factory Registry")
        print("=" * 50)
        
        print("\n📊 Available Datasets:")
        for dataset_name, dataset_types in cls.DATASET_REGISTRY.items():
            print(f"  {dataset_name}: {list(dataset_types.keys())}")
            
        print("\n🔧 Available Collators:")
        for dataset_name, collator_types in cls.COLLATOR_REGISTRY.items():
            print(f"  {dataset_name}: {list(collator_types.keys())}")
            
        print(f"\n🤖 Supported Models: {SUPPORTED_MODELS}")


# Convenience functions
def create_mimic_dataloader(
    task_type: str = 'zeroshot',
    model_type: str = 'medclip',
    split: str = 'test',
    batch_size: int = 16,
    **kwargs
) -> DataLoader:
    """
    Convenience function to create MIMIC dataloader
    """
    return DatasetFactory.create_dataloader(
        dataset_name='mimic',
        dataset_type='classification',
        task_type=task_type,
        model_type=model_type,
        split=split,
        batch_size=batch_size,
        **kwargs
    )


def create_covid_dataloader(
    task_type: str = 'zeroshot',
    model_type: str = 'medclip',
    split: str = 'test',
    batch_size: int = 16,
    **kwargs
) -> DataLoader:
    """
    Convenience function to create COVID dataloader
    """
    return DatasetFactory.create_dataloader(
        dataset_name='covid',
        dataset_type='classification',
        task_type=task_type,
        model_type=model_type,
        split=split,
        batch_size=batch_size,
        **kwargs
    )


def create_rsna_dataloader(
    task_type: str = 'zeroshot',
    model_type: str = 'medclip',
    split: str = 'test',
    batch_size: int = 16,
    **kwargs
) -> DataLoader:
    """
    Convenience function to create RSNA dataloader
    """
    return DatasetFactory.create_dataloader(
        dataset_name='rsna',
        dataset_type='classification',
        task_type=task_type,
        model_type=model_type,
        split=split,
        batch_size=batch_size,
        **kwargs
    )


def create_contrastive_dataloader(
    dataset_name: str = 'mimic',
    model_type: str = 'medclip',
    split: str = 'train',
    batch_size: int = 16,
    **kwargs
) -> DataLoader:
    """
    Convenience function to create contrastive learning dataloader
    """
    return DatasetFactory.create_dataloader(
        dataset_name=dataset_name,
        dataset_type='contrastive',
        task_type='contrastive',
        model_type=model_type,
        split=split,
        batch_size=batch_size,
        **kwargs
    )


def demo_factory():
    """
    Demo using DatasetFactory
    """
    print("🏭 DatasetFactory Demo")
    
    # Print registry
    DatasetFactory.print_registry()
    
    # Test create datasets
    print("\n📊 Testing Dataset Creation:")
    
    datasets_to_test = [
        ('mimic', 'classification', 'medclip'),
        ('covid', 'classification', 'biomedclip'),
        ('rsna', 'classification', 'medclip')
    ]
    
    for dataset_name, dataset_type, model_type in datasets_to_test:
        try:
            dataset = DatasetFactory.create_dataset(
                dataset_name=dataset_name,
                dataset_type=dataset_type,
                model_type=model_type,
                split='test'
            )
            print(f"  ✅ {dataset_name} {dataset_type} ({model_type}): {len(dataset)} samples")
        except Exception as e:
            print(f"  ❌ {dataset_name} {dataset_type} ({model_type}): {e}")
            
    # Test create dataloaders
    print("\n🔧 Testing DataLoader Creation:")
    
    dataloaders_to_test = [
        ('mimic', 'zeroshot', 'medclip'),
        ('covid', 'supervised', 'biomedclip'),
        ('rsna', 'zeroshot', 'medclip')
    ]
    
    for dataset_name, task_type, model_type in dataloaders_to_test:
        try:
            dataloader = DatasetFactory.create_dataloader(
                dataset_name=dataset_name,
                task_type=task_type,
                model_type=model_type,
                batch_size=2
            )
            
            # Test one batch
            for batch in dataloader:
                print(f"  ✅ {dataset_name} {task_type} ({model_type}): batch shape {batch['pixel_values'].shape}")
                break
                
        except Exception as e:
            print(f"  ❌ {dataset_name} {task_type} ({model_type}): {e}")
            
    logger.info("\n✅ Factory Demo completed!")


# Wrapper functions for easier import
def create_dataloader(dataset_name: str, split: str = 'test', **kwargs):
    """Create a dataloader using DatasetFactory"""
    return DatasetFactory.create_dataloader(
        dataset_name=dataset_name,
        split=split,
        **kwargs
    )


def create_dataset(dataset_name: str, split: str = 'test', **kwargs):
    """Create a dataset using DatasetFactory"""
    return DatasetFactory.create_dataset(
        dataset_name=dataset_name,
        split=split,
        **kwargs
    )


if __name__ == "__main__":
    demo_factory()
