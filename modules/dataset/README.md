# 🏗️ Dataset Module

## 📁 File Structure

```
src/
├── constants.py                 # Constants 
├── utils.py                     # Utility functions cho prompt generation
└── datasets/
    ├── __init__.py             # Module exports
    ├── base.py                 # Abstract base classes
    ├── mimic.py                # MIMIC-CXR dataset implementation
    ├── covid.py                # COVID-19 dataset implementation
    ├── rsna.py                 # RSNA Pneumonia dataset implementation
    ├── factory.py              # Factory pattern for easy creation
    ├── demo.py                 # Demo script with examples
    ├── README.md               
```

## 🏛️ Class Hierarchy

### Base Classes

```
BaseMedicalDataset (ABC)
├── BaseContrastiveDataset
│   └── MIMICContrastiveDataset
└── BaseClassificationDataset
    ├── MIMICClassificationDataset
    ├── COVIDDataset
    └── RSNADataset

BaseCollator (ABC)
├── MIMICContrastiveCollator
├── MIMICZeroShotCollator
├── MIMICSupervisedCollator
├── COVIDZeroShotCollator
├── COVIDSupervisedCollator
├── RSNAZeroShotCollator
└── RSNASupervisedCollator
```

## 🎯 Supported Combinations

### Dataset × Model Type × Task Type

| Dataset | Model Type | Task Types | Description |
|---------|------------|------------|-------------|
| **MIMIC** | MedCLIP | text2image | image - text |
| **MIMIC** | BiomedCLIP | text2image | image - text |
| **COVID** | MedCLIP | zeroshot, supervised | 2-class binary |
| **COVID** | BiomedCLIP | zeroshot, supervised | 2-class binary |
| **RSNA** | MedCLIP | zeroshot, supervised | 2-class binary |
| **RSNA** | BiomedCLIP | zeroshot, supervised | 2-class binary |

## 📊 Data Flow

### 1. **Dataset Creation**
```
DatasetFactory.create_dataset()
    ↓
Validate parameters
    ↓
Instantiate dataset class
    ↓
Load data files
    ↓
Setup transforms
    ↓
Return dataset instance
```

### 2. **DataLoader Creation**
```
DatasetFactory.create_dataloader()
    ↓
Create dataset
    ↓
Create collator
    ↓
Combine in DataLoader
    ↓
Return dataloader
```

### 3. **Batch Processing**
```
Dataset.__getitem__()
    ↓
Load & transform image
    ↓
Get labels
    ↓
Return (image, labels)
    ↓
Collator.__call__()
    ↓
Process batch of items
    ↓
Tokenize text (if needed)
    ↓
Return batch dict
```


## 🔌 Extension Points

### Adding new datasets

1. **Inherit from base class**:
```python
class NewDataset(BaseClassificationDataset):
    def _load_data(self): ...
    def _setup_dataset(self): ...
    def get_class_names(self): ...
    def get_class_prompts(self): ...
```

2. **Create collators**:
```python
class NewZeroShotCollator(BaseCollator): ...
class NewSupervisedCollator(BaseCollator): ...
```

3. **Register trong factory**:
```python
DatasetFactory.DATASET_REGISTRY['new'] = {
    'classification': NewDataset
}
DatasetFactory.COLLATOR_REGISTRY['new'] = {
    'zeroshot': NewZeroShotCollator,
    'supervised': NewSupervisedCollator
}
```

### Adding new models

1. **Update constants**:
```python
SUPPORTED_MODELS = ['medclip', 'biomedclip', 'newmodel']
```

2. **Update collators** handle new tokenization
3. **Update base classes** if need new functionality

### Add new Task Type

1. **Create new collator class**
2. **Register in factory**
3. **Update documentation**

### Research Workflow
```python
# 1. Exploratory analysis
factory.print_registry()

# 2. Quick prototyping  
loader = create_covid_dataloader('zeroshot', 'medclip')

# 3. Hyperparameter tuning
loader = DatasetFactory.create_dataloader(
    dataset_name='mimic',
    batch_size=optimal_batch_size,
    cls_prompts=tuned_prompts
)

# 4. Final evaluation
loader = DatasetFactory.create_dataloader(
    dataset_name='all_datasets',
    task_type='zeroshot',
    model_type='best_model'
)
```

### Production Workflow
```python
# 1. Configuration-driven
config = load_config('production.yaml')
loader = DatasetFactory.create_dataloader(**config)

# 2. Error handling
try:
    loader = DatasetFactory.create_dataloader(...)
except Exception as e:
    logger.error(f"Failed to create loader: {e}")
    fallback_loader = create_default_loader()

# 3. Monitoring
for batch in loader:
    monitor_batch_stats(batch)
    process_batch(batch)
```

---
