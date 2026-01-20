# Vision-Language Models for Medical Image Analysis

This module provides implementations of various vision-language models for medical image analysis, including MedCLIP and BioMedCLIP.

## 📁 Structure

```
models/
├── __init__.py           # Module exports
├── model.py              # Base classes for all models
├── factory.py            # Factory for model creation
├── medclip.py            # MedCLIP implementation
├── biomedclip.py         # BioMedCLIP implementation  
├── example_usage.py      # Usage examples
└── README.md             # This file
```

## 🏗️ Architecture

### Base Classes (model.py)

- **`BaseVisionLanguageModel`**: Abstract base class for all vision-language models
- **`BaseClassifier`**: Abstract base class for classifiers
- **`BaseZeroShotClassifier`**: Base class for zero-shot classification
- **`BaseSupervisedClassifier`**: Base class for supervised classification
- **`BasePromptLearner`**: Base class for prompt learning methods

### Model Implementations

#### MedCLIP (medclip.py)
- `MedCLIPModel`: Full MedCLIP model with text and vision encoders
- `MedCLIPTextModel`: Text encoder using BioClinicalBERT
- `MedCLIPVisionModel`: Vision encoder using ResNet50
- `MedCLIPVisionModelViT`: Vision encoder using Vision Transformer
- `PromptClassifier`: Zero-shot classifier
- `SuperviseClassifier`: Supervised classifier with linear head
- `PromptTuningClassifier`: Prompt tuning implementation

#### BioMedCLIP (biomedclip.py)
- `BioMedCLIPModel`: Full BioMedCLIP model using OpenCLIP
- `BioMedCLIPClassifier`: Zero-shot classifier
- `BioMedCLIPFeatureExtractor`: Feature extractor for supervised learning

### Factory Pattern (factory.py)

The `ModelFactory` class provides a unified interface for creating models and classifiers.

## 🚀 Quick Start

### Basic Usage

```python
from models import ModelFactory, create_medclip, create_biomedclip

# Create models using factory
medclip = ModelFactory.create_model(model_type='medclip', variant='base')
biomedclip = create_biomedclip()

# Encode images and text
image_features = biomedclip.encode_image(images)
text_features = biomedclip.encode_text(texts)

# Compute similarity
similarity = (image_features @ text_features.t()).softmax(dim=-1)
```

### Zero-Shot Classification

```python
# Create zero-shot classifier
classifier = ModelFactory.create_zeroshot_classifier(
    model_type='biomedclip',
    class_names=['Normal', 'Pneumonia', 'COVID-19'],
    templates=['a chest x-ray showing {}'],
    ensemble=True
)

# Classify images
outputs = classifier(pixel_values=images)
predictions = outputs['logits'].argmax(dim=1)
```

### Supervised Classification

```python
# Create supervised classifier
classifier = ModelFactory.create_supervised_classifier(
    model_type='medclip',
    num_classes=14,
    task_mode='multilabel',
    freeze_encoder=True
)

# Train the classifier
outputs = classifier(pixel_values=images, labels=labels, return_loss=True)
loss = outputs['loss_value']
```

## 📊 Model Comparison

| Feature | MedCLIP | BioMedCLIP |
|---------|---------|------------|
| Text Encoder | BioClinicalBERT | PubMedBERT |
| Vision Encoder | ResNet50/ViT | ViT-B/16 |
| Pre-training Data | MIMIC-CXR | PubMed articles + medical images |
| Context Length | 512 | 256 |
| Zero-shot Support | ✅ | ✅ |
| Supervised Support | ✅ | ✅ |
| Prompt Learning | ✅ | ❌ |

## 🔧 Factory Methods

### Creating Models

```python
# Method 1: Using ModelFactory
model = ModelFactory.create_model(
    model_type='medclip',      # 'medclip' or 'biomedclip'
    variant='base',             # Model variant
    checkpoint='path/to/ckpt',  # Optional checkpoint
    device='cuda',              # Device
    pretrained=True             # Load pretrained weights
)

# Method 2: Using convenience functions
medclip = create_medclip(variant='base', pretrained=True)
biomedclip = create_biomedclip(checkpoint='path/to/ckpt')
```

### Creating Classifiers

```python
# Zero-shot classifier
zs_classifier = ModelFactory.create_classifier(
    model=model,                # Pre-initialized model
    task_type='zeroshot',
    class_names=['class1', 'class2'],
    ensemble=True
)

# Supervised classifier
sup_classifier = ModelFactory.create_classifier(
    model=model,
    task_type='supervised',
    num_classes=10,
    freeze_encoder=True
)
```

## 🎯 Task Types

### Zero-Shot Classification
- No training required
- Define class names and templates
- Supports prompt ensembling

### Supervised Classification
- Requires training on labeled data
- Supports binary, multiclass, and multilabel
- Option to freeze encoder for faster training

### Prompt Learning
- Learn task-specific prompts
- Available for MedCLIP only
- Supports class-specific contexts

## 📝 Examples

Run the example script to see all features:

```bash
python example_usage.py
```

This will demonstrate:
- Basic model creation and usage
- Zero-shot classification
- Supervised classification
- Model comparison
- Custom integration

## 🔗 Integration with Datasets

The models are designed to work seamlessly with the dataset module:

```python
from dataset import DatasetFactory
from models import ModelFactory

# Create dataset and dataloader
dataloader = DatasetFactory.create_dataloader(
    dataset_name='covid',
    task_type='zeroshot',
    model_type='biomedclip'
)

# Create corresponding model
model = ModelFactory.create_model(model_type='biomedclip')

# Process batches
for batch in dataloader:
    outputs = model(**batch)
```

## 📚 References

- MedCLIP: [Paper](https://arxiv.org/abs/2210.10163) | [GitHub](https://github.com/RyanWangZf/MedCLIP)
- BioMedCLIP: [HuggingFace](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)

## 🤝 Contributing

To add a new model:

1. Create a new file in `models/` with your implementation
2. Inherit from appropriate base classes in `model.py`
3. Register your model in `factory.py`
4. Add exports to `__init__.py`
5. Update this README

## 📄 License

See the main project LICENSE file.
