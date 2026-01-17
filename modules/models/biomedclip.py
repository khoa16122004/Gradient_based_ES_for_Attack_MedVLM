import os
import torch
import torch.nn as nn
from PIL import Image
import open_clip
from typing import Optional, Dict, List, Union
import numpy as np
from collections import defaultdict
from ..utils import constants
import torch.nn.functional as F
from .base import VisionLanguageModel
from collections import OrderedDict
from huggingface_hub import hf_hub_download


class BioMedCLIPModel(VisionLanguageModel):
    """
    BioMedCLIP model implementation using open_clip.
    This model provides text and vision encoding capabilities for medical images.
    """
    
    def __init__(
        self,
        model_name: str = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
        vision_pretrained = None,
        context_length: int = 256,
        checkpoint=None,
        mode_pretrained='scratch',
    ):
        """
        Initialize BioMedCLIP model.
        
        Args:
            model_name: Name/path of the pretrained model from HuggingFace Hub
            context_length: Maximum context length for text tokenization
            checkpoint: Optional checkpoint path to load model weights from
        """
        super().__init__()

        
        # Load model and preprocessing
        self.model, self.preprocess = open_clip.create_model_from_pretrained(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.context_length = context_length
        self.normalize_transform = constants.TENSOR_NORMALIZE_TRANSFORM['biomedclip']
        self.mode_pretrained = mode_pretrained
        # raise
        if checkpoint is not None:
            self.load_checkpoint(checkpoint)
        else:
            repo_id = "Woffy/Thesis_Pretrained_Medical_Moddel"
            if self.mode_pretrained == "scratch":
                file_name = "biomedclip.pth"
            elif self.mode_pretrained == "ssl":
                file_name = "biomedclip_ssl_finetuning.pth"
            elif self.mode_pretrained == "at":
                file_name = "biomedclip_AT.pth"
                
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=file_name,
                local_dir=".",          # lưu ngay thư mục hiện tại
                local_dir_use_symlinks=False  # QUAN TRỌNG: copy file thật, không tạo symlink
            )   

            ckpt = torch.load(local_path)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model_state_dict = ckpt["model_state_dict"]
            else:
                model_state_dict = ckpt  

            # model_state_dict = torch.load(local_path)['model_state_dict']
            model_state_dict = self._strip_prefix_from_state_dict(model_state_dict, 'model.')

            incompatible_keys = self.model.load_state_dict(model_state_dict, strict=True)
            print(f"Incompatible keys when load {local_path}: {incompatible_keys}")

        if vision_pretrained is not None:
            state_dict =torch.load(vision_pretrained)['model_state_dict']
            
            state_dict = self._strip_prefix_from_state_dict(state_dict)

            incompatible_keys = self.model.visual.load_state_dict(state_dict, strict=True)
            print(f"Incompatible keys {incompatible_keys}")
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Set model to eval mode by default
        self.model.eval()
    def _strip_prefix_from_state_dict(self, sd, prefixes=('visual.', 'module.', 'model.')):

        if isinstance(sd, dict) and 'state_dict' in sd and isinstance(sd['state_dict'], dict):
            sd = sd['state_dict']

        new_sd = OrderedDict()
        for k, v in sd.items():
            nk = k
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p):]
            new_sd[nk] = v
        return new_sd
    def load_checkpoint(self, checkpoint_path: str, strict: bool = False):
        """Load model weights from checkpoint."""
        if not checkpoint_path:
            print("No checkpoint path provided")
            return
            
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=strict)
            
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
                
            print(f'✅ Loaded model weights from: {checkpoint_path}')
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    def encode_text(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode text inputs to embeddings.
        
        Args:
            texts: Single text string or list of text strings
            normalize: Whether to normalize the embeddings
            
        Returns:
            Text embeddings tensor
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize texts
        text_tokens = self.tokenizer(texts, context_length=self.context_length)
        text_tokens = text_tokens.to(self.device)
        
        # Encode texts (no_grad only if model is in eval mode)
        if not self.model.training:
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
        else:
            text_features = self.model.encode_text(text_tokens)
        
        if normalize:
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def encode_image(
        self,
        images: Union[torch.Tensor, List[Image.Image], Image.Image],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode image inputs to embeddings.
        
        Args:
            images: Image tensor, PIL Image, or list of PIL Images
            normalize: Whether to normalize the embeddings
            
        Returns:
            Image embeddings tensor
        """
        # Handle different input types
        if isinstance(images, Image.Image):
            images = [images]
        
        if isinstance(images, list):
            # Process PIL images
            image_tensors = torch.stack([self.preprocess(img) for img in images])
            image_tensors = image_tensors.to(self.device)
        else:
            # Assume tensor input
            image_tensors = images.to(self.device)
        
        # Encode images (no_grad only if model is in eval mode)
        if not self.model.training:
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensors)
        else:
            image_features = self.model.encode_image(image_tensors)
        
        if normalize:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features
    
    def encode_posttransform_image( # truyền voad image tensor dwuodjc scale
        self,
        images: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode image inputs to embeddings.
        
        Args:
            images: Image tensor, PIL Image, or list of PIL Images
            normalize: Whether to normalize the embeddings
            
        Returns:
            Image embeddings tensor
        """
        # Handle different input types
        image_tensors = self.normalize_transform(images)
        
        image_features = self.model.encode_image(image_tensors)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features    
            
    def encode_pretransform_image( # truyền voad image tensor dwuodjc scale
        self,
        images: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode image inputs to embeddings.
        
        Args:
            images: Image tensor, PIL Image, or list of PIL Images
            normalize: Whether to normalize the embeddings
            
        Returns:
            Image embeddings tensor
        """
        # Handle different input types
        # images_ = torch.round(images * 255.0).clamp(0, 255)
        images_ = (images * 255.0).clamp(0, 255)

        # Resize to model input size
        image_tensors = F.interpolate(images_, size=(224, 224), mode="bilinear", align_corners=False)
        image_tensors = image_tensors / 255.0
        image_tensors = self.normalize_transform(image_tensors)
        


        
        image_features = self.model.encode_image(image_tensors)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features           

    
    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        texts: Optional[List[str]] = None,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        return_dict: bool = True,
        **kwargs
    ) -> Union[Dict, tuple]:
        """
        Forward pass for BioMedCLIP model.
        
        Args:
            images: Image tensors (alternative to pixel_values)
            texts: Text strings for tokenization
            pixel_values: Preprocessed image tensors
            input_ids: Tokenized text input ids
            attention_mask: Attention mask for text inputs
            return_loss: Whether to compute and return contrastive loss
            return_dict: Whether to return outputs as dictionary
            
        Returns:
            Dictionary or tuple containing image/text embeddings, logits, and optionally loss
        """
        # Handle image inputs
        if pixel_values is not None:
            images = pixel_values
        
        if images is not None:
            images = images.to(self.device)
            # Handle grayscale images
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)
        
        # Handle text inputs
        if texts is not None and input_ids is None:
            text_tokens = self.tokenizer(texts, context_length=self.context_length)
            input_ids = text_tokens
        
        if input_ids is not None:
            input_ids = input_ids.to(self.device)
        
        # Forward pass through the model
        if images is not None and input_ids is not None:
            # Both image and text inputs
            image_features = self.model.encode_image(images)
            text_features = self.model.encode_text(input_ids)
            
            # Get logit scale
            logit_scale = self.model.logit_scale.exp()
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity logits
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()
            
        elif images is not None:
            # Only image inputs
            image_features = self.model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = None
            logits_per_image = None
            logits_per_text = None
            
        elif input_ids is not None:
            # Only text inputs
            text_features = self.model.encode_text(input_ids)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_features = None
            logits_per_image = None
            logits_per_text = None
        else:
            raise ValueError("Either images or texts must be provided")
        
        loss = None
        if return_loss and logits_per_image is not None:
            loss = self.clip_loss(logits_per_text)
        
        if logits_per_image is not None:
            logits_per_image = logits_per_image.softmax(dim=-1)
            logits_per_text = logits_per_image.t()
        
        if return_dict:
            outputs = {
                'img_embeds': image_features,
                'text_embeds': text_features,
                'logits': logits_per_image,
                'logits_per_text': logits_per_text,
                'loss_value': loss
            }
            return outputs
        else:
            return image_features, text_features, logits_per_image
    
    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss."""
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.T)
        return (caption_loss + image_loss) / 2.0
    
    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss for contrastive learning."""
        return nn.functional.cross_entropy(
            logits, torch.arange(len(logits), device=logits.device)
        )
    

class BioMedCLIPClassifier(nn.Module):
    """
    Zero-shot classifier using BioMedCLIP for medical image classification.
    """
    
    def __init__(
        self,
        biomedclip_model: BioMedCLIPModel,
        ensemble: bool = False,
        templates: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize BioMedCLIP classifier.
        
        Args:
            biomedclip_model: Pretrained BioMedCLIP model
            ensemble: Whether to use prompt ensembling
            templates: List of prompt templates (for compatibility with factory)
            **kwargs: Additional arguments (for compatibility)
        """
        super().__init__()
        self.model = biomedclip_model
        self.ensemble = ensemble
        self.templates = templates if templates else ["a medical image of {}"]
    
    def create_text_prompts(
        self,
        class_names: List[str]
    ) -> List[str]:
        """
        Create text prompts for classes using templates.
        
        Args:
            class_names: List of class names
            
        Returns:
            List of text prompts
        """
        prompts = []
        for class_name in class_names:
            for template in self.templates:
                prompts.append(template.format(class_name))
        
        return prompts
    
    def classify_with_templates(
        self,
        pixel_values: torch.Tensor,
        class_names: List[str],
        **kwargs
    ) -> Dict:
        """
        Classify images using templates to generate prompts.
        
        Args:
            pixel_values: Preprocessed image tensors
            class_names: List of class names
            
        Returns:
            Dictionary containing logits and class names
        """
        pixel_values = pixel_values.to(self.model.device)
        class_similarities = []
        
        for cls_name in class_names:
            # Generate prompts for this class using templates
            prompts = []
            for template in self.templates:
                prompts.append(template.format(cls_name))
            
            # Encode all prompts for this class
            template_similarities = []
            for prompt in prompts:
                # Encode text and image
                text_features = self.model.encode_text(prompt)
                image_features = self.model.encode_image(pixel_values)
                
                # Compute similarity
                similarity = torch.cosine_similarity(image_features, text_features, dim=-1)
                template_similarities.append(similarity)
            
            # Aggregate similarities across templates
            template_similarities = torch.stack(template_similarities, dim=1)  # [batch, num_templates]
            
            if self.ensemble and len(self.templates) > 1:
                # Average across templates for ensembling
                cls_sim = torch.mean(template_similarities, dim=1)
            else:
                # Take max similarity across templates
                cls_sim = torch.max(template_similarities, dim=1)[0]
            
            class_similarities.append(cls_sim)
        
        # Stack similarities for all classes
        class_similarities = torch.stack(class_similarities, dim=1)  # [batch, num_classes]
        class_similarities = class_similarities.softmax(dim=-1)
        
        outputs = {
            'logits': class_similarities,
            'class_names': class_names,
        }
        
        return outputs
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        prompt_inputs: Dict[str, Dict],
        **kwargs
    ) -> Dict:
        """
        Forward pass for zero-shot classification.
        
        Args:
            pixel_values: Preprocessed image tensors
            prompt_inputs: Dictionary mapping class names to their text inputs
            
        Returns:
            Dictionary containing logits and class names
        """
        pixel_values = pixel_values.to(self.model.device)
        class_similarities = []
        class_names = []
        
        for cls_name, cls_text in prompt_inputs.items():
            inputs = {'pixel_values': pixel_values}
            
            # Handle text inputs
            if isinstance(cls_text, list):
                # Text strings -> model will tokenize
                inputs['texts'] = cls_text
            elif isinstance(cls_text, dict):
                # Tokenized inputs (for compatibility)
                if 'input_ids' in cls_text:
                    inputs['input_ids'] = cls_text['input_ids'].to(self.model.device)
                if 'attention_mask' in cls_text:
                    inputs['attention_mask'] = cls_text['attention_mask'].to(self.model.device)
            else:
                # Tensor inputs (for compatibility)
                inputs['input_ids'] = cls_text.to(self.model.device)
            
            # Get model outputs
            outputs = self.model(**inputs)
            logits = outputs['logits']
            
            # Aggregate similarities
            if self.ensemble:
                cls_sim = torch.mean(logits, 1)  # Prompt ensembling
            else:
                cls_sim = torch.max(logits, 1)[0]  # Take max similarity
            
            class_similarities.append(cls_sim)
            class_names.append(cls_name)
        
        class_similarities = torch.stack(class_similarities, 1)
        class_similarities = class_similarities.softmax(dim=-1)
        
        outputs = {
            'logits': class_similarities,
            'class_names': class_names,
        }
        
        return outputs

if __name__ == '__main__':
    import torch
    from PIL import Image
    import numpy as np
    
    from ..dataset.rsna import RSNADataset, create_rsna_dataloader
    from ..utils.constants import (
        MODEL_TRANSFORMS,
        RSNA_TASKS
    )
    
    print("=" * 80)
    print("🧪 Testing BioMedCLIP Model and Classifier")
    print("=" * 80)
    
    # ============================================================================
    # 1. TEST BIOMedCLIP MODEL
    # ============================================================================
    print("\n" + "=" * 80)
    print("1️⃣  Testing BioMedCLIPModel")
    print("=" * 80)
    
    try:
        # Initialize model
        print("\n📦 Initializing BioMedCLIPModel...")
        model = BioMedCLIPModel(model_name='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        print(f"✅ Model initialized on device: {model.device}")
        print(f"   Context length: {model.context_length}")
        
        # Test encode_text
        print("\n📝 Testing encode_text()...")
        text1 = "a medical image of pneumonia"
        text2 = ["a chest X-ray showing pneumonia", "a normal chest X-ray"]
        
        # Single text
        text_features_single = model.encode_text(text1)
        print(f"✅ Single text encoding: shape {text_features_single.shape}")
        
        # Multiple texts
        text_features_multi = model.encode_text(text2)
        print(f"✅ Multiple texts encoding: shape {text_features_multi.shape}")
        
        # Test encode_image
        print("\n🖼️  Testing encode_image()...")
        # Create dummy image tensor
        dummy_image_tensor = torch.randn(2, 3, 224, 224)
        image_features_tensor = model.encode_image(dummy_image_tensor)
        print(f"✅ Image tensor encoding: shape {image_features_tensor.shape}")
        
        # Test forward with texts only
        print("\n🔄 Testing forward() with texts only...")
        outputs_text = model.forward(texts=text2, return_dict=True)
        print(f"✅ Forward with texts:")
        print(f"   Text embeddings shape: {outputs_text['text_embeds'].shape}")
        print(f"   Logits: {outputs_text['logits']}")
        
        # Test forward with images only
        print("\n🔄 Testing forward() with images only...")
        outputs_img = model.forward(images=dummy_image_tensor, return_dict=True)
        print(f"✅ Forward with images: image embeddings shape {outputs_img['img_embeds'].shape}")
        
        # Test forward with both images and texts
        print("\n🔄 Testing forward() with images and texts...")
        outputs_both = model.forward(
            images=dummy_image_tensor,
            texts=text2,
            return_dict=True
        )
        print(f"✅ Forward with images and texts:")
        print(f"   Image embeddings shape: {outputs_both['img_embeds'].shape}")
        print(f"   Text embeddings shape: {outputs_both['text_embeds'].shape}")
        print(f"   Logits shape: {outputs_both['logits'].shape}")
        print(f"   Logits (softmax applied): sum per image = {outputs_both['logits'].sum(dim=-1)}")
        
        # Test forward with pixel_values
        print("\n🔄 Testing forward() with pixel_values...")
        outputs_pixel = model.forward(
            pixel_values=dummy_image_tensor,
            texts=text2,
            return_dict=True
        )
        print(f"✅ Forward with pixel_values and texts: logits shape {outputs_pixel['logits'].shape}")
        
    except Exception as e:
        print(f"❌ Error testing BioMedCLIPModel: {e}")
        import traceback
        traceback.print_exc()
    
    # ============================================================================
    # 2. TEST BIOMedCLIP CLASSIFIER
    # ============================================================================
    print("\n" + "=" * 80)
    print("2️⃣  Testing BioMedCLIPClassifier")
    print("=" * 80)
    
    try:
        # Initialize classifier
        print("\n📦 Initializing BioMedCLIPClassifier...")
        classifier = BioMedCLIPClassifier(
            biomedclip_model=model,
            ensemble=False,
            templates=["a chest X-ray showing {}"]
        )
        print(f"✅ Classifier initialized")
        print(f"   Templates: {classifier.templates}")
        print(f"   Ensemble: {classifier.ensemble}")
        
        # Test create_text_prompts
        print("\n📝 Testing create_text_prompts()...")
        class_names = RSNA_TASKS  # ['Pneumonia', 'Normal']
        prompts = classifier.create_text_prompts(class_names)
        print(f"✅ Created prompts for {len(class_names)} classes:")
        for i, prompt in enumerate(prompts):
            print(f"   {i+1}. {prompt}")
        
        # Test classify_with_templates
        print("\n🎯 Testing classify_with_templates()...")
        test_images = torch.randn(2, 3, 224, 224)
        outputs_template = classifier.classify_with_templates(
            pixel_values=test_images,
            class_names=class_names
        )
        print(f"✅ classify_with_templates:")
        print(f"   Logits shape: {outputs_template['logits'].shape}")
        print(f"   Class names: {outputs_template['class_names']}")
        print(f"   Logits (softmax): sum per image = {outputs_template['logits'].sum(dim=-1)}")
        # Get predictions
        predictions = outputs_template['logits'].argmax(dim=-1)
        print(f"   Predictions: {[class_names[p] for p in predictions]}")
        
        # Test forward with prompt_inputs (simulating collator output)
        print("\n🎯 Testing forward() with prompt_inputs...")
        # Simulate prompt_inputs from collator (text strings)
        prompt_inputs = {
            'Pneumonia': ['a chest X-ray showing Pneumonia', 'chest X-ray with pneumonia'],
            'Normal': ['a chest X-ray showing Normal', 'normal chest X-ray']
        }
        outputs_forward = classifier.forward(
            pixel_values=test_images,
            prompt_inputs=prompt_inputs
        )
        print(f"✅ forward with prompt_inputs:")
        print(f"   Logits shape: {outputs_forward['logits'].shape}")
        print(f"   Class names: {outputs_forward['class_names']}")
        print(f"   Logits (softmax): sum per image = {outputs_forward['logits'].sum(dim=-1)}")
        predictions = outputs_forward['logits'].argmax(dim=-1)
        print(f"   Predictions: {[outputs_forward['class_names'][p] for p in predictions]}")
        
    except Exception as e:
        print(f"❌ Error testing BioMedCLIPClassifier: {e}")
        import traceback
        traceback.print_exc()
    
    # ============================================================================
    # 3. TEST WITH RSNA DATASET AND DATALOADER
    # ============================================================================
    print("\n" + "=" * 80)
    print("3️⃣  Testing with RSNA Dataset and DataLoader")
    print("=" * 80)
    
    try:
        # Create dataset
        print("\n📂 Creating RSNA dataset...")
        dataset = RSNADataset(
            data_root='../local_data',
            split='test',
            model_type='biomedclip',
            transform=MODEL_TRANSFORMS['biomedclip']
        )
        print(f"✅ Dataset created: {len(dataset)} samples")
        print(f"   Class names: {dataset.get_class_names()}")
        
        # Test single sample
        if len(dataset) > 0:
            print("\n📊 Testing single sample from dataset...")
            img, labels = dataset[0]
            print(f"✅ Sample loaded:")
            print(f"   Image shape: {img.shape}")
            print(f"   Labels: {labels}")
            
            # Test with classifier
            img_batch = img.unsqueeze(0)  # Add batch dimension
            class_names = dataset.get_class_names()
            outputs = classifier.classify_with_templates(
                pixel_values=img_batch,
                class_names=class_names
            )
            print(f"   Classification logits shape: {outputs['logits'].shape}")
            pred = outputs['logits'].argmax(dim=-1)[0].item()
            print(f"   Prediction: {class_names[pred]} (confidence: {outputs['logits'][0][pred]:.4f})")
            print(f"   Ground truth: {class_names[labels.argmax().item()]}")
        
        # Test with DataLoader
        print("\n🔄 Creating DataLoader...")
        dataloader = create_rsna_dataloader(
            data_root='../local_data',
            split='test',
            model_type='biomedclip',
            task_type='zeroshot',
            batch_size=2,
            shuffle=False,
            num_workers=0
        )
        print(f"✅ DataLoader created")
        
        # Test one batch
        print("\n🔄 Testing one batch from DataLoader...")
        for batch in dataloader:
            pixel_values = batch['pixel_values']
            prompt_inputs = batch['prompt_inputs']
            labels = batch['labels']
            class_names = batch['class_names']
            
            print(f"✅ Batch loaded:")
            print(f"   Pixel values shape: {pixel_values.shape}")
            print(f"   Labels shape: {labels.shape}")
            print(f"   Class names: {class_names}")
            print(f"   Prompt inputs keys: {list(prompt_inputs.keys())}")
            
            # Test with classifier
            outputs = classifier.forward(
                pixel_values=pixel_values,
                prompt_inputs=prompt_inputs
            )
            print(f"   Classification outputs:")
            print(f"     Logits shape: {outputs['logits'].shape}")
            print(f"     Logits (softmax): sum = {outputs['logits'].sum(dim=-1)}")
            
            # Get predictions
            predictions = outputs['logits'].argmax(dim=-1)
            print(f"     Predictions: {[outputs['class_names'][p.item()] for p in predictions]}")
            
            # Calculate accuracy for this batch
            gt_classes = labels.argmax(dim=-1)
            accuracy = (predictions == gt_classes).float().mean()
            print(f"     Batch accuracy: {accuracy.item():.4f}")
            
            break  # Only test first batch
        
    except Exception as e:
        print(f"❌ Error testing with RSNA dataset: {e}")
        import traceback
        traceback.print_exc()
    
    # ============================================================================
    # SUMMARY
    # ============================================================================
    print("\n" + "=" * 80)
    print("✅ Testing completed!")
    print("=" * 80)


