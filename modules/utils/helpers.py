import random
from typing import Dict, List, Optional
from itertools import product

from .constants import COVID_CLASS_PROMPTS, RSNA_CLASS_PROMPTS
from .logging_config import get_logger
from torch import nn
import numpy as np
import torch
import open_clip

logger = get_logger(__name__)

def load_open_clip_model(args, device):
    """
    Load open_clip + optional custom checkpoint.
    """
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name,
        pretrained="openai",
    )
    model.to(device)

    if args.pretrained is not None and len(args.pretrained) > 0:
        print(f"Loading custom checkpoint from: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location="cpu")
        state_dict = _strip_prefix_from_state_dict(ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

    model.eval()
    tokenizer = open_clip.get_tokenizer(args.model_name)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    return model, preprocess, tokenizer


def _extract_label(dict_label):
    for i, (class_name, is_gt) in enumerate(dict_label.items()):
        if is_gt == 1:
            return i


def _strip_prefix_from_state_dict(sd, prefixes=('visual.')):

    if isinstance(sd, dict) and 'state_dict' in sd and isinstance(sd['state_dict'], dict):
        sd = sd['state_dict']

    new_sd = OrderedDict()
    for k, v in sd.items():
        nk = k
        # Bỏ lần lượt các prefix nếu có
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        new_sd[nk] = v
    return new_sd

def setup_seed(seed: int = 42):
    """
    Setup seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_covid_class_prompts(
    class_prompts: Optional[Dict] = None,
    n: int = 5
) -> Dict[str, List[str]]:
    """
    Generate class prompts cho COVID tasks
    
    Args:
        class_prompts: Dictionary of class prompt templates
        n: Number of prompts per class
        
    Returns:
        Dictionary {class_name: [list_of_prompts]}
    """
    if class_prompts is None:
        class_prompts = COVID_CLASS_PROMPTS
        
    generated_prompts = {}
    
    for class_name, prompt_dict in class_prompts.items():
        prompts = []
        
        if class_name == 'COVID':
            # Generate combinations cho COVID
            adjective_list = prompt_dict.get('adjective', [''])
            description_list = prompt_dict.get('description', [''])
            subtype_list = prompt_dict.get('subtype', [''])
            location_list = prompt_dict.get('location', [''])
            
            # Create combinations
            combinations = list(product(adjective_list, description_list, subtype_list, location_list))
            
            for adj, desc, subtype, loc in combinations:
                prompt_parts = [part for part in [adj, desc, subtype, loc] if part.strip()]
                if prompt_parts:
                    prompt = ' '.join(prompt_parts).strip()
                    if prompt and prompt not in prompts:
                        prompts.append(prompt)
                        
        elif class_name == 'Normal':
            # Simple prompts cho Normal
            adjective_list = prompt_dict.get('adjective', ['normal'])
            description_list = prompt_dict.get('description', ['chest'])
            subtype_list = prompt_dict.get('subtype', ['x-ray'])
            
            combinations = list(product(adjective_list, description_list, subtype_list))
            
            for adj, desc, subtype in combinations:
                prompt_parts = [part for part in [adj, desc, subtype] if part.strip()]
                if prompt_parts:
                    prompt = ' '.join(prompt_parts).strip()
                    if prompt and prompt not in prompts:
                        prompts.append(prompt)
                        
        # Fallback nếu không có prompts
        if not prompts:
            if class_name == 'COVID':
                prompts = ['covid pneumonia', 'coronavirus infection', 'covid-19 findings']
            else:
                prompts = ['normal chest', 'clear lungs', 'no abnormalities']
                
        # Limit số lượng prompts
        if len(prompts) > n:
            prompts = random.sample(prompts, n)
        elif len(prompts) < n:
            while len(prompts) < n:
                prompts.extend(prompts[:n - len(prompts)])
                
        generated_prompts[class_name] = prompts[:n]
        
    return generated_prompts


def generate_rsna_class_prompts(
    class_prompts: Optional[Dict] = None,
    n: int = 5
) -> Dict[str, List[str]]:
    """
    Generate class prompts cho RSNA tasks
    
    Args:
        class_prompts: Dictionary of class prompt templates
        n: Number of prompts per class
        
    Returns:
        Dictionary {class_name: [list_of_prompts]}
    """
    if class_prompts is None:
        class_prompts = RSNA_CLASS_PROMPTS
        
    generated_prompts = {}
    
    for class_name, prompt_dict in class_prompts.items():
        prompts = []
        
        if class_name == 'Pneumonia':
            # Generate combinations cho Pneumonia
            adjective_list = prompt_dict.get('adjective', [''])
            subtype_list = prompt_dict.get('subtype', ['pneumonia'])
            location_list = prompt_dict.get('location', [''])
            
            combinations = list(product(adjective_list, subtype_list, location_list))
            
            for adj, subtype, loc in combinations:
                prompt_parts = [part for part in [adj, subtype, loc] if part.strip()]
                if prompt_parts:
                    prompt = ' '.join(prompt_parts).strip()
                    if prompt and prompt not in prompts:
                        prompts.append(prompt)
                        
        elif class_name == 'Normal':
            # Simple prompts cho Normal
            adjective_list = prompt_dict.get('adjective', ['normal'])
            description_list = prompt_dict.get('description', ['chest'])
            subtype_list = prompt_dict.get('subtype', ['x-ray'])
            
            combinations = list(product(adjective_list, description_list, subtype_list))
            
            for adj, desc, subtype in combinations:
                prompt_parts = [part for part in [adj, desc, subtype] if part.strip()]
                if prompt_parts:
                    prompt = ' '.join(prompt_parts).strip()
                    if prompt and prompt not in prompts:
                        prompts.append(prompt)
                        
        # Fallback nếu không có prompts
        if not prompts:
            if class_name == 'Pneumonia':
                prompts = ['pneumonia', 'lung infection', 'bacterial pneumonia']
            else:
                prompts = ['normal chest', 'clear lungs', 'no abnormalities']
                
        # Limit số lượng prompts
        if len(prompts) > n:
            prompts = random.sample(prompts, n)
        elif len(prompts) < n:
            while len(prompts) < n:
                prompts.extend(prompts[:n - len(prompts)])
                
        generated_prompts[class_name] = prompts[:n]
        
    return generated_prompts


def process_class_prompts(cls_prompts: Dict[str, List[str]]) -> Dict[str, Dict]:
    """
    Process class prompts cho MedCLIP tokenization
    
    Args:
        cls_prompts: Dictionary {class_name: [prompts]}
        
    Returns:
        Dictionary {class_name: tokenized_inputs}
    """
    from transformers import AutoTokenizer
    from .constants import BERT_TYPE
    
    tokenizer = AutoTokenizer.from_pretrained(BERT_TYPE)
    tokenizer.model_max_length = 77
    
    prompt_inputs = {}
    for class_name, prompts in cls_prompts.items():
        text_inputs = tokenizer(
            prompts,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        prompt_inputs[class_name] = text_inputs
        
    return prompt_inputs


def process_class_prompts_for_tuning(
    cls_prompts: Dict[str, List[str]],
    n_context: int = 16,
    class_specific_context: bool = False
) -> Dict[str, Dict]:
    """
    Process class prompts cho prompt tuning
    
    Args:
        cls_prompts: Dictionary {class_name: [prompts]}
        n_context: Number of context tokens
        class_specific_context: Có dùng class-specific context không
        
    Returns:
        Dictionary {class_name: processed_inputs}
    """
    # Placeholder implementation
    # Actual implementation sẽ depend on prompt tuning architecture
    processed_prompts = {}
    
    for class_name, prompts in cls_prompts.items():
        # Add context tokens
        context_tokens = ['[CTX]'] * n_context
        
        processed_class_prompts = []
        for prompt in prompts:
            if class_specific_context:
                full_prompt = context_tokens + [f'[CLASS_{class_name}]'] + prompt.split()
            else:
                full_prompt = context_tokens + prompt.split()
            processed_class_prompts.append(' '.join(full_prompt))
            
        processed_prompts[class_name] = processed_class_prompts
        
    return processed_prompts


def validate_dataset_config(dataset_name: str, config: Dict) -> bool:
    """
    Validate dataset configuration
    
    Args:
        dataset_name: Name of dataset
        config: Dataset configuration
        
    Returns:
        True if valid, False otherwise
    """
    required_keys = ['tasks', 'class_prompts', 'data_files', 'mode']
    
    for key in required_keys:
        if key not in config:
            print(f"Missing required key '{key}' in {dataset_name} config")
            return False
            
    # Validate mode
    valid_modes = ['multiclass', 'multilabel', 'binary']
    if config['mode'] not in valid_modes:
        print(f"Invalid mode '{config['mode']}' for {dataset_name}. Must be one of {valid_modes}")
        return False
        
    # Validate tasks và class_prompts consistency
    tasks = config['tasks']
    class_prompts = config['class_prompts']
    
    for task in tasks:
        if task not in class_prompts:
            print(f"Warning: Task '{task}' not found in class_prompts for {dataset_name}")
            
    return True


def get_dataset_info(dataset_name: str) -> Dict:
    """
    Get thông tin về dataset
    
    Args:
        dataset_name: Name of dataset ('mimic', 'covid', 'rsna')
        
    Returns:
        Dictionary chứa dataset info
    """
    from .constants import DATASET_CONFIGS
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")
        
    config = DATASET_CONFIGS[dataset_name]
    print(config)
    return {
        'name': dataset_name,
        'tasks': config['tasks'],
        'num_classes': len(config['tasks']),
        'mode': config['mode'],
        'available_splits': list(config['data_files'].keys()),
        'class_prompts_available': len(config['class_prompts'])
    }


def print_dataset_summary():
    """
    In summary của tất cả datasets
    """
    from .constants import DATASET_CONFIGS
    
    logger.info("📊 Medical Datasets Summary")
    logger.info("=" * 50)
    
    for dataset_name in DATASET_CONFIGS:
        info = get_dataset_info(dataset_name)
        logger.info(f"\n🏥 {dataset_name.upper()} Dataset:")
        logger.info(f"  Classes: {info['num_classes']} ({info['mode']})")
        logger.info(f"  Tasks: {info['tasks']}")
        logger.info(f"  Splits: {info['available_splits']}")
        logger.info(f"  Prompts: {info['class_prompts_available']} classes")
        
    logger.info("\n✅ Summary completed!")


if __name__ == "__main__":
    # Demo các functions
    logger.info("🔧 Utils Demo")
    
    # Test prompt generation
    logger.info("\n📝 Testing prompt generation:")
    
    # COVID prompts
    covid_prompts = generate_covid_class_prompts(n=3)
    logger.info(f"COVID prompts: {covid_prompts}")
    
    # RSNA prompts
    rsna_prompts = generate_rsna_class_prompts(n=3)
    logger.info(f"RSNA prompts: {rsna_prompts}")
    
    # Dataset summary
    print_dataset_summary()