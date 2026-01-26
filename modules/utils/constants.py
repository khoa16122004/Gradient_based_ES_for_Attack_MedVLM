from torchvision import transforms

RANDOM_STATE = 22520691
TRAIN_RATIO = 0.8
TEST_RATIO = 0.1
VAL_RATIO = 0.1

# Model types
BERT_TYPE = 'emilyalsentzer/Bio_ClinicalBERT'
VIT_TYPE = 'microsoft/swin-tiny-patch4-window7-224'
BIOMEDCLIP_MODEL = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'

# Image processing
IMG_SIZE = 224
IMG_MEAN = 0.5862785803043838 # medclip
IMG_STD = 0.27950088968644304 # MEDCLIP

BIOMEDCLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
BIOMEDCLIP_STD = [0.26862954, 0.26130258, 0.27577711]


# Entrep
ENTREP_MEAN = [0.485, 0.456, 0.406]
ENTREP_STD = [0.229, 0.224, 0.225]


#ViT
VIT_MEAN = (0.48145466, 0.4578275, 0.40821073) 
VIT_STD = (0.26862954, 0.26130258, 0.27577711)


# Dataset paths
DEFAULT_DATA_ROOT = 'local_data'

# COVID Dataset
COVID_TASKS = [
    'COVID',
    'Normal',
]

# MIMIC Dataset
MIMIC_TASKS = [
    'Abnormal',
    'Normal',
]

# RSNA Dataset  
RSNA_TASKS = [
    'Pneumonia',
    'Normal',
]

ENTREP_TASKS = [
    'vocal-throat',
    'nose',
    'ear',
    'throat'
]

MIMIC_CLASS_PROMPTS = {
    'Abnormal': {
        'adjective': ['abnormal', 'pathological', 'irregular'],
        'description': ['findings', 'opacity', 'infiltrate'],
        'subtype': ['consolidation', 'effusion', 'pneumonia'],
        'location': ['in lung', 'bilateral', 'unilateral'],
    },
    
    'Normal': {
        'adjective': ['clear', 'normal', 'healthy'],
        'description': ['chest', 'lungs', 'findings'],
        'subtype': ['x-ray', 'radiograph', 'image'],
        'location': ['', 'bilateral', 'throughout'],
    },

}

COVID_CLASS_PROMPTS = {
    'COVID': {
        'adjective': ['patchy', 'confluent'],
        'description': ['ground glass'],
        'subtype': ['opacity', 'consolidation'],
        'location': ['in peripheral', 'in mid', 'in lower'],
    },
    'Normal': {
        'adjective': ['clear', 'normal', 'healthy'],
        'description': ['chest', 'lungs', 'findings'],
        'subtype': ['x-ray', 'radiograph', 'image'],
        'location': ['', 'bilateral', 'throughout'],
    }
}

# RSNA_CLASS_PROMPTS = {
#     'Pneumonia': {
#         'adjective': ['round', 'early', 'focal', 'multifocal', 'small', ''],
#         'subtype': ['bacterial', 'viral', 'mycoplasma', ''],
#         "location": [
#             "at the mid lung zone",
#             "at the upper lung zone", 
#             "at the right lung zone",
#             "at the left lung zone",
#             "at the lung bases",
#             "at the right lung base",
#             "at the left lung base",
#             "at the bilateral lung bases",
#             "at the left lower lobe",
#             "at the right lower lobe",
#             "at the left middle lobe",
#             "at the right middle lobe",
#             ""
#         ]
#     },
#     'Normal': {
#         'adjective': ['clear', 'normal', 'healthy'],
#         'description': ['chest', 'lungs', 'findings'], 
#         'subtype': ['x-ray', 'radiograph', 'image'],
#         'location': ['', 'bilateral', 'throughout'],
#     }
# }

# RSNA_CLASS_PROMPTS = {
#     'Pneumonia': [
#         "lung opacities",
#         "focal consolidation",
#         "pulmonary infiltrates",
#         "pleural effusion",
#     ],
#     'Normal': [
#         "lungs clear",
#         "no lung opacity",
#         "normal chest radiograph",
#         "no pulmonary"
#     ]
# }

RSNA_CLASS_PROMPTS = {
    'Pneumonia': [
        "lung opacities",
        "focal consolidation",
        "patchy airspace opacity",
        "increased lung density",
        "pulmonary infiltrates",
        "interstitial opacity",
        "alveolar opacity",
        "pleural effusion",
        "parenchymal opacity"
    ],
    'Normal': [
        "lungs clear",
        "no focal consolidation",
        "no pleural effusion",
        "no pulmonary edema",
        "no lung opacity",
        "normal chest radiograph",
        "no pulmonary"
    ]
}




# ENTREP_CLASS_PROMPTS = {
#     'vocal-throat': [
#         'endoscopic image of vocal-throat',
#         'medical image showing vocal-throat',
#         'clinical image of vocal-throat'
#     ],
#     'nose': [
#         'endoscopic image of nose',
#         'medical image showing nose',
#         'clinical image of nose'
#     ],
#     'ear': [
#         'endoscopic image of nose',
#         'medical image showing nose',
#         'clinical image of nose'
#     ],
#     'throat': [
#         'endoscopic image of throat',
#         'medical image showing throat',
#         'clinical image of throat'
#     ],
# }

ENTREP_CLASS_PROMPTS = {
    'vocal-throat': [
        "vocal cords visualized",
        "vocal cord edema",
        "vocal cord erythema",
        "vocal cord thickening",
        "glottic edema",
        "supraglottic swelling",
        "interarytenoid edema",
        "tonsillar enlargement",
        "oropharyngeal mucosa normal",
        "no tonsillar exudate"
    ],
    'nose': [
        "nasal mucosa erythema",
        "nasal turbinate hypertrophy",
        "nasal discharge",
        "septal deviation",
        "pale nasal mucosa",
        "mucus in nasal cavity",
        "inferior turbinate enlarged",
        "nasal airway patent",
        "visualized nasal cavity",
        "no purulent discharge"
    ],
    'ear': [
        "tympanic membrane intact",
        "tympanic membrane erythema",
        "tympanic membrane bulging",
        "auditory canal edema",
        "ear canal discharge",
        "cerumen present",
        "external auditory canal clear",
        "tympanic membrane thickened",
        "no middle ear effusion",
        "auditory canal normal"
    ],
    'throat': [
        "tonsillar edema",
        "pharyngeal erythema",
        "posterior oropharynx edema",
        "mucus in oropharynx",
        "oropharyngeal discharge",
        "uvula midline",
        "posterior pharyngeal wall erythema",
        "oropharynx clear",
        "no tonsillar hypertrophy",
        "mucosa moist and pink"
    ],
}


# Model weights
WEIGHTS_NAME = 'pytorch_model.bin'

# Pretrained model URLs
PRETRAINED_URL_MEDCLIP_RESNET = 'https://github.com/RyanWangZf/MedCLIP/raw/main/medclip/medclip_resnet_weight.txt'
PRETRAINED_URL_MEDCLIP_VIT = 'https://github.com/RyanWangZf/MedCLIP/raw/main/medclip/medclip_vit_weight.txt'

# Dataset configurations
DATASET_CONFIGS = {
    'mimic': {
        'tasks': MIMIC_TASKS,
        'class_prompts': MIMIC_CLASS_PROMPTS,
        'data_files': {
            'train': 'mimic-train-meta.csv',
            'test': 'mimic-test-meta.csv',
            'finetune': 'mimic-finetune-meta.csv'
        },
        'mode': 'binary'
    },
    'covid': {
        'tasks': COVID_TASKS,
        'class_prompts': COVID_CLASS_PROMPTS,
        'data_files': {
            'train': 'covid-train-meta.csv',
            'test': 'covid-test-meta.csv',
            'small': 'covid-0.1-train-meta.csv'
        },
        'mode': 'binary'
    },
    'rsna': {
        'tasks': RSNA_TASKS,
        'class_prompts': RSNA_CLASS_PROMPTS,
        'data_files': {
            'train': 'stage_2_train_labels.csv',
            'test': 'rsna-balanced-test-meta.csv'
        },
        'mode': 'binary'
    },
    'entrep': {
        'tasks': ENTREP_TASKS,
        'data_files': {
            'train': 'entrep-train-meta.csv',
            'test': 'entrep-test-meta.csv',
            'val': 'entrep-val-meta.csv'
        },
        'mode': 'multiclass'
    }
}

# Supported model types
SUPPORTED_MODELS = ['medclip', 'biomedclip', 'entrep', 'rmedclip']

# Default templates for text prompts
DEFAULT_TEMPLATES = {
    'medclip': 'this is a photo of ',
    'biomedclip': 'this is a chest x-ray showing ',
    'entrep': 'this is an endoscopic image of ',
    'general': 'this is an image of '
}




MODEL_TRANSFORMS = {

    'medclip': transforms.Compose(
        [
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[IMG_MEAN], std=[IMG_STD])
        ]
    ),
    'biomedclip': transforms.Compose(
        [
            transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(IMG_SIZE),
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=BIOMEDCLIP_MEAN,
                std=BIOMEDCLIP_STD
            ),
        ]
    ),
    'entrep': transforms.Compose(
        [
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=ENTREP_MEAN, std=ENTREP_STD)
        ]
    ),
}
SIZE_TRANSFORM = {
    'medclip': transforms.Compose(
        [
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[IMG_MEAN], std=[IMG_STD])
        ]
    ),
    'biomedclip': transforms.Compose(
        [
            transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(IMG_SIZE),
            transforms.Lambda(lambda x: x.convert("RGB")),
            # transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=BIOMEDCLIP_MEAN,
            #     std=BIOMEDCLIP_STD
            # ),
        ]
    ),
    'rmedclip': transforms.Compose(
        [
            transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(IMG_SIZE),
            transforms.Lambda(lambda x: x.convert("RGB")),
            # transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=BIOMEDCLIP_MEAN,
            #     std=BIOMEDCLIP_STD
            # ),
        ]
    ),
    'entrep': transforms.Compose(
        [
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[IMG_MEAN], std=[IMG_STD])
        ]
    ),
    'entrep': transforms.Compose(
        [
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
        ]
    ),

    'ViT-B-32': transforms.Compose(
        [
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.CenterCrop((IMG_SIZE, IMG_SIZE)), 
        ]
    ),
    'ViT-B-16': transforms.Compose(
        [
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.CenterCrop((IMG_SIZE, IMG_SIZE)), 
        ]
    ),
    'ViT-L-14': transforms.Compose(
        [
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.CenterCrop((IMG_SIZE, IMG_SIZE)), 
        ]
    ),
}
TENSOR_NORMALIZE_TRANSFORM = {
    'medclip': transforms.Compose(
        [
            # transforms.Lambda(lambda x: x.convert("RGB")),
            # transforms.Resize((IMG_SIZE, IMG_SIZE)),
            # transforms.ToTensor(),
            transforms.Normalize(mean=[IMG_MEAN], std=[IMG_STD])
        ]
    ),
    'biomedclip': transforms.Compose(
        [
            # transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            # transforms.CenterCrop(IMG_SIZE),
            # transforms.Lambda(lambda x: x.convert("RGB")),
            # transforms.ToTensor(),
            transforms.Normalize(
                mean=BIOMEDCLIP_MEAN,
                std=BIOMEDCLIP_STD
            ),
        ]
    ),
    'entrep': transforms.Compose(
        [
            transforms.Normalize(mean=ENTREP_MEAN, std=ENTREP_STD)
        ]
    ),
    'ViT-B-32': transforms.Compose(
        [
            transforms.Normalize(mean=VIT_MEAN, std=VIT_STD)
        ]
    ),
    'ViT-B-16': transforms.Compose(
        [
            transforms.Normalize(mean=VIT_MEAN, std=VIT_STD)
        ]
    ),
    'ViT-L-14': transforms.Compose(
        [
            transforms.Normalize(mean=VIT_MEAN, std=VIT_STD)
        ]
    )
}


# for attack
DATA_ROOT = './local_data'

