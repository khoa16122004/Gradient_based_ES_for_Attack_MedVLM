import open_clip
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-L-14',
    pretrained="openai",
    )
print(preprocess)