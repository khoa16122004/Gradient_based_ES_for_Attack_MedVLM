import json
import os
import warnings
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from modules.models.factory import ModelFactory
from modules.utils.constants import SIZE_TRANSFORM
from modules.utils.helpers import setup_seed


CONFIG = {
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_name": "entrep",
    "dataset_name": "entrep",
    "mode_pretrained": "scratch",
    "batch_size": 32,
    "num_workers": 0,
    "epochs_sl": 50,
    "epochs_clip": 200,
    "clip_save_every": 10,
    "temperature": 0.07,
    "lr_sl": 1e-4,
    "lr_clip": 1e-4,
    "weight_decay": 1e-4,
    "run_sl_stage": True,
    "run_clip_stage": True,
    "data": {
        "img_dir": "/datastore/khoatn/Gradient_based_ES_for_Attack_MedVLM/Dataset/Images",
        "annotation_file": "/datastore/khoatn/Gradient_based_ES_for_Attack_MedVLM/Dataset/data.json",
    },
    "output_dir": "checkpoints/sl_ctl",
}


_TO_TENSOR = transforms.ToTensor()
ENTREP_CLASS_NAMES = ["voice-throat", "nose", "ear", "throat"]
ENTREP_CLASS_TO_ID = {name: idx for idx, name in enumerate(ENTREP_CLASS_NAMES)}
ENTREP_CLASS_MAPPING = {
    "vc-open": "voice-throat",
    "vc-closed": "voice-throat",
    "nose-left": "nose",
    "nose-right": "nose",
    "ear-left": "ear",
    "ear-right": "ear",
    "throat": "throat",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return _repo_root() / path


class ENTREPDataset(Dataset):
    def __init__(self, size_transform, img_dir: str, annotation_file: str):
        self.img_dir = _resolve_path(img_dir)
        self.annotation_file = _resolve_path(annotation_file)
        self.size_transform = size_transform
        self.samples = []

        with self.annotation_file.open("r", encoding="utf-8") as file_obj:
            samples_data = json.load(file_obj)

        for sample_data in samples_data:
            relative_image_path = sample_data["Path"].replace("image", "Image")
            image_path = self.img_dir / relative_image_path
            text = sample_data.get("DescriptionEN") or sample_data.get("Description") or ""
            class_id = self._extract_class_id(sample_data)
            self.samples.append((image_path, text, class_id))

    def _extract_class_id(self, sample_data):
        raw_class_name = sample_data["Classification"]
        mapped_class_name = ENTREP_CLASS_MAPPING[raw_class_name]
        return ENTREP_CLASS_TO_ID[mapped_class_name]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text, class_id = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.size_transform(image)
        image = _TO_TENSOR(image)
        return {
            "image": image,
            "text": text,
            "class_id": torch.tensor(class_id, dtype=torch.long),
            "idx": idx,
        }


def sl_infonce(z_q, z_k, class_ids, temperature=0.07):
    logits = z_q @ z_k.t() / temperature
    positive_mask = class_ids.unsqueeze(0) == class_ids.unsqueeze(1)
    positive_mask = positive_mask.to(dtype=logits.dtype)

    log_denominator = torch.logsumexp(logits, dim=1)
    masked_logits = logits.masked_fill(positive_mask == 0, float("-inf"))
    log_numerator = torch.logsumexp(masked_logits, dim=1)

    valid_mask = positive_mask.sum(dim=1) > 0
    if not torch.all(valid_mask):
        log_denominator = log_denominator[valid_mask]
        log_numerator = log_numerator[valid_mask]

    loss = -(log_numerator - log_denominator)
    return loss.mean() if loss.numel() > 0 else torch.tensor(0.0, device=z_q.device)


def clip_infonce_loss(image_feats, text_feats, temperature=0.07):
    logits = image_feats @ text_feats.t() / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    loss_i2t = F.cross_entropy(logits, labels)
    return 0.5 * (loss_t2i + loss_i2t)


def clip_infonce_loss_with_model(model, image_feats, text_feats, temperature=0.07):
    """OpenCLIP-style symmetric contrastive loss with learnable logit scale fallback."""
    if hasattr(model, "logit_scale") and isinstance(model.logit_scale, torch.nn.Parameter):
        # Clamp to OpenCLIP's default max scale ln(100) for stability.
        model.logit_scale.data.clamp_(0, 4.6052)
        logits = (image_feats @ text_feats.t()) * model.logit_scale.exp()
    else:
        logits = image_feats @ text_feats.t() / temperature

    labels = torch.arange(logits.size(0), device=logits.device)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    loss_i2t = F.cross_entropy(logits, labels)
    return 0.5 * (loss_t2i + loss_i2t)


def freeze_text_encoder(model):
    if hasattr(model, "text_model"):
        for parameter in model.text_model.parameters():
            parameter.requires_grad = False
    elif hasattr(model, "model") and hasattr(model.model, "text"):
        for parameter in model.model.text.parameters():
            parameter.requires_grad = False


def unfreeze_text_encoder(model):
    if hasattr(model, "text_model"):
        for parameter in model.text_model.parameters():
            parameter.requires_grad = True
    elif hasattr(model, "model") and hasattr(model.model, "text"):
        for parameter in model.model.text.parameters():
            parameter.requires_grad = True


def get_vision_module(model):
    if hasattr(model, "vision_model"):
        return model.vision_model
    if hasattr(model, "model") and hasattr(model.model, "visual"):
        return model.model.visual
    raise AttributeError(f"Cannot locate vision encoder for model type: {type(model).__name__}")


def print_cuda_environment(config):
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>")
    print(f"CUDA_VISIBLE_DEVICES={visible_devices}")

    if not torch.cuda.is_available():
        print("CUDA unavailable, using CPU")
        return

    device = torch.device(config["device"])
    print(f"torch.cuda.device_count()={torch.cuda.device_count()}")
    print(f"current_device={torch.cuda.current_device()}")

    if device.type == "cuda":
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        print(f"using_device=cuda:{device_index} ({torch.cuda.get_device_name(device_index)})")


def build_model(config):
    model_name = config["model_name"]
    if model_name in {"medclip", "biomedclip"}:
        model = ModelFactory.create_model(
            model_type=model_name,
            variant="base",
            pretrained=True,
            mode_pretrained=config["mode_pretrained"],
        )
    elif model_name == "entrep":
        model = ModelFactory.create_model(
            model_type="entrep",
            variant="base",
            checkpoint=None,
            pretrained=False,
            mode_pretrained=config["mode_pretrained"],
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    return model.to(config["device"])


def build_dataset(config):
    if config["dataset_name"] != "entrep":
        raise ValueError(f"Unsupported dataset_name: {config['dataset_name']}")

    return ENTREPDataset(
        size_transform=SIZE_TRANSFORM[config["model_name"]],
        img_dir=config["data"]["img_dir"],
        annotation_file=config["data"]["annotation_file"],
    )


def build_dataloader(dataset, config):
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


def _safe_torch_save(payload, target_path: Path):
    """Save with atomic replace and legacy-format fallback for flaky filesystems."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")

    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, target_path)
        return
    except RuntimeError as error:
        # Some distributed/network filesystems intermittently fail with
        # zip writer finalization errors; retry using legacy serialization.
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        if "iostream" not in str(error).lower():
            raise

    torch.save(payload, tmp_path, _use_new_zipfile_serialization=False)
    os.replace(tmp_path, target_path)


def save_checkpoint(model, optimizer, epoch, loss_value, stage_name, output_dir, best=False):
    output_path = Path(output_dir)
    print("output_path=", output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "loss": loss_value,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "stage": stage_name,
        "config": deepcopy(CONFIG),
    }

    full_checkpoint_path = output_path / f"{stage_name}_full_last.pth"
    try:
        _safe_torch_save(checkpoint, full_checkpoint_path)
    except Exception as error:
        warnings.warn(
            f"Failed to save full checkpoint at {full_checkpoint_path}: {error}",
            RuntimeWarning,
        )

    vision_checkpoint = {
        "epoch": epoch,
        "loss": loss_value,
        "model_state_dict": get_vision_module(model).state_dict(),
        "stage": stage_name,
        "config": deepcopy(CONFIG),
    }
    vision_checkpoint_path = output_path / f"{stage_name}_vision_last.pth"
    try:
        _safe_torch_save(vision_checkpoint, vision_checkpoint_path)
    except Exception as error:
        warnings.warn(
            f"Failed to save vision checkpoint at {vision_checkpoint_path}: {error}",
            RuntimeWarning,
        )

    if best:
        full_best_path = output_path / f"{stage_name}_full_best.pth"
        vision_best_path = output_path / f"{stage_name}_vision_best.pth"
        try:
            _safe_torch_save(checkpoint, full_best_path)
        except Exception as error:
            warnings.warn(
                f"Failed to save best full checkpoint at {full_best_path}: {error}",
                RuntimeWarning,
            )
        try:
            _safe_torch_save(vision_checkpoint, vision_best_path)
        except Exception as error:
            warnings.warn(
                f"Failed to save best vision checkpoint at {vision_best_path}: {error}",
                RuntimeWarning,
            )


def train_sl_vision_encoder(model, dataloader, optimizer, config):
    device = config["device"]
    temperature = config["temperature"]
    epochs = config["epochs_sl"]
    best_loss = float("inf")

    model.train()
    freeze_text_encoder(model)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for batch in tqdm(dataloader, desc=f"SL epoch {epoch}/{epochs}"):
            images = batch["image"].to(device)
            class_ids = batch["class_id"].to(device)

            image_feats = model.encode_posttransform_image(images)
            loss = sl_infonce(image_feats, image_feats, class_ids, temperature=temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_loss = total_loss / len(dataloader)
        print(f"[SL][Epoch {epoch}/{epochs}] loss={epoch_loss:.4f}")
        is_best = epoch_loss < best_loss
        if is_best:
            best_loss = epoch_loss
        save_checkpoint(model, optimizer, epoch, epoch_loss, "sl", config["output_dir"], best=is_best)


def train_clip(model, dataloader, optimizer, config):
    device = config["device"]
    temperature = config["temperature"]
    epochs = config["epochs_clip"]
    clip_save_every = config.get("clip_save_every", 10)
    best_loss = float("inf")

    model.train()
    unfreeze_text_encoder(model)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for batch in tqdm(dataloader, desc=f"CLIP epoch {epoch}/{epochs}"):
            images = batch["image"].to(device)
            texts = batch["text"]

            image_feats = model.encode_posttransform_image(images)
            text_feats = model.encode_text(texts)
            loss = clip_infonce_loss_with_model(
                model,
                image_feats,
                text_feats,
                temperature=temperature,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_loss = total_loss / len(dataloader)
        print(f"[CLIP][Epoch {epoch}/{epochs}] loss={epoch_loss:.4f}")
        is_best = epoch_loss < best_loss
        if is_best:
            best_loss = epoch_loss

        should_save = (epoch % clip_save_every == 0) or (epoch == epochs)
        if should_save:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                epoch_loss,
                "clip",
                config["output_dir"],
                best=is_best,
            )


def main():
    print("Starting training process...")
    setup_seed(CONFIG["seed"])
    print_cuda_environment(CONFIG)
    dataset = build_dataset(CONFIG)
    dataloader = build_dataloader(dataset, CONFIG)
    model = build_model(CONFIG)

    if CONFIG["run_sl_stage"]:
        print("Starting SL stage...")
        sl_optimizer = torch.optim.AdamW(
            get_vision_module(model).parameters(),
            lr=CONFIG["lr_sl"],
            weight_decay=CONFIG["weight_decay"],
        )
        train_sl_vision_encoder(model, dataloader, sl_optimizer, CONFIG)

        print("Finished SL stage.")
    
    if CONFIG["run_clip_stage"]:
        print("Starting CLIP stage...")
        clip_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=CONFIG["lr_clip"],
            weight_decay=CONFIG["weight_decay"],
        )
        train_clip(model, dataloader, clip_optimizer, CONFIG)
        print("Finished CLIP stage.")
        
    # Removed redundant CLIP stage block


if __name__ == "__main__":
    main()

