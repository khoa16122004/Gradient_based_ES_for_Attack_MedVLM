import argparse
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from modules.models.factory import ModelFactory
from modules.utils.constants import SIZE_TRANSFORM
from modules.utils.helpers import setup_seed


MIMIC_LABEL_COLUMNS: List[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Lung Lesion",
    "Lung Opacity",
    "Normal",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
]

DEFAULT_TEXT_BY_LABEL: Dict[str, str] = {
    "Atelectasis": "chest x-ray with atelectasis",
    "Cardiomegaly": "chest x-ray with cardiomegaly",
    "Consolidation": "chest x-ray with pulmonary consolidation",
    "Edema": "chest x-ray with pulmonary edema",
    "Enlarged Cardiomediastinum": "chest x-ray with enlarged cardiomediastinum",
    "Lung Lesion": "chest x-ray with lung lesion",
    "Lung Opacity": "chest x-ray with lung opacity",
    "Normal": "normal chest x-ray without acute cardiopulmonary abnormality",
    "Pleural Effusion": "chest x-ray with pleural effusion",
    "Pneumonia": "chest x-ray with pneumonia",
    "Pneumothorax": "chest x-ray with pneumothorax",
}

CONFIG = {
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_name": "biomedclip",  # medclip | biomedclip
    "dataset_name": "mimic",
    "mode_pretrained": "scratch",  # scratch | ssl | at | sl
    "batch_size": 64,
    "num_workers": 4,
    "epochs_sl": 50,
    "epochs_ctl": 200,
    "lr_sl": 1e-4,
    "lr_ctl": 5e-5,
    "weight_decay": 1e-4,
    "temperature": 0.07,
    "run_sl_stage": True,
    "run_ctl_stage": True,
    "freeze_text_in_sl": True,
    "unfreeze_text_in_ctl": True,
    "debug_num_samples": None,
    "output_dir": "/datastore/hoangln/KBS/checkpoints/sl_ctl_mimic_biomedclip",
    "data": {
        "data_root": "/datastore/hoangln/KBS/mimic-cxr",
        "csv_file": "/datastore/hoangln/KBS/mimic-cxr/mimic-cxr.csv",
        "train_split": "train",
        "val_split": "valid",
        "test_split": "test",
    },
}

_TO_TENSOR = transforms.ToTensor()
TEXT_CANDIDATE_COLUMNS: List[str] = ["finding", "findings", "report", "impression", "caption", "label"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return _repo_root() / path


def build_caption_from_multilabel(label_values: torch.Tensor, label_names: List[str]) -> str:
    positives = [label_names[i] for i, value in enumerate(label_values.tolist()) if value > 0.5]
    if not positives:
        return "normal chest x-ray"

    if len(positives) == 1:
        return DEFAULT_TEXT_BY_LABEL.get(positives[0], f"chest x-ray with {positives[0].lower()}")

    phrase = ", ".join([p.lower() for p in positives])
    return f"chest x-ray with {phrase}"


def build_text_from_row(row, label_values: torch.Tensor, label_names: List[str]) -> str:
    for column in TEXT_CANDIDATE_COLUMNS:
        if column not in row.index:
            continue
        value = row[column]
        if isinstance(value, str):
            value = value.strip()
            if value and value.lower() not in {"nan", "none"}:
                return value
    return build_caption_from_multilabel(label_values, label_names)


class MIMICSLCTLDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        csv_file: str,
        split: str,
        model_name: str,
        label_columns: List[str],
        max_samples: int = None,
    ):
        self.data_root = _resolve_path(data_root)
        self.csv_file = _resolve_path(csv_file)
        self.split = split
        self.model_name = model_name
        self.label_columns = label_columns
        self.size_transform = SIZE_TRANSFORM[model_name]

        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {self.data_root}")
        if not self.csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_file}")

        df = pd.read_csv(self.csv_file)

        required_cols = ["filename", "split", *label_columns]
        missing_cols = [column for column in required_cols if column not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in CSV: {missing_cols}")

        self.df = df[df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No samples found for split={split}")

        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError(f"max_samples must be > 0, got {max_samples}")
            self.df = self.df.head(max_samples).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = self.data_root / row["split"] / row["filename"]
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image = self.size_transform(image)
        image = _TO_TENSOR(image)

        label_values = torch.tensor(row[self.label_columns].astype(float).values, dtype=torch.float32)
        caption = build_text_from_row(row, label_values, self.label_columns)

        return {
            "image": image,
            "labels": label_values,
            "text": caption,
            "filename": row["filename"],
            "idx": idx,
        }


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


def clip_infonce_loss(image_feats: torch.Tensor, text_feats: torch.Tensor, temperature: float = 0.07):
    logits = image_feats @ text_feats.t() / temperature
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_i2t = F.cross_entropy(logits, targets)
    loss_t2i = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_i2t + loss_t2i)


def clip_infonce_loss_with_model(
    model,
    image_feats: torch.Tensor,
    text_feats: torch.Tensor,
    temperature: float = 0.07,
):
    """OpenCLIP-style symmetric contrastive loss with learnable logit scale fallback."""
    if hasattr(model, "logit_scale") and isinstance(model.logit_scale, torch.nn.Parameter):
        # Clamp to OpenCLIP's default max scale ln(100) for stability.
        model.logit_scale.data.clamp_(0, 4.6052)
        logits = (image_feats @ text_feats.t()) * model.logit_scale.exp()
    else:
        logits = image_feats @ text_feats.t() / temperature

    targets = torch.arange(logits.size(0), device=logits.device)
    loss_i2t = F.cross_entropy(logits, targets)
    loss_t2i = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_i2t + loss_t2i)


def sl_infonce_multilabel(
    z_q: torch.Tensor,
    z_k: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    logits = z_q @ z_k.t() / temperature

    # Positive pairs are samples sharing at least one positive class.
    positive_mask = (labels @ labels.t()) > 0
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


def _safe_torch_save(payload: Dict, target_path: Path):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, target_path)


def save_stage_checkpoint(
    stage_name: str,
    epoch: int,
    output_dir: str,
    config: Dict,
    optimizer,
    model,
    metric_value: float,
    best: bool,
):
    output_path = _resolve_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "metric": metric_value,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "stage": stage_name,
        "config": deepcopy(config),
        "labels": MIMIC_LABEL_COLUMNS,
    }

    last_path = output_path / f"{stage_name}_last.pth"
    _safe_torch_save(checkpoint, last_path)

    if best:
        best_path = output_path / f"{stage_name}_best.pth"
        _safe_torch_save(checkpoint, best_path)


def build_model(config: Dict):
    if config["model_name"] not in {"medclip", "biomedclip"}:
        raise ValueError("model_name must be one of: medclip, biomedclip")

    model = ModelFactory.create_model(
        model_type=config["model_name"],
        variant="base",
        pretrained=True,
        mode_pretrained=config["mode_pretrained"],
    )
    return model.to(config["device"])


def build_dataloaders(config: Dict):
    max_samples = config.get("debug_num_samples")

    train_dataset = MIMICSLCTLDataset(
        data_root=config["data"]["data_root"],
        csv_file=config["data"]["csv_file"],
        split=config["data"]["train_split"],
        model_name=config["model_name"],
        label_columns=MIMIC_LABEL_COLUMNS,
        max_samples=max_samples,
    )
    val_dataset = MIMICSLCTLDataset(
        data_root=config["data"]["data_root"],
        csv_file=config["data"]["csv_file"],
        split=config["data"]["val_split"],
        model_name=config["model_name"],
        label_columns=MIMIC_LABEL_COLUMNS,
        max_samples=max_samples,
    )

    if max_samples is not None:
        print(
            "[DEBUG] Limiting samples per split to "
            f"{max_samples} (train={len(train_dataset)}, val={len(val_dataset)})"
        )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader


def run_sl_epoch(backbone_model, dataloader, optimizer, config: Dict):
    backbone_model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="SL-train", leave=False):
        images = batch["image"].to(config["device"])
        labels = batch["labels"].to(config["device"])

        image_feats = backbone_model.encode_posttransform_image(images)
        loss = sl_infonce_multilabel(
            image_feats,
            image_feats,
            labels,
            temperature=config["temperature"],
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_sl_stage(backbone_model, train_loader, config: Dict):
    print("Starting SL stage on MIMIC multi-label (supervised contrastive)...")

    if config["freeze_text_in_sl"]:
        freeze_text_encoder(backbone_model)

    sl_optimizer = torch.optim.AdamW(
        backbone_model.parameters(),
        lr=config["lr_sl"],
        weight_decay=config["weight_decay"],
    )

    best_train_loss = float("inf")

    for epoch in range(1, config["epochs_sl"] + 1):
        train_loss = run_sl_epoch(
            backbone_model,
            train_loader,
            sl_optimizer,
            config,
        )

        is_best = train_loss < best_train_loss
        if is_best:
            best_train_loss = train_loss

        save_stage_checkpoint(
            stage_name="sl",
            epoch=epoch,
            output_dir=config["output_dir"],
            config=config,
            optimizer=sl_optimizer,
            model=backbone_model,
            metric_value=train_loss,
            best=is_best,
        )

        print(f"[SL][Epoch {epoch}/{config['epochs_sl']}] train_loss={train_loss:.4f}")


def run_ctl_epoch(backbone_model, dataloader, optimizer, config: Dict, train_mode: bool):
    if train_mode:
        backbone_model.train()
    else:
        backbone_model.eval()

    total_loss = 0.0

    for batch in tqdm(dataloader, desc="CTL-train" if train_mode else "CTL-val", leave=False):
        images = batch["image"].to(config["device"])
        texts = batch["text"]

        if train_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            image_feats = backbone_model.encode_posttransform_image(images)
            text_feats = backbone_model.encode_text(texts)
            loss = clip_infonce_loss_with_model(
                backbone_model,
                image_feats,
                text_feats,
                temperature=config["temperature"],
            )

            if train_mode:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_ctl_stage(backbone_model, train_loader, val_loader, config: Dict):
    print("Starting CTL stage on MIMIC image-text pairs...")

    if config["unfreeze_text_in_ctl"]:
        unfreeze_text_encoder(backbone_model)

    ctl_optimizer = torch.optim.AdamW(
        backbone_model.parameters(),
        lr=config["lr_ctl"],
        weight_decay=config["weight_decay"],
    )

    best_val_loss = float("inf")

    for epoch in range(1, config["epochs_ctl"] + 1):
        train_loss = run_ctl_epoch(backbone_model, train_loader, ctl_optimizer, config, train_mode=True)
        val_loss = run_ctl_epoch(backbone_model, val_loader, ctl_optimizer, config, train_mode=False)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        save_stage_checkpoint(
            stage_name="ctl",
            epoch=epoch,
            output_dir=config["output_dir"],
            config=config,
            optimizer=ctl_optimizer,
            model=backbone_model,
            metric_value=val_loss,
            best=is_best,
        )

        print(
            f"[CTL][Epoch {epoch}/{config['epochs_ctl']}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
        )


def print_config(config: Dict):
    print("==== Training Config (MIMIC SL + CTL) ====")
    for key in [
        "model_name",
        "dataset_name",
        "mode_pretrained",
        "device",
        "batch_size",
        "epochs_sl",
        "epochs_ctl",
        "lr_sl",
        "lr_ctl",
        "weight_decay",
        "output_dir",
    ]:
        print(f"{key}: {config[key]}")
    print(f"debug_num_samples: {config.get('debug_num_samples')}")
    print(f"data_root: {config['data']['data_root']}")
    print(f"csv_file: {config['data']['csv_file']}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train MIMIC SL + CTL")
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        choices=["medclip", "biomedclip"],
        help="Backbone model for MIMIC training.",
    )
    parser.add_argument(
        "--debug-num-samples",
        type=int,
        default=None,
        help="Limit number of samples per split for debugging. If omitted, use full dataset.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size override for train/val dataloaders.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = deepcopy(CONFIG)
    if args.model_name is not None:
        config["model_name"] = args.model_name
    if args.debug_num_samples is not None:
        config["debug_num_samples"] = args.debug_num_samples
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size

    setup_seed(config["seed"])
    print_config(config)

    train_loader, val_loader = build_dataloaders(config)
    backbone_model = build_model(config)

    if config["run_sl_stage"]:
        train_sl_stage(backbone_model, train_loader, config)

    if config["run_ctl_stage"]:
        train_ctl_stage(backbone_model, train_loader, val_loader, config)


if __name__ == "__main__":
    main()
