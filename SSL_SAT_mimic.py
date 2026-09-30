import argparse
from copy import deepcopy
from typing import Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from SL_CTL_mimic import (
    CONFIG as BASE_CONFIG,
    build_dataloaders,
    build_model,
    freeze_text_encoder,
    print_config,
    save_stage_checkpoint,
    setup_seed,
    train_ctl_stage,
)


CONFIG = deepcopy(BASE_CONFIG)
CONFIG.update(
    {
        "output_dir": "checkpoints/ssl_sat_mimic_medclip",
        "sat_eps": 4.0 / 255.0,
        "sat_alpha": 1.0 / 255.0,
        "sat_steps": 3,
        "aug_noise_std": 0.03,
        "aug_brightness": 0.20,
    }
)


def sat_infonce_pair(z_q: torch.Tensor, z_k: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    z_q = F.normalize(z_q, dim=-1)
    z_k = F.normalize(z_k, dim=-1)

    logits = z_q @ z_k.t() / temperature
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_q2k = F.cross_entropy(logits, targets)
    loss_k2q = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_q2k + loss_k2q)


def stochastic_augment(images: torch.Tensor, config: Dict) -> torch.Tensor:
    aug_images = images.clone()

    flip_mask = torch.rand(aug_images.size(0), device=aug_images.device) < 0.5
    if flip_mask.any():
        aug_images[flip_mask] = torch.flip(aug_images[flip_mask], dims=[3])

    brightness = config.get("aug_brightness", 0.20)
    scale = 1.0 + (2.0 * torch.rand(aug_images.size(0), 1, 1, 1, device=aug_images.device) - 1.0) * brightness
    aug_images = aug_images * scale

    noise_std = config.get("aug_noise_std", 0.03)
    aug_images = aug_images + noise_std * torch.randn_like(aug_images)

    return aug_images.clamp(0.0, 1.0)


def generate_pgd_view(model, clean_images: torch.Tensor, anchor_images: torch.Tensor, config: Dict) -> torch.Tensor:
    eps = config["sat_eps"]
    alpha = config["sat_alpha"]
    steps = config["sat_steps"]

    with torch.no_grad():
        anchor_feats = F.normalize(model.encode_posttransform_image(anchor_images), dim=-1)

    adv_images = clean_images.detach() + torch.empty_like(clean_images).uniform_(-eps, eps)
    adv_images = adv_images.clamp(0.0, 1.0).detach()

    for _ in range(steps):
        adv_images.requires_grad_(True)

        adv_feats = F.normalize(model.encode_posttransform_image(adv_images), dim=-1)
        attack_objective = -F.cosine_similarity(adv_feats, anchor_feats, dim=-1).mean()
        grad = torch.autograd.grad(attack_objective, adv_images, only_inputs=True)[0]

        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = (adv_images - clean_images).clamp(-eps, eps)
        adv_images = (clean_images + delta).clamp(0.0, 1.0).detach()

    return adv_images


def run_ssl_sat_epoch(backbone_model, dataloader, optimizer, config: Dict) -> float:
    backbone_model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="SSL+SAT-train", leave=False):
        images = batch["image"].to(config["device"])

        aug_images = stochastic_augment(images, config)
        adv_images = generate_pgd_view(backbone_model, images, aug_images, config)

        aug_feats = backbone_model.encode_posttransform_image(aug_images)
        adv_feats = backbone_model.encode_posttransform_image(adv_images)
        loss = sat_infonce_pair(aug_feats, adv_feats, temperature=config["temperature"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_ssl_sat_stage(backbone_model, train_loader, config: Dict):
    print("Starting SSL+SAT stage on MIMIC...")

    if config["freeze_text_in_sl"]:
        freeze_text_encoder(backbone_model)

    sl_optimizer = torch.optim.AdamW(
        backbone_model.parameters(),
        lr=config["lr_sl"],
        weight_decay=config["weight_decay"],
    )

    best_train_loss = float("inf")

    for epoch in range(1, config["epochs_sl"] + 1):
        train_loss = run_ssl_sat_epoch(backbone_model, train_loader, sl_optimizer, config)

        is_best = train_loss < best_train_loss
        if is_best:
            best_train_loss = train_loss

        save_stage_checkpoint(
            stage_name="ssl_sat",
            epoch=epoch,
            output_dir=config["output_dir"],
            config=config,
            optimizer=sl_optimizer,
            model=backbone_model,
            metric_value=train_loss,
            best=is_best,
        )

        print(f"[SSL+SAT][Epoch {epoch}/{config['epochs_sl']}] train_loss={train_loss:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train MIMIC with SSL+SAT stage-1 + CTL stage-2")
    parser.add_argument(
        "--debug-num-samples",
        type=int,
        default=None,
        help="Limit number of samples per split for debugging. If omitted, use full dataset.",
    )
    parser.add_argument("--sat-eps", type=float, default=0.03, help="Linf epsilon for PGD SAT view")
    parser.add_argument("--sat-alpha", type=float, default=0.01, help="PGD step size for SAT view")
    parser.add_argument("--sat-steps", type=int, default=100, help="Number of PGD steps for SAT view")
    return parser.parse_args()


def main():
    args = parse_args()

    config = deepcopy(CONFIG)
    if args.debug_num_samples is not None:
        config["debug_num_samples"] = args.debug_num_samples
    if args.sat_eps is not None:
        config["sat_eps"] = args.sat_eps
    if args.sat_alpha is not None:
        config["sat_alpha"] = args.sat_alpha
    if args.sat_steps is not None:
        config["sat_steps"] = args.sat_steps

    setup_seed(config["seed"])
    print_config(config)

    train_loader, val_loader = build_dataloaders(config)
    backbone_model = build_model(config)

    if config["run_sl_stage"]:
        train_ssl_sat_stage(backbone_model, train_loader, config)

    if config["run_ctl_stage"]:
        print("Starting stage-2 CTL multimodal training...")
        train_ctl_stage(backbone_model, train_loader, val_loader, config)


if __name__ == "__main__":
    main()
