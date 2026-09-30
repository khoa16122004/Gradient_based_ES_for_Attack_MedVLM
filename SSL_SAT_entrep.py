import argparse
from copy import deepcopy

import torch
import torch.nn.functional as F
from tqdm import tqdm

from SL_CTL_entrep import (
    CONFIG as BASE_CONFIG,
    build_dataset,
    build_dataloader,
    build_model,
    freeze_text_encoder,
    get_vision_module,
    print_cuda_environment,
    save_checkpoint,
    setup_seed,
    train_clip,
)


CONFIG = deepcopy(BASE_CONFIG)
CONFIG.update(
    {
        "epochs_sl": 50,
        "output_dir": "checkpoints/ssl_sat_entrep",
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


def stochastic_augment(images: torch.Tensor, config: dict) -> torch.Tensor:
    aug_images = images.clone()

    # Random horizontal flip per-sample.
    flip_mask = torch.rand(aug_images.size(0), device=aug_images.device) < 0.5
    if flip_mask.any():
        aug_images[flip_mask] = torch.flip(aug_images[flip_mask], dims=[3])

    # Brightness jitter.
    brightness = config.get("aug_brightness", 0.20)
    scale = 1.0 + (2.0 * torch.rand(aug_images.size(0), 1, 1, 1, device=aug_images.device) - 1.0) * brightness
    aug_images = aug_images * scale

    # Gaussian noise injection.
    noise_std = config.get("aug_noise_std", 0.03)
    aug_images = aug_images + noise_std * torch.randn_like(aug_images)

    return aug_images.clamp(0.0, 1.0)


def generate_pgd_view(model, clean_images: torch.Tensor, anchor_images: torch.Tensor, config: dict) -> torch.Tensor:
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
        # Maximize feature discrepancy between PGD view and augmented anchor view.
        attack_objective = -F.cosine_similarity(adv_feats, anchor_feats, dim=-1).mean()
        grad = torch.autograd.grad(attack_objective, adv_images, only_inputs=True)[0]

        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = (adv_images - clean_images).clamp(-eps, eps)
        adv_images = (clean_images + delta).clamp(0.0, 1.0).detach()

    return adv_images


def train_ssl_sat_stage(model, dataloader, optimizer, config: dict):
    device = config["device"]
    temperature = config["temperature"]
    epochs = config["epochs_sl"]
    best_loss = float("inf")

    model.train()
    freeze_text_encoder(model)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for batch in tqdm(dataloader, desc=f"SSL+SAT epoch {epoch}/{epochs}"):
            images = batch["image"].to(device)
            aug_images = stochastic_augment(images, config)
            adv_images = generate_pgd_view(model, images, aug_images, config)

            aug_feats = model.encode_posttransform_image(aug_images)
            adv_feats = model.encode_posttransform_image(adv_images)
            loss = sat_infonce_pair(aug_feats, adv_feats, temperature=temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_loss = total_loss / len(dataloader)
        print(f"[SSL+SAT][Epoch {epoch}/{epochs}] loss={epoch_loss:.4f}")

        is_best = epoch_loss < best_loss
        if is_best:
            best_loss = epoch_loss

        save_checkpoint(
            model,
            optimizer,
            epoch,
            epoch_loss,
            "ssl_sat",
            config["output_dir"],
            best=is_best,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train ENTREP with SSL+SAT stage-1 + CLIP stage-2")
    parser.add_argument("--sat-eps", type=float, default=None, help="Linf epsilon for PGD SAT view")
    parser.add_argument("--sat-alpha", type=float, default=None, help="PGD step size for SAT view")
    parser.add_argument("--sat-steps", type=int, default=None, help="Number of PGD steps for SAT view")
    return parser.parse_args()


def main():
    args = parse_args()
    config = deepcopy(CONFIG)

    if args.sat_eps is not None:
        config["sat_eps"] = args.sat_eps
    if args.sat_alpha is not None:
        config["sat_alpha"] = args.sat_alpha
    if args.sat_steps is not None:
        config["sat_steps"] = args.sat_steps

    print("Starting SSL+SAT training process (ENTREP)...")
    setup_seed(config["seed"])
    print_cuda_environment(config)

    dataset = build_dataset(config)
    dataloader = build_dataloader(dataset, config)
    model = build_model(config)

    if config["run_sl_stage"]:
        print("Starting stage-1 SSL+SAT...")
        sl_optimizer = torch.optim.AdamW(
            get_vision_module(model).parameters(),
            lr=config["lr_sl"],
            weight_decay=config["weight_decay"],
        )
        train_ssl_sat_stage(model, dataloader, sl_optimizer, config)
        print("Finished stage-1 SSL+SAT.")

    if config["run_clip_stage"]:
        print("Starting stage-2 CLIP multimodal training...")
        clip_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["lr_clip"],
            weight_decay=config["weight_decay"],
        )
        train_clip(model, dataloader, clip_optimizer, config)
        print("Finished stage-2 CLIP multimodal training.")


if __name__ == "__main__":
    main()
