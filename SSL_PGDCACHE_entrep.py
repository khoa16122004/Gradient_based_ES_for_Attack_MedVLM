import argparse
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as TVF
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
        "output_dir": "checkpoints/ssl_pgdcache_entrep",
        "sat_eps": 0.03,
        "sat_alpha": 0.01,
        "sat_steps": 100,
        "aug_noise_std": 0.03,
        "aug_brightness": 0.20,
        "aug_affine_p": 0.7,
        "aug_rotation_deg": 20.0,
        "aug_translate_ratio": 0.1,
        "aug_scale_min": 0.9,
        "aug_scale_max": 1.1,
        "aug_autocontrast_p": 0.2,
        "aug_blur_sigma_min": 0.1,
        "aug_blur_sigma_max": 1.5,
        "aug_erasing_p": 0.2,
        "pgd_cache_dir": "cache/pgd_vanilla_entrep",
    }
)


_COLOR_JITTER = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
_RANDOM_ERASING = transforms.RandomErasing(
    p=0.2,
    scale=(0.02, 0.08),
    ratio=(0.3, 3.3),
    value="random",
    inplace=False,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return _repo_root() / path


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

    batch_size, _, height, width = aug_images.shape

    affine_p = config.get("aug_affine_p", 0.7)
    rotation_deg = config.get("aug_rotation_deg", 20.0)
    translate_ratio = config.get("aug_translate_ratio", 0.1)
    scale_min = config.get("aug_scale_min", 0.9)
    scale_max = config.get("aug_scale_max", 1.1)

    max_tx = int(round(translate_ratio * width))
    max_ty = int(round(translate_ratio * height))

    for i in range(batch_size):
        if torch.rand(1, device=aug_images.device).item() < affine_p:
            angle = float(torch.empty(1, device=aug_images.device).uniform_(-rotation_deg, rotation_deg).item())
            tx = int(torch.randint(-max_tx, max_tx + 1, (1,), device=aug_images.device).item()) if max_tx > 0 else 0
            ty = int(torch.randint(-max_ty, max_ty + 1, (1,), device=aug_images.device).item()) if max_ty > 0 else 0
            scale = float(torch.empty(1, device=aug_images.device).uniform_(scale_min, scale_max).item())
            aug_images[i] = TVF.affine(
                aug_images[i],
                angle=angle,
                translate=[tx, ty],
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=transforms.InterpolationMode.BILINEAR,
            )

    for i in range(batch_size):
        aug_images[i] = _COLOR_JITTER(aug_images[i])

    autocontrast_p = config.get("aug_autocontrast_p", 0.2)
    autocontrast_mask = torch.rand(batch_size, device=aug_images.device) < autocontrast_p
    if autocontrast_mask.any():
        selected = aug_images[autocontrast_mask]
        channel_min = selected.amin(dim=(-2, -1), keepdim=True)
        channel_max = selected.amax(dim=(-2, -1), keepdim=True)
        selected = (selected - channel_min) / (channel_max - channel_min).clamp_min(1e-6)
        aug_images[autocontrast_mask] = selected

    sigma = float(
        torch.empty(1, device=aug_images.device).uniform_(
            config.get("aug_blur_sigma_min", 0.1),
            config.get("aug_blur_sigma_max", 1.5),
        ).item()
    )
    aug_images = TVF.gaussian_blur(aug_images, kernel_size=[3, 3], sigma=[sigma, sigma])

    erasing = transforms.RandomErasing(
        p=config.get("aug_erasing_p", 0.2),
        scale=(0.02, 0.08),
        ratio=(0.3, 3.3),
        value="random",
        inplace=False,
    )
    for i in range(batch_size):
        aug_images[i] = erasing(aug_images[i])

    noise_std = config.get("aug_noise_std", 0.03)
    if noise_std > 0:
        aug_images = aug_images + noise_std * torch.randn_like(aug_images)

    return aug_images.clamp(0.0, 1.0)


def generate_pgd_from_clean(model, clean_images: torch.Tensor, config: dict) -> torch.Tensor:
    eps = config["sat_eps"]
    alpha = config["sat_alpha"]
    steps = config["sat_steps"]

    with torch.no_grad():
        clean_feats = F.normalize(model.encode_posttransform_image(clean_images), dim=-1)

    adv_images = clean_images.detach() + torch.empty_like(clean_images).uniform_(-eps, eps)
    adv_images = adv_images.clamp(0.0, 1.0).detach()

    for _ in range(steps):
        adv_images.requires_grad_(True)
        adv_feats = F.normalize(model.encode_posttransform_image(adv_images), dim=-1)

        # Maximize feature discrepancy between clean image and PGD image.
        attack_objective = -F.cosine_similarity(adv_feats, clean_feats, dim=-1).mean()
        grad = torch.autograd.grad(attack_objective, adv_images, only_inputs=True)[0]

        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = (adv_images - clean_images).clamp(-eps, eps)
        adv_images = (clean_images + delta).clamp(0.0, 1.0).detach()

    return adv_images


def _cache_file_path(cache_root: Path, sample_idx: int) -> Path:
    return cache_root / f"{sample_idx}.pt"


def precompute_and_cache_pgd(model, dataset, config: dict, force_rebuild: bool = False):
    cache_root = _resolve_path(config["pgd_cache_dir"])

    ordered_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    if force_rebuild and cache_root.exists():
        for existing_file in cache_root.rglob("*.pt"):
            existing_file.unlink()

    cache_root.mkdir(parents=True, exist_ok=True)

    model.eval()

    created = 0
    skipped = 0

    for batch in tqdm(ordered_loader, desc="Build PGD cache", leave=False):
        images = batch["image"].to(config["device"])
        indices = batch["idx"].tolist()
        target_paths = [_cache_file_path(cache_root, int(i)) for i in indices]

        if not force_rebuild and all(path.exists() for path in target_paths):
            skipped += len(target_paths)
            continue

        adv_images = generate_pgd_from_clean(model, images, config).detach().cpu()

        for i, path in enumerate(target_paths):
            if path.exists() and not force_rebuild:
                skipped += 1
                continue
            torch.save(adv_images[i], path)
            created += 1

    print(f"PGD cache ready at: {cache_root}")
    print(f"PGD cache stats: created={created}, skipped={skipped}")


class EntrepCachedPGDDataset(Dataset):
    def __init__(self, base_dataset, cache_dir: str):
        self.base_dataset = base_dataset
        self.cache_root = _resolve_path(cache_dir)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        sample = self.base_dataset[idx]
        sample_idx = int(sample["idx"])
        cache_path = _cache_file_path(self.cache_root, sample_idx)

        if not cache_path.exists():
            raise FileNotFoundError(
                f"Missing PGD cache for sample idx={sample_idx} at {cache_path}. "
                "Run with --rebuild-pgd-cache to regenerate."
            )

        pgd_image = torch.load(cache_path, map_location="cpu")
        if not isinstance(pgd_image, torch.Tensor):
            raise TypeError(f"Invalid PGD cache format at {cache_path}; expected Tensor.")

        sample["pgd_image"] = pgd_image.to(dtype=sample["image"].dtype)
        return sample


def build_cached_ssl_loader(config: dict):
    base_dataset = build_dataset(config)
    cached_dataset = EntrepCachedPGDDataset(base_dataset=base_dataset, cache_dir=config["pgd_cache_dir"])

    return DataLoader(
        cached_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


def train_ssl_pgd_cache_stage(model, dataloader, optimizer, config: dict):
    device = config["device"]
    temperature = config["temperature"]
    epochs = config["epochs_sl"]
    best_loss = float("inf")

    model.train()
    freeze_text_encoder(model)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for batch in tqdm(dataloader, desc=f"SSL+PGDCache epoch {epoch}/{epochs}"):
            images = batch["image"].to(device)
            pgd_images = batch["pgd_image"].to(device)
            aug_images = stochastic_augment(images, config)

            aug_feats = model.encode_posttransform_image(aug_images)
            pgd_feats = model.encode_posttransform_image(pgd_images)
            loss = sat_infonce_pair(aug_feats, pgd_feats, temperature=temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_loss = total_loss / len(dataloader)
        print(f"[SSL+PGDCache][Epoch {epoch}/{epochs}] loss={epoch_loss:.4f}")

        is_best = epoch_loss < best_loss
        if is_best:
            best_loss = epoch_loss

        save_checkpoint(
            model,
            optimizer,
            epoch,
            epoch_loss,
            "ssl_pgdcache",
            config["output_dir"],
            best=is_best,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train ENTREP with SSL+PGDCache stage-1 + CLIP stage-2")
    parser.add_argument("--sat-eps", type=float, default=CONFIG["sat_eps"], help="Linf epsilon for cached PGD")
    parser.add_argument("--sat-alpha", type=float, default=CONFIG["sat_alpha"], help="PGD step size for cached PGD")
    parser.add_argument("--sat-steps", type=int, default=CONFIG["sat_steps"], help="Number of PGD steps for cached PGD")
    parser.add_argument(
        "--pgd-cache-dir",
        type=str,
        default=CONFIG["pgd_cache_dir"],
        help="Folder to store/load precomputed PGD tensors.",
    )
    parser.add_argument(
        "--rebuild-pgd-cache",
        action="store_true",
        help="Force rebuild PGD cache even if files already exist.",
    )
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
    if args.pgd_cache_dir is not None:
        config["pgd_cache_dir"] = args.pgd_cache_dir

    print("Starting SSL+PGDCache training process (ENTREP)...")
    setup_seed(config["seed"])
    print_cuda_environment(config)
    print(f"pgd_cache_dir: {config['pgd_cache_dir']}")

    dataset = build_dataset(config)
    clip_dataloader = build_dataloader(dataset, config)
    model = build_model(config)

    precompute_and_cache_pgd(model, dataset, config, force_rebuild=args.rebuild_pgd_cache)
    ssl_dataloader = build_cached_ssl_loader(config)

    if config["run_sl_stage"]:
        print("Starting stage-1 SSL+PGDCache...")
        sl_optimizer = torch.optim.AdamW(
            get_vision_module(model).parameters(),
            lr=config["lr_sl"],
            weight_decay=config["weight_decay"],
        )
        train_ssl_pgd_cache_stage(model, ssl_dataloader, sl_optimizer, config)
        print("Finished stage-1 SSL+PGDCache.")

    if config["run_clip_stage"]:
        print("Starting stage-2 CLIP multimodal training...")
        clip_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["lr_clip"],
            weight_decay=config["weight_decay"],
        )
        train_clip(model, clip_dataloader, clip_optimizer, config)
        print("Finished stage-2 CLIP multimodal training.")


if __name__ == "__main__":
    main()
