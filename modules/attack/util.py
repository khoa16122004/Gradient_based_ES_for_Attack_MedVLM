import torch
import os
import random
import numpy as np
import torch
from torchvision.transforms import ToTensor, ToPILImage

def pil_to_tensor(imgs):
    return torch.stack([ToTensor()(img) for img in imgs])

def tensor_to_pillow(imgs):
    return [ToPILImage()(img) for img in imgs]




def clamp_eps(delta: torch.Tensor, eps: float, norm: str = "linf") -> torch.Tensor:
    if norm == "linf":
        return torch.clamp(delta, -eps, eps)
    elif norm == "l2":
        shape = delta.shape
        pop = shape[0]
        flat = delta.view(pop, -1)
        norms = torch.norm(flat, dim=1, keepdim=True).clamp_min(1e-12)
        factor = (eps / norms).clamp_max(1.0)
        return (flat * factor).view_as(delta)
    else:
        raise ValueError("norm must be 'linf' or 'l2'")

def _project_linf(delta: torch.Tensor, eps: float) -> torch.Tensor:
    if eps is None:
        return delta
    return delta.clamp(min=-eps, max=eps)

def _project_l2(delta: torch.Tensor, eps: float) -> torch.Tensor:
    norm = delta.view(delta.size(0), -1).norm(p=2, dim=1, keepdim=True)
    norm = torch.clamp(norm, min=1e-12)
    factor = torch.min(torch.ones_like(norm), eps / norm)
    return delta * factor.view(-1, 1, 1, 1)

def _project_l0(delta: torch.Tensor, eps: float) -> torch.Tensor:
    if eps is None:
        return delta

    B = delta.shape[0]
    flat = delta.view(B, -1)                    # [B, N]
    N = flat.shape[1]

    if 0 < eps < 1:
        k = max(1, int(round(eps * N)))
    else:
        k = int(eps)
        k = max(0, min(k, N))

    if k == N:
        return delta
    if k == 0:
        return torch.zeros_like(delta)

    abs_flat = flat.abs()
    topk_vals, topk_idx = torch.topk(abs_flat, k, dim=1, largest=True, sorted=False)  # [B, k]

    mask = torch.zeros_like(flat, dtype=torch.bool)  # [B, N]
    rows = torch.arange(B, device=flat.device).unsqueeze(1).expand(B, k)
    mask[rows, topk_idx] = True

    pruned = torch.zeros_like(flat)
    pruned[mask] = flat[mask]
    return pruned.view_as(delta)

def clamp_eps(delta: torch.Tensor, eps: float, norm: str) -> torch.Tensor:
    norm = norm.lower()
    if norm in ["linf", "l∞", "inf"]:
        return _project_linf(delta, eps)
    elif norm in ["l2"]:
        return _project_l2(delta, eps)
    elif norm in ["l0"]:
        return _project_l0(delta, eps)
    else:
        raise ValueError(f"Unsupported norm: {norm}. Use 'linf', 'l2', or 'l0'.")

def project_delta(delta: torch.Tensor,
                  eps: float,
                  norm: str = "linf"
                  ) -> torch.Tensor:
    
    delta = clamp_eps(delta, eps, norm)
    return delta




def seed_everything(
    seed: int,
    cudnn_deterministic: bool = True,
    cudnn_benchmark: bool = False,
) -> None:

    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(cudnn_deterministic)
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)

