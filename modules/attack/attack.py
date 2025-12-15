import torch
import numpy as np
from typing import Any, Dict
from .util import clamp_eps, project_delta
from tqdm import tqdm
from time import time
g_gpu = torch.Generator(device='cuda')


class BaseAttack:
    def __init__(self, evaluator, eps=8/255, norm="l2", device=None):
        self.evaluator = evaluator
        self.eps = float(eps)
        self.norm = norm
        self.device = device if device is not None else next(self.evaluator.model.parameters()).device

    def evaluate_population(self, deltas: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            margins, l2s = self.evaluator.evaluate_blackbox(deltas)  # torch (pop,)
            return margins.clone(), l2s.clone()
        
    def is_success(self, margin):
        if margin < 0:
            return True
        return False
    
    def z_to_delta(self, z):
        s = torch.tanh(z)           # s in (-1,1)
        return self.eps * s

class ES_1_Lambda(BaseAttack):
    def __init__(self, evaluator, eps=8/255, norm="linf",
                 max_evaluation=10000, lam=64, c_inc=1.5, c_dec=0.9, device='cuda'):
        super().__init__(evaluator, eps, norm, device)
        # assert lam >= 2 and c_inc > 1.0 and 0.0 < c_dec < 1.0
        self.lam = int(lam)
        self.c_inc = float(c_inc)
        self.c_dec = float(c_dec)
        self.sigma = 1.1  # σ tuyệt đối
        self.max_evaluation = max_evaluation

    def run(self) -> Dict[str, Any]:
        sigma = self.sigma
        _, C, H, W = self.evaluator.img_tensor.shape
        
        m = torch.randn((1, C, H, W), device=self.device)
        delta_m = self.z_to_delta(m)
        delta_m = project_delta(delta_m, self.eps, self.norm)

        f_m, l2_m = self.evaluator.evaluate_blackbox(delta_m)
        history = [[float(f_m.item()), delta_m.cpu()]]

        num_evaluation = 1
        while num_evaluation < self.max_evaluation:
            # noise = torch.randn((self.lam, C, H, W), device=self.device)
            noise = torch.randn((self.lam, C, H, W), device=device, generator=g_gpu)
            X = m + sigma * noise
            X_delta = self.z_to_delta(X)
            X_delta = project_delta(X_delta, self.eps, self.norm)

            margins, l2s = self.evaluate_population(X_delta)
            num_evaluation += self.lam
            idx_best = torch.argmin(margins).item()
            x_best = X[idx_best].clone()
            f_best = float(margins[idx_best].item())
            l2_best = float(l2s[idx_best].item())
            x_delta_best = X_delta[idx_best].clone()
            if f_best < f_m:
                m = x_best.clone()
                delta_m = x_delta_best.clone()
                l2_m = l2_best
                f_m = f_best
                sigma *= self.c_inc
                # sigma = min(self.eps, self.sigma)
            else:
                sigma *= self.c_dec            
                # sigma = max(1e-6, sigma)     
            
            # print(f"[{num_evaluation} - attack phase] Best loss: ", f_m, " L2: ", l2_m )
            history.append([float(f_m), delta_m.cpu()])
            if self.is_success(f_m):
                break
            
            
        return {"best_delta": delta_m, "best_margin": f_m, "history": history, "num_evaluation": num_evaluation}



class CMA_ES(BaseAttack):
    def __init__(self, evaluator, eps=8/255, norm="linf",
                 max_evaluation=10000, lam=64, mu=None,
                 sigma=0.5, c_cov=0.2, device="cuda"):
        super().__init__(evaluator, eps, norm, device)

        self.lam = int(lam)
        self.mu = mu if mu is not None else lam // 2
        self.max_evaluation = max_evaluation

        self.sigma = float(sigma)
        self.c_cov = float(c_cov)

        w = torch.log(torch.tensor(self.mu + 0.5)) - torch.log(torch.arange(1, self.mu + 1))
        self.weights = (w / w.sum()).to(device)

    def run(self):
        _, C, H, W = self.evaluator.img_tensor.shape

        m = torch.randn((1, C, H, W), device=self.device)
        C_var = torch.ones_like(m)
        delta_m = project_delta(self.z_to_delta(m), self.eps, self.norm)
        f_m, _ = self.evaluator.evaluate_blackbox(delta_m)

        history = [[float(f_m.item()), delta_m.cpu()]]
        num_evaluation = 1

        while num_evaluation < self.max_evaluation:
            noise = torch.randn((self.lam, C, H, W), device=device, generator=g_gpu)
            X = m + self.sigma * noise * torch.sqrt(C_var)

            X_delta = project_delta(self.z_to_delta(X), self.eps, self.norm)
            margins, l2s = self.evaluate_population(X_delta)
            num_evaluation += self.lam

            idx = torch.argsort(margins)[:self.mu]
            X_sel = X[idx]
            Y = X_sel - m

            m = torch.sum(self.weights.view(-1, 1, 1, 1, 1) * X_sel, dim=0, keepdim=True)
            C_var = (1 - self.c_cov) * C_var + self.c_cov * torch.sum(
                self.weights.view(-1, 1, 1, 1, 1) * (Y ** 2), dim=0, keepdim=True
            )

            delta_m = project_delta(self.z_to_delta(m), self.eps, self.norm)
            f_m, l2_m = self.evaluator.evaluate_blackbox(delta_m)
            num_evaluation += 1
            print(f"[{num_evaluation} - attack phase] Best loss: ", f_m )

            history.append([float(f_m), delta_m.cpu()])

            if self.is_success(f_m):
                break

        return {
            "best_delta": delta_m,
            "best_margin": float(f_m),
            "history": None,
            "num_evaluation": num_evaluation
        }

class PGDAttack(BaseAttack):
    def __init__(self, eps, alpha, norm, steps, evaluator):
        self.eps = eps
        self.alpha = alpha
        self.norm = norm
        self.steps = steps
        self.evaluator = evaluator
    
    def run(self):
        delta = torch.zeros_like(self.evaluator.img_tensor).to(self.evaluator.img_tensor.device)
        delta.requires_grad = True

        for step in range(self.steps):
            margin, _ = self.evaluator.evaluate_whitebox(delta)
            loss = margin.mean()
            print("Loss: ", loss)
            loss.backward()
            # if loss < 0:
            #     break
            with torch.no_grad():
                if self.norm == "linf":
                    delta.data = delta - self.alpha * delta.grad.sign()
                    delta.data = clamp_eps(delta.data, self.eps, norm="linf")
                elif self.norm == "l2":
                    grad_norm = torch.norm(delta.grad.view(delta.size(0), -1), dim=1).view(-1, 1, 1, 1)
                    scaled_grad = delta.grad / (grad_norm + 1e-10)
                    delta.data = delta - self.alpha * scaled_grad
                    delta.data = clamp_eps(delta.data, self.eps, norm="l2")
                delta.grad.zero_()

        final_margin, _ = self.evaluator.evaluate_whitebox(delta)
        return {
            "best_delta": delta.detach(),
            "best_margin": float(final_margin.item()),
            "history": None,
            "num_evaluation": step
        }


class ES_1_Lambda_Gradient(BaseAttack):
    def __init__(self, evaluator, eps=8/255, norm="linf",
                 theta=0.001, max_evaluation=10000, lam=64, c_inc=1.5, c_dec=0.9, device='cuda'):
        super().__init__(evaluator, eps, norm, device)
        # assert lam >= 2 and c_inc > 1.0 and 0.0 < c_dec < 1.0
        self.lam = int(lam)
        self.c_inc = float(c_inc)
        self.c_dec = float(c_dec)
        self.sigma = 1.1  # σ tuyệt đối
        self.max_evaluation = max_evaluation
        self.theta = theta  # hệ số điều chỉnh hướng gradient trắng
        self.device = 'cuda'

    def run(self) -> Dict[str, Any]:
        sigma = self.sigma
        theta = self.theta
        _, C, H, W = self.evaluator.img_tensor.shape

        m = torch.randn((1, C, H, W), device=self.device)

        delta_m = self.z_to_delta(m)
        delta_m = project_delta(delta_m, self.eps, self.norm)

        f_m, l2_m = self.evaluator.evaluate_blackbox(delta_m)
        history = [[float(f_m.item()), delta_m.cpu()]]

        num_evaluation = 1

        while num_evaluation < self.max_evaluation:

            # ===== 1. Compute gradient guidance (WHITEBOX) =====
            m.requires_grad_(True)
            delta = self.z_to_delta(m)
            delta = project_delta(delta, self.eps, self.norm)

            margin_wb, _ = self.evaluator.evaluate_whitebox(delta)
            num_evaluation += 1

            loss = margin_wb.mean()
            loss.backward()

            grad_m = m.grad.detach()
            # print("Gradient sum: ", grad_m.sum())
            m = m.detach()

            # normalize gradient (VERY IMPORTANT)
            grad_m = grad_m / (grad_m.norm() + 1e-8)

            theta = self.theta

            noise = torch.randn((self.lam, C, H, W), device=self.device, generator=g_gpu)

            X = m \
                + sigma * noise \
                - theta * grad_m

            X_delta = self.z_to_delta(X)
            X_delta = project_delta(X_delta, self.eps, self.norm)

            margins, l2s = self.evaluate_population(X_delta)
            num_evaluation += self.lam

            idx_best = torch.argmin(margins).item()
            f_best = float(margins[idx_best].item())

            if f_best < f_m:
                m = X[idx_best].clone()
                delta_m = X_delta[idx_best].clone()
                f_m = f_best
                l2_m = float(l2s[idx_best].item())
                sigma *= self.c_inc
            else:
                sigma *= self.c_dec
            
            # print(f"[{num_evaluation} - attack phase] Best loss: ", f_m, " L2: ", l2_m )

            # history.append([float(f_m), delta_m.cpu()])
            if self.is_success(f_m):
                break

        return {
            "best_delta": delta_m,
            "best_margin": f_m,
            "history": None,
            "num_evaluation": num_evaluation
        }
