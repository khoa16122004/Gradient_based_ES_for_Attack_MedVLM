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
    def __init__(self, evaluator, pattern, eps=8/255, norm="linf",
                 max_evaluation=10000, lam=64, c_inc=1.5, c_dec=0.9, device='cuda'):
        super().__init__(evaluator, eps, norm, device)
        # assert lam >= 2 and c_inc > 1.0 and 0.0 < c_dec < 1.0
        self.lam = int(lam)
        self.c_inc = float(c_inc)
        self.c_dec = float(c_dec)
        self.sigma = 1.1  # σ tuyệt đối
        self.max_evaluation = max_evaluation
        self.pattern = pattern

    def run(self) -> Dict[str, Any]:
        sigma = self.sigma
        _, C, H, W = self.evaluator.img_tensor.shape
        
        m = torch.randn((1, C, H, W), device=self.device)
        if self.pattern == 'blur':
            m = blur(noise, k=11)

        delta_m = self.z_to_delta(m)
        delta_m = project_delta(delta_m, self.eps, self.norm)

        f_m, l2_m = self.evaluator.evaluate_blackbox(delta_m)
        f_m, l2_m = float(f_m.item()), float(l2_m.item())
        history = [(1, f_m)]
        success_evaluation = None
        num_evaluation = 1
        while num_evaluation < self.max_evaluation:
            # noise = torch.randn((self.lam, C, H, W), device=self.device)
            noise = torch.randn((self.lam, C, H, W), device=self.device, generator=g_gpu)
            if self.pattern == "blur":
                noise = blur(noise, k=11)
                
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
            else:
                sigma *= self.c_dec            
            
            print(f"[{num_evaluation} - attack phase] Best loss: ", f_m, " L2: ", l2_m )
            history.append((num_evaluation, float(f_m)))
            if self.is_success(f_m) and success_evaluation is None:
                success_evaluation = num_evaluation
                break

            
        return {"best_delta": delta_m, "best_margin": f_m, "history": history, "success_evaluation": success_evaluation, 'l2': l2_m}





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
        success_evaluation = None
        history = []
        for step in range(self.steps):
            margin, l2 = self.evaluator.evaluate_whitebox(delta)
            loss = margin.mean()
            history.append((step, float(loss.item())))
            # print("Loss: ", loss)
            loss.backward()
            if loss < 0 and success_evaluation is None:
                success_evaluation = success_evaluation
                
            with torch.no_grad():
                if self.norm == "linf":
                    grad_norm = torch.norm(delta.grad.view(delta.size(0), -1), dim=1).view(-1, 1, 1, 1)
                    scaled_grad = delta.grad / (grad_norm + 1e-10)
                    delta.data = delta - self.alpha * scaled_grad
                    delta.data = clamp_eps(delta.data, self.eps, norm="linf")
                elif self.norm == "l2":
                    grad_norm = torch.norm(delta.grad.view(delta.size(0), -1), dim=1).view(-1, 1, 1, 1)
                    scaled_grad = delta.grad / (grad_norm + 1e-10)
                    delta.data = delta - self.alpha * scaled_grad
                    delta.data = clamp_eps(delta.data, self.eps, norm="l2")
                delta.grad.zero_()

        final_margin, l2 = self.evaluator.evaluate_whitebox(delta)
        return {
            "best_delta": delta.detach(),
            "best_margin": float(final_margin.item()),
            "history": history,
            "success_evaluation": step,
            'l2': float(l2.item())
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
        history = [(1, float(f_m.item()))]
        success_evaluation = None
        num_evaluation = 1
        theta = self.theta

        while num_evaluation < self.max_evaluation:

            m.requires_grad_(True)
            delta = self.z_to_delta(m)
            delta = project_delta(delta, self.eps, self.norm)

            margin_wb, _ = self.evaluator.evaluate_whitebox(delta)
            num_evaluation += 1

            # calculate gradient
            loss = margin_wb.mean()
            loss.backward()
            grad_m = m.grad.detach()
            m = m.detach()
            m = m - theta * grad_m.sign() # gradient guided


            noise = torch.randn((self.lam, C, H, W), device=self.device, generator=g_gpu)
            X = m + sigma * noise

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

            history.append((num_evaluation, float(f_m)))
            if self.is_success(f_m) and success_evaluation is None:
                success_evaluation = num_evaluation

        return {
            "best_delta": delta_m,
            "best_margin": f_m,
            "history": history,
            "success_evaluation": success_evaluation
        }


class CEM_Attack(BaseAttack):
    def __init__(
        self,
        evaluator,
        eps=8/255,
        norm="linf",
        max_evaluation=10000,
        N=64,          # number of samples
        Ne=8,          # elite set size
        sigma_init=1.1,
        sigma_min=1e-3,
        device="cuda"
    ):
        super().__init__(evaluator, eps, norm, device)

        self.N = int(N)
        self.Ne = int(Ne)
        self.sigma_init = float(sigma_init)
        self.sigma_min = sigma_min
        self.max_evaluation = max_evaluation
        self.device = device

        assert self.Ne < self.N

    def run(self):

        _, C, H, W = self.evaluator.img_tensor.shape

        mu = torch.randn((1, C, H, W), device=self.device)
        sigma = self.sigma_init

        delta_mu = self.z_to_delta(mu)
        delta_mu = project_delta(delta_mu, self.eps, self.norm)

        f_mu, l2_mu = self.evaluator.evaluate_blackbox(delta_mu)
        num_evaluation = 1

        while num_evaluation < self.max_evaluation:

            noise = torch.randn(
                (self.N, C, H, W),
                device=self.device,
                generator=g_gpu
            )

            X = mu + sigma * noise

            # X_delta = self.z_to_delta(X)
            X_delta = project_delta(X_delta, self.eps, self.norm)

            margins, l2s = self.evaluate_population(X_delta)
            num_evaluation += self.N

            idx_sorted = torch.argsort(margins)
            elite_idx = idx_sorted[:self.Ne]

            Z = X[elite_idx]

            mu = Z.mean(dim=0)

            sigma = torch.sqrt(
                ((Z - mu) ** 2).mean()
            ).item()

            sigma = torch.sqrt(((Z - mu) ** 2).mean(dim=0))
            sigma = torch.clamp(sigma, min=self.sigma_min)
            
            f_mu = margins[elite_idx].mean().item()
            l2_mu = l2s[elite_idx].mean().item()

            if self.is_success(margins[elite_idx[0]].item()):
                delta_mu = X_delta[elite_idx[0]]
                break
            print(f"[{num_evaluation} - attack phase] Best loss: ", f_mu )

            delta_mu = self.z_to_delta(mu)
            delta_mu = project_delta(delta_mu, self.eps, self.norm)

        return {
            "best_delta": delta_mu,
            "best_margin": f_mu,
            "num_evaluation": num_evaluation
        }


class ESGD_Attack(BaseAttack):
    def __init__(
        self,
        evaluator,
        eps=8/255,
        norm="linf",
        mu=4,
        lam=16,
        m=2,
        Ks=3,
        Kv=1,
        alpha=0.01,
        sigma=0.5,
        max_evaluation=10000,
        device="cuda"
    ):
        super().__init__(evaluator, eps, norm, device)
        self.mu = mu
        self.lam = lam
        self.m = m
        self.Ks = Ks
        self.Kv = Kv
        self.alpha = alpha
        self.sigma = sigma
        self.max_evaluation = max_evaluation

    def sgd_refine(self, z):
        z = z.clone().detach().requires_grad_(True)
        best_z = z.detach().clone()

        delta = self.z_to_delta(z)
        best_f, _ = self.evaluator.evaluate_whitebox(
            project_delta(delta, self.eps, self.norm)
        )

        for _ in range(self.Ks):
            delta = self.z_to_delta(z)
            margin, _ = self.evaluator.evaluate_whitebox(delta)
            loss = margin.mean()
            loss.backward()

            with torch.no_grad():
                z -= self.alpha * z.grad
                z.grad.zero_()

            delta_new = self.z_to_delta(z)
            f_new, _ = self.evaluator.evaluate_whitebox(
                project_delta(delta_new, self.eps, self.norm)
            )

            if f_new < best_f:
                best_f = f_new
                best_z = z.detach().clone()

        return best_z.detach()

    def run(self):
        _, C, H, W = self.evaluator.img_tensor.shape

        population = torch.randn(
            (self.mu, C, H, W),
            device=self.device
        )

        num_eval = 0
        best_margin = float("inf")
        best_delta = None

        while num_eval < self.max_evaluation:

            for i in range(self.mu):
                population[i] = self.sgd_refine(population[i])

            deltas = project_delta(
                self.z_to_delta(population),
                self.eps,
                self.norm
            )

            margins, _ = self.evaluate_population(deltas)
            num_eval += self.mu

            idx_best = torch.argmin(margins)
            if margins[idx_best] < best_margin:
                best_margin = float(margins[idx_best])
                best_delta = deltas[idx_best].clone()

            if self.is_success(best_margin):
                break

            for _ in range(self.Kv):
                parents = population[
                    torch.randint(0, self.mu, (self.lam,), device=self.device)
                ]

                noise = torch.randn_like(parents)
                offspring = parents + self.sigma * noise

                all_pop = torch.cat([population, offspring], dim=0)

                all_delta = project_delta(
                    self.z_to_delta(all_pop),
                    self.eps,
                    self.norm
                )

                margins, _ = self.evaluate_population(all_delta)
                num_eval += all_pop.size(0)

                idx = torch.argsort(margins)

                elites = all_pop[idx[:self.m]]
                rest = all_pop[
                    idx[self.m:][
                        torch.randperm(len(idx) - self.m, device=self.device)
                        [: self.mu - self.m]
                    ]
                ]

                population = torch.cat([elites, rest], dim=0).squeeze(1)

            print(
                f"[Eval {num_eval}] "
                f"Best margin: {best_margin:.6f}"
            )

        return {
            "best_delta": best_delta,
            "best_margin": best_margin,
            "num_evaluation": num_eval
        }


class NES_Attack(BaseAttack):
    def __init__(
        self,
        evaluator,
        eps=8/255,
        norm="linf",
        max_evaluation=10000,
        q=200,              # tổng số direction
        batch_q=50,         # batch size để eval
        alpha=0.01,
        sigma=0.001,
        device="cuda",
    ):
        super().__init__(evaluator, eps, norm, device)
        self.q = q
        self.batch_q = batch_q
        self.sigma = sigma
        self.alpha = alpha
        self.max_evaluation = max_evaluation

    def run(self):
        _, C, H, W = self.evaluator.img_tensor.shape

        delta = torch.zeros((1, C, H, W), device=self.device)
        # delta = torch.clamp(delta, -self.eps, self.eps)
        delta = project_delta(delta, self.eps, self.norm)

        margin, _ = self.evaluator.evaluate_blackbox(delta)
        f_m = float(margin.item())

        history = [(1, f_m)]
        num_evaluation = 1
        success_evaluation = None

        while num_evaluation < self.max_evaluation:

            grad = torch.zeros_like(delta)
            used_q = 0

            while used_q < self.q and num_evaluation < self.max_evaluation:
                cur_q = min(self.batch_q, self.q - used_q)

                U = torch.randn(
                    (cur_q, C, H, W),
                    device=self.device,
                    generator=g_gpu
                )

                delta_pos = delta + self.sigma * U
                delta_neg = delta - self.sigma * U

                l_pos, _ = self.evaluate_population(delta_pos)
                l_neg, _ = self.evaluate_population(delta_neg)

                num_evaluation += 2 * cur_q
                used_q += cur_q

                grad += (
                    ((l_pos - l_neg).view(cur_q, 1, 1, 1) * U)
                    .sum(dim=0, keepdim=True)
                )

            grad = grad / (2 * self.sigma * used_q)

            # update
            delta = delta - self.alpha * grad.sign()
            delta = project_delta(delta, self.eps, self.norm)
            margin, l2 = self.evaluator.evaluate_blackbox(delta)
            f_m = float(margin.item())
            num_evaluation += 1
            print(
                f"[Eval {num_evaluation}] "
                f"Score: {f_m:.6f} l2: {l2}"
            )
            history.append((num_evaluation, f_m))

            if self.is_success(f_m) and success_evaluation is None:
                success_evaluation = num_evaluation
                break
        return {
            "best_delta": delta.detach(),
            "best_margin": f_m,
            "history": history,
            "success_evaluation": success_evaluation,
            'l2': float(l2.item())
        }


import torch.nn.functional as F

def blur(x, k=9):
    kernel = torch.ones(1, 1, k, k, device=x.device) / (k * k)
    return F.conv2d(
        x.view(-1, 1, *x.shape[-2:]),
        kernel,
        padding=k//2
    ).view(x.shape)

def grid_patch_mask(patch_idx, patch_size, H, W, device):
    gh = H // patch_size
    gw = W // patch_size

    i = patch_idx // gw
    j = patch_idx % gw

    mask = torch.zeros((1, 1, H, W), device=device)

    h0 = i * patch_size
    h1 = h0 + patch_size
    w0 = j * patch_size
    w1 = w0 + patch_size

    mask[:, :, h0:h1, w0:w1] = 1.0
    return mask

def grid_local_es(
    evaluator,
    base_delta,
    mask,
    eps,
    norm,
    lam=16,
    steps=10,
    sigma=0.5,
    device="cuda",
    start_eval=0
):
    _, C, H, W = base_delta.shape

    z = torch.zeros_like(base_delta)
    best_delta = base_delta.clone()

    f_best, _ = evaluator.evaluate_blackbox(best_delta)
    f_best = float(f_best)

    history = []
    eval_cnt = start_eval

    local_success_eval = None

    # print(f"    [LocalES] start | margin = {f_best:.6f}")

    for step in range(steps):

        noise = torch.randn((lam, C, H, W), device=device)
        noise = blur(noise, k=11)
        Z = z + sigma * noise

        deltas = base_delta + mask * torch.tanh(Z) * eps
        deltas = project_delta(deltas, eps, norm)

        margins, _ = evaluator.evaluate_blackbox(deltas)

        eval_cnt += lam
        idx = torch.argmin(margins)
        f_step = float(margins[idx])

        if f_step < f_best:
            z = Z[idx:idx+1].clone()
            best_delta = deltas[idx:idx+1].clone()
            f_best = f_step
            status = "ACCEPT"
        else:
            sigma *= 0.7
            status = "REJECT"

        history.append((eval_cnt, f_best))

        if f_best < 0 and local_success_eval is None:
            local_success_eval = eval_cnt

        # print(
        #     f"    [LocalES][{step+1:03d}/{steps}] "
        #     f"eval = {eval_cnt:6d} | "
        #     f"best = {f_best:.6f} | {status}"
        # )

    return best_delta, f_best, history, local_success_eval




class GridES_1_Lambda(BaseAttack):
    def __init__(
        self,
        evaluator,
        patch_size=16,
        eps=8/255,
        norm="linf",
        lam=16,
        local_steps=1000,
        sigma=0.5,
        max_evaluation=10000,
        device="cuda"
    ):
        super().__init__(evaluator, eps, norm, device)

        _, _, self.H, self.W = evaluator.img_tensor.shape
        self.patch = patch_size
        self.gh = self.H // patch_size
        self.gw = self.W // patch_size
        self.Np = self.gh * self.gw

        self.lam = lam
        self.local_steps = local_steps
        self.sigma = sigma
        self.max_evaluation = max_evaluation

    def run(self):

        delta = torch.zeros_like(self.evaluator.img_tensor)
        f_best, _ = self.evaluator.evaluate_blackbox(delta)
        f_best = float(f_best)

        used_eval = 1
        tried = set()
        round_id = 0

        history = [(used_eval, f_best)]
        success_evaluation = None

        print(f"[GridES] start | eval = {used_eval} | margin = {f_best:.6f}")

        while used_eval < self.max_evaluation:

            round_id += 1

            if len(tried) == self.Np:
                tried.clear()

            patch_idx = np.random.choice(
                list(set(range(self.Np)) - tried)
            )
            tried.add(patch_idx)

            print(
                f"\n[GridES][Round {round_id}] "
                f"Explore patch {patch_idx} | eval = {used_eval}"
            )

            mask = grid_patch_mask(
                patch_idx,
                self.patch,
                self.H,
                self.W,
                self.device
            )

            # ⭐ local ES trả về success tại local
            delta_new, f_new, local_history, local_success_eval = grid_local_es(
                evaluator=self.evaluator,
                base_delta=delta,
                mask=mask,
                eps=self.eps,
                norm=self.norm,
                lam=self.lam,
                steps=self.local_steps,
                sigma=self.sigma,
                device=self.device,
                start_eval=used_eval
            )

            # merge history
            history.extend(local_history)
            used_eval = history[-1][0]

            # ⭐ propagate success từ local (KHÔNG break)
            if success_evaluation is None and local_success_eval is not None:
                success_evaluation = local_success_eval

            if f_new < f_best:
                delta = delta_new
                f_best = f_new
                tried.clear()
                status = "IMPROVED"
            else:
                status = "NO-IMPROVE"

            # print(
            #     f"[GridES][Round {round_id}] "
            #     f"{status} | best margin = {f_best:.6f} | eval = {used_eval}"
            # )

        return {
            "best_delta": delta.detach(),
            "best_margin": f_best,
            "history": history,
            "success_evaluation": success_evaluation,
            "num_evaluation": used_eval
        }





