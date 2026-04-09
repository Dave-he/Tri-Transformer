import math
import torch
from typing import Optional


class GaLoreProjector:
    def __init__(
        self,
        rank: int = 128,
        update_proj_gap: int = 200,
        scale: float = 1.0,
    ):
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix: Optional[torch.Tensor] = None
        self._step_count = 0

    def _update_subspace(self, grad: torch.Tensor):
        m, n = grad.shape
        if m >= n:
            U, _, _ = torch.linalg.svd(grad, full_matrices=False)
            self.ortho_matrix = U[:, :self.rank]
            self._tall = True
        else:
            _, _, Vt = torch.linalg.svd(grad, full_matrices=False)
            self.ortho_matrix = Vt[:self.rank].t()
            self._tall = False

    def project(self, grad: torch.Tensor) -> torch.Tensor:
        if self.ortho_matrix is None or self._step_count % self.update_proj_gap == 0:
            self._update_subspace(grad)
        self._step_count += 1
        if self._tall:
            return self.ortho_matrix.t() @ grad
        else:
            return grad @ self.ortho_matrix

    def project_back(self, low_rank_grad: torch.Tensor) -> torch.Tensor:
        if self.ortho_matrix is None:
            raise RuntimeError("project() must be called before project_back()")
        if self._tall:
            restored = self.ortho_matrix @ low_rank_grad
        else:
            restored = low_rank_grad @ self.ortho_matrix.t()
        return restored * self.scale


class GaLoreAdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self._projectors: dict = {}

    def _get_projector(self, param_id: int, rank: int, update_proj_gap: int, scale: float) -> GaLoreProjector:
        if param_id not in self._projectors:
            self._projectors[param_id] = GaLoreProjector(
                rank=rank,
                update_proj_gap=update_proj_gap,
                scale=scale,
            )
        return self._projectors[param_id]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            rank = group.get("rank", None)
            update_proj_gap = group.get("update_proj_gap", 200)
            scale = group.get("scale", 1.0)
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            params = group["params"]
            if isinstance(params, torch.Tensor):
                params = [params]

            for p in params:
                if p.grad is None:
                    continue

                grad = p.grad
                use_galore = rank is not None and p.ndim >= 2

                if use_galore:
                    pid = id(p)
                    projector = self._get_projector(pid, rank, update_proj_gap, scale)
                    grad_proj = projector.project(grad)
                else:
                    grad_proj = grad

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(grad_proj)
                    state["exp_avg_sq"] = torch.zeros_like(grad_proj)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = betas
                state["step"] += 1
                t = state["step"]

                exp_avg.mul_(beta1).add_(grad_proj, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_proj, grad_proj, value=1 - beta2)

                bias_c1 = 1 - beta1 ** t
                bias_c2 = 1 - beta2 ** t
                step_size = lr / bias_c1

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_c2)).add_(eps)
                norm_grad = exp_avg / denom

                if use_galore:
                    norm_grad = projector.project_back(norm_grad)

                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                p.add_(norm_grad, alpha=-step_size)

        return loss
