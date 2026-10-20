"""Single-GPU Muon, following https://github.com/KellerJordan/Muon.
Uses Nesterov momentum and bfloat16 Newton–Schulz iterations."""

import torch


def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Approximate the polar factor with quintic Newton–Schulz iterations."""
    if G.ndim < 2:
        raise ValueError(
            "Newton-Schulz orthogonalisation requires a tensor with at least "
            "two dimensions."
        )
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Normalise so the largest singular value is ≤ 1 before iterating.
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Orthogonalize momentum updates for matrix parameters."""

    def __init__(
        self,
        params,
        lr: float = 5e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim < 2:
                    raise ValueError(
                        "Muon expects 2-D parameters; use AdamW for 1-D params."
                    )

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - momentum)
                g = g.lerp_(buf, momentum) if nesterov else buf

                g = _zeropower_via_newtonschulz5(g, steps=ns_steps)
                # Scale update so RMS matches 1/lr (same convention as AdamW).
                g = g * max(1, g.size(-2) / g.size(-1)) ** 0.5
                wd = group["weight_decay"]
                if wd > 0:
                    p.mul_(1 - lr * wd)
                p.add_(g.to(p.dtype), alpha=-lr)

        return loss
