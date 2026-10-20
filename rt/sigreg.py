"""Relational SIGReg adapted from LeJEPA (arXiv:2511.08544) and
"Optimizing Pre-Training of Tabular Foundation Models by Shaping Geometry".
Group cells by node or (table, column); compute all regularization in fp32."""

import torch
import torch.nn.functional as F
from torch import nn

# Compare empirical and Gaussian characteristic functions.
# Integrate their squared difference with a Gaussian window on [0, t_max].


class MaskedBatchNorm1d(nn.Module):
    """Normalize valid cells only, without running statistics."""

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # x: (B, S, C) fp32 ; valid: (B, S) bool
        m = valid[..., None].to(x.dtype)  # (B, S, 1)
        n = m.sum().clamp(min=1.0)  # scalar: #valid cells
        mean = (x * m).sum(dim=(0, 1)) / n  # (C,)
        var = (((x - mean) ** 2) * m).sum(dim=(0, 1)) / n
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


class Projector(nn.Module):
    """Project cell embeddings through an MLP with masked BatchNorm."""

    def __init__(
        self,
        d_in: int,
        hidden_mult: int = 4,
        out_div: int = 4,
        hidden_layers: int = 1,
    ):
        super().__init__()
        d_hidden = d_in * hidden_mult
        d_out = max(1, d_in // out_div)
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        prev = d_in
        for _ in range(hidden_layers):
            self.lins.append(nn.Linear(prev, d_hidden, bias=False))
            self.bns.append(MaskedBatchNorm1d(d_hidden))
            prev = d_hidden
        self.out = nn.Linear(prev, d_out, bias=True)
        self.d_out = d_out

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        for lin, bn in zip(self.lins, self.bns):
            x = F.relu(bn(lin(x), valid))
        return self.out(x)


def _dense_group_ids(keys: torch.Tensor) -> torch.Tensor:
    """Map distinct keys to contiguous group IDs within each sequence."""
    sorted_keys, sort_idx = keys.sort(dim=1)
    is_new = torch.ones_like(sorted_keys, dtype=torch.bool)
    is_new[:, 1:] = sorted_keys[:, 1:] != sorted_keys[:, :-1]
    dense_sorted = is_new.to(torch.int64).cumsum(dim=1) - 1
    gid = torch.empty_like(keys, dtype=torch.int64)
    gid.scatter_(1, sort_idx, dense_sorted)
    return gid


def _dense_group_ids_global(keys: torch.Tensor) -> torch.Tensor:
    """Map distinct keys to contiguous group IDs across the batch."""
    flat = keys.reshape(-1)
    sorted_keys, sort_idx = flat.sort()
    is_new = torch.ones_like(sorted_keys, dtype=torch.bool)
    is_new[1:] = sorted_keys[1:] != sorted_keys[:-1]
    dense_sorted = is_new.to(torch.int64).cumsum(0) - 1
    gid = torch.empty_like(flat, dtype=torch.int64)
    gid.scatter_(0, sort_idx, dense_sorted)
    return gid


def _pair_nonnegative_ids(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Pair non-negative IDs without collisions using int64 Cantor pairing."""
    first = first.to(torch.int64)
    second = second.to(torch.int64)
    total = first + second
    return total * (total + 1) // 2 + second


def _grouped_epps_pulley(
    z: torch.Tensor,  # (B, S, D) fp32 projected embeddings
    flat_idx: torch.Tensor,  # (B*S,) int64 group slot of each cell in [0, n_slots)
    n_slots: int,  # static size of the group buffer (= B*S)
    valid: torch.Tensor,  # (B, S) bool valid (non-padding) cells
    dirs: torch.Tensor,  # (D, K) fp32 unit-norm projection directions
    t: torch.Tensor,  # (T,) fp32 quadrature nodes
    phi: torch.Tensor,  # (T,) fp32 = exp(-t²/2) = φ_N(t) = window
    weights: torch.Tensor,  # (T,) fp32 trapezoidal weights × window
    min_group: float,  # groups with fewer valid cells are ignored
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute grouped Epps–Pulley statistics and normalize by eligible cell count."""
    B, S, _ = z.shape
    K = dirs.shape[1]
    T = t.shape[0]
    F_ = K * T

    vmask = valid.to(z.dtype)  # (B, S)
    p = z @ dirs  # (B, S, K)
    x_t = p.unsqueeze(-1) * t  # (B, S, K, T)
    w_cells = vmask[..., None, None]  # zero out padding cells
    cos = (torch.cos(x_t) * w_cells).reshape(B * S, F_)
    sin = (torch.sin(x_t) * w_cells).reshape(B * S, F_)

    sum_cos = torch.zeros(n_slots, F_, dtype=z.dtype, device=z.device).index_add_(
        0, flat_idx, cos
    )
    sum_sin = torch.zeros(n_slots, F_, dtype=z.dtype, device=z.device).index_add_(
        0, flat_idx, sin
    )
    cnt = torch.zeros(n_slots, dtype=z.dtype, device=z.device).index_add_(
        0, flat_idx, vmask.reshape(B * S)
    )  # (n_slots,) valid count/group

    denom = cnt.clamp(min=1.0).unsqueeze(-1)  # (n_slots, 1)
    cos_mean = (sum_cos / denom).reshape(n_slots, K, T)
    sin_mean = (sum_sin / denom).reshape(n_slots, K, T)
    err = (cos_mean - phi) ** 2 + sin_mean**2  # (n_slots, K, T)
    stat = (err @ weights) * cnt.unsqueeze(-1)  # (n_slots, K) — scaled by N_g
    stat = stat.mean(dim=-1)  # (n_slots,) mean over slices

    group_present = (cnt > 0).to(z.dtype)
    group_ok = (cnt >= min_group).to(z.dtype)  # (n_slots,)
    num_present = group_present.sum()
    num_groups = group_ok.sum()
    num_groups_safe = num_groups.clamp(min=1.0)
    valid_cells = (cnt * group_ok).sum()
    valid_cells_safe = valid_cells.clamp(min=1.0)
    total_cells = cnt.sum().clamp(min=1.0)

    # N_g-weighted mean of the per-group statistic (scale-invariant to grouping).
    loss = (stat * group_ok).sum() / valid_cells_safe
    stats = {
        "groups_present": num_present,
        "groups_valid": num_groups,
        "groups_valid_frac": num_groups / num_present.clamp(min=1.0),
        "group_size_present_mean": cnt.sum() / num_present.clamp(min=1.0),
        "group_size_valid_mean": valid_cells / num_groups_safe,
        "group_size_valid_max": (cnt * group_ok).max(),
        "cells_covered_frac": valid_cells / total_cells,
    }
    return loss, stats


def _zero_stats(device, dtype=torch.float32) -> dict[str, torch.Tensor]:
    zero = torch.zeros((), device=device, dtype=dtype)
    return {
        "groups_present": zero,
        "groups_valid": zero,
        "groups_valid_frac": zero,
        "group_size_present_mean": zero,
        "group_size_valid_mean": zero,
        "group_size_valid_max": zero,
        "cells_covered_frac": zero,
    }


class SIGReg(nn.Module):
    """Row- and column-wise SIGReg for nanoRT relational hidden states."""

    def __init__(
        self,
        d_model: int,
        n_proj: int = 64,
        proj_hidden_mult: int = 4,
        proj_out_div: int = 4,
        proj_hidden_layers: int = 1,
        knots: int = 17,
        t_max: float = 3.0,
        min_group: int = 8,
        col_global_pool: bool = True,
    ):
        super().__init__()
        self.row_proj = Projector(
            d_model, proj_hidden_mult, proj_out_div, proj_hidden_layers
        )
        self.col_proj = Projector(
            d_model, proj_hidden_mult, proj_out_div, proj_hidden_layers
        )
        self.n_proj = n_proj
        self.min_group = float(min_group)
        # Optionally pool each feature across the full batch.
        self.col_global_pool = col_global_pool

        t = torch.linspace(0.0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        w = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        w[0] = dt
        w[-1] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)  # φ_N(t) and window
        self.register_buffer("weights", w * window)

    def _sample_dirs(self, d: int, device) -> torch.Tensor:
        a = torch.randn(d, self.n_proj, device=device, dtype=torch.float32)
        return a / (a.norm(dim=0, keepdim=True) + 1e-8)

    def forward(
        self,
        x: torch.Tensor,  # (B, S, E) hidden state at the chosen block
        node_idxs: torch.Tensor,  # (B, S) int
        table_name_idxs: torch.Tensor,
        col_name_idxs: torch.Tensor,
        is_padding: torch.Tensor,  # (B, S) bool
        use_row: bool = True,
        use_col: bool = True,
    ):
        x = x.float()
        valid = ~is_padding
        B, S = node_idxs.shape
        n_slots = B * S
        zero = torch.zeros((), device=x.device, dtype=torch.float32)
        stats: dict[str, torch.Tensor] = {}

        # Row groups are sequence-local.
        if use_row:
            row_gid = _dense_group_ids(node_idxs.to(torch.int64))  # (B, S) in [0, S)
            row_base = (
                torch.arange(B, device=x.device, dtype=torch.int64) * S
            ).unsqueeze(1)
            row_flat = (row_base + row_gid).reshape(n_slots)
            z_row = self.row_proj(x, valid)
            row_loss, row_stats = _grouped_epps_pulley(
                z_row,
                row_flat,
                n_slots,
                valid,
                self._sample_dirs(z_row.shape[-1], x.device),
                self.t,
                self.phi,
                self.weights,
                self.min_group,
            )
        else:
            row_loss = zero
            row_stats = _zero_stats(x.device)
        for key, value in row_stats.items():
            stats[f"row/{key}"] = value

        # Column groups use (table, column) feature IDs.
        if use_col:
            # Sanitize padding IDs before pairing to avoid int64 overflow.
            safe_table_ids = torch.where(valid, table_name_idxs, 0)
            safe_column_ids = torch.where(valid, col_name_idxs, 0)
            col_keys = _pair_nonnegative_ids(safe_table_ids, safe_column_ids)
            col_keys = torch.where(
                valid,
                col_keys,
                torch.full_like(col_keys, -1),
            )
            if self.col_global_pool:
                col_flat = _dense_group_ids_global(col_keys)  # (B*S,) in [0, B*S)
            else:
                col_gid = _dense_group_ids(col_keys)
                col_base = (
                    torch.arange(B, device=x.device, dtype=torch.int64) * S
                ).unsqueeze(1)
                col_flat = (col_base + col_gid).reshape(n_slots)
            z_col = self.col_proj(x, valid)
            col_loss, col_stats = _grouped_epps_pulley(
                z_col,
                col_flat,
                n_slots,
                valid,
                self._sample_dirs(z_col.shape[-1], x.device),
                self.t,
                self.phi,
                self.weights,
                self.min_group,
            )
        else:
            col_loss = zero
            col_stats = _zero_stats(x.device)
        for key, value in col_stats.items():
            stats[f"col/{key}"] = value

        return row_loss, col_loss, stats
