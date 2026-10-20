from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange
from einops._torch_specific import allow_ops_in_compiled_graph
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from rt.sigreg import SIGReg
from rt.simplicial_attention import RelAttn, SimplicialNbrAttn, build_rel_batch

allow_ops_in_compiled_graph()
flex_attention = torch.compile(flex_attention)


# Standard building blocks


class MaskedAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, block_mask):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        if block_mask is None:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                x = F.scaled_dot_product_attention(q, k, v)
        else:
            x = flex_attention(q, k, v, block_mask=block_mask)

        x = rearrange(x, "b h s d -> b s (h d)")
        x = self.wo(x)
        return x


class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# Transformer block


class RelationalBlock(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        use_simplicial=True,
        update_relations=True,
        qk_norm=True,
        layerscale_init=1.0,
    ):
        super().__init__()

        norm_names = ["feat", "col", "full", "ffn"]
        if use_simplicial:
            norm_names += ["nbr_x", "nbr_r"]
        self.norms = nn.ModuleDict({l: nn.RMSNorm(d_model) for l in norm_names})
        self.attns = nn.ModuleDict(
            {l: MaskedAttention(d_model, num_heads) for l in ["feat", "col", "full"]}
        )
        self.use_simplicial = use_simplicial
        # Static relations skip RelAttn updates.
        self.update_relations = update_relations and use_simplicial
        if use_simplicial:
            self.nbr_attn = SimplicialNbrAttn(d_model, num_heads, qk_norm=qk_norm)
            # LayerScale=1 preserves the record residual update.
            self.nbr_scale = nn.Parameter(torch.full((d_model,), layerscale_init))
            if self.update_relations:
                self.rel_attn = RelAttn(d_model, num_heads, qk_norm=qk_norm)
                self.rel_scale = nn.Parameter(torch.full((d_model,), layerscale_init))
        self.ffn = FFN(d_model, d_ff)

    def forward(self, x, r, block_masks, rel_batch, need_stats=False):
        for l in ["col", "feat"]:
            x = x + self.attns[l](self.norms[l](x), block_mask=block_masks[l])

        block_stats = None
        if self.use_simplicial:
            x_n = self.norms["nbr_x"](x)
            r_n = self.norms["nbr_r"](r)

            dx, nbr_entropy = self.nbr_attn(x_n, r_n, rel_batch, need_stats=need_stats)
            dx = dx * self.nbr_scale
            if self.update_relations:
                dr, rel_entropy = self.rel_attn(
                    r_n, x_n, rel_batch, need_stats=need_stats
                )
                dr = dr * self.rel_scale
            else:
                dr, rel_entropy = None, (x.new_zeros(()) if need_stats else None)
            if need_stats:
                nbr_delta_norm = dx.norm()
                rel_delta_norm = dr.norm() if dr is not None else x.new_zeros(())
                block_stats = {
                    "nbr_entropy": nbr_entropy,
                    "rel_entropy": rel_entropy,
                    "nbr_delta_norm": nbr_delta_norm,
                    "rel_delta_norm": rel_delta_norm,
                    "nbr_delta_frac": nbr_delta_norm / x.norm().clamp(min=1e-8),
                    "rel_delta_frac": rel_delta_norm / r.norm().clamp(min=1e-8),
                }
            x = x + dx
            if dr is not None:
                r = r + dr

        x = x + self.attns["full"](
            self.norms["full"](x), block_mask=block_masks["full"]
        )
        x = x + self.ffn(self.norms["ffn"](x))

        return x, r, block_stats


# Block mask helpers


def _make_block_mask(mask, batch_size, seq_len, device):
    def _mod(b, h, q_idx, kv_idx):
        return mask[b, q_idx, kv_idx]

    return create_block_mask(
        mask_mod=_mod,
        B=batch_size,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
        _compile=True,
    )


# Full model


class RelationalTransformer(nn.Module):
    def __init__(
        self,
        num_blocks,
        d_model,
        d_text,
        num_heads,
        d_ff,
        simplicial_layers=None,
        update_relations=True,
        qk_norm=True,
        simplicial_layerscale_init=1.0,
        # SIGReg: early-layer geometry regularization (None = disabled).
        sigreg_layer=None,
        sigreg_n_proj=64,
        sigreg_proj_hidden_mult=4,
        sigreg_proj_out_div=4,
        sigreg_proj_hidden_layers=1,
        sigreg_knots=17,
        sigreg_t_max=3.0,
        sigreg_min_group=8,
        sigreg_col_global_pool=True,
        sigreg_use_row=True,
        sigreg_use_col=True,
    ):
        """Build typed attention blocks and optional SIGReg.
        Use simplicial_layers=[] to disable typed attention, and
        update_relations=False to keep relation states fixed within a forward pass."""
        super().__init__()

        if d_model % num_heads:
            raise ValueError(
                f"d_model={d_model} must be divisible by num_heads={num_heads}"
            )

        if simplicial_layers is None:
            simplicial_layers = range(num_blocks)
        simplicial_layers = set(simplicial_layers)
        invalid_layers = sorted(i for i in simplicial_layers if not 0 <= i < num_blocks)
        if invalid_layers:
            raise ValueError(
                f"simplicial layers out of range [0, {num_blocks}): {invalid_layers}"
            )
        self.has_simplicial = len(simplicial_layers) > 0
        # The last relation update has no consumer; omit its parameters.
        _last_simplicial = max(simplicial_layers) if simplicial_layers else None

        self.enc_dict = nn.ModuleDict(
            {
                "number": nn.Linear(1, d_model, bias=True),
                "text": nn.Linear(d_text, d_model, bias=True),
                "datetime": nn.Linear(1, d_model, bias=True),
                "col_name": nn.Linear(d_text, d_model, bias=True),
                "boolean": nn.Linear(1, d_model, bias=True),
                "relation": nn.Linear(d_text, d_model, bias=True),
            }
        )
        self.dec_dict = nn.ModuleDict(
            {
                "number": nn.Linear(d_model, 1, bias=True),
                "text": nn.Linear(d_model, d_text, bias=True),
                "datetime": nn.Linear(d_model, 1, bias=True),
                "boolean": nn.Linear(d_model, 1, bias=True),
            }
        )
        self.norm_dict = nn.ModuleDict(
            {
                "number": nn.RMSNorm(d_model),
                "text": nn.RMSNorm(d_model),
                "datetime": nn.RMSNorm(d_model),
                "col_name": nn.RMSNorm(d_model),
                "boolean": nn.RMSNorm(d_model),
                "relation": nn.RMSNorm(d_model),
            }
        )
        self.mask_embs = nn.ParameterDict(
            {
                t: nn.Parameter(torch.randn(d_model))
                for t in ["number", "text", "datetime", "boolean"]
            }
        )
        self.blocks = nn.ModuleList(
            [
                RelationalBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    use_simplicial=(i in simplicial_layers),
                    update_relations=(update_relations and i != _last_simplicial),
                    qk_norm=qk_norm,
                    layerscale_init=simplicial_layerscale_init,
                )
                for i in range(num_blocks)
            ]
        )
        self.norm_out = nn.RMSNorm(d_model)
        self.d_model = d_model

        # Regularize the selected block output with SIGReg.
        self.sigreg_layer = sigreg_layer
        self.sigreg_use_row = sigreg_use_row
        self.sigreg_use_col = sigreg_use_col
        if sigreg_layer is not None:
            if not 0 <= sigreg_layer < num_blocks:
                raise ValueError(
                    f"sigreg_layer={sigreg_layer} out of range for {num_blocks} blocks"
                )
            self.sigreg = SIGReg(
                d_model,
                n_proj=sigreg_n_proj,
                proj_hidden_mult=sigreg_proj_hidden_mult,
                proj_out_div=sigreg_proj_out_div,
                proj_hidden_layers=sigreg_proj_hidden_layers,
                knots=sigreg_knots,
                t_max=sigreg_t_max,
                min_group=sigreg_min_group,
                col_global_pool=sigreg_col_global_pool,
            )
        else:
            self.sigreg = None

    def forward(
        self,
        batch,
        return_embeddings: bool = False,
        need_stats: bool = True,
    ):
        node_idxs = batch["node_idxs"]
        f2p_nbr_idxs = batch["f2p_nbr_idxs"]
        f2p_rel_idxs = batch["f2p_rel_idxs"]
        col_name_idxs = batch["col_name_idxs"]
        table_name_idxs = batch["table_name_idxs"]
        is_padding = batch["is_padding"]
        batch_size, seq_len = node_idxs.shape
        device = node_idxs.device

        # Attention masks
        pad = (~is_padding[:, :, None]) & (~is_padding[:, None, :])
        same_node = node_idxs[:, :, None] == node_idxs[:, None, :]
        kv_in_f2p = (node_idxs[:, None, :, None] == f2p_nbr_idxs[:, :, None, :]).any(-1)

        same_col_table = (col_name_idxs[:, :, None] == col_name_idxs[:, None, :]) & (
            table_name_idxs[:, :, None] == table_name_idxs[:, None, :]
        )

        attn_masks = {
            "feat": (same_node | kv_in_f2p) & pad,
            "col": same_col_table & pad,
            "full": pad,
        }
        attn_masks = {name: mask.contiguous() for name, mask in attn_masks.items()}

        make_bm = partial(
            _make_block_mask, batch_size=batch_size, seq_len=seq_len, device=device
        )
        block_masks = {l: make_bm(m) for l, m in attn_masks.items()}

        # Edge list for simplicial attention
        num_relations = batch["relation_embs"].shape[-2]
        rel_batch = None
        if self.has_simplicial:
            rel_batch = build_rel_batch(
                f2p_nbr_idxs,
                f2p_rel_idxs,
                node_idxs,
                is_padding,
                num_relations=num_relations,
                need_stats=need_stats,
            )

        # Relation stream init
        r = self.norm_dict["relation"](
            self.enc_dict["relation"](batch["relation_embs"])
        )
        r_mask = batch["r_mask"]
        if r.dim() == 2:
            r = r.unsqueeze(0).expand(batch_size, -1, -1).clone()
            r_mask = r_mask.unsqueeze(0).expand(batch_size, -1)
        elif r.dim() == 3 and r_mask.dim() == 1:
            r_mask = r_mask.unsqueeze(0).expand(batch_size, -1)
        r = r * r_mask[..., None]
        r_norm_init = r.norm() if need_stats else None

        # Cell stream init
        x = 0
        x = x + (
            self.norm_dict["col_name"](
                self.enc_dict["col_name"](batch["col_name_values"])
            )
            * (~is_padding)[..., None]
        )
        for i, t in enumerate(["number", "text", "datetime", "boolean"]):
            x = x + (
                self.norm_dict[t](self.enc_dict[t](batch[t + "_values"]))
                * ((batch["sem_types"] == i) & ~batch["masks"] & ~is_padding)[..., None]
            )
            x = x + (
                self.mask_embs[t]
                * ((batch["sem_types"] == i) & batch["masks"] & ~is_padding)[..., None]
            )

        # Transformer blocks
        block_stats = []
        sig_x = None
        for i, block in enumerate(self.blocks):
            x, r, stats_i = block(x, r, block_masks, rel_batch, need_stats=need_stats)
            # Re-mask relation padding after biased projections.
            if self.has_simplicial:
                r = r * r_mask[..., None]
            if stats_i is not None:
                block_stats.append(stats_i)
            # Capture post-block states before final normalization.
            if self.sigreg is not None and i == self.sigreg_layer:
                sig_x = x

        x = self.norm_out(x)

        if return_embeddings:
            return x

        # Compute diagnostic reductions only on logging steps.
        stats = {}
        if need_stats and rel_batch is not None:
            r_norm_final = r.norm()
            nbr_entropies = torch.stack([s["nbr_entropy"] for s in block_stats])
            rel_entropies = torch.stack([s["rel_entropy"] for s in block_stats])
            nbr_delta_fracs = torch.stack([s["nbr_delta_frac"] for s in block_stats])
            rel_delta_fracs = torch.stack([s["rel_delta_frac"] for s in block_stats])
            nbr_delta_norms = torch.stack([s["nbr_delta_norm"] for s in block_stats])
            rel_delta_norms = torch.stack([s["rel_delta_norm"] for s in block_stats])
            valid_nbr_slots = ((f2p_nbr_idxs != -1) & ~is_padding[:, :, None]).sum()
            valid_rel_slots = ((f2p_rel_idxs != -1) & ~is_padding[:, :, None]).sum()
            mismatched_fk_rel_slots = (
                ((f2p_nbr_idxs != -1) ^ (f2p_rel_idxs != -1)) & ~is_padding[:, :, None]
            ).sum()
            matched_fk_slots = rel_batch["matched_fk_slots"]
            valid_fk_slots = rel_batch["valid_fk_slots"]
            stats = {
                "simplicial/valid_nbr_slots": valid_nbr_slots,
                "simplicial/valid_rel_slots": valid_rel_slots,
                "simplicial/mismatched_fk_rel_slots": mismatched_fk_rel_slots,
                "simplicial/valid_fk_slots": float(valid_fk_slots),
                "simplicial/matched_fk_slots": float(matched_fk_slots),
                "simplicial/parent_match_rate": float(matched_fk_slots)
                / max(valid_fk_slots, 1),
                "simplicial/edge_count": float(rel_batch["T"]),
                "simplicial/r_table_size": float(r.shape[1]),
                "simplicial/r_norm_init": r_norm_init,
                "simplicial/r_norm_final": r_norm_final,
                "simplicial/r_update_frac": r_norm_final / r_norm_init.clamp(min=1e-8),
                "simplicial/nbr_entropy": nbr_entropies.mean(),
                "simplicial/rel_entropy": rel_entropies.mean(),
                "simplicial/nbr_delta_frac": nbr_delta_fracs.mean(),
                "simplicial/rel_delta_frac": rel_delta_fracs.mean(),
                "simplicial/nbr_delta_norm": nbr_delta_norms.mean(),
                "simplicial/rel_delta_norm": rel_delta_norms.mean(),
            }
            if rel_batch["T"] > 0:
                n_active_q = rel_batch["seg_q"].unique().numel()
                n_active_relation_types = rel_batch["r_t"].unique().numel()
                n_active_relation_instances = rel_batch["seg_r"].unique().numel()
                stats["simplicial/active_query_cells"] = float(n_active_q)
                stats["simplicial/active_relation_types"] = float(
                    n_active_relation_types
                )
                stats["simplicial/active_relation_instances"] = float(
                    n_active_relation_instances
                )
                stats["simplicial/edges_per_q"] = float(rel_batch["T"]) / max(
                    n_active_q, 1
                )
                stats["simplicial/triples_per_matched_fk_slot"] = float(
                    rel_batch["T"]
                ) / max(matched_fk_slots, 1)
                stats["simplicial/active_relation_type_frac"] = float(
                    n_active_relation_types
                ) / max(r.shape[1], 1)
            else:
                stats["simplicial/active_query_cells"] = 0.0
                stats["simplicial/active_relation_types"] = 0.0
                stats["simplicial/active_relation_instances"] = 0.0
                stats["simplicial/edges_per_q"] = 0.0
                stats["simplicial/triples_per_matched_fk_slot"] = 0.0
                stats["simplicial/active_relation_type_frac"] = 0.0

        # Loss
        loss_out = x.new_zeros(())
        yhat_out = {t: None for t in ["number", "text", "datetime", "boolean"]}

        sem_types = batch["sem_types"]
        masks = batch["masks"].bool()

        for i, t in enumerate(["number", "text", "datetime", "boolean"]):
            yhat = self.dec_dict[t](x)
            y = batch[f"{t}_values"]
            sem_type_mask = (sem_types == i) & masks

            if not sem_type_mask.any():
                loss_out = loss_out + (yhat.sum() * 0.0)
                yhat_out[t] = yhat
                continue

            if t in ("number", "datetime"):
                loss_t = F.huber_loss(yhat, y, reduction="none").mean(-1)
            elif t == "boolean":
                loss_t = F.binary_cross_entropy_with_logits(
                    yhat, (y > 0).float(), reduction="none"
                ).mean(-1)
            elif t == "text":
                raise ValueError("masking text not supported")

            loss_out = loss_out + (loss_t * sem_type_mask).sum()
            yhat_out[t] = yhat

        loss_out = loss_out / masks.sum().clamp(min=1)

        # SIGReg runs only during training and returns scalar losses.
        if self.sigreg is not None and self.training and sig_x is not None:
            sigreg_row, sigreg_col, sigreg_stats = self.sigreg(
                sig_x,
                node_idxs,
                table_name_idxs,
                col_name_idxs,
                is_padding,
                use_row=self.sigreg_use_row,
                use_col=self.sigreg_use_col,
            )
        else:
            sigreg_row = x.new_zeros((), dtype=torch.float32)
            sigreg_col = x.new_zeros((), dtype=torch.float32)
            sigreg_stats = {
                f"{view}/{key}": sigreg_row
                for view in ("row", "col")
                for key in (
                    "groups_present",
                    "groups_valid",
                    "groups_valid_frac",
                    "group_size_present_mean",
                    "group_size_valid_mean",
                    "group_size_valid_max",
                    "cells_covered_frac",
                )
            }
        stats["sigreg_row"] = sigreg_row
        stats["sigreg_col"] = sigreg_col
        stats.update({f"sigreg_{k}": v for k, v in sigreg_stats.items()})

        return loss_out, yhat_out, stats
