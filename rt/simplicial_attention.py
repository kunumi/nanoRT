import math

import torch
from einops import rearrange
from torch import nn


def _empty_rel_batch(B, S, device, valid_fk_slots):
    empty = torch.empty(0, device=device, dtype=torch.long)
    return {
        "seg_q": empty,
        "bs_t": empty,
        "seg_r": empty,
        "r_t": empty,
        "B": B,
        "S": S,
        "T": 0,
        "valid_fk_slots": valid_fk_slots,
        "matched_fk_slots": 0,
    }


@torch.compiler.disable
def build_rel_batch(
    f2p_nbr_idxs,
    f2p_rel_idxs,
    node_idxs,
    is_padding,
    num_relations,
    need_stats=False,
):
    """Match FK parent IDs to cells and build flat gather/scatter indices.
    seg_q selects parent queries, bs_t child cells, and seg_r relation instances.
    Coverage statistics are optional to avoid extra GPU synchronization."""
    B, S, _ = f2p_nbr_idxs.shape
    device = f2p_nbr_idxs.device

    valid_sk = (f2p_nbr_idxs != -1) & (f2p_rel_idxs != -1) & ~is_padding[:, :, None]

    b_e, s_e, k_e = valid_sk.nonzero(as_tuple=True)
    b_q, q_pos = (~is_padding).nonzero(as_tuple=True)
    valid_fk_slots = b_e.numel()

    if b_e.numel() == 0 or b_q.numel() == 0:
        return _empty_rel_batch(B, S, device, valid_fk_slots)

    parent_e = f2p_nbr_idxs[b_e, s_e, k_e].long()
    rel_e = f2p_rel_idxs[b_e, s_e, k_e].long()
    q_nodes = node_idxs[b_q, q_pos].long()

    key_stride = torch.maximum(parent_e.max(), q_nodes.max()) + 2
    edge_key = b_e.long() * key_stride + parent_e
    query_key = b_q.long() * key_stride + q_nodes

    sorted_query_key, order = query_key.sort()
    sorted_q_pos = q_pos[order].long()

    left = torch.searchsorted(sorted_query_key, edge_key, right=False)
    right = torch.searchsorted(sorted_query_key, edge_key, right=True)
    counts = right - left
    has_query = counts > 0
    matched_fk_slots = int(has_query.sum().item()) if need_stats else -1

    b_e = b_e[has_query].long()
    s_e = s_e[has_query].long()
    rel_e = rel_e[has_query]
    left = left[has_query]
    counts = counts[has_query]

    # Size downstream tensors once; output_size avoids repeated synchronization.
    total = int(counts.sum().item())
    if total == 0:
        return _empty_rel_batch(B, S, device, valid_fk_slots)

    edge_rep = torch.repeat_interleave(
        torch.arange(counts.numel(), device=device), counts, output_size=total
    )
    group_starts = torch.repeat_interleave(
        counts.cumsum(0) - counts, counts, output_size=total
    )
    q_offsets = torch.arange(total, device=device) - group_starts
    q_t = sorted_q_pos[
        torch.repeat_interleave(left, counts, output_size=total) + q_offsets
    ]

    b_t = b_e[edge_rep]
    s_t = s_e[edge_rep]
    r_t = rel_e[edge_rep]

    return {
        "seg_q": b_t * S + q_t,
        "bs_t": b_t * S + s_t,
        "seg_r": b_t * num_relations + r_t,
        "r_t": r_t,
        "B": B,
        "S": S,
        "T": total,
        "valid_fk_slots": valid_fk_slots,
        "matched_fk_slots": matched_fk_slots,
    }


def _scatter_softmax(logits, seg, n_segs):
    """Compute numerically stable softmax within sparse integer segments."""
    H = logits.shape[1]

    with torch.no_grad():
        max_l = logits.new_full((n_segs, H), -1e9)
        max_l.index_reduce_(0, seg, logits, "amax", include_self=True)

    exp_l = (logits - max_l.index_select(0, seg)).exp()
    denom = logits.new_zeros(n_segs, H)
    denom.index_add_(0, seg, exp_l)
    return exp_l / denom.index_select(0, seg).clamp(min=1e-8)


def _mean_segment_entropy(attn, seg, n_segs):
    """Mean categorical entropy over active sparse segments and heads."""
    entropy_terms = -(attn.clamp(min=1e-8).log() * attn)
    entropy_by_segment = attn.new_zeros(n_segs, attn.shape[1])
    entropy_by_segment.index_add_(0, seg, entropy_terms)
    counts = attn.new_zeros(n_segs)
    counts.index_add_(0, seg, attn.new_ones(seg.shape[0]))
    active = counts > 0
    return (entropy_by_segment * active[:, None]).sum() / (
        active.sum() * attn.shape[1]
    ).clamp(min=1)


class SimplicialNbrAttn(nn.Module):
    """Update cells: parent q attends over sparse (child s, relation r) pairs."""

    def __init__(self, d_model, num_heads, qk_norm=True):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wkr = nn.Linear(d_model, d_model, bias=False)
        self.wvr = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        # Normalize each score factor to stabilize trilinear logits.
        norm = lambda: nn.RMSNorm(self.d_head) if qk_norm else nn.Identity()
        self.q_norm = norm()
        self.k_norm = norm()
        self.kr_norm = norm()

    def forward(self, x, r, rel_batch, need_stats=False):
        B, S, _ = x.shape
        H, D = self.num_heads, self.d_head
        seg_q, bs_t, seg_r = (rel_batch[k] for k in ("seg_q", "bs_t", "seg_r"))
        T = rel_batch["T"]

        q_proj = self.q_norm(rearrange(self.wq(x), "b s (h d) -> (b s) h d", h=H))
        k_proj = self.k_norm(rearrange(self.wk(x), "b s (h d) -> (b s) h d", h=H))
        v_proj = rearrange(self.wv(x), "b s (h d) -> (b s) h d", h=H)
        kr_proj = self.kr_norm(rearrange(self.wkr(r), "b r (h d) -> (b r) h d", h=H))
        vr_proj = rearrange(self.wvr(r), "b r (h d) -> (b r) h d", h=H)

        out = x.new_zeros(B * S, H * D)
        attn_entropy = x.new_zeros(()) if need_stats else None

        if T > 0:
            q_e = q_proj.index_select(0, seg_q)
            k_e = k_proj.index_select(0, bs_t)
            v_e = v_proj.index_select(0, bs_t)
            kr_e = kr_proj.index_select(0, seg_r)
            vr_e = vr_proj.index_select(0, seg_r)

            logits = (q_e * k_e * kr_e).sum(-1) / math.sqrt(D)
            attn = _scatter_softmax(logits, seg_q, B * S)

            out_vals = (attn.unsqueeze(-1) * (v_e * vr_e)).view(T, H * D)
            out.index_add_(0, seg_q, out_vals)

            if need_stats:
                attn_entropy = _mean_segment_entropy(attn, seg_q, B * S)

        return self.wo(out.view(B, S, H * D)), attn_entropy


class RelAttn(nn.Module):
    """Update relations: relation r attends over sparse (parent q, child s) pairs."""

    def __init__(self, d_model, num_heads, qk_norm=True):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.wqr = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        # Normalize relation queries and shared cell keys.
        norm = lambda: nn.RMSNorm(self.d_head) if qk_norm else nn.Identity()
        self.qr_norm = norm()
        self.k_norm = norm()

    def forward(self, r, x, rel_batch, need_stats=False):
        H, D = self.num_heads, self.d_head
        B, R, _ = r.shape
        seg_q, bs_t, seg_r = (rel_batch[k] for k in ("seg_q", "bs_t", "seg_r"))
        T = rel_batch["T"]

        qr_proj = self.qr_norm(rearrange(self.wqr(r), "b r (h d) -> (b r) h d", h=H))
        k_proj = self.k_norm(rearrange(self.wk(x), "b s (h d) -> (b s) h d", h=H))
        v_proj = rearrange(self.wv(x), "b s (h d) -> (b s) h d", h=H)

        out = r.new_zeros(B * R, H * D)
        attn_entropy = r.new_zeros(()) if need_stats else None

        if T > 0:
            qr_e = qr_proj.index_select(0, seg_r)
            kq_e = k_proj.index_select(0, seg_q)
            ks_e = k_proj.index_select(0, bs_t)
            vq_e = v_proj.index_select(0, seg_q)
            vs_e = v_proj.index_select(0, bs_t)

            logits = (qr_e * kq_e * ks_e).sum(-1) / math.sqrt(D)
            attn = _scatter_softmax(logits, seg_r, B * R)

            out_vals = (attn.unsqueeze(-1) * (vq_e * vs_e)).view(T, H * D)
            out.index_add_(0, seg_r, out_vals)

            if need_stats:
                attn_entropy = _mean_segment_entropy(attn, seg_r, B * R)

        return self.wo(out.view(B, R, H * D)), attn_entropy
