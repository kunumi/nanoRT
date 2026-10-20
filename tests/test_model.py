"""Exercise the retained training recipe on a small CPU batch."""

from unittest.mock import patch

import torch

from rt.model import RelationalTransformer
from rt.optim import build_muon_param_groups


def test_training_loss_reaches_attention_and_sigreg_parameters():
    torch.manual_seed(0)
    seq_len, d_text = 4, 6
    batch = {
        "node_idxs": torch.tensor([[10, 20, 30, 40]]),
        "f2p_nbr_idxs": torch.full((1, seq_len, 5), -1, dtype=torch.long),
        "f2p_rel_idxs": torch.full((1, seq_len, 5), -1, dtype=torch.long),
        "col_name_idxs": torch.zeros((1, seq_len), dtype=torch.long),
        "table_name_idxs": torch.zeros((1, seq_len), dtype=torch.long),
        "is_padding": torch.zeros((1, seq_len), dtype=torch.bool),
        "col_name_values": torch.randn(1, seq_len, d_text),
        "sem_types": torch.zeros((1, seq_len), dtype=torch.long),
        "masks": torch.tensor([[True, False, True, False]]),
        "number_values": torch.randn(1, seq_len, 1),
        "text_values": torch.randn(1, seq_len, d_text),
        "datetime_values": torch.zeros(1, seq_len, 1),
        "boolean_values": torch.zeros(1, seq_len, 1),
        "relation_embs": torch.randn(1, d_text),
        "r_mask": torch.ones(1, dtype=torch.bool),
    }
    batch["f2p_nbr_idxs"][0, 1:, 0] = 10
    batch["f2p_rel_idxs"][0, 1:, 0] = 0
    model = RelationalTransformer(
        num_blocks=2,
        d_model=8,
        d_text=d_text,
        num_heads=2,
        d_ff=16,
        update_relations=False,
        sigreg_layer=0,
        sigreg_min_group=2,
        sigreg_use_row=False,
    )

    # The production block-mask kernel targets CUDA. Dense SDPA lets this
    # smoke test cover the surrounding model and sparse FK operator on CPU.
    with patch("rt.model._make_block_mask", return_value=None):
        supervised, predictions, stats = model(batch, need_stats=False)
    loss = 0.999 * supervised + 0.001 * stats["sigreg_col"]
    assert torch.isfinite(loss)
    assert predictions["number"].shape == (1, seq_len, 1)
    loss.backward()

    for parameter in (
        model.enc_dict["relation"].weight,
        model.blocks[0].nbr_attn.wq.weight,
        model.sigreg.col_proj.out.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0

    muon, adamw = build_muon_param_groups(model)
    grouped = {id(parameter) for parameter in muon + adamw}
    expected = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if not name.startswith("sigreg.")
    }
    assert grouped == expected
    assert len(grouped) == len(muon) + len(adamw)
