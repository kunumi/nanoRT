from .muon import Muon as Muon

__all__ = ["MUON_BLOCK_MODULES", "Muon", "build_muon_param_groups"]

MUON_BLOCK_MODULES = ("attns", "nbr_attn", "rel_attn", "ffn")


def build_muon_param_groups(model) -> tuple[list, list]:
    """Route block matrices to Muon and remaining weights to AdamW.
    SIGReg projectors are excluded and use a separate optimizer."""
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if name.startswith("sigreg."):
            continue
        parts = name.split(".")
        in_block = parts[0] == "blocks" and len(parts) >= 2
        in_muon_module = in_block and len(parts) >= 3 and parts[2] in MUON_BLOCK_MODULES
        is_weight_2d = param.ndim == 2

        if in_muon_module and is_weight_2d:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    return muon_params, adamw_params
