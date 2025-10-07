import torch
import torch.nn as nn

""" this utility ensures that MedNeXtBlock instances have a .norm attribute
    that is properly registered and on the correct device.
    this is necessary because MedNeXtBlock norms are not named "norm" by default
"""

def _restore_or_create_norm_for_mednext_block(mblk: nn.Module):
    
    target_device = next(mblk.parameters()).device if list(mblk.parameters()) else torch.device("cpu")

    if hasattr(mblk, "norm"):
        if "norm" not in mblk._modules:
            mblk._modules["norm"] = getattr(mblk, "norm")
        if mblk.norm.weight.device != target_device:
             mblk.norm.to(target_device)
        return

    candidate_name = None
    candidate_mod = None
    for n, sm in list(mblk._modules.items()):
        if isinstance(sm, (nn.InstanceNorm3d, nn.GroupNorm, nn.LayerNorm, nn.BatchNorm3d)):
            candidate_name = n
            candidate_mod = sm
            break

    if candidate_mod is not None:
        if candidate_name != "norm":
            mblk._modules["norm"] = mblk._modules.pop(candidate_name)
        setattr(mblk, "norm", candidate_mod)
        if mblk.norm.weight.device != target_device:
             mblk.norm.to(target_device)
        return

    inferred_channels = None
    for conv_attr in ("conv1", "conv3", "conv2"):
        if hasattr(mblk, conv_attr):
            conv = getattr(mblk, conv_attr)
            if isinstance(conv, nn.Conv3d):
                inferred_channels = conv.out_channels
                break

    if inferred_channels is None:
        inferred_channels = 1

    num_groups = 1 if inferred_channels == 1 else min(32, inferred_channels)
    new_norm = nn.GroupNorm(num_groups=num_groups, num_channels=inferred_channels, eps=1e-5, affine=True).to(target_device)
    mblk._modules["norm"] = new_norm
    setattr(mblk, "norm", new_norm)


def ensure_mednext_norms(root_module: nn.Module):
    """Walk the model and ensure MedNeXtBlock instances have .norm usable by their forward()."""
    for name, module in root_module.named_modules():
        if type(module).__name__ == "MedNeXtBlock" or type(module).__name__ == "MedNeXtDownBlock" or type(module).__name__ == "MedNeXtUpBlock":
            _restore_or_create_norm_for_mednext_block(module)
