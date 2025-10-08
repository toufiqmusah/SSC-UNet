"""norm_fix.py"""

import torch
import torch.nn as nn

"""
This utility ensures that MedNeXtBlock instances have a .norm attribute
that is properly registered and on the correct device.
This is necessary because MedNeXtBlock norms are not named "norm" by default.
Also handles GroupNorm compatibility issues with concatenated features.
"""

def _restore_or_create_norm_for_mednext_block(mblk: nn.Module, norm_type='group'):

    target_device = next(mblk.parameters()).device if list(mblk.parameters()) else torch.device("cpu")

    if hasattr(mblk, "norm"):
        if "norm" not in mblk._modules:
            mblk._modules["norm"] = getattr(mblk, "norm")
        if hasattr(mblk.norm, 'weight') and mblk.norm.weight.device != target_device:
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
        if hasattr(candidate_mod, 'weight') and candidate_mod.weight.device != target_device:
            mblk.norm.to(target_device)
        return

    inferred_channels = None

    if type(mblk).__name__ == "MedNeXtUpBlock":
         if hasattr(mblk, 'out_channels'):
             inferred_channels = mblk.out_channels
    else:
        for conv_attr in ("conv1", "conv3", "conv2", "conv"):
            if hasattr(mblk, conv_attr):
                conv = getattr(mblk, conv_attr)
                if isinstance(conv, nn.Conv3d):
                    inferred_channels = conv.out_channels
                    break

    if inferred_channels is None:
        inferred_channels = 1

    # appropriate norm layer based on norm_type
    if norm_type == 'instance':
        new_norm = nn.InstanceNorm3d(
            num_features=inferred_channels,
            eps=1e-5,
            affine=True
        ).to(target_device)
    else:  # norm_type == 'group'
        if inferred_channels == 1:
            num_groups = 1
        else:
            for preferred_groups in [32, 16, 8, 4, 2]:
                if inferred_channels % preferred_groups == 0:
                    num_groups = preferred_groups
                    break
            else:
                # using smallest divisor > 1
                num_groups = 2 if inferred_channels % 2 == 0 else 1

        new_norm = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=inferred_channels,
            eps=1e-5,
            affine=True
        ).to(target_device)

    mblk._modules["norm"] = new_norm
    setattr(mblk, "norm", new_norm)


def ensure_mednext_norms(root_module: nn.Module, norm_type='instance'):
    """
    Walk the model and ensure MedNeXtBlock instances have .norm usable by their forward().
    Recursively checks submodules.
    """
    for name, module in root_module.named_modules():
        module_type = type(module).__name__
        if module_type in ["MedNeXtBlock", "MedNeXtDownBlock", "MedNeXtUpBlock"]:
            if module_type == "MedNeXtUpBlock":
                for sub_name, sub_module in module.named_modules():
                    if isinstance(sub_module, nn.InstanceNorm3d):
                        if sub_module.num_features != module.out_channels:
                            # instancenorm3d with correct features
                            new_norm = nn.InstanceNorm3d(
                                num_features=module.out_channels,
                                eps=sub_module.eps,
                                affine=sub_module.affine
                            ).to(sub_module.weight.device if hasattr(sub_module, 'weight') else torch.device("cpu"))

                            # swapping old norm module with the new one
                            parent_module_str = ".".join(name.split(".") + sub_name.split(".")[:-1])
                            child_module_name = sub_name.split(".")[-1]

                            parent_module = root_module
                            for part in parent_module_str.split('.'):
                                if part:
                                    parent_module = getattr(parent_module, part)

                            setattr(parent_module, child_module_name, new_norm)

            _restore_or_create_norm_for_mednext_block(module, norm_type=norm_type)


def fix_groupnorm_for_concatenated_features(root_module: nn.Module):
    """
    Replace GroupNorm layers with InstanceNorm3d when num_channels is not divisible by num_groups.
    """
    for name, module in root_module.named_modules():
        if isinstance(module, nn.GroupNorm):
            num_channels = module.num_channels
            num_groups = module.num_groups

            if num_channels % num_groups != 0:
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]

                parent = root_module
                for part in parent_name.split('.'):
                    if part:
                        parent = getattr(parent, part)

                if hasattr(parent, child_name):
                    device = module.weight.device if hasattr(module, 'weight') else torch.device("cpu")
                    new_norm = nn.InstanceNorm3d(
                        num_features=num_channels,
                        eps=module.eps,
                        affine=True
                    ).to(device)
                    setattr(parent, child_name, new_norm)