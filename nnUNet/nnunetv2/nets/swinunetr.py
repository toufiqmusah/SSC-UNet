from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import \
    nnUNetTrainerNoDeepSupervision
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import nn

from monai.networks.nets import SwinUNETR

class nnUNetTrainerSwinUNETR(nnUNetTrainerNoDeepSupervision):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        original_patch_size = self.configuration_manager.patch_size
        new_patch_size = [-1] * len(original_patch_size)
        for i in range(len(original_patch_size)):
            if (original_patch_size[i] / 2**5) < 1 or ((original_patch_size[i] / 2**5) % 1) != 0:
                new_patch_size[i] = round(original_patch_size[i] / 2**5 + 0.5) * 2**5
            else:
                new_patch_size[i] = original_patch_size[i]
        self.configuration_manager.configuration['patch_size'] = new_patch_size
        self.print_to_log_file("Patch size changed from {} to {}".format(original_patch_size, new_patch_size))
        self.plans_manager.plans['configurations'][self.configuration_name]['patch_size'] = new_patch_size

        self.grad_scaler = None
        self.initial_lr = 8e-4
        self.weight_decay = 0.01

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = False) -> nn.Module:

        label_manager = plans_manager.get_label_manager(dataset_json)

        model = SwinUNETR(
            in_channels = num_input_channels,
            out_channels = label_manager.num_segmentation_heads,
            img_size = configuration_manager.patch_size,
            depths = (2, 2, 2, 2),
            num_heads = (3, 6, 12, 24),
            feature_size = 48, ##
            norm_name = "instance",
            drop_rate = 0.0,
            attn_drop_rate = 0.0,
            dropout_path_rate = 0.0,
            normalize = True,
            use_checkpoint = False,
            spatial_dims = len(configuration_manager.patch_size),
            downsample = "merging",
            use_v2 = False,
        )

        return model
    
    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        
        output = self.network(data)
        l = self.loss(output, target)
        l.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.optimizer.step()
        
        return {'loss': l.detach().cpu().numpy()}
    

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        output = self.network(data)
        del data
        l = self.loss(output, target)

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
    
    def configure_optimizers(self):

        optimizer = AdamW(self.network.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay, eps=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-6)

        self.print_to_log_file(f"Using optimizer {optimizer}")
        self.print_to_log_file(f"Using scheduler {scheduler}")

        return optimizer, scheduler
    
    def set_deep_supervision_enabled(self, enabled: bool):
        pass


class nnUNetTrainerSwinUNETR_10epochs(nnUNetTrainerSwinUNETR):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 10

'''# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from functools import partial

import torch 
import torch.nn as nn
import torch.nn.functional as F 

# from mamba_ssm import Mamba
from monai.networks.nets import SwinUNETR
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock

class SwinVITBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SwinVITBlock, self).__init__()
        swin = SwinUNETR(in_channels=in_channels, out_channels=out_channels)
        self.svit = swin.swinViT

    def forward(self, x):
        return self.svit(x)


class SwinUNETRv(nn.Module):
    def __init__(
        self,
        in_chans=1,
        out_chans=13,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 768,
        norm_name = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims=3,
        do_deep_supervision: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        self.deep_supervision = do_deep_supervision
        self.spatial_dims = spatial_dims
        
        self.vit = SwinVITBlock(in_chans, out_chans)
        
        # NEW: Add projection layers to match Swin-ViT output channels to encoder input channels
        # Swin-ViT outputs [24, 48, 96, 192] but we need [48, 96, 192, 384]
        self.proj0 = nn.Conv3d(24, feat_size[0], kernel_size=1) if spatial_dims == 3 else nn.Conv2d(24, feat_size[0], kernel_size=1)
        self.proj1 = nn.Conv3d(48, feat_size[1], kernel_size=1) if spatial_dims == 3 else nn.Conv2d(48, feat_size[1], kernel_size=1)
        self.proj2 = nn.Conv3d(96, feat_size[2], kernel_size=1) if spatial_dims == 3 else nn.Conv2d(96, feat_size[2], kernel_size=1)
        self.proj3 = nn.Conv3d(192, feat_size[3], kernel_size=1) if spatial_dims == 3 else nn.Conv2d(192, feat_size[3], kernel_size=1)
        
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.out_main_seg = UnetOutBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[0], out_channels=self.out_chans)

        if self.deep_supervision:
            self.ds_seg_from_dec0 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[0], out_channels=self.out_chans)
            self.ds_seg_from_dec1 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[1], out_channels=self.out_chans)
            self.ds_seg_from_dec2 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[2], out_channels=self.out_chans)
            self.ds_seg_from_dec3 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[3], out_channels=self.out_chans)

    def forward(self, x_in):
        mamba_features = self.vit(x_in)

        # Project Swin-ViT features to match encoder channel expectations
        mamba_features = [
            self.proj0(mamba_features[0]),
            self.proj1(mamba_features[1]),
            self.proj2(mamba_features[2]),
            self.proj3(mamba_features[3])
        ]

        skip_conn1 = self.encoder1(x_in)
        skip_conn2 = self.encoder2(mamba_features[0])
        skip_conn3 = self.encoder3(mamba_features[1])
        skip_conn4 = self.encoder4(mamba_features[2])

        bottleneck_features = self.encoder5(mamba_features[3])

        dec_features_d3 = self.decoder5(bottleneck_features, skip_conn4)
        dec_features_d2 = self.decoder4(dec_features_d3, skip_conn3)
        dec_features_d1 = self.decoder3(dec_features_d2, skip_conn2)
        dec_features_d0 = self.decoder2(dec_features_d1, skip_conn1)

        final_features = self.decoder1(dec_features_d0)
        main_segmentation = self.out_main_seg(final_features)

        if self.deep_supervision:
            ds_seg2 = self.ds_seg_from_dec1(dec_features_d1)
            ds_seg3 = self.ds_seg_from_dec2(dec_features_d2)
            ds_seg4 = self.ds_seg_from_dec3(dec_features_d3)
            return (main_segmentation, ds_seg2, ds_seg3, ds_seg4)
        else:
            return main_segmentation

            '''