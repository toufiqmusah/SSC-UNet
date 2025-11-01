# Copyright (c) MONAI Consortium
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