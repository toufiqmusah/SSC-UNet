# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");

import torch
from torch import nn
import torch.nn.functional as F

from monai.networks.nets import SwinUNETR
from nnunet_mednext import MedNeXtBlock, MedNeXtDownBlock, MedNeXtUpBlock

class SwinVITBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SwinVITBlock, self).__init__()
        swin = SwinUNETR(in_channels=in_channels, out_channels=out_channels)
        self.svit = swin.swinViT

    def forward(self, x):
        return self.svit(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=3,
                 do_res=True, norm_type='group'):
        super(EncoderBlock, self).__init__()

        self.mednext_block = MedNeXtBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            exp_r=exp_r,
            kernel_size=kernel_size,
            do_res=do_res and (in_channels == out_channels),
            norm_type=norm_type,
            n_groups=None
        )
        self.down_block = MedNeXtDownBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            exp_r=exp_r,
            kernel_size=kernel_size,
            do_res=do_res,
            norm_type=norm_type
        )

    def forward(self, x, swin_features=None):
        x = self.mednext_block(x)

        if swin_features is not None:
            if x.shape[2:] != swin_features.shape[2:]:
                swin_features = torch.nn.functional.interpolate(
                    swin_features,
                    size=x.shape[2:],
                    mode='trilinear',
                    align_corners=False
                )
            if x.shape[1] != swin_features.shape[1]:
                if not hasattr(self, 'channel_proj'):
                    self.channel_proj = nn.Conv3d(swin_features.shape[1], x.shape[1], 1).to(x.device)
                swin_features = self.channel_proj(swin_features)
            x = x + swin_features

        skip_features = x.clone()
        x = self.down_block(x)
        return x, skip_features

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=None,
                 exp_r=4, kernel_size=3, norm_type='group'):
        super(DecoderBlock, self).__init__()
        self.skip_channels = skip_channels

        # Main upsampling block
        self.up_block = MedNeXtUpBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            exp_r=exp_r,
            kernel_size=kernel_size,
            do_res=False,
            norm_type=norm_type
        )

        # Skip feature processing
        if skip_channels is not None:
            # Adjusted group normalization for smaller channels
            groups = min(8, out_channels) if out_channels >= 8 else out_channels
            
            self.skip_conv = nn.Sequential(
                nn.Conv3d(skip_channels, out_channels, kernel_size=1),
                nn.GroupNorm(groups, out_channels),
                nn.ReLU(inplace=True)
            )

            # Changed to regular Conv block instead of MedNeXtBlock
            combine_groups = min(8, out_channels) if out_channels >= 8 else out_channels
            self.combine_conv = nn.Sequential(
                nn.Conv3d(out_channels * 2, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(combine_groups, out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.skip_conv = None
            self.combine_conv = None

    def forward(self, x, skip_features=None, use_skip=True):
        # Upsample first
        x = self.up_block(x)

        if use_skip and skip_features is not None and self.skip_channels is not None:

            # Process skip features
            skip = self.skip_conv(skip_features)

            if x.shape[2:] != skip.shape[2:]:
                skip = torch.nn.functional.interpolate(
                    skip,
                    size=x.shape[2:],
                    mode='trilinear',
                    align_corners=False
                )

            x = torch.cat([x, skip], dim=1)
            x = self.combine_conv(x)

        return x

class SwinMedNeXt(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, use_skip_connections=True, 
                 do_deep_supervision=True, enable_swin=True, feat_size=[24, 48, 96, 192]):
        super(SwinMedNeXt, self).__init__()
        self.use_skip_connections = use_skip_connections
        self.do_deep_supervision = do_deep_supervision
        self.enable_swin = enable_swin
        
        # Make SwinViT optional to save memory
        if self.enable_swin:
            self.swin_vit = SwinVITBlock(in_channels, out_channels)
        else:
            self.swin_vit = None
            
        # Configurable feature sizes
        self.input_conv = nn.Conv3d(in_channels, feat_size[0], kernel_size=3, padding=1)

        # Encoder with configurable feature sizes
        self.encoder1 = EncoderBlock(feat_size[0], feat_size[0])
        self.encoder2 = EncoderBlock(feat_size[0], feat_size[1])
        self.encoder3 = EncoderBlock(feat_size[1], feat_size[2])
        self.encoder4 = EncoderBlock(feat_size[2], feat_size[3])

        # Bottleneck - reduced complexity
        self.bottleneck = nn.Sequential(
            MedNeXtBlock(feat_size[3], feat_size[3], exp_r=2, kernel_size=3, do_res=True),
        )
        self.bottleneck_proj = None

        # Decoder with configurable feature sizes
        skip_channels = feat_size if use_skip_connections else [None]*4
        self.decoder4 = DecoderBlock(feat_size[3], feat_size[2], skip_channels[3])
        self.decoder3 = DecoderBlock(feat_size[2], feat_size[1], skip_channels[2])
        self.decoder2 = DecoderBlock(feat_size[1], feat_size[0], skip_channels[1])
        self.decoder1 = DecoderBlock(feat_size[0], feat_size[0], skip_channels[0])

        self.output_conv = nn.Conv3d(feat_size[0], out_channels, kernel_size=1)

        if self.do_deep_supervision:
            self.ds_out4 = nn.Conv3d(feat_size[2], out_channels, kernel_size=1) # From decoder4
            self.ds_out3 = nn.Conv3d(feat_size[1], out_channels, kernel_size=1)  # From decoder3
            self.ds_out2 = nn.Conv3d(feat_size[0], out_channels, kernel_size=1)  # From decoder2

    def set_skip_connections(self, use_skip):
        self.use_skip_connections = use_skip

    def forward(self, x):
        # Optional SwinViT features
        swin_features = []
        if self.enable_swin and self.swin_vit is not None:
            swin_features = self.swin_vit(x)
        
        x = self.input_conv(x)

        # Encoder path
        x, skip1 = self.encoder1(x, swin_features[0] if len(swin_features) > 0 else None)
        x, skip2 = self.encoder2(x, swin_features[1] if len(swin_features) > 1 else None)
        x, skip3 = self.encoder3(x, swin_features[2] if len(swin_features) > 2 else None)
        x, skip4 = self.encoder4(x, swin_features[3] if len(swin_features) > 3 else None)
        skip_connections = [skip4, skip3, skip2, skip1]

        # Bottleneck
        if self.enable_swin and len(swin_features) > 4:
            bottleneck_swin = swin_features[4]
            if x.shape[2:] != bottleneck_swin.shape[2:]:
                bottleneck_swin = F.interpolate(bottleneck_swin, size=x.shape[2:], mode='trilinear', align_corners=False)
            if x.shape[1] != bottleneck_swin.shape[1]:
                if self.bottleneck_proj is None:
                    self.bottleneck_proj = nn.Conv3d(bottleneck_swin.shape[1], x.shape[1], 1).to(x.device)
                bottleneck_swin = self.bottleneck_proj(bottleneck_swin)
            x = x + bottleneck_swin
        x = self.bottleneck(x)

        # Decoder path
        dec4_out = self.decoder4(x, skip_connections[0], self.use_skip_connections)
        dec3_out = self.decoder3(dec4_out, skip_connections[1], self.use_skip_connections)
        dec2_out = self.decoder2(dec3_out, skip_connections[2], self.use_skip_connections)
        dec1_out = self.decoder1(dec2_out, skip_connections[3], self.use_skip_connections)

        main_output = self.output_conv(dec1_out)

        if self.do_deep_supervision:
            ds_out_2 = self.ds_out2(dec2_out)  # 1/2 resolution
            ds_out_3 = self.ds_out3(dec3_out)  # 1/4 resolution
            ds_out_4 = self.ds_out4(dec4_out)  # 1/8 resolution
            return (main_output, ds_out_2, ds_out_3, ds_out_4)
        else:
            return main_output

"""
import torch
from torch import nn
import torch.nn.functional as F

from monai.networks.nets import SwinUNETR
from nnunet_mednext import MedNeXtBlock, MedNeXtDownBlock, MedNeXtUpBlock

class SwinVITBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SwinVITBlock, self).__init__()
        swin = SwinUNETR(in_channels=in_channels, out_channels=out_channels)
        self.svit = swin.swinViT

    def forward(self, x):
        return self.svit(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=3,
                 do_res=True, norm_type='group'):
        super(EncoderBlock, self).__init__()

        self.mednext_block = MedNeXtBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            exp_r=exp_r,
            kernel_size=kernel_size,
            do_res=do_res and (in_channels == out_channels),
            norm_type=norm_type,
            n_groups=None
        )
        self.down_block = MedNeXtDownBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            exp_r=exp_r,
            kernel_size=kernel_size,
            do_res=do_res,
            norm_type=norm_type
        )

    def forward(self, x, swin_features=None):
        x = self.mednext_block(x)

        if swin_features is not None:
            if x.shape[2:] != swin_features.shape[2:]:
                swin_features = torch.nn.functional.interpolate(
                    swin_features,
                    size=x.shape[2:],
                    mode='trilinear',
                    align_corners=False
                )
            if x.shape[1] != swin_features.shape[1]:
                if not hasattr(self, 'channel_proj'):
                    self.channel_proj = nn.Conv3d(swin_features.shape[1], x.shape[1], 1).to(x.device)
                swin_features = self.channel_proj(swin_features)
            x = x + swin_features

        skip_features = x.clone()
        x = self.down_block(x)
        return x, skip_features

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=None,
                 exp_r=4, kernel_size=3, norm_type='group'):
        super(DecoderBlock, self).__init__()
        self.skip_channels = skip_channels

        # Main upsampling block
        self.up_block = MedNeXtUpBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            exp_r=exp_r,
            kernel_size=kernel_size,
            do_res=False,
            norm_type=norm_type
        )

        # Skip feature processing
        if skip_channels is not None:
            self.skip_conv = nn.Sequential(
                nn.Conv3d(skip_channels, out_channels, kernel_size=1),
                nn.GroupNorm(8, out_channels),
                nn.ReLU(inplace=True)
            )

            # Changed to regular Conv block instead of MedNeXtBlock
            self.combine_conv = nn.Sequential(
                nn.Conv3d(out_channels * 2, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(8, out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.skip_conv = None
            self.combine_conv = None

    def forward(self, x, skip_features=None, use_skip=True):
        # Upsample first
        x = self.up_block(x)

        if use_skip and skip_features is not None and self.skip_channels is not None:

            # Process skip features
            skip = self.skip_conv(skip_features)

            if x.shape[2:] != skip.shape[2:]:
                skip = torch.nn.functional.interpolate(
                    skip,
                    size=x.shape[2:],
                    mode='trilinear',
                    align_corners=False
                )

            x = torch.cat([x, skip], dim=1)
            x = self.combine_conv(x)

        return x

class SwinConvAE_DS(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, use_skip_connections=True, do_deep_supervision=True):
        super(SwinConvAE_DS, self).__init__()
        self.use_skip_connections = use_skip_connections
        self.do_deep_supervision = do_deep_supervision

        self.swin_vit = SwinVITBlock(in_channels, out_channels)
        self.input_conv = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)

        # Encoder
        self.encoder1 = EncoderBlock(64, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            MedNeXtBlock(512, 512, exp_r=4, kernel_size=3, do_res=True),
            MedNeXtBlock(512, 512, exp_r=4, kernel_size=3, do_res=True)
        )
        self.bottleneck_proj = None

        # Decoder
        skip_channels = [64, 128, 256, 512] if use_skip_connections else [None]*4
        self.decoder4 = DecoderBlock(512, 256, skip_channels[3])
        self.decoder3 = DecoderBlock(256, 128, skip_channels[2])
        self.decoder2 = DecoderBlock(128, 64, skip_channels[1])
        self.decoder1 = DecoderBlock(64, 64, skip_channels[0])

        self.output_conv = nn.Conv3d(64, out_channels, kernel_size=1)

        if self.do_deep_supervision:
            self.ds_out4 = nn.Conv3d(256, out_channels, kernel_size=1) # From decoder4
            self.ds_out3 = nn.Conv3d(128, out_channels, kernel_size=1) # From decoder3
            self.ds_out2 = nn.Conv3d(64, out_channels, kernel_size=1)  # From decoder2

    def set_skip_connections(self, use_skip):
        self.use_skip_connections = use_skip

    def forward(self, x):
        swin_features = self.swin_vit(x)
        x = self.input_conv(x)

        # Encoder path
        x, skip1 = self.encoder1(x, swin_features[0] if len(swin_features) > 0 else None)
        x, skip2 = self.encoder2(x, swin_features[1] if len(swin_features) > 1 else None)
        x, skip3 = self.encoder3(x, swin_features[2] if len(swin_features) > 2 else None)
        x, skip4 = self.encoder4(x, swin_features[3] if len(swin_features) > 3 else None)
        skip_connections = [skip4, skip3, skip2, skip1]

        # Bottleneck
        if len(swin_features) > 4:
            bottleneck_swin = swin_features[4]
            if x.shape[2:] != bottleneck_swin.shape[2:]:
                bottleneck_swin = F.interpolate(bottleneck_swin, size=x.shape[2:], mode='trilinear', align_corners=False)
            if x.shape[1] != bottleneck_swin.shape[1]:
                if self.bottleneck_proj is None:
                    self.bottleneck_proj = nn.Conv3d(bottleneck_swin.shape[1], x.shape[1], 1).to(x.device)
                bottleneck_swin = self.bottleneck_proj(bottleneck_swin)
            x = x + bottleneck_swin
        x = self.bottleneck(x)

        # Decoder path
        dec4_out = self.decoder4(x, skip_connections[0], self.use_skip_connections)
        dec3_out = self.decoder3(dec4_out, skip_connections[1], self.use_skip_connections)
        dec2_out = self.decoder2(dec3_out, skip_connections[2], self.use_skip_connections)
        dec1_out = self.decoder1(dec2_out, skip_connections[3], self.use_skip_connections)

        main_output = self.output_conv(dec1_out)

        if self.do_deep_supervision:
            ds_out_2 = self.ds_out2(dec2_out)  # 1/2 resolution
            ds_out_3 = self.ds_out3(dec3_out)  # 1/4 resolution
            ds_out_4 = self.ds_out4(dec4_out)  # 1/8 resolution
            return (main_output, ds_out_2, ds_out_3, ds_out_4)
        else:
            return main_output
        
"""