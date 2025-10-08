"""ssc_unet.py"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba
from nnunet_mednext import MedNeXtBlock, MedNeXtDownBlock, MedNeXtUpBlock

from norm_fix import ensure_mednext_norms


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, num_slices=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bimamba_type="v3",
            nslices=num_slices,
        )

    def forward(self, x):
        B, C = x.shape[:2]
        x_skip = x
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        out = out + x_skip
        return out


class MlpChannel(nn.Module):
    def __init__(self, hidden_size, mlp_dim):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class GSC(nn.Module):
    """Global Spatial Context module using MedNeXt blocks"""
    def __init__(self, in_channels):
        super().__init__()
        
        self.block1 = MedNeXtBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            norm_type='instance',
            kernel_size=3,
            exp_r=2,
            do_res=True
        )

        self.block2 = MedNeXtBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            norm_type='instance',
            kernel_size=3,
            exp_r=2,
            do_res=True
        )

        self.block3 = MedNeXtBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            norm_type='instance',
            kernel_size=1,
            exp_r=2,
            do_res=True
        )

        self.block4 = MedNeXtBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            norm_type='instance',
            kernel_size=1,
            exp_r=2,
            do_res=True
        )

    def forward(self, x):
        x_residual = x

        x1 = self.block1(x)
        x1 = self.block2(x1)

        x2 = self.block3(x)

        x = x1 + x2
        x = self.block4(x)

        return x + x_residual


class MambaEncoder(nn.Module):
    """Mamba encoder - extracts features at multiple scales"""
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        num_slices_list = [64, 32, 16, 8]

        for i in range(4):
            gsc = GSC(dims[i])
            stage = nn.Sequential(
                *[MambaLayer(dim=dims[i], num_slices=num_slices_list[i]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            self.gscs.append(gsc)

        self.out_indices = out_indices
        self.mlps = nn.ModuleList()

        for i_layer in range(4):
            layer = nn.InstanceNorm3d(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.gscs[i](x)
            x = self.stages[i](x)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x


class MedNeXtEncoderStage(nn.Module):
    """Single MedNeXt, Optional Downsampling"""
    def __init__(self, in_channels, out_channels, depth=2, exp_r=2,
                 norm_type='instance', downsample=False):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                MedNeXtBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    norm_type=norm_type,
                    kernel_size=3,
                    exp_r=exp_r,
                    do_res=True
                )
            )

        self.downsample = None
        if downsample:
            self.downsample = MedNeXtDownBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                exp_r=exp_r,
                kernel_size=3,
                norm_type=norm_type
            )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        skip = x

        if self.downsample is not None:
            x = self.downsample(x)
            return x, skip

        return x, skip


class MedNeXtDecoderStage(nn.Module):
    """Single MedNeXt Decoder Stage"""
    def __init__(self, in_channels, out_channels, skip_channels, depth=2,
                 exp_r=2, norm_type='instance'):
        super().__init__()

        self.up = MedNeXtUpBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            exp_r=exp_r,
            kernel_size=3,
            norm_type=norm_type
        )

        # project skip connection to match decoder channels
        self.skip_proj = nn.Conv3d(skip_channels, out_channels, kernel_size=1)

        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                MedNeXtBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    norm_type=norm_type,
                    kernel_size=3,
                    exp_r=exp_r,
                    do_res=True
                )
            )

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.skip_proj(skip)
        x = x + skip

        for block in self.blocks:
            x = block(x)

        return x

class OutBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class SSC_UNet(nn.Module):
    def __init__(
        self,
        in_chans=1,
        out_chans=13,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        norm_type='instance',
        exp_r=2,
        do_deep_supervision=True,
    ):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.feat_size = feat_size
        self.deep_supervision = do_deep_supervision
        self.norm_type = norm_type
        self.exp_r = exp_r

        self.mamba_encoder = MambaEncoder(
            in_chans=in_chans,
            depths=depths,
            dims=feat_size,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
        )

        # MedNeXt Encoder 1 – Processes Mamba features w/o downsampling
        # Stage 0
        self.mednext1_stage0_block1 = MedNeXtBlock(feat_size[0], feat_size[0], 
                                                   norm_type=norm_type, kernel_size=3, 
                                                   exp_r=exp_r, do_res=True)
        self.mednext1_stage0_block2 = MedNeXtBlock(feat_size[0], feat_size[0], 
                                                   norm_type=norm_type, kernel_size=3, 
                                                   exp_r=exp_r, do_res=True)

        # Stage 1
        self.mednext1_stage1_block1 = MedNeXtBlock(feat_size[1], feat_size[1], 
                                                   norm_type=norm_type, kernel_size=3, 
                                                   exp_r=exp_r, do_res=True)
        self.mednext1_stage1_block2 = MedNeXtBlock(feat_size[1], feat_size[1], 
                                                   norm_type=norm_type, kernel_size=3, 
                                                   exp_r=exp_r, do_res=True)

        # Stage 2
        self.mednext1_stage2_block1 = MedNeXtBlock(feat_size[2], feat_size[2], 
                                                   norm_type=norm_type, kernel_size=3, 
                                                   exp_r=exp_r, do_res=True)
        self.mednext1_stage2_block2 = MedNeXtBlock(feat_size[2], feat_size[2], 
                                                   norm_type=norm_type, kernel_size=3, 
                                                   exp_r=exp_r, do_res=True)

        # Stage 3
        self.mednext1_stage3_block1 = MedNeXtBlock(feat_size[3], feat_size[3], 
                                                   norm_type=norm_type, kernel_size=3, 
                                                   exp_r=exp_r, do_res=True)
        self.mednext1_stage3_block2 = MedNeXtBlock(feat_size[3], feat_size[3], 
                                                   norm_type=norm_type, kernel_size=3, 
                                                   exp_r=exp_r, do_res=True)

        # MedNeXt Encoder 2 – Processes input w/ downsampling
        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_chans, feat_size[0], kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm3d(feat_size[0])
        )

        # Stage 0 (with downsampling to stage 1)
        self.mednext2_stage0_block1 = MedNeXtBlock(feat_size[0], feat_size[0],
                                                    norm_type=norm_type, kernel_size=3,
                                                    exp_r=exp_r, do_res=True)
        self.mednext2_stage0_block2 = MedNeXtBlock(feat_size[0], feat_size[0],
                                                    norm_type=norm_type, kernel_size=3,
                                                    exp_r=exp_r, do_res=True)
        self.mednext2_down0 = MedNeXtDownBlock(feat_size[0], feat_size[1],
                                                exp_r=exp_r, kernel_size=3,
                                                norm_type=norm_type)

        # Stage 1 (with downsampling to stage 2)
        self.mednext2_stage1_block1 = MedNeXtBlock(feat_size[1], feat_size[1],
                                                    norm_type=norm_type, kernel_size=3,
                                                    exp_r=exp_r, do_res=True)
        self.mednext2_stage1_block2 = MedNeXtBlock(feat_size[1], feat_size[1],
                                                    norm_type=norm_type, kernel_size=3,
                                                    exp_r=exp_r, do_res=True)
        self.mednext2_down1 = MedNeXtDownBlock(feat_size[1], feat_size[2],
                                                exp_r=exp_r, kernel_size=3,
                                                norm_type=norm_type)

        # Stage 2 (with downsampling to stage 3)
        self.mednext2_stage2_block1 = MedNeXtBlock(feat_size[2], feat_size[2],
                                                    norm_type=norm_type, kernel_size=3,
                                                    exp_r=exp_r, do_res=True)
        self.mednext2_stage2_block2 = MedNeXtBlock(feat_size[2], feat_size[2],
                                                    norm_type=norm_type, kernel_size=3,
                                                    exp_r=exp_r, do_res=True)
        self.mednext2_down2 = MedNeXtDownBlock(feat_size[2], feat_size[3],
                                                exp_r=exp_r, kernel_size=3,
                                                norm_type=norm_type)

        # Stage 3 (no downsampling)
        self.mednext2_stage3_block1 = MedNeXtBlock(feat_size[3], feat_size[3],
                                                    norm_type=norm_type, kernel_size=3,
                                                    exp_r=exp_r, do_res=True)
        self.mednext2_stage3_block2 = MedNeXtBlock(feat_size[3], feat_size[3],
                                                    norm_type=norm_type, kernel_size=3,
                                                    exp_r=exp_r, do_res=True)

        # Fusion Projection Layers 
        self.fusion_proj0 = nn.Conv3d(feat_size[0] * 2, feat_size[0], kernel_size=1)
        self.fusion_proj1 = nn.Conv3d(feat_size[1] * 2, feat_size[1], kernel_size=1)
        self.fusion_proj2 = nn.Conv3d(feat_size[2] * 2, feat_size[2], kernel_size=1)
        self.fusion_proj3 = nn.Conv3d(feat_size[3] * 2, feat_size[3], kernel_size=1)

        # MedNeXt Decoder
        # Bottleneck
        self.bottleneck = MedNeXtBlock(feat_size[3], feat_size[3], norm_type=norm_type, kernel_size=3, exp_r=exp_r, do_res=False)

        # Decoder Stage 3 -> 2
        # 
        self.decoder3_up = nn.Sequential(
            nn.ConvTranspose3d(feat_size[3], feat_size[2], kernel_size=2, stride=2),
            nn.InstanceNorm3d(feat_size[2])
        )
        self.decoder3_skip_proj = nn.Conv3d(feat_size[2], feat_size[2], kernel_size=1)
        self.decoder3_block1 = MedNeXtBlock(feat_size[2], feat_size[2], norm_type=norm_type, kernel_size=3, exp_r=exp_r, do_res=True)
        self.decoder3_block2 = MedNeXtBlock(feat_size[2], feat_size[2], norm_type=norm_type, kernel_size=3, exp_r=exp_r, do_res=True)

        # Decoder Stage 2 -> 1
        self.decoder2_up = nn.Sequential(
            nn.ConvTranspose3d(feat_size[2], feat_size[1], kernel_size=2, stride=2),
            nn.InstanceNorm3d(feat_size[1])
        )
        self.decoder2_skip_proj = nn.Conv3d(feat_size[1], feat_size[1], kernel_size=1)
        self.decoder2_block1 = MedNeXtBlock(feat_size[1], feat_size[1], norm_type=norm_type, kernel_size=3, exp_r=exp_r, do_res=True)
        self.decoder2_block2 = MedNeXtBlock(feat_size[1], feat_size[1], norm_type=norm_type, kernel_size=3, exp_r=exp_r, do_res=True)

        # Decoder Stage 1 -> 0
        self.decoder1_up = nn.Sequential(
            nn.ConvTranspose3d(feat_size[1], feat_size[0], kernel_size=2, stride=2),
            nn.InstanceNorm3d(feat_size[0])
        )
        self.decoder1_skip_proj = nn.Conv3d(feat_size[0], feat_size[0], kernel_size=1)
        self.decoder1_block1 = MedNeXtBlock(feat_size[0], feat_size[0], norm_type=norm_type, kernel_size=3, exp_r=exp_r, do_res=True)
        self.decoder1_block2 = MedNeXtBlock(feat_size[0], feat_size[0], norm_type=norm_type, kernel_size=3, exp_r=exp_r, do_res=True)

        # Final refinement (no upsampling)
        self.final_block1 = MedNeXtBlock(feat_size[0], feat_size[0], norm_type=norm_type, kernel_size=3, exp_r=exp_r, do_res=True)
        self.final_block2 = MedNeXtBlock(feat_size[0], feat_size[0], norm_type=norm_type, kernel_size=3, exp_r=exp_r, do_res=True)

        # Output 
        self.out = OutBlock(feat_size[0], out_chans)

        # === Deep Supervision ===
        if self.deep_supervision:
            self.ds_out1 = OutBlock(feat_size[1], out_chans)
            self.ds_out2 = OutBlock(feat_size[2], out_chans)
            self.ds_out3 = OutBlock(feat_size[3], out_chans)

    def forward(self, x_in):
        ensure_mednext_norms(self, norm_type=self.norm_type)
   
        mamba_features = self.mamba_encoder(x_in) 
        mamba_feat0, mamba_feat1, mamba_feat2, mamba_feat3 = mamba_features

        # Stage 0
        mednext1_feat0 = self.mednext1_stage0_block1(mamba_feat0)
        mednext1_feat0 = self.mednext1_stage0_block2(mednext1_feat0)

        # Stage 1
        mednext1_feat1 = self.mednext1_stage1_block1(mamba_feat1)
        mednext1_feat1 = self.mednext1_stage1_block2(mednext1_feat1)

        # Stage 2
        mednext1_feat2 = self.mednext1_stage2_block1(mamba_feat2)
        mednext1_feat2 = self.mednext1_stage2_block2(mednext1_feat2)

        # Stage 3
        mednext1_feat3 = self.mednext1_stage3_block1(mamba_feat3)
        mednext1_feat3 = self.mednext1_stage3_block2(mednext1_feat3)

        x = self.stem(x_in)

        # Stage 0
        x = self.mednext2_stage0_block1(x)
        x = self.mednext2_stage0_block2(x)
        mednext2_feat0 = x
        x = self.mednext2_down0(x)

        # Stage 1
        x = self.mednext2_stage1_block1(x)
        x = self.mednext2_stage1_block2(x)
        mednext2_feat1 = x
        x = self.mednext2_down1(x)

        # Stage 2
        x = self.mednext2_stage2_block1(x)
        x = self.mednext2_stage2_block2(x)
        mednext2_feat2 = x
        x = self.mednext2_down2(x)

        # Stage 3
        x = self.mednext2_stage3_block1(x)
        x = self.mednext2_stage3_block2(x)
        mednext2_feat3 = x

        # Fusion
        fused_feat0 = self.fusion_proj0(torch.cat([mednext1_feat0, mednext2_feat0], dim=1))
        fused_feat1 = self.fusion_proj1(torch.cat([mednext1_feat1, mednext2_feat1], dim=1))
        fused_feat2 = self.fusion_proj2(torch.cat([mednext1_feat2, mednext2_feat2], dim=1))
        fused_feat3 = self.fusion_proj3(torch.cat([mednext1_feat3, mednext2_feat3], dim=1))

        # Bottleneck
        x = self.bottleneck(fused_feat3)

        # Decoder 3 -> 2
        x = self.decoder3_up(x)

        skip = self.decoder3_skip_proj(fused_feat2)
        x = x + skip
        x = self.decoder3_block1(x)
        x = self.decoder3_block2(x)
        dec_feat2 = x

        # Decoder 2 -> 1
        x = self.decoder2_up(x)
        skip = self.decoder2_skip_proj(fused_feat1)
        x = x + skip
        x = self.decoder2_block1(x)
        x = self.decoder2_block2(x)
        dec_feat1 = x

        # Decoder 1 -> 0
        x = self.decoder1_up(x)
        skip = self.decoder1_skip_proj(fused_feat0)
        x = x + skip
        x = self.decoder1_block1(x)
        x = self.decoder1_block2(x)
        dec_feat0 = x

        # Final refinement
        x = self.final_block1(x)
        x = self.final_block2(x)

        # Output
        main_output = self.out(x)

        if self.deep_supervision:
            ds1 = self.ds_out1(dec_feat1)
            ds2 = self.ds_out2(dec_feat2)
            ds3 = self.ds_out3(fused_feat3)  
            return (main_output, ds1, ds2, ds3)

        return main_output
