from __future__ import annotations

import torch 
import torch.nn as nn
import torch.nn.functional as F 

from mamba_ssm import Mamba
from nnunet_mednext import MedNeXtBlock, MedNeXtDownBlock, MedNeXtUpBlock

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock

from norm_fix import ensure_mednext_norms
from fusion import BidirectionalCrossAttentionFusion, Conv1x1Fusion

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

            return x


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, num_slices=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim,      # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
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
    def __init__(self,hidden_size, mlp_dim, ):
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
    def __init__(self, in_channles) -> None:
        super().__init__()

        self.x1_block = MedNeXtBlock(in_channels=in_channles, 
                                out_channels=in_channles, 
                                norm_type='instance', 
                                kernel_size=3, exp_r=2, res_block=True)

        self.x2_block = MedNeXtBlock(in_channels=in_channles, 
                                out_channels=in_channles, 
                                norm_type='instance', 
                                kernel_size=3, exp_r=2, res_block=True)

        self.x3_block = MedNeXtBlock(in_channels=in_channles, 
                                out_channels=in_channles, 
                                norm_type='instance', 
                                kernel_size=3, exp_r=2, res_block=True)
        
        self.x4_block = MedNeXtBlock(in_channels=in_channles, 
                                out_channels=in_channles, 
                                norm_type='instance', 
                                kernel_size=3, exp_r=2, res_block=True)


    def forward(self, x):

        x1 = self.x1_block(x)
        x2 = self.x2_block(x1)

        x = x1 + x2

        x3 = self.x3_block(x)
        x4 = self.x4_block(x3)

        x = x + x3 + x4

        return x

class MambaEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2, 2, 2], dims=[16, 32, 64, 128, 256, 512],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
              nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
              )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                # LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        num_slices_list = [64, 32, 16, 8]
        cur = 0

        for i in range(4):
            gsc = GSC(dims[i])

            stage = nn.Sequential(
                *[MambaLayer(dim=dims[i], num_slices=num_slices_list[i]) for j in range(depths[i])]
            )

            self.stages.append(stage)
            self.gscs.append(gsc)
            cur += depths[i]

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

class MedNeXtEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=3,
                 do_res=True, norm_type='group'):
        super(MedNeXtEncoderBlock, self).__init__()

        self.mednext_block_1 = MedNeXtBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            exp_r=exp_r,
            kernel_size=kernel_size,
            do_res=do_res,
            norm_type=norm_type,
            n_groups=None,
            grn=True
        )
        
        self.mednext_block_2 = MedNeXtBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            exp_r=exp_r,
            kernel_size=kernel_size,
            do_res=do_res, # and (in_channels == out_channels),
            norm_type=norm_type,
            n_groups=None,
            grn=True
        )

        self.down_block_1 = MedNeXtDownBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            exp_r=exp_r,
            kernel_size=kernel_size,
            do_res=do_res,
            norm_type=norm_type
        )

    def forward(self, x, downsample=False):
        x = self.mednext_block_1(x)
        x = self.mednext_block_2(x)
        skip_features = x.clone()

        if downsample:
            x = self.down_block_1(x)
            return x, skip_features

        return x, skip_features

class MedNeXtDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels,
                 exp_r=4, kernel_size=3, do_res=True, norm_type='group'):
        super(MedNeXtDecoderBlock, self).__init__()
        
        self.up_block = MedNeXtUpBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            exp_r=exp_r,
            kernel_size=kernel_size,
            do_res=do_res,
            norm_type=norm_type
        )
        
        self.skip_proj = nn.Conv3d(skip_channels, out_channels, kernel_size=1)

        self.mednext_block_1 = MedNeXtBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            exp_r=exp_r,
            kernel_size=kernel_size,
            do_res=do_res,
            norm_type=norm_type,
            n_groups=None,
            grn=True
        )
        
        self.mednext_block_2 = MedNeXtBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            exp_r=exp_r,
            kernel_size=kernel_size,
            do_res=do_res,
            norm_type=norm_type,
            n_groups=None,
            grn=True
        )
    
    def forward(self, x, skip_features):
        x = self.up_block(x)
        skip = self.skip_proj(skip_features)
        
        # if x.shape[2:] != skip.shape[2:]:
        #    skip = torch.nn.functional.interpolate(
        #        skip,
        #        size=x.shape[2:],
        #        mode='trilinear',
        #        align_corners=False
        #    )
        
        x = x + skip
        x = self.mednext_block_1(x)
        x = self.mednext_block_2(x)
        
        return x
# Better Decoder ..

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=3,
                 do_res=True, norm_type='group'):
        super(Bottleneck, self).__init__()

        self.mednext_block_1 = MedNeXtBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            exp_r=exp_r,
            kernel_size=kernel_size,
            do_res=do_res,
            norm_type=norm_type,
            n_groups=None,
            grn=True
        )
        
        self.mednext_block_2 = MedNeXtBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            exp_r=exp_r,
            kernel_size=kernel_size,
            do_res=do_res, # and (in_channels == out_channels),
            norm_type=norm_type,
            n_groups=None,
            grn=True
        )

    def forward(self, x):
        x = self.mednext_block_1(x) 
        x = self.mednext_block_2(x)
        return x

class OutBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutBlock, self).__init__()
        self.conv_out = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv_out(x)


class SSC_UNet(nn.Module):
    def __init__(self, in_chans=1, out_chans=1,
                 channels=[32, 64, 128, 256, 512, 512],
                 mamba_depths=[2, 2, 2, 2, 2, 2],
                 exp_r=4, norm_type='group',
                 fusion_type='bicrossattn', do_deep_supervision=False):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.channels = channels
        self.do_deep_supervision = do_deep_supervision

        self.initial_encoder = MedNeXtBlock(in_chans, channels[0], exp_r=exp_r, norm_type=norm_type)

        self.mamba_encoder = MambaEncoder(in_chans, mamba_depths, channels, out_indices=list(range(len(channels))))

        # Downsampling MedNeXt Path
        self.mednext_encoder_down_1 = MedNeXtEncoderBlock(in_chans, channels[0], exp_r, norm_type=norm_type)
        self.mednext_encoder_down_2 = MedNeXtEncoderBlock(channels[0], channels[1], exp_r, norm_type=norm_type)
        self.mednext_encoder_down_3 = MedNeXtEncoderBlock(channels[1], channels[2], exp_r, norm_type=norm_type)
        self.mednext_encoder_down_4 = MedNeXtEncoderBlock(channels[2], channels[3], exp_r, norm_type=norm_type)
        self.mednext_encoder_down_5 = MedNeXtEncoderBlock(channels[3], channels[4], exp_r, norm_type=norm_type)
        self.mednext_encoder_down_6 = MedNeXtEncoderBlock(channels[4], channels[5], exp_r, norm_type=norm_type)

        # No-Downsampling MedNeXt Path (for Mamba features)
        self.mednext_encoder_no_down_1 = MedNeXtBlock(channels[0], channels[0], exp_r, norm_type=norm_type, res_block=True)
        self.mednext_encoder_no_down_2 = MedNeXtBlock(channels[1], channels[1], exp_r, norm_type=norm_type, res_block=True)
        self.mednext_encoder_no_down_3 = MedNeXtBlock(channels[2], channels[2], exp_r, norm_type=norm_type, res_block=True)
        self.mednext_encoder_no_down_4 = MedNeXtBlock(channels[3], channels[3], exp_r, norm_type=norm_type, res_block=True)
        self.mednext_encoder_no_down_5 = MedNeXtBlock(channels[4], channels[4], exp_r, norm_type=norm_type, res_block=True)
        self.mednext_encoder_no_down_6 = MedNeXtBlock(channels[5], channels[5], exp_r, norm_type=norm_type, res_block=True)

        # --- Fusion Blocks ---
        if fusion_type == 'bicrossattn':
            fusion_block = BidirectionalCrossAttentionFusion
        elif fusion_type == 'conv1x1':
            fusion_block = Conv1x1Fusion

        self.fusion_1 = fusion_block(channels[0], num_heads=8, norm_type=norm_type)
        self.fusion_2 = fusion_block(channels[1], num_heads=8, norm_type=norm_type)
        self.fusion_3 = fusion_block(channels[2], num_heads=8, norm_type=norm_type)
        self.fusion_4 = fusion_block(channels[3], num_heads=8, norm_type=norm_type)
        self.fusion_5 = fusion_block(channels[4], num_heads=8, norm_type=norm_type)
        self.fusion_6 = fusion_block(channels[5], num_heads=8, norm_type=norm_type)
        
        # --- Bottleneck ---
        self.fusion_bottleneck = fusion_block(channels[-1], num_heads=8, norm_type=norm_type)
        self.bottleneck = Bottleneck(channels[-1], channels[-1], exp_r, norm_type=norm_type)

        # --- Decoders ---
        self.decoder_1 = MedNeXtDecoderBlock(channels[5], channels[4], skip_channels=channels[5], exp_r=exp_r, norm_type=norm_type)
        self.decoder_2 = MedNeXtDecoderBlock(channels[4], channels[3], skip_channels=channels[4], exp_r=exp_r, norm_type=norm_type)
        self.decoder_3 = MedNeXtDecoderBlock(channels[3], channels[2], skip_channels=channels[3], exp_r=exp_r, norm_type=norm_type)
        self.decoder_4 = MedNeXtDecoderBlock(channels[2], channels[1], skip_channels=channels[2], exp_r=exp_r, norm_type=norm_type)
        self.decoder_5 = MedNeXtDecoderBlock(channels[1], channels[0], skip_channels=channels[1], exp_r=exp_r, norm_type=norm_type)
        self.decoder_6 = MedNeXtDecoderBlock(channels[0], channels[0], skip_channels=channels[0], exp_r=exp_r, norm_type=norm_type)
        
        # Final decoder stage (no upsampling)
        self.decoder_final = MedNeXtBlock(channels[0], channels[0], exp_r=exp_r, norm_type=norm_type, res_block=True)
        
        # --- Output ---
        self.out = OutBlock(in_channels=channels[0], out_channels=out_chans)
        
        # --- Deep Supervision Outputs ---
        if self.do_deep_supervision:
            self.out_4 = OutBlock(channels[4], out_chans)
            self.out_3 = OutBlock(channels[3], out_chans)
            self.out_2 = OutBlock(channels[2], out_chans)
            self.out_1 = OutBlock(channels[1], out_chans)
            self.out_0 = OutBlock(channels[0], out_chans)

    def forward(self, x):
        ensure_mednext_norms(self)

        # 1. Initial MedNeXt Block
        skip_init = self.initial_encoder(x)

        # 2. Mamba Encoder Path
        mamba_features = self.mamba_encoder(x)
        mamba_1, mamba_2, mamba_3, mamba_4, mamba_5, mamba_6 = mamba_features
         
        # 3a. MedNeXt Downsampling Path
        x_down_0, skip_down_1 = self.mednext_encoder_down_1(skip_init, downsample=True) # replacing x with skip_init
        x_down_1, skip_down_2 = self.mednext_encoder_down_2(x_down_0, downsample=True)
        x_down_2, skip_down_3 = self.mednext_encoder_down_3(x_down_1, downsample=True)
        x_down_3, skip_down_4 = self.mednext_encoder_down_4(x_down_2, downsample=True)
        x_down_4, skip_down_5 = self.mednext_encoder_down_5(x_down_3, downsample=True)
        bottleneck_down, skip_down_6 = self.mednext_encoder_down_6(x_down_4, downsample=True)

        # print("Shapes after downsampling path:")
        print(f"""MedNeXt Downsampling Path Shapes:
                skip_down_1, x_down_0: {skip_down_1.shape, x_down_0.shape}
                skip_down_2, x_down_1: {skip_down_2.shape, x_down_1.shape}
                skip_down_3, x_down_2: {skip_down_3.shape, x_down_2.shape}
                skip_down_4, x_down_3: {skip_down_4.shape, x_down_3.shape}
                skip_down_5, x_down_4: {skip_down_5.shape, x_down_4.shape}
                skip_down_6, bottleneck_down: {skip_down_6.shape, bottleneck_down.shape}""")

        # 3b. MedNeXt No-Downsampling Path (processing Mamba features)
        skip_no_down_1 = self.mednext_encoder_no_down_1(mamba_1)
        skip_no_down_2 = self.mednext_encoder_no_down_2(mamba_2)
        skip_no_down_3 = self.mednext_encoder_no_down_3(mamba_3)
        skip_no_down_4 = self.mednext_encoder_no_down_4(mamba_4)
        skip_no_down_5 = self.mednext_encoder_no_down_5(mamba_5)
        bottleneck_no_down = self.mednext_encoder_no_down_6(mamba_6)

        print("""Shapes after no-downsampling path (Mamba features):
                skip_no_down_1: {skip_no_down_1.shape}
                skip_no_down_2: {skip_no_down_2.shape}
                skip_no_down_3: {skip_no_down_3.shape}
                skip_no_down_4: {skip_no_down_4.shape}
                skip_no_down_5: {skip_no_down_5.shape}
                bottleneck_no_down: {bottleneck_no_down.shape}""")

        # 4. Fusion at each stage
        fused_1 = self.fusion_1(skip_no_down_1, skip_down_1)
        print(f"Shape of fused_1: {fused_1.shape}")
        fused_2 = self.fusion_2(skip_no_down_2, skip_down_2)
        print(f"Shape of fused_2: {fused_2.shape}")
        fused_3 = self.fusion_3(skip_no_down_3, skip_down_3)
        print(f"Shape of fused_3: {fused_3.shape}")
        fused_4 = self.fusion_4(skip_no_down_4, skip_down_4)
        print(f"Shape of fused_4: {fused_4.shape}")
        fused_5 = self.fusion_5(skip_no_down_5, skip_down_5)
        print(f"Shape of fused_5: {fused_5.shape}")
        fused_6 = self.fusion_6(bottleneck_no_down, skip_down_6)
        print(f"Shape of fused_6: {fused_6.shape}")

        # 5. Bottleneck Fusion and Processing
        # fused_bottleneck = self.fusion_bottleneck(bottleneck_no_down, bottleneck_down)
        dec_features = self.bottleneck(fused_6)

        # 6. Decoder Path
        dec_features_d1 = self.decoder_1(dec_features, fused_6)
        dec_features_d2 = self.decoder_2(dec_features_d1, fused_5)
        dec_features_d3 = self.decoder_3(dec_features_d2, fused_4)
        dec_features_d4 = self.decoder_4(dec_features_d3, fused_3)
        dec_features_d5 = self.decoder_5(dec_features_d4, fused_2)
        dec_features_d6 = self.decoder_6(dec_features_d5, fused_1)
        
        # 7th decoder stage (no upsampling, just combination)
        final_features = self.decoder_final(dec_features_d6 + skip_init)
        
        # 7. Final Output
        main_output = self.out(final_features)
        
        if self.do_deep_supervision:
            out_4 = self.out_4(dec_features_d1)
            out_3 = self.out_3(dec_features_d2)
            out_2 = self.out_2(dec_features_d3)
            out_1 = self.out_1(dec_features_d4)
            out_0 = self.out_0(dec_features_d5)
            return (main_output, out_0, out_1, out_2, out_3, out_4)

        return main_output
    

# for SSC-UNet

# create 6 encoder stages of mednext with downsampling in between
# create 6 encoder stages of mednext without downsampling in between
# initiate mamba layer too to extract features from input
# create  7 decoder stages of mednext
# instantiate fusion methods too. lets start with bidirectional cross attention as default

# in the forward method:
# x = input_volume
# pass x into an initial encoder block of mednext and pass those features to the first decoder block
# pass x into the mednext with downsampling and make it flow to the last layer. the saved features should be held onto for now
# pass x into mamba and get the features from all the stages
# pass the features into the mednext encoder without downsampling

# the features from both encoders should be fused at each stage using the fusion method
# finally the bottleneck features should also be fused with the last encoder features
# finally the decoder should take in the fused features and perform segmentation

# this is SSC-UNet with a dual encoder (sort of), fusion, and a single decoder architecture. refer back to segmamba.py for general structure if unsure. but yes, this is the idea.