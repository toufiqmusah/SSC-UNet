from __future__ import annotations

from nnUNet.nnunetv2.nets.swin_mednext import EncoderBlock
import torch 
import torch.nn as nn
import torch.nn.functional as F 

from mamba_ssm import Mamba
from nnunet_mednext import MedNeXtBlock, MedNeXtDownBlock, MedNeXtUpBlock

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock

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
                                stride=1, norm_name='instance', 
                                kernel_size=3, expansion=2, res_block=True)

        self.x2_block = MedNeXtBlock(in_channels=in_channles, 
                                out_channels=in_channles, 
                                stride=1, norm_name='instance', 
                                kernel_size=3, expansion=2, res_block=True)

        self.x3_block = MedNeXtBlock(in_channels=in_channles, 
                                out_channels=in_channles, 
                                stride=1, norm_name='instance', 
                                kernel_size=3, expansion=2, res_block=True)
        
        self.x4_block = MedNeXtBlock(in_channels=in_channles, 
                                out_channels=in_channles, 
                                stride=1, norm_name='instance', 
                                kernel_size=3, expansion=2, res_block=True)


    def forward(self, x):

        x1 = self.x1_block(x)
        x2 = self.x2_block(x1)

        x = x1 + x2

        x3 = self.x3_block(x)
        x4 = self.x4_block(x3)

        x = x + x3 + x4

        return x

class MambaEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
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
            n_groups=None
        )
        
        self.mednext_block_2 = MedNeXtBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            exp_r=exp_r,
            kernel_size=kernel_size,
            do_res=do_res and (in_channels == out_channels),
            norm_type=norm_type,
            n_groups=None
        )

        self.down_block_1 = MedNeXtDownBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            exp_r=exp_r,
            kernel_size=kernel_size,
            do_res=do_res,
            norm_type=norm_type
        )

    def forward(self, x, seg_features=None):
        x = self.mednext_block_1(x)
        x = self.mednext_block_2(x)

        if seg_features is not None:
            if x.shape[2:] != seg_features.shape[2:]:
                seg_features = torch.nn.functional.interpolate(
                    seg_features,
                    size=x.shape[2:],
                    mode='trilinear',
                    align_corners=False
                )

            if x.shape[1] != seg_features.shape[1]:
                if not hasattr(self, 'channel_proj'):
                    self.channel_proj = nn.Conv3d(seg_features.shape[1], x.shape[1], 1).to(x.device)
                seg_features = self.channel_proj(seg_features)

            x = x + seg_features

        skip_features = x.clone()
        x = self.down_block(x)

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
            do_res=False,
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
            n_groups=None
        )
        
        self.mednext_block_2 = MedNeXtBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            exp_r=exp_r,
            kernel_size=kernel_size,
            do_res=do_res,
            norm_type=norm_type,
            n_groups=None
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