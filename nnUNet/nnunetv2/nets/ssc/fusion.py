import torch 
import torch.nn as nn
import torch.nn.functional as F 

class Conv1x1Fusion(nn.Module):
    """
    Baseline fusion method using simple 1x1x1 convolution.
    Concatenates features from both encoders and applies channel-wise mixing.
    
    Args:
        channels: Number of channels from each encoder (both should have same channels)
        norm_type: Type of normalization ('instance', 'batch', 'group')
    """
    def __init__(self, channels, norm_type='instance', **kwargs):
        super(Conv1x1Fusion, self).__init__()
        
        # Concatenate 2C -> C reduction
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1, bias=False),
            self._get_norm_layer(channels, norm_type),
            nn.GELU()
        )
        
    def _get_norm_layer(self, channels, norm_type):
        if norm_type == 'instance':
            return nn.InstanceNorm3d(channels, affine=True)
        elif norm_type == 'batch':
            return nn.BatchNorm3d(channels)
        elif norm_type == 'group':
            return nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")
    
    def forward(self, feat_mamba, feat_mednext):
        
        concat = torch.cat([feat_mamba, feat_mednext], dim=1)  # (B, 2C, D, H, W)
        fused = self.fusion_conv(concat)  # (B, C, D, H, W)
        
        return fused

class BidirectionalCrossAttentionFusion(nn.Module):
    """
    Bidirectional multi-head cross-attention fusion for dual encoder features.
    
    This module allows features from one encoder pathway to attend to the other
    and vice versa, enabling adaptive feature recalibration.
    
    Args:
        channels: Number of channels from each encoder
        num_heads: Number of attention heads (must divide channels evenly)
        dropout: Dropout rate for attention weights and FFN
        norm_type: Type of normalization ('instance', 'batch', 'group')
        qkv_bias: Whether to use bias in Q, K, V projections
    """
    def __init__(self, channels, num_heads=8, dropout=0.1, norm_type='instance', qkv_bias=True):
        super(BidirectionalCrossAttentionFusion, self).__init__()
        
        assert channels % num_heads == 0, f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Cross-attention: Path 1 queries Path 2
        self.norm1 = nn.LayerNorm(channels)
        self.q1 = nn.Linear(channels, channels, bias=qkv_bias)
        self.kv2 = nn.Linear(channels, channels * 2, bias=qkv_bias)
        self.proj1 = nn.Linear(channels, channels)
        self.dropout1 = nn.Dropout(dropout)
        
        # Cross-attention: Path 2 queries Path 1
        self.norm2 = nn.LayerNorm(channels)
        self.q2 = nn.Linear(channels, channels, bias=qkv_bias)
        self.kv1 = nn.Linear(channels, channels * 2, bias=qkv_bias)
        self.proj2 = nn.Linear(channels, channels)
        self.dropout2 = nn.Dropout(dropout)
        
        # Fusion of both attended features
        self.norm_fusion = nn.LayerNorm(channels * 2)
        self.fusion_ffn = nn.Sequential(
            nn.Linear(channels * 2, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout)
        )
        
        self.norm_final = nn.LayerNorm(channels)
        
    def forward(self, feat1, feat2):
        B, C, D, H, W = feat1.shape
        
        # Reshape to sequence: (B, C, D, H, W) -> (B, D*H*W, C)
        n_tokens = D * H * W
        feat1_flat = feat1.flatten(2).transpose(1, 2)
        feat2_flat = feat2.flatten(2).transpose(1, 2)
        
        # Residual connections
        feat1_residual = feat1_flat
        feat2_residual = feat2_flat
        
        # 1: Path 1 attends to Path 2 ---
        feat1_norm = self.norm1(feat1_flat)
        q1 = self.q1(feat1_norm).reshape(B, n_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        
        kv2 = self.kv2(feat2_flat).reshape(B, n_tokens, 2, self.num_heads, self.head_dim)
        k2, v2 = kv2.permute(2, 0, 3, 1, 4)
        
        attn1 = (q1 @ k2.transpose(-2, -1)) * self.scale
        attn1 = F.softmax(attn1, dim=-1)
        attn1 = self.dropout1(attn1)
        
        feat1_attended = (attn1 @ v2).transpose(1, 2).reshape(B, n_tokens, C)
        feat1_attended = self.proj1(feat1_attended)
        feat1_attended = self.dropout1(feat1_attended)
        feat1_enhanced = feat1_residual + feat1_attended

        # 2: Path 2 attends to Path 1 ---
        feat2_norm = self.norm2(feat2_flat)
        q2 = self.q2(feat2_norm).reshape(B, n_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        
        kv1 = self.kv1(feat1_flat).reshape(B, n_tokens, 2, self.num_heads, self.head_dim)
        k1, v1 = kv1.permute(2, 0, 3, 1, 4)
        
        attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale
        attn2 = F.softmax(attn2, dim=-1)
        attn2 = self.dropout2(attn2)
        
        feat2_attended = (attn2 @ v1).transpose(1, 2).reshape(B, n_tokens, C)
        feat2_attended = self.proj2(feat2_attended)
        feat2_attended = self.dropout2(feat2_attended)
        feat2_enhanced = feat2_residual + feat2_attended
        
        # 3: Fuse both enhanced features ---
        concat_enhanced = torch.cat([feat1_enhanced, feat2_enhanced], dim=-1)
        fused = self.fusion_ffn(self.norm_fusion(concat_enhanced))
        fused = fused + (feat1_residual + feat2_residual) / 2
        fused = self.norm_final(fused)
        
        # Reshape back to image format
        fused = fused.transpose(1, 2).reshape(B, C, D, H, W)
        
        return fused


'''
import torch 
import torch.nn as nn
import torch.nn.functional as F 

class Conv1x1Fusion(nn.Module):
    """
    Baseline fusion method using simple 1x1x1 convolution.
    Concatenates features from both encoders and applies channel-wise mixing.
    
    Args:
        channels: Number of channels from each encoder (both should have same channels)
        norm_type: Type of normalization ('instance', 'batch', 'group')
    """
    def __init__(self, channels, norm_type='instance'):
        super(Conv1x1Fusion, self).__init__()
        
        # Concatenate 2C -> C reduction
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1, bias=False),
            self._get_norm_layer(channels, norm_type),
            nn.GELU()
        )
        
    def _get_norm_layer(self, channels, norm_type):
        if norm_type == 'instance':
            return nn.InstanceNorm3d(channels, affine=True)
        elif norm_type == 'batch':
            return nn.BatchNorm3d(channels)
        elif norm_type == 'group':
            return nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")
    
    def forward(self, feat_mamba, feat_mednext):
        
        # Ensure spatial dimensions match
        #if feat_mamba.shape[2:] != feat_mednext.shape[2:]:
        #    feat_mednext = F.interpolate(
        #        feat_mednext,
        #        size=feat_mamba.shape[2:],
        #        mode='trilinear',
        #        align_corners=False
        #    )
        
        concat = torch.cat([feat_mamba, feat_mednext], dim=1)  # (B, 2C, D, H, W)
        fused = self.fusion_conv(concat)  # (B, C, D, H, W)
        
        return fused

class BidirectionalCrossAttentionFusion(nn.Module):
    """
    Bidirectional multi-head cross-attention fusion for dual encoder features.
    
    This module allows features from Mamba encoder to attend to MedNeXt features
    and vice versa, enabling adaptive feature recalibration based on complementary
    information from both encoding pathways.
    
    Args:
        channels: Number of channels from each encoder
        num_heads: Number of attention heads (must divide channels evenly)
        dropout: Dropout rate for attention weights and FFN
        norm_type: Type of normalization ('instance', 'batch', 'group')
        qkv_bias: Whether to use bias in Q, K, V projections
    """
    def __init__(self, channels, num_heads=8, dropout=0.1, norm_type='instance', qkv_bias=True):
        super(BidirectionalCrossAttentionFusion, self).__init__()
        
        assert channels % num_heads == 0, f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Cross-attention: Mamba queries MedNeXt
        self.norm_mamba_1 = nn.LayerNorm(channels)
        self.q_mamba = nn.Linear(channels, channels, bias=qkv_bias)
        self.kv_mednext_1 = nn.Linear(channels, channels * 2, bias=qkv_bias)
        self.proj_mamba = nn.Linear(channels, channels)
        self.dropout_mamba = nn.Dropout(dropout)
        
        # Cross-attention: MedNeXt queries Mamba
        self.norm_mednext_1 = nn.LayerNorm(channels)
        self.q_mednext = nn.Linear(channels, channels, bias=qkv_bias)
        self.kv_mamba_1 = nn.Linear(channels, channels * 2, bias=qkv_bias)
        self.proj_mednext = nn.Linear(channels, channels)
        self.dropout_mednext = nn.Dropout(dropout)
        
        # Fusion of both attended features
        self.norm_fusion = nn.LayerNorm(channels * 2)
        self.fusion_ffn = nn.Sequential(
            nn.Linear(channels * 2, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout)
        )
        
        self.norm_final = nn.LayerNorm(channels)
        
    def forward(self, feat_mamba, feat_mednext):
    
        B, C, D, H, W = feat_mamba.shape
        
        # - I don't want to add this overhead - I expect the dims to match naturally
        # Ensure spatial dimensions match
        # if feat_mamba.shape[2:] != feat_mednext.shape[2:]:
        #    feat_mednext = F.interpolate(
        #        feat_mednext,
        #        size=feat_mamba.shape[2:],
        #        mode='trilinear',
        #        align_corners=False
        #    )
        
        # Reshape to sequence: (B, C, D, H, W) -> (B, D*H*W, C)
        n_tokens = D * H * W
        mamba_flat = feat_mamba.flatten(2).transpose(1, 2)  # (B, N, C)
        mednext_flat = feat_mednext.flatten(2).transpose(1, 2)  # (B, N, C)
        
        # residual connections
        mamba_residual = mamba_flat
        mednext_residual = mednext_flat
        
        # 1: Mamba attends to MedNeXt ---
        mamba_norm = self.norm_mamba_1(mamba_flat)
        q_m = self.q_mamba(mamba_norm).reshape(B, n_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        
        kv_mn = self.kv_mednext_1(mednext_flat).reshape(B, n_tokens, 2, self.num_heads, self.head_dim)
        k_mn, v_mn = kv_mn.permute(2, 0, 3, 1, 4)  # 2 x (B, num_heads, N, head_dim)
        
        attn_m = (q_m @ k_mn.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn_m = F.softmax(attn_m, dim=-1)
        attn_m = self.dropout_mamba(attn_m)
        
        mamba_attended = (attn_m @ v_mn).transpose(1, 2).reshape(B, n_tokens, C)
        mamba_attended = self.proj_mamba(mamba_attended)
        mamba_attended = self.dropout_mamba(mamba_attended)
        
        mamba_enhanced = mamba_residual + mamba_attended # res connection
        
        # 2: MedNeXt attends to Mamba ---
        mednext_norm = self.norm_mednext_1(mednext_flat)
        q_mn = self.q_mednext(mednext_norm).reshape(B, n_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        
        kv_m = self.kv_mamba_1(mamba_flat).reshape(B, n_tokens, 2, self.num_heads, self.head_dim)
        k_m, v_m = kv_m.permute(2, 0, 3, 1, 4)
        
        attn_mn = (q_mn @ k_m.transpose(-2, -1)) * self.scale
        attn_mn = F.softmax(attn_mn, dim=-1)
        attn_mn = self.dropout_mednext(attn_mn)
        
        mednext_attended = (attn_mn @ v_m).transpose(1, 2).reshape(B, n_tokens, C)
        mednext_attended = self.proj_mednext(mednext_attended)
        mednext_attended = self.dropout_mednext(mednext_attended)
        
        mednext_enhanced = mednext_residual + mednext_attended # res connection
        
        # 3: Fuse both enhanced features ---
        concat_enhanced = torch.cat([mamba_enhanced, mednext_enhanced], dim=-1)  # (B, N, 2C)
        fused = self.fusion_ffn(self.norm_fusion(concat_enhanced))  # (B, N, C)
        fused = fused + (mamba_residual + mednext_residual) / 2
        fused = self.norm_final(fused)
        
        fused = fused.transpose(1, 2).reshape(B, C, D, H, W) # back to (B, C, D, H, W)
        
        return fused
        '''