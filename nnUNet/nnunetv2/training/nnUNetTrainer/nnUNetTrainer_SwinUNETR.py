# nnUNetTrainer_SwinUNETR.py

import torch
import torch.nn as nn
from typing import Union, List, Tuple
from torch._dynamo import OptimizedModule
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision
from monai.networks.nets import SwinUNETR
import numpy as np


class SwinUNETRPaddingWrapper(nn.Module):
    """
    Wrapper for SwinUNETR that handles padding to ensure spatial dimensions are divisible by 32.
    SwinUNETR requires all spatial dimensions to be divisible by 2^5 = 32.
    """
    def __init__(self, model: SwinUNETR):
        super().__init__()
        self.model = model
        self.padding_dims = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with automatic padding.
        
        Args:
            x: Input tensor of shape (B, C, H, W, D) for 3D
        
        Returns:
            Output tensor with padding removed
        """
        # Get original shape
        original_shape = x.shape
        
        # Calculate padding needed for each spatial dimension
        # Spatial dimensions start at index 2
        padding = []
        self.padding_dims = []
        
        for i in range(2, len(original_shape)):
            spatial_size = original_shape[i]
            # Round up to nearest multiple of 32
            padded_size = int(np.ceil(spatial_size / 32) * 32)
            pad_amount = padded_size - spatial_size
            self.padding_dims.append((spatial_size, padded_size, pad_amount))
            padding.append(pad_amount)
        
        # Apply padding if needed
        if any(p > 0 for p in padding):
            # torch.nn.functional.pad expects padding in reverse order (last dim first)
            pad_tuple = []
            for p in reversed(padding):
                pad_tuple.extend([0, p])
            x_padded = torch.nn.functional.pad(x, pad_tuple)
        else:
            x_padded = x
        
        # Forward pass through SwinUNETR
        output = self.model(x_padded)
        
        # Remove padding from output
        if self.padding_dims:
            slices = [slice(None), slice(None)]  # Keep batch and channel dimensions
            for original_size, _, _ in self.padding_dims:
                slices.append(slice(None, original_size))
            output = output[tuple(slices)]
        
        return output



class nnUNetTrainer_SwinUNETR(nnUNetTrainerNoDeepSupervision):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans=plans, configuration=configuration, fold=fold, dataset_json=dataset_json, 
                         device=device)
        
        self.initial_lr = 1e-2 
        self.weight_decay = 1e-5 
        self.model_name = "SwinUNETR"
        self.num_epochs = 10

    @staticmethod
    def build_network_architecture(
            architecture_class_name: str,
            arch_init_kwargs: dict,
            arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
            num_input_channels: int,
            num_output_channels: int,
            enable_deep_supervision: bool = False) -> torch.nn.Module:
        
        # Create the base SwinUNETR model
        model = SwinUNETR(
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            use_v2=True,
        )
        
        # Wrap it with padding handler
        wrapped_model = SwinUNETRPaddingWrapper(model)
        return wrapped_model

    def _get_base_model(self):
        """Get the actual SwinUNETR model, handling wrapper and DDP."""
        mod = self.network.module if self.is_ddp else self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        
        # Unwrap the padding wrapper if present
        if isinstance(mod, SwinUNETRPaddingWrapper):
            mod = mod.model
        
        return mod

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        SwinUNETR does not support deep supervision, so this method does nothing.
        Deep supervision is disabled by default in nnUNetTrainerNoDeepSupervision.
        """
        pass

    def save_checkpoint(self, filename: str) -> None:
        """Override to save model without deep supervision weights for inference compatibility."""
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                mod = self._get_base_model()
                
                try:
                    state_dict = mod.state_dict()
                    
                    # SwinUNETR doesn't have deep supervision heads, so just save as-is
                    filtered_state_dict = state_dict
                    
                    checkpoint = {
                        'network_weights': filtered_state_dict,
                        'optimizer_state': self.optimizer.state_dict(),
                        'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                        'logging': self.logger.get_checkpoint(),
                        '_best_ema': self._best_ema,
                        'current_epoch': self.current_epoch + 1,
                        'init_args': self.my_init_kwargs,
                        'trainer_name': self.__class__.__name__,
                        'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                    }
                    torch.save(checkpoint, filename)
                except Exception as e:
                    self.print_to_log_file(f'Failed to save checkpoint: {e}')
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')

    def perform_actual_validation(self, save_probabilities: bool = False):
        """Validation doesn't need special handling for SwinUNETR."""
        return super().perform_actual_validation(save_probabilities)

    def validation_step(self, batch: dict) -> dict:
        """Validation step doesn't need special handling for SwinUNETR."""
        return super().validation_step(batch)