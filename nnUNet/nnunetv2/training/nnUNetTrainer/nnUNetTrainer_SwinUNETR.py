# nnUNetTrainer_SegMamba.py

# nnUNetTrainer_SwinUNETR.py

import torch
from typing import Union, List, Tuple
from torch._dynamo import OptimizedModule
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision
from monai.networks.nets import SwinUNETR


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
        
        model = SwinUNETR(
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            feature_size=48,
            use_v2=True,
        )
        return model

    def _get_base_model(self):
        mod = self.network.module if self.is_ddp else self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
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
                original_deep_supervision = getattr(mod, 'do_deep_supervision', True)
                
                try:
                    state_dict = mod.state_dict()
                    
                    # CORRECTED filtering logic for SegMamba:
                    # Remove deep supervision heads ("ds_seg_from_dec...").
                    # Keep everything else, including the main output ("out_main_seg").
                    filtered_state_dict = {
                        k: v for k, v in state_dict.items()
                        if not k.startswith('ds_seg_from_dec')
                    }
                    
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
                finally:
                    mod.do_deep_supervision = original_deep_supervision
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')

    def perform_actual_validation(self, save_probabilities: bool = False):
        mod = self._get_base_model()
        original_forward = mod.forward
        mod.forward = lambda x: original_forward(x)[0]
        
        original_deep_supervision = getattr(mod, 'do_deep_supervision', True)
        mod.do_deep_supervision = False
        
        try:
            result = super().perform_actual_validation(save_probabilities)
        finally:
            mod.forward = original_forward
            mod.do_deep_supervision = original_deep_supervision
            
        return result

    def validation_step(self, batch: dict) -> dict:
        mod = self._get_base_model()
        original_deep_supervision = getattr(mod, 'do_deep_supervision', True)
        mod.do_deep_supervision = False
        try:
            result = super().validation_step(batch)
        finally:
            mod.do_deep_supervision = original_deep_supervision
        return result