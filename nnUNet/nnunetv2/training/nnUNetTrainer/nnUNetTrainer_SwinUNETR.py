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
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
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
                                num_input_channels: int,
                                num_output_channels: int,
                                enable_deep_supervision: bool = False) -> nn.Module:

        # Remove this line since num_input_channels is now passed as a parameter
        # label_manager = plans_manager.get_label_manager(dataset_json)
        # num_input_channels = label_manager.num_input_channels

        model = SwinUNETR(
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 6, 12),
            feature_size=48,
            norm_name="instance",
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            normalize=True,
            use_checkpoint=False,
            spatial_dims=3,
            downsample="merging",
            use_v2=False,
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
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.num_epochs = 10

'''# nnUNetTrainer_SegMamba.py

import torch
from typing import Union, List, Tuple
from torch._dynamo import OptimizedModule
from nnunetv2.nets.swinunetr import SwinUNETRv
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainer_SwinUNETR(nnUNetTrainer):
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
            enable_deep_supervision: bool = True) -> torch.nn.Module:
        segmamba_depths = [2, 2, 2, 2]
        segmamba_feat_size = [48, 96, 192, 384]
        model = SwinUNETRv(
            in_chans=num_input_channels,
            out_chans=num_output_channels,
            depths=segmamba_depths,
            feat_size=segmamba_feat_size,
            do_deep_supervision=enable_deep_supervision 
        )
        return model
    
    def _get_base_model(self):
        mod = self.network.module if self.is_ddp else self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        return mod

    def set_deep_supervision_enabled(self, enabled: bool):
        mod = self._get_base_model()
        mod.do_deep_supervision = enabled

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
    
    
    '''