# nnUNetTrainer_SwinUNETR.py

import torch
import torch.nn as nn
from typing import Union, List, Tuple
from torch._dynamo import OptimizedModule
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.nets.swinunetr import SwinUNETR
import numpy as np


class nnUNetTrainer_SwinUNETR(nnUNetTrainer):
    """
    Trainer for SwinUNETR model with SegMamba-style architecture.
    Now supports deep supervision like SegMamba.
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans=plans, configuration=configuration, fold=fold, dataset_json=dataset_json, 
                         device=device)
        
        self.initial_lr = 1e-2 
        self.weight_decay = 1e-5 
        self.model_name = "SwinUNETR"
        self.num_epochs = 1000

    @staticmethod
    def build_network_architecture(
            architecture_class_name: str,
            arch_init_kwargs: dict,
            arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
            num_input_channels: int,
            num_output_channels: int,
            enable_deep_supervision: bool = False) -> torch.nn.Module:
        
        # Create the custom SwinUNETR model with SegMamba-style architecture
        model = SwinUNETR(
            in_chans=num_input_channels,
            out_chans=num_output_channels,
            feat_size=[48, 96, 192, 384],
            hidden_size=768,
            norm_name="instance",
            res_block=True,
            spatial_dims=3,
            do_deep_supervision=enable_deep_supervision,
        )
        
        return model

