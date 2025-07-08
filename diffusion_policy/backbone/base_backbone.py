from abc import ABC, abstractmethod
from typing import Tuple, Optional

import torch
import torch.nn as nn
from diffusion_policy.utils.module_attr_mixin import ModuleAttrMixin

class  BaseBackbone(ModuleAttrMixin):
    def __init__(self, x_input_dim, x_output_dim, **kwargs):
        super().__init__()
        self.x_input_dim = x_input_dim
        self.x_output_dim = x_output_dim

    @property
    def input_dim(self):
        # assume only x
        return self.x_input_dim
    
    @property
    def output_dim(self):
        # assume only x
        return self.x_output_dim

    @abstractmethod
    def _init_weights(self, module):
        pass
    
    @abstractmethod
    def get_optim_groups(self, weight_decay: float=1e-3):
        pass

    @abstractmethod
    def configure_optimizers(self, 
            learning_rate: float=1e-4, 
            weight_decay: float=1e-3,
            betas: Tuple[float, float]=(0.9,0.95)):
       pass

    @abstractmethod
    def forward(self, 
        x_input: torch.Tensor, 
        **kwargs):
        pass

class  ConditionalBackbone(BaseBackbone):
    # implement a sequence modeling backbone
    def __init__(self, cond_dim, **kwargs):
        super().__init__(**kwargs)
        self.cond_dim = cond_dim

    def forward(self, 
        x_input: torch.Tensor, 
        cond: torch.Tensor,
        **kwargs):
        pass

class  SequentialBackbone(BaseBackbone):
    # implement a sequence modeling backbone
    def __init__(self, x_horizon, **kwargs):
        super().__init__(**kwargs)
        self.x_horizon = x_horizon

    @property
    def horizon(self):
        # assume only x
        return self.x_horizon

class  ConditionalSeqBackbone(SequentialBackbone):
    # implement a sequence modeling backbone
    def __init__(self, n_cond_steps, cond_dim, **kwargs):
        super().__init__(**kwargs)
        self.n_cond_steps = n_cond_steps
        self.cond_dim = cond_dim

    def forward(self, 
        x_input: torch.Tensor, 
        cond: torch.Tensor,
        x_timesteps: Optional[torch.Tensor] = None,
        **kwargs):
        pass

class JointSeqBackbone(SequentialBackbone):
    # implement a joint distribution of "x" and "y"
    def __init__(self, y_horizon, y_input_dim, y_output_dim, **kwargs):
        super().__init__(**kwargs)
        self.y_horizon = y_horizon
        self.y_input_dim = y_input_dim
        self.y_output_dim = y_output_dim

    def forward(self, 
        x_input: torch.Tensor, 
        y_input: torch.Tensor,
        x_timesteps: Optional[torch.Tensor] = None,
        y_timesteps: Optional[torch.Tensor] = None,
        **kwargs):
        pass