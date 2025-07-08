from typing import Tuple, Union

import torch
from diffusion_policy.backbone.base_backbone import BaseBackbone
from diffusion_policy.utils.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.utils.normalizer import LinearNormalizer
from diffusion_policy.dataset.offline_dataset import BaseDataset

class BaseActor(ModuleAttrMixin):
    def __init__(self, 
                 backbone: BaseBackbone,
                 **kwargs):
        if kwargs:
            print("BaseActor.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ModuleAttrMixin, self).__init__()

        self.backbone = backbone
        print(f"Actor Network: {self.backbone}")

        self.normalizer: LinearNormalizer = None
        self.dataset_class: BaseDataset = None
    
    def get_optimizer(
            self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        return self.backbone.configure_optimizers(
                weight_decay=weight_decay, 
                learning_rate=learning_rate, 
                betas=tuple(betas))
    
    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def set_dataset_class(self, dataset_class):
        self.dataset_class = dataset_class