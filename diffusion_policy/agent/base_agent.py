from typing import Dict
from abc import abstractmethod

import torch
import torch.nn as nn
import numpy as np
from diffusion_policy.utils.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.utils.normalizer import LinearNormalizer
from diffusion_policy.dataset.offline_dataset import BaseDataset
from diffusion_policy.modules.base_actor import BaseActor


class BaseAgent(ModuleAttrMixin):
    def __init__(
        self,
        actor: BaseActor,
    ):
    
        super().__init__()
        self.normalizer: LinearNormalizer = LinearNormalizer()
        self.dataset_class: BaseDataset = None

        # Set up models
        self.actor = actor.to(self.device)

    def test_mode(self):
        pass

    def train_mode(self):
        pass

    @abstractmethod
    def act(self, obs_dict) -> torch.Tensor:
        pass

    def process_env_step(self, rewards, dones, infos):
        pass

    def compute_returns(self, last_critic_obs):
        pass

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer = normalizer
        self.actor.set_normalizer(normalizer)

    def set_dataset_class(self, dataset_class):
        self.dataset_class = dataset_class
        self.actor.set_dataset_class(dataset_class)

    @abstractmethod
    def get_optimizer(
            self, *args, **kwargs
        ) -> Dict[str, torch.optim.Optimizer]:
        pass
    
    @abstractmethod
    def compute_loss(self, batch) -> Dict[str, torch.Tensor]:
        pass

    def to(self, device):
        super().to(device)
        self.actor.to(device)
        return self

    @property
    def horizon(self):
        return self.actor.horizon
    
    @property
    def n_past_steps(self):
        return self.actor.n_past_steps
    
    @property
    def n_future_steps(self):
        return self.actor.n_future_steps