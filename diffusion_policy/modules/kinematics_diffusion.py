from __future__ import annotations
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import copy
# import einops
from torch.distributions.normal import Normal
from diffusion_policy.backbone.base_backbone import ConditionalSeqBackbone
from diffusion_policy.modules.diffusion_model import SequentialDiffusionModel, make_timesteps, extract, cosine_beta_schedule
from diffusion_policy.utils.normalizer import LinearNormalizer

class Diffusion(SequentialDiffusionModel):
    """ 
    This Module implements an Unconditional Diffusion Model for sequential data, 
    which is used in Diffusion Policy
    """
    def __init__(
        self,
        backbone: ConditionalSeqBackbone,
        **kwargs,
    ):
    
        super().__init__(backbone, **kwargs)

        self.obs_dim = backbone.cond_dim
        self.n_past_steps = backbone.n_cond_steps
        self.action_dim = self.output_dim
        self.n_future_steps = self.horizon - self.n_past_steps

    def act(self, nobs):
        # clip nobs shape
        B = nobs.shape[0]
        nobs = nobs[:,:self.n_past_steps]
        trajectory = torch.randn((B, *self.seq_shape), device=self.device)

        trajectory[:, :self.n_past_steps] = nobs

        # Diffusion Loop
        t_all = torch.flip(torch.arange(self.denoising_steps), dims=(0,))
        t_all = t_all.unsqueeze(1).repeat(1, self.horizon)

        # Diffusion Loop
        for i, t in enumerate(t_all):
            trajectory = self.diffuse_step(trajectory, t, i, cond=nobs)