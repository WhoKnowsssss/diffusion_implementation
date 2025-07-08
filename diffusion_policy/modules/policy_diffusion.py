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

from diffusion_policy.modules.base_actor import BaseActor

class DiffusionActor(SequentialDiffusionModel, BaseActor):
    """ 
    This Module implements a Conditional Diffusion Model for sequential data, 
    which is used in Diffusion Policy
    """
    backbone: ConditionalSeqBackbone
    def __init__(
        self,
        **kwargs,
    ):
        SequentialDiffusionModel.__init__(self, **kwargs)
        BaseActor.__init__(self, backbone=self.backbone)

        self.obs_dim = self.backbone.cond_dim
        self.n_past_steps = self.backbone.n_cond_steps
        self.action_dim = self.output_dim
        self.n_future_steps = self.horizon - self.n_past_steps

    def act(self, nobs):
        # clip nobs shape
        B = nobs.shape[0]
        nobs = nobs[:,:self.n_past_steps]
        trajectory = torch.randn((B, *self.seq_shape), device=self.device)

        # Diffusion Loop
        t_all = torch.flip(torch.arange(self.denoising_steps), dims=(0,))
        t_all = t_all.unsqueeze(1).repeat(1, self.horizon)

        # Diffusion Loop
        for i, t in enumerate(t_all):
            trajectory = self.diffuse_step(trajectory, t, i, cond=nobs)

    def p_losses(
        self,
        trajectory,
        cond,
    ):
        """
        If predicting epsilon: E_{t, x0, ε} [||ε - ε_θ(√α̅ₜx0 + √(1-α̅ₜ)ε, t)||²

        Args:
            trajectory: (B, horizon, transition_dim)
            cond: dict with keys as step and value as observation
            t: batch of integers
        """

        # Forward process
        B = trajectory.shape[0]
        device = trajectory.device
        noise = torch.randn_like(trajectory, device=device)

        # diffusion sampling
        t = torch.randint(
            0, self.denoising_steps, (B, self.horizon), device=device
        ).long()
        
        x_noisy = self.q_sample(trajectory=trajectory, t=t, noise=noise)
        
        # Reverse process
        x_pred = self.backbone(x_noisy, t, cond=cond)

        if self.predict_epsilon:
            return torch.nn.functional.mse_loss(x_pred, noise, reduction="mean")
        else:
            return torch.nn.functional.mse_loss(x_pred, trajectory, reduction="mean")