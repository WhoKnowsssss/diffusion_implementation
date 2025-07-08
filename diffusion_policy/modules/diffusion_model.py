from __future__ import annotations
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import copy
# import einops
from diffusion_policy.utils.module_attr_mixin import ModuleAttrMixin

from diffusion_policy.backbone.base_backbone import SequentialBackbone
from diffusion_policy.utils.normalizer import LinearNormalizer

class SequentialDiffusionModel(ModuleAttrMixin):
    """
    This Module implements an Unconditional Diffusion Model for sequential data
    """

    def __init__(
        self,
        backbone: SequentialBackbone,

        # DDPM parameters
        denoising_steps=10,
        predict_epsilon=True,
        denoised_clip_value=1.0,
        **kwargs,
    ):    
        super(ModuleAttrMixin, self).__init__()

        self.denoising_steps = int(denoising_steps)
        self.denoised_clip_value = denoised_clip_value
        self.predict_epsilon = predict_epsilon

        self.normalizer: LinearNormalizer = None
        self.backbone: SequentialBackbone = backbone
        self.horizon = backbone.horizon
    
        self.output_dim = backbone.output_dim
        self.seq_shape = (self.horizon, self.output_dim)

    def to(self, device):
        super().to(device)
        self.DDPM_init()
        return self

    def forward(self, B):
        trajectory = torch.randn((B, *self.seq_shape), device=self.device)

        # Diffusion Loop
        t_all = torch.flip(torch.arange(self.denoising_steps), dims=(0,))
        t_all = t_all.unsqueeze(1).repeat(1, self.horizon)

        for i, t in enumerate(t_all):
            trajectory = self.diffuse_step(trajectory, t, i)

        return trajectory
    
    def diffuse_step(self, trajectory, t, i, **kwargs):
        B = trajectory.shape[0]
        device = self.device
        # t.shape = (T,)
        t_b = make_timesteps(B, t, device)
        # t_b.shape = (B,T)
        index_b = make_timesteps(B, i, device)
        mu, logvar = self.p_mean_var(x=trajectory, t=t_b, index=index_b, **kwargs)
        std = torch.exp(0.5 * logvar)

        # no noise when t == 0
        noise = torch.randn_like(trajectory)
        noise[t_b == 0] = 0

        trajectory = mu + std * noise

        return trajectory

    def p_mean_var(self, x, t, index=None, **kwargs):
        output = self.backbone(x, t, **kwargs)

        # Predict x_0
        if self.predict_epsilon:
            """
            x₀ = √ 1\α̅ₜ xₜ - √ 1\α̅ₜ-1 ε
            """
            x_pred = self.predict_x0_from_epsilon(x, t, output)
        else:   # directly predicting x₀
            x_pred = output

        if self.denoised_clip_value is not None:
            x_pred.clamp_(-self.denoised_clip_value, self.denoised_clip_value)

        # Get mu
        """
        μₜ = β̃ₜ √ α̅ₜ₋₁/(1-α̅ₜ)x₀ + √ αₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)xₜ
        """
        mu = (
            extract(self.ddpm_mu_coef1, t, x.shape) * x_pred
            + extract(self.ddpm_mu_coef2, t, x.shape) * x
        )
        logvar = extract(
            self.ddpm_logvar_clipped, t, x.shape
        )
        return mu, logvar

    def p_losses(
        self,
        trajectory,
    ):
        """
        If predicting epsilon: E_{t, x0, ε} [||ε - ε_θ(√α̅ₜx0 + √(1-α̅ₜ)ε, t)||²

        Args:
            trajectory: (B, horizon, output_dim)
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
        x_pred = self.backbone(x_noisy, t)

        if self.predict_epsilon:
            return torch.nn.functional.mse_loss(x_pred, noise, reduction="mean")
        else:
            return torch.nn.functional.mse_loss(x_pred, trajectory, reduction="mean")
    
# ------------------------------------------------ Diffusion Helper Functions -------------------------------------
    
    def DDPM_init(self):
        """
        DDPM parameters

        """
        """
        βₜ
        """
        self.betas = cosine_beta_schedule(self.denoising_steps).to(self.device)
        """
        αₜ = 1 - βₜ
        """
        self.alphas = 1.0 - self.betas
        """
        α̅ₜ= ∏ᵗₛ₌₁ αₛ 
        """
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        """
        α̅ₜ₋₁
        """
        self.alphas_cumprod_prev = torch.cat([torch.ones(1).to(self.device), self.alphas_cumprod[:-1]])
        """
        √ α̅ₜ
        """
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        """
        √ 1-α̅ₜ
        """
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        """
        √ 1\α̅ₜ
        """
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        """
        √ 1\α̅ₜ-1
        """
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        """
        β̃ₜ = σₜ² = βₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)
        """
        self.ddpm_var = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.ddpm_logvar_clipped = torch.log(torch.clamp(self.ddpm_var, min=1e-20))
        """
        μₜ = β̃ₜ √ α̅ₜ₋₁/(1-α̅ₜ)x₀ + √ αₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)xₜ
        """
        self.ddpm_mu_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.ddpm_mu_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
    
    def predict_x0_from_epsilon(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_epsilon_from_x0(self, x_t, t, x0):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )

    def q_sample(self, trajectory, t, noise=None):
        """
        q(xₜ | x₀) = 𝒩(xₜ; √ α̅ₜ x₀, (1-α̅ₜ)I)
        xₜ = √ α̅ₜ xₒ + √ (1-α̅ₜ) ε
        """
        if noise is None:
            device = trajectory.device
            noise = torch.randn_like(trajectory, device=device)
        return (
            extract(self.sqrt_alphas_cumprod, t, trajectory.shape) * trajectory
            + extract(self.sqrt_one_minus_alphas_cumprod, t, trajectory.shape) * noise
        )
    
def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)

def extract(a, t, x_shape):
    b, l = t.shape
    out = a[t]
    return out.reshape(b, l, *((1,) * (len(x_shape) - 2)))

def make_timesteps(B, i, device):
    if isinstance(i, int):
        t = torch.full((B,), i, device=device, dtype=torch.long)
    else:
        t = i.unsqueeze(0).repeat(B, 1).to(dtype=torch.long, device=device)
    return t

def to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    elif type(x) is dict:
        return {k: to_device(v, device) for k, v in x.items()}
    else:
        print(f"Unrecognized type in `to_device`: {type(x)}")


def batch_to_device(batch, device):
    vals = [to_device(getattr(batch, field), device) for field in batch._fields]
    return type(batch)(*vals)
