from __future__ import annotations
from typing import TYPE_CHECKING

import torch
from torch.distributions.normal import Normal
import numpy as np

from diffusion_policy.modules.policy_diffusion import (
    make_timesteps,
    extract,
)
from diffusion_policy.modules.diffusion_model import SequentialDiffusionModel
from diffusion_policy.modules.base_actor import BaseActor

from diffusion_policy.backbone.base_backbone import JointSeqBackbone


class JointDiffusionActor(SequentialDiffusionModel, BaseActor):
    """
    This module implements the basics of a Joint-distribution Diffusion Model for states and actions,
    """
    backbone: JointSeqBackbone

    def __init__(
        self,
        action_weight_schedule="constant",
        action_denoising_steps=0,
        state_weight_schedule="constant",
        state_denoising_steps=0,
        clean_past_state=True,
        state_pred_epsilon=None,
        action_pred_epsilon=None,
        n_past_steps=8,
        **kwargs,
    ):
        SequentialDiffusionModel.__init__(self, **kwargs)
        BaseActor.__init__(self, backbone=self.backbone)
        self.backbone: JointSeqBackbone

        self.horizon = max(self.backbone.x_horizon, self.backbone.y_horizon)

        self.obs_dim = self.backbone.x_output_dim
        self.n_past_steps = n_past_steps
        self.action_dim = self.backbone.y_output_dim
        self.n_future_steps = self.horizon - self.n_past_steps


        self.action_loss_weights = self.get_loss_weights(action_weight_schedule)
        self.action_denoising_steps = (
            action_denoising_steps if action_denoising_steps > 0 else self.denoising_steps
        )
        self.state_loss_weights = self.get_loss_weights(state_weight_schedule)
        self.state_denoising_steps = (
            state_denoising_steps if state_denoising_steps > 0 else self.denoising_steps
        )
        self.clean_past_state = clean_past_state  # whether past state is clean or not

        if state_pred_epsilon is None:
            self.state_pred_epsilon = self.predict_epsilon
        if action_pred_epsilon is None:
            self.action_pred_epsilon = self.predict_epsilon


    def act(
        self,
        nobs,
        past_actions=None,
        **kwargs,
    ):
        B = nobs.shape[0]
        nobs = nobs[:, :self.n_past_steps, :]
        # print(nobs[0, :, 186:189])

        action_traj = torch.randn((B, self.horizon, self.action_dim), device=self.device)
        state_traj = torch.randn((B, self.horizon, self.obs_dim), device=self.device)

        # Diffusion Loop
        t_all = torch.flip(torch.arange(self.denoising_steps), dims=(0,))
        t_all = t_all.unsqueeze(1).repeat(1, self.horizon)
        action_t_all = t_all.clone()
        state_t_all = t_all.clone()

        for i in range(max(len(action_t_all), len(state_t_all))):
            action_traj, state_traj = self.diffuse_step(
                nobs=nobs,
                action_traj=action_traj,
                state_traj=state_traj,
                action_t=action_t_all[i],
                state_t=state_t_all[i],
                i=i,
                past_actions=past_actions,
                **kwargs,
            )

        return action_traj, state_traj

    def diffuse_step(
        self,
        nobs,
        action_traj,
        state_traj,
        action_t,
        state_t,
        i,
        past_actions=None,
        **kwargs,
    ):
        device = self.device
        B, _, Do = nobs.shape
        
        # inpainting observation
        state_traj[:, : self.n_past_steps] = nobs
        state_t[: self.n_past_steps] = 0

        # if past_actions is not None:
        #     action_traj[:, :self.n_past_steps-1] = past_actions
        #     action_t[:self.n_past_steps-1] = 0

        action_t_b = make_timesteps(B, action_t, device)
        state_t_b = make_timesteps(B, state_t, device)

        index_b = make_timesteps(B, i, device)

        action_mu, state_mu, action_logvar, state_logvar = self.p_mean_var(
            action_traj=action_traj,
            state_traj=state_traj,
            action_t=action_t_b,
            state_t=state_t_b,
            index=index_b,
        )

        action_std = torch.exp(0.5 * action_logvar)
        state_std = torch.exp(0.5 * state_logvar)

        # no noise when t == 0
        noise = torch.randn_like(action_traj)
        noise[action_t_b == 0] = 0

        action_traj = action_mu + action_std * noise

        noise = torch.randn_like(state_traj)
        noise[state_t_b == 0] = 0

        state_traj = state_mu + state_std * noise

        self.distribution = Normal(action_mu, action_std)

        return action_traj, state_traj

    def predict_x0(
       self, action_traj, state_traj, action_t, state_t, **kwargs
    ):
        # action_t = action_t.clone() + 1
        # state_t = state_t.clone() + 1
        # state_t[:,:self.n_past_steps] = 0
        state_output, action_output = self.backbone.forward(
            x_input=state_traj,
            y_input=action_traj,
            x_timesteps=state_t,
            y_timesteps=action_t,
            **kwargs, 
        )
        # action_t = action_t.clone() - 1
        # state_t = torch.clip(state_t.clone() - 1, min=0)

        if self.state_pred_epsilon:
            state_pred = self.predict_x0_from_epsilon(state_traj, state_t, state_output)
        else:
            state_pred = state_output
        
        if self.action_pred_epsilon:
            action_pred = self.predict_x0_from_epsilon(action_traj, action_t, action_output)
        else:
            action_pred = action_output
        
        return action_pred, state_pred
    
    def mean_var_from_x0(
        self, action_traj, state_traj, action_t, state_t, action_pred, state_pred, **kwargs
    ):
        """
        μₜ = β̃ₜ √ α̅ₜ₋₁/(1-α̅ₜ)x₀ + √ αₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)xₜ
        """

        if self.denoised_clip_value is not None:
            state_pred.clamp_(-self.denoised_clip_value, self.denoised_clip_value)
            action_pred.clamp_(-self.denoised_clip_value, self.denoised_clip_value)
            
        action_mu = (
            extract(self.ddpm_mu_coef1, action_t, action_traj.shape) * action_pred
            + extract(self.ddpm_mu_coef2, action_t, action_traj.shape) * action_traj
        )
        state_mu = (
            extract(self.ddpm_mu_coef1, state_t, state_traj.shape) * state_pred
            + extract(self.ddpm_mu_coef2, state_t, state_traj.shape) * state_traj
        )
        action_logvar = extract(self.ddpm_logvar_clipped, action_t, action_traj.shape)
        state_logvar = extract(self.ddpm_logvar_clipped, state_t, state_traj.shape)

        return action_mu, state_mu, action_logvar, state_logvar
    
    def p_mean_var(
        self,
        action_traj,
        action_t,
        state_traj,
        state_t,
        index=None,
    ):

        action_pred, state_pred = self.predict_x0(
            action_traj=action_traj,
            state_traj=state_traj,
            action_t=action_t,
            state_t=state_t,
        )

        # with torch.enable_grad():
        #     state_pred_ = state_pred.detach().requires_grad_(True)
        #     guidance_loss = torch.nn.functional.mse_loss(state_pred_[:, self.n_past_steps:, 186:189], torch.tensor([-0.5, 0, 0.0], device=self.device), reduction="none")
        #     guidance_loss = -(guidance_loss).sum() #* torch.tensor([1,1,0], device=self.device)
        #     grad = -torch.autograd.grad(
        #         guidance_loss,
        #         state_pred_,
        #     )[0]

        # # print(grad[0,:,186:189].mean(dim=0))

        # mask = state_t < 10
        # pred_noise = self.predict_epsilon_from_x0(state_traj, state_t, state_pred)
        # pred_noise = pred_noise + 2 * grad #* mask[...,None]#* extract(self.ddpm_logvar_clipped, state_t, state_traj.shape).exp().sqrt()
        # state_pred = self.predict_x0_from_epsilon(state_traj, state_t, pred_noise)
        # print(extract(self.ddpm_logvar_clipped, state_t, state_traj.shape).exp().sqrt())
        # print(grad[0,:,186:189].mean(dim=0))

        # print(state_pred[0,:,186:189].mean(dim=0))

        # print('\n\n')


        action_mu, state_mu, action_logvar, state_logvar = self.mean_var_from_x0(
            action_traj=action_traj,
            state_traj=state_traj,
            action_t=action_t,
            state_t=state_t,
            action_pred=action_pred,
            state_pred=state_pred,
        )


        return action_mu, state_mu, action_logvar, state_logvar
    

    # --------------------------------TRAINING ----------------------------------------------
    def p_losses(
        self,
        action_traj,
        state_traj,
    ):
        """
        If predicting epsilon: E_{t, x0, ε} [||ε - ε_θ(√α̅ₜx0 + √(1-α̅ₜ)ε, t)||²

        Args:
            trajectory: (B, horizon, transition_dim)
            cond: dict with keys as step and value as observation
            t: batch of integers
        """
        B = action_traj.shape[0]
        device = self.device

        action_t = torch.randint(
            0, self.denoising_steps, (B, self.horizon), device=device
        ).long()

        state_t = torch.randint(
            0, self.denoising_steps, (B, self.horizon), device=device
        ).long()

        action_noise = torch.randn_like(action_traj, device=device)
        action_noisy = self.q_sample(
            trajectory=action_traj, t=action_t, noise=action_noise
        )

        state_noise = torch.randn_like(state_traj, device=device)
        if self.clean_past_state:
            state_t[:, : self.n_past_steps] = 0
        state_noisy = self.q_sample(trajectory=state_traj, t=state_t, noise=state_noise)

        # Reverse process
        state_pred, action_pred = self.backbone(
            x_input=state_noisy,
            y_input=action_noisy,
            x_timesteps=state_t,
            y_timesteps=action_t,
        )

        # ---------------- action loss ---------------------
        if self.action_pred_epsilon:
            action_loss = torch.nn.functional.mse_loss(action_pred, action_noise, reduction="none")

        else:
            action_loss = torch.nn.functional.mse_loss(
                action_pred, action_traj, reduction="none"
            )


        hip_idxs = [0, 1, 2, 6, 7, 8]
        knee_idxs = [3, 9]
        ankle_idxs = [4, 5, 10, 11]
        
        action_loss_scale = torch.ones(29, device=self.device) * 2 
        action_loss_scale[...,hip_idxs]   = 6 # 4
        action_loss_scale[...,knee_idxs]  = 6 
        action_loss_scale[...,ankle_idxs] = 6 

        # action_loss = action_loss * 2 # times 3
        action_loss = action_loss * self.action_loss_weights.to(self.device)


        action_loss = action_loss.mean()

        # ---------------- state loss ------------------------
        if self.state_pred_epsilon:
            state_loss = torch.nn.functional.mse_loss(state_pred, state_noise, reduction="none")
        else:
            state_loss = torch.nn.functional.mse_loss(state_pred, state_traj, reduction="none")

        state_loss = state_loss * self.state_loss_weights.to(self.device)
        state_loss = state_loss.mean()
        
        # velocity_loss = torch.nn.functional.mse_loss(state_pred[:, 1:, :], state_pred[:, :-1, :], reduction="none")
        # velocity_loss = velocity_loss.mean() * .1

        # state_loss += velocity_loss
                
        action_rate_loss = torch.nn.functional.mse_loss(action_pred[:, 1:, :], action_pred[:, :-1, :], reduction="none")
        action_rate_loss = action_rate_loss.mean() * .1
        action_loss += action_rate_loss

        return (
            action_loss + state_loss,
            action_loss,
            state_loss,
            action_pred,
            state_pred,
        )

    # -------------------------------- HELPER FUNCTIONS -----------------------------------------

    def get_loss_weights(self, loss_schedule):
        """
        Generates loss weights based on the specified loss schedule.
        Args:
            loss_schedule (str): A string specifying the type of loss schedule to use.
                     Supported schedules include:
                     - 'constant-to-{n}': Sets weights to 1 for the first n steps and 0 thereafter.
                     - 'linear': Linearly decreases weights from 1 to 0 over the action steps.
                     - 'cosine': Applies a cosine function to decrease weights from 1 to 0 over the action steps.
                     - 'exponential-{temp}': Applies an exponential decay to the weights, with an optional temperature parameter.
                     - 'sigmoid': Applies a sigmoid function to decrease weights from 1 to 0 over the action steps.
        Returns:
            torch.Tensor: A tensor of shape (1, horizon, 1) containing the computed loss weights.
        """
        weights = torch.ones((1, self.horizon, 1), device=self.device)

        if "constant-to-" in loss_schedule:
            n_pred = loss_schedule.split("constant-to-")
            if len(n_pred) == 2:
                n_pred = int(n_pred[-1])
                weights[:, n_pred + self.n_past_steps :] = 0.0
        elif "linear" == loss_schedule:
            weights[:, self.n_past_steps :] = torch.linspace(
                1, 0, self.n_future_steps, device=self.device
            ).view(1, -1, 1)
        elif "cosine" == loss_schedule:
            weights[:, self.n_past_steps :] = torch.cos(
                torch.linspace(0, torch.pi / 2, self.n_future_steps, device=self.device)
            ).view(1, -1, 1)
        elif "exponential" in loss_schedule:
            temp = loss_schedule.split("exponential-")
            if len(temp) == 2:
                temp = float(temp[-1])
            else:
                temp = 1
            weights[:, self.n_past_steps :] = torch.exp(
                -temp * torch.arange(self.n_future_steps, device=self.device)
            ).view(1, -1, 1)
        elif "sigmoid" == loss_schedule:
            weights[:, self.n_past_steps + 1 :] = 1 - (
                1
                / (
                    1
                    + torch.exp(
                        -0.5 * torch.arange(self.n_future_steps - 1, device=self.device)
                        + 3
                    )
                )
            ).view(1, -1, 1)

        if "exclude-past" in loss_schedule:
            weights[:, : self.n_past_steps] = 0.0
        return weights