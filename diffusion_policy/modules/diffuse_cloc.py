from __future__ import annotations
from typing import TYPE_CHECKING

import torch
from torch.distributions.normal import Normal
import numpy as np
from diffusion_policy.modules.joint_diffusion import JointDiffusionActor

count = 0
class DiffuseCLoC(JointDiffusionActor):
    """
    This module implements rolling scheme and state emphasis used in DiffuseCLoC
    """

    def __init__(self,  
                 state_emphasis='same', randomize_noise_schedule=True, **kwargs):
        super().__init__(**kwargs)

        self.state_emphasis = state_emphasis
        self.randomize_noise_schedule = randomize_noise_schedule
        self.get_emphasis_projection()

        self.action_schedule = 'from_xT_decreasing'
        self.state_schedule = 'from_xT_step'

    def act(
        self,
        nobs,
        **kwargs,
    ):
        # Loop
        B = nobs.shape[0]
        nobs = nobs[:, :self.n_past_steps, :]

        action_traj = torch.randn((B, self.horizon, self.action_dim), device=self.device)
        state_traj = torch.randn((B, self.horizon, self.obs_dim), device=self.device)

        # ------------- implementing rolling scheme ------------------

        action_chain = []
        state_chain = []

        if self.randomize_noise_schedule:
            if not hasattr(self, 'action_rolling_traj'):
                action_t_all = self.generate_denoising_matrix("full_decreasing")
            else:
                action_t_all = self.generate_denoising_matrix(self.action_schedule)
                # impaint those that are already diffused
                action_traj[:,:-1] = self.action_rolling_traj
        else:
            action_t_all = self.generate_denoising_matrix("full")

        if self.randomize_noise_schedule:
            if not hasattr(self, 'state_rolling_traj'):
                state_t_all = self.generate_denoising_matrix("full", is_state=True)
            else:
                state_t_all = self.generate_denoising_matrix(self.state_schedule, is_state=True)
                state_traj[:,:-1] = self.state_rolling_traj
        else:
            state_t_all = self.generate_denoising_matrix("full", is_state=True)

        prev_action_t = action_t_all[0].clone() + 1
        prev_state_t = state_t_all[0].clone() + 1
        nobs = nobs @ self.emphasis_mat

        for i in range(max(len(action_t_all), len(state_t_all))):
            action_t = action_t_all[min(i, len(action_t_all)-1)]
            if i > len(state_t_all)-1:
                state_t = prev_state_t.clone()
                state_t[torch.nonzero(state_t==0)[-1,-1]+1:] -= 1 
            else:
                state_t = state_t_all[i]

            if self.randomize_noise_schedule:
                action_chain.append((action_traj.clone(), prev_action_t))
                state_chain.append((state_traj.clone(), prev_state_t))
            prev_action_t = action_t.clone()
            prev_state_t = state_t.clone()

           
            action_traj, state_traj = self.diffuse_step(
                nobs=nobs,
                action_traj=action_traj,
                state_traj=state_traj,
                action_t=action_t,
                state_t=state_t,
                i=i,
                **kwargs,
            )

        if self.randomize_noise_schedule:
            action_chain.append((action_traj.clone(), action_t.clone()))
            state_chain.append((state_traj.clone(), state_t.clone()))

            action_t_all = self.generate_denoising_matrix(self.action_schedule)
            self.action_rolling_traj = self.get_rolling_traj(action_chain, action_t_all)
            state_t_all = self.generate_denoising_matrix(self.state_schedule, is_state=True)
            self.state_rolling_traj = self.get_rolling_traj(state_chain, state_t_all)
            
        # -------------------------------------------------------------------------

        state_traj = state_traj @ self.emphasis_mat_inv

        return action_traj, state_traj

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
        state_traj = state_traj @ self.emphasis_mat
        return super().p_losses(action_traj, state_traj)


# -------------------------------- HELPER FUNCTIONS -----------------------------------------

    def get_rolling_traj(self, chain, t_all):
        traj = torch.stack([c[0] for c in chain], dim=1)[:,:,1:,:] # shape = (B,K,T,A_dim) K = denoising_length
        idx = torch.stack([c[1] for c in chain])[:,1:] # shape = (K,T)
        needed_idx = t_all[0,:-1] + 1
        mask = idx == needed_idx
        mask = torch.logical_xor(mask, (mask.roll(-1,0) == mask) & (mask))
        traj = traj[:,mask]
        return traj
    
    def generate_denoising_matrix(self, schedule, is_state=False, **kwargs):
        
        def decreasing_matrix(start_value, is_state, step_size=1, all_clear=False):
            if step_size == 1:
                if is_state:
                    end = start_value + self.n_future_steps
                else:
                    end = start_value + self.n_future_steps + 1
            else:
                end = self.denoising_steps + step_size
                
            first_row = torch.arange(start_value, end, step_size)
            # if all_clear:
            #     # first_row = torch.arange(end-1, start_value-1, -1)
            #     m = end
            m = start_value + 1
            decrement_column = -torch.arange(m).view(m, 1)
            
            # Broadcast the first row and decrement_column to generate the entire matrix
            action_t_all = first_row + decrement_column
            action_t_all = torch.clip(action_t_all, 0, self.denoising_steps - 1)
            n = self.horizon - action_t_all.shape[1]
            return action_t_all, n
        if schedule == "full":
            action_t_all = torch.flip(torch.arange(self.denoising_steps), dims=(0,))
            action_t_all = action_t_all.unsqueeze(1).repeat(1, self.horizon)
        elif schedule == "full_decreasing":
            action_t_all, n = decreasing_matrix(self.denoising_steps - 1, is_state=is_state)
            action_t_all = torch.cat([action_t_all[:,0:1].repeat(1,n), action_t_all], dim=-1)
        elif "from_xT_decreasing" in schedule:
            if is_state:
                start = max(self.denoising_steps - self.n_future_steps, 0)
            else:
                start = max(self.denoising_steps - 8 - 1, 0)
            action_t_all, n = decreasing_matrix(start, is_state=is_state, **kwargs)
            action_t_all = torch.cat([action_t_all[:,0:1].repeat(1,n), action_t_all], dim=-1)
        elif "from_xT_step" in schedule:
            # if is_state:
            #     start = max(self.denoising_steps - self.n_future_steps, 0)
            # else:
            #     start = max(self.denoising_steps - 8 - 1, 0)
            # action_t_all, n = decreasing_matrix(start, is_state=is_state, **kwargs)
            # action_t_all = torch.cat([action_t_all[:,0:1].repeat(1,n), action_t_all], dim=-1)
            start = 14
            action_t_all, n = decreasing_matrix(start, is_state=is_state, step_size=10, **kwargs)
            action_t_all = torch.cat([action_t_all[:,0:1].repeat(1,n), action_t_all], dim=-1)
        return action_t_all

    def get_emphasis_projection(self):
        state_dim = self.backbone.x_output_dim
        if self.state_emphasis == "rand":
            emphasis_mat = torch.randn((state_dim,state_dim),device=self.device) / np.sqrt(state_dim)
        elif self.state_emphasis == "emph_global":
            emphasis_mat = torch.eye(state_dim,device=self.device)
            emphasis_mat[torch.arange(144,150),torch.arange(144,150)] = 3
            emphasis_mat[torch.arange(162,165),torch.arange(162,165)] = 3
        elif self.state_emphasis == "random_emph":
            emphasis_mat_A = torch.randn((state_dim,state_dim),device=self.device)
            emphasis_mat_B = torch.eye(state_dim,device=self.device)
            emphasis_mat_B[torch.arange(144,150),torch.arange(144,150)] = 5
            emphasis_mat_B[torch.arange(162,165),torch.arange(162,165)] = 5
            emphasis_mat = (emphasis_mat_B @ emphasis_mat_A) / np.sqrt(state_dim - 9 + 9 * 5**2)
        elif self.state_emphasis == "random_emph_half":
            emphasis_mat_A = torch.randn((state_dim,state_dim),device=self.device)
            emphasis_mat_B = torch.eye(state_dim,device=self.device)
            mask = (torch.rand((1, state_dim),device=self.device) < 0.5).repeat(state_dim, 1)
            emphasis_mat_B[torch.arange(144,150),torch.arange(144,150)] = 5
            emphasis_mat_B[torch.arange(162,165),torch.arange(162,165)] = 5
            emphasis_mat_B = (emphasis_mat_B @ emphasis_mat_A) / np.sqrt(state_dim - 9 + 9 * 5**2)
            emphasis_mat_B_nominal = (emphasis_mat_A) / np.sqrt(state_dim)
        elif self.state_emphasis == "random_emph_double":
            state_dim = state_dim // 2
            emphasis_mat_A = torch.randn((state_dim,state_dim),device=self.device)
            emphasis_mat_B = torch.eye(state_dim,device=self.device)
            emphasis_mat_B_nominal = torch.eye(state_dim,device=self.device)
            start_dim = state_dim - 12
            emphasis_mat_B[torch.arange(start_dim,start_dim+6),torch.arange(start_dim,start_dim+6)] = 4
            emphasis_mat_B[torch.arange(start_dim+6,start_dim+12),torch.arange(start_dim+6,start_dim+12)] = 4
            emphasis_mat = emphasis_mat_B @ emphasis_mat_A / np.sqrt(state_dim - 9 / 2 + 9 * 4**2 / 2)
            emphasis_mat = torch.cat((emphasis_mat, emphasis_mat_B_nominal), dim=1)

        elif self.state_emphasis == "random_emph_symm":
            state_dim = state_dim // 2
            emphasis_mat_A = torch.zeros((state_dim,state_dim),device=self.device)
            from diffusion_policy.dataset.g1_offline_dataset import G1_Dataset
            obs_r, _ = G1_Dataset.get_reflection_ops()
            mask = obs_r.sum(dim=0) < 0
            emphasis_mat_A[mask, :state_dim//2] = emphasis_mat_A[mask, :state_dim//2].normal_()
            emphasis_mat_A[~mask, state_dim//2:] = emphasis_mat_A[~mask, state_dim//2:].normal_()
            emphasis_mat_A = (emphasis_mat_A + obs_r.abs().to(emphasis_mat_A.dtype) @ emphasis_mat_A) / 2

            emphasis_mat_A[mask, :state_dim//2] /= np.sqrt((mask).sum())
            emphasis_mat_A[~mask, state_dim//2:] /= np.sqrt((~mask).sum())
            
            emphasis_mat_B = torch.eye(state_dim,device=self.device)
            emphasis_mat_B_nominal = torch.eye(state_dim,device=self.device)
            start_dim = state_dim - 12

            # emphasis_mat_B[torch.arange(90, 180),torch.arange(90, 180)] = 0.5
            # emphasis_mat_B[torch.arange(186, 189),torch.arange(186, 189)] = 0.5


            emphasis_mat_B[torch.arange(start_dim,start_dim+6),torch.arange(start_dim,start_dim+6)] = 4
            emphasis_mat_B[torch.arange(start_dim+9,start_dim+12),torch.arange(start_dim+9,start_dim+12)] = 4
            emphasis_mat = emphasis_mat_B @ emphasis_mat_A
            emphasis_mat = torch.cat((emphasis_mat, emphasis_mat_B_nominal), dim=1)

        elif self.state_emphasis == "same":
            emphasis_mat = torch.eye(state_dim,device=self.device)

        elif self.state_emphasis == "copy":
            state_dim = state_dim - 9*10
            mat = torch.eye(state_dim,device=self.device)
            emphasis_mat = torch.zeros((state_dim, state_dim + 9*10), device=self.device)
            emphasis_mat[:144, :144] = mat[:144, :144]
            for i in range(10):
                emphasis_mat[144:150, 144+i*6:144+(i+1)*6] = mat[144:150, 144:150]
            emphasis_mat[150:162, 210:222] = mat[150:162, 150:162]
            for i in range(10):
                emphasis_mat[162:165, 222+i*3:222+(i+1)*3] = mat[162:165, 162:165]

        self.register_buffer('emphasis_mat', emphasis_mat)
        self.register_buffer('emphasis_mat_inv', torch.linalg.pinv(emphasis_mat))
