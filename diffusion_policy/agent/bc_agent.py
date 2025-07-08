from __future__ import annotations
from typing import Tuple, Union, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time 

from diffusion_policy.modules import *
# from diffusion_policy.utils.sdf import * 

from diffusion_policy.agent.base_agent import BaseAgent
from diffusion_policy.utils.module_dict import ModuleDict
from diffusion_policy.utils.traj_utils import quat_from_euler_xyz, get_euler_xyz, quat_mul, quat_rotate, box_minus, quat_rotate_inverse, box_plus, quat_conjugate

try:
    from diffusion_policy.utils.live_visualizer_pyg import LivePlotVisualizerPygame
except:
    print("Plot Visualizer NOT INSTALLED")

class BCAgent(BaseAgent):
    def __init__(
        self,
        actor: BaseActor,
        **kwargs,
    ):
        super().__init__(
            actor=actor,
        )

    def test_mode(self):
        pass

    def train_mode(self):
        pass
    
    @torch.no_grad()
    def act(self, obs_dict):

        obs = self.normalizer['obs'].normalize(obs_dict['obs'])

        output = self.actor.act(obs)

        if isinstance(self.actor, JointDiffusionActor):
            naction_pred, nstate_pred = output
        
            action_pred = self.normalizer['action'].unnormalize(naction_pred)
            state_pred = self.normalizer['obs'].unnormalize(nstate_pred)
            # global_body_pos, root_rot_global = self.dataset_class.state_unnormalize(state_pred.clone(), obs_dict['global_root'], return_rot=True)
            # body_pos = self.dataset_class.state_unnormalize(state_pred.clone(), return_rot=False)
            
            return action_pred, None, None #global_body_pos, body_pos #, #state_pred #body_pos
        
        elif isinstance(self.actor, DiffusionActor):
            naction_pred = output
            action_pred = self.normalizer['action'].unnormalize(naction_pred)

            return action_pred
    
        else:
            raise NotImplementedError

    def get_optimizer(
            self, **kwargs
        ) -> Dict[str, torch.optim.Optimizer]:

        optimizer = dict()
        
        if isinstance(self.actor, DiffusionActor) or isinstance(self.actor, JointDiffusionActor):
            optimizer.update(
                {'diffusion': self.actor.get_optimizer(**kwargs)}
            )


        return ModuleDict(optimizer)

    def compute_loss(self, batch, local_epoch_idx):
        B = batch['obs'].shape[0]

        nbatch = self.normalizer.normalize(batch)

        nobs = nbatch['obs']
        naction = nbatch['action']

        loss_dict = {}

        B = nobs.shape[0]
        device = nobs.device

        if isinstance(self.actor, JointDiffusionActor):
            loss, action_loss, state_loss ,action_pred, state_pred = self.actor.p_losses(action_traj=naction, state_traj=nobs)
            loss_dict.update(
                {
                    'diffusion': loss.mean(), 

                    "grad_analysis": {
                        "action_pred": action_pred,
                        "state_pred": state_pred,
                    },

                    "log":{
                        "action_loss":np.sqrt(action_loss.detach().cpu()),
                        "state_loss":np.sqrt(state_loss.detach().cpu()),
                    }
                }
            )
            yield loss_dict

        elif isinstance(self.actor, DiffusionActor):
            loss = self.actor.p_losses(trajectory=naction, cond=nobs)
            loss_dict.update(
                {'diffusion': loss.mean(), }
            )
            yield loss_dict



