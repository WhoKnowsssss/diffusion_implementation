import torch
import torch.nn as nn
from torch.distributions import Normal
from diffusion_policy.backbone.base_backbone import BaseBackbone
from diffusion_policy.modules.base_actor import BaseActor

class BaseRLActor(BaseActor):
    def __init__(self, 
                 backbone: BaseBackbone,
                 init_noise_std=1.0,
                 fixed_std=False,
                 **kwargs):
        if kwargs:
            print("BaseActor.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super().__init__(backbone, **kwargs)

        # Action noise
        self.fixed_std = fixed_std
        std = init_noise_std * torch.ones(self.action_dim)
        self.std = torch.tensor(std) if fixed_std else nn.Parameter(std)
        self.distribution = None
        self.is_recurrent = False

        # Disable args validation for speedup
        Normal.set_default_validate_args = False

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    def update_distribution(self, observations):
        mean = self.backbone(observations)
        std = self.std.to(mean.device)
        self.distribution = Normal(mean, mean * 0. + std)

    def act(self, observations, deterministic=False, **kwargs):
        if deterministic:
            return self.backbone(observations)
        self.update_distribution(observations)
        action = self.distribution.sample()
        log_prob = self.distribution.log_prob(action).sum(dim=-1)
        return action, log_prob

    def get_actions_log_prob(self, obs, actions):
        self.update_distribution(obs)
        return self.distribution.log_prob(actions).sum(dim=-1)

    def reset(self, dones=None):
        pass
