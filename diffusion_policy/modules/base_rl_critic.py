import torch
import torch.nn as nn
from diffusion_policy.backbone.base_backbone import BaseBackbone
from diffusion_policy.utils.module_attr_mixin import ModuleAttrMixin


class BaseRLCritic(ModuleAttrMixin):
    def __init__(self, 
                 backbone: BaseBackbone,
                 **kwargs):
        if kwargs:
            print("BaseCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(BaseRLCritic, self).__init__()

        self.backbone = backbone

        print(f"Critic MLP: {self.backbone}")

    def evaluate(self, critic_observations, **kwargs):
        value = self.backbone(critic_observations)
        return value

    def reset(self, dones=None):
        pass

    def get_optim_groups(self, **kwargs):
        return self.backbone.get_optim_groups(**kwargs)

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
