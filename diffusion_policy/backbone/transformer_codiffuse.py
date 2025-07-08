from typing import Union, Optional, Tuple
import logging
import torch
import torch.nn as nn
from diffusion_policy.backbone.positional_embedding import SinusoidalPosEmb
from diffusion_policy.backbone.base_backbone import JointSeqBackbone

logger = logging.getLogger(__name__)

class Transformer(JointSeqBackbone):
    def __init__(self,
            n_layer: int = 6,
            n_head: int = 8,
            n_emb: int = 256,
            p_drop_emb: float = 0.1,
            p_drop_attn: float = 0.1,
            causal_attn: bool=True,
            x_to_x_attn: str='full',
            x_to_y_attn: str='full',
            y_to_x_attn: str='full',
            y_to_y_attn: str='full',
            **kwargs
        ) -> None:
        super().__init__(
            **kwargs 
        )

        # compute number of tokens for main trunk and condition encoder
        self.n_emb = n_emb
    
        self.x_emb = nn.Linear(self.x_input_dim, n_emb//4 * 3)
        self.y_emb = nn.Linear(self.y_input_dim, n_emb//4 * 3)

        self.pos_emb = nn.Parameter(torch.zeros(1, self.horizon*2, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # cond encoder
        self.time_emb = SinusoidalPosEmb(n_emb//4)        
        decoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4*n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
        self.decoder = nn.TransformerEncoder(
            encoder_layer=decoder_layer,
            num_layers=n_layer
        )

        def get_causal_mask(pattern, T1, T2):
            if pattern == 'full':
                return torch.ones((T1, T2), dtype=torch.bool)
            elif pattern == 'causal':
                return (torch.triu(torch.ones(T1, T2)) == 1).transpose(0, 1)
            elif pattern == 'causal-1':
                mask = (torch.triu(torch.ones(T1, T2)) == 1).transpose(0, 1)
                mask = mask.fill_diagonal_(False)
                mask[:, 4:] = False
                return mask    
            elif pattern == 'no_attn':
                return torch.zeros((T1, T2), dtype=torch.bool)
            else:  
                raise NotImplementedError

        if causal_attn:
            sz = self.x_horizon + self.y_horizon
            mask = torch.ones((sz, sz), dtype=torch.bool)
            mask[::2,::2] = get_causal_mask(x_to_x_attn, self.x_horizon, self.x_horizon)
            mask[1::2,1::2] = get_causal_mask(y_to_y_attn, self.y_horizon, self.y_horizon)
            mask[::2,1::2] = get_causal_mask(x_to_y_attn, self.x_horizon, self.y_horizon)
            mask[1::2,::2] = get_causal_mask(y_to_x_attn, self.y_horizon, self.x_horizon)

            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

            self.register_buffer("mask", mask)

        else:
            self.mask = None
            self.encoder_mask = None

        # decoder head
        self.x_ln_f = nn.LayerNorm(n_emb)
        self.x_head = nn.Linear(n_emb, self.x_output_dim)
        self.y_ln_f = nn.LayerNorm(n_emb)
        self.y_head = nn.Linear(n_emb, self.y_output_dim)

        # init
        self.apply(self._init_weights)
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def _init_weights(self, module):
        ignore_types = (nn.Dropout, 
            SinusoidalPosEmb, 
            nn.TransformerEncoderLayer, 
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, Transformer):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))
    
    def get_optim_groups(self, weight_decay: float=1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        no_decay.add("_dummy_variable")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups


    def configure_optimizers(self, 
            learning_rate: float=1e-4, 
            weight_decay: float=1e-3,
            betas: Tuple[float, float]=(0.9,0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def forward(self, 
                x_input,
                y_input,
                x_timesteps,
                y_timesteps,):
        """
        sample: (B,T,input_dim)
        timestep: (B,T)
        cond: (B,T_cond,cond_dim)
        output: (B,T,input_dim)
        """
        B, H = x_input.shape[:2]
        # time embedding
        x_timesteps = self.time_emb(x_timesteps)
        # (B,T,n_emb)
        y_timesteps = self.time_emb(y_timesteps)

        # input embedding
        x_input = self.x_emb(x_input)
        y_input = self.y_emb(y_input)
        
        # decoder
        x_embeddings = torch.cat([x_input,  x_timesteps], dim=-1)
        y_embeddings = torch.cat([y_input,  y_timesteps], dim=-1)
        token_embeddings = torch.zeros((B,H*2, self.n_emb), device=self.device)
        token_embeddings[:,::2] = x_embeddings
        token_embeddings[:,1::2] = y_embeddings

        t = token_embeddings.shape[1]
        position_embeddings = self.pos_emb[
            :, :t, :
        ]  # each position maps to a (learnable) vector
        token_embeddings = self.drop(token_embeddings + position_embeddings)
        # (B,T,n_emb)

        token_embeddings = self.decoder(
            src=token_embeddings,
            mask=self.mask,
        )
        
        x_output = token_embeddings[:,::2]
        y_output = token_embeddings[:,1::2]

        # TODO: get value from embedding value here (at k=0, k=100, k=500, k=1000)
        x_output = self.x_ln_f(x_output)
        x_output = self.x_head(x_output) # (B,T,n_out)
        
        y_output = self.y_ln_f(y_output)
        y_output = self.y_head(y_output)

        return x_output, y_output