from typing import Union, Optional, Tuple
import logging
import torch
import torch.nn as nn
from diffusion_policy.backbone.positional_embedding import SinusoidalPosEmb
from diffusion_policy.backbone.base_backbone import ConditionalSeqBackbone

logger = logging.getLogger(__name__)

class Transformer(ConditionalSeqBackbone):
    def __init__(self,
            n_layer: int = 6,
            n_head: int = 8,
            n_emb: int = 256,
            p_drop_emb: float = 0.1,
            p_drop_attn: float = 0.1,
            causal_attn: bool=True,
            cond_causal_attn: bool=True,
            n_cond_layers: int = 0,
            **kwargs

        ) -> None:
        super().__init__(
            **kwargs
        )

        # compute number of tokens for main trunk and condition encoder
        T = self.horizon
        T_cond = self.n_cond_steps
    
        # input embedding stem
        self.input_emb = nn.Linear(self.x_input_dim, n_emb//2)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # cond encoder
        self.time_emb = SinusoidalPosEmb(n_emb//2)        
        self.cond_obs_emb = nn.Linear(self.cond_dim, n_emb)

        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None
        encoder_only = False
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))

        if self.cond_dim > 0:
            if n_cond_layers > 0:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4*n_emb,
                    dropout=p_drop_attn,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer=encoder_layer,
                    num_layers=n_cond_layers
                )
            else:
                self.encoder = nn.Sequential(
                    nn.Linear(n_emb, 4 * n_emb),
                    nn.Mish(),
                    nn.Linear(4 * n_emb, n_emb)
                )

            # decoder
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4*n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True # important for stability
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=n_layer
            )

        else:
            # decoder only (weird naming due to nn transformer)
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


        # causal attention mask
        if cond_causal_attn:
            S = T_cond
            t, s = torch.meshgrid(
                torch.arange(T),
                torch.arange(S),
                indexing='ij'
            )
            mask = t >= (s) # add one dimension since time is the first token in cond
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer('memory_mask', mask)
        else:
            self.memory_mask = None
        
        if causal_attn:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf and others (including diag) should be 0.
            sz = T
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask)

            sz = T_cond
            encoder_mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            encoder_mask = encoder_mask.float().masked_fill(encoder_mask == 0, float('-inf')).masked_fill(encoder_mask == 1, float(0.0))
            self.register_buffer("encoder_mask", encoder_mask)
        else:
            self.mask = None

        # decoder head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, self.x_output_dim)
            
        # constants
        self.T = T
        self.T_cond = T_cond
        self.encoder_only = encoder_only

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
            if module.cond_obs_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
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
        if self.cond_pos_emb is not None:
            no_decay.add("cond_pos_emb")

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
        sample: torch.Tensor, 
        timesteps: torch.Tensor, 
        cond: Optional[torch.Tensor]=None, **kwargs):
        """
        sample: (B,T,input_dim)
        timestep: (B,T)
        cond: (B,T_cond,cond_dim)
        output: (B,T,input_dim)
        """
        B = sample.shape[0]
        # time embedding
        time_emb = self.time_emb(timesteps)
        # (B,T,n_emb)

        # input embedding
        input_emb = self.input_emb(sample)

        if cond is not None:
            if isinstance(cond, dict):
                raise NotImplementedError
            cond = cond.reshape(B, -1, self.cond_dim)
            # obs as condition
            cond_obs_emb = self.cond_obs_emb(cond)
            # (B,To,n_emb)
            # cond_embeddings = torch.cat([time_emb, cond_obs_emb], dim=1)
            cond_embeddings = cond_obs_emb

            T_cond = cond_embeddings.shape[1]
            position_embeddings = self.cond_pos_emb[
                :, :T_cond, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(cond_embeddings + position_embeddings)
            x = self.encoder(
                src = x,
            )
            memory = x # (B,T_cond,n_emb)
        
        # decoder
        token_embeddings = torch.cat([input_emb,  time_emb], dim=-1)
        t = token_embeddings.shape[1]
        position_embeddings = self.pos_emb[
            :, :t, :
        ]  # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        # (B,T,n_emb)

        if cond is not None:
            x = self.decoder(
                tgt=x,
                memory=memory,
                tgt_mask=self.mask,
                memory_mask=self.memory_mask
            ) # (B,T,n_emb)
        else:
            x = self.decoder(
                src=x,
                mask=self.mask,
            ) # (B,T,n_emb)  
        
        # head
        x = self.ln_f(x)

        x = self.head(x) # (B,T,n_out)
        
        return x