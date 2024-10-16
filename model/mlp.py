from torch import nn as nn
from timm.layers.helpers import to_2tuple
from functools import partial
from model.ssf import init_ssf_scale_shift, ssf_ada


class MlpPETL(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
            params=None,
            fact=None
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

        ############# Added module #############
        self.params = params
        if params.ssf:
            self.ssf_scale_hidden, self.ssf_shift_hidden = init_ssf_scale_shift(hidden_features)
            self.ssf_scale_out, self.ssf_shift_out = init_ssf_scale_shift(out_features)
        self.fact = fact
        ############# Added module end #############

    def forward(self, x, block_idx):
        ############# Added module #############
        B, N, C = x.shape
        h = self.fc1(x)
        if self.params.ssf:
            h = ssf_ada(h, self.ssf_scale_hidden, self.ssf_shift_hidden)
        if self.fact:
            h += self.fact(x, block_idx, 'mlp_1', B, N, C)
        ############# Added module end #############

        x = self.act(h)
        x = self.drop1(x)
        x = self.norm(x)

        ############# Added module #############
        h = self.fc2(x)
        if self.params.ssf:
            h = ssf_ada(h, self.ssf_scale_out, self.ssf_shift_out)
        if self.fact:
            h += self.fact(x, block_idx, 'mlp_2', B, N, C)
        ############# Added module end #############

        x = self.drop2(h)
        return x