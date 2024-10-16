import torch.nn as nn
from timm.layers import DropPath
from timm.models.vision_transformer import LayerScale
from timm.layers.trace_utils import _assert
from model.adapter import Adapter
from model.convpass import ConvPass
from model.repadapter import RepAdapter
from model.ssf import init_ssf_scale_shift, ssf_ada
import torch
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List
from model.mlp import MlpPETL
from model.attention import AttentionPETL

MODULE_REGISTRY = {
    'adapter': Adapter,
    'convpass': ConvPass,
    'repadapter': RepAdapter

}

class BlockPETL(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = MlpPETL,
            params=None,
            fact=None
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AttentionPETL(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            ############# Added module #############
            params=params,
            fact=fact
            ############# Added module end #############
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
            ############# Added module #############
            params=params,
            fact=fact
            ############# Added module end #############
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        ############# Added module #############
        self.params = params
        if params.ft_attn_module:
            self.ft_attn_module = MODULE_REGISTRY[params.ft_attn_module](dim=dim, params=params)
        if params.ft_mlp_module:
            self.ft_mlp_module = MODULE_REGISTRY[params.ft_mlp_module](dim=dim, params=params)

        if self.params.ssf:
            self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(dim)
            self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(dim)

        if self.params.difffit:
            self.difffit_gamma1 = nn.Parameter(torch.ones(dim))
            self.difffit_gamma2 = nn.Parameter(torch.ones(dim))

        self.fact = fact
        ############# Added module end #############

    def forward(self, x: torch.Tensor, idx) -> torch.Tensor:
        # MHSA path
        residual_attn = x

        if self.params.ssf:
            x_norm1 = ssf_ada(self.norm1(x), self.ssf_scale_1, self.ssf_shift_1)
        else:
            x_norm1 = self.norm1(x)
        # ft attention module
        if self.params.ft_attn_module:
            if self.params.ft_attn_mode == 'parallel':
                x_original = self.drop_path1(self.ls1(self.attn(x_norm1, idx)))
                if self.params.ft_attn_ln == 'before':
                    x_ft_attn = self.drop_path1(self.ls1(self.ft_attn_module(x))) + x_original
                elif self.params.ft_attn_ln == 'after':
                    x_ft_attn = self.drop_path1(self.ls1(self.ft_attn_module(x_norm1))) + x_original
                else:
                    raise NotImplementedError
                del x_original
            elif self.params.ft_attn_mode == 'sequential_after':
                x_original = self.drop_path1(self.ls1(self.attn(x_norm1, idx)))
                x_ft_attn = self.drop_path1(self.ls1(self.ft_attn_module(x_original, add_residual=True)))
                del x_original
            elif self.params.ft_attn_mode == 'sequential_before':
                x_ft_attn = self.drop_path1(self.ls1(self.attn(self.ft_attn_module(x_norm1), idx)))
            else:
                raise NotImplementedError

            torch.cuda.empty_cache()
        else:
            # no tuning
            x_ft_attn = self.drop_path1(self.ls1(self.attn(x_norm1, idx)))

        # residual for attention module
        if self.params.difffit:
            x = self.difffit_gamma1 * x_ft_attn + residual_attn
        else:
            x = x_ft_attn + residual_attn

        del x_norm1, x_ft_attn, residual_attn
        torch.cuda.empty_cache()

        # MLP path
        residual_mlp = x

        if self.params.ssf:
            x_norm2 = ssf_ada(self.norm2(x), self.ssf_scale_2, self.ssf_shift_2)
        else:
            x_norm2 = self.norm2(x)

        # ft mlp module
        if self.params.ft_mlp_module:
            if self.params.ft_mlp_mode == 'parallel':
                x_original = self.drop_path2(self.ls2(self.mlp(x_norm2, idx)))
                if self.params.ft_mlp_ln == 'before':
                    x_ft_mlp = self.drop_path2(self.ls2(self.ft_mlp_module(x))) + x_original
                elif self.params.ft_mlp_ln == 'after':
                    x_ft_mlp = self.drop_path2(self.ls2(self.ft_mlp_module(x_norm2))) + x_original
                else:
                    raise NotImplementedError
                del x_original
            elif self.params.ft_mlp_mode == 'sequential_after':
                x_original = self.drop_path2(self.ls2(self.mlp(x_norm2, idx)))
                x_ft_mlp = self.drop_path2(self.ls2(self.ft_mlp_module(x_original, add_residual=True)))
                del x_original
            elif self.params.ft_attn_mode == 'sequential_before':
                x_ft_mlp = self.drop_path2(self.ls2(self.mlp(self.ft_mlp_module(x_norm2), idx)))
            else:
                raise NotImplementedError

            torch.cuda.empty_cache()
        else:
            # no tuning
            x_ft_mlp = self.drop_path2(self.ls2(self.mlp(x_norm2, idx)))

        # residual for mlp module
        if self.params.difffit:
            x = self.difffit_gamma2 * x_ft_mlp + residual_mlp
        else:
            x = x_ft_mlp + residual_mlp
        del x_norm2, x_ft_mlp, residual_mlp
        torch.cuda.empty_cache()
        # Original forward
        # x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        # x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x



