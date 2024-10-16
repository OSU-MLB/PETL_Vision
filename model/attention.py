from torch.jit import Final
import torch.nn as nn
from timm.layers import use_fused_attn
import torch
import torch.nn.functional as F
from model.ssf import init_ssf_scale_shift, ssf_ada
from model.lora import LoRA

class AttentionPETL(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            params=None,
            fact=None
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        ############# Added module #############
        self.params = params
        if self.params.ssf:
            self.ssf_scale_qkv, self.ssf_shift_qkv = init_ssf_scale_shift(dim * 3)
            self.ssf_scale_linear, self.ssf_shift_linear = init_ssf_scale_shift(dim)
        if self.params.lora_bottleneck > 0:
            self.lora = LoRA(dim, num_heads, params)
        self.fact = fact
        ############# Added module end #############
    def forward(self, x: torch.Tensor, block_idx) -> torch.Tensor:
        B, N, C = x.shape
        ############# Added module #############
        if self.params.ssf:
            qkv = (ssf_ada(self.qkv(x), self.ssf_scale_qkv, self.ssf_shift_qkv))
        else:
            qkv = self.qkv(x)

        if self.fact:
            qkv += self.fact(x, block_idx, 'attn_qkv', B, N, C)
        ############# Added module end #############

        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        ############# Added module #############
        if self.params.lora_bottleneck > 0:
            q, k, v = self.lora(x, q, k, v, B, N, C)

        if self.params.vqt_num > 0:
            k = k[:, :, self.params.vqt_num:]
            v = v[:, :, self.params.vqt_num:]
        ############# Added module end #############

        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        proj = self.proj(x)
        ############# Added module #############
        if self.params.ssf:
            proj = ssf_ada(proj, self.ssf_scale_linear, self.ssf_shift_linear)
        if self.fact:
            proj += self.fact(x, block_idx, 'attn_proj', B, N, C)
        ############# Added module end #############
        x = self.proj_drop(proj)
        return x
