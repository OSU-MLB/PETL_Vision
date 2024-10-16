import torch
import torch.nn as nn
from functools import reduce
from operator import mul
import math
from torch.nn.modules.utils import _pair


class VQT(nn.Module):

    def __init__(self, params, depth, patch_size, embed_dim):
        super().__init__()
        self.params = params
        self.vqt_num = params.vqt_num

        patch_size = _pair(patch_size)
        self.query_prompt_embeddings = nn.Parameter(torch.zeros(
            depth, self.vqt_num, embed_dim))
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + embed_dim))
        nn.init.uniform_(self.query_prompt_embeddings.data, -val, val)
        self.prompt_dropout = nn.Dropout(params.vqt_dropout)

        self.combine_layer = nn.Linear(self.vqt_num * depth + 1, 1, bias=False)

        nn.init.ones_(self.combine_layer.weight)

    def retrieve_prompt(self, index, batch_size):
        if index < len(self.query_prompt_embeddings):
            return self.prompt_dropout(self.query_prompt_embeddings[index]).expand(batch_size, -1, -1)
        else:
            return None
