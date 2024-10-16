from torch import nn
from model.utils import init_weight


class RepAdapter(nn.Module):
    """ Pytorch Implemention of RepAdapter for 1d tensor"""

    def __init__(
            self,
            dim,
            params,
    ):
        super().__init__()
        self.conv_A = nn.Conv1d(dim, params.repadapter_bottleneck, 1, groups=1, bias=True)
        self.conv_B = nn.Conv1d(params.repadapter_bottleneck, dim, 1, groups=params.repadapter_group, bias=True)
        self.dropout = nn.Dropout(0.1)
        self.groups = params.repadapter_group
        self.scale = params.repadapter_scaler

        init_weight(self.conv_A, self.conv_B, params.repadapter_init)
        self.params = params

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv_B(self.dropout(self.conv_A(x))) * self.scale * self.params.merge_factor + x
        x = x.transpose(1, 2).contiguous()
        return x
