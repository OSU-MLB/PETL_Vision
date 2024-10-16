import torch.nn as nn
import torch

TOTAL_WEIGHT_MATRIX = 12


class FacT(nn.Module):
    def __init__(self, dim, depth, params):
        super().__init__()

        self.per_block_para = nn.ModuleList()
        self.fact_dim = params.fact_dim
        self.params = params
        self.depth = depth
        if params.fact_type == 'tt':
            self.FacTu = nn.Linear(dim, self.fact_dim, bias=False)
            self.FacTv = nn.Linear(self.fact_dim, dim, bias=False)
            nn.init.zeros_(self.FacTv.weight)

            for i in range(depth):
                self.per_block_para.append(nn.ModuleDict({
                    'attn': nn.ModuleList([
                        nn.Linear(self.fact_dim, self.fact_dim, bias=False),
                        nn.Linear(self.fact_dim, self.fact_dim, bias=False),
                        nn.Linear(self.fact_dim, self.fact_dim, bias=False),
                        nn.Linear(self.fact_dim, self.fact_dim, bias=False),
                        nn.Dropout(0.1)
                    ]),
                    'mlp': nn.ModuleList([
                        nn.Linear(self.fact_dim, self.fact_dim * 4, bias=False),
                        nn.Linear(self.fact_dim * 4, self.fact_dim, bias=False),
                        nn.Dropout(0.1)
                    ])
                }))

        elif params.fact_type == 'tk':
            self.FacTu = nn.Linear(dim, self.fact_dim, bias=False)
            self.FacTv = nn.Linear(self.fact_dim, dim, bias=False)
            self.FacTp = nn.Parameter(torch.zeros([self.fact_dim, TOTAL_WEIGHT_MATRIX * depth], dtype=torch.float),
                                      requires_grad=True)
            self.FacTc = nn.Parameter(
                torch.zeros([self.fact_dim, self.fact_dim, self.fact_dim], dtype=torch.float), requires_grad=True)

            nn.init.zeros_(self.FacTv.weight)
            nn.init.xavier_uniform_(self.FacTc)
            nn.init.xavier_uniform_(self.FacTp)
            for i in range(depth):
                self.per_block_para.append(nn.ModuleDict({
                    'attn': nn.Dropout(0.1),
                    'mlp': nn.Dropout(0.1)

                }))

        else:
            raise NotImplementedError

    def forward(self, x, block_idx, mode, B, N, C):
        
        if self.params.fact_type == 'tt':
            if mode == 'attn_qkv':
                q_FacTs, k_FacTs, v_FacTs, _, dp = self.per_block_para[block_idx]['attn']
                q = self.FacTv(dp(q_FacTs(self.FacTu(x))))
                k = self.FacTv(dp(k_FacTs(self.FacTu(x))))
                v = self.FacTv(dp(v_FacTs(self.FacTu(x))))
                return torch.cat([q, k, v], dim=2) * self.params.fact_scaler * self.params.merge_factor
            elif mode == 'attn_proj':
                _, _, _, proj_FacTs, dp = self.per_block_para[block_idx]['attn']
                return self.FacTv(dp(proj_FacTs(self.FacTu(x)))) * self.params.fact_scaler * self.params.merge_factor
            elif mode == 'mlp_1':
                fc1_FacTs, _, dp = self.per_block_para[block_idx]['mlp']
                return self.FacTv(dp(fc1_FacTs(self.FacTu(x))).reshape(
                    B, N, 4, self.fact_dim)).reshape(
                    B, N, 4 * C) * self.params.fact_scaler * self.params.merge_factor
            elif mode == 'mlp_2':
                x = x.reshape(B, N, 4, C)
                _, fc2_FacTs, dp = self.per_block_para[block_idx]['mlp']
                return self.FacTv(dp(fc2_FacTs(self.FacTu(x).reshape(
                    B, N, 4 * self.fact_dim)))) * self.params.fact_scaler * self.params.merge_factor
            else:
                raise NotImplementedError
        elif self.params.fact_type == 'tk':
            if mode == 'attn_qkv':
                dp = self.per_block_para[block_idx]['attn']
                start_idx = block_idx * TOTAL_WEIGHT_MATRIX
                FacTc = self.FacTc @ self.FacTp[:, start_idx:start_idx + 4]
                q_FacTc, k_FacTc, v_FacTc = FacTc[:, :, 0], FacTc[:, :, 1], FacTc[:, :, 2]
                q = self.FacTv(dp(self.FacTu(x) @ q_FacTc))
                k = self.FacTv(dp(self.FacTu(x) @ k_FacTc))
                v = self.FacTv(dp(self.FacTu(x) @ v_FacTc))
                return torch.cat([q, k, v], dim=2) * self.params.fact_scaler * self.params.merge_factor
            elif mode == 'attn_proj':
                dp = self.per_block_para[block_idx]['attn']
                start_idx = block_idx * TOTAL_WEIGHT_MATRIX
                FacTc = self.FacTc @ self.FacTp[:, start_idx:start_idx + 4]
                proj_FacTc = FacTc[:, :, 3]
                return self.FacTv(dp(self.FacTu(x) @ proj_FacTc)) * self.params.fact_scaler * self.params.merge_factor
            elif mode == 'mlp_1':
                dp = self.per_block_para[block_idx]['mlp']
                start_idx = block_idx * TOTAL_WEIGHT_MATRIX + 4
                FacTc = self.FacTc @ self.FacTp[:, start_idx:start_idx + 8]
                fc1_FacTc, _ = FacTc[:, :, :4].reshape(self.fact_dim, self.fact_dim * 4), FacTc[:, :, 4:].reshape(
                    self.fact_dim,
                    self.fact_dim * 4)
                return self.FacTv(dp(self.FacTu(x) @ fc1_FacTc).reshape(
                            B, N, 4, self.fact_dim)).reshape(
                            B, N, 4 * C) * self.params.fact_scaler * self.params.merge_factor
            elif mode == 'mlp_2':
                x = x.reshape(B, N, 4, C)
                dp = self.per_block_para[block_idx]['mlp']
                start_idx = block_idx * TOTAL_WEIGHT_MATRIX + 4
                FacTc = self.FacTc @ self.FacTp[:, start_idx:start_idx + 8]
                _, fc2_FacTc = FacTc[:, :, :4].reshape(self.fact_dim, self.fact_dim * 4), FacTc[:, :, 4:].reshape(
                    self.fact_dim,
                    self.fact_dim * 4)
                return self.FacTv(dp(self.FacTu(x).reshape(
                        B, N, 4 * self.fact_dim) @ fc2_FacTc.t())) * self.params.fact_scaler * self.params.merge_factor
            else:
                raise NotImplementedError