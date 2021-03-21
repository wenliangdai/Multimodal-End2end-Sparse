import torch
from torch import nn
import torch.nn.functional as F
from typing import List


class CrossModalAttentionLayer(nn.Module):
    # y attends x
    def __init__(self, k, x_channels: int, y_size: int, spatial=True):
        super(CrossModalAttentionLayer, self).__init__()
        self.k = k
        self.spatial = spatial

        if spatial:
            self.channel_affine = nn.Linear(x_channels, k)

        self.y_affine = nn.Linear(y_size, k, bias=False)
        self.attn_weight_affine = nn.Linear(k, 1)

    def forward(self, x: List[torch.Tensor], x_lens: List[int], y: torch.Tensor):
        # x -> [(S, C, H, W)], len(x) = bs
        # y -> (bs, D)

        bs = y.size(0)
        x = x.split(x_lens, dim=0)
        y_k = self.y_affine(y) # (bs, k)

        all_spatial_attn_weights_softmax = []

        for i in range(bs):
            if self.spatial:
                x_tensor = x[i].permute(0, 2, 3, 1) # (S_v, H_v, W_v, C_v)
                x_k = self.channel_affine(x_tensor) # (S_v, H_v, W_v, k)
                x_k += y_k[i]
                x_k = torch.tanh(x_k)
                x_attn_weights = self.attn_weight_affine(x_k).squeeze(-1) # (S_v, H_v, W_v)

                all_spatial_attn_weights_softmax.append(
                    F.softmax(
                        x_attn_weights.reshape(x_tensor.size(0), -1),
                        dim=-1
                    ).reshape(x_tensor.size(0), x_tensor.size(1), x_tensor.size(2)) # (S_v, H_v, W_v)
                )

        return torch.cat(all_spatial_attn_weights_softmax, dim=0)

class SparseCrossModalAttentionLayer(nn.Module):
    def __init__(self, k: int, x_channels: int, y_size: int, sparse_threshold: float):
        super(SparseCrossModalAttentionLayer, self).__init__()
        self.k = k
        self.sparse_threshold = sparse_threshold
        self.channel_affine = nn.Linear(x_channels, k)
        self.y_affine = nn.Linear(y_size, k, bias=False)
        self.attn_weight_affine = nn.Linear(k, 1)

    def forward(self, x: List[torch.Tensor], x_lens: List[int], locations: List[torch.Tensor], y: torch.Tensor):
        # x -> (N, C)
        # locations -> (N, 3)
        # y -> (bs, D)
        bs = y.size(0)
        y_k = self.y_affine(y) # (bs, k)
        x_k = self.channel_affine(x) # (N, k)

        sample_points_lens = []
        for i in range(sum(x_lens)):
            sample_points_lens.append(len(locations[locations[:, 2] == i]))

        # how much points are left in each batch
        batch_points_lens = []
        pointer = 0
        for l in x_lens:
            batch_points_lens.append(sum(sample_points_lens[pointer:(pointer + l)]))
            pointer += l

        x_ks = x_k.split(batch_points_lens, dim=0)

        attn_weights = []
        for i in range(bs):
            this_weights = self.attn_weight_affine(torch.tanh(x_ks[i] + y_k[i])).squeeze(-1)
            attn_weights.append(this_weights)

        attn_weights = torch.cat(attn_weights, dim=0)
        attn_weights_split = list(attn_weights.split(sample_points_lens, dim=0))
        attn_weights_split = [F.softmax(a, dim=-1) for a in attn_weights_split]
        attn_weights = torch.cat(attn_weights_split, dim=0)

        attn_weights_sparse = to_sparse_by_cdf(attn_weights, sample_points_lens, self.sparse_threshold)

        select_indices = attn_weights_sparse == 1
        new_x = x[select_indices, :]
        new_locations = locations[select_indices, :]

        return new_x, new_locations, None

def to_sparse_by_cdf(t: torch.tensor, lens, cdf: float):
    _t = t.clone().detach()
    _t = list(_t.split(lens, dim=0))

    for i, this_t in enumerate(_t):
        this_t_sorted, indices = torch.sort(this_t, descending=True)
        mask = torch.cumsum(this_t_sorted, dim=-1) < cdf
        mask[torch.sum(mask)] = True
        _t[i][indices[mask]] = 1
        _t[i][indices[~mask]] = 0

    return torch.cat(_t, dim=0).long()
