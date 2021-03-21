import math
from typing import Optional, List
import torch
from torch import nn
from src.utils import padTensor

class WrappedTransformerEncoder(nn.Module):
    def __init__(self, dim, num_layers, num_heads):
        super(WrappedTransformerEncoder, self).__init__()
        self.dim = dim
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=dim)

    def prepend_cls(self, inputs):
        index = torch.LongTensor([0]).to(device=inputs.device)
        cls_emb = self.cls_emb(index)
        cls_emb = cls_emb.expand(inputs.size(0), 1, self.dim)
        outputs = torch.cat((cls_emb, inputs), dim=1)
        return outputs

    def forward(self, inputs: torch.Tensor, lens: Optional[List[int]] = None, get_cls: Optional[bool] = False):
        if lens is not None:
            max_len = max(lens)

            mask = [([False] * (l + int(get_cls)) + [True] * (max_len - l)) for l in lens]
            mask = torch.tensor(mask).to(device=inputs.device)

            inputs = list(inputs.split(lens, dim=0))
            inputs = [padTensor(inp, max_len) for inp in inputs]
            inputs = torch.stack(inputs, dim=0)
        else:
            mask = None

        if get_cls:
            inputs = self.prepend_cls(inputs)

        inputs = inputs.permute(1, 0, 2)
        # inputs = self.pos_encoder(inputs)
        inputs = self.encoder(src=inputs, src_key_padding_mask=mask) # (seq_len, bs, dim)

        if get_cls:
            return inputs[0]

        return inputs[1:].permute(1, 0, 2)

