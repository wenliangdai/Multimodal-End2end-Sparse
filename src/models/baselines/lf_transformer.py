import numpy as np
import torch
from torch import nn
from typing import List
from src.models.transformer_encoder import WrappedTransformerEncoder


class LF_Transformer(nn.Module):
    def __init__(self, args):
        super(LF_Transformer, self).__init__()
        num_classes = args['num_emotions']
        self.mods = args['modalities']
        feature_sizes = np.array(args['hfc_sizes'])
        nlayers = args['trans_nlayers']
        # nheads = args['trans_nheads']
        # trans_dim = args['trans_dim']

        self.transformers = nn.ModuleDict({
            't': WrappedTransformerEncoder(
                dim=feature_sizes[0], # 300
                num_layers=nlayers,
                num_heads=4
            ),
            'a': WrappedTransformerEncoder(
                dim=feature_sizes[1], # 2 empty features are added to make it 144, easy to be divided by #heads
                num_layers=nlayers,
                num_heads=2
            ),
            'v': WrappedTransformerEncoder(
                dim=feature_sizes[2], # 35
                num_layers=nlayers,
                num_heads=5
            )
        })

        self.affines = nn.ModuleDict({
            't': nn.Sequential(
                nn.Linear(feature_sizes[0], feature_sizes[0] // 2),
                nn.ReLU(),
                nn.Linear(feature_sizes[0] // 2, num_classes)
            ),
            'a': nn.Sequential(
                nn.Linear(feature_sizes[1], feature_sizes[1] // 2),
                nn.ReLU(),
                nn.Linear(feature_sizes[1] // 2, num_classes)
            ),
            'v': nn.Sequential(
                nn.Linear(feature_sizes[2], feature_sizes[2] // 2),
                nn.ReLU(),
                nn.Linear(feature_sizes[2] // 2, num_classes)
            )
        })

        self.weighted_fusion = nn.Linear(len(self.mods), 1, bias=False)

    def forward(self, img_features, img_features_lens, audio_features, audio_features_lens, texts):
        all_logits = []

        if 't' in self.mods:
            texts = self.transformers['t'](texts, get_cls=True)
            texts = self.affines['t'](texts)
            all_logits.append(texts)

        if 'a' in self.mods:
            audio_features = self.transformers['a'](audio_features, audio_features_lens, get_cls=True)
            audio_features = self.affines['a'](audio_features)
            all_logits.append(audio_features)

        if 'v' in self.mods:
            img_features = self.transformers['v'](img_features, img_features_lens, get_cls=True)
            img_features = self.affines['v'](img_features)
            all_logits.append(img_features)

        if len(self.mods) == 1:
            return all_logits[0]

        return self.weighted_fusion(torch.stack(all_logits, dim=-1)).squeeze(-1)
