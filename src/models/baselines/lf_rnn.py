import numpy as np
import torch
from torch import nn
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import List
from src.utils import padTensor

pad_token_id = 0
unk_token_id = 1

class LF_RNN(nn.Module):
    def __init__(self, args, num_layers=1, dropout=0.1, bi=True):
        super(LF_RNN, self).__init__()
        feature_sizes = args['hfc_sizes']
        num_classes = args['num_emotions']
        self.mods = args['modalities']

        feature_sizes = np.array(feature_sizes)

        self.rnns = nn.ModuleDict({
            't': nn.LSTM(
                input_size=feature_sizes[0],
                hidden_size=feature_sizes[0],
                num_layers=num_layers,
                dropout=(dropout if num_layers > 1 else 0),
                bidirectional=bi
            ),
            'a': nn.LSTM(
                input_size=feature_sizes[1],
                hidden_size=feature_sizes[1],
                num_layers=num_layers,
                dropout=(dropout if num_layers > 1 else 0),
                bidirectional=bi
            ),
            'v': nn.LSTM(
                input_size=feature_sizes[2],
                hidden_size=feature_sizes[2],
                num_layers=num_layers,
                dropout=(dropout if num_layers > 1 else 0),
                bidirectional=bi
            )
        })

        linear_in_sizes = feature_sizes if not bi else feature_sizes * 2

        self.affines = nn.ModuleDict({
            't': nn.Sequential(
                nn.Linear(linear_in_sizes[0], linear_in_sizes[0] // 2),
                nn.ReLU(),
                nn.Linear(linear_in_sizes[0] // 2, num_classes)
            ),
            'a': nn.Sequential(
                nn.Linear(linear_in_sizes[1], linear_in_sizes[1] // 2),
                nn.ReLU(),
                nn.Linear(linear_in_sizes[1] // 2, num_classes)
            ),
            'v': nn.Sequential(
                nn.Linear(linear_in_sizes[2], linear_in_sizes[2] // 2),
                nn.ReLU(),
                nn.Linear(linear_in_sizes[2] // 2, num_classes)
            )
        })

        self.weighted_fusion = nn.Linear(len(self.mods), 1, bias=False)

    def forward(self, img_features, img_features_lens, audio_features, audio_features_lens, texts):
        all_logits = []

        if 't' in self.mods:
            output_t, _ = self.rnns['t'](texts.transpose(0, 1))
            output_t = output_t[-1, :, :]
            output_t = self.affines['t'](output_t)
            all_logits.append(output_t)

        if 'a' in self.mods:
            max_len = max(audio_features_lens)
            audio_features = audio_features.split(audio_features_lens, dim=0)
            audio_features = [padTensor(s, max_len) for s in audio_features]
            audio_features = torch.stack(audio_features, dim=1) # (seq_len, batch, dim)
            _, (audio_hn, _) = self.rnns['a'](audio_features)
            audio_hn = audio_hn.transpose(0, 1).flatten(start_dim=1) # (batch, hid_dim * 2)
            audio_hn = self.affines['a'](audio_hn)
            all_logits.append(audio_hn)

        if 'v' in self.mods:
            max_len = max(img_features_lens)
            img_features = img_features.split(img_features_lens, dim=0)
            img_features = [padTensor(s, max_len) for s in img_features]
            img_features = torch.stack(img_features, dim=1) # (seq_len, batch, dim)
            _, (img_hn, _) = self.rnns['v'](img_features)
            img_hn = img_hn.transpose(0, 1).flatten(start_dim=1) # (batch, hid_dim * 2)
            img_hn = self.affines['v'](img_hn)
            all_logits.append(img_hn)

        if len(self.mods) == 1:
            return all_logits[0]

        return self.weighted_fusion(torch.stack(all_logits, dim=-1)).squeeze(-1)
