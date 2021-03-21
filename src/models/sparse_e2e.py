import sparseconvnet as scn
import torch
from torch import nn
from src.models.e2e_t import MME2E_T
from src.models.vgg_block import VggBasicBlock
from src.models.transformer_encoder import WrappedTransformerEncoder
from src.models.attention_block import CrossModalAttentionLayer, SparseCrossModalAttentionLayer
from torchvision import transforms
from facenet_pytorch import MTCNN

def to_sparse_by_cdf(t: torch.tensor, cdf: float):
    N = t.size(0)
    t_flatten = t.flatten(start_dim=1).clone().detach()
    sorted_t_flatten, indices = torch.sort(t_flatten, descending=True, dim=-1)
    sorted_t_flatten_cum = torch.cumsum(sorted_t_flatten, dim=-1)

    for i in range(N):
        mask = sorted_t_flatten_cum[i] < cdf
        mask[torch.sum(mask)] = True
        t_flatten[i, indices[i, mask]] = 1
        t_flatten[i, indices[i, ~mask]] = 0

    return t_flatten.reshape(*t.size()).long()

class MME2E_Sparse(nn.Module):
    def __init__(self, args, device):
        super(MME2E_Sparse, self).__init__()
        self.args = args
        self.device = device
        self.num_classes = args['num_emotions']
        self.mod = args['modalities'].lower()
        self.feature_dim = args['feature_dim']
        self.threshold = args['sparse_threshold']
        nlayers = args['trans_nlayers']
        nheads = args['trans_nheads']
        trans_dim = args['trans_dim']

        text_cls_dim = 768
        if args['text_model_size'] == 'large':
            text_cls_dim = 1024
        if args['text_model_size'] == 'xlarge':
            text_cls_dim = 2048

        # Textual
        self.T = MME2E_T(feature_dim=self.feature_dim, size=args['text_model_size'])

        # Visual
        self.mtcnn = MTCNN(image_size=48, margin=2, post_process=False, device=device)
        self.normalize = transforms.Normalize(mean=[159, 111, 102], std=[37, 33, 32])

        self.V = nn.ModuleDict({
            'low_level': nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                VggBasicBlock(in_planes=64, out_planes=64),
                VggBasicBlock(in_planes=64, out_planes=64)
            ),
            'sparse_layers': nn.ModuleList([
                scn.Sequential(
                    # 第一个 2，指的是 2D
                    scn.SparseVggNet(2, 64, [
                        ['C', 128], ['C', 128], ['MP', 2, 2]
                    ])
                ),
                scn.Sequential(
                    scn.SparseVggNet(2, 128, [
                        ['C', 256], ['C', 256], ['MP', 2, 2]
                    ])
                ),
                scn.Sequential(
                    scn.SparseVggNet(2, 256, [
                        ['C', 512], ['C', 512], ['MP', 2, 2]
                    ]),
                    scn.SparseToDense(2, 512)
                )
            ]),
            'attn_layers': nn.ModuleList([
                CrossModalAttentionLayer(k=64, x_channels=64, y_size=text_cls_dim, spatial=True),
                SparseCrossModalAttentionLayer(k=128, x_channels=128, y_size=text_cls_dim, sparse_threshold=self.threshold),
                SparseCrossModalAttentionLayer(k=256, x_channels=256, y_size=text_cls_dim, sparse_threshold=self.threshold)
            ])
        })

        self.A = nn.ModuleDict({
            'low_level': nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                VggBasicBlock(in_planes=64, out_planes=64),
                VggBasicBlock(in_planes=64, out_planes=64)
            ),
            'sparse_layers': nn.ModuleList([
                scn.Sequential().add(
                    scn.SparseVggNet(2, 64, [
                        ['C', 128], ['C', 128], ['MP', 2, 2]
                    ])
                ),
                scn.Sequential().add(
                    scn.SparseVggNet(2, 128, [
                        ['C', 256], ['C', 256], ['MP', 2, 2]
                    ])
                ),
                scn.Sequential().add(
                    scn.SparseVggNet(2, 256, [
                        ['C', 512], ['C', 512], ['MP', 2, 2]
                    ])
                ).add(scn.SparseToDense(2, 512))
            ]),
            'attn_layers': nn.ModuleList([
                CrossModalAttentionLayer(k=64, x_channels=64, y_size=text_cls_dim, spatial=True),
                SparseCrossModalAttentionLayer(k=128, x_channels=128, y_size=text_cls_dim, sparse_threshold=self.threshold),
                SparseCrossModalAttentionLayer(k=256, x_channels=256, y_size=text_cls_dim, sparse_threshold=self.threshold)
            ])
        })

        self.v_sparse_input_layers = nn.ModuleList([
            scn.InputLayer(2, self.V['sparse_layers'][0].input_spatial_size(torch.LongTensor([12, 12]))),
            scn.InputLayer(2, self.V['sparse_layers'][1].input_spatial_size(torch.LongTensor([6, 6]))),
            scn.InputLayer(2, self.V['sparse_layers'][2].input_spatial_size(torch.LongTensor([3, 3])))
        ])

        self.a_sparse_input_layers = nn.ModuleList([
            scn.InputLayer(2, self.A['sparse_layers'][0].input_spatial_size(torch.LongTensor([32, 8]))),
            scn.InputLayer(2, self.A['sparse_layers'][1].input_spatial_size(torch.LongTensor([16, 4]))),
            scn.InputLayer(2, self.A['sparse_layers'][2].input_spatial_size(torch.LongTensor([8, 2])))
        ])

        self.v_flatten = nn.Sequential(
            nn.Linear(512 * 3 * 3, 1024),
            nn.ReLU(),
            nn.Linear(1024, trans_dim)
        )

        self.a_flatten = nn.Sequential(
            nn.Linear(512 * 8 * 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, trans_dim)
        )

        self.v_transformer = WrappedTransformerEncoder(dim=trans_dim, num_layers=nlayers, num_heads=nheads)
        self.a_transformer = WrappedTransformerEncoder(dim=trans_dim, num_layers=nlayers, num_heads=nheads)

        # Output layers
        self.t_out = nn.Linear(text_cls_dim, self.num_classes)
        self.v_out = nn.Linear(trans_dim, self.num_classes)
        self.a_out = nn.Linear(trans_dim, self.num_classes)

        self.weighted_fusion = nn.Linear(len(self.mod), 1, bias=False)

    def forward(self, imgs, imgs_lens, specs, spec_lens, text):
        all_logits = []

        if 't' not in self.mod:
            raise ValueError('For sparse model, the textual modality must exist!')

        text_cls = self.T(text, get_cls=True)
        all_logits.append(self.t_out(text_cls))

        if 'v' in self.mod:
            total_samples = sum(imgs_lens)
            # Face extraction
            faces = self.mtcnn(imgs)
            for i, face in enumerate(faces):
                if face is None:
                    center = self.crop_img_center(torch.tensor(imgs[i]).permute(2, 0, 1))
                    faces[i] = center

            faces = [self.normalize(face) for face in faces]
            faces = torch.stack(faces, dim=0).to(device=self.device)

            # Go through a few dense CNN layers first, to extract low level visual features
            faces = self.V['low_level'](faces)

            # Go through sparse CNN layers
            for i, sparse_layer in enumerate(self.V['sparse_layers']):
                if i == 0:
                    attn_weights_softmax = self.V['attn_layers'][i](faces, imgs_lens, text_cls)
                    attn_weights = to_sparse_by_cdf(attn_weights_softmax, self.threshold)

                    locations = attn_weights.permute(1, 2, 0).nonzero()
                    bs, y, x = attn_weights.nonzero(as_tuple=True)
                    features = faces[bs, :, y, x]
                    sparse_input = self.v_sparse_input_layers[i]([locations, features, total_samples])
                    faces = sparse_layer(sparse_input)
                else:
                    current_locations = faces.get_spatial_locations()
                    features, locations, sparse_attn_weights = self.V['attn_layers'][i](faces.features, imgs_lens, current_locations, text_cls)
                    sparse_input = self.v_sparse_input_layers[i]([locations, features, total_samples])
                    faces = sparse_layer(sparse_input)

            faces = self.v_flatten(faces.flatten(start_dim=1))
            faces = self.v_transformer(faces, imgs_lens, get_cls=True)

            all_logits.append(self.v_out(faces))

        if 'a' in self.mod:
            total_samples = sum(spec_lens)
            specs = self.A['low_level'](specs)
            for i, sparse_layer in enumerate(self.A['sparse_layers']):
                if i == 0:
                    attn_weights_softmax = self.A['attn_layers'][i](specs, spec_lens, text_cls)
                    attn_weights = to_sparse_by_cdf(attn_weights_softmax, self.threshold)

                    locations = attn_weights.permute(1, 2, 0).nonzero()
                    bs, y, x = attn_weights.nonzero(as_tuple=True)
                    features = specs[bs, :, y, x]
                    sparse_input = self.a_sparse_input_layers[i]([locations, features, total_samples])
                    specs = sparse_layer(sparse_input)
                else:
                    current_locations = specs.get_spatial_locations()
                    features, locations, sparse_attn_weights = self.A['attn_layers'][i](specs.features, spec_lens, current_locations, text_cls)
                    sparse_input = self.a_sparse_input_layers[i]([locations, features, total_samples])
                    specs = sparse_layer(sparse_input)

            specs = self.a_flatten(specs.flatten(start_dim=1))
            specs = self.a_transformer(specs, spec_lens, get_cls=True)

            all_logits.append(self.a_out(specs))

        if len(self.mod) == 1:
            return all_logits[0]

        return self.weighted_fusion(torch.stack(all_logits, dim=-1)).squeeze(-1)

    def crop_img_center(self, img: torch.tensor, target_size=48):
        '''
        Some images have un-detectable faces,
        to make the training goes normally,
        for those images, we crop the center part,
        which highly likely contains the face or part of the face.

        @img - (channel, height, width)
        '''
        current_size = img.size(1)
        off = (current_size - target_size) // 2 # offset
        cropped = img[:, off:off + target_size, off - target_size // 2:off + target_size // 2]
        return cropped
