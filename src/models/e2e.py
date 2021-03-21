import torch
from torch import nn
from src.models.e2e_t import MME2E_T

from src.models.transformer_encoder import WrappedTransformerEncoder
from torchvision import transforms
from facenet_pytorch import MTCNN
from src.models.vgg_block import VggBasicBlock


class MME2E(nn.Module):
    def __init__(self, args, device):
        super(MME2E, self).__init__()
        self.num_classes = args['num_emotions']
        self.args = args
        self.mod = args['modalities'].lower()
        self.device = device
        self.feature_dim = args['feature_dim']
        nlayers = args['trans_nlayers']
        nheads = args['trans_nheads']
        trans_dim = args['trans_dim']

        text_cls_dim = 768
        if args['text_model_size'] == 'large':
            text_cls_dim = 1024
        if args['text_model_size'] == 'xlarge':
            text_cls_dim = 2048

        self.T = MME2E_T(feature_dim=self.feature_dim, size=args['text_model_size'])

        self.mtcnn = MTCNN(image_size=48, margin=2, post_process=False, device=device)
        self.normalize = transforms.Normalize(mean=[159, 111, 102], std=[37, 33, 32])

        self.V = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VggBasicBlock(in_planes=64, out_planes=64),
            VggBasicBlock(in_planes=64, out_planes=64),
            VggBasicBlock(in_planes=64, out_planes=128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VggBasicBlock(in_planes=128, out_planes=256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VggBasicBlock(in_planes=256, out_planes=512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.A = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VggBasicBlock(in_planes=64, out_planes=64),
            VggBasicBlock(in_planes=64, out_planes=64),
            VggBasicBlock(in_planes=64, out_planes=128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VggBasicBlock(in_planes=128, out_planes=256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VggBasicBlock(in_planes=256, out_planes=512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

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

        self.v_out = nn.Linear(trans_dim, self.num_classes)
        self.t_out = nn.Linear(text_cls_dim, self.num_classes)
        self.a_out = nn.Linear(trans_dim, self.num_classes)

        self.weighted_fusion = nn.Linear(len(self.mod), 1, bias=False)

    def forward(self, imgs, imgs_lens, specs, spec_lens, text):
        all_logits = []

        if 't' in self.mod:
            text_cls = self.T(text, get_cls=True)
            all_logits.append(self.t_out(text_cls))

        if 'v' in self.mod:
            faces = self.mtcnn(imgs)
            for i, face in enumerate(faces):
                if face is None:
                    center = self.crop_img_center(torch.tensor(imgs[i]).permute(2, 0, 1))
                    faces[i] = center
            faces = [self.normalize(face) for face in faces]
            faces = torch.stack(faces, dim=0).to(device=self.device)

            faces = self.V(faces)

            faces = self.v_flatten(faces.flatten(start_dim=1))
            faces = self.v_transformer(faces, imgs_lens, get_cls=True)

            all_logits.append(self.v_out(faces))

        if 'a' in self.mod:
            for a_module in self.A:
                specs = a_module(specs)

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
