import numpy as np
from torch import nn
from torch import Tensor
from torch.nn import functional as F
import torch
from models.SAM.common import LayerNorm2d
from models.SAM.MaskDecoder.decoder_modules import TFA, BPM, LocationAdaptiveLearner


class Boundary_guided_Decoder_Swin(nn.Module):
    def __init__(self, n_obj_class=1, n_out_class=4, image_size=300, in_dim=1):
        super(Boundary_guided_Decoder_Swin, self).__init__()
        self.image_size = image_size
        self.n_obj_class = n_obj_class
        self.n_out_class = n_out_class
        self.in_dim = in_dim

        self.TFA = TFA(self.in_dim, out_dim=self.in_dim)
        self.side1 = nn.Sequential(nn.ConvTranspose2d(self.in_dim, self.in_dim // 4, 4, stride=2, padding=1),
                                    LayerNorm2d(self.in_dim // 4),
                                    nn.GELU(),
                                    nn.ConvTranspose2d(self.in_dim // 4, self.n_obj_class, 4, stride=2, padding=1))
        
        self.side2 = nn.Sequential(nn.ConvTranspose2d(self.in_dim, self.in_dim // 4, 4, stride=2, padding=1),
                                    LayerNorm2d(self.in_dim // 4),
                                    nn.GELU(),
                                    nn.ConvTranspose2d(self.in_dim // 4, self.n_obj_class, 4, stride=2, padding=1))
        
        self.side3 = nn.Sequential(nn.ConvTranspose2d(self.in_dim, self.in_dim // 4, 4, stride=2, padding=1),
                                    LayerNorm2d(self.in_dim // 4),
                                    nn.GELU(),
                                    nn.ConvTranspose2d(self.in_dim // 4, self.n_obj_class, 4, stride=2, padding=1))
        
        self.side4_w = nn.Sequential(nn.ConvTranspose2d(self.in_dim, self.in_dim // 4, 4, stride=2, padding=1),
                                    LayerNorm2d(self.in_dim // 4),
                                    nn.GELU(),
                                    nn.ConvTranspose2d(self.in_dim // 4, self.n_obj_class*4, 4, stride=2, padding=1))

        self.side4 = nn.Sequential(nn.ConvTranspose2d(self.in_dim, self.in_dim // 4, 4, stride=2, padding=1),
                                    LayerNorm2d(self.in_dim // 4),
                                    nn.GELU(),
                                    nn.ConvTranspose2d(self.in_dim // 4, self.n_obj_class, 4, stride=2, padding=1))
        
        self.ada_learner = LocationAdaptiveLearner(n_obj_class, n_obj_class * 4, n_obj_class * 4, norm_layer=nn.BatchNorm2d)
        
        self.BPM = BPM(self.n_out_class, 1, self.n_out_class, 1)

    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(
                    m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)

        self.apply(_init)
        # nn.init.constant_(self.fc.weight, 0)
        # nn.init.constant_(self.fc.bias, 0)
        # nn.init.constant_(self.class_token, 0)

    def forward(self, feats, sam_out, cond):
        _, _, h, w = cond.shape
        f1, f2, f3, f4 = feats
        side1 = self.side1(f1.permute(0, 3, 1, 2))
        side1 = F.interpolate(side1, (h, w), mode="bilinear", align_corners=False)
        # side1 = self.bpm1(side1, cond)
        side2 = self.side2(f2.permute(0, 3, 1, 2))
        side2 = F.interpolate(side2, (h, w), mode="bilinear", align_corners=False)
        # side2 = self.bpm2(side2, cond)
        side3 = self.side3(f3.permute(0, 3, 1, 2))
        side3 = F.interpolate(side3, (h, w), mode="bilinear", align_corners=False)
        # side3 = self.bpm3(side3, cond)
        f4 = self.TFA(f4.permute(0, 3, 1, 2))
        side4 = self.side4(f4)
        side4 = F.interpolate(side4, (h, w), mode="bilinear", align_corners=False)

        side4_w = self.side4_w(f4)
        side4_w = F.interpolate(side4_w, (h, w), mode="bilinear", align_corners=False)

        fuse = torch.cat((side4, side1, side2, side3), 1)

        ada_weights = self.ada_learner(side4_w)  # (N, nclass, 4, H, W)

        # fuse_feature = self.featureconv1(fuse)
        # fuse_feature = self.featureconv2(fuse_feature)
        
        # out = self.seg_head(fuse_feature)  # (N, nclass*4, H, W)

        edge = fuse.view(fuse.size(0), self.n_obj_class, -1, fuse.size(2), fuse.size(3))  # (N, nclass, 4, H, W)
        edge = torch.mul(edge, ada_weights)  # (N, nclass, 4, H, W)
        edge = torch.sum(edge, 2)  # (N, nclass, H, W)

        out = self.BPM(sam_out, edge)

        return out, edge