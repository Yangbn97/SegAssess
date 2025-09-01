import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn.functional import upsample
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.scatter_gather import scatter
from thop import profile
# from models import resnet
from torch.jit import Final
from typing import Type
from timm.layers import use_fused_attn
up_kwargs = {'mode': 'bilinear', 'align_corners': True}


# class BaseNet(nn.Module):
#     def __init__(self, nclass, backbone, norm_layer=None,
#                  crop_size=472, mean=[.485, .456, .406],
#                  std=[.229, .224, .225], pretrained=False,root='./pretrain_models'):
#         super(BaseNet, self).__init__()
#         self.nclass = nclass
#         self.mean = mean
#         self.std = std
#         self.crop_size = crop_size
#         self.pretrained = pretrained
#         # copying modules from pretrained models
#         if backbone == 'resnet50':
#             self.pretrained = resnet.resnet50(pretrained=self.pretrained,
#                                               norm_layer=norm_layer, root=root)
#         elif backbone == 'resnet101':
#             self.pretrained = resnet.resnet101(pretrained=self.pretrained,
#                                                norm_layer=norm_layer, root=root)
#         elif backbone == 'resnet152':
#             self.pretrained = resnet.resnet152(pretrained=self.pretrained,
#                                                norm_layer=norm_layer, root=root)
#         else:
#             raise RuntimeError('unknown backbone: {}'.format(backbone))
#         # bilinear upsample options
#         self._up_kwargs = up_kwargs


#     def base_forward(self, x):
#         x = self.pretrained.conv1(x)
#         x = self.pretrained.bn1(x)
#         c1 = self.pretrained.relu(x)

#         x = self.pretrained.maxpool(c1)
#         c2 = self.pretrained.layer1(x)
#         c3 = self.pretrained.layer2(c2)
#         c4 = self.pretrained.layer3(c3)
#         c5 = self.pretrained.layer4(c4)
#         return c1, c2, c3, c4, c5

    
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context, mask=None):
        B, N, C = x.shape
        _, M, _ = context.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, M, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, e=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            with autocast(False):
                attn = q @ k.transpose(-2, -1)
                if e is not None:
                    attn = attn + e
                attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        #d_model是每个词embedding后的维度
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term2 = torch.pow(torch.tensor(10000.0),torch.arange(0, d_model, 2).float()/d_model)
        div_term1 = torch.pow(torch.tensor(10000.0),torch.arange(1, d_model, 2).float()/d_model)
        #高级切片方式，即从0开始，两个步长取一个。即奇数和偶数位置赋值不一样。直观来看就是每一句话的
        pe[:, 0::2] = torch.sin(position * div_term2)
        pe[:, 1::2] = torch.cos(position * div_term1)
        #这里是为了与x的维度保持一致，释放了一个维度
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        可学习的位置编码初始化
        :param d_model: 模型的维度
        :param max_len: 序列的最大长度
        """
        super(LearnablePositionalEncoding, self).__init__()
        # 创建一个Embedding层，形状为 (max_len, d_model)
        self.position_embeddings = nn.Embedding(max_len, d_model)
        nn.init.constant(self.position_embeddings.weight, 0.)

    def forward(self, x):
        # 获取输入序列的长度
        seq_len = x.size(0)
        # 生成位置索引，形状为 (seq_len, 1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(1)
        # 将位置索引通过Embedding层转换为位置编码，然后加到输入张量上
        return x + self.position_embeddings(position_ids)

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

