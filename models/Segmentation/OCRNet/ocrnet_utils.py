from collections import OrderedDict

import models.Segmentation.OCRNet.hrnetv2 as hrnetv2
import torch
import torch.nn as nn
import torch.nn.functional as F

ALIGN_CORNERS = False
MSCALE_INNER_3x3 = True
MSCALE_DROPOUT = False
OCR_ASPP = False


def initialize_weights(*models):
    """
    Initialize Model Weights
    """
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(
        x, size=size, mode="bilinear", align_corners=ALIGN_CORNERS
    )


def scale_as(x, y):
    """
    scale x to the same size as y
    """
    y_size = y.size(2), y.size(3)
    x_scaled = torch.nn.functional.interpolate(
        x, size=y_size, mode="bilinear", align_corners=ALIGN_CORNERS
    )
    return x_scaled


def ResizeX(x, scale_factor):
    """
    scale x by same factor
    """
    x_scaled = torch.nn.functional.interpolate(
        x,
        scale_factor=scale_factor,
        mode="bilinear",
        align_corners=ALIGN_CORNERS,
        recompute_scale_factor=True,
    )
    return x_scaled


def fmt_scale(prefix, scale):
    """
    format scale name

    :prefix: a string that is the beginning of the field name
    :scale: a scale value (0.25, 0.5, 1.0, 2.0)
    """

    scale_str = str(float(scale))
    scale_str.replace(".", "")
    return f"{prefix}_{scale_str}x"


def get_trunk(trunk_name, output_stride=8):
    """
    Retrieve the network trunk and channel counts.
    """
    assert output_stride == 8, "Only stride8 supported right now"

    if trunk_name == "hrnetv2":
        backbone = hrnetv2.get_seg_model()
        high_level_ch = backbone.high_level_ch
        s2_ch = -1
        s4_ch = -1
    else:
        raise "unknown backbone {}".format(trunk_name)

    return backbone, s2_ch, s4_ch, high_level_ch


def BNReLU(ch):
    return nn.Sequential(nn.BatchNorm2d(ch), nn.ReLU())


def make_attn_head(in_ch, out_ch):
    bot_ch = 256

    od = OrderedDict(
        [
            ("conv0", nn.Conv2d(in_ch, bot_ch, kernel_size=3, padding=1, bias=False)),
            ("bn0", nn.BatchNorm2d(bot_ch)),
            ("re0", nn.ReLU(inplace=True)),
        ]
    )

    if MSCALE_INNER_3x3:
        od["conv1"] = nn.Conv2d(bot_ch, bot_ch, kernel_size=3, padding=1, bias=False)
        od["bn1"] = nn.BatchNorm2d(bot_ch)
        od["re1"] = nn.ReLU(inplace=True)

    if MSCALE_DROPOUT:
        od["drop"] = nn.Dropout(0.5)

    od["conv2"] = nn.Conv2d(bot_ch, out_ch, kernel_size=1, bias=False)
    od["sig"] = nn.Sigmoid()

    attn_head = nn.Sequential(od)
    # init_attn(attn_head)
    return attn_head


class SpatialGather_Module(nn.Module):
    """
    Aggregate the context features according to the initial
    predicted probability distribution.
    Employ the soft-weighted method to aggregate the context.

    Output:
      The correlation of every class map with every feature map
      shape = [n, num_feats, num_classes, 1]
    """

    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, _, _ = probs.size(0), probs.size(1), probs.size(2), probs.size(3)

        # each class image now a vector
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)

        feats = feats.permute(0, 2, 1)  # batch x hw x c
        probs = F.softmax(self.scale * probs, dim=2)  # batch x k x hw
        ocr_context = torch.matmul(probs, feats)
        ocr_context = ocr_context.permute(0, 2, 1).unsqueeze(3)
        return ocr_context


class ObjectAttentionBlock(nn.Module):
    """
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature
                            maps (save memory cost)
    Return:
        N X C X H X W
    """

    def __init__(self, in_channels, key_channels, scale=1):
        super(ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            BNReLU(self.key_channels),
            nn.Conv2d(
                in_channels=self.key_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            BNReLU(self.key_channels),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            BNReLU(self.key_channels),
            nn.Conv2d(
                in_channels=self.key_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            BNReLU(self.key_channels),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            BNReLU(self.key_channels),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(
                in_channels=self.key_channels,
                out_channels=self.in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            BNReLU(self.in_channels),
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(
                input=context, size=(h, w), mode="bilinear", align_corners=ALIGN_CORNERS
            )

        return context


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation
    for each pixel.
    """

    def __init__(self, in_channels, key_channels, out_channels, scale=1, dropout=0.1):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock(
            in_channels, key_channels, scale
        )
        if OCR_ASPP:
            self.aspp, aspp_out_ch = get_aspp(
                in_channels, bottleneck_ch=256, output_stride=8  # cfg.MODEL.ASPP_BOT_CH
            )
            _in_channels = 2 * in_channels + aspp_out_ch
        else:
            _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            BNReLU(out_channels),
            nn.Dropout2d(dropout),
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        if OCR_ASPP:
            aspp = self.aspp(feats)
            output = self.conv_bn_dropout(torch.cat([context, aspp, feats], 1))
        else:
            output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return output


def dpc_conv(in_dim, reduction_dim, dil, separable):
    if separable:
        groups = reduction_dim
    else:
        groups = 1

    return nn.Sequential(
        nn.Conv2d(
            in_dim,
            reduction_dim,
            kernel_size=3,
            dilation=dil,
            padding=dil,
            bias=False,
            groups=groups,
        ),
        nn.Batchnn.BatchNorm2d(reduction_dim),
        nn.ReLU(inplace=True),
    )


class DPC(nn.Module):
    """
    From: Searching for Efficient Multi-scale architectures for dense
    prediction
    """

    def __init__(
        self,
        in_dim,
        reduction_dim=256,
        output_stride=16,
        rates=[(1, 6), (18, 15), (6, 21), (1, 1), (6, 3)],
        dropout=False,
        separable=False,
    ):
        super(DPC, self).__init__()

        self.dropout = dropout
        if output_stride == 8:
            rates = [(2 * r[0], 2 * r[1]) for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise "output stride of {} not supported".format(output_stride)

        self.a = dpc_conv(in_dim, reduction_dim, rates[0], separable)
        self.b = dpc_conv(reduction_dim, reduction_dim, rates[1], separable)
        self.c = dpc_conv(reduction_dim, reduction_dim, rates[2], separable)
        self.d = dpc_conv(reduction_dim, reduction_dim, rates[3], separable)
        self.e = dpc_conv(reduction_dim, reduction_dim, rates[4], separable)

        self.drop = nn.Dropout(p=0.1)

    def forward(self, x):
        a = self.a(x)
        b = self.b(a)
        c = self.c(a)
        d = self.d(a)
        e = self.e(b)
        out = torch.cat((a, b, c, d, e), 1)
        if self.dropout:
            out = self.drop(out)
        return out


class AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=(6, 12, 18)):
        super(AtrousSpatialPyramidPoolingModule, self).__init__()

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise "output stride of {} not supported".format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True),
            )
        )
        # other rates
        for r in rates:
            self.features.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_dim,
                        reduction_dim,
                        kernel_size=3,
                        dilation=r,
                        padding=r,
                        bias=False,
                    ),
                    nn.BatchNorm2d(reduction_dim),
                    nn.ReLU(inplace=True),
                )
            )
        self.features = nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduction_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = Upsample(img_features, x_size[2:])
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out


def get_aspp(high_level_ch, bottleneck_ch, output_stride, dpc=False):
    """
    Create aspp block
    """
    if dpc:
        aspp = DPC(high_level_ch, bottleneck_ch, output_stride=output_stride)
    else:
        aspp = AtrousSpatialPyramidPoolingModule(
            high_level_ch, bottleneck_ch, output_stride=output_stride
        )
    aspp_out_ch = 5 * bottleneck_ch
    return aspp, aspp_out_ch