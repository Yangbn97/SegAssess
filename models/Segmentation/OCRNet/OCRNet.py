import torch
import torch.nn as nn
from models.Segmentation.OCRNet.ocrnet_utils import (
    BNReLU,
    ResizeX,
    SpatialGather_Module,
    SpatialOCR_Module,
    Upsample,
    fmt_scale,
    get_trunk,
    initialize_weights,
    make_attn_head,
    scale_as,
)

INIT_DECODER = False
MID_CHANNELS = 512
KEY_CHANNELS = 256

MSCALE_LO_SCALE = 0.5
N_SCALES = [0.5, 1.0, 2.0]


class OCR_block(nn.Module):
    """
    Some of the code in this class is borrowed from:
    https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR
    """

    def __init__(self, high_level_ch, num_classes):
        super(OCR_block, self).__init__()

        ocr_mid_channels = MID_CHANNELS
        ocr_key_channels = KEY_CHANNELS
        num_classes = num_classes

        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(
                high_level_ch, ocr_mid_channels, kernel_size=3, stride=1, padding=1
            ),
            BNReLU(ocr_mid_channels),
        )
        self.ocr_gather_head = SpatialGather_Module(num_classes)
        self.ocr_distri_head = SpatialOCR_Module(
            in_channels=ocr_mid_channels,
            key_channels=ocr_key_channels,
            out_channels=ocr_mid_channels,
            scale=1,
            dropout=0.05,
        )
        self.cls_head = nn.Conv2d(
            ocr_mid_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True
        )

        self.aux_head = nn.Sequential(
            nn.Conv2d(high_level_ch, high_level_ch, kernel_size=1, stride=1, padding=0),
            BNReLU(high_level_ch),
            nn.Conv2d(
                high_level_ch,
                num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

        if INIT_DECODER:
            initialize_weights(
                self.conv3x3_ocr,
                self.ocr_gather_head,
                self.ocr_distri_head,
                self.cls_head,
                self.aux_head,
            )

    def forward(self, high_level_features):
        feats = self.conv3x3_ocr(high_level_features)
        aux_out = self.aux_head(high_level_features)
        context = self.ocr_gather_head(feats, aux_out)
        ocr_feats = self.ocr_distri_head(feats, context)
        cls_out = self.cls_head(ocr_feats)
        return cls_out, aux_out, ocr_feats


class OCRNet(nn.Module):
    """
    OCR net
    """

    def __init__(self, num_classes, trunk="hrnetv2"):
        super(OCRNet, self).__init__()
        self.backbone, _, _, high_level_ch = get_trunk(trunk)
        self.ocr = OCR_block(high_level_ch, num_classes)

    def forward(self, inputs):
        x = inputs

        _, _, high_level_features = self.backbone(x)
        cls_out, aux_out, _ = self.ocr(high_level_features)
        aux_out = scale_as(aux_out, x)
        cls_out = scale_as(cls_out, x)

        output_dict = {"pred": cls_out, "aux": aux_out}
        return output_dict
        # return cls_out


def main():
    model = OCRNet(num_classes=1)
    x = torch.ones([2, 3, 320, 320])
    output_dict = model(x)
    print(output_dict['pred'].shape, output_dict['aux'].shape)


if __name__ == '__main__':
    main()