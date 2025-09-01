import argparse
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from models.build_sam import sam_model_registry
from models.SAM.MaskDecoder.decoders import Boundary_guided_Decoder_Swin
    

class SegAssess(nn.Module):
    def __init__(self,
                 args,
                 channel = 3,
                 decode_channels=64,
                 dropout=0.1,
                 window_size=8,
                 num_classes=1
                 ):
        super().__init__()
        args.mod = 'sam_all'
        args.mid_dim = 10
        cfg = args
        self.sam = sam_model_registry["vit_b"](cfg)
        self.image_encoder = self.sam.image_encoder
        self.prompt_encoder = self.sam.prompt_encoder
        self.mask_decoder = self.sam.mask_decoder
        # ASF module
        self.TCCA = Boundary_guided_Decoder_Swin(n_obj_class=args.class_num, n_out_class=4, in_dim=args.encoder_embed_dim)
        self.configure_gradients(args)

    def configure_gradients(self, args):
        """
        Configure the gradient flow based on the provided arguments.
        
        Arguments:
        args: argparse.Namespace object containing the configuration
                - args.update_image_encoder_all: If True, update all parameters in image_encoder.
                - args.update_mask_decoder: If True, update all parameters in mask_decoder.
                - args.mod: The mode for fine-tuning, e.g., 'sam_adpt' or 'sam_lora'.
        """
        train_fields = ['embed', 'rel_pos', 'Adapter', 'lora']
        for n, param in self.image_encoder.named_parameters():
                if args.mod == 'sam_adpt' and "Adapter" in n:
                    # print(n, "22")
                    param.requires_grad = True
                elif args.mod == 'sam_lora' and "lora" in n:
                    # print(n, "11")
                    param.requires_grad = True
                elif args.mod == 'sam_all':
                    pass
                else:
                    # print(n, "00")
                    param.requires_grad = False
                
                if args.vit_patch_size != 16 and any(field in n for field in ['embed']):
                    param.requires_grad = True
                if 'tok' in n:
                    param.requires_grad = True

        for n, param in self.prompt_encoder.named_parameters():
                if args.mod == 'decoder_only':
                    param.requires_grad = False
                

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def forward(self, x, dense_prompt=None):
        h, w = x.size()[-2:]

        image_embeddings, interm_embeddings = self.image_encoder(x)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None, boxes=None, masks=dense_prompt
        )

        masks_sam, masks_hq, low_res_masks = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            interm_embeddings=interm_embeddings[0],
            multimask_output=0,
        )

        masks_sam = self.postprocess_masks(
            masks_sam,
            input_size=(h, w),
            original_size=(h, w)
        )
        masks_hq = self.postprocess_masks(
            masks_hq,
            input_size=(h, w),
            original_size=(h, w)
        )
        masks = self.postprocess_masks(
            low_res_masks,
            input_size=(h, w),
            original_size=(h, w)
        )

        outs, edges = self.TCCA(interm_embeddings, masks, dense_prompt)
   
        return [masks_sam, masks_hq, masks, outs], edges



