import os
from models.loss import *
from models.SegAssess import SegAssess
from models.Segmentation.DeepLabv3plus.deeplabv3plus_resnet import resnet101_deeplabv3p as deeplabv3plus
from models.Segmentation.unetformer import UNetFormer
from models.Segmentation.hrnet import HRNet
from models.Segmentation.TransUnet.vit_seg_modeling import get_transunet
from models.Segmentation.OCRNet.OCRNet import OCRNet
from tools.setting_utils import load_ckpt

def to_gpu(args, model, mode='DP'):
    if mode == 'DP':
          model = nn.DataParallel(model)
          model = model.cuda()
    elif mode == 'DDP':
        if args.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model.cuda(),
                                                device_ids=[args.gpu],
                                                find_unused_parameters=True)
    else:
        pass

    return model

def set_parallel(args):
    if args.configs['Model']['detection_model']['name'] in ['SwinDiT', 'GL']:
        args.image_encoder = to_gpu(args, args.image_encoder, mode=args.dp_mode)

    args.detection_model = to_gpu(args, args.detection_model, mode=args.dp_mode)
    if args.configs['Experiment']['task'] == 'full':
        args.adj_model = to_gpu(args, args.adj_model, mode=args.dp_mode)
    if args.configs['Model']['MTL_loss']['flag']:
        args.MTL = to_gpu(args, args.MTL, mode=args.dp_mode)
    return args


def build_assess_model(args):
    args.dp_mode = 'DDP' if args.configs['DDP']['flag'] else 'DP'
        
    if 'sam' in args.configs['Model']['assess_model']['name']:
        args.assess_model = SegAssess(args=args)

    if args.configs['Model']['assess_model']['resume']:
        print('Loading Assess Model Weights......')
        detection_weight_path = args.configs['Paths']['pretrained_assess_path']
        args.assess_model = load_ckpt(detection_weight_path, args.assess_model, dp_mode=args.dp_mode)

    args.assess_model = to_gpu(args, args.assess_model, mode=args.dp_mode)

    args.optimizer = torch.optim.Adam(params=args.assess_model.parameters(),lr=args.configs['Model']['assess_model']['lr'])

    return args


def build_seg_model(args):
    args.dp_mode = 'DDP' if args.configs['DDP']['flag'] else 'DP'

    seg_names = args.configs['Model']['seg_model']['name']
    if 'deeplabv3+' in seg_names:
        print('Loading DeepLabV3+ Weights......')
        args.deeplabv3plus = deeplabv3plus(in_channel=3, n_class=args.class_num, output_stride=8, pretrained=False)
        args.deeplabv3plus = load_ckpt(args.configs['Paths']['pretrained_seg_path']['deeplabv3+'], args.deeplabv3plus,
                                       dp_mode=args.dp_mode)
        args.deeplabv3plus = to_gpu(args, args.deeplabv3plus, mode=args.dp_mode)
    if 'hrnet' in seg_names:
        print('Loading HRNet Weights......')
        args.hrnet = HRNet(num_classes=args.class_num,
                           pretrained='/data02/ybn/Projects/comparison/segmentation/hrnetv2_w48_imagenet_pretraine.pth')
        args.hrnet = load_ckpt(args.configs['Paths']['pretrained_seg_path']['hrnet'], args.hrnet, dp_mode=args.dp_mode)
        args.hrnet = to_gpu(args, args.hrnet, mode=args.dp_mode)
    if 'transunet' in seg_names:
        print('Loading TransUnet Weights......')
        args.transunet = get_transunet(in_channel=3, img_size=args.configs['Data']['image_size'],
                                       num_classes=args.class_num, pretrain=False)
        args.transunet = load_ckpt(args.configs['Paths']['pretrained_seg_path']['transunet'], args.transunet,
                                   dp_mode=args.dp_mode)
        args.transunet = to_gpu(args, args.transunet, mode=args.dp_mode)
    if 'unetformer' in seg_names:
        print('Loading UnetFormer Weights......')
        args.unetformer = UNetFormer(num_classes=args.class_num, pretrained=False)
        args.unetformer = load_ckpt(args.configs['Paths']['pretrained_seg_path']['unetformer'], args.unetformer,
                                    dp_mode=args.dp_mode)
        args.unetformer = to_gpu(args, args.unetformer, mode=args.dp_mode)
    if 'ocrnet' in seg_names:
        print('Loading OCRNet Weights......')
        args.ocrnet = OCRNet(num_classes=args.class_num)
        args.ocrnet = load_ckpt(args.configs['Paths']['pretrained_seg_path']['OCRNet'], args.ocrnet,
                                dp_mode=args.dp_mode)
        args.ocrnet = to_gpu(args, args.ocrnet, mode=args.dp_mode)

    return args