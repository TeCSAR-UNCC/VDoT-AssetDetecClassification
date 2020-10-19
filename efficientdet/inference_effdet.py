import gc
import torch
import numpy as np
import pandas as pd
from glob import glob
from torch.utils.data import Dataset,DataLoader

import cv2
import gc
from matplotlib import pyplot as plt
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict
from effdet.efficientdet import HeadNet
def load_net(checkpoint_path):
    config = get_efficientdet_config('tf_efficientdet_d0')
    net = EfficientDet(config, pretrained_backbone=True)

    config.num_classes = 1
    config.image_size=512
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))

    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['state_dict'])

    del checkpoint
    gc.collect()

    net = DetBenchPredict(net, config)
    net.eval() 
    return net.cuda()

net = load_net('vdot_checkpoint_adam.pth')



"model.backbone.conv_stem.weight", "model.backbone.bn1.weight", "model.backbone.bn1.bias", "model.backbone.bn1.running_mean", "model.backbone.bn1.running_var", "model.backbone.bn1.num_batches_tracked", "model.backbone.blocks.0.0.conv_dw.weight", "model.backbone.blocks.0.0.bn1.weight", "model.backbone.blocks.0.0.bn1.bias", "model.backbone.blocks.0.0.bn1.running_mean", "model.backbone.blocks.0.0.bn1.running_var", "model.backbone.blocks.0.0.bn1.num_batches_tracked", "model.backbone.blocks.0.0.se.conv_reduce.weight", "model.backbone.blocks.0.0.se.conv_reduce.bias", "model.backbone.blocks.0.0.se.conv_expand.weight", "model.backbone.blocks.0.0.se.conv_expand.bias", "model.backbone.blocks.0.0.conv_pw.weight", "model.backbone.blocks.0.0.bn2.weight", "model.backbone.blocks.0.0.bn2.bias", "model.backbone.blocks.0.0.bn2.running_mean", "model.backbone.blocks.0.0.bn2.running_var", "model.backbone.blocks.0.0.bn2.num_batches_tracked", "model.backbone.blocks.1.0.conv_pw.weight", "model.backbone.blocks.1.0.bn1.weight", "model.backbone.blocks.1.0.bn1.bias", "model.backbone.blocks.1.0.bn1.running_mean", "model.backbone.blocks.1.0.bn1.running_var", "model.backbone.blocks.1.0.bn1.num_batches_tracked", "model.backbone.blocks.1.0.conv_dw.weight", "model.backbone.blocks.1.0.bn2.weight", "model.backbone.blocks.1.0.bn2.bias", "model.backbone.blocks.1.0.bn2.running_mean", "model.backbone.blocks.1.0.bn2.running_var", "model.backbone.blocks.1.0.bn2.num_batches_tracked", "model.backbone.blocks.1.0.se.conv_reduce.weight", "model.backbone.blocks.1.0.se.conv_reduce.bias", "model.backbone.blocks.1.0.se.conv_expand.weight", "model.backbone.blocks.1.0.se.conv_expand.bias", "model.backbone.blocks.1.0.conv_pwl.weight", "model.backbone.blocks.1.0.bn3.weight", "model.backbone.blocks.1.0.bn3.bi


backbone.conv_stem.weight", "backbone.bn1.weight", "backbone.bn1.bias", "backbone.bn1.running_mean", "backbone.bn1.running_var", "backbone.blocks.0.0.conv_dw.weight", "backbone.blocks.0.0.bn1.weight", "backbone.blocks.0.0.bn1.bias", "backbone.blocks.0.0.bn1.running_mean", "backbone.blocks.0.0.bn1.running_var", "backbone.blocks.0.0.se.conv_reduce.weight", "backbone.blocks.0.0.se.conv_reduce.bias", "backbone.blocks.0.0.se.conv_expand.weight", "backbone.blocks.0.0.se.conv_expand.bias", "backbone.blocks.0.0.conv_pw.weight", "backbone.blocks.0.0.bn2.weight", "backbone.blocks.0.0.bn2.bias", "backbone.blocks.0.0.bn2.running_mean", "backbone.blocks.0.0.bn2.running_var", "backbone.blocks.1.0.conv_pw.weight", "backbone.blocks.1.0.bn1.weight", "backbone.blocks.1.0.bn1.bias", "backbone.blocks.1.0.bn1.running_mean", "backbone.blocks.1.0.bn1.running_var", "backbone.blocks.1.0.conv_dw.weight", "backbone.blocks.1.0.bn2.weight", "backbone.blocks.1.0.bn2.bias", "backbone.blocks.1.0.bn2.running_mean", "backbone.blocks.1.0.bn2.running_var", "backbone.blocks.1.0.se.conv_reduce.weight", "backbone.blocks.1.0.se.conv_reduce.bias", "backbone.blocks.1.0.se.conv_expand.weight", "backbone.blocks.1.0.se.conv_expand.bias", "backbone.blocks.1.0.conv_pwl.weight", "backbone.blocks.1.0.bn3.weight", "backbone.blocks.1.0.bn3.bi