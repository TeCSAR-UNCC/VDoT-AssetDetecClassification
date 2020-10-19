#!/usr/bin/env python
""" COCO validation script

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import argparse
import os
import json
import time
import logging
import gc
import torch
import numpy as np
from ensemble_boxes import *
import torch.nn.parallel
try:
    from apex import amp
    has_amp = True
except ImportError:
    has_amp = False

from effdet import create_model
from data import create_loader, CocoDetection
from timm.utils import AverageMeter, setup_default_logging
from matplotlib import pyplot as plt
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from vdot import VdotDataset
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict
from effdet.efficientdet import HeadNet
torch.backends.cudnn.benchmark = True


def add_bool_arg(parser, name, default=False, help=''):  # FIXME move to utils
    dest_name = name.replace('-', '_')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=dest_name, action='store_true', help=help)
    group.add_argument('--no-' + name, dest=dest_name, action='store_false', help=help)
    parser.set_defaults(**{dest_name: default})


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--anno', default='val_annotations',
                    help='mscoco annotation set (one of val2017, train2017, test-dev2017)')
parser.add_argument('--model', '-m', metavar='MODEL', default='tf_efficientdet_d1',
                    help='model architecture (default: tf_efficientdet_d1)')
add_bool_arg(parser, 'redundant-bias', default=None,
                    help='override model config for redundant bias layers')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--num-classes', type=int, default=1, metavar='N',
                    help='Override num_classes in model config if set. For fine-tuning from pretrained.')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='bilinear', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--fill-color', default='mean', type=str, metavar='NAME',
                    help='Image augmentation fill (background) color ("mean" or int)')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='./vdot_checkpoint_adam.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--results', default='./results.json', type=str, metavar='FILENAME',
                    help='JSON filename for evaluation results')
'''def load_net(checkpoint_path):
        config = get_efficientdet_config('tf_efficientdet_d0')
        net = EfficientDet(config, pretrained_backbone=False)

        config.num_classes = 1
        config.image_size=512
        net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))

        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['state_dict'], strict=False)

        del checkpoint
        gc.collect()

        net = DetBenchPredict(net, config)
        net.eval()
        return net.cuda()'''


def validate(args):
    setup_default_logging()
    
   
    # might as well try to validate something
    args.pretrained = args.pretrained or not args.checkpoint
    args.prefetcher = not args.no_prefetcher
    # create model

    bench = create_model(
        args.model,
        bench_task='predict',
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        redundant_bias=args.redundant_bias,
        checkpoint_path=args.checkpoint,
        checkpoint_ema=args.use_ema,
    )
    model_config = bench.config
    input_size = bench.config.image_size

    param_count = sum([m.numel() for m in bench.parameters()])
    print('Model %s created, param count: %d' % (args.model, param_count))

    bench = bench.cuda()
    if has_amp:
        print('Using AMP mixed precision.')
        bench = amp.initialize(bench, opt_level='O1')
    else:
        print('AMP not installed, running network in FP32.')

    if args.num_gpu > 1:
        bench = torch.nn.DataParallel(bench, device_ids=list(range(args.num_gpu)))
    

    if 'test' in args.anno:
        annotation_path = os.path.join(args.data, 'annotations', f'image_info_{args.anno}.json')
        image_dir = 'test2017'
    else:
        annotation_path = os.path.join(args.data, 'val_annotations', f'{args.anno}.json')
        image_dir = os.path.join(args.data, 'val_set')
    dataset = VdotDataset(image_dir, annotation_path)
    
    loader = create_loader(
        dataset,
        input_size=input_size,
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=args.interpolation,
        fill_color=args.fill_color,
        num_workers=args.workers,
        pin_mem=args.pin_mem)
    
    img_ids = []
    results = []
    bench.eval()
    batch_time = AverageMeter()
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            #output = bench(input, target['img_scale'], target['img_size'])
            output = bench(input, img_info=target)
            output = output.cpu()
            sample_ids = target['img_id'].cpu()
            for index, sample in enumerate(output):
                image_id = int(sample_ids[index])
                for det in sample:
                    score = float(det[4])
                    if score < .001:  # stop when below this threshold, scores in descending order
                        break
                    coco_det = dict(
                        image_id=image_id,
                        bbox=det[0:4].tolist(),
                        score=score,
                        category_id=int(det[5]))
                    img_ids.append(image_id)
                    results.append(coco_det)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.log_freq == 0:
                print(
                    'Test: [{0:>4d}/{1}]  '
                    'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    .format(
                        i, len(loader), batch_time=batch_time,
                        rate_avg=input.size(0) / batch_time.avg,
                     )
                )

    json.dump(results, open(args.results, 'w'), indent=4)
    '''if 'test' not in args.anno:
        #coco_results = dataset.coco.loadRes(args.results)
        file_res = open('./results.json')
        coco_results = json.load(file_res)
        coco_eval = COCOeval(dataset.coco, coco_results, 'bbox')
        coco_eval.params.imgIds = img_ids  # score only ids we've used
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()'''
    return results
    
    '''def run_wbf(predictions, image_index, image_size=512, iou_thr=0.44, skip_box_thr=0.43, weights=None):
        boxes = [(prediction[image_index]['bbox']/(image_size-1)).tolist()  for prediction in results]
        scores = [prediction[image_index]['score'].tolist()  for prediction in results]
        labels = [np.ones(prediction[image_index]['score'].shape[0]).tolist() for prediction in results]
        boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        boxes = boxes*(image_size-1)
        return boxes, scores, labels
    
    sample = input[0].permute(1,2,0).cpu().numpy()

    boxes, scores, labels = run_wbf(results, image_index=1)
    boxes = boxes.astype(np.int32).clip(min=0, max=511)

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    for box in boxes:
        cv2.rectangle(sample, (box[0], box[1]), (box[2], box[3]), (1, 0, 0), 1)

    ax.set_axis_off()
    ax.imshow(sample)'''
    
    '''for j, (input, target['img_id']) in enumerate(loader):
        break
    predictions= validate(input)
    #predictions = make_predictions(images)'''
    

    '''def make_predictions(images, score_threshold=0.22):
        images = torch.stack(images).cuda().float()
        predictions = []
        with torch.no_grad():
            det = net(images, torch.tensor([1]*images.shape[0]).float().cuda())
            for i in range(images.shape[0]):
                boxes = det[i].detach().cpu().numpy()[:,:4]    
            scores = det[i].detach().cpu().numpy()[:,4]
            indexes = np.where(scores > score_threshold)[0]
            boxes = boxes[indexes]
            #boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
            #boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
            predictions.append({
                'boxes': boxes[indexes],
                'scores': scores[indexes],
            })
    return [predictions]'''



def main():
    args = parser.parse_args()
    validate(args)

if __name__ == '__main__':
    main()


    
    '''if 'test' not in args.anno:
        coco_results = dataset.coco.loadRes(args.results)
        coco_eval = COCOeval(dataset.coco, coco_results, 'bbox')
        coco_eval.params.imgIds = img_ids  # score only ids we've used
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()'''

    
    
