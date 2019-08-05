from pathlib import Path
import math
import sys
import time

from PIL import Image
import torch
import torchvision.models.detection.mask_rcnn

from . import utils
from ..viz import visualize_boxes


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter='  ')
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(
            optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(
            data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print('Loss is {}, stopping training'.format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ['bbox']
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append('segm')
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append('keypoints')
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device, output_dir, threshold=0.5):
    cpu_device = torch.device('cpu')
    model.eval()
    metric_logger = utils.MetricLogger(delimiter='  ')
    header = 'Test:'

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        evaluator_time = time.time()
        for target, image, output in zip(targets, images, outputs):
            boxes = output['boxes'][output['scores'] >= threshold].clone()
            # convert from pytorch detection format
            boxes[:, 2] -= boxes[:, 0]
            boxes[:, 3] -= boxes[:, 1]
            item = data_loader.dataset.df.iloc[target['idx'].item()]
            if output_dir:
                _save_predictions(image, boxes,
                                  output_dir / f'{item.image_id}.jpg')

        evaluator_time = time.time() - evaluator_time
        metric_logger.update(
            model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)


def _save_predictions(image, boxes, path: Path):
    image = image.detach().cpu().clone()
    image_std = [0.229, 0.224, 0.225]
    image_mean = [0.485, 0.456, 0.406]
    for i, (mean, std) in enumerate(zip(image_mean, image_std)):
        image[i] = image[i] * std + mean
    image = (image * 255).to(torch.uint8)
    image = visualize_boxes(image, boxes)
    Image.fromarray(image).save(path)
