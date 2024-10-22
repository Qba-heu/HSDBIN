# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE：https://github.com/facebookresearch/mae
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch
import visdom

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched
from util import B_utils


def train_one_epoch(model: torch.nn.Module,classifier: torch.nn.Module, criterion: torch.nn.Module,emb_criterion: torch.nn.Module,
                    data_loader: Iterable, val_loader: Iterable,test_loader:Iterable,optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    classifier.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    Best_val = 0

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples_all, targets_all) in enumerate(metric_logger.log_every(zip(data_loader,test_loader), print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples, targets = samples_all
        test_sam, _ = targets_all

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        test_sam = test_sam.to(device, non_blocking=True)
        # center_data = center_X.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        labels = targets.clone()
        with torch.cuda.amp.autocast():
            P = classifier.polars[:, targets].T
            outputs = model(samples,training=True)
            feat = outputs[0]
            classifier.forward_momentum(feat.detach(), labels.detach())
            loss_sph_cls = B_utils.BLoss(emb_criterion, feat, P)

            output = classifier.predict(feat)
            loss_sph_reg = B_utils.hcr_loss(output, outputs[1], eps=1e-12)
            loss_cls = criterion(outputs[1], targets)


            loss = loss_cls+args.lambda1*loss_sph_cls+args.lambda2*loss_sph_reg



        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    val_acc = val(model,classifier,val_loader,'cuda')
    if val_acc>=Best_val:
        Best_val = val_acc
        torch.save(model,'Best_vit_model.pkl')
        torch.save(classifier, 'Best_classifier.pkl')

    print("Averaged stats:", metric_logger)
    print('Val acc:{:.04f}'.format(Best_val))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, classifier, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    classifier.eval()

    for batch in metric_logger.log_every(data_loader, 20, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            outputs = model(images)
            output = classifier.predict(outputs[0])
            loss = criterion(output[1], target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def val(net, classifier,data_loader, device='cpu', supervision='full'):
# TODO : fix me using metrics()
    accuracy, total = 0., 0.
    # ignored_labels = data_loader.dataset.ignored_labels
    for batch_idx, (data, target ) in enumerate(data_loader):
        with torch.no_grad():
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)
            if supervision == 'full':
                outputs = net(data)
                # output =classifier.predict(outputs[0])
                output = outputs[1]
                # output = (output1+outputs[1])/2
                # output = net_c(output_f)
            elif supervision == 'semi':
                outs = net(data)
                output, rec = outs
            _, output = torch.max(output, dim=1)
            #target = target - 1
            for pred, out in zip(output.view(-1), target.view(-1)):
                accuracy += out.item() == pred.item()
                total += 1
    return accuracy / total
