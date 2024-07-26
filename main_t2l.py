# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import seaborn as sns
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm,visdom

assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util import fixed
from thop import profile
import models_vit


from engine_d2l import train_one_epoch, evaluate
import dataload
from dataload import convert_to_color_,display_predictions,LMMDLoss

def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--pretrain', type=bool, default=False,
                        help="set the pretrain of model")
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--alpha', type=float, default=0.9, metavar='PCT',
                        help='Trade-off param of 1D and ViT (default: 0.9)')
    parser.add_argument('--lambda1', type=float, default=0.1, metavar='PCT',
                        help='Weight of loss1 (default: 0.1)')
    parser.add_argument('--lambda2', type=float, default=0.1, metavar='PCT',
                        help='Weight of loss2 (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.2,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='/Datasets/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=16, type=int,
                        help='number of the classification types')
    parser.add_argument('--space_dim', default=192, type=int,
                        help='number of the classification types')
    parser.add_argument('--disjoint', type=bool, default=False,
                             help="Training by the disjoint train set.")

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--centroid_path', default='./Estimated_prototypes/19centers_192dim.pth',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    #hsi data parameters
    parser.add_argument('--source_HSI', help='source', type=str, default='houston2013')
    parser.add_argument('--folder', type=str, help="Folder where to store the "
                                                   "datasets (defaults to the current working directory).",
                        default="./Datasets/")
    parser.add_argument('--training_sample', type=float, default=0.1,
                        help="Percentage of samples to use for training (random sampling rule)")
    parser.add_argument('--runs', type=int, default=1,
                        help="Number of experiments")
    parser.add_argument('--patch_size', type=int,
                        help="Size of the spatial neighbourhood (optional, if "
                             "absent will be set by the model)")
    parser.add_argument('--spectral_fusion', type=bool, default=False,
                        help="H*W*C to n*n*4 for ViT, ResNet,Swin_transformer of Swin_mlp.")
    parser.add_argument('--flip_augmentation', action='store_true',
                        help="Random flips (if patch_size > 1)")
    parser.add_argument('--radiation_augmentation', action='store_true',
                        help="Random radiation noise (illumination)")
    parser.add_argument('--mixture_augmentation', action='store_true',
                        help="Random mixes between spectra")

    return parser


def main(args):
    misc.init_distributed_mode(args)

    nDataSet = args.runs
    acc = np.zeros([nDataSet, 1])
    if args.source_HSI=='houston2013':
        N_CLASS = 16
    elif args.source_HSI == 'PaviaU':
        N_CLASS = 10
    elif args.source_HSI == 'IndianPines':
        N_CLASS = 17
    elif args.source_HSI == 'YRE':
        N_CLASS = 21
    elif args.source_HSI == 'YC':
        N_CLASS = 19
    elif args.source_HSI == 'Huanghe_obt':
        N_CLASS = 18
    elif args.source_HSI == 'XiongAn':
        N_CLASS = 21
    else:
        raise ValueError('{} dataset need you add it'.format(args.source_HSI))
    A = np.zeros([nDataSet, N_CLASS])
    KA = np.zeros([nDataSet, 1])
    AA = np.zeros([nDataSet, 1])
    best_predict_all = []


    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    viz = visdom.Visdom(env=args.source_HSI +''+args.model)
    palette = {0: (0, 0, 0)}
    for k, color in enumerate(sns.color_palette("hls", N_CLASS-1)):
        if k == 9:
            palette[k + 1] = tuple(np.asarray([255, 255, 255], dtype='uint8'))
        else:
            palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))

    seeds = [1234, 1240, 1223, 1236, 1326]

    for iter_num in range(nDataSet):
        # fix the seed for reproducibility
        # seed = args.seed + misc.get_rank()
        torch.manual_seed(seeds[iter_num])
        np.random.seed(seeds[iter_num])

        cudnn.benchmark = True


        Train_datasets,vol_Datasets,gt,test_gt,val_dataset, img = dataload.HSI_dataloder(args)
        args.nb_classes=int(np.max(vol_Datasets.label)+1)


        H,W,BAND=Train_datasets.data.shape
        data_loader_train = torch.utils.data.DataLoader(
            Train_datasets,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

        data_loader_val = torch.utils.data.DataLoader(
            vol_Datasets,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

        val_loader_val = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )



        mixup_fn = None
        mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        if mixup_active:
            print("Mixup is activated!")
            mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing, num_classes=args.nb_classes)

        model = models_vit.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
            img_size = args.patch_size,
            patch_size = 2,
            in_chans = BAND,
            alpha = args.alpha
        )

        print('########   Using a fixed hyperspherical classifier with DL2PA  ##########')
        classifier = getattr(fixed, 'fixed_Classifier')(feat_in=args.space_dim, num_classes=args.nb_classes,
                                                        centroid_path=args.centroid_path)




        if args.finetune and not args.eval:
            checkpoint = torch.load(args.finetune, map_location='cpu')

            print("Load pre-trained checkpoint from: %s" % args.finetune)
            checkpoint_model = checkpoint['model']
            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            # interpolate position embedding
            interpolate_pos_embed(model, checkpoint_model)

            # load pre-trained model
            msg = model.load_state_dict(checkpoint_model, strict=False)
            print(msg)

            if args.global_pool:
                assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
            else:
                assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

            # manually initialize fc layer
            trunc_normal_(model.head.weight, std=2e-5)

        model.to(device)
        classifier.to(device)
        # algthom_SAGM.to(device)
        # input_tensor = torch.rand(1, BAND, 13, 13)
        # input_tensor = input_tensor.to(device)
        # FLODS, PARAMETER = profile(model, inputs=(input_tensor,))
        # print("FLODS of model are:", FLODS, PARAMETER)
        # quit()

        model_without_ddp = model
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("Model = %s" % str(model_without_ddp))
        print('number of params (M): %.2f' % (n_parameters / 1.e6))

        eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

        if args.lr is None:  # only base_lr is specified
            args.lr = args.blr * eff_batch_size / 256

        print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        print("actual lr: %.2e" % args.lr)

        print("accumulate grad iterations: %d" % args.accum_iter)
        print("effective batch size: %d" % eff_batch_size)

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module
            classfier_without_ddp = classifier.module

        # build optimizer with layer-wise lr decay (lrd)
        param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
            no_weight_decay_list=model_without_ddp.no_weight_decay(),
            layer_decay=args.layer_decay
        )
        param_groups_cls = lrd.param_groups_fix(classifier, args.weight_decay,
                                            no_weight_decay_list=model_without_ddp.no_weight_decay(),
                                            layer_decay=args.layer_decay
                                            )
        para_all = param_groups+param_groups_cls
        optimizer = torch.optim.AdamW(para_all, lr=args.lr)
        loss_scaler = NativeScaler()

        if mixup_fn is not None:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif args.smoothing > 0.:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()


        Iter_num_epoch = len(data_loader_train)
        emb_criterion = torch.nn.CosineSimilarity(eps=1e-9)
        print("criterion = %s" % str(criterion))

        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

        if args.eval:
            test_stats = evaluate(data_loader_val, model, device)
            print(f"Accuracy of the network on the {len(vol_Datasets)} test images: {test_stats['acc1']:.1f}%")
            exit(0)

        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()
        max_accuracy = 0.0
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)
            start_epoch = time.time()
            train_stats = train_one_epoch(
                model,classifier, criterion, emb_criterion,data_loader_train,val_loader_val,data_loader_val,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad, mixup_fn,
                #log_writer=log_writer,
                args=args
            )
            end_epoch = time.time()
            print(end_epoch-start_epoch)


            if epoch+1== args.epochs:
                Best_model = torch.load('Best_vit_model.pkl')
                Best_classifier = torch.load('Best_classifier.pkl')
                # probabilities = dataload.test(Best_model, Best_classifier,img, vars(args))
                probabilities = dataload.test(Best_model, img, vars(args))



                # sum_value = np.sum(probabilities, axis=-1)

                prediction = np.argmax(probabilities, axis=-1)
                # np.save('mas_results.npy',prediction)
                run_results = dataload.metrics(prediction,test_gt,args.ignored_labels,
                                      n_classes=args.nb_classes)
                for label_out in run_results:
                    if label_out=='class_acc':
                        for label, score in zip(args.label_values, run_results["class_acc"]):
                            print("\t{}: {:.03f}\n".format(label, 100*score))
                    else:
                        # print(label_out,':',run_results[label_out],"\n")
                        print("{}: {}\n".format(label_out, run_results[label_out]))
                total_time = time.time() - start_time
                total_time_str = str(datetime.timedelta(seconds=int(total_time)))
                acc[iter_num] = run_results['Accuracy']
                A[iter_num, :] = run_results['class_acc']
                KA[iter_num] = run_results['Kappa']
                AA[iter_num] = run_results['average_acc']


                color_prediction = convert_to_color_(prediction,palette=palette)

                display_predictions(color_prediction, viz,
                                    caption="Prediction vs. test ground truth")

                print("****************{}**************".format(iter_num))



    OAmean = np.mean(acc)
    OAStd = np.std(acc)
    Amean = np.mean(A, 0)
    AStd = np.std(A, 0)
    AAmean = np.mean(AA)
    AAstd = np.std(AA)
    kappa_mean = np.mean(KA)
    kappa_std = np.std(KA)
    print("Average OA:" + "{:.2f}".format(OAmean) + "+-" + "{:.2f}".format(OAStd))
    print("Average AA:" + "{:.2f}".format(100*AAmean) + "+-" + "{:.2f}".format(100*AAstd))
    print("Average KAPPA:" + "{:.2f}".format(kappa_mean) + "+-" + "{:.2f}".format(kappa_std))
    i = 0
    for label in args.label_values:
        print("\t{}: {:.03f}".format(label, 100*Amean[i])+"+-"+"{:.2f}".format(100*AStd[i]))
        i+=1
    # for i in range(args.nb_class):
    #     print("class" + str(i) + ":" + "{:.2f}".format(Amean[i]) + "+-" + "{:.2f}".format(AStd[i]))
    print("\t All OA is:",acc)
    print('Training time {}'.format(total_time_str))
    with open('HSDBIN_out_'+args.source_HSI+'.log', 'a') as f:
        f.write("\n")
        # f.write("train time per DataSet(s): " + "{:.5f}".format(train_end - train_start) + "\n")
        # f.write("test time per DataSet(s): " + "{:.5f}".format(test_end - train_end) + "\n")
        f.write("lambda1:{}, lambda2:{}".format(args.lambda1,args.lambda2))
        f.write("\n")
        f.write("patch size:{}".format(args.patch_size))
        f.write("\n")
        f.write("average OA: " + "{:.2f}".format(OAmean) + " +- " + "{:.2f}".format(OAStd) + "\n")
        f.write("average AA: " + "{:.2f}".format(100 * AAmean) + " +- " + "{:.2f}".format(100 * AAstd) + "\n")
        f.write("average kappa: " + "{:.4f}".format(100 * kappa_mean) + " +- " + "{:.4f}".format(100 * kappa_std) + "\n")
        f.write("accuracy for each class: " + "\n")
        f.write("accuracy for each class: " + "\n")
        f.write("accuracy for each class: " + "\n")
        i = 0
        for label in args.label_values:
            f.write("\t{}: {:.03f}".format(label, 100*Amean[i])+"+-"+"{:.2f}".format(100*AStd[i]))
            f.write("\n")
            i += 1
        f.write(str(acc))
        curret_tiem = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write("\n")
        f.write(curret_tiem)
        f.write("\n")

    f.close()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
