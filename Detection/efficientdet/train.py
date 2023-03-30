#!/usr/bin/env python
""" EfficientDet Training Script

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import os
import argparse
import time
import yaml
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import torch
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

from effdet import create_model, unwrap_bench, create_loader, create_dataset, create_evaluator
from effdet.data import resolve_input_config, SkipSubset
from effdet.anchors import Anchors, AnchorLabeler
from timm.models import resume_checkpoint, load_checkpoint
from timm.models.layers import set_layer_config
from timm.utils import *
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from embeddings_model import get_embeddings_model, LinearClassifier, get_embedding_classifier
from torchvision import transforms
import pickle as pk

torch.backends.cudnn.benchmark = True


# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# Dataset / Model parameters
parser.add_argument('root', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', default='coco', type=str, metavar='DATASET',
                    help='Name of dataset to train (default: "coco"')
parser.add_argument('--model', default='tf_efficientdet_d1', type=str, metavar='MODEL',
                    help='Name of model to train (default: "tf_efficientdet_d1"')
add_bool_arg(parser, 'redundant-bias', default=None, help='override model config for redundant bias')
add_bool_arg(parser, 'soft-nms', default=None, help='override model config for soft-nms')
parser.add_argument('--val-skip', type=int, default=0, metavar='N',
                    help='Skip every N validation samples.')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='Override num_classes in model config if set. For fine-tuning from pretrained.')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--no-pretrained-backbone', action='store_true', default=False,
                    help='Do not start with pretrained backbone weights, fully random.')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--fill-color', default=None, type=str, metavar='NAME',
                    help='Image augmentation fill (background) color ("mean" or int)')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--clip-grad', type=float, default=10.0, metavar='NORM',
                    help='Clip gradient norm (default: 10.0)')
parser.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')

# Optimizer parameters
parser.add_argument('--opt', default='momentum', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "momentum"')
parser.add_argument('--opt-eps', default=1e-3, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-3)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=4e-5,
                    help='weight decay (default: 0.00004)')

# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation parameters
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                    help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')

# loss
parser.add_argument('--smoothing', type=float, default=None, help='override model config label smoothing')
add_bool_arg(parser, 'jit-loss', default=None, help='override model config for torchscript jit loss fn')
add_bool_arg(parser, 'legacy-focal', default=None, help='override model config to use legacy focal loss')

# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
add_bool_arg(parser, 'bench-labeler', default=False,
             help='label targets in model bench, increases GPU load at expense of loader processes')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--eval-metric', default='map', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "map"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument("--local_rank", default=0, type=int)

# For posterior regularisation
parser.add_argument('--postreg', action='store_true', default=False,
                    help='trigger posterior regularisation')
parser.add_argument('--embedding_weights', type=str, default='INVALID', help='path to the weights of the embeddings model')
parser.add_argument('--embedding_pca_weights', type=str, default='INVALID', help='path to the weights of the embeddings pca model')


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def get_clip_parameters(model, exclude_head=False):
    if exclude_head:
        # FIXME this a bit of a quick and dirty hack to skip classifier head params
        return [p for n, p in model.named_parameters() if 'predict' not in n]
    else:
        return model.parameters()


def main():
    setup_default_logging()
    args, args_text = _parse_args()

    args.pretrained_backbone = not args.no_pretrained_backbone
    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
    assert args.rank >= 0

    if args.distributed:
        logging.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        logging.info('Training with a single process on 1 GPU.')

    use_amp = None
    if args.amp:
        # for backwards compat, `--amp` arg tries apex before native amp
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
        else:
            logging.warning("Neither APEX or native Torch AMP is available, using float32. "
                            "Install NVIDA apex or upgrade to PyTorch 1.6.")

    if args.native_amp:
        if has_native_amp:
            use_amp = 'native'
        else:
            logging.warning("Native AMP not available, using float32. Upgrade to PyTorch 1.6.")
    elif args.apex_amp:
        if has_apex:
            use_amp = 'apex'
        else:
            logging.warning("APEX AMP not available, using float32. Install NVIDA apex")

    random_seed(args.seed, args.rank)

    with set_layer_config(scriptable=args.torchscript):
        model = create_model(
            args.model,
            bench_task='train',
            num_classes=args.num_classes,
            pretrained=args.pretrained,
            pretrained_backbone=args.pretrained_backbone,
            redundant_bias=args.redundant_bias,
            label_smoothing=args.smoothing,
            legacy_focal=args.legacy_focal,
            jit_loss=args.jit_loss,
            soft_nms=args.soft_nms,
            bench_labeler=args.bench_labeler,
            checkpoint_path=args.initial_checkpoint,
        )
    model_config = model.config  # grab before we obscure with DP/DDP wrappers

    if args.local_rank == 0:
        logging.info('Model %s created, param count: %d' % (args.model, sum([m.numel() for m in model.parameters()])))

    model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.distributed and args.sync_bn:
        if has_apex and use_amp == 'apex':
            model = convert_syncbn_model(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.local_rank == 0:
            logging.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model, force native amp with `--native-amp` flag'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model. Use `--dist-bn reduce` instead of `--sync-bn`'
        model = torch.jit.script(model)

    optimizer = create_optimizer(args, model)

    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            logging.info('Using native Torch AMP. Training in mixed precision.')
    elif use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            logging.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    else:
        if args.local_rank == 0:
            logging.info('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            unwrap_bench(model), args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(model, decay=args.model_ema_decay)
        if args.resume:
            load_checkpoint(unwrap_bench(model_ema), args.resume, use_ema=True)

    if args.distributed:
        if has_apex and use_amp == 'apex':
            if args.local_rank == 0:
                logging.info("Using apex DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                logging.info("Using torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[args.device])
        # NOTE: EMA model does not need to be wrapped by DDP...
        if model_ema is not None and not args.resume:
            # ...but it is a good idea to sync EMA copy of weights
            # NOTE: ModelEma init could be moved after DDP wrapper if using PyTorch DDP, not Apex.
            model_ema.set(model)

    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        logging.info('Scheduled epochs: {}'.format(num_epochs))

    loader_train, loader_eval, evaluator = create_datasets_and_loaders(args, model_config)

    if model_config.num_classes < loader_train.dataset.parser.max_label:
        logging.error(
            f'Model {model_config.num_classes} has fewer classes than dataset {loader_train.dataset.parser.max_label}.')
        exit(1)
    if model_config.num_classes > loader_train.dataset.parser.max_label:
        logging.warning(
            f'Model {model_config.num_classes} has more classes than dataset {loader_train.dataset.parser.max_label}.')

    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = ''
    if args.local_rank == 0:
        output_base = args.output if args.output else './output'
        exp_name = '-'.join([
            datetime.now().strftime("%Y%m%d-%H%M%S"),
            args.model
        ])
        output_dir = get_outdir(output_base, 'train', exp_name)
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(
            model, optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
            checkpoint_dir=output_dir, decreasing=decreasing, unwrap_fn=unwrap_bench)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    # Initialize the embeddings model
    embed_model = None
    embed_transform = None
    pca_model = None
    embed_classifier = None
    if args.embedding_weights != 'INVALID':
        embed_model , _ = get_embeddings_model(args.embedding_weights)
        means = (0.485, 0.456, 0.406)
        stds = (0.229, 0.224, 0.225)
        embed_transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(means, stds)])

        if args.embedding_pca_weights != 'INVALID':
            pca_model = pk.load(open(args.embedding_pca_weights,'rb'))
            print("PCA model loaded")

        if args.embedding_classifier_weights != 'INVALID':
            embed_classifier = get_embedding_classifier(args.embedding_classifier_weights)
            print("Model Loaded")

    try:
        for epoch in range(start_epoch, num_epochs):
            if args.distributed:
                loader_train.sampler.set_epoch(epoch)

            train_metrics = train_epoch(
                epoch, model, loader_train, optimizer, args,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema,num_classes=args.num_classes,postreg=args.postreg,shape_model=embed_model, shape_transform=embed_transform, embeddings_pca_model=pca_model)

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.local_rank == 0:
                    logging.info("Distributing BatchNorm running means and vars")
                distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            # the overhead of evaluating with coco style datasets is fairly high, so just ema or non, not both
            if model_ema is not None:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')

                eval_metrics = validate(model_ema.module, loader_eval, args, evaluator, log_suffix=' (EMA)')
            else:
                eval_metrics = validate(model, loader_eval, args, evaluator)

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            if saver is not None:
                update_summary(
                    epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                    write_header=best_metric is None)

                # save proper checkpoint with eval metric
                best_metric, best_epoch = saver.save_checkpoint(epoch=epoch, metric=eval_metrics[eval_metric])

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        logging.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


def create_datasets_and_loaders(
        args,
        model_config,
        transform_train_fn=None,
        transform_eval_fn=None,
        collate_fn=None,
):
    """ Setup datasets, transforms, loaders, evaluator.

    Args:
        args: Command line args / config for training
        model_config: Model specific configuration dict / struct
        transform_train_fn: Override default image + annotation transforms (see note in loaders.py)
        transform_eval_fn: Override default image + annotation transforms (see note in loaders.py)
        collate_fn: Override default fast collate function

    Returns:
        Train loader, validation loader, evaluator
    """
    input_config = resolve_input_config(args, model_config=model_config)

    dataset_train, dataset_eval = create_dataset(args.dataset, args.root)

    # setup labeler in loader/collate_fn if not enabled in the model bench
    labeler = None
    if not args.bench_labeler:
        labeler = AnchorLabeler(
            Anchors.from_config(model_config), model_config.num_classes, match_threshold=0.5)

    loader_train = create_loader(
        dataset_train,
        input_size=input_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        # color_jitter=args.color_jitter,
        # auto_augment=args.aa,
        interpolation=args.train_interpolation or input_config['interpolation'],
        fill_color=input_config['fill_color'],
        mean=input_config['mean'],
        std=input_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        pin_mem=args.pin_mem,
        anchor_labeler=labeler,
        transform_fn=transform_train_fn,
        collate_fn=collate_fn,
    )

    if args.val_skip > 1:
        dataset_eval = SkipSubset(dataset_eval, args.val_skip)
    loader_eval = create_loader(
        dataset_eval,
        input_size=input_config['input_size'],
        batch_size=args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=input_config['interpolation'],
        fill_color=input_config['fill_color'],
        mean=input_config['mean'],
        std=input_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        pin_mem=args.pin_mem,
        anchor_labeler=labeler,
        transform_fn=transform_eval_fn,
        collate_fn=collate_fn,
    )

    evaluator = create_evaluator(args.dataset, loader_eval.dataset, distributed=args.distributed, pred_yxyx=False)

    return loader_train, loader_eval, evaluator

import numpy as np
import cv2
from sklearn import mixture
import torchvision

def gmm_kl(gmm_p, gmm_q, n_samples=10**5):
    if(abs(gmm_p.weights_[1]) < 1e-12 ):
        gmm_p.weights_[0] = 1.
        gmm_p.weights_[1] = 0.

    #print(gmm_p.weights_)
    X,y = gmm_p.sample(n_samples)
    log_p_X = gmm_p.score_samples(X)
    log_q_X = gmm_q.score_samples(X)
    return log_p_X.mean() - log_q_X.mean()

def calc_postreg_loss_gmm(train_sample, test_sample,gmm_comp):
    if(train_sample.shape[0] < 2):
        return 0
    g1 = mixture.GaussianMixture(n_components=2,random_state=0).fit(train_sample)
    g2 = mixture.GaussianMixture(n_components=2,random_state=0).fit(test_sample)

    if(abs(g1.weights_[0] - 1) < 1e-15):
        g1.weights_[0] = 1.
        g1.weights_[1] = 0.
    if(abs(g2.weights_[0] - 1) < 1e-15):
        g2.weights_[0] = 1.
        g2.weights_[1] = 0.

    return gmm_kl(g1,g2)

def flatten_list(_2d_list):
    '''
    convert list of list into list
    flatten_list
    '''
    flat_list = []

    for element in _2d_list:
        if type(element) is list:
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)

    return flat_list

def check_shape(target_shape, pred_shape):
    pred_shape_new = []
    for i, pred_list in enumerate(pred_shape):
        print(np.shape(pred_list), np.shape(target_shape[i]))
        if np.shape(pred_list) == np.shape(target_shape[i]):
            pred_shape_new.append(pred_list)

    return pred_shape_new

def train_epoch(
        epoch, model, loader, optimizer, args,
        lr_scheduler=None, saver=None, output_dir='', amp_autocast=suppress, loss_scaler=None, model_ema=None,num_classes=2,postreg=False,shape_model=None,shape_transform=None,embeddings_pca_model=None):

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.train()
    clip_params = get_clip_parameters(model, exclude_head='agc' in args.clip_mode)
    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)

        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        with amp_autocast():
            output = model(input, target)
        #print(output['loss'].item() , output['box_loss'].item() , output['class_loss'].item())
        loss = output['loss']

        #### Postreg starts #####
        #######################################
        if postreg:
            model.eval()
            with torch.no_grad():
                output = model(input,target)['detections']
            output = output.cpu().numpy()

            # initialise intensity lists
            target_intensity = [[] for _ in range(num_classes)]
            pred_intensity = [[] for _ in range(num_classes)]

            # initialise size lists
            target_size = [[] for _ in range(num_classes)]
            pred_size = [[] for _ in range(num_classes)]

            # initialise shape lists
            target_shape = [[] for _ in range(num_classes)]
            pred_shape = [[] for _ in range(num_classes)]

            target_shape_label = []
            pred_shape_label = []

            target['num_objects'] = target['num_objects'].cpu()
            target['cls'] = target['cls'].cpu()
            target['bbox'] = target['bbox'].cpu()

            for i in range(input.shape[0]):
                img = torchvision.transforms.ToPILImage()(input[i])
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

                intensity = [0 for _ in range(num_classes)]
                size = [0 for _ in range(num_classes)]
                shape = [np.zeros(512) for _ in range(num_classes)]
                iel_embed_list = []
                epith_embed_list = []
                num_each_class = [0 for _ in range(num_classes)]

                for j in range(target['num_objects'][i]):
                    cls = target['cls'][i][j] - 1
                    x1, y1, x2, y2 = int(target['bbox'][i][j][0]), int(target['bbox'][i][j][1]), int(target['bbox'][i][j][2]), int(target['bbox'][i][j][3])

                    pixel_sum = np.sum(img[y1:y2,x1:x2,2])
                    pixel_sum /=((x2 - x1 + 1)*(y2 - y1 + 1))

                    if shape_model is not None:
                        try:
                            box_img = cv2.cvtColor(img[y1:y2, x1:x2],cv2.COLOR_HSV2RGB)
                            box_img = shape_transform(box_img)
                            box_img = box_img.unsqueeze(0)
                            box_img = box_img.to(device)
                            embed = shape_model.encoder(box_img)
                            embed = embed.squeeze(0)
                            embed = embed.detach().cpu().numpy()
                            target_shape_label.append(cls)
                            # shape[int(labelsn[j][0])] += embed
                            # shape[j] = embed
                            if cls==1:
                                iel_embed_list.append(embed)
                            else:
                                epith_embed_list.append(embed)
                            box_img = box_img.detach().cpu()
                            del box_img
                        except:
                            pass

                    intensity[cls] += pixel_sum
                    size[cls] += (x2 - x1 + 1)*(y2 - y1 + 1)
                    num_each_class[cls] += 1

                iel_embed_array = np.array(iel_embed_list)
                epith_embed_array = np.array(epith_embed_list)
                if len(iel_embed_array)!=0:
                    iel_embed_array = np.mean(iel_embed_array, axis = 0)
                else:
                    iel_embed_array = np.zeros(512)
                if len(epith_embed_array)!=0:
                    epith_embed_array = np.mean(epith_embed_array, axis = 0)
                else:
                    epith_embed_array = np.zeros(512)

                shape[1] = iel_embed_array
                shape[0] = epith_embed_array
                    
                for k in range(num_classes):
                    if(num_each_class[k] != 0):
                        intensity[k] /= num_each_class[k]
                        size[k] /= num_each_class[k]

                    target_intensity[k].append(intensity[k])
                    target_size[k].append(size[k])
                    target_shape[k] = shape[k]

                intensity = [0 for _ in range(num_classes)]
                size = [0 for _ in range(num_classes)]
                num_each_class = [0 for _ in range(num_classes)]

                for j in range(output[i].shape[0]):
                    if float(output[i][j][4]) < 0.001:
                        continue
                    cls = int(output[i][j][5]) - 1
                    x1, y1, x2, y2 = int(output[i][j][0]), int(output[i][j][1]), int(output[i][j][2]), int(output[i][j][3])

                    pixel_sum = np.sum(img[y1:y2,x1:x2,2])
                    pixel_sum /=((x2 - x1 + 1)*(y2 - y1 + 1))

                    if shape_model is not None:
                        try:
                            box_img = cv2.cvtColor(img[y1:y2, x1:x2],cv2.COLOR_HSV2RGB)
                            box_img = shape_transform(box_img)
                            box_img = box_img.unsqueeze(0)
                            box_img = box_img.to(device)
                            embed = shape_model.encoder(box_img)
                            embed = embed.squeeze(0)
                            embed = embed.detach().cpu().numpy()
                            pred_shape_label.append(cls)
                            # shape[int(predn[j][5])] += embed
                            # shape[j] = embed
                            if cls==1:
                                iel_embed_list.append(embed)
                            else:
                                epith_embed_list.append(embed)
                            box_img = box_img.detach().cpu()
                            del box_img
                        except:
                            pass

                    intensity[cls] += pixel_sum
                    size[cls] += (x2 - x1 + 1)*(y2 - y1 + 1)
                    num_each_class[cls] += 1

                iel_embed_array = np.array(iel_embed_list)
                epith_embed_array = np.array(epith_embed_list)
                if len(iel_embed_array)!=0:
                    iel_embed_array = np.mean(iel_embed_array, axis = 0)
                else:
                    iel_embed_array = np.zeros(512)
                if len(epith_embed_array)!=0:
                    epith_embed_array = np.mean(epith_embed_array, axis = 0)
                else:
                    epith_embed_array = np.zeros(512)

                shape[1] = iel_embed_array
                shape[0] = epith_embed_array
                    
                for k in range(num_classes):
                    if(num_each_class[k] != 0):
                        intensity[k] /= num_each_class[k]
                        size[k] /= num_each_class[k]

                    pred_intensity[k].append(intensity[k])
                    pred_size[k].append(size[k])
                    pred_shape[k] = shape[k]

            # apply pca
            for i1 in range(num_classes):
                try:
                    target_shape[i1] = np.array(target_shape[i1])
                    pred_shape[i1] = np.array(pred_shape[i1])
                    target_transform = embeddings_pca_model.transform(target_shape[i1].reshape(1,-1))
                    target_shape[i1] = target_transform.flatten()
                    pred_transform = embeddings_pca_model.transform(pred_shape[i1].reshape(1,-1))
                    pred_shape[i1] = pred_transform.flatten()
                except:
                    print("pass")
                    pass

            num_pairs = 0
            lreg = torch.zeros(1)
            for i1 in range(num_classes):
                for i2 in range(i1+1,num_classes):
                    num_pairs += 1
                    target_intensity1 = [np.subtract(x1, x2) for (x1, x2) in zip(target_intensity[i1], target_intensity[i2])]
                    pred_intensity1 = [np.subtract(x1, x2) for (x1, x2) in zip(pred_intensity[i1], pred_intensity[i2])]

                    if len(target_intensity1) > 0 and len(pred_intensity1) > 0:
                        target_intensity1 = np.array(target_intensity1)
                        target_intensity1 = target_intensity1.reshape((target_intensity1.shape[0],1))

                        target_size1 = [np.subtract(x1, x2) for (x1, x2) in zip(target_size[i1], target_size[i2])]
                        target_size1 = np.array(target_size1)
                        target_size1 = target_size1.reshape((target_size1.shape[0],1))

                        pred_intensity1 = np.array(pred_intensity1)
                        pred_intensity1 = pred_intensity1.reshape((pred_intensity1.shape[0],1))

                        pred_size1 = [np.subtract(x1, x2) for (x1, x2) in zip(pred_size[i1], pred_size[i2])]
                        pred_size1 = np.array(pred_size1)
                        pred_size1 = pred_size1.reshape((pred_size1.shape[0],1))

                        target_shape1 = np.subtract(target_shape[i1] , target_shape[i2])
                        pred_shape1 = np.subtract(pred_shape[i1] , pred_shape[i2])

                        # print(target_shape1.shape, pred_shape1.shape)

                        pred_shape1 = pred_shape1.reshape((pred_shape1.shape[0],1))
                        target_shape1 = target_shape1.reshape((target_shape1.shape[0],1))

                        
                        print("Shape size intensity with loss 1 and loss 2")
                        # target_shape1 = np.subtract(target_shape[i1] , target_shape[i2])
                        # pred_shape1 = np.subtract(pred_shape[i1] , pred_shape[i2])

                        loss1 = calc_postreg_loss_gmm(np.concatenate((target_intensity1,target_size1)) , np.concatenate((pred_intensity1,pred_size1)), 2)
                        loss2 = calc_postreg_loss_gmm(target_shape1 , pred_shape1, 2)

                        print('Intensity loss : ' , loss1)
                        print('Embeddings loss :', loss2)
                        if loss2> 1000:
                            lreg += (loss1)
                        else:
                            lreg += (loss1 + 10*loss2)
                        
            lreg /= num_pairs
            print(loss, lreg[0])    
            #lreg *= gmm_weight
            loss += lreg[0]
            model.train()

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode, parameters=clip_params)
        else:
            loss.backward()
            if args.clip_grad is not None:
                dispatch_clip_grad(clip_params, value=args.clip_grad, mode=args.clip_mode)
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        num_updates += 1

        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))

            if args.local_rank == 0:
                logging.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])


def validate(model, loader, args, evaluator=None, log_suffix=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx

            output = model(input, target)
            loss = output['loss']

            if evaluator is not None:
                evaluator.add_predictions(output['detections'], target)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                logging.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m, loss=losses_m))

    metrics = OrderedDict([('loss', losses_m.avg)])
    if evaluator is not None:
        metrics['map'] = evaluator.evaluate()

    return metrics


if __name__ == '__main__':
    main()
