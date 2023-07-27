import argparse
import logging
import os
import random
import threading
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml

from tools.trainer import train, train_cls
from utils.general import increment_path, fitness, get_latest_run, check_file, print_mutation, set_logging, colorstr
from utils.plots import plot_evolution
from utils.torch_utils import time_synchronized
from utils.wandb_logging.wandb_utils import check_wandb_resume

# from utils.autobatch import check_train_batch_size

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='pretained model path')
    parser.add_argument('--cfg', type=str, default='',
                        help='model config path. if this rule is empty and --weights got an exists path -->train same model with more epochs')
    parser.add_argument('--data', type=str, default='', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.custom.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--augment', action='store_true', help='using augment for training')
    parser.add_argument('--batch_size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', type=int, default=-1, help='evolve hyperparameters')
    parser.add_argument('--parent', type=bool, default=True,
                        help='parent selection method: single or weighted, default: True (single)')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache_images', type=str, nargs='+', default=['no', 'no'],
                        help='cache images for faster training [Train cache, Validation cache]')
    parser.add_argument('--image_weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi_scale', action='store_true', help='vary img-size +/- 50%')
    parser.add_argument('--single_cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW', 'Lion'], default='SGD',
                        help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=512, help='maximum number of dataloader workers')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist_ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear_lr', action='store_true', help='linear LR')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0],
                        help='Freeze layers: backbone of yolov7=50, first3=0 1 2')
    parser.add_argument('--v5_metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--tensorboard', action='store_true', help='Start with Tensorboard')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile() API if available')
    parser.add_argument('--video_backend', default="pyav", type=str, help='torchvision video backend')


    opt = parser.parse_args()

    # Set DDP variables
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)
    opt.epochs = 300 if opt.epochs < 1 else opt.epochs

    # Resume
    wandb_run = check_wandb_resume(opt)
    if opt.resume and not wandb_run:  # resume an interrupted run
        # specified or most recent path
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()
        assert Path(ckpt).is_file(), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))  # replace
        opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = ('', ckpt, True,
                                                                                             opt.total_batch_size,
                                                                                             opt.global_rank,
                                                                                             opt.local_rank)
        logger.info('Resuming training from %s' % ckpt)
        data = str(torch.load(ckpt, map_location="cpu")["model"].yaml)


    else:
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(
            opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(
            opt.weights), 'either --cfg or --weights must be specified'
        # extend to 2 sizes (train, test)
        opt.imgsz.extend([opt.imgsz[-1]] * (2 - len(opt.imgsz)))
        opt.name = 'evolve' if opt.evolve > 1 else opt.name
        with open(opt.cfg, 'r') as fi:
            data = fi.read()
    project = "runs/train-cls" if "Classify" in data else "runs/train"
    if not opt.resume:
      opt.save_dir = increment_path(Path(project) / opt.name, exist_ok=opt.exist_ok | opt.evolve > 1)  # increment run
    opt.project = project
    # DDP mode
    opt.total_batch_size = opt.batch_size
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        opt.device = torch.device('cuda', opt.local_rank)
        # distributed backend
        dist.init_process_group(backend='nccl', init_method='env://')
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
    # Train
    logger.info(opt)
    if opt.evolve <= 1:
        tb_writer = None  # init loggers
        if opt.global_rank in [-1, 0]:
            prefix = colorstr('tensorboard: ')
            try:
                assert opt.tensorboard, 'not using Tensorboard'
                from torch.utils.tensorboard import SummaryWriter

                tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
                tensorboard_lauch = threading.Thread(
                    target=lambda: os.system(f'tensorboard --bind_all --logdir {project}'), daemon=True).start()
                logger.info(
                    f"{prefix}Starting...")
            except Exception as ex:
                tb_writer = None
                logger.warning(f'{prefix}Init error, {ex}')

        if "Classify" in data:
            train_cls(hyp, opt, tb_writer=tb_writer, logger=logger, use3D="Classify3D" in data)
        else:
            train(hyp, opt, tb_writer=tb_writer, logger=logger)
    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1 [0 to ignore], lower_limit, upper_limit)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                # final OneCycleLR learning rate (lr0 * lrf)
                'lrf': (1, 0.01, 1.0),
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (1, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (1, 3.0, 10.0),  # anchors != 3 make error with reparamater
                'fl_gamma': (1, 0.0, 2.0),
                'lost_ota': (1, 0, 1)}  # focal loss gamma (efficientDet default gamma=1.5)

        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3

        assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
        opt.notest, opt.nosave = True, True  # only test/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        yaml_file = Path(opt.save_dir) / 'hyp_evolved.yaml'  # save best result here
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' %
                      opt.bucket)  # download evolve.txt if exists
        data_loader = {'dataloader': None, 'dataset': None, 'val_dataloader': None, 'test_dataloader': None}
        for _ in range(opt.evolve):  # generations to evolve
            a = colorstr('Evolving: ')
            logger.info(
                f'{a}starting training for {_}th generation out of {opt.evolve} generations...\n')
            if Path('evolve.txt').exists():  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = opt.parent  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent or len(x) == 1:
                    x = x[random.choices(range(n), weights=w)[
                        0]]  # weighted selection
                else:
                    x = (x * w.reshape(n, 1)).sum(0) / \
                        w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time_synchronized()))
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng)
                         * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    if k in meta:
                        hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            hyp_ = meta.copy()
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits
                hyp_[k] = hyp[k]

            # Train mutation
            results, data_loader = train(hyp.copy(), opt, data_loader=data_loader, logger=logger)

            # Write mutation results
            print_mutation(hyp_.copy(), results, yaml_file, opt.bucket)
        torch.cuda.empty_cache()
        # Plot results
        plot_evolution(yaml_file)
        logger.info(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
                    f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')
