import argparse
import logging
import math
import os
import random
import csv
import sys
from copy import deepcopy
from pathlib import Path
from threading import Thread
from datetime import datetime
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as functional
from torch.optim.lr_scheduler import LambdaLR
import yaml
from torch import autocast, float16, bfloat16, float32
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from tools.test import test as tester
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_img_size, \
    print_mutation, set_logging, one_cycle, colorstr
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss, ComputeLossOTA, ComputeLossAuxOTA
from utils.plots import plot_images, plot_results, plot_evolution
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel, \
    time_synchronized, smart_optimizer, save_model
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume
# from utils.autobatch import check_train_batch_size
from utils.re_parameteration import Re_parameterization

logger = logging.getLogger(__name__)


def train(hyp, opt, tb_writer=None,
          data_loader=None):
    # Save run settings
    if data_loader is None:
        data_loader = {'dataloader': None, 'dataset': None, 'val_dataloader': None, 'test_dataloader': None}
    save_dir, epochs, batch_size, total_batch_size, weights, rank, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank, opt.freeze
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'
    results_file_csv = save_dir / 'results.csv'
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    device, git_status = select_device(opt.device, batch_size=opt.batch_size)
    opt.git_status = git_status
    logger.info(colorstr('hyperparameters: ') +
                ', '.join(f'{k}={v}' for k, v in hyp.items()))
    # Directories
    tag_results = ('Epoch', 'GPU_mem', 'box', 'obj',
                   'cls', 'total', 'labels', 'img_size')
    if not os.path.exists(str(results_file_csv)):
        with open(results_file_csv, 'w') as f:
            csv_writer = csv.writer(f)
            a = tag_results + ('P', 'R', 'mAP@.5',
                               'mAP@.5:.95', 'x/lr0', 'x/lr1', 'x/lr2')
            csv_writer.writerow(a)

    # Configure
    plots = opt.evolve <= 1  # create plots
    cuda = True if device.type in ['cuda'] else False
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    is_coco = opt.data.endswith('coco.yaml')

    loggers = {'wandb': None}  # loggers dict
    map_device = 'cpu' if device.type in ['privateuseone', 'xla', 'cpu'] else device
    if rank in [-1, 0]:
        opt.hyp = hyp  # add hyperparameters
        run_id = torch.load(weights, map_location=map_device).get(
            'wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
        wandb_logger = WandbLogger(opt,
                                   Path(opt.save_dir).stem,
                                   run_id,
                                   data_dict)
        loggers['wandb'] = wandb_logger.wandb
        data_dict = wandb_logger.data_dict
        if wandb_logger.wandb:
            # WandbLogger might update weights, epochs if resuming
            weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp
    data_dict['nc'] = len(
        data_dict['names']) if 'nc' not in data_dict else data_dict['nc']
    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(
        data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (
        len(names), nc, opt.data)  # check

    # Model
    pretrained = weights.endswith('.pt')
    model_version = 0
    nodes, nodes2 = 0, 0
    if pretrained:
        with torch_distributed_zero_first(rank):
            attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=map_device)  # load checkpoint
        model = Model(opt.cfg or ckpt['model'].yaml,
                      ch=1 if opt.single_channel else 3,
                      nc=nc,
                      anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict,
                                     model.state_dict(),
                                     exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        ckpt['best_fitness'] = ckpt['best_fitness'] if 'best_fitness' in ckpt else 'unknown'
        ckpt['best_fitness'] = ckpt['best_fitness'].tolist()[0] if isinstance(ckpt['best_fitness'], np.ndarray) else \
            ckpt['best_fitness']
        model_version = ckpt['model_version'] if 'model_version' in ckpt else 0
        logger.info('Transferred %g/%g items from: %s, best fitness: %s, version: %s' % (
            len(state_dict), len(model.state_dict()), weights, ckpt['best_fitness'],
            model_version if (model_version != 0 and not opt.resume) else 'Init new model'))  # report
        nodes = len(ckpt['model'].yaml['head']) + len(ckpt['model'].yaml['backbone']) - 1
        p5_model = True if nodes in [77, 105, 121] else False
        nodes2 = len(model.yaml['head']) + len(model.yaml['backbone']) - 1
        assert nodes == nodes2, f'Please paste the same model branch,' \
                                f' pre-trained model from {"P5" if p5_model else "P6"} branch and new model ' \
                                f'from {"P5" if not p5_model else "P6"}'
    else:
        model = Model(opt.cfg, ch=1 if opt.single_channel else 3, nc=nc, anchors=hyp.get('anchors')).to(
            device)  # create
        nodes = len(model.yaml['head']) + len(model.yaml['backbone']) - 1

    p5_model = True if nodes in [77, 105, 121] else False
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # check
    train_path = data_dict['train']
    val_path = data_dict['val']
    test_path = data_dict['test'] if 'test' in data_dict else val_path
    test_path = test_path if os.path.exists(test_path) else val_path

    # Freeze
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(
        freeze[0]))]  # parameter names to freeze (full or partial)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False

    # Optimizer
    nbs = 64  # nominal batch size
    # accumulate loss before optimizing
    accumulate = max(round(nbs / total_batch_size), 1)
    hyp['weight_decay'] *= total_batch_size * \
                           accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")
    optimizer = smart_optimizer(model, opt.optimizer,
                                lr=hyp['lr0'],
                                decay=hyp['weight_decay'],
                                momentum=hyp['momentum'])

    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = LambdaLR(optimizer, lr_lambda=lf)
    ema = ModelEMA(model) if rank in [-1, 0] else None
    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Results
        if ckpt.get('training_results') is not None:
            results_file.write_text(
                ckpt['training_results'])  # write results.txt

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if opt.resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (
                weights, epochs)
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs
        del ckpt, state_dict, nodes, freeze

    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    # number of detection layers (used for scaling hyp['obj'])
    nl = model.model[-1].nl
    # verify imgsz are gs-multiples
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')
    opt.cache_images = [opt.cache_images[0]] * 2 if len(opt.cache_images) < 2 else [x for x in opt.cache_images]
    # Trainloader
    if data_loader['dataloader'] is None:
        dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                                hyp=hyp, augment=opt.augment,
                                                cache=opt.cache_images[0],
                                                rect=opt.rect, rank=rank,
                                                world_size=opt.world_size,
                                                workers=opt.workers,
                                                shuffle=False if opt.rect else True,
                                                seed=opt.seed,
                                                image_weights=opt.image_weights,
                                                quad=opt.quad,
                                                single_channel=opt.single_channel,
                                                prefix=colorstr('train: '))

        data_loader['dataloader'], data_loader['dataset'] = dataloader, dataset
    else:
        dataloader, dataset = data_loader['dataloader'], data_loader['dataset']

    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (
        mlc, nc, opt.data, nc - 1)

    # Process 0
    if rank in [-1, 0]:
        if data_loader['val_dataloader'] is None:
            val_dataloader = create_dataloader(val_path,
                                               imgsz_test,
                                               batch_size * 2, gs, opt,
                                               hyp=hyp,
                                               cache=opt.cache_images[1] if (
                                                       opt.cache_images[1] and not opt.notest) else 'no',
                                               rect=opt.rect,
                                               shuffle=False if opt.rect else True,
                                               rank=-1,
                                               world_size=opt.world_size,
                                               workers=opt.workers,
                                               single_channel=opt.single_channel,
                                               pad=0.5, prefix=colorstr('val: '))[0]
            data_loader['val_dataloader'] = val_dataloader
        else:
            val_dataloader = data_loader['val_dataloader']

        if test_path != val_path and os.path.exists(test_path):
            if data_loader['test_dataloader'] is None:
                test_dataloader = create_dataloader(test_path,
                                                    imgsz_test,
                                                    batch_size * 2, gs, opt,
                                                    hyp=hyp,
                                                    cache='no',
                                                    rect=opt.rect,
                                                    shuffle=False if opt.rect else True,
                                                    rank=-1,
                                                    world_size=opt.world_size,
                                                    workers=opt.workers,
                                                    single_channel=opt.single_channel,
                                                    pad=0.5, prefix=colorstr('test: '))[0]
                data_loader['test_dataloader'] = test_dataloader
            else:
                test_dataloader = data_loader['test_dataloader']
        else:
            logger.info(colorstr('Val and Test is the same thing or test path does not exists!'))

        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes
            if plots:
                # plot_labels(labels, names, save_dir, loggers)
                if tb_writer:
                    tb_writer.add_histogram('classes', c, 0)

            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model,
                              thr=hyp['anchor_t'], imgsz=imgsz, device='cpu')
            model.half().float() if device.type == 'xla' else model.to(device).half().float()

    # DDP mode
    if cuda and rank != -1:
        # nn.Multihead Attention incompatibility with DDP https://github.com/pytorch/pytorch/issues/26698
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank,
                    find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules()))
    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    # scale to image size and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).float().to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time_synchronized()
    # number of warmup iterations, max(3 epochs, 1k iterations)
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    results = (0, 0, 0, 0, 0, 0, 0)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = GradScaler(enabled=cuda)
    compute_loss_ota = ComputeLossOTA(model) if p5_model else ComputeLossAuxOTA(model)  # init loss class
    compute_loss = ComputeLoss(model)  # init loss class
    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...\n')
    torch.save(model, wdir / 'init.pt')
    # epoch ------------------------------------------------------------------
    for epoch in range(start_epoch, epochs):
        model.to(device).train(mode=True)
        # Update image weights (optional)
        if opt.image_weights:
            # Generate indices
            if rank in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(dataset.labels,
                                             nc=nc,
                                             class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n),
                                                 weights=iw,
                                                 k=dataset.n)  # rand weighted idx
            # Broadcast if DDP
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank ==
                                                            0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        mloss = torch.zeros(4, device=device)  # mean losses
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 8) % tag_results)
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb, mininterval=0.05, maxinterval=1, unit='batch')  # progress bar

        # batch -------------------------------------------------------------
        for i, (imgs, targets, paths, _) in pbar:
            # number integrated batches (since train start)
            ni = i + nb * epoch
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                accumulate = max(1, np.interp(
                    ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(
                        ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(
                            ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(
                    imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    # new shape (stretched to gs-multiple)
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]
                    imgs = functional.interpolate(
                        imgs, size=ns, mode='bilinear', align_corners=False, antialias=False)

            # Forward
            with autocast(enabled=True if device.type in ['cpu', 'cuda'] else False,
                          device_type='cuda' if device.type == 'cuda' else 'cpu'):
                optimizer.zero_grad()
                pred = model(imgs)  # forward
                if 'loss_ota' not in hyp or hyp['loss_ota'] == 1:
                    loss, loss_items = compute_loss_ota([pre.to(map_device) for pre in pred],
                                                        targets.to(map_device),
                                                        imgs.to(map_device))  # loss scaled by batch_size
                else:
                    loss, loss_items = compute_loss([pre.to(map_device) for pre in pred],
                                                    targets.to(map_device))  # loss scaled by batch_size
                if rank != -1:
                    loss *= opt.world_size  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items.to(device)) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % ('%g/%g' % (epoch,
                                                              epochs - 1), mem,
                                                   *mloss, targets.shape[0],
                                                   imgs.shape[-1])
                pbar.set_description(s)

                # Plot
                if plots and ni < 10:
                    f = save_dir / f'train_batch{ni}.jpg'  # filename
                    if tb_writer:
                        tb_writer.add_image(str(f), np.moveaxis(plot_images(
                            images=imgs, targets=targets, paths=paths, fname=f, names=names), -1, 0), ni)
                    Thread(target=plot_images, args=(
                        imgs,
                        targets,
                        paths, f), daemon=True).start()
                elif plots and ni == 10 and wandb_logger.wandb:
                    wandb_logger.log({"Mosaics": [wandb_logger.wandb.Image(str(x), caption=x.name) for x in
                                                  save_dir.glob('train*.jpg') if x.exists()]})

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            ema.update_attr(
                model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  # Calculate mAP
                wandb_logger.current_epoch = epoch + 1
                results, maps = tester(data_dict,
                                       batch_size=batch_size * 2,
                                       imgsz=imgsz_test,
                                       model=ema.ema,
                                       single_cls=opt.single_cls,
                                       dataloader=val_dataloader,
                                       save_dir=save_dir,
                                       verbose=False,
                                       plots=plots and final_epoch and device.type != 'privateuseone',
                                       wandb_logger=wandb_logger,
                                       compute_loss=compute_loss if device.type != 'privateuseone' else None,
                                       is_coco=is_coco,
                                       single_channel=opt.single_channel,
                                       v5_metric=opt.v5_metric)[:2]

                # Write
                with open(results_file, 'a') as f:
                    f.write(s + '%10.4g' * 7 % results + '\n')
                with open(results_file_csv, 'a') as f:
                    csv_writer = csv.writer(f)
                    a = []
                    for x in s.split(' '):
                        if x not in ['', ' ', '  ', '   ', '    ']:
                            a.append(x)
                    csv_writer.writerow(tuple(a) + results)

                if len(opt.name) and opt.bucket:
                    os.system('gsutil cp %s gs://%s/results/results%s.txt' %
                              (results_file, opt.bucket, opt.name))

                # Log
                tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                        'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                        'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                        'x/lr0', 'x/lr1', 'x/lr2']  # params
                if tb_writer or wandb_logger.wandb:
                    for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                        if tb_writer:
                            tb_writer.add_scalar(tag, x, epoch)  # tensorboard
                        if wandb_logger.wandb:
                            wandb_logger.log({tag: x})  # W&B

            # Update best mAP
            # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            fi = fitness(np.array(results).reshape(1, -1))
            if fi > best_fitness:
                best_fitness = fi
            wandb_logger.end_epoch(best_result=best_fitness == fi)

            # Save model
            if (not opt.nosave) or (final_epoch and opt.evolve < 1):  # if save
                ckpt = {
                    'model_version': model_version + 1 if not opt.resume else model_version,
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'training_results': results_file.read_text(),
                    'model': deepcopy(model.module if is_parallel(model) else model).half(),
                    'input_shape': list(imgs.shape[1:]) if isinstance(imgs.shape[1:], torch.Size) else imgs.shape[1:],
                    # BCWH to CWH
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None,
                    'hyp': hyp,
                    'git': git_status,
                    'training_opt': vars(opt),
                    'training_date': datetime.now().isoformat("#")
                }
                saver = Thread(target=save_model, args=(ckpt,
                                                        last,
                                                        best,
                                                        best_fitness,
                                                        fi,
                                                        epoch,
                                                        epochs,
                                                        wdir,
                                                        wandb_logger,
                                                        opt), daemon=True)
                saver.start()
                del ckpt
                if final_epoch:
                    saver.join()
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training
    if rank in [-1, 0]:
        # Plots
        if plots:
            plot_results(save_dir=save_dir)  # save as results.png
            if wandb_logger.wandb:
                files = ['results.png', 'confusion_matrix.png', *
                [f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
                wandb_logger.log({"Results": [wandb_logger.wandb.Image(str(save_dir / f), caption=f) for f in files
                                              if (save_dir / f).exists()]})
        logger.info('%g epochs completed in %.3f hours.\n' %
                    (epoch - start_epoch + 1, (time_synchronized() - t0) / 3600))

        if best.exists():
            prefix = colorstr('Validating')
            logger.info(f'{prefix} model {best}.')
            results_val = tester(opt.data,
                                 batch_size=batch_size * 2,
                                 imgsz=imgsz_test,
                                 conf_thres=0.001,
                                 iou_thres=0.7,
                                 weights=best,
                                 single_cls=opt.single_cls,
                                 single_channel=opt.single_channel,
                                 dataloader=val_dataloader,
                                 save_dir=save_dir,
                                 save_json=is_coco,
                                 verbose=True,
                                 plots=False,
                                 is_coco=is_coco,
                                 v5_metric=opt.v5_metric,
                                 device=opt.device,
                                 task='val',
                                 name=opt.name,
                                 project=opt.project,
                                 exist_ok=opt.exist_ok,
                                 trace=True)[0]

            prefix = colorstr('Testing')
            if opt.evolve < 1 and test_path != val_path:
                logger.info(f'{prefix} model {best}.')
                results_test = tester(opt.data,
                                      batch_size=batch_size * 2,
                                      imgsz=imgsz_test,
                                      conf_thres=0.001,
                                      iou_thres=0.7,
                                      weights=best,
                                      single_cls=opt.single_cls,
                                      dataloader=test_dataloader,
                                      single_channel=opt.single_channel,
                                      save_dir=save_dir,
                                      save_json=is_coco,
                                      verbose=True,
                                      plots=False,
                                      is_coco=is_coco,
                                      v5_metric=opt.v5_metric,
                                      device=opt.device,
                                      task='test',
                                      name=opt.name,
                                      project=opt.project,
                                      exist_ok=opt.exist_ok,
                                      trace=True)[0]

        final = best if best.exists() else last  # final model
        for f in last, best:
            if f.exists():
                strip_optimizer(f)
                strip_optimizer(f, str(f).replace(
                    'best.pt', 'striped.pt'), halfModel=True)
                if 'best.pt' in str(f):
                    output_path = str(f)
                    output_path = output_path.replace(
                        'best.pt', 'deploy_best.pt')
                    output_path = output_path.replace(
                        'last.pt', 'deploy_best.pt')
                    try:
                        if opt.evolve <= 1:
                            Re_parameterization(inputWeightPath=str(f),
                                                outputWeightPath=output_path,
                                                device=map_device)
                    except Exception as ex:
                        print(ex)
        if opt.bucket:
            os.system(f'gsutil cp {final} gs://{opt.bucket}/weights')  # upload
        if wandb_logger.wandb and opt.evolve <= 1:  # Log the stripped model
            wandb_logger.wandb.log_artifact(str(final), type='model',
                                            name='run_' + wandb_logger.wandb_run.id + '_model',
                                            aliases=['last', 'best', 'stripped'])
        wandb_logger.finish_run()
    else:
        dist.destroy_process_group()
    if opt.evolve <= 1:
        torch.cuda.empty_cache()

    return results, data_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='cfg/training/yolov7.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='mydataset.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.custom.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--augment', action='store_true', help='using augment for training')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--single-channel', action='store_true', help='single channel image training')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', type=int, default=-1, help='evolve hyperparameters')
    parser.add_argument('--parent', type=bool, default=True,
                        help='parent selection method: single or weighted, default: True (single)')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', type=str, nargs='+', default=['no', 'no'],
                        help='cache images for faster training [Train cache, Validation cache]')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW', 'Lion'], default='SGD',
                        help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=512, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0],
                        help='Freeze layers: backbone of yolov7=50, first3=0 1 2')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    opt = parser.parse_args()

    # Set DDP variables
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)
    opt.epochs = 300 if opt.epochs < 1 else opt.epochs
    # if opt.global_rank in [-1, 0]:
    #     check_git_status()
    #     check_requirements()

    # Resume
    wandb_run = check_wandb_resume(opt)
    if opt.resume and not wandb_run:  # resume an interrupted run
        # specified or most recent path
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(
                **yaml.load(f, Loader=yaml.SafeLoader))  # replace
        opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = '', ckpt, True, opt.total_batch_size, opt.global_rank, opt.local_rank  # reinstate
        logger.info('Resuming training from %s' % ckpt)
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(
            opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(
            opt.weights), 'either --cfg or --weights must be specified'
        # extend to 2 sizes (train, test)
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))
        opt.name = 'evolve' if opt.evolve > 1 else opt.name
        opt.save_dir = increment_path(Path(
            opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve > 1)  # increment run

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
            logger.info(
                f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
            try:
                tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
                logger.warning(f'{prefix}Init success')
            except:
                tb_writer = None
                logger.warning(f'{prefix}Init error')

        train(hyp, opt, tb_writer=tb_writer)
    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
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
                # anchors per output grid (0 to ignore)
                'anchors': (0, 3.0, 10.0),
                'fl_gamma': (1, 0.0, 2.0)}  # focal loss gamma (efficientDet default gamma=1.5)

        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3

        assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
        opt.notest, opt.nosave = True, True  # only test/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        yaml_file = Path(opt.save_dir) / \
                    'hyp_evolved.yaml'  # save best result here
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
                        try:
                            hyp[k] = float(x[i + 7] * v[i])  # mutate
                        except:
                            pass
            # Constrain to limits
            hyp_ = meta.copy()
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits
                hyp_[k] = hyp[k]

            # Train mutation
            results, data_loader = train(hyp.copy(), opt, data_loader=data_loader)

            # Write mutation results
            print_mutation(hyp_.copy(), results, yaml_file, opt.bucket)
        torch.cuda.empty_cache()
        # Plot results
        plot_evolution(yaml_file)
        logger.info(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
                    f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')
