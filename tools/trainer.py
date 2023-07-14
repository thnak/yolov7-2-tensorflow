import csv
import math
import os
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from torch import nn as nn, distributed as dist, autocast
from torch.cuda.amp import GradScaler
from torch.nn import functional as functional
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from tools.tester import test, cls_test
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader, create_dataloader_cls

from utils.general import (colorstr, init_seeds, check_dataset, check_img_size, one_cycle, labels_to_class_weights,
                           labels_to_image_weights, TQDM_BAR_FORMAT, strip_optimizer, parse_path)
from utils.google_utils import attempt_download
from utils.loss import SmartLoss
from utils.metrics import fitness
from utils.plots import plot_images, plot_results, plot_dataset, plotSample
from utils.re_parameteration import Re_parameterization
from utils.torch_utils import (select_device, torch_distributed_zero_first, intersect_dicts, smart_optimizer, ModelEMA,
                               time_synchronized, is_parallel, save_model)
from utils.wandb_logging.wandb_utils import WandbLogger


def train_cls(hyp, opt, tb_writer=None, data_loader=None, logger=None, use3D=False):
    if data_loader is None:
        data_loader = {'dataloader': None, 'dataset': None, 'val_dataloader': None, 'test_dataloader': None}
    save_dir, epochs, batch_size, total_batch_size, weights, rank, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank, opt.freeze
    if use3D:
        from models.commond3D import Model3D as Model
        from utils.datasets import LoadSampleforVideoClassify as LoadSampleAndTarget
    else:
        from utils.datasets import LoadSampleAndTarget
        from models.yolo import Model

    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results_.txt'
    results_file_csv = save_dir / 'results_.csv'
    model_version = 0
    total_image = [0]
    start_epoch, best_fitness = 0, 0.0
    model_version = 1
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    device, git_status = select_device(opt.device, batch_size=opt.batch_size)
    opt.git_status = git_status
    logger.info(colorstr('hyperparameters: ') +
                ', '.join(f'{k}={v}' for k, v in hyp.items()))

    tag_results = ('Epochs', 'VRAMs', "t_loss", "v_loss", "top1", "top5")
    if not results_file_csv.exists():
        with open(results_file_csv.as_posix(), 'w') as f:
            csv_writer = csv.writer(f)
            aa = tag_results + ('P', 'R', 'mAP@.5',
                                'mAP@.5:.95', 'xx/lr0', 'xx/lr1', 'xx/lr2')
            csv_writer.writerow(aa)

    # Configure
    plots = opt.evolve <= 1  # create plots
    cuda = True if device.type in ['cuda'] else False
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict

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
            weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp

    with torch_distributed_zero_first(rank):
        train_path, val_path, test_path = parse_path(data_dict=data_dict)
        dataset = LoadSampleAndTarget(root=train_path.as_posix(), augment=True, prefix=colorstr('train: '))
        val_dataset = dataset
        if train_path.as_posix() != val_path.as_posix():
            val_dataset = LoadSampleAndTarget(root=val_path.as_posix(), augment=True,
                                              prefix=colorstr(
                                                  'val: '))

        if tb_writer:
            data_, names = dataset.dataset_analysis()
            tb_writer.add_figure("Datasets/train", plot_dataset(data_, names, "Total samples per class in Train"))
            if val_path.as_posix() != train_path.as_posix():
                data_, names = val_dataset.dataset_analysis()
                tb_writer.add_figure("Datasets/val", plot_dataset(data_, names, "Total samples per class in Val"))
            del data_, names

    total_image.append(len(dataset))
    nb = len(dataset)  # number of batches
    names = dataset.classes
    nc = len(names)
    input_channel = hyp.get('ch', 3)
    # Model
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(rank):
            attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=map_device)  # load checkpoint
        pretrained_model = ckpt['model']
        model = Model(opt.cfg or pretrained_model.yaml,
                      ch=input_channel,
                      nc=nc,
                      anchors=hyp.get('anchors')).to(device)  # create

        total_image = pretrained_model.total_image if hasattr(pretrained_model, 'total_image') else total_image
        model_version = pretrained_model.model_version if hasattr(pretrained_model, 'model_version') else model_version
        best_fitness = pretrained_model.best_fitness if hasattr(pretrained_model, 'best_fitness') else 'unknown'
        best_fitness = best_fitness.tolist()[0] if isinstance(best_fitness,
                                                              (torch.Tensor, np.ndarray)) else best_fitness
        best_fitness = 0. if best_fitness in ['unknown', -1, -1.] else best_fitness
        best_fitness = float(best_fitness) if isinstance(best_fitness, str) else best_fitness

        if not opt.resume:
            if model_version >= 1:
                check_it_one = pretrained_model.yaml['backbone']
                check_it_twice = model.yaml['backbone']
                pre_names = pretrained_model.names
                if check_it_one == check_it_twice and pre_names == names:
                    model_version += 1
                del check_it_one, check_it_twice, pre_names

        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # exclude keys
        state_dict = pretrained_model.float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict,
                                     model.state_dict(),
                                     exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load

        logger.info('Transferred %g/%g items from: %s, best fitness: %s, dataset %s images\n' % (
            len(state_dict), len(model.state_dict()), weights, best_fitness, total_image[-1]))  # report

        assert model.is_p5() == pretrained_model.is_p5(), f'Please paste the same model cfg branch like P5vsP5 or P6vsP6'
        assert model.is_Classify, f"Please paste the cls model cfg here"
    else:
        model = Model(opt.cfg,
                      ch=input_channel,
                      nc=nc,
                      anchors=hyp.get('anchors')).to(device)  # create
    if Path(opt.cfg).exists() and opt.cfg != "":
        with open(opt.cfg, "r") as f:
            model3d_config = yaml.load(f, yaml.SafeLoader)
        clip_len = model3d_config.get("sub_sample", 16)
        step = model3d_config.get("step", 3)
    else:
        clip_len = model.yaml.get("sub_sample", 16)
        step = model.yaml.get("step", 3)
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
    # Image sizes
    gs = int(model.stride.max())  # grid size (max stride)
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.imgsz]
    for _ in range(3):
        try:
            imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.imgsz]
            if use3D:
                dataset.step = val_dataset.step = step
                dataset.clip_len = val_dataset.clip_len = clip_len
                model.input_shape = [input_channel, clip_len, imgsz, imgsz] if isinstance(imgsz, int) else [
                    input_channel, clip_len, *imgsz]
            else:
                model.input_shape = [input_channel, imgsz, imgsz] if isinstance(imgsz, int) else [input_channel, *imgsz]
            y = model(torch.zeros([1, *model.input_shape], device=device))
            del y
        except:
            gs += gs
            model.stride = torch.tensor([gs], device=device)
            logger.warn(f"trying to get larger input shape")

    dataset.imgsz = imgsz
    val_dataset.imgsz = imgsz_test
    model.model_version = model_version
    model.total_image = total_image
    model.best_fitness = best_fitness

    if tb_writer:
        logger.info(f"{colorstr('Train: ')}Plotting samples to Tensorboard.")
        for x in range(10):
            tb_writer.add_figure("Samples/train", plotSample(*dataset.loadSampleforploting()), x)
            tb_writer.add_figure("Samples/val", plotSample(*dataset.loadSampleforploting()), x)

    if use3D:
        model.info(verbose=True,
                   img_size=[input_channel, clip_len, imgsz, imgsz] if isinstance(imgsz, int) else [input_channel,
                                                                                                    clip_len,
                                                                                                    *imgsz])
    else:
        model.info(verbose=True,
                   img_size=[input_channel, imgsz, imgsz] if isinstance(imgsz, int) else [input_channel, *imgsz])

    logger.info('')
    if tb_writer:
        if use3D:
            tb_writer.add_graph(model, torch.zeros([1, input_channel, clip_len, imgsz, imgsz], device=device))
        else:
            tb_writer.add_graph(model, torch.zeros([1, input_channel, imgsz, imgsz], device=device))

    with torch_distributed_zero_first(rank):
        if data_loader['dataloader'] is None:
            dataloader = create_dataloader_cls(dataset=dataset, batch_size=batch_size,
                                               world_size=opt.world_size,
                                               workers=opt.workers,
                                               shuffle=False if opt.rect else True,
                                               seed=opt.seed,
                                               prefix=colorstr('train: '))
            data_loader["dataloader"] = dataloader
        else:
            dataloader = data_loader["dataloader"]

        if rank in [-1, 0]:
            if data_loader['val_dataloader'] is None:
                val_dataloader = create_dataloader_cls(dataset=val_dataset,
                                                       batch_size=batch_size * 2,
                                                       shuffle=False if opt.rect else True,
                                                       world_size=opt.world_size,
                                                       workers=opt.workers, prefix=colorstr('val: '))
                data_loader["val_dataloader"] = val_dataloader
            else:
                val_dataloader = data_loader['val_dataloader']

    # accumulate loss before optimizing
    accumulate = max(round(nbs / total_batch_size), 1)
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
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

    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])

        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Results
        if ckpt.get('training_results') is not None:
            results_file.write_text(
                ckpt['training_results'])  # write results_.txt

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if opt.resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (
                weights, epochs)
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs
        del ckpt, state_dict, freeze

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    if not opt.resume:
        model.half().float() if device.type == 'xla' else model.to(device).half().float()

    # DDP mode
    if cuda and rank != -1:
        # nn.Multihead Attention incompatibility with DDP https://github.com/pytorch/pytorch/issues/26698
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank,
                    find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules()))
    # Model parameters
    hyp['label_smoothing'] = opt.label_smoothing
    model.hyp = hyp  # attach hyperparameters to model
    model.names = names
    model.nc = nc
    # Start training
    t0 = time_synchronized()
    results_ = (0, 0, 0, 0, 0, 0, 0)
    scheduler.last_epoch = start_epoch - 1  # do not move

    scaler = GradScaler()
    compute_loss, compute_loss_val = SmartLoss(model, hyp)

    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results_ to {save_dir}\n'
                f'Starting training for {epochs} epochs...\n')
    save_dir_ = wdir / 'init.pt'
    model.eval().cpu()
    torch.save({'model': model}, save_dir_)
    model.to(device)
    logger.info(f'saved init model at: {save_dir_}')
    # epoch ------------------------------------------------------------------
    for epoch in range(start_epoch, epochs):
        model.to(device).train(mode=True)
        tloss, vloss, best_fitness = 0.0, 0.0, 0.0  # train loss, val loss, fitness
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        logger.info(('\n' + '%11s' * len(tag_results)) % tag_results)
        pbar = enumerate(dataloader)
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=len(dataloader), unit='batch',
                        bar_format=TQDM_BAR_FORMAT)  # progress bar

        # batch -------------------------------------------------------------
        optimizer.zero_grad(set_to_none=True)
        for i, (images, targets) in pbar:
            targets = targets.to(device)
            images = images.to(device=device, non_blocking=True)
            # Forward
            with autocast(enabled=device.type in ['cuda', 'cpu'],
                          device_type='cuda' if device.type == 'cuda' else 'cpu',
                          dtype=torch.float16 if cuda else torch.bfloat16):
                pred = model(images)  # forward
                loss = compute_loss(pred, targets)

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
            scaler.step(optimizer)  # optimizer.step
            scaler.update()
            optimizer.zero_grad()
            if ema:
                ema.update(model)

            # Print
            if rank in [-1, 0]:
                tloss = (tloss * i + loss.item()) / (i + 1)  # update mean losses
                mem = '%.3gG' % (
                    torch.cuda.memory_reserved(device=device) / 1E9 if torch.cuda.is_available() else 0)  # (GB)

                pbar.desc = f"{f'{epoch + 1}/{epochs}':>11}{mem:>11}{tloss:>11.6g}" + ' ' * 36

        # Scheduler
        lr = [xx['lr'] for xx in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            ema.update_attr(
                model, include=['yaml', 'nc', 'hyp', 'gr', 'names',
                                'stride', 'class_weights', 'best_fitness',
                                'input_shape', 'model_version', 'total_image',
                                'is_anchorFree', 'is_Classify'])
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  # Calculate mAP
                wandb_logger.current_epoch = epoch + 1
                top1, top5, vloss, fig = cls_test(data_dict,
                                                  batch_size=batch_size * 2,
                                                  imgsz=imgsz_test,
                                                  model=ema.ema,
                                                  conf_thres=0.5,
                                                  single_cls=opt.single_cls,
                                                  dataloader=val_dataloader,
                                                  save_dir=save_dir,
                                                  verbose=False,
                                                  plots=plots and final_epoch,
                                                  wandb_logger=wandb_logger,
                                                  compute_loss=compute_loss_val,
                                                  epoch=epoch)
                if tb_writer:
                    tb_writer.add_scalar("Accuracy/Top1", top1, epoch)
                    tb_writer.add_scalar("Accuracy/Top5", top5, epoch)
                    tb_writer.add_figure("Confusion Matrix", fig, epoch)
                fi = top1
            if best_fitness < fi:
                best_fitness = fi
            if tb_writer:
                tb_writer.add_scalar("Loss/train", tloss, epoch)
                tb_writer.add_scalar("Loss/val", vloss, epoch)
                tb_writer.add_scalar("Lr", lr[-1], epoch)
                tb_writer.add_scalar("BestFitness", best_fitness, epoch)

            # Save model
            if (not opt.nosave) or (final_epoch and opt.evolve < 1):  # if save
                input_shape = list(images.shape[1:]) if isinstance(images.shape[1:], torch.Size) else images.shape[1:]
                model.input_shape = input_shape
                if rank in [-1, 0]:
                    ema.update_attr(model,
                                    include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights',
                                             'best_fitness',
                                             'input_shape', 'model_version', 'total_image', 'is_anchorFree'])

                ckpt = {
                    'epoch': epoch,
                    'training_results': results_file.read_text() if results_file.exists() else None,
                    'model': deepcopy(model.module if is_parallel(model) else model).half(),
                    'ema': deepcopy(ema.ema).half() if not final_epoch else None,
                    'updates': ema.updates if not final_epoch else None,
                    'optimizer': optimizer.state_dict() if not final_epoch else None,
                    'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None,
                    'hyp': hyp,
                    'train_gitstatus': git_status,
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
                del saver
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training
    prefix = colorstr('best fitness: ')
    logger.info(f'{prefix}{best_fitness}')
    if rank in [-1, 0]:
        logger.info('%g epochs completed in %.3f hours.\n' %
                    (epoch - start_epoch + 1, (time_synchronized() - t0) / 3600))

        if best.exists():
            prefix = colorstr('Validating')
            logger.info(f'{prefix} model {best.as_posix()}.')

            prefix = colorstr('Testing')
            if opt.evolve < 1 and test_path.as_posix() != val_path.as_posix():
                logger.info(f'{prefix} model {best.as_posix()}.')
                with torch_distributed_zero_first(rank):
                    if test_path.as_posix() != val_path.as_posix() and test_path.exists():
                        if data_loader['test_dataloader'] is None:
                            test_dataset = val_dataset
                            if test_path.as_posix() != val_path.as_posix():
                                test_dataset = LoadSampleAndTarget(root=test_path.as_posix(),
                                                                   augment=True)
                            test_dataloader = create_dataloader_cls(dataset=test_dataset,
                                                                    batch_size=batch_size * 2,
                                                                    shuffle=False if opt.rect else True,
                                                                    world_size=opt.world_size,
                                                                    workers=opt.workers, prefix=colorstr('val: '))
                            data_loader["test_dataloader"] = test_dataloader
                        else:
                            test_dataloader = data_loader['test_dataloader']
                    else:
                        logger.info(
                            colorstr("test: ") + 'val and test is the same things or test path does not exists!')
                # testing here

        final = best if best.exists() else last  # final model
        for f in last, best:
            if f.exists():
                strip_optimizer(f, halfModel=True)
                if 'best.pt' in str(f):
                    output_path = str(f)
                    output_path = output_path.replace('best.pt',
                                                      'deploy_best.pt')
                    try:
                        if opt.evolve <= 1:
                            Re_parameterization(inputWeightPath=str(f),
                                                outputWeightPath=output_path,
                                                device=map_device)
                    except Exception as ex:
                        prefix = colorstr('reparamater: ')
                        logger.error(f'{prefix}{ex}')
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

    return results_, data_loader


def train(hyp, opt, tb_writer=None,
          data_loader=None, logger=None):
    from models.yolo import Model

    # Save run settings
    if data_loader is None:
        data_loader = {'dataloader': None, 'dataset': None, 'val_dataloader': None, 'test_dataloader': None}
    save_dir, epochs, batch_size, total_batch_size, weights, rank, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank, opt.freeze
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results_.txt'
    results_file_csv = save_dir / 'results_.csv'
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
    if not results_file_csv.exists():
        with open(results_file_csv.as_posix(), 'w') as f:
            csv_writer = csv.writer(f)
            aa = tag_results + ('P', 'R', 'mAP@.5',
                                'mAP@.5:.95', 'xx/lr0', 'xx/lr1', 'xx/lr2')
            csv_writer.writerow(aa)

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
    total_image = [0]
    start_epoch, best_fitness = 0, 0.0
    model_version = 1
    input_channel = hyp.get('ch', 3)
    if pretrained:
        with torch_distributed_zero_first(rank):
            attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=map_device)  # load checkpoint
        pretrained_model = ckpt['model']
        model = Model(opt.cfg or pretrained_model.yaml,
                      ch=input_channel,
                      nc=nc,
                      anchors=hyp.get('anchors')).to(device)  # create

        total_image = pretrained_model.total_image if hasattr(pretrained_model, 'total_image') else total_image
        model_version = pretrained_model.model_version if hasattr(pretrained_model, 'model_version') else model_version
        best_fitness = pretrained_model.best_fitness if hasattr(pretrained_model, 'best_fitness') else 'unknown'
        best_fitness = best_fitness.tolist()[0] if isinstance(best_fitness,
                                                              (torch.Tensor, np.ndarray)) else best_fitness
        best_fitness = 0. if best_fitness in ['unknown', -1, -1.] else best_fitness
        best_fitness = float(best_fitness) if isinstance(best_fitness, str) else best_fitness

        if not opt.resume:
            if model_version >= 1:
                check_it_one = pretrained_model.yaml['backbone']
                check_it_twice = model.yaml['backbone']
                pre_names = pretrained_model.names
                if check_it_one == check_it_twice and pre_names == names:
                    model_version += 1
                del check_it_one, check_it_twice, pre_names

        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # exclude keys
        state_dict = pretrained_model.float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict,
                                     model.state_dict(),
                                     exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load

        logger.info('Transferred %g/%g items from: %s, best fitness: %s, dataset %s images\n' % (
            len(state_dict), len(model.state_dict()), weights, best_fitness, total_image[-1]))  # report

        assert model.is_p5() == pretrained_model.is_p5(), f'Please paste the same model cfg branch like P5vsP5 or P6vsP6'
        assert nodes == nodes2, f'Please paste the same model cfg branch like P5vsP5 or P6vsP6'
    else:
        model = Model(opt.cfg,
                      ch=input_channel,
                      nc=nc,
                      anchors=hyp.get('anchors')).to(device)  # create

    if model.is_anchorFree:
        tag_results = list(tag_results)
        tag_results.remove('labels')
        tag_results = tuple(tag_results)

    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # check
    train_path, val_path, test_path = parse_path(data_dict=data_dict)

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
    # Image sizes

    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    # verify imgsz are gs-multiples
    imgsz, imgsz_test = [check_img_size(size, gs) for size in opt.imgsz]
    model.model_version = model_version
    model.total_image = total_image
    model.input_shape = [3, imgsz, imgsz] if isinstance(imgsz, int) else [3, *imgsz]
    model.best_fitness = best_fitness
    model.info(verbose=True,
               img_size=[input_channel, imgsz, imgsz] if isinstance(imgsz, int) else [input_channel, *imgsz])
    logger.info('')
    if tb_writer:
        tb_writer.add_graph(model, torch.zeros([1, input_channel, imgsz, imgsz], device=device))
    # accumulate loss before optimizing
    accumulate = max(round(nbs / total_batch_size), 1)
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
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

    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])

        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Results
        if ckpt.get('training_results') is not None:
            results_file.write_text(
                ckpt['training_results'])  # write results_.txt

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

    # number of detection layers (used for scaling hyp['obj'])
    nl = model.model[-1].nl

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
                                                prefix=colorstr('train: '))

        data_loader['dataloader'], data_loader['dataset'] = dataloader, dataset
    else:
        dataloader, dataset = data_loader['dataloader'], data_loader['dataset']

    total_image.append(len(dataset))
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (
        mlc, nc, opt.data, nc - 1)

    # Process 0
    if rank in [-1, 0]:
        if data_loader['val_dataloader'] is None:
            if train_path.as_posix() == val_path.as_posix():
                val_dataloader = dataloader
                logger.info(colorstr('val: ') + "inherit from train")
            else:
                val_dataloader = create_dataloader(val_path.as_posix(),
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
                                                   pad=0.5, prefix=colorstr('val: '))[0]
            data_loader['val_dataloader'] = val_dataloader
        else:
            val_dataloader = data_loader['val_dataloader']

        if test_path.as_posix() != val_path.as_posix() and test_path.exists():
            if data_loader['test_dataloader'] is None:
                test_dataloader = create_dataloader(test_path.as_posix(),
                                                    imgsz_test,
                                                    batch_size * 2, gs, opt,
                                                    hyp=hyp,
                                                    cache='no',
                                                    rect=opt.rect,
                                                    shuffle=False if opt.rect else True,
                                                    rank=-1,
                                                    world_size=opt.world_size,
                                                    workers=opt.workers,
                                                    pad=0.5, prefix=colorstr('test: '))[0]
                data_loader['test_dataloader'] = test_dataloader
            else:
                test_dataloader = data_loader['test_dataloader']
        else:
            logger.info(colorstr("val: ") + 'val and test is the same things or test path does not exists!')

        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes
            if plots:
                # plot_labels(labels, names, save_dir, loggers)
                if tb_writer:
                    tb_writer.add_histogram('classes', c, 0)

            # Anchors
            if not opt.noautoanchor:
                if not any([model.is_anchorFree, model.is_Classify]):
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
    results_ = (0, 0, 0, 0, 0, 0, 0)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = GradScaler()

    compute_loss, compute_loss_val = SmartLoss(model, hyp)

    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results_ to {save_dir}\n'
                f'Starting training for {epochs} epochs...\n')
    model.to(device=torch.device('cpu'))
    save_dir_ = wdir / 'init.pt'
    model.cpu()
    torch.save({'model': model}, save_dir_)
    model.to(device)
    logger.info(f'saved init model at: {save_dir_}')
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

        mloss = torch.zeros(4 if not model.is_anchorFree else 3, device=device)  # mean losses
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        logger.info(('\n' + '%11s' * len(tag_results)) % tag_results)
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=len(dataloader), unit='batch',
                        bar_format=TQDM_BAR_FORMAT)  # progress bar

        # batch -------------------------------------------------------------
        optimizer.zero_grad(set_to_none=True)
        for i, (images, targets, paths, _) in pbar:
            del _
            # number integrated batches (since train start)
            ni = i + nb * epoch
            images = images.to(device=device, non_blocking=True,
                               dtype=torch.float32) / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # xx interp
                accumulate = max(1,
                                 np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, xx in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    xx['lr'] = np.interp(
                        ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, xx['initial_lr'] * lf(epoch)])
                    if 'momentum' in xx:
                        xx['momentum'] = np.interp(
                            ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(int(imgsz * 0.5), int(imgsz * 1.5) + gs) // gs * gs  # size
                sf = sz / max(images.shape[2:])  # scale factor
                if sf != 1:
                    # new shape (stretched to gs-multiple)
                    ns = [math.ceil(x * sf / gs) * gs for x in images.shape[2:]]
                    images = functional.interpolate(images, size=ns,
                                                    mode='bilinear',
                                                    align_corners=False,
                                                    antialias=False)

            # Forward
            with autocast(enabled=device.type in ['cuda', 'cpu'],
                          device_type='cuda' if device.type == 'cuda' else 'cpu',
                          dtype=torch.float16 if cuda else torch.bfloat16):

                pred = model(images)  # forward
                loss, loss_items = compute_loss(pred, targets, images)  # loss scaled by batch_size
                if rank != -1:
                    loss *= opt.world_size  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema:
                    ema.update(model)

            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items.to(device)) / (i + 1)  # update mean losses
                mem = '%.3gG' % (
                    torch.cuda.memory_reserved(device=device) / 1E9 if torch.cuda.is_available() else 0)  # (GB)

                s = ('%11s' * 2 + '%11.4g' * (2 + len(mloss))) % (
                    f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], images.shape[-1])
                pbar.set_description(s)

                # Plot
                if plots and ni < 10:
                    f = save_dir / f'train_batch{ni}.jpg'  # filename
                    if tb_writer:
                        tb_writer.add_image(str(f), np.moveaxis(plot_images(
                            images=images, targets=targets, paths=paths, fname=f, names=names), -1, 0), ni)
                    Thread(target=plot_images, args=(
                        images,
                        targets,
                        paths, f), daemon=True).start()
                elif plots and ni == 10 and wandb_logger.wandb:
                    wandb_logger.log({"Mosaics": [wandb_logger.wandb.Image(str(x), caption=x.name) for x in
                                                  save_dir.glob('train*.jpg') if x.exists()]})
        input_shape = list(images.shape[1:]) if isinstance(images.shape[1:], torch.Size) else images.shape[1:]
        del images, targets, paths
        # Scheduler
        lr = [xx['lr'] for xx in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            ema.update_attr(
                model, include=['yaml', 'nc', 'hyp', 'gr', 'names',
                                'stride', 'class_weights', 'best_fitness',
                                'input_shape', 'model_version', 'total_image',
                                'is_anchorFree', 'is_Classify'])
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  # Calculate mAP
                wandb_logger.current_epoch = epoch + 1
                results_, maps = test(data_dict,
                                      batch_size=batch_size * 2,
                                      imgsz=imgsz_test,
                                      model=ema.ema,
                                      single_cls=opt.single_cls,
                                      dataloader=val_dataloader,
                                      save_dir=save_dir,
                                      verbose=False,
                                      plots=plots and final_epoch,
                                      wandb_logger=wandb_logger,
                                      compute_loss=compute_loss_val,
                                      is_coco=is_coco,
                                      v5_metric=opt.v5_metric)[:2]

                # Write
                with open(results_file.as_posix(), 'a') as f:
                    f.write(s + '%10.4g' * 7 % results_ + '\n')
                with open(results_file_csv.as_posix(), 'a') as f:
                    csv_writer = csv.writer(f)
                    aa = []
                    for xx in s.split(' '):
                        if xx not in ['', ' ', '  ', '   ', '    ']:
                            aa.append(xx)
                    csv_writer.writerow(tuple(aa) + results_)

                if len(opt.name) and opt.bucket:
                    os.system('gsutil cp %s gs://%s/results_/results_%s.txt' %
                              (results_file.as_posix(), opt.bucket, opt.name))

                # Log
                tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                        'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                        'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                        'xx/lr0', 'xx/lr1', 'xx/lr2']  # params
                if tb_writer or wandb_logger.wandb:
                    for xx, tag in zip(list(mloss[:-1]) + list(results_) + lr, tags):
                        if tb_writer:
                            tb_writer.add_scalar(tag, xx, epoch)  # tensorboard
                        if wandb_logger.wandb:
                            wandb_logger.log({tag: xx})  # W&B

            # Update best mAP
            # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            fi = fitness(np.array(results_).reshape(1, -1))

            if fi > best_fitness:
                best_fitness = fi.tolist()[0]
                model.best_fitness = best_fitness
            wandb_logger.end_epoch(best_result=best_fitness == fi)

            # Save model
            if (not opt.nosave) or (final_epoch and opt.evolve < 1):  # if save
                model.input_shape = input_shape
                if rank in [-1, 0]:
                    ema.update_attr(model,
                                    include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights',
                                             'best_fitness',
                                             'input_shape', 'model_version', 'total_image', 'is_anchorFree'])

                ckpt = {
                    'epoch': epoch,
                    'training_results': results_file.read_text() if results_file.exists() else None,
                    'model': deepcopy(model.module if is_parallel(model) else model).half(),
                    'ema': deepcopy(ema.ema).half() if not final_epoch else None,
                    'updates': ema.updates if not final_epoch else None,
                    'optimizer': optimizer.state_dict() if not final_epoch else None,
                    'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None,
                    'hyp': hyp,
                    'train_gitstatus': git_status,
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
                del saver
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training
    prefix = colorstr('best fitness: ')
    logger.info(f'{prefix}{best_fitness}')
    if rank in [-1, 0]:
        # Plots
        if plots:
            plot_results(save_dir=save_dir)  # save as results_.png
            if wandb_logger.wandb:
                files = ['results_.png', 'confusion_matrix.png', *
                [f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
                wandb_logger.log({"Results": [wandb_logger.wandb.Image(str(save_dir / f), caption=f) for f in files
                                              if (save_dir / f).exists()]})
        logger.info('%g epochs completed in %.3f hours.\n' %
                    (epoch - start_epoch + 1, (time_synchronized() - t0) / 3600))

        if best.exists():
            prefix = colorstr('Validating')
            logger.info(f'{prefix} model {best.as_posix()}.')
            results_val = test(opt.data,
                               batch_size=batch_size * 2,
                               imgsz=imgsz_test,
                               conf_thres=0.001,
                               iou_thres=0.7,
                               weights=best,
                               single_cls=opt.single_cls,
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
            if opt.evolve < 1 and test_path.as_posix() != val_path.as_posix():
                logger.info(f'{prefix} model {best.as_posix()}.')
                results_test = test(opt.data,
                                    batch_size=batch_size * 2,
                                    imgsz=imgsz_test,
                                    conf_thres=0.001,
                                    iou_thres=0.7,
                                    weights=best,
                                    single_cls=opt.single_cls,
                                    dataloader=test_dataloader,
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
                strip_optimizer(f, halfModel=True)
                if 'best.pt' in str(f):
                    output_path = str(f)
                    output_path = output_path.replace('best.pt',
                                                      'deploy_best.pt')
                    try:
                        if opt.evolve <= 1:
                            Re_parameterization(inputWeightPath=str(f),
                                                outputWeightPath=output_path,
                                                device=map_device)
                    except Exception as ex:
                        prefix = colorstr('reparamater: ')
                        logger.error(f'{prefix}{ex}')
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

    return results_, data_loader
