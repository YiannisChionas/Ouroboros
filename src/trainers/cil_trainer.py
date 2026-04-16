import time
import torch
import timm

import os
import numpy as np
import importlib
from datasets.data_loader import get_loaders

import utils
from loggers.exp_logger import MultiLogger
from networks.network import LLL_Net
from networks.distil_network import LLL_Net_Distilled
from approach.incremental_learning import Incremental_Learning_Approach
from networks import tvmodels, timmmodels, set_model_head_var


def train(args):

    start_time = time.time()

    _set_defaults(args)
    _print_args(args)

    args['results_path'] = os.path.expanduser(args['results_path'])

    if not args['experiment_path']:
        if args['classes_first_task']:
            args['experiment_path'] = "{}/{}/{}/{},{}".format(args["network"], args["dataset"], args["approach"], args['classes_first_task'], args['increment'])
        else:
            args['experiment_path'] = "{}/{}/{}/{}".format(args["network"], args["dataset"], args["approach"], args['increment'])

    logger = MultiLogger(
                         args['results_path'],
                         args['experiment_path'],
                         args['log'],
                         args['save_models']
                         )

    if args['no_cudnn_deterministic']:
        print('WARNING: CUDNN Deterministic will be disabled.')
        utils.cudnn_deterministic = False

    utils.seed_everything(seed=args['seed'])

    if torch.cuda.is_available():
        torch.cuda.set_device(args['gpus'][0])
        args['device'] = f"cuda:{args['gpus'][0]}"
    else:
        print('WARNING: [CUDA unavailable] Using CPU instead!')
        args['device'] = 'cpu'

    multi_gpu = torch.cuda.is_available() and len(args['gpus']) > 1
    if multi_gpu:
        print(f"Multi-GPU mode: DataParallel on GPUs {args['gpus']}")

    if args['network'] in tvmodels:
        tvnet = getattr(importlib.import_module(name='torchvision.models'), args['network'])
        init_model = tvnet(pretrained=args['pretrained'])
        set_model_head_var(init_model)
    elif args['network'] in timmmodels:
        init_model = timm.create_model(args['network'], pretrained=args['pretrained'], num_classes=0)
        set_model_head_var(init_model)
    else:
        net_wrapper = getattr(importlib.import_module(name='networks'), args['network'])
        init_model = net_wrapper(pretrained=args['pretrained'])
        set_model_head_var(init_model)

    Appr = getattr(importlib.import_module(name='approach.' + args['approach']), 'Appr')
    assert issubclass(Appr, Incremental_Learning_Approach)

    from datasets.exemplars_dataset import ExemplarsDataset
    Appr_ExemplarsDataset = Appr.exemplars_dataset_class()
    if Appr_ExemplarsDataset:
        assert issubclass(Appr_ExemplarsDataset, ExemplarsDataset)

    utils.seed_everything(seed=args['seed'])
    train_loader, validation_loader, test_loader, classes_per_task = get_loaders(args)

    if args['use_valid_only']:
        test_loader = validation_loader

    utils.seed_everything(seed=args['seed'])
    if args.get('distilled', False):
        net = LLL_Net_Distilled(init_model, remove_existing_head=False)
    else:
        net = LLL_Net(init_model, remove_existing_head=False)

    utils.seed_everything(seed=args['seed'])
    exemplars_dataset = None
    if Appr_ExemplarsDataset:
        eargs = args.get('exemplars_args', {})
        transform = train_loader[0].dataset.transform
        class_indices = train_loader[0].dataset.class_indices
        exemplars_dataset = Appr_ExemplarsDataset(
                                                  transform, class_indices,
                                                  num_exemplars=eargs.get('num_exemplars', 0),
                                                  num_exemplars_per_class=eargs.get('num_exemplars_per_class', 0),
                                                  exemplar_selection=eargs.get('exemplar_selection', 'random'),
                                                  )

    utils.seed_everything(seed=args['seed'])
    appr = Appr(args, model=net, logger=logger, exemplars_dataset=exemplars_dataset)

    total_tasks = len(classes_per_task)
    start_task = args['start_at_task']
    stop_task  = args['stop_at_task'] if args['stop_at_task'] != 0 else total_tasks
    stop_task  = min(stop_task, total_tasks)

    if not (0 <= start_task < stop_task <= total_tasks):
        raise ValueError(f"Invalid range: start={start_task}, stop={stop_task}, total={total_tasks}")

    start_epoch = args['start_epoch']
    stop_epoch  = args['stop_epoch'] if args['stop_epoch'] != 0 else args['nepochs']
    is_last_epoch_job = (stop_epoch == args['nepochs'])

    metrics = {
        'acc_taw':  np.zeros((total_tasks, total_tasks), dtype=np.float32),
        'acc_tag':  np.zeros((total_tasks, total_tasks), dtype=np.float32),
        'forg_taw': np.zeros((total_tasks, total_tasks), dtype=np.float32),
        'forg_tag': np.zeros((total_tasks, total_tasks), dtype=np.float32),
    }

    # Resume from task checkpoint
    if start_task > 0:
        ckpt = os.path.join(logger.exp_path, "models", f"task{start_task - 1}.ckpt")
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        for tt in range(start_task):
            net.add_head(classes_per_task[tt][1])
        net.to(args['device'])
        net.load_state_dict(torch.load(ckpt, map_location=args['device']))
        metrics = _load_metrics(total_tasks, logger.exp_path, start_task)
        # Epoch resume (start_epoch > 0): load datasets including current task
        # Task resume (start_epoch == 0): load datasets up to previous task
        progress_task = start_task if start_epoch > 0 else start_task - 1
        appr.load_progress(logger.exp_path, progress_task)

    for task in range(start_task, stop_task):
        _run_task(task,
                  classes_per_task=classes_per_task,
                  network=net,
                  device=args['device'],
                  approach=appr,
                  train_loader=train_loader,
                  validation_loader=validation_loader,
                  start_epoch=start_epoch,
                  stop_epoch=stop_epoch,
                  gpu_ids=args['gpus'] if multi_gpu else None)

        # Eval + save metrics only when all epochs for this task are done
        if is_last_epoch_job:
            _eval_task(task=task, approach=appr, test_loader=test_loader, logger=logger, metrics=metrics)

    if is_last_epoch_job:
        _save_metrics(task=task, results_path=args['results_path'], logger=logger, metrics=metrics,
                      classes_per_task=classes_per_task, network=net)

    appr.save_progress(logger.exp_path, task)

    if is_last_epoch_job:
        utils.print_summary(metrics['acc_taw'], metrics['acc_tag'], metrics['forg_taw'], metrics['forg_tag'])

    print('[Elapsed time = {:.1f} h]'.format((time.time() - start_time) / (60 * 60)))
    print('Done!')

    return metrics['acc_taw'], metrics['acc_tag'], metrics['forg_taw'], metrics['forg_tag'], logger.exp_path

# ------------------------------------------------------------------
# Metrics matrices
# ------------------------------------------------------------------

def _load_metrics(T, path, start_t):
    return {
        'acc_taw':  _load_metrics_matrix(T, "acc_taw",  path, start_t),
        'acc_tag':  _load_metrics_matrix(T, "acc_tag",  path, start_t),
        'forg_taw': _load_metrics_matrix(T, "forg_taw", path, start_t),
        'forg_tag': _load_metrics_matrix(T, "forg_tag", path, start_t),
    }

def _load_metrics_matrix(T, prefix, path, start_t):
    target = os.path.join(path, f"{prefix}-{start_t - 1}.txt")
    if not os.path.isfile(target):
        raise FileNotFoundError(f"Metrics file not found: {target}")
    metrics = np.loadtxt(target, dtype=np.float32)
    if metrics.shape != (T, T):
        raise ValueError(f"Metrics file '{target}' has shape {metrics.shape}, expected ({T}, {T}).")
    print(f"Loaded {prefix} from: {target}")
    return metrics

# ------------------------------------------------------------------
# Per-task loop
# ------------------------------------------------------------------

def _run_task(task, classes_per_task, network, device, approach, train_loader, validation_loader,
              start_epoch=0, stop_epoch=0, gpu_ids=None):
    _, ncla = classes_per_task[task]

    print('*' * 108)
    print('Task {:2d}  [epochs {:d}-{:d}]'.format(task, start_epoch, stop_epoch))
    print('*' * 108)

    if task >= len(network.heads):
        network.add_head(ncla)
        network.to(device)

    # Wrap backbone only (not LLL_Net) so heads/task_offset/task_cls remain directly accessible
    if gpu_ids is not None:
        network.model = torch.nn.DataParallel(network.model, device_ids=gpu_ids)

    approach.train(task, train_loader[task], validation_loader[task],
                   start_epoch=start_epoch, stop_epoch=stop_epoch)

    # Unwrap — external checkpoints are always saved on the unwrapped model
    if gpu_ids is not None:
        network.model = network.model.module

    print('-' * 108)

def _eval_task(task, approach, test_loader, logger, metrics):
    acc_taw  = metrics['acc_taw']
    acc_tag  = metrics['acc_tag']
    forg_taw = metrics['forg_taw']
    forg_tag = metrics['forg_tag']

    for eval_task in range(task + 1):
        test_loss, acc_taw[task, eval_task], acc_tag[task, eval_task] = approach.eval(eval_task, test_loader[eval_task])
        if eval_task < task:
            best_prev_taw = acc_taw[:task, eval_task].max()
            forg_taw[task, eval_task] = best_prev_taw - acc_taw[task, eval_task]
            best_prev_tag = acc_tag[:task, eval_task].max()
            forg_tag[task, eval_task] = best_prev_tag - acc_tag[task, eval_task]
        print('>>> Test on task {:2d} : loss={:.3f} | TAw acc={:5.1f}%, forg={:5.1f}% | TAg acc={:5.1f}%, forg={:5.1f}% <<<'.format(
            eval_task, test_loss,
            100 * acc_taw[task, eval_task], 100 * forg_taw[task, eval_task],
            100 * acc_tag[task, eval_task], 100 * forg_tag[task, eval_task]
        ))
        logger.log_scalar(task=task, iter=eval_task, name='loss',     group='test', value=float(test_loss))
        logger.log_scalar(task=task, iter=eval_task, name='acc_taw',  group='test', value=float(100.0 * acc_taw[task, eval_task]))
        logger.log_scalar(task=task, iter=eval_task, name='acc_tag',  group='test', value=float(100.0 * acc_tag[task, eval_task]))
        logger.log_scalar(task=task, iter=eval_task, name='forg_taw', group='test', value=float(100.0 * forg_taw[task, eval_task]))
        logger.log_scalar(task=task, iter=eval_task, name='forg_tag', group='test', value=float(100.0 * forg_tag[task, eval_task]))

def _save_metrics(task, results_path, logger, metrics, classes_per_task, network):
    print('Save at ' + os.path.join(results_path, logger.exp_path))
    acc_taw  = metrics['acc_taw']
    acc_tag  = metrics['acc_tag']
    forg_taw = metrics['forg_taw']
    forg_tag = metrics['forg_tag']
    cur = task + 1
    blk_taw = acc_taw[:cur, :cur]
    blk_tag = acc_tag[:cur, :cur]

    aux = np.tril(np.repeat([[tdata[1] for tdata in classes_per_task[:cur]]], cur, axis=0))
    avg_taw  = blk_taw.sum(1) / np.tril(np.ones((cur, cur))).sum(1)
    avg_tag  = blk_tag.sum(1) / np.tril(np.ones((cur, cur))).sum(1)
    wavg_taw = (blk_taw * aux).sum(1) / aux.sum(1)
    wavg_tag = (blk_tag * aux).sum(1) / aux.sum(1)

    logger.log_result(acc_taw,  name="acc_taw",  step=task)
    logger.log_result(acc_tag,  name="acc_tag",  step=task)
    logger.log_result(forg_taw, name="forg_taw", step=task)
    logger.log_result(forg_tag, name="forg_tag", step=task)
    logger.log_result(avg_taw,  name="avg_accs_taw",  step=task)
    logger.log_result(avg_tag,  name="avg_accs_tag",  step=task)
    logger.log_result(wavg_taw, name="wavg_accs_taw", step=task)
    logger.log_result(wavg_tag, name="wavg_accs_tag", step=task)

    for name, arr in [("acc_taw", acc_taw), ("acc_tag", acc_tag),
                      ("forg_taw", forg_taw), ("forg_tag", forg_tag)]:
        np.savetxt(os.path.join(logger.exp_path, f"{name}-{task}.txt"), arr, fmt='%.6f', delimiter='\t')

    logger.save_model(network.state_dict(), task=task)

# ------------------------------------------------------------------
# Defaults & print
# ------------------------------------------------------------------

def _set_defaults(args: dict) -> None:
    args.setdefault('gpu', 0)
    # gpus: list of GPU ids. If not set, derive from scalar 'gpu'.
    if 'gpus' not in args:
        g = args['gpu']
        args['gpus'] = g if isinstance(g, list) else [g]
    args.setdefault('results_path', './results')
    args.setdefault('experiment_path', None)
    args.setdefault('seed', 0)
    args.setdefault('log', ['disk'])
    args.setdefault('save_models', True)
    args.setdefault('no_cudnn_deterministic', False)
    args.setdefault('dataset', 'cifar100')
    args.setdefault('num_workers', 2)
    args.setdefault('pin_memory', False)
    args.setdefault('batch_size', 64)
    args.setdefault('increment', 10)
    args.setdefault('classes_first_task', None)
    args.setdefault('use_valid_only', False)
    args.setdefault('start_at_task', 0)
    args.setdefault('stop_at_task', 0)
    args.setdefault('network', 'resnet32')
    args.setdefault('pretrained', False)
    args.setdefault('approach', 'finetuning')
    args.setdefault('nepochs', 50)
    args.setdefault('start_epoch', 0)
    args.setdefault('stop_epoch', 0)    # 0 = run until nepochs
    args.setdefault('optimizer_name', 'sgd')
    args.setdefault('lr_scheduler', None)
    args.setdefault('lr', 0.1)
    args.setdefault('lr_min', None)
    args.setdefault('lr_factor', None)
    args.setdefault('lr_patience', None)
    args.setdefault('lr_warmup_epochs', 0)
    args.setdefault('clipping', 10000)
    args.setdefault('momentum', 0.0)
    args.setdefault('weight_decay', 0.0)
    args.setdefault('warmup_nepochs', 0)
    args.setdefault('warmup_lr_factor', 1.0)
    args.setdefault('multi_softmax', False)
    args.setdefault('fix_bn', False)
    args.setdefault('freeze_backbone', False)
    args.setdefault('eval_on_train', False)
    args.setdefault('approach_args', {})
    args.setdefault('exemplars_args', {})

def _print_args(args: dict):
    print('=' * 108)
    print('Arguments =')
    for arg in sorted(args.keys()):
        if arg not in ('approach_args', 'exemplars_args'):
            print('\t' + arg + ':', args[arg])
    print('=' * 108)
    print('Approach arguments =')
    for arg, val in sorted(args.get('approach_args', {}).items()):
        print('\t' + arg + ':', val)
    print('=' * 108)
    print('Exemplars arguments =')
    for arg, val in sorted(args.get('exemplars_args', {}).items()):
        print('\t' + arg + ':', val)
    print('=' * 108)
