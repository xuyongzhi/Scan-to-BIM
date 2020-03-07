from __future__ import division
import argparse
import os
import os.path as osp
import time

import open3d as o3d

import mmcv
import torch
from mmcv import Config
from mmcv.runner import init_dist

from mmdet import __version__
from mmdet.apis import get_root_logger, set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    parser.add_argument('--rotate', type=int, default=None,
                        help='use data aug of rotation or not')
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--bs', type=int, default=None)
    parser.add_argument('--cls', type=str, default=None, help='refine, refine_final')
    parser.add_argument('--dcn_zero_base', type=int, default=None)
    parser.add_argument('--corhm', type=int, default=None,
                        help='0: no corner heat map, 1: both corner and line, 2:only corner')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def update_config(cfg, args, split):
    assert split == 'train' or split == 'test'
    if args.config.split('/')[1] != 'strpoints':
      #return
      pass
    #gpus = args.gpus
    gpus = 1
    rotate = args.rotate
    lr = args.lr
    bs = args.bs
    cls_loss = args.cls
    dcn_zero_base = args.dcn_zero_base
    corner_hm = args.corhm
    if rotate is not None:
      assert rotate == 1 or rotate == 0
      if split == 'train':
        assert cfg[f'train_pipeline'][4]['type'] == 'RandomRotate'
        cfg[f'train_pipeline'][4]['rotate_ratio'] *= rotate
        cfg['data']['train']['pipeline'][4]['rotate_ratio'] *= rotate
        cfg['data']['val']['pipeline'][4]['rotate_ratio'] *= rotate
      elif split == 'test':
        assert cfg['test_pipeline'][2]['transforms'][2]['type'] == 'RandomRotate'
        cfg['test_pipeline'][2]['transforms'][2]['rotate_ratio'] *= rotate
        cfg['data']['test']['pipeline'][2]['transforms'][2]['rotate_ratio'] *= rotate
        pass


    if lr is not None:
      cfg['optimizer']['lr'] = lr
    if bs is not None:
      cfg['data']['imgs_per_gpu'] = bs
    if cls_loss is not None:
      cls_loss = cls_loss.split('_')
      cfg['model']['bbox_head']['cls_types'] = cls_loss
    if dcn_zero_base is not None:
      assert dcn_zero_base == 0 or dcn_zero_base == 1
      cfg['model']['bbox_head']['dcn_zero_base'] = dcn_zero_base == 1

    if corner_hm is not None:
      assert corner_hm ==0 or corner_hm==1 or corner_hm ==2
      cfg['model']['bbox_head']['corner_hm'] = corner_hm!=0
      cfg['model']['bbox_head']['corner_hm_only'] = corner_hm==2


    # update work_dir
    if split == 'train':
      if '_obj_rep' in cfg:
        cfg['work_dir'] += '_' + cfg['_obj_rep']
      if 'cls_types' in cfg['model']['bbox_head']:
        cfg['work_dir'] += '_' + '_'.join(cfg['model']['bbox_head']['cls_types'])
      if 'IMAGE_SIZE' in cfg:
        cfg['work_dir'] += '_' + str(cfg['IMAGE_SIZE'])
      if 'TOPVIEW' in cfg:
        cfg['work_dir'] += '_' + cfg['TOPVIEW']
      if 'DATA' in cfg:
        cfg['work_dir'] += '_' + cfg['DATA']
      cfg['work_dir'] += '_bs' + str(cfg['data']['imgs_per_gpu'] * gpus)
      cfg['work_dir'] += '_lr' + str(int(cfg['optimizer']['lr']*1000))
      if 'rotate_ratio' in cfg['train_pipeline'][4]:
        if cfg['train_pipeline'][4]['rotate_ratio'] == 0:
          cfg['work_dir'] += '_NR'
        else:
          cfg['work_dir'] += '_RA'
      if 'method' in cfg['img_norm_cfg']:
        cfg['work_dir'] += '_Norm' + cfg['img_norm_cfg']['method']
      if dcn_zero_base:
        cfg['work_dir'] += '_DcnZb'
      if 'corner_hm' in cfg['model']['bbox_head'] and cfg['model']['bbox_head']['corner_hm']:
        cfg['work_dir'] += '_Chm'
        if cfg['model']['bbox_head']['corner_hm_only']:
          cfg['work_dir'] += 'Only'
        cor_assigner = cfg['train_cfg']['corner']['assigner']
        radius = cor_assigner['ref_radius']
        pos_iou = int(cor_assigner['pos_iou_thr'] * 10)
        neg_iou = int(cor_assigner['neg_iou_thr'] * 10)
        cfg['work_dir'] += f'R{radius}P{neg_iou}N{neg_iou}'

      # refine_iou_assigner
      refine_iou_assigner = cfg['train_cfg']['refine']['assigner']
      p = int(refine_iou_assigner['pos_iou_thr']*10)
      n = int(refine_iou_assigner['neg_iou_thr']*10)
      m = int(refine_iou_assigner['min_pos_iou']*10)
      cfg['work_dir'] += f'_Rfiou{p}{n}{m}'

      # FPN
      neck = cfg['model']['neck']
      num_in = len(neck['in_channels'])
      num_outs = neck['num_outs']
      cfg['work_dir'] += f'_Fpn{num_in}{num_outs}'

      # point base scale
      base_scale=cfg['model']['bbox_head']['point_base_scale']
      cfg['work_dir'] += f'_Pbs{base_scale}'
      pass
      #print(cfg['work_dir'])
      pass


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    update_config(cfg, args, 'train')

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.gpus / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, '{}.log'.format(timestamp))
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # log some basic info
    logger.info('Distributed training: {}'.format(distributed))
    logger.info('MMDetection Version: {}'.format(__version__))
    logger.info('Config:\n{}'.format(cfg.text))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}, deterministic: {}'.format(
            args.seed, args.deterministic))
        set_random_seed(args.seed, deterministic=args.deterministic)

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        datasets.append(build_dataset(cfg.data.val))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.text,
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=args.validate,
        timestamp=timestamp)


if __name__ == '__main__':
    main()
