# model settings
''' modified
  num_classes
  num_points
'''
''' # pedding
  num_outs
  assigner
  img_norm_cfg
  transform_method
'''

DATA='A'
#*******************************************************************************
# 1. coco
#_obj_rep='box_scope'

# 2. line scope
#_obj_rep='line_scope'

#3. lines beike
_obj_rep='lscope_istopleft'

#*******************************************************************************
from configs.common import  OBJ_REP, IMAGE_SIZE
assert OBJ_REP == _obj_rep
_all_obj_rep_dims = {'box_scope': 4, 'line_scope': 4, 'lscope_istopleft':5}
_obj_dim = _all_obj_rep_dims[_obj_rep]

if _obj_rep == 'box_scope':
  _transform_method = 'moment'
elif _obj_rep == 'line_scope':
  _transform_method = 'moment'
elif _obj_rep == 'lscope_istopleft':
  _transform_method='moment_lscope_istopleft'
#*******************************************************************************


norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

model = dict(
    type='StrPointsDetector',
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=4,
        num_stages=4,
        out_indices=( 0, 1, 2),
        frozen_stages=-1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[ 256, 512, 1024],
        out_channels=256,
        start_level=0,
        add_extra_convs=True,
        num_outs=4,
        norm_cfg=norm_cfg),
    bbox_head=dict(
        type='StrPointsHead',
        num_classes=2,
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        stacked_convs=3,
        num_points=9,
        gradient_mul=0.1,
        point_strides=[4, 8, 16, 32],
        point_base_scale=2,
        norm_cfg=norm_cfg,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox_init=dict(type='SmoothL1Loss', beta=0.11, loss_weight=0.5),
        loss_bbox_refine=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0),
        transform_method=_transform_method))
        #transform_method='minmax'))
        #transform_method='center_size_istopleft'))
# training and testing settings
train_cfg = dict(
    init=dict(
        assigner=dict(type='PointAssigner', scale=4, pos_num=1, obj_rep=_obj_rep),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    refine=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            overlap_fun='dil_iou_dis',
            obj_rep=_obj_rep),
        allowed_border=-1,
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms_dsiou', iou_thr=0.5, dis_weight=0.7),
    max_per_img=100)
# dataset settings
dataset_type = 'BeikeDataset'
data_root = f'data/beike/processed_{IMAGE_SIZE}/'
img_norm_cfg = dict(
    mean=[ 0, 0, 0, 0.0],
    std=[255.0, 1, 1, 1,], to_rgb=False)
train_pipeline = [
    dict(type='LoadTopviewFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(IMAGE_SIZE, IMAGE_SIZE), keep_ratio=True, obj_dim=_obj_dim),
    dict(type='RandomLineFlip', flip_ratio=0.7, obj_rep=_obj_rep, direction='random'),
    dict(type='RandomRotate', rotate_ratio=1.0, obj_rep=_obj_rep),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadTopviewFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512,512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True, obj_dim=_obj_dim),
            dict(type='RandomLineFlip', obj_rep=_obj_rep),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'json/',
        img_prefix=data_root + 'topview/_train_87_' + DATA,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'json/',
        img_prefix=data_root + 'topview/_test_10_' + DATA,
        pipeline=train_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'json/',
        img_prefix=data_root + 'topview/_test_10_' + DATA,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=20,
    warmup_ratio=1.0 / 3,
    step=[200, 300])
checkpoint_config = dict(interval=10)
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 400
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/{}_strpoints_moment_r50_fpn_1x_{}_{}'.format(_obj_rep, DATA, IMAGE_SIZE)
load_from = None
#load_from ='./checkpoints/strpoints_moment_r50_fpn_1x.pth'
#load_from = './work_dirs/strpoints_moment_r50_fpn_1x/best.pth'
resume_from = None
auto_resume = True
workflow = [('train', 1), ('val', 1)]

