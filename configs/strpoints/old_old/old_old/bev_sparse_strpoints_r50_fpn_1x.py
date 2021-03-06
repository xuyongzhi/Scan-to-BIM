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

stem_stride = 4
TOPVIEW = 'VerD' # better
#TOPVIEW = 'All'
#*******************************************************************************
from configs.common import  OBJ_REP, IMAGE_SIZE, DATA, SPARSE_BEV
assert SPARSE_BEV==1
assert 'pcl' not in DATA
_obj_rep = OBJ_REP
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
        type='Sparse3DResNet',
        depth=50,
        in_channels=4,
        num_stages=4,
        out_indices=( 0, 1, 2,),
        frozen_stages=-1,
        style='pytorch',
        stem_stride=stem_stride,
        basic_planes=64,
        max_planes=2048),
    neck=dict(
        type='FPN',
        in_channels=[ 256, 512, 1024],
        out_channels=256,
        start_level=0,
        add_extra_convs=True,
        num_outs=5,
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
        point_strides=[4, 8, 16, 32, 64],
        point_base_scale=1,
        norm_cfg=norm_cfg,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,),
        cls_types=['refine', 'final'],
        loss_bbox_init=dict(type='SmoothL1Loss', beta=0.11, loss_weight=0.5),
        loss_bbox_refine=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0),
        transform_method=_transform_method,
        dcn_zero_base=False,
        corner_hm = True,
        corner_hm_only = True,
        )
    )
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
            pos_iou_thr=0.7,
            neg_iou_thr=0.4,
            min_pos_iou=0.3,
            ignore_iof_thr=-1,
            overlap_fun='dil_iou_dis',
            obj_rep=_obj_rep),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    corner=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.6,
            neg_iou_thr=0.1,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            overlap_fun='dis',
            ref_radius=2,
            obj_rep='corner'),
        allowed_border=-1,
        pos_weight=-1,
        gaussian_weight=True,
        debug=False),
        )
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.2,
    nms=dict(type='nms_dsiou', iou_thr=0.5, dis_weight=0.7),
    max_per_img=100)
# dataset settings
dataset_type = 'BeikeDataset'
data_root = f'data/beike/processed_{IMAGE_SIZE}/'
#img_norm_cfg = dict(
#    mean=[  0, 0,0,0],
#    std=[ 255, 1,1,1 ], to_rgb=False, method='raw')

img_norm_cfg = dict(
    mean=[  4.753,  0.,     0.,    0.],
    std=[ 16.158,  0.155,  0.153,  0.22], to_rgb=False, method='rawstd') # better

#img_norm_cfg = dict(
#    mean=[4.753, 0.044, 0.043, 0.102],
#    std=[ 16.158,  0.144,  0.142,  0.183], to_rgb=False, method='abs')
#
#img_norm_cfg = dict(
#    mean=[4.753, 11.142, 11.044, 25.969],
#    std=[ 16.158, 36.841, 36.229, 46.637], to_rgb=False, method='abs255')

train_pipeline = [
    dict(type='LoadTopviewFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(IMAGE_SIZE, IMAGE_SIZE), keep_ratio=True, obj_dim=_obj_dim),
    dict(type='RandomLineFlip', flip_ratio=0.6, obj_rep=_obj_rep, direction='random'),
    dict(type='RandomRotate', rotate_ratio=0.8, obj_rep=_obj_rep),
    dict(type='NormalizeTopview', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadTopviewFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512,512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True, obj_dim=_obj_dim),
            dict(type='RandomLineFlip', obj_rep=_obj_rep),
            dict(type='RandomRotate', rotate_ratio=1.0, obj_rep=_obj_rep),
            dict(type='NormalizeTopview', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
if IMAGE_SIZE == 512:
  batch_size = 6
  lra = 0.01
if IMAGE_SIZE == 1024:
  batch_size = 1
  lra = 0.005

test_dir=data_root + f'TopView_{TOPVIEW}/test.txt'
input_style = ['bev_img', 'bev_sparse'][1]
data = dict(
    imgs_per_gpu=batch_size,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'json/',
        img_prefix=data_root + f'TopView_{TOPVIEW}/train.txt',
        input_style = input_style,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'json/',
        img_prefix=data_root + f'TopView_{TOPVIEW}/test.txt',
        input_style = input_style,
        pipeline=train_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'json/',
        img_prefix=test_dir,
        input_style = input_style,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=lra, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
total_epochs = 800
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=20,
    warmup_ratio=1.0 / 3,
    step=[int(total_epochs*0.7), int(total_epochs*0.85)])
checkpoint_config = dict(interval=20)
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/TPV_r50_fpn'
load_from = None
#load_from ='./checkpoints/strpoints_moment_r50_fpn_1x.pth'
#load_from = f'{work_dir}/best.pth'
#load_from = f'./work_dirs/T1_r50_fpn_lscope_istopleft_refine_512_VerD_bs1_lr10_RA_Normrawstd/best.pth'
resume_from = None
auto_resume = True
workflow = [('train', 1), ('val', 1)]
workflow = [('train', 1),]

