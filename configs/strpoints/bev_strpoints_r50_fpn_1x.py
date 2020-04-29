# mdel settings
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

TOPVIEW = 'VerD' # better
#*******************************************************************************
from configs.common import DIM_PARSE
IMAGE_SIZE = DIM_PARSE.IMAGE_SIZE
DATA = 'beike2d'
#DATA = 'stanford2d'
classes= ['wall']

if DATA == 'beike2d':
  _obj_rep = 'RoLine2D_UpRight_xyxy_sin2a'
  _obj_rep = 'XYLgWsAsinSin2Z0Z1'
elif DATA == 'stanford2d':
  _obj_rep = 'RoLine2D_UpRight_xyxy_sin2a'
  _obj_rep = 'XYLgWsAsinSin2Z0Z1'

if _obj_rep == 'RoLine2D_UpRight_xyxy_sin2a':
  num_ps_long_axis = 9
  overlap_fun='dil_iou_dis'
elif _obj_rep == 'XYLgWsAsinSin2Z0Z1':
  num_ps_long_axis = 5
  overlap_fun='dil_iou_dis_rotated_3d'

dim_parse = DIM_PARSE(_obj_rep, len(classes)+1)
_obj_dim = dim_parse.OBJ_DIM

if _obj_rep == 'RoLine2D_UpRight_xyxy_sin2a':
  _transform_method='moment_lscope_istopleft'
if _obj_rep == 'XYLgWsAsinSin2Z0Z1':
  _transform_method='moment_LWAsS2ZZ'
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
        out_indices=( 0, 1, 2, 3),
        frozen_stages=-1,
        style='pytorch',
        basic_planes=64,
        max_planes=2048),
    neck=dict(
        type='FPN',
        in_channels=[ 256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs=True,
        num_outs=4,
        norm_cfg=norm_cfg),
    bbox_head=dict(
        type='StrPointsHead',
        obj_rep=_obj_rep,
        num_classes=len(classes) + 1,
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        stacked_convs=3,
        num_points=9,
        num_ps_long_axis = num_ps_long_axis,
        gradient_mul=0.1,
        point_strides=[4, 8, 16, 32],
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
        corner_hm = False,
        corner_hm_only = False,
        move_points_to_center = 0,
        relation_cfg=dict(enable=0,
                          stage='refine',
                          score_threshold=0.2,
                          max_relation_num=120),
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
            min_pos_iou=0.15,
            ignore_iof_thr=-1,
            overlap_fun=overlap_fun,
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
    max_per_img=150)
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
    dict(type='PadToSameHW_ForRotation',obj_rep=_obj_rep,pad_border_make_bboxes_pos=True),
    dict(type='ResizeImgLine', obj_rep=_obj_rep, img_scale=(IMAGE_SIZE, IMAGE_SIZE), keep_ratio=True, obj_dim=_obj_dim),
    dict(type='RandomLineFlip', flip_ratio=0.6, obj_rep=_obj_rep, direction='random'),
    dict(type='RandomRotate', rotate_ratio=0.8, obj_rep=_obj_rep),
    dict(type='NormalizeTopview', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_relations']),
]
test_pipeline = [
    dict(type='LoadTopviewFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(IMAGE_SIZE, IMAGE_SIZE),
        flip=False,
        transforms=[
            dict(type='PadToSameHW_ForRotation', obj_rep=_obj_rep, pad_border_make_bboxes_pos=True),
            dict(type='ResizeImgLine', obj_rep=_obj_rep, keep_ratio=True, obj_dim=_obj_dim),
            dict(type='RandomLineFlip', obj_rep=_obj_rep),
            dict(type='RandomRotate', rotate_ratio=0.0, obj_rep=_obj_rep),
            dict(type='NormalizeTopview', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img', 'gt_bboxes', 'gt_labels']),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_relations']),
        ])
]

filter_edges=True
# dataset settings
if DATA == 'beike2d':
  dataset_type = 'BeikeDataset'
  data_root = f'data/beike/processed_{IMAGE_SIZE}/'
  ann_file = data_root + 'json/'
  img_prefix_train = data_root + f'TopView_{TOPVIEW}/train.txt'
  img_prefix_test = data_root + f'TopView_{TOPVIEW}/test.txt'
  img_prefix_test = img_prefix_train
elif DATA == 'stanford2d':
  dataset_type = 'Stanford_2D_Dataset'
  ann_file = 'data/stanford/'
  img_prefix_train = 'train'
  img_prefix_test = 'test'

data = dict(
    imgs_per_gpu=7,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        obj_rep = _obj_rep,
        ann_file=ann_file,
        img_prefix=img_prefix_train,
        pipeline=train_pipeline,
        classes=classes,
        filter_edges=filter_edges),
    val=dict(
        type=dataset_type,
        obj_rep = _obj_rep,
        ann_file=ann_file,
        img_prefix=img_prefix_test,
        pipeline=train_pipeline,
        classes=classes,
        filter_edges=filter_edges),
    test=dict(
        type=dataset_type,
        obj_rep = _obj_rep,
        ann_file=ann_file,
        img_prefix=img_prefix_test,
        pipeline=test_pipeline,
        classes=classes,
        filter_edges=filter_edges))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
total_epochs = 1510
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=20,
    warmup_ratio=1.0 / 3,
    step=[int(total_epochs*0.4), int(total_epochs*0.7)])
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
work_dir = f'./work_dirs/{DATA[0]}TPV_r50_fpn'
if DATA == 'beike2d':
  load_from = './checkpoints/beike/Apr23_WaDo_Bev.pth'
  #load_from ='./checkpoints/beike/Apr16FineTuneApr12_Fpn44_Bp32.pth'
elif DATA == 'stanford2d':
  load_from = './checkpoints/sfd/Apr26_wabeco_Bev.pth'
load_from = None
resume_from = None
auto_resume = True
workflow = [('train', 5), ('val', 1)]
if 1:
  data['workers_per_gpu'] = 0
  workflow = [('train', 1),]
  checkpoint_config = dict(interval=100)

