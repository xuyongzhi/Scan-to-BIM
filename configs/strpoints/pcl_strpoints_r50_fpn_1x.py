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
import math

voxel_size = [0.04, 0.08][1]

if 1:
  stem_strides = {0.04:4, 0.08:2, 0.16:1}
  batch_size = {0.04:5, 0.08:7}[voxel_size]
if 0:
  stem_strides = {0.04:2, 0.08:1, 0.16:1}
  batch_size = {0.04:2, 0.08:2}[voxel_size]

stem_stride = stem_strides[voxel_size]
#*******************************************************************************
from configs.common import  OBJ_REP, IMAGE_SIZE, TRAIN_NUM, DATA
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
if DATA == 'stanford_pcl_2d':
  dataset_type = 'StanfordPclDataset'
  data_root = f'data/stanford/'
  img_prefix_train = 'train'
  img_prefix_test = 'test'
  ann_file = data_root
  in_channels = 6

if DATA == 'beike_pcl_2d':
  dataset_type = 'BeikePclDataset'
  data_root = f'data/beike/processed_512/'
  ann_file = data_root
  img_prefix_train = 'train'
  img_prefix_test = 'test'
  in_channels = 9

backbone_type = 'VoxResNet'

#*******************************************************************************

norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

point_strides_all = [(2**i)*stem_stride for i in range(4)]
bbp = 32
model = dict(
    type='StrPointsDetector',
    pretrained=None,
    backbone=dict(
        type=backbone_type,
        depth=50,
        in_channels=in_channels,
        num_stages=4,
        out_indices=( 0, 1, 2,),
        frozen_stages=-1,
        style='pytorch',
        stem_stride=stem_stride,
        basic_planes=bbp,
        max_planes=1024),
    neck=dict(
        type='FPN',
        in_channels=[ bbp*4, bbp*8, bbp*16],
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
        point_strides=point_strides_all,
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
        corner_hm_only = False,
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


train_pipeline = None
test_pipeline = train_pipeline

lra = 0.01

data = dict(
    imgs_per_gpu=batch_size,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=ann_file,
        img_prefix=img_prefix_train,
        augment_data=True,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file,
        img_prefix=img_prefix_test,
        augment_data=True,
        pipeline=train_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file,
        img_prefix=img_prefix_test,
        augment_data=False,
        pipeline=test_pipeline))


if DATA == 'beike_pcl_2d':
  # pcl_scope: max=[20.041 15.847  6.531] mean=[10.841 10.851  3.392]
  max_scene_size = [20.48, 20.48, 7.68]
elif DATA == 'stanford_pcl_2d':
  max_scene_size = [10.24, 10.24, 5.12]

auto_scale_vs = False
model['backbone']['voxel_size'] = voxel_size
model['backbone']['full_height'] = max_scene_size[-1]
for split in ['train', 'test', 'val']:
  data[split]['voxel_size'] = voxel_size
  data[split]['auto_scale_vs'] = auto_scale_vs

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
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/R50_fpn'
load_from = None
resume_from = None
auto_resume = True
workflow = [('train', 1), ('val', 1)]
workflow = [('train', 1)]

