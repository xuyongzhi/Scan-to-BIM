# model settings
''' modified
  num_points
'''
''' # pedding
  num_outs
  assigner
  img_norm_cfg
  transform_method
'''
import math
from configs.common import  OBJ_REP, IMAGE_SIZE, SPARSE_BEV
DATA = 'beike_pcl_2d'
classes= ['wall']

voxel_size = [0.02, 0.04, 0.08][1]
stem_stride = {0.02:2, 0.04:2, 0.08:1}[voxel_size] * 2

if DATA == 'beike_pcl_2d':
  # pcl_scope: max=[20.041 15.847  6.531] mean=[10.841 10.851  3.392]
  max_height = 7.68
  max_height = 5
elif DATA == 'stanford_pcl_2d':
  max_height = 5.12

batch_size = 4

stem_stride_z = 8
z_stride = 3
z_strides = (z_stride,) * 4
backbone_out_indices = (0,1,2,3)
stem_z_out_dim = max_height/voxel_size/stem_stride_z
z_out_dims  = [stem_z_out_dim]
for zs in z_strides:
  z_out_dims.append( z_out_dims[-1]/zs )
z_out_dims = tuple([int(math.ceil( d )) for d in z_out_dims])
#*******************************************************************************
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
  ann_file = data_root
  in_channels = 6

if DATA == 'beike_pcl_2d':
  dataset_type = 'BeikePclDataset'
  data_root = f'data/beike/processed_512/'
  ann_file = data_root + 'json/'
  if not SPARSE_BEV:
    in_channels = 9
  else:
    in_channels = 4

backbone_type = 'Sparse3DResNet'

#*******************************************************************************
max_planes = 1024
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
        z_strides=z_strides,
        out_indices=backbone_out_indices,
        frozen_stages=-1,
        style='pytorch',
        stem_stride=stem_stride,
        stem_stride_z = stem_stride_z,
        basic_planes=bbp,
        z_out_dims = z_out_dims,
        max_planes=max_planes),
    neck=dict(
        type='FPN_Dense3D',
        in_channels=[ bbp*4, bbp*8, bbp*16, bbp*32],
        out_channels=256,
        start_level=0,
        add_extra_convs=True,
        num_outs=4,
        norm_cfg = norm_cfg,
        z_out_dims = z_out_dims[1:],
        z_stride = z_stride,
        ),
    bbox_head=dict(
        type='StrPointsHead',
        num_classes=1+len(classes),
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
        corner_hm = False,
        corner_hm_only = False,
        move_points_to_center = False,
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


lra = 0.01


max_footprint_for_scale = 160 # 200
max_num_points = 20 * 10000
data = dict(
    imgs_per_gpu=batch_size,
    workers_per_gpu=3,
    train=dict(
        type=dataset_type,
        ann_file=ann_file,
        img_prefix=data_root + f'ply/train.txt',
        voxel_size=voxel_size,
        augment_data=True,
        data_types = ['color', 'norm', 'xyz'],
        max_num_points=max_num_points,
        max_footprint_for_scale=max_footprint_for_scale,
        filter_edges=True,
        classes = classes,
        pipeline=None),
    val=None,
    test=None,
)
data['val'] = data['train'].copy()
data['val']['img_prefix'] = data_root + f'ply/test.txt'
data['test'] = data['val'].copy()


model['backbone']['voxel_size'] = voxel_size

# optimizer
optimizer = dict(type='SGD', lr=lra, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
total_epochs = 410
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
work_dir = f'./work_dirs/R50_fpn'
load_from = None
resume_from = None
auto_resume = True
workflow = [('train', 1), ('val', 1)]
#workflow = [('train', 1),]

