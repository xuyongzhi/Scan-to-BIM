# RES
```
8/Mar T90_r50_fpn_lscope_istopleft_refine_final_512_VerD_bs6_lr10_RA_Normrawstd_ChmR2P1N1_Rfiou743_Fpn34_Pbs1
```
|config | eval set | corner prec-recall| line prec-recall |
|-|-|-|-|
|Rfiou743_Fpn34_Pbs1 | test, composite, no rotation    | 0.828 - 0.82  |0.694 - 0.638  |
|Rfiou743_Fpn34_Pbs1 | test, line_ave, no rotation     | 0.842 - 0.836 |0.746 - 0.649  |
|Rfiou743_Fpn34_Pbs1 | test, 1 stage, no rotation | 0.715 - 0.833 |0.637 - 0.545  |
| | | |
|Rfiou743_Fpn34_Pbs1 | train, composite, no rotation    | 0.909 - 0.93  |  0.826 - 0.827 |
|Rfiou743_Fpn34_Pbs1 | train, line_ave, no rotation     | 0.932 - 0.926 | 0.908 - 0.822 |
|Rfiou743_Fpn34_Pbs1 | train, 1 stage, no rotation | 0.814 - 0.927 | 0.801 - 0.722  |


|config | eval set | corner prec-recall| line prec-recall |
|-|-|-|-|
| Rfiou743_Fpn34_Pbs2 | test, line_ave, no rotation     |  0.837 - 0.826 | 0.754 - 0.628  |
| Rfiou743_Fpn34_Pbs2 | train, line_ave, no rotation     | 0.918 - 0.91 | 0.891 - 0.786 |
| | | |
| Rfiou743_Fpn45_Pbs2 | test, line_ave, no rotation     |  0.821 - 0.813 | 0.748 - 0.622  |
| Rfiou743_Fpn45_Pbs2 | train, line_ave, no rotation     | 0.924 - 0.924 | 0.892 - 0.805 |

# Questions
* a big issue is that the refine stage cannot remove false positive of the first stage
* base_scale=2 seems not right
* Is the max_iou assigner configuration not good, num_total_pos is too large
* try dcn offset == reppoints location
* The weights for refine classification and final are shared
* No ignored points: each gt box is assigned with one pos point, all others are negative
* The direction of normal. Does the positive meaning is learnable. After rotation, is the direction should be updated.    
* image size = 512, max feature size=128. Is it better to use max feature size=256

# Big ideas
* use guassion weights for initial stage
* only use two reppoints and sample the other 7 based on the two
* compare performance of feeding images and point clouds
* compare performace of feeding images of different resolution
*
# Items need to be further researched
- voxel_resolution = [512,512] is not fully used. 
- apis/train.py/batch_processor num_samples
- conv type for MinkConv
- vox_resnet weight init not implemented
- group norm for vox_resnet not implemented
- When workers_per_gpu=0, rotate test doest not work.
- There are some bad annotations, such as fully occuled, circule walls    


# Debug
- PointAssigner
- MaxIouAssigner
- Out key points:
        * mmdet/core/post_processing/bbox_nms.py
        * mmdet/models/anchor_heads/reppoints_head.py
        * mmdet/models/anchor_heads/strpoints_head.py
        * mmdet/models/detectors/base.py  show_det_lines_1by1  or show_det_lines
- show 2D gaussian centerness  
    mmdet/core/anchor/point_target.py
# Line object
- box size for point distance normalization
```
(1) datasets/pipelines/transforms.py
        flipped[:,-1] = -flipped[:,-1]

(2) core/bbox/assigners/point_assigner.py
From: gt_bboxes_wh = (gt_bboxes[:, 2:] - gt_bboxes[:, :2]).clamp(min=1e-6)
To:   gt_bboxes_wh = (gt_bboxes[:, 2:] - gt_bboxes[:, :2]).norm(dim=1).clamp(min=1e-6)

(3) models/anchor_heads/strpoints_head.py
angles = angle_from_vecs_to_vece(vec_start, vec_pts, scope_id=1)
istoplefts_ = torch.sin(2*angles)

(4) core/bbox/assigners/max_iou_assigner.py
From : bbox_overlaps
To   : line_overlaps in straight_line_distance_torch.py
```

## rep points order
- Direct out from head is y_first: pts_preds_init, pts_preds_refine. They are feed into loss, get_bbox
- input of self.loss_single is x_first, achieved by self.offset_to_pts: pts_coordinate_preds_init, pts_coordinate_preds_refinebackbone
- deformable offset is y_first

# Framework
- img: [512,512]
- feature: [ [5, 128, 128], [5, 64, 64], [5, 32, 32], [5, 16, 16] ] 
- pts_preds_init:  [ [18, 128, 128], [18, 64, 64], [18, 32, 32], [18, 16, 16] ] 
- pts_preds_refine:  [ [18, 128, 128], [18, 64, 64], [18, 32, 32], [18, 16, 16] ] 
- bbox_preds: [ [5, 128, 128], [5, 64, 64], [5, 32, 32], [5, 16, 16] ]  = [5, 21760]
- Top 1000 scores on each level    
- nsm input: [3256,6] = 1000 + 1000 + 1000 + 256
- det: [100,6]
# Order of 2 points in a line
- the one with smaller x^2+y^2 is the first one
```
1) beike_data_utils/beike_utils.py : sort_2points_in_line
2) datasets/pipelines/transforms.py/RandomLineFlip
```

# point_target
```
A function in core/anchor/point_target.py
```
- Compute corresponding GT box and classification targets for proposals. 
- Call PointAssiger to generate targets from gt_bboxes
- Call PseudoSampler to sample the targets

# PointAssigner (Initial Stage)
```
core/bbox/assigners/point_assigner.py

Save the results in AssignResult
```
- scale=4
- pos_num=1
- self.line_object=True
- gt_bboxes_lvl = (torch.log2(gt_bboxes_wh.max(dim=1)[0] / scale)).int()    
- points_gt_dist = (lvl_points - gt_point).norm(dim=1) / gt_wh.norm(dim=1)  
- No ignored points: each gt box is assigned with one pos point, all others are negative

# MaxIoUAssigner (Second Stage)
```
core/bbox/assigners/max_iou_assigner.py
```
- pos_iou_thr=0.5  
- neg_iou_thr=0.4  
- min_pos_iou=0   
- ignore_iof_thr=-1  
- overlap_fun='aug_iou_dis'  
 
### Overlap fuction
- IoU, IoF, Aug_IoU, AugIoU_Dis
```
core/bbox/geometry.py
```
- line_overlaps
```
ore/bbox/straight_line_distance.py
```


# AssignResult
```
core/bbox/assigners/assign_result.py
```
record results of point/box assigner
- called in point_assigner
- called in max_iou_assigner
   

## Two cases that losts gt
* Initial stage
```
core/bbox/assigners\point_assigner.py
```
* Refinement stage
```
core/bbox/assigners/max_iou_assigner.p
```

#  PseudoSampler
```
core/bbox/samplers/pseudo_sampler.py    
```
...

# Data Augmentation
- RandomLineFlip
```
datasets/pipelines/transforms.py/RandomLineFlip
```
flip line instead of box, sort the 2 points of line after fliping


# img shape for pcl
- get_points in strpoints_head.py
```
h, w, _ = img_meta['pad_shape']
voxel_zero_offset_i = int(np.ceil( img_meta['voxel_zero_offset'] * 2 / point_stride ))
feat_w += voxel_zero_offset_i
valid_feat_h = min(int(np.ceil(h / point_stride)), feat_h)
```
