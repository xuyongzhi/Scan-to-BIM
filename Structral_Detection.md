# Questions
* No ignored points: each gt box is assigned with one pos point, all others are negative

# Big ideas
* compare performance of feeding images and point clouds
* compare performace of feeding images of different resolution
*
# Items need to be further researched
- num_points
- point_strides
- point_base_scale
- image mean is too small    


# Line object
- box size for point distance normalization
```
(1) core/bbox/assigners/point_assigner.py
From: gt_bboxes_wh = (gt_bboxes[:, 2:] - gt_bboxes[:, :2]).clamp(min=1e-6)
To:   gt_bboxes_wh = (gt_bboxes[:, 2:] - gt_bboxes[:, :2]).norm(dim=1).clamp(min=1e-6)

(2) core/bbox/assigners/max_iou_assigner.py
From : bbox_overlaps
To   : line_overlaps in straight_line_distance_torch.py
```

# Order of 2 points in a line
- the one with smaller x^2+y^2 is the first one
```
1) beike_data_utils/beike_utils.py : sort_2points_in_line
2) datasets/pipelines/transforms.py/RandomLineFlip
```
- 

# point_target
```
A function in core/anchor/point_target.py
```
- Compute corresponding GT box and classification targets for proposals. 
- Call PointAssiger to generate targets from gt_bboxes
- Call PseudoSampler to sample the targets

# PointAssigner
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

# AssignResult
```
core/bbox/assigners/assign_result.py
```
- called in point_assigner
- called in max_iou_assigner

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


