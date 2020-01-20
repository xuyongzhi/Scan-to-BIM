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

# PointAssigner
```
core/bbox/assigners/point_assigner.py
```
- scale=4
- pos_num=1
- self.line_object=True
- gt_bboxes_lvl = (torch.log2(gt_bboxes_wh.max(dim=1)[0] / scale)).int()    
- points_gt_dist = (lvl_points - gt_point).norm(dim=1) / gt_wh.norm(dim=1)  
- No ignored points: each gt box is assigned with one pos point, all others are negative

#  PseudoSampler
```
core/bbox/samplers/pseudo_sampler.py    
```
