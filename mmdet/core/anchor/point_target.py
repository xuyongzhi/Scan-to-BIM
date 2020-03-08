import torch

from ..bbox import PseudoSampler, assign_and_sample, build_assigner
from ..utils import multi_apply

from mmdet import debug_tools
DEBUG = 0
SHOW_CENTERNESS = 0

def point_target(proposals_list,
                 valid_flag_list,
                 gt_bboxes_list,
                 img_metas,
                 cfg,
                 gt_bboxes_ignore_list=None,
                 gt_labels_list=None,
                 label_channels=1,
                 sampling=True,
                 unmap_outputs=True,
                 flag=''):
    """Compute corresponding GT box and classification targets for proposals.

    Args:
        points_list (list[list]): Multi level points of each image.
        valid_flag_list (list[list]): Multi level valid flags of each image.
          padded part is invalid
        gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
        img_metas (list[dict]): Meta info of each image.
        cfg (dict): train sample configs.


    (1) flag == 'init'
    points_list[i]: [n,3] 3 = center + stride

    (2) flag == 'refine'
    points_list[i]: [n,5] 5 = line

    Returns:
        tuple
    """
    num_imgs = len(img_metas)
    assert len(proposals_list) == len(valid_flag_list) == num_imgs

    #if DEBUG:
    #  debug_tools.show_multi_ls_shapes([proposals_list, gt_bboxes_list], ['proposals_list','gt_bboxes_list'], f'{flag} point_target input')

    # points number of multi levels
    num_level_proposals = [points.size(0) for points in proposals_list[0]]

    # concat all level points and flags to a single tensor
    for i in range(num_imgs):
        assert len(proposals_list[i]) == len(valid_flag_list[i])
        proposals_list[i] = torch.cat(proposals_list[i])
        valid_flag_list[i] = torch.cat(valid_flag_list[i])

    # compute targets for each image
    if gt_bboxes_ignore_list is None:
        gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
    if gt_labels_list is None:
        gt_labels_list = [None for _ in range(num_imgs)]

    if DEBUG and flag=='refine':
      debug_tools.show_multi_ls_shapes([proposals_list, gt_bboxes_list], ['proposals_list','gt_bboxes_list'], f'{flag} point_target (2)')
      pass

    (all_labels, all_label_weights, all_bbox_gt, all_proposals,
     all_proposal_weights, pos_inds_list, neg_inds_list) = multi_apply(
         point_target_single,
         proposals_list,
         valid_flag_list,
         gt_bboxes_list,
         gt_bboxes_ignore_list,
         gt_labels_list,
         img_metas,
         cfg=cfg,
         label_channels=label_channels,
         sampling=sampling,
         unmap_outputs=unmap_outputs,)


    # no valid points
    if any([labels is None for labels in all_labels]):
        return None
    # sampled points of all images
    num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
    num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
    num_total_gt = sum([gtb.shape[0] for gtb in gt_bboxes_list])
    labels_list = images_to_levels(all_labels, num_level_proposals)
    label_weights_list = images_to_levels(all_label_weights,
                                          num_level_proposals)
    bbox_gt_list = images_to_levels(all_bbox_gt, num_level_proposals)
    proposals_list = images_to_levels(all_proposals, num_level_proposals)
    proposal_weights_list = images_to_levels(all_proposal_weights,
                                             num_level_proposals)

    if DEBUG and flag=='refine':
      debug_tools.show_multi_ls_shapes([all_bbox_gt], ['all_bbox_gt'], f'{flag} point_target (3)')
      print(f'pos:{num_total_pos}\nneg:{num_total_neg}\ngt:{num_total_gt}')
      show_point_targets(pos_inds_list, all_proposals, gt_bboxes_list, flag)
      #for labels_, flag in zip([labels_list, label_weights_list, proposal_weights_list],
      #                        ['labels_list', 'label_weights_list', 'proposal_weights_list']):
      #  show_weights(labels_,flag)
      pass

      #debug_tools.show_multi_ls_shapes([bbox_gt_list], ['bbox_gt_list'], f'{flag} point_target (4)')

    return (labels_list, label_weights_list, bbox_gt_list, proposals_list,
            proposal_weights_list, num_total_pos, num_total_neg)


def images_to_levels(target, num_level_grids):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_level_grids:
        end = start + n
        level_targets.append(target[:, start:end].squeeze(0))
        start = end
    return level_targets


def point_target_single(flat_proposals,
                        valid_flags,
                        gt_bboxes,
                        gt_bboxes_ignore,
                        gt_labels,
                        img_meta,
                        cfg,
                        label_channels=1,
                        sampling=True,
                        unmap_outputs=True,):
    inside_flags = valid_flags
    if not inside_flags.any():
        return (None, ) * 7
    # assign gt and sample proposals
    proposals = flat_proposals[inside_flags, :]

    if sampling:
        assign_result, sampling_result = assign_and_sample(
            proposals, gt_bboxes, gt_bboxes_ignore, None, cfg)
    else:
        bbox_assigner = build_assigner(cfg.assigner)
        assign_result = bbox_assigner.assign(proposals, gt_bboxes,
                                             gt_bboxes_ignore, gt_labels,
                                             img_meta)
        bbox_sampler = PseudoSampler()
        sampling_result = bbox_sampler.sample(assign_result, proposals,
                                              gt_bboxes)

    num_valid_proposals = proposals.shape[0]
    box_cn = gt_bboxes.shape[1]
    bbox_gt = proposals.new_zeros([num_valid_proposals, box_cn])
    pos_proposals = torch.zeros_like(proposals)
    proposals_weights = proposals.new_zeros([num_valid_proposals, box_cn])
    labels = proposals.new_zeros(num_valid_proposals, dtype=torch.long)
    label_weights = proposals.new_zeros(num_valid_proposals, dtype=torch.float)

    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds
    if len(pos_inds) > 0:
        pos_gt_bboxes = sampling_result.pos_gt_bboxes
        bbox_gt[pos_inds, :] = pos_gt_bboxes
        pos_proposals[pos_inds, :] = proposals[pos_inds, :]
        proposals_weights[pos_inds, :] = 1.0
        if gt_labels is None:
            labels[pos_inds] = 1
        else:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = cfg.pos_weight
    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0

    # map up to original set of proposals
    if unmap_outputs:
        num_total_proposals = flat_proposals.size(0)
        labels = unmap(labels, num_total_proposals, inside_flags)
        label_weights = unmap(label_weights, num_total_proposals, inside_flags)
        bbox_gt = unmap(bbox_gt, num_total_proposals, inside_flags)
        pos_proposals = unmap(pos_proposals, num_total_proposals, inside_flags)
        proposals_weights = unmap(proposals_weights, num_total_proposals,
                                  inside_flags)

    if cfg['assigner']['obj_rep'] == 'corner':
        assert 'gaussian_weight' in cfg and cfg['gaussian_weight']==True
        assert gt_bboxes.shape[1] == 2
        gt_centerness = get_gaussian_weights( gt_bboxes, flat_proposals[:,:2], cfg['assigner']['ref_radius'] )
        pos_proposals = torch.cat([pos_proposals, gt_centerness.reshape(-1,1)], dim=1)
        if SHOW_CENTERNESS:
          from mmdet.debug_tools import show_heatmap
          show_heatmap(gt_centerness.reshape(128,128), (512,512), gt_corners=gt_bboxes)
          show_heatmap(labels.reshape(128,128), (512,512), gt_corners=gt_bboxes)
          show_heatmap(label_weights.reshape(128,128), (512,512), gt_corners=gt_bboxes )
          show_heatmap(proposals_weights[:,0].reshape(128,128), (512,512), gt_corners=gt_bboxes)
          import pdb; pdb.set_trace()  # XXX BREAKPOINT
          pass
    return (labels, label_weights, bbox_gt, pos_proposals, proposals_weights,
            pos_inds, neg_inds)


def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret


def get_gaussian_weights(gt_corners, prop_corners, ref_radius):
  from mmdet.core.bbox.geometry import corner_overlaps
  distances = corner_overlaps(gt_corners, prop_corners, ref_radius)
  distances = distances.max(dim=0)[0]
  #show_weights([distances], 'distances')
  return distances

def show_point_targets(pos_inds_list, all_proposals, gt_bboxes_list, flag):
  from mmdet.debug_tools import show_lines, show_points, _show_lines_ls_points_ls
  from configs.common import IMAGE_SIZE
  import numpy as np
  for (pos_inds, proposals, bbox_gt) in zip(pos_inds_list, all_proposals, gt_bboxes_list):
    proposals = proposals[pos_inds].cpu().data.numpy()
    bbox_gt = bbox_gt.cpu().data.numpy()
    if proposals.shape[1] == 3:
      points = proposals[:,:2]
    if proposals.shape[1] == 5:
      points = (proposals[:,0:2] + proposals[:,2:4])/2
      _show_lines_ls_points_ls((IMAGE_SIZE, IMAGE_SIZE), [bbox_gt, proposals], line_colors=['red', 'green'],line_thickness=[1,1],  out_file='./'+flag+'_proposals.png')

    if bbox_gt.shape[1] == 5:
      _show_lines_ls_points_ls((IMAGE_SIZE, IMAGE_SIZE), [bbox_gt], [points], line_colors='random', point_colors='random', line_thickness=2, point_thickness=3, out_file='./'+flag+'_centroids.png')
    if bbox_gt.shape[1] == 2:
      points = proposals[:,:2]
      _show_lines_ls_points_ls((IMAGE_SIZE, IMAGE_SIZE), None, [bbox_gt, points], point_colors=['red', 'green'], point_thickness=[2,1], out_file='./'+flag+'_proposal_corners.png')
    pass

def show_weights(weights_list, flag):
  from mmdet.debug_tools import show_lines, show_points
  from configs.common import IMAGE_SIZE
  import numpy as np
  import mmcv
  print('\n\n\n')
  print(flag)
  for weights in weights_list:
    if weights.dim() > 1:
      weights = weights[:,0]
    s = np.sqrt(weights.shape[0]).astype(np.int32)
    try:
      img = weights.reshape(s,s).cpu().data.numpy().astype(np.float32)
    except:
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass
    mmcv.imshow(img)
  pass


