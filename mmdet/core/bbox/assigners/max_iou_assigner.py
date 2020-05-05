import torch

from ..geometry import  bbox_overlaps, dilated_bbox_overlaps, \
                        dsiou_bbox_overlaps, corner_overlaps,\
                        dsiou_rotated_3d_bbox
from ..straight_line_distance import line_overlaps
from .assign_result import AssignResult
from .base_assigner import BaseAssigner
import cv2
from obj_geo_utils.obj_utils import OBJ_REPS_PARSE

from configs.common import DEBUG_CFG

class MaxIoUAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, `0`, or a positive integer
    indicating the ground truth index.

    - -1: don't care
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
        gpu_assign_thr (int): The upper bound of the number of GT for GPU
            assign. When the number of gt is above this threshold, will assign
            on CPU device. Negative values mean not assign on CPU.
    """

    def __init__(self,
                 pos_iou_thr,
                 neg_iou_thr,
                 min_pos_iou=.0,
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True,
                 gpu_assign_thr=-1,
                 overlap_fun='iou',
                 obj_rep='box_scope',
                 ref_radius=None):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.overlap_fun = overlap_fun
        assert obj_rep in ['XYXYSin2', 'XYLgWsAsinSin2Z0Z1', 'XYXYSin2WZ0Z1', 'XYLgWsAbsSin2Z0Z1', 'XYDAsinAsinSin2Z0Z1']
        if obj_rep == 'corner':
          assert ref_radius is not None
        self.obj_rep = obj_rep
        self.ref_radius = ref_radius

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None,
               img_meta=None):
        """Assign gt to bboxes.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, 0, or a positive number. -1 means don't care,
        0 means negative sample, positive number is the index (1-based) of
        assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every bbox to -1
        2. assign proposals whose iou with all gts < neg_iou_thr to 0
        3. for each bbox, if the iou with its nearest gt >= pos_iou_thr,
           assign it to that bbox
        4. for each gt bbox, assign its nearest proposals (may be more than
           one) to itself

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.

        Example:
            >>> self = MaxIoUAssigner(0.5, 0.5)
            >>> bboxes = torch.Tensor([[0, 0, 10, 10], [10, 10, 20, 20]])
            >>> gt_bboxes = torch.Tensor([[0, 0, 10, 9]])
            >>> assign_result = self.assign(bboxes, gt_bboxes)
            >>> expected_gt_inds = torch.LongTensor([1, 0])
            >>> assert torch.all(assign_result.gt_inds == expected_gt_inds)
        """
        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
            gt_bboxes.shape[0] > self.gpu_assign_thr) else False

        gt_bboxes_raw = gt_bboxes
        bboxes_raw = bboxes
        # compute overlap and assign gt on CPU when number of GT is large
        if assign_on_cpu:
            device = bboxes.device
            bboxes = bboxes.cpu()
            gt_bboxes = gt_bboxes.cpu()
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu()
            if gt_labels is not None:
                gt_labels = gt_labels.cpu()

        if self.obj_rep == 'box_scope' or self.obj_rep == 'line_scope':
          assert bboxes.shape[1] == 4
          assert gt_bboxes.shape[1] == 4
          if gt_bboxes_ignore is not None:
            assert gt_bboxes_ignore.shape[1] == 4
        elif self.obj_rep == 'XYXYSin2':
          assert self.overlap_fun == 'dil_iou_dis' or self.overlap_fun == 'dil_iou_dis_rotated_3d'
          assert bboxes.shape[1] == 5
          assert gt_bboxes.shape[1] == 5
          if self.overlap_fun == 'dil_iou_dis':
            bboxes = bboxes[:, :4]
            gt_bboxes = gt_bboxes[:, :4]
            if gt_bboxes_ignore is not None:
              assert gt_bboxes_ignore.shape[1] == 5
              gt_bboxes_ignore = gt_bboxes_ignore[:,:4]
          elif self.overlap_fun == 'dil_iou_dis_rotated_3d':
            box_encode_fn = OBJ_REPS_PARSE.encode_obj
            bboxes = box_encode_fn(bboxes, self.obj_rep, 'XYZLgWsHA')
            gt_bboxes = box_encode_fn(gt_bboxes, self.obj_rep, 'XYZLgWsHA')
            if gt_bboxes_ignore is not None:
              assert gt_bboxes_ignore.shape[1] == 5
              gt_bboxes_ignore = box_encode_fn(gt_bboxes_ignore, self.obj_rep, 'XYZLgWsHA')

        elif self.obj_rep == 'XYXYSin2WZ0Z1' or self.obj_rep == 'XYLgWsAbsSin2Z0Z1' or self.obj_rep == 'XYDAsinAsinSin2Z0Z1':
          assert self.overlap_fun == 'dil_iou_dis_rotated_3d'
          assert bboxes.shape[1] == 8
          assert gt_bboxes.shape[1] == 8
          box_encode_fn = OBJ_REPS_PARSE.encode_obj
          bboxes = box_encode_fn(bboxes, self.obj_rep, 'XYZLgWsHA', allow_illegal=True)
          gt_bboxes = box_encode_fn(gt_bboxes, self.obj_rep, 'XYZLgWsHA')
          if gt_bboxes_ignore is not None:
            assert gt_bboxes_ignore.shape[1] == 5
            gt_bboxes_ignore = box_encode_fn(gt_bboxes_ignore, self.obj_rep, 'XYZLgWsHA')

        elif self.obj_rep == 'corner':
          assert gt_bboxes.shape[1] == 2
          #gt_bboxes = gt_bboxes.repeat(1,2)
        else:
          raise NotImplemented

        if self.overlap_fun == 'iou':
          overlaps_fun = bbox_overlaps
        elif self.overlap_fun == 'line':
          overlaps_fun = line_overlaps
        elif self.overlap_fun == 'dil_ou':
          overlaps_fun = dilated_bbox_overlaps
        elif self.overlap_fun == 'dil_iou_dis':
          overlaps_fun = dsiou_bbox_overlaps
        elif self.overlap_fun == 'dil_iou_dis_rotated_3d':
          overlaps_fun = dsiou_rotated_3d_bbox
        elif self.overlap_fun == 'dis':
          overlaps_fun = corner_overlaps
        else:
          print(f'overlaps_fun: {self.overlaps_fun}')
          raise NotImplemented

        if self.overlap_fun == 'dis':
          overlaps = overlaps_fun(gt_bboxes, bboxes[:,:2], self.ref_radius)
        else:
          overlaps = overlaps_fun(gt_bboxes, bboxes)

        if (self.ignore_iof_thr > 0) and (gt_bboxes_ignore is not None) and (
                gt_bboxes_ignore.numel() > 0):
            if self.ignore_wrt_candidates:
                ignore_overlaps = overlaps_fun(
                    bboxes, gt_bboxes_ignore, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = overlaps_fun(
                    gt_bboxes_ignore, bboxes, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)
        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)

        if DEBUG_CFG.PRINT_IOU_ASSIGNER:
          print('\tmax_iou_assigner\t' + str(assign_result))
        if DEBUG_CFG.VISUALIZE_IOU_ASSIGNER:
          show_assign_res(self.obj_rep, bboxes_raw, gt_bboxes_raw, assign_result, img_meta)
        return assign_result

    def assign_wrt_overlaps(self, overlaps, gt_labels=None):
        """Assign w.r.t. the overlaps of bboxes with gts.

        Args:
            overlaps (Tensor): Overlaps between k gt_bboxes and n bboxes,
                shape(k, n).
            gt_labels (Tensor, optional): Labels of k gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_zeros((num_bboxes, ),
                                                     dtype=torch.long)
            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

        # 2. assign negative: below
        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < self.neg_iou_thr)] = 0
        elif isinstance(self.neg_iou_thr, tuple):
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0])
                             & (max_overlaps < self.neg_iou_thr[1])] = 0

        # 3. assign positive: above positive IoU threshold
        pos_inds = max_overlaps >= self.pos_iou_thr
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        # 4. assign fg: for each gt, proposals with highest IoU
        for i in range(num_gts):
            if gt_max_overlaps[i] >= self.min_pos_iou:
                if self.gt_max_assign_all:
                    max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                    assigned_gt_inds[max_iou_inds] = i + 1
                else:
                    assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_zeros((num_bboxes, ))
            pos_inds = torch.nonzero(assigned_gt_inds > 0).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        assign_res = AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels,
        env='MaxIoUAssigner')

        return assign_res

def fix_bboxes_size(bboxes, size=2):
  centroids = (bboxes[:,:2] + bboxes[:,2:]) / 2
  bboxes_ = torch.cat( [centroids - size/2, centroids + size/2], 1 )
  return bboxes_

def show_assign_res(obj_rep, bboxes, gt_bboxes, assign_res, img_meta):
  import mmcv, numpy as np
  from tools.visual_utils import _show_objs_ls_points_ls

  filename = img_meta['filename']
  print(f'\nfilename: {filename}\n')

  if gt_bboxes.shape[1] == 2:
    gt_bboxes = corner_as_bboxes(gt_bboxes, 2)
    bboxes = corner_as_bboxes(bboxes[:,:2], 1)

  #bboxes = fix_bboxes_size(bboxes)

  gt_inds_valid = assign_res.gt_inds_valid.cpu().data.numpy()
  pos_inds = assign_res.pos_inds.cpu().data.numpy()
  pos_num = pos_inds.shape[0]
  max_overlaps = assign_res.max_overlaps.cpu().data.numpy()[pos_inds]

  gt_num = gt_bboxes.shape[0]
  gt_inds_invalid = [i for i in range(gt_num) if i not in gt_inds_valid]
  gt_num_missed = len(gt_inds_invalid)

  img = np.zeros((512,512,3), dtype=np.uint8)
  gt_bboxes_ = gt_bboxes.cpu().data.numpy()
  bboxes_ = bboxes.cpu().data.numpy()
  pos_bboxes_ = bboxes_[pos_inds]
  print(f'gt_num: {gt_num}, pos_num: {pos_num}, miss_gt_num: {gt_num_missed}')

  #_show_objs_ls_points_ls(img, [gt_bboxes_, bboxes_],obj_colors=['red', 'green'], obj_rep=obj_rep)
  _show_objs_ls_points_ls(img, [gt_bboxes_, pos_bboxes_],obj_colors=['red', 'green'] , obj_rep=obj_rep)
  _show_objs_ls_points_ls(img, [gt_bboxes_, gt_bboxes_[gt_inds_invalid]],obj_colors=['red', 'green'], obj_rep=obj_rep )

  for i in range(gt_num):
    mask = gt_inds_valid == i
    ni = mask.sum()
    print(f'\npos bboxes num: {ni}')
    if ni == 0:
      continue
    overlaps_i = max_overlaps[mask]
    mol = overlaps_i.mean()
    ol_str = ','.join([f'{o:.2}' for o in overlaps_i])
    print(f'overlaps: mean={mol:.2} : {ol_str}')

    pos_ids_i = pos_inds[mask]
    _show_objs_ls_points_ls(img, [gt_bboxes_, gt_bboxes_[i:i+1], bboxes_[pos_ids_i]],obj_colors=['red', 'green', 'yellow'], obj_rep=obj_rep)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    pass
  pass

def corner_as_bboxes(corners, s):
  return torch.cat( [corners - s, corners + s], dim=1 )



