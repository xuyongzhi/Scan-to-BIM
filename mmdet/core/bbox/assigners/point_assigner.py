import torch

from .assign_result import AssignResult
from .base_assigner import BaseAssigner

from configs.common import DEBUG_CFG
DEBUG = 0

class PointAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each point.

    Each proposals will be assigned with `0`, or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    """

    def __init__(self, scale=4, pos_num=3, obj_rep=''):
        assert obj_rep in ['XYXYSin2', 'XYXYSin2WZ0Z1', 'XYLgWsAsinSin2Z0Z1', 'XYLgWsAbsSin2Z0Z1']
        self.scale = scale
        self.pos_num = pos_num
        self.obj_rep = obj_rep

    def assign(self, points, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None,
               img_meta=None):
        """Assign gt to points.

        This method assign a gt bbox to every points set, each points set
        will be assigned with  0, or a positive number.
        0 means negative sample, positive number is the index (1-based) of
        assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every points to 0
        2. A point is assigned to some gt bbox if
            (i) the point is within the k closest points to the gt bbox
            (ii) the distance between this point and the gt is smaller than
                other gt bboxes

        Args:
            points (Tensor): points to be assigned, shape(n, 3) while last
                dimension stands for (x, y, stride).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
                NOTE: currently unused.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        num_points = points.shape[0]
        num_gts = gt_bboxes.shape[0]

        if num_gts == 0 or num_points == 0:
            # If no truth assign everything to the background
            assigned_gt_inds = points.new_full((num_points, ),
                                               0,
                                               dtype=torch.long)
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = points.new_zeros((num_points, ),
                                                   dtype=torch.long)
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        points_xy = points[:, :2]
        points_stride = points[:, 2]
        points_lvl = torch.log2(
            points_stride).int()  # [3...,4...,5...,6...,7...]
        lvl_min, lvl_max = points_lvl.min(), points_lvl.max()

        # assign gt box
        if self.obj_rep == 'box_scope':
          assert gt_bboxes.shape[1] == 4
          if gt_bboxes_ignore is not None:
            assert gt_bboxes_ignore.shape[1] == 4
          gt_bboxes_wh = (gt_bboxes[:, 2:] - gt_bboxes[:, :2]).clamp(min=1e-6)
          gt_bboxes_xy = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2

        elif self.obj_rep == 'XYXYSin2' or self.obj_rep == 'XYXYSin2WZ0Z1':
          obj_dim =  {'XYXYSin2':5, 'XYXYSin2WZ0Z1':8}[self.obj_rep]
          assert gt_bboxes.shape[1] == obj_dim
          if gt_bboxes_ignore is not None:
            assert gt_bboxes_ignore.shape[1] == obj_dim
            gt_bboxes_ignore = gt_bboxes_ignore[:,:8]
          gt_bboxes_raw = gt_bboxes.clone()
          gt_bboxes = gt_bboxes[:,:4]
          gt_bboxes_wh = (gt_bboxes[:, 2:] - gt_bboxes[:, :2]).norm(dim=1)\
                                                              .clamp(min=1e-6)
          gt_bboxes_wh = gt_bboxes_wh.unsqueeze(1).repeat(1,2)      # [56, 2]
          gt_bboxes_xy = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2  # [56, 2]

        elif self.obj_rep == 'XYLgWsAsinSin2Z0Z1' or self.obj_rep == 'XYLgWsAbsSin2Z0Z1':
          assert gt_bboxes.shape[1] == 8
          if gt_bboxes_ignore is not None:
            assert gt_bboxes_ignore.shape[1] == 8
            gt_bboxes_ignore = gt_bboxes_ignore[:,:8]
          gt_bboxes_raw = gt_bboxes.clone()
          gt_bboxes_wh = gt_bboxes[:,2:3].repeat(1,2)
          gt_bboxes_xy = gt_bboxes[:,:2]

        elif self.obj_rep == 'corner':
          raise NotImplemented
          assert gt_bboxes.shape[1] == 2
          assert lvl_min == lvl_max, "only use one level for corner heat map"
          gt_bboxes_wh = gt_bboxes * 0 + 2**(lvl_min)*self.scale
          gt_bboxes = gt_bboxes.repeat(1,2)
          pass

        else:
          import pdb; pdb.set_trace()  # XXX BREAKPOINT
          raise NotImplemented

        if DEBUG_CFG.CHECK_POINT_ASSIGN:
          if not gt_bboxes_wh.min() > DEBUG_CFG.MIN_BOX_SIZE:
            import pdb; pdb.set_trace()  # XXX BREAKPOINT
            pass

        scale = self.scale
        gt_bboxes_lvl = ((torch.log2(gt_bboxes_wh[:, 0] / scale) +
                          torch.log2(gt_bboxes_wh[:, 1] / scale)) / 2).int()
        gt_bboxes_lvl = torch.clamp(gt_bboxes_lvl, min=lvl_min, max=lvl_max)

        # stores the assigned gt index of each point
        assigned_gt_inds = points.new_zeros((num_points, ), dtype=torch.long)
        # stores the assigned gt dist (to this point) of each point
        assigned_gt_dist = points.new_full((num_points, ), float('inf'))
        points_range = torch.arange(points.shape[0])

        for idx in range(num_gts):
            gt_lvl = gt_bboxes_lvl[idx]
            # get the index of points in this level
            lvl_idx = gt_lvl == points_lvl
            points_index = points_range[lvl_idx]
            # get the points in this level
            lvl_points = points_xy[lvl_idx, :]
            # get the center point of gt
            gt_point = gt_bboxes_xy[[idx], :]
            # get width and height of gt
            gt_wh = gt_bboxes_wh[[idx], :]
            # compute the distance between gt center and
            #   all points in this level
            points_gt_dist = ((lvl_points - gt_point) / gt_wh).norm(dim=1)
            # find the nearest k points to gt center in this level
            min_dist, min_dist_index = torch.topk(
                points_gt_dist, self.pos_num, largest=False)

            #print(f'min_dist: {min_dist}')
            if DEBUG_CFG.CHECK_POINT_ASSIGN:
              if min_dist.max() > 2:
                print(f'\n\tmin_dist is too large: {min_dist}\n')
                import pdb; pdb.set_trace()  # XXX BREAKPOINT
                #assert False
                pass
            # the index of nearest k points to gt center in this level
            min_dist_points_index = points_index[min_dist_index]
            # The less_than_recorded_index stores the index
            #   of min_dist that is less then the assigned_gt_dist. Where
            #   assigned_gt_dist stores the dist from previous assigned gt
            #   (if exist) to each point.
            less_than_recorded_index = min_dist < assigned_gt_dist[
                min_dist_points_index]
            # The min_dist_points_index stores the index of points satisfy:
            #   (1) it is k nearest to current gt center in this level.
            #   (2) it is closer to current gt center than other gt center.
            min_dist_points_index = min_dist_points_index[
                less_than_recorded_index]
            # assign the result
            assigned_gt_inds[min_dist_points_index] = idx + 1
            assigned_gt_dist[min_dist_points_index] = min_dist[
                less_than_recorded_index]

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_zeros((num_points, ))
            pos_inds = torch.nonzero(assigned_gt_inds > 0).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]

            if DEBUG_CFG.PRINT_POINT_ASSIGNER:
              pos_dist = assigned_gt_dist[pos_inds]
            #  print(f'pos_dist: {pos_dist}')
        else:
            assigned_labels = None

        assign_res = AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels,
            env='PointAssigner', img_meta=img_meta)

        if DEBUG_CFG.PRINT_POINT_ASSIGNER:
          assign_res.dg_add_pos_dist(pos_dist)
          print('\tPointAssigner\t' + str(assign_res))

        if DEBUG_CFG.VISUALIZE_POINT_ASSIGNER:
          #if (not assign_res.num_pos_inds == num_gts):
            from tools.visual_utils import _show_objs_ls_points_ls
            import numpy as np
            filename = img_meta['filename']
            points_scope = points.max(0)[0][:2].int()
            print(f'{filename}')

            pos_inds = torch.nonzero(assigned_gt_inds).squeeze()
            pos_gt_inds = assigned_gt_inds[pos_inds].cpu().data.numpy() - 1
            pos_gt_inds = np.array(pos_gt_inds).reshape([-1])
            neg_gt_inds = [i for i in range(num_gts) if i not in pos_gt_inds]
            miss_gt_num = len(neg_gt_inds)
            print(f'\n\nMiss {miss_gt_num} in point assigner\n')
            gt_bboxes_raw = gt_bboxes_raw.cpu().data.numpy()
            missed_gt_bboxes = gt_bboxes_raw[neg_gt_inds]
            missed_gt_bboxes_center = missed_gt_bboxes[:,:4].reshape(-1,2,2).mean(1)

            if miss_gt_num>0:
              point_scope = points.max(0)[0][:2].cpu().data.numpy()
              is_gt_out_scope = (missed_gt_bboxes_center >  point_scope).any(-1)
              if is_gt_out_scope.all():
                print('\n\tthe missed gts are out of scope\n')
              else:
                assert False, "miss gt"
            if 1:
              pos_points = points[:,:2][pos_inds].cpu().data.numpy().reshape(-1,2)
              _show_objs_ls_points_ls((points_scope[1], points_scope[0]), [gt_bboxes_raw, missed_gt_bboxes], obj_rep=self.obj_rep, points_ls=[pos_points], obj_colors=['red', 'green'])
              for i in range(len(pos_inds)):
                pos_gt_i = gt_bboxes_raw[pos_gt_inds[i]].reshape(-1, gt_bboxes_raw.shape[1])
                _show_objs_ls_points_ls((points_scope[1], points_scope[0]), [gt_bboxes_raw, pos_gt_i], obj_rep=self.obj_rep, points_ls = [pos_points[i:i+1]], obj_colors=['red', 'green'])
              import pdb; pdb.set_trace()  # XXX BREAKPOINT
              pass

        return assign_res
