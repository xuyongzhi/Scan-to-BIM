import torch
import numpy as np
from tools.visual_utils import _show_objs_ls_points_ls


def dsiou_rotated_3d_bbox(bboxes1, bboxes2, iou_w = 0.8):
  '''
  XYZLgWsHA
  bboxes1: [n,7]  gt
  '''
  assert bboxes1.shape[1] == 7
  assert bboxes2.shape[1] == 7
  #import time
  #t0 = time.time()
  aug_ious = rotated_3d_bbox_overlaps(bboxes1, bboxes2)
  #t1 = time.time()
  rel_diss = relative_dis_XYZLgWsHA(bboxes1, bboxes2)
  #t2 = time.time()
  ious = aug_ious * iou_w + rel_diss * (1-iou_w)
  #print('t1:{}\nt2:{}'.format((t1-t0)*1000, (t2-t1)*1000))
  return ious

def relative_dis_XYZLgWsHA(bboxes1, bboxes2, mode='gt_size_as_ref'):
  '''
  bboxes1 is gt
  '''
  assert bboxes1.shape[1] == 7
  assert bboxes2.shape[1] == 7

  n1 = bboxes1.shape[0]
  n2 = bboxes2.shape[0]
  centroids1 = bboxes1[:,:2]
  centroids2 = bboxes2[:,:2]
  abs_diss = centroids1.unsqueeze(dim=1) - centroids2.unsqueeze(dim=0)
  abs_diss = abs_diss.norm(dim=-1)

  sizes1 = bboxes1[:,3:5].max(dim=1)[0]
  if mode == 'gt_size_as_ref':
    ref_sizes = sizes1.unsqueeze(dim=1).repeat(1, n2) / 2
  elif mode == 'fix_size_as_ref':
    ref_sizes = sizes1.mean()
  elif mode == 'ave_size_as_ref':
    sizes2 = (bboxes2[:,2:] - bboxes2[:,:2]).norm(dim=-1)
    ave_sizes = (sizes1.unsqueeze(dim=1) + sizes2.unsqueeze(dim=0))/2
    ref_sizes = ave_sizes
  else:
    raise NotImplemented

  rel_diss = 1 - abs_diss / ref_sizes
  rel_diss = rel_diss.clamp(min=0)
  rel_diss = rel_diss ** 2
  return rel_diss


def dilate_3d_bboxes(bboxes0, size_rate_thres=0.25):
  '''
  XYZLgWsHA
  '''
  assert bboxes0.shape[1] == 7
  bboxes1 = bboxes0.clone()
  min_width = bboxes1[:, 3] * size_rate_thres
  bboxes1[:,4] = torch.max(bboxes1[:,4], min_width)
  #_show_objs_ls_points_ls((512,512), [bboxes0.cpu().numpy()], obj_rep='XYZLgWsHA')
  #_show_objs_ls_points_ls((512,512), [bboxes1.cpu().numpy()], obj_rep='XYZLgWsHA')
  return bboxes1

def rotated_3d_bbox_overlaps(bboxes1, bboxes2):
  '''
  XYZLgWsHA
  '''
  assert bboxes1.shape[1] == 7
  assert bboxes1.shape[1] == 7
  bboxes1 = dilate_3d_bboxes(bboxes1)
  bboxes2 = dilate_3d_bboxes(bboxes2)
  ious_2d = rotated_bbox_overlaps( bboxes1[:,[0,1,3,4,6]], bboxes2[:,[0,1,3,4,6]])

  #_show_objs_ls_points_ls((512,512), [bboxes2[:100,:].cpu().data.numpy()], obj_rep='XYZLgWsHA')
  #_show_objs_ls_points_ls((512,512), [bboxes1.cpu().numpy()], obj_rep='XYZLgWsHA')
  return ious_2d

def rotated_bbox_overlaps(bboxes1, bboxes2):
  '''
  XYLgWsA
  bbox: [cx, cy, size_x, size_y, angle]
  angle: from x_r to x_b, positive for clock-wise
        unit: degree
  '''
  from detectron2 import _C
  bboxes1 = bboxes1.clone()
  bboxes2 = bboxes2.clone()
  #_show_objs_ls_points_ls((512,512), [bboxes1.cpu().numpy()], obj_rep='XYLgWsA')
  bboxes1[:,-1] *= 180/np.pi
  bboxes2[:,-1] *= 180/np.pi
  assert bboxes1.shape[1] == 5
  assert bboxes1.shape[1] == 5
  ious_2d = _C.box_iou_rotated(bboxes1, bboxes2)
  return ious_2d

def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (m, 4) in <x1, y1, x2, y2> format.
        bboxes2 (Tensor): shape (n, 4) in <x1, y1, x2, y2> format.
            If is_aligned is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> bbox_overlaps(bboxes1, bboxes2)
        tensor([[0.5238, 0.0500, 0.0041],
                [0.0323, 0.0452, 1.0000],
                [0.0000, 0.0000, 0.0000]])

    Example:
        >>> empty = torch.FloatTensor([])
        >>> nonempty = torch.FloatTensor([
        >>>     [0, 0, 10, 9],
        >>> ])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'iof']

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if is_aligned:
        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1 + area2 - overlap)
        else:
            ious = overlap / area1
    else:
        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1[:, None] + area2 - overlap)
        else:
            ious = overlap / (area1[:, None])

    return ious

def dilated_bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
  bboxes1_a = dilate_bboxes(bboxes1)
  bboxes2_a = dilate_bboxes(bboxes2)
  return bbox_overlaps(bboxes1_a, bboxes2_a, mode, is_aligned)

def dilate_bboxes(bboxes0, size_rate_thres=0.25):
  '''
  xyxy
  '''
  assert bboxes0.shape[1] == 4
  bboxes1 = bboxes0.clone()
  box_size = bboxes1[:,[2,3]] - bboxes1[:,[0,1]]
  thres_size = box_size.max(dim=1, keepdim=True)[0] * size_rate_thres
  aug = (thres_size - box_size).clamp(min=0) * 0.5
  bboxes1[:,:2] = (bboxes1[:, :2] - aug)
  bboxes1[:,2:] = (bboxes1[:, 2:] + aug)
  return bboxes1

def relative_dis_XyXy(bboxes1, bboxes2, mode='gt_size_as_ref'):
  '''
  bboxes1 is gt
  '''
  assert bboxes1.shape[1] == 4
  assert bboxes2.shape[1] == 4
  n1 = bboxes1.shape[0]
  n2 = bboxes2.shape[0]
  centroids1 = (bboxes1[:,:2] + bboxes1[:,2:]) / 2
  centroids2 = (bboxes2[:,:2] + bboxes2[:,2:]) / 2
  abs_diss = centroids1.unsqueeze(dim=1) - centroids2.unsqueeze(dim=0)
  abs_diss = abs_diss.norm(dim=-1)

  sizes1 = (bboxes1[:,2:] - bboxes1[:,:2]).norm(dim=-1)
  if mode == 'gt_size_as_ref':
    ref_sizes = sizes1.unsqueeze(dim=1).repeat(1, n2) / 2
  elif mode == 'fix_size_as_ref':
    ref_sizes = sizes1.mean()
  elif mode == 'ave_size_as_ref':
    sizes2 = (bboxes2[:,2:] - bboxes2[:,:2]).norm(dim=-1)
    ave_sizes = (sizes1.unsqueeze(dim=1) + sizes2.unsqueeze(dim=0))/2
    ref_sizes = ave_sizes
  else:
    raise NotImplemented

  rel_diss = 1 - abs_diss / ref_sizes
  rel_diss = rel_diss.clamp(min=0)
  rel_diss = rel_diss ** 2
  return rel_diss

def dsiou_bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
  #import time
  #t0 = time.time()
  iou_w = 0.6
  aug_ious = dilated_bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)
  #t1 = time.time()
  rel_diss = relative_dis_XyXy(bboxes1, bboxes2)
  #t2 = time.time()
  ious = aug_ious * iou_w + rel_diss * (1-iou_w)
  #print('t1:{}\nt2:{}'.format((t1-t0)*1000, (t2-t1)*1000))
  return ious

def corner_overlaps(corners1, corners2, ref_radius, norm_method='gaussian'):
  assert corners1.dim() == corners2.dim() == 2
  assert corners1.shape[1] == 2
  assert corners2.shape[1] == 2  # corners2[:,2] is stride
  abs_diss = corners1.unsqueeze(dim=1) - corners2.unsqueeze(dim=0)
  abs_diss = abs_diss.norm(dim=-1)
  if norm_method == 'divide':
    rel_diss = abs_diss / pow(ref_radius,2)
    rel_diss = 1 - rel_diss.clamp(max=1)
  elif norm_method == 'gaussian':
    ref = pow(ref_radius,2) * 2
    ref_diss = torch.exp( -abs_diss/ref )
  #show_overlaps(ref_diss.cpu().data.numpy())
  return ref_diss


def show_overlaps(overlaps):
  import numpy as np
  import mmcv
  overlaps = overlaps.max(axis=0)
  s = np.sqrt(overlaps.shape[0]).astype(np.int32)
  overlaps = overlaps.reshape(s,s)
  mmcv.imshow(overlaps)

def test():
  from tools.visual_utils import _show_objs_ls_points_ls_torch
  cx = 200
  cy = 200
  cz = 0
  sx = 100
  sy = 1
  sz = 0
  u = np.pi/180
  bboxes0 = torch.Tensor([
    [cx, cy, cz, sx, sy, sz, 0]
  ])
  bboxes1 = torch.Tensor([
    [cx, cy, cz, sx, sy, sz, 0*u,],
    [cx, cy, cz, sy, sx, sz, 0*u,],
    [cx, cy, cz, sx, sy, sz, 30*u,],
  ])
  #ious = rotated_bbox_overlaps(bboxes0, bboxes1)
  ious = dsiou_rotated_3d_bbox(bboxes0, bboxes1, iou_w=0.8)
  s = ious[0][0:1]
  s[:] = 1
  print(ious)
  for i in range(bboxes1.shape[0]):
    print(ious[0,i])
    _show_objs_ls_points_ls_torch( (512,512), [bboxes0, bboxes1[i:i+1]], obj_rep='XYZLgWsHA',
                                obj_colors=['green', 'red'] )
  #_show_objs_ls_points_ls_torch( (512,512), [bboxes0, bboxes1], obj_rep='XYLgWsA',
  #                              obj_colors=['red','green'], obj_scores_ls=[s, ious[0]] )

if __name__ == '__main__':
  test()
