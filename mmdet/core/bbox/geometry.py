import torch


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
  bboxes1_a = dilated_bboxes(bboxes1)
  bboxes2_a = dilated_bboxes(bboxes2)
  return bbox_overlaps(bboxes1_a, bboxes2_a, mode, is_aligned)

def dilated_bboxes(bboxes0, size_rate_thres=0.25):
  assert bboxes0.shape[1] == 4
  bboxes1 = bboxes0.clone()
  box_size = bboxes1[:,[2,3]] - bboxes1[:,[0,1]]
  thres_size = box_size.max(dim=1, keepdim=True)[0] * size_rate_thres
  aug = (thres_size - box_size).clamp(min=0) * 0.5
  bboxes1[:,:2] = (bboxes1[:, :2] - aug)
  bboxes1[:,2:] = (bboxes1[:, 2:] + aug)
  return bboxes1

def relative_dis(bboxes1, bboxes2, mode='gt_size_as_ref'):
  '''
  bboxes1 is gt
  '''
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
  rel_diss = relative_dis(bboxes1, bboxes2)
  #t2 = time.time()
  ious = aug_ious * iou_w + rel_diss * (1-iou_w)
  #print('t1:{}\nt2:{}'.format((t1-t0)*1000, (t2-t1)*1000))
  return ious
