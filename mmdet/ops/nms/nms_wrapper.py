import numpy as np
import torch

from . import nms_cpu, nms_cuda, nms_dsiou_cuda
from .soft_nms_cpu import soft_nms_cpu
from obj_geo_utils.obj_utils import OBJ_REPS_PARSE
from tools.visual_utils import _show_objs_ls_points_ls_torch
from configs.common import DEBUG_CFG

def nms_rotated(dets, obj_rep, iou_thr, min_width_length_ratio=0.3, device_id=None):
    from detectron2 import _C
    if isinstance(dets, torch.Tensor):
        is_numpy = False
        dets_th = dets
    elif isinstance(dets, np.ndarray):
        is_numpy = True
        device = 'cpu' if device_id is None else 'cuda:{}'.format(device_id)
        dets_th = torch.from_numpy(dets).to(device)
    else:
        raise TypeError(
            'dets must be either a Tensor or numpy array, but got {}'.format(
                type(dets)))


    obj_dim = OBJ_REPS_PARSE._obj_dims[obj_rep]
    assert dets.shape[1] == obj_dim + 1
    # execute cpu or cuda nms
    if dets_th.shape[0] == 0:
        inds = dets_th.new_zeros(0, dtype=torch.long)
    else:
        # patrse obj_rep to XYZHWA
        scores = dets_th[:, -1].clone()
        dets_th = OBJ_REPS_PARSE.encode_obj(dets_th[:,:obj_dim], obj_rep, 'XYLgWsA')
        dets_th[:,3] = torch.max( dets_th[:,3], dets_th[:,2]*min_width_length_ratio)
        #_show_objs_ls_points_ls_torch( (512,512), [dets_th[:,:obj_dim]], 'XYLgWsA' )
        dets_th[:, 4] *= -180/np.pi

        inds = _C.nms_rotated(dets_th, scores, iou_thr)

    if is_numpy:
        inds = inds.cpu().numpy()

    if DEBUG_CFG.SHOW_NMS_PROCESS:
      n0 = scores.shape[0]
      n1 = inds.shape[0]
      print(f'\n\n{n0} -> {n1}')
      _show_objs_ls_points_ls_torch( (512,512), [dets[:,:obj_dim]], obj_rep )
      _show_objs_ls_points_ls_torch( (512,512), [dets[inds][:,:obj_dim]], obj_rep )
      import pdb; pdb.set_trace()  # XXX BREAKPOINT

    return dets[inds, :], inds


def nms_dsiou(dets, iou_thr, dis_weight, device_id=None):
    """Dispatch to either CPU or GPU NMS implementations.

    The input can be either a torch tensor or numpy array. GPU NMS will be used
    if the input is a gpu tensor or device_id is specified, otherwise CPU NMS
    will be used. The returned type will always be the same as inputs.

    Arguments:
        dets (torch.Tensor or np.ndarray): bboxes with scores.
        iou_thr (float): IoU threshold for NMS.
        device_id (int, optional): when `dets` is a numpy array, if `device_id`
            is None, then cpu nms is used, otherwise gpu_nms will be used.

    Returns:
        tuple: kept bboxes and indice, which is always the same data type as
            the input.

    Example:
        >>> dets = np.array([[49.1, 32.4, 51.0, 35.9, 0.9],
        >>>                  [49.3, 32.9, 51.0, 35.3, 0.9],
        >>>                  [49.2, 31.8, 51.0, 35.4, 0.5],
        >>>                  [35.1, 11.5, 39.1, 15.7, 0.5],
        >>>                  [35.6, 11.8, 39.3, 14.2, 0.5],
        >>>                  [35.3, 11.5, 39.9, 14.5, 0.4],
        >>>                  [35.2, 11.7, 39.7, 15.7, 0.3]], dtype=np.float32)
        >>> iou_thr = 0.7
        >>> supressed, inds = nms(dets, iou_thr)
        >>> assert len(inds) == len(supressed) == 3
    """
    # convert dets (tensor or numpy array) to tensor
    if isinstance(dets, torch.Tensor):
        is_numpy = False
        dets_th = dets
    elif isinstance(dets, np.ndarray):
        is_numpy = True
        device = 'cpu' if device_id is None else 'cuda:{}'.format(device_id)
        dets_th = torch.from_numpy(dets).to(device)
    else:
        raise TypeError(
            'dets must be either a Tensor or numpy array, but got {}'.format(
                type(dets)))

    # execute cpu or cuda nms
    dets_th = dets_th[:,[0,1,2,3,-1]]
    if dets_th.shape[0] == 0:
        inds = dets_th.new_zeros(0, dtype=torch.long)
    else:
        if dets_th.is_cuda:
            inds = nms_dsiou_cuda.nms(dets_th, iou_thr, dis_weight)
        else:
            raise  NotImplementedError
            inds = nms_cpu.nms(dets_th, iou_thr)

    if is_numpy:
        inds = inds.cpu().numpy()
    return dets[inds, :], inds


def nms(dets, iou_thr, device_id=None):
    """Dispatch to either CPU or GPU NMS implementations.

    The input can be either a torch tensor or numpy array. GPU NMS will be used
    if the input is a gpu tensor or device_id is specified, otherwise CPU NMS
    will be used. The returned type will always be the same as inputs.

    Arguments:
        dets (torch.Tensor or np.ndarray): bboxes with scores.
        iou_thr (float): IoU threshold for NMS.
        device_id (int, optional): when `dets` is a numpy array, if `device_id`
            is None, then cpu nms is used, otherwise gpu_nms will be used.

    Returns:
        tuple: kept bboxes and indice, which is always the same data type as
            the input.

    Example:
        >>> dets = np.array([[49.1, 32.4, 51.0, 35.9, 0.9],
        >>>                  [49.3, 32.9, 51.0, 35.3, 0.9],
        >>>                  [49.2, 31.8, 51.0, 35.4, 0.5],
        >>>                  [35.1, 11.5, 39.1, 15.7, 0.5],
        >>>                  [35.6, 11.8, 39.3, 14.2, 0.5],
        >>>                  [35.3, 11.5, 39.9, 14.5, 0.4],
        >>>                  [35.2, 11.7, 39.7, 15.7, 0.3]], dtype=np.float32)
        >>> iou_thr = 0.7
        >>> supressed, inds = nms(dets, iou_thr)
        >>> assert len(inds) == len(supressed) == 3
    """
    # convert dets (tensor or numpy array) to tensor
    if isinstance(dets, torch.Tensor):
        is_numpy = False
        dets_th = dets
    elif isinstance(dets, np.ndarray):
        is_numpy = True
        device = 'cpu' if device_id is None else 'cuda:{}'.format(device_id)
        dets_th = torch.from_numpy(dets).to(device)
    else:
        raise TypeError(
            'dets must be either a Tensor or numpy array, but got {}'.format(
                type(dets)))

    # execute cpu or cuda nms
    dets_th = dets_th[:,[0,1,2,3,-1]]
    if dets_th.shape[0] == 0:
        inds = dets_th.new_zeros(0, dtype=torch.long)
    else:
        if dets_th.is_cuda:
            inds = nms_cuda.nms(dets_th, iou_thr)
        else:
            inds = nms_cpu.nms(dets_th, iou_thr)

    if is_numpy:
        inds = inds.cpu().numpy()
    return dets[inds, :], inds


def soft_nms(dets, iou_thr, method='linear', sigma=0.5, min_score=1e-3):
    """
    Example:
        >>> dets = np.array([[4., 3., 5., 3., 0.9],
        >>>                  [4., 3., 5., 4., 0.9],
        >>>                  [3., 1., 3., 1., 0.5],
        >>>                  [3., 1., 3., 1., 0.5],
        >>>                  [3., 1., 3., 1., 0.4],
        >>>                  [3., 1., 3., 1., 0.0]], dtype=np.float32)
        >>> iou_thr = 0.7
        >>> supressed, inds = soft_nms(dets, iou_thr, sigma=0.5)
        >>> assert len(inds) == len(supressed) == 3
    """
    if isinstance(dets, torch.Tensor):
        is_tensor = True
        dets_np = dets.detach().cpu().numpy()
    elif isinstance(dets, np.ndarray):
        is_tensor = False
        dets_np = dets
    else:
        raise TypeError(
            'dets must be either a Tensor or numpy array, but got {}'.format(
                type(dets)))

    method_codes = {'linear': 1, 'gaussian': 2}
    if method not in method_codes:
        raise ValueError('Invalid method for SoftNMS: {}'.format(method))
    new_dets, inds = soft_nms_cpu(
        dets_np,
        iou_thr,
        method=method_codes[method],
        sigma=sigma,
        min_score=min_score)

    if is_tensor:
        return dets.new_tensor(new_dets), dets.new_tensor(
            inds, dtype=torch.long)
    else:
        return new_dets.astype(np.float32), inds.astype(np.int64)


def nms_rotated_np(dets, obj_rep, iou_thr, min_width_length_ratio):
  dets = torch.from_numpy(dets)
  dets_new, ids = nms_rotated(dets, obj_rep, iou_thr, min_width_length_ratio)
  dets_new = dets_new.numpy()
  return dets_new, ids


