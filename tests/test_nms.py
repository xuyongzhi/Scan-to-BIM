"""
CommandLine:
    pytest tests/test_nms.py
"""
import numpy as np
import torch

from mmdet.ops.nms.nms_wrapper import nms, nms_rotated_np


def test_nms_device_and_dtypes_cpu():
    """
    CommandLine:
        xdoctest -m tests/test_nms.py test_nms_device_and_dtypes_cpu
    """
    iou_thr = 0.7
    base_dets = np.array([[49.1, 32.4, 51.0, 35.9, 0.9],
                          [49.3, 32.9, 51.0, 35.3, 0.9],
                          [35.3, 11.5, 39.9, 14.5, 0.4],
                          [35.2, 11.7, 39.7, 15.7, 0.3]])

    # CPU can handle float32 and float64
    dets = base_dets.astype(np.float32)
    supressed, inds = nms(dets, iou_thr)
    assert dets.dtype == supressed.dtype
    assert len(inds) == len(supressed) == 3

    dets = torch.FloatTensor(base_dets)
    surpressed, inds = nms(dets, iou_thr)
    assert dets.dtype == surpressed.dtype
    assert len(inds) == len(surpressed) == 3

    dets = base_dets.astype(np.float64)
    supressed, inds = nms(dets, iou_thr)
    assert dets.dtype == supressed.dtype
    assert len(inds) == len(supressed) == 3

    dets = torch.DoubleTensor(base_dets)
    surpressed, inds = nms(dets, iou_thr)
    assert dets.dtype == surpressed.dtype
    assert len(inds) == len(surpressed) == 3


def test_nms_device_and_dtypes_gpu():
    """
    CommandLine:
        xdoctest -m tests/test_nms.py test_nms_device_and_dtypes_gpu
    """
    if not torch.cuda.is_available():
        import pytest
        pytest.skip('test requires GPU and torch+cuda')

    iou_thr = 0.7
    base_dets = np.array([[49.1, 32.4, 51.0, 35.9, 0.9],
                          [49.3, 32.9, 51.0, 35.3, 0.9],
                          [35.3, 11.5, 39.9, 14.5, 0.4],
                          [35.2, 11.7, 39.7, 15.7, 0.3]])

    for device_id in range(torch.cuda.device_count()):
        print('Run NMS on device_id = {!r}'.format(device_id))
        # GPU can handle float32 but not float64
        dets = base_dets.astype(np.float32)
        supressed, inds = nms(dets, iou_thr, device_id)
        assert dets.dtype == supressed.dtype
        assert len(inds) == len(supressed) == 3

        dets = torch.FloatTensor(base_dets).to(device_id)
        surpressed, inds = nms(dets, iou_thr)
        assert dets.dtype == surpressed.dtype
        assert len(inds) == len(surpressed) == 3

def test_nms_rotated():
  dets = np.array([
    [190.665, 237.454,   0.   , 103.328,   0.   ,   0.   ,   0.734,   0.742],
    [190.665, 237.454,   0.   , 103.328,   0.   ,   0.   ,   0.734,   0.706]], dtype=np.float32)
  dets = np.array([
    [100, 100, 0, 100, 100, 0, 0, 1],
    [100, 100, 0, 100, 100, 0, 0, 1],
  ], dtype=np.float32)
  obj_rep = 'XYZLgWsHA'
  dets_new, ids = nms_rotated_np( dets, obj_rep, 0.2, 0.3 )
  pass

if __name__ == '__main__':
  test_nms_rotated()
