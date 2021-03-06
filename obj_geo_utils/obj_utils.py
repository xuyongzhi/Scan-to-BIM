# xyz 17 Apr
import os
import numpy as np
import mmcv
from obj_geo_utils.geometry_utils import sin2theta_np, angle_with_x_np, \
      vec_from_angle_with_x_np, angle_with_x
import cv2
import torch
from obj_geo_utils.geometry_utils import limit_period_np, four_corners_to_box,\
  sort_four_corners, line_intersection_2d, mean_angles, angle_dif_by_period_np,\
  check_duplicate

MIN_ALIGN_ANGLE = 3.5

class OBJ_REPS_PARSE():
  '''
  (*) CenSizeA3D vs XYZLgWsHA
    CenSizeA3D[:,3:6]: [size_x, size_y, size_z]
    XYZLgWsHA[:,3:6]: [max(size_x, size_y), min(size_x, size_y), size_z]
    CenSizeA3D[:,-1]: rotation from x_ref to x_body
    XYZLgWsHA[:,-1]: rotation from x_ref to axis_long

  (*) XYZLgWsHA
    angle: between long axis (x_b) and x_f, [-90, 90)
    x_b denotes the long axis:  always Lg >= Ws
    Lg: length is the greater horizontal axis
    Ws: Width is the smaller horizontal axis

  (*) XYLgWsSin2Sin4Z0Z1
    angle, Lg, Ws: same with XYZLgWsHA

  (*) XYXYSin2
    x denotes the long axis.
    angle: [-90, 90)
  '''
  _obj_dims = {
    'CenSizeA3D': 7,
    'CenSize2D': 5,

    'XYZLgWsHA': 7,
    'XYLgWsA': 5,

    'XYLgWsSin2Cos2Z0Z1': 8,
    'XYDRSin2Cos2Z0Z1': 8,

    'XYLgWsAbsSin2Z0Z1': 8,

    'XYLgWsAsinSin2Z0Z1': 8,
    'XYLgWsAsinSin2': 6,

    'XYLgWsSin2Sin4Z0Z1': 8,
    'XYLgWsSin2Sin4': 6,


    'XYXYSin2': 5,
    'XYXYSin2W': 6,
    'RoLine2D_2p': 4,
    'XYLgA': 4,

    'XYXYSin2WZ0Z1': 8,
    'Top_Corners': 12,
    'Bottom_Corners': 12,

    'XYDAsinAsinSin2Z0Z1': 8,
    'XYDAsinAsinSin2' : 6,

    'Rect4CornersZ0Z1': 10,

  }
  _obj_reps = _obj_dims.keys()

  @staticmethod
  def check_obj_dim(bboxes, obj_rep, allow_illegal):
    assert bboxes.ndim == 2
    s = OBJ_REPS_PARSE._obj_dims[obj_rep]
    assert bboxes.shape[1] == s, f'obj_rep={obj_rep}, input shape={bboxes.shape[1]}, correct shape={s}'

    if obj_rep == 'XYLgWsAbsSin2Z0Z1':
        if allow_illegal:
          bboxes[:,3] = np.clip(bboxes[:,3], None, bboxes[:,2])
          bboxes[:,4] = np.clip(bboxes[:,4], 0, np.pi/2)
          bboxes[:,5] = np.clip(bboxes[:,5], -1, 1)
        else:
          check_lw = np.all( bboxes[:,2] > bboxes[:,3] )
          check_abs = np.all( bboxes[:,4] >=0 ) and  np.all( bboxes[:,4] <= np.pi/2 )
          check_sin2 = np.all( np.abs(bboxes[:,5]) <= 1 )
          if not check_lw:
            err = ( bboxes[:,2] - bboxes[:,3] ).max()
            pass
          if not check_abs:
            import pdb; pdb.set_trace()  # XXX BREAKPOINT
            pass
          if not check_sin2:
            import pdb; pdb.set_trace()  # XXX BREAKPOINT
            pass

  @staticmethod
  def encode_obj(bboxes, obj_rep_in, obj_rep_out, allow_illegal=True):
    if isinstance(bboxes, torch.Tensor):
      bboxes_np = bboxes.cpu().data.numpy()
      bboxes_np = OBJ_REPS_PARSE.encode_obj_np(bboxes_np, obj_rep_in, obj_rep_out, allow_illegal)
      OBJ_REPS_PARSE.check_obj_dim(bboxes_np, obj_rep_out, allow_illegal)
      return torch.from_numpy(bboxes_np).to(bboxes.dtype).to(bboxes.device)
    else:
      bboxes_out = OBJ_REPS_PARSE.encode_obj_np(bboxes, obj_rep_in, obj_rep_out, allow_illegal)
      OBJ_REPS_PARSE.check_obj_dim(bboxes_out, obj_rep_out, allow_illegal)
      return bboxes_out

  @staticmethod
  def encode_obj_np(bboxes, obj_rep_in, obj_rep_out, allow_illegal):
    '''
    bboxes: [n,4] or [n,2,2]
    bboxes_out : [n,4/5]
    '''
    assert obj_rep_in  in OBJ_REPS_PARSE._obj_reps, obj_rep_in
    assert obj_rep_out in OBJ_REPS_PARSE._obj_reps, obj_rep_out
    OBJ_REPS_PARSE.check_obj_dim(bboxes, obj_rep_in, allow_illegal)

    bboxes = OBJ_REPS_PARSE.make_x_long_dim(bboxes, obj_rep_in)
    nb = bboxes.shape[0]

    if obj_rep_in == obj_rep_out:
      return bboxes

    # XYZLgWsHA  ---------------------------------------------------------------
    if obj_rep_in == 'XYZLgWsHA'  and obj_rep_out == 'XYLgWsA':
      return bboxes[:,[0,1,3,4,6]]

    elif obj_rep_in == 'XYZLgWsHA' and obj_rep_out == 'XYLgWsAbsSin2Z0Z1':
        return OBJ_REPS_PARSE.XYZLgWsHA_TO_XYLgWsAbsSin2Z0Z1(bboxes)

    elif obj_rep_in == 'XYZLgWsHA' and obj_rep_out == 'XYDAsinAsinSin2Z0Z1':
      return OBJ_REPS_PARSE.XYZLgWsHA_TO_XYDAsinAsinSin2Z0Z1(bboxes)

    elif obj_rep_in == 'XYZLgWsHA' and obj_rep_out == 'RoLine2D_2p':
      return OBJ_REPS_PARSE.encode_obj(bboxes[:,[0,1,3,4,6]], 'XYLgWsA', 'RoLine2D_2p')

    elif obj_rep_in == 'XYZLgWsHA' and obj_rep_out == 'XYXYSin2':
      RoLine2D_2p = OBJ_REPS_PARSE.encode_obj(bboxes, 'XYZLgWsHA', 'RoLine2D_2p')
      return OBJ_REPS_PARSE.encode_obj(RoLine2D_2p, 'RoLine2D_2p', 'XYXYSin2')

    elif obj_rep_in == 'XYZLgWsHA' and obj_rep_out == 'XYXYSin2W':
      XYXYSin2 = OBJ_REPS_PARSE.encode_obj(bboxes, obj_rep_in, 'XYXYSin2')
      return np.concatenate([ XYXYSin2, bboxes[:,4:5] ], 1)

    # RoLine2D_2p --------------------------------------------------------------
    elif obj_rep_in == 'RoLine2D_2p' and obj_rep_out == 'XYXYSin2':
      return OBJ_REPS_PARSE.Line2p_TO_UpRight_xyxy_sin2a(bboxes)

    elif obj_rep_in == 'RoLine2D_2p' and obj_rep_out == 'XYXYSin2W':
      XYXYSin2 = OBJ_REPS_PARSE.encode_obj(bboxes, obj_rep_in, 'XYXYSin2')
      zeros = XYXYSin2[:,0:1] * 0
      XYXYSin2W = np.concatenate([XYXYSin2, zeros], 1)
      return XYXYSin2W

    elif obj_rep_in == 'RoLine2D_2p' and obj_rep_out == 'XYZLgWsHA':
      XYXYSin2 = OBJ_REPS_PARSE.encode_obj(bboxes, 'RoLine2D_2p', 'XYXYSin2')
      return OBJ_REPS_PARSE.encode_obj(XYXYSin2, 'XYXYSin2', 'XYZLgWsHA')

    # XYXYSin2 XYXYSin2W  ------------------------------------------------------
    elif obj_rep_in == 'XYXYSin2' and obj_rep_out == 'RoLine2D_2p':
      return OBJ_REPS_PARSE.XYXYSin2_TO_RoLine2D_2p(bboxes)

    elif obj_rep_in == 'XYXYSin2W' and obj_rep_out == 'RoLine2D_2p':
      return OBJ_REPS_PARSE.XYXYSin2_TO_RoLine2D_2p(bboxes[:,:5])

    elif obj_rep_in == 'XYXYSin2W' and obj_rep_out == 'XYLgWsA':
        return OBJ_REPS_PARSE.XYXYSin2W_TO_XYLgWsA(bboxes)

    elif obj_rep_in == 'XYXYSin2WZ0Z1' and obj_rep_out == 'XYXYSin2W':
      return bboxes[:,:-2]

    # XYLgWsSin2Cos2Z0Z1  ---------------------------------------------------------------
    elif obj_rep_in == 'XYZLgWsHA' and obj_rep_out == 'XYLgWsSin2Cos2Z0Z1':
        return OBJ_REPS_PARSE.XYZLgWsHA_TO_XYLgWsSin2Cos2Z0Z1(bboxes)

    elif obj_rep_in == 'XYLgWsSin2Cos2Z0Z1' and obj_rep_out == 'XYZLgWsHA':
        return OBJ_REPS_PARSE.XYLgWsSin2Cos2Z0Z1_TO_XYZLgWsHA(bboxes)

    elif obj_rep_in == 'XYXYSin2WZ0Z1' and obj_rep_out == 'XYLgWsSin2Cos2Z0Z1':
        bboxes = OBJ_REPS_PARSE.encode_obj(bboxes, obj_rep_in, 'XYZLgWsHA')
        return OBJ_REPS_PARSE.XYZLgWsHA_TO_XYLgWsSin2Cos2Z0Z1(bboxes)

    elif obj_rep_in == 'XYLgWsSin2Cos2Z0Z1' and obj_rep_out == 'XYLgWsA':
      bboxes = OBJ_REPS_PARSE.encode_obj(bboxes, obj_rep_in, 'XYZLgWsHA')
      return OBJ_REPS_PARSE.encode_obj(bboxes, 'XYZLgWsHA', 'XYLgWsA')

    # XYDRSin2Cos2Z0Z1  ---------------------------------------------------------------
    elif obj_rep_in == 'XYZLgWsHA' and obj_rep_out == 'XYDRSin2Cos2Z0Z1':
        return OBJ_REPS_PARSE.XYZLgWsHA_TO_XYDRSin2Cos2Z0Z1(bboxes)

    elif obj_rep_in == 'XYDRSin2Cos2Z0Z1' and obj_rep_out == 'XYZLgWsHA':
        return OBJ_REPS_PARSE.XYDRSin2Cos2Z0Z1_TO_XYZLgWsHA(bboxes)

    elif obj_rep_in == 'XYXYSin2WZ0Z1' and obj_rep_out == 'XYDRSin2Cos2Z0Z1':
        bboxes = OBJ_REPS_PARSE.encode_obj(bboxes, obj_rep_in, 'XYZLgWsHA')
        return OBJ_REPS_PARSE.XYZLgWsHA_TO_XYDRSin2Cos2Z0Z1(bboxes)

    elif obj_rep_in == 'XYDRSin2Cos2Z0Z1' and obj_rep_out == 'XYLgWsA':
      bboxes = OBJ_REPS_PARSE.encode_obj(bboxes, obj_rep_in, 'XYZLgWsHA')
      return OBJ_REPS_PARSE.encode_obj(bboxes, 'XYZLgWsHA', 'XYLgWsA')
    # essential  ---------------------------------------------------------------




    elif obj_rep_in == 'XYLgWsAbsSin2Z0Z1' and obj_rep_out == 'XYZLgWsHA':
        return OBJ_REPS_PARSE.XYLgWsAbsSin2Z0Z1_TO_XYZLgWsHA(bboxes)

    elif obj_rep_in == 'XYDAsinAsinSin2Z0Z1'  and obj_rep_out == 'XYZLgWsHA':
      return OBJ_REPS_PARSE.XYDAsinAsinSin2Z0Z1_TO_XYZLgWsHA(bboxes)

    elif obj_rep_in == 'XYDAsinAsinSin2'  and obj_rep_out == 'XYLgWsA':
      b = np.concatenate([bboxes, bboxes[:,:2]*2], 1)
      return OBJ_REPS_PARSE.XYDAsinAsinSin2Z0Z1_TO_XYZLgWsHA(b)[:,[0,1,3,4,6]]


    elif obj_rep_in == 'XYLgWsAsinSin2Z0Z1' and obj_rep_out == 'XYDAsinAsinSin2Z0Z1':
      return OBJ_REPS_PARSE.XYLgWsAsinSin2Z0Z1_TO_XYDAsinAsinSin2Z0Z1(bboxes)

    elif obj_rep_in == 'XYDAsinAsinSin2Z0Z1'  and obj_rep_out == 'XYLgWsAsinSin2Z0Z1':
      return OBJ_REPS_PARSE.XYDAsinAsinSin2Z0Z1_TO_XYLgWsAsinSin2Z0Z1(bboxes)

    elif obj_rep_in == 'XYDAsinAsinSin2Z0Z1'  and obj_rep_out == 'XYXYSin2WZ0Z1':
      return OBJ_REPS_PARSE.XYDAsinAsinSin2Z0Z1_TO_XYXYSin2WZ0Z1(bboxes)

    elif obj_rep_in == 'XYDAsinAsinSin2Z0Z1'  and obj_rep_out == 'XYXYSin2W':
      return OBJ_REPS_PARSE.XYDAsinAsinSin2Z0Z1_TO_XYXYSin2WZ0Z1(bboxes)[:,:6]

    # Rect4CornersZ0Z1 -------------------------------------------------------------------
    elif obj_rep_in == 'XYDAsinAsinSin2Z0Z1' and obj_rep_out == 'Rect4CornersZ0Z1':
      return OBJ_REPS_PARSE.XYDAsinAsinSin2Z0Z1_TO_Rect4CornersZ0Z1(bboxes)

    elif obj_rep_in == 'Rect4CornersZ0Z1' and obj_rep_out == 'XYDAsinAsinSin2Z0Z1':
      bboxes_t = torch.from_numpy(bboxes[:,:8].reshape(-1,4,2))
      XYDAsinAsinSin2, rect_loss = four_corners_to_box(bboxes_t)
      XYDAsinAsinSin2 = XYDAsinAsinSin2.numpy()
      XYDAsinAsinSin2Z0Z1 = np.concatenate([XYDAsinAsinSin2, bboxes[:,8:10]], -1)
      return XYDAsinAsinSin2Z0Z1

    elif obj_rep_in == 'Rect4CornersZ0Z1' and obj_rep_out == 'XYZLgWsHA':
      XYDAsinAsinSin2Z0Z1 = OBJ_REPS_PARSE.encode_obj(bboxes, 'Rect4CornersZ0Z1', 'XYDAsinAsinSin2Z0Z1')
      return OBJ_REPS_PARSE.encode_obj(XYDAsinAsinSin2Z0Z1, 'XYDAsinAsinSin2Z0Z1', 'XYZLgWsHA')

    elif obj_rep_in == 'Rect4CornersZ0Z1' and obj_rep_out == 'XYLgWsA':
      XYZLgWsHA = OBJ_REPS_PARSE.encode_obj(bboxes, 'Rect4CornersZ0Z1', 'XYZLgWsHA')
      return XYZLgWsHA[:,[0,1,3,4,6]]

    elif obj_rep_in == 'XYZLgWsHA' and obj_rep_out == 'Rect4CornersZ0Z1':
      XYDAsinAsinSin2Z0Z1 = OBJ_REPS_PARSE.encode_obj(bboxes, 'XYZLgWsHA', 'XYDAsinAsinSin2Z0Z1')
      return OBJ_REPS_PARSE.encode_obj(XYDAsinAsinSin2Z0Z1, 'XYDAsinAsinSin2Z0Z1', 'Rect4CornersZ0Z1'  )

    elif obj_rep_in == 'XYXYSin2WZ0Z1' and obj_rep_out == 'Rect4CornersZ0Z1':
      XYZLgWsHA = OBJ_REPS_PARSE.encode_obj(bboxes, 'XYXYSin2WZ0Z1', 'XYZLgWsHA')
      return OBJ_REPS_PARSE.encode_obj(XYZLgWsHA, 'XYZLgWsHA', 'Rect4CornersZ0Z1')

    elif obj_rep_in == 'Rect4CornersZ0Z1' and obj_rep_out == 'RoLine2D_2p':
      XYZLgWsHA = OBJ_REPS_PARSE.encode_obj(bboxes, 'Rect4CornersZ0Z1', 'XYZLgWsHA')
      return OBJ_REPS_PARSE.encode_obj(XYZLgWsHA, 'XYZLgWsHA', 'RoLine2D_2p')

    elif obj_rep_in == 'RoLine2D_2p' and obj_rep_out == 'Rect4CornersZ0Z1':
      XYLgA = OBJ_REPS_PARSE.encode_obj(bboxes, 'RoLine2D_2p', 'XYLgA')
      ze = XYLgA[:,0:1] * 0
      XYZLgWsHA = np.concatenate([XYLgA[:,:2], ze, XYLgA[:,2:3], ze, ze, XYLgA[:,3:4]], axis=1)
      return OBJ_REPS_PARSE.encode_obj(XYZLgWsHA, 'XYZLgWsHA', 'Rect4CornersZ0Z1')

    elif obj_rep_in == 'XYXYSin2' and obj_rep_out == 'Rect4CornersZ0Z1':
      RoLine2D_2p = OBJ_REPS_PARSE.encode_obj(bboxes, 'XYXYSin2', 'RoLine2D_2p')
      return OBJ_REPS_PARSE.encode_obj(RoLine2D_2p, 'RoLine2D_2p', 'Rect4CornersZ0Z1')

    elif obj_rep_in == 'Rect4CornersZ0Z1' and obj_rep_out == 'XYXYSin2W':
      XYZLgWsHA = OBJ_REPS_PARSE.encode_obj(bboxes, 'Rect4CornersZ0Z1', 'XYZLgWsHA')
      return OBJ_REPS_PARSE.encode_obj(XYZLgWsHA, 'XYZLgWsHA', 'XYXYSin2WZ0Z1')[:,:6]

    elif obj_rep_in == 'Rect4CornersZ0Z1' and obj_rep_out == 'XYXYSin2':
      return OBJ_REPS_PARSE.encode_obj(bboxes, obj_rep_in, 'XYXYSin2W')[:,:5]

    # extra  -------------------------------------------------------------------
    elif obj_rep_in == 'XYDAsinAsinSin2Z0Z1' and obj_rep_out == 'XYLgWsA':
      XYLgWsAsinSin2Z0Z1 = OBJ_REPS_PARSE.encode_obj(bboxes, 'XYDAsinAsinSin2Z0Z1', 'XYLgWsAsinSin2Z0Z1')
      return OBJ_REPS_PARSE.encode_obj(XYLgWsAsinSin2Z0Z1, 'XYLgWsAsinSin2Z0Z1', 'XYLgWsA')

    elif obj_rep_in == 'XYDAsinAsinSin2Z0Z1' and obj_rep_out == 'RoLine2D_2p':
      XYLgWsA = OBJ_REPS_PARSE.encode_obj(bboxes, 'XYDAsinAsinSin2Z0Z1', 'XYLgWsA')
      return OBJ_REPS_PARSE.encode_obj(XYLgWsA, 'XYLgWsA', 'RoLine2D_2p')

    elif obj_rep_in == 'XYLgWsAbsSin2Z0Z1' and obj_rep_out == 'XYLgWsA':
      return OBJ_REPS_PARSE.XYLgWsAbsSin2Z0Z1_TO_XYZLgWsHA(bboxes)[:,[0,1,3,4,6]]

    elif obj_rep_in == 'RoLine2D_2p' and obj_rep_out == 'XYXYSin2WZ0Z1':
      bboxes = OBJ_REPS_PARSE.Line2p_TO_UpRight_xyxy_sin2a(bboxes)
      return OBJ_REPS_PARSE.encode_obj(bboxes, 'XYXYSin2', 'XYXYSin2WZ0Z1')

    elif obj_rep_in == 'XYXYSin2WZ0Z1' and obj_rep_out == 'RoLine2D_2p':
      return OBJ_REPS_PARSE.XYXYSin2_TO_RoLine2D_2p(bboxes[:,:5])

    elif obj_rep_in == 'XYXYSin2WZ0Z1' and obj_rep_out == 'XYZLgWsHA':
      box2d = OBJ_REPS_PARSE.encode_obj(bboxes[:,:6], 'XYXYSin2W', 'XYLgWsA')
      z0 = bboxes[:,6:7]
      z1 = bboxes[:,7:8]
      zc = bboxes[:,6:8].mean(1)[:,None]
      zs = np.abs(z1-z0)
      box3d = np.concatenate([box2d[:,:2], zc, box2d[:,2:4], zs, box2d[:,4:5]], axis=1)
      return box3d

    elif obj_rep_in == 'XYXYSin2WZ0Z1' and obj_rep_out == 'XYLgWsA':
      bboxes = OBJ_REPS_PARSE.encode_obj(bboxes, 'XYXYSin2WZ0Z1', 'XYZLgWsHA')
      return OBJ_REPS_PARSE.encode_obj(bboxes, 'XYZLgWsHA', 'XYLgWsA')

    elif obj_rep_in == 'XYLgWsA' and obj_rep_out == 'RoLine2D_2p':
      XYXYSin2W = OBJ_REPS_PARSE.encode_obj(bboxes, 'XYLgWsA','XYXYSin2W')
      return OBJ_REPS_PARSE.encode_obj(XYXYSin2W[:,:5], 'XYXYSin2', 'RoLine2D_2p')


    elif obj_rep_in == 'XYXYSin2WZ0Z1' and obj_rep_out == 'XYLgWsAbsSin2Z0Z1':
      XYZLgWsHA = OBJ_REPS_PARSE.encode_obj(bboxes, 'XYXYSin2WZ0Z1', 'XYZLgWsHA')
      bboxes_new = OBJ_REPS_PARSE.encode_obj(XYZLgWsHA, 'XYZLgWsHA', 'XYLgWsAbsSin2Z0Z1')
      return bboxes_new

    elif obj_rep_in == 'XYXYSin2WZ0Z1' and obj_rep_out == 'XYDAsinAsinSin2Z0Z1':
      XYLgWsAsinSin2Z0Z1= OBJ_REPS_PARSE.encode_obj(bboxes, 'XYXYSin2WZ0Z1', 'XYLgWsAsinSin2Z0Z1')
      return OBJ_REPS_PARSE.encode_obj(XYLgWsAsinSin2Z0Z1, 'XYLgWsAsinSin2Z0Z1', 'XYDAsinAsinSin2Z0Z1')

    elif obj_rep_in == 'XYLgWsAbsSin2Z0Z1' and obj_rep_out == 'XYXYSin2':
      RoLine2D_2p = OBJ_REPS_PARSE.encode_obj(bboxes, 'XYLgWsAbsSin2Z0Z1', 'RoLine2D_2p')
      return OBJ_REPS_PARSE.encode_obj(RoLine2D_2p, 'RoLine2D_2p', 'XYXYSin2')

    elif obj_rep_in == 'XYLgWsAbsSin2Z0Z1' and obj_rep_out == 'RoLine2D_2p':
      XYZLgWsHA = OBJ_REPS_PARSE.encode_obj(bboxes, 'XYLgWsAbsSin2Z0Z1', 'XYZLgWsHA')
      RoLine2D_2p = OBJ_REPS_PARSE.encode_obj(XYZLgWsHA, 'XYZLgWsHA', 'RoLine2D_2p')
      return RoLine2D_2p

    elif obj_rep_in == 'XYXYSin2' and obj_rep_out == 'XYLgWsAbsSin2Z0Z1':
      XYXYSin2WZ0Z1 = OBJ_REPS_PARSE.encode_obj(bboxes, 'XYXYSin2', 'XYXYSin2WZ0Z1')
      return OBJ_REPS_PARSE.encode_obj(XYXYSin2WZ0Z1, 'XYXYSin2WZ0Z1', 'XYLgWsAbsSin2Z0Z1')

    # expand size: add width, 2D -> 3D -----------------------------------------------------------------
    elif obj_rep_in == 'XYXYSin2' and obj_rep_out == 'XYLgWsA':
        # to XYXYSin2W
        bboxes = np.concatenate([bboxes, bboxes[:,0:1]*0], axis=1)
        bboxes_csa =  OBJ_REPS_PARSE.XYXYSin2W_TO_XYLgWsA(bboxes)
        return bboxes_csa

    elif obj_rep_in == 'XYXYSin2' and obj_rep_out == 'XYZLgWsHA':
      XYLgWsA = OBJ_REPS_PARSE.encode_obj(bboxes, 'XYXYSin2', 'XYLgWsA')
      return OBJ_REPS_PARSE.encode_obj(XYLgWsA, 'XYLgWsA', 'XYZLgWsHA')

    elif obj_rep_in == 'XYXYSin2' and obj_rep_out == 'XYXYSin2WZ0Z1':
      ze = bboxes[:,:3]*0
      return np.concatenate([bboxes, ze], axis=1)

    elif obj_rep_in == 'XYLgWsA' and obj_rep_out == 'XYZLgWsHA':
      ze = bboxes[:,:1]*0
      return np.concatenate([bboxes[:,[0,1]], ze, bboxes[:,[2,3]], ze, bboxes[:,[4]]], axis=1)

    # --------------------------------------------------------------------------

    elif obj_rep_in == 'XYZLgWsHA'  and obj_rep_out == 'XYLgWsSin2Sin4Z0Z1':
      return OBJ_REPS_PARSE.XYZLgWsHA_to_XYLgWsSin2Sin4Z0Z1(bboxes)

    elif obj_rep_in == 'XYZLgWsHA'  and obj_rep_out == 'XYLgWsAsinSin2Z0Z1':
      return OBJ_REPS_PARSE.XYZLgWsHA_to_XYLgWsAsinSin2Z0Z1(bboxes)

    elif obj_rep_in == 'XYXYSin2WZ0Z1'  and obj_rep_out == 'XYLgWsAsinSin2Z0Z1':
      XYZLgWsHA = OBJ_REPS_PARSE.encode_obj(bboxes, 'XYXYSin2WZ0Z1', 'XYZLgWsHA')
      return  OBJ_REPS_PARSE.encode_obj(XYZLgWsHA, 'XYZLgWsHA', 'XYLgWsAsinSin2Z0Z1')

    elif obj_rep_in == 'XYLgWsSin2Sin4Z0Z1' and obj_rep_out == 'XYZLgWsHA':
      return OBJ_REPS_PARSE.XYLgWsSin2Sin4Z0Z1_to_XYZLgWsHA(bboxes)

    elif obj_rep_in == 'XYLgWsAsinSin2Z0Z1' and obj_rep_out == 'XYXYSin2':
      return OBJ_REPS_PARSE.XYLgWsAsinSin2Z0Z1_to_XYXYSin2(bboxes)

    elif obj_rep_in == 'XYLgWsAsinSin2Z0Z1' and obj_rep_out == 'XYZLgWsHA':
      return OBJ_REPS_PARSE.XYLgWsAsinSin2Z0Z1_to_XYZLgWsHA(bboxes)

    elif obj_rep_in == 'XYLgWsSin2Sin4Z0Z1' and obj_rep_out == 'XYLgWsA':
      XYZLgWsHA = OBJ_REPS_PARSE.XYLgWsSin2Sin4Z0Z1_to_XYZLgWsHA(bboxes)
      return XYZLgWsHA[:, [0,1,3,4,6]]

    elif obj_rep_in == 'XYLgWsAsinSin2Z0Z1' and obj_rep_out == 'XYLgWsA':
      XYZLgWsHA = OBJ_REPS_PARSE.XYLgWsAsinSin2Z0Z1_to_XYZLgWsHA(bboxes)
      return XYZLgWsHA[:, [0,1,3,4,6]]

    elif obj_rep_in == 'XYLgWsAsinSin2Z0Z1' and obj_rep_out == 'RoLine2D_2p':
      return OBJ_REPS_PARSE.XYLgWsAsinSin2Z0Z1_to_RoLine2D_2p(bboxes)

    elif obj_rep_in == 'XYLgWsA' and obj_rep_out == 'XYLgWsAsinSin2Z0Z1':
      return OBJ_REPS_PARSE.XYLgWsA_TO_XYLgWsAsinSin2Z0Z1(bboxes)

    elif obj_rep_in == 'XYLgWsSin2Sin4Z0Z1' and obj_rep_out == 'RoLine2D_2p':
      return OBJ_REPS_PARSE.XYLgWsSin2Sin4Z0Z1_to_RoLine2D_2p(bboxes)

    elif obj_rep_in == 'RoLine2D_2p' and obj_rep_out == 'XYLgWsSin2Sin4Z0Z1':
      return OBJ_REPS_PARSE.Line2p_TO_XYLgWsSin2Sin4Z0Z1(bboxes)

    elif obj_rep_in == 'RoLine2D_2p' and obj_rep_out == 'XYLgWsAsinSin2Z0Z1':
      return OBJ_REPS_PARSE.Line2p_TO_XYLgWsAsinSin2Z0Z1(bboxes)


    elif obj_rep_in == 'XYXYSin2' and obj_rep_out == 'XYLgWsAsinSin2Z0Z1':
      return OBJ_REPS_PARSE.UpRight_xyxy_sin2a_TO_XYLgWsAsinSin2Z0Z1(bboxes)

    elif obj_rep_in == 'XYLgWsA' and obj_rep_out == 'XYXYSin2W':
      bboxes_s2t = OBJ_REPS_PARSE.XYLgWsA_TO_XYXYSin2W(bboxes)
      check = 1
      if check and bboxes_s2t.shape[0]>0:
        bboxes_c = OBJ_REPS_PARSE.encode_obj(bboxes_s2t, 'XYXYSin2W', 'XYLgWsA')
        err = bboxes_c - bboxes
        err[:,-1] = limit_period_np(err[:,-1], 0.5, np.pi)
        err = np.abs(err).max()
        if not (err.size==0 or np.abs(err).max() < 1e-1):
          import pdb; pdb.set_trace()  # XXX BREAKPOINT
          pass
      return bboxes_s2t

    elif obj_rep_in == 'XYLgWsA' and obj_rep_out == 'XYXYSin2WZ0Z1':
      XYXYSin2W = OBJ_REPS_PARSE.encode_obj(bboxes, obj_rep_in, 'XYXYSin2W')
      return np.concatenate( [XYXYSin2W, XYXYSin2W[:,:2]*0], 1 )

    elif obj_rep_in == 'XYLgA' and obj_rep_out == 'RoLine2D_2p':
      return OBJ_REPS_PARSE.XYLgA_TO_RoLine2D_2p(bboxes)

    elif obj_rep_in == 'RoLine2D_2p' and obj_rep_out == 'XYLgA':
      return OBJ_REPS_PARSE.RoLine2D_2p_TO_CenterLengthAngle(bboxes)

    elif obj_rep_in == 'RoLine2D_2p' and obj_rep_out == 'XYLgWsA':
      return OBJ_REPS_PARSE.RoLine2D_2p_TO_XYLgWsA(bboxes)

    elif obj_rep_in == 'XYXYSin2' and obj_rep_out == 'XYLgA':
      lines_2p = OBJ_REPS_PARSE.encode_obj(bboxes, 'XYXYSin2', 'RoLine2D_2p')
      lines_cla = OBJ_REPS_PARSE.encode_obj(lines_2p, 'RoLine2D_2p', 'XYLgA')
      return lines_cla



    elif obj_rep_in == 'XYXYSin2WZ0Z1' and obj_rep_out == 'XYXYSin2':
      return bboxes[:, :5]

    elif obj_rep_in == 'XYXYSin2WZ0Z1' and obj_rep_out == 'Top_Corners':
      corners_z0z1 = OBJ_REPS_PARSE.encode_obj(bboxes, 'XYXYSin2WZ0Z1', 'Rect4CornersZ0Z1')
      z0 = corners_z0z1[:,8:9]
      z1 = corners_z0z1[:,9:10]
      top_corners = corners_z0z1[:,:8].reshape(-1,4,2)
      z1 = np.repeat(z1[:,None,:], 4, 1)
      top_corners = np.concatenate([top_corners, z1], axis=2).reshape(-1,12)
      return top_corners

    elif obj_rep_in == 'XYZLgWsHA' and obj_rep_out == 'Top_Corners':
      XYXYSin2WZ0Z1 = OBJ_REPS_PARSE.encode_obj(bboxes, 'XYZLgWsHA', 'XYXYSin2WZ0Z1')
      return OBJ_REPS_PARSE.encode_obj(XYXYSin2WZ0Z1, 'XYXYSin2WZ0Z1', 'Top_Corners')

    elif obj_rep_in == 'XYXYSin2WZ0Z1' and obj_rep_out == 'Bottom_Corners':
      corners_z0z1 = OBJ_REPS_PARSE.encode_obj(bboxes, 'XYXYSin2WZ0Z1', 'Rect4CornersZ0Z1')
      z0 = corners_z0z1[:,8:9]
      z1 = corners_z0z1[:,9:10]
      bot_corners = corners_z0z1[:,:8].reshape(-1,4,2)
      z0 = np.repeat(z0[:,None,:], 4, 1)
      bot_corners = np.concatenate([bot_corners, z0], axis=2).reshape(-1,12)
      return bot_corners

    elif obj_rep_in == 'XYZLgWsHA' and obj_rep_out == 'Bottom_Corners':
      XYXYSin2WZ0Z1 = OBJ_REPS_PARSE.encode_obj(bboxes, 'XYZLgWsHA', 'XYXYSin2WZ0Z1')
      return OBJ_REPS_PARSE.encode_obj(XYXYSin2WZ0Z1, 'XYXYSin2WZ0Z1', 'Bottom_Corners')

    elif obj_rep_in == 'XYZLgWsHA'  and obj_rep_out == 'XYXYSin2WZ0Z1':
      # XYLgWsA
      XYLgWsA = bboxes[:, [0,1, 3,4, 6]]
      z0 = bboxes[:, 2:3] - bboxes[:, 5:6]/2
      z1 = bboxes[:, 2:3] + bboxes[:, 5:6]/2
      line2d = OBJ_REPS_PARSE.encode_obj(XYLgWsA, 'XYLgWsA', 'XYXYSin2W')
      line3d = np.concatenate([line2d, z0, z1], axis=1)
      return line3d

    assert False, f'Not implemented:\nobj_rep_in: {obj_rep_in}\nobj_rep_out: {obj_rep_out}'

  def lines2d_to_lines3d(lines2d, ):
    '''
      lines2d: [n,5]  XYXYSin2
      out: XYXYSin2WZ0Z1
    '''
    n = lines2d.shape[0]
    tmp = np.ones([n,3])
    tmp[:,0] = 2
    tmp[:,1] = 0
    tmp[:,2] = 75
    lines3d = np.concatenate([lines2d, tmp], axis=1)
    return lines3d

  @staticmethod
  def make_x_long_dim(bboxes, obj_rep):
    if obj_rep in ['XYLgWsA', 'XYZLgWsHA']:
        if obj_rep == 'XYLgWsA':
          xi, yi, ai = 2, 3, 4
        if obj_rep == 'XYZLgWsHA':
          xi, yi, ai = 3, 4, 6
        switch = (bboxes[:,yi] > bboxes[:,xi]).astype(bboxes.dtype)
        bboxes[:,ai] = limit_period_np( bboxes[:,ai] + switch * np.pi / 2, 0.5, np.pi )
        switch = switch.reshape(-1,1)
        bboxes[:,[xi,yi]] = bboxes[:,[xi,yi]] * (1-switch) + bboxes[:,[yi,xi]] * switch

    if obj_rep in ['XYLgWsSin2Sin4Z0Z1', 'XYLgWsSin2Sin4']:
        xi, yi = 2, 3
        switch = (bboxes[:,yi] > bboxes[:,xi]).astype(bboxes.dtype).reshape(-1,1)
        bboxes[:,[4,5]] = bboxes[:,[4,5]] - switch * bboxes[:,[4,5]] * 2
        bboxes[:,[xi,yi]] = bboxes[:,[xi,yi]] * (1-switch) + bboxes[:,[yi,xi]] * switch

    if obj_rep in ['XYLgWsAsinSin2Z0Z1', 'XYLgWsAsinSin2']:
        xi, yi = 2, 3
        switch = (bboxes[:,yi] > bboxes[:,xi]).astype(bboxes.dtype).reshape(-1,1)
        bboxes[:,[5]] = bboxes[:,[5]] - switch * bboxes[:,[5]] * 2
        bboxes[:,[xi,yi]] = bboxes[:,[xi,yi]] * (1-switch) + bboxes[:,[yi,xi]] * switch

    return bboxes

  @staticmethod
  def check_x_long_dim(bboxes, obj_rep):
    if obj_rep in ['XYLgWsA', 'XYZLgWsHA']:
        if obj_rep == 'XYLgWsA':
          xi, yi, ai = 2, 3, 4
        if obj_rep == 'XYZLgWsHA':
          xi, yi, ai = 3, 4, 6
        assert np.all( bboxes[:,yi] <= bboxes[:,xi] )

    if obj_rep in ['XYLgWsSin2Sin4Z0Z1', 'XYLgWsSin2Sin4']:
        xi, yi, ai = 2, 3, 4
        assert np.all( bboxes[:,yi] <= bboxes[:,xi] )
    return bboxes

  @staticmethod
  def XYDAsinAsinSin2Z0Z1_TO_XYZLgWsHA(bboxes):
    n = bboxes.shape[0]
    bboxes_new = np.zeros([n, 7])
    bboxes_new[:,:2] = bboxes[:,:2]
    diag = bboxes[:,2]
    abs_sin_alpha = bboxes[:,3]
    assert np.all(abs_sin_alpha < 1.1)
    abs_sin_alpha = np.clip(abs_sin_alpha, a_min=0, a_max=1)
    alpha = np.arcsin(abs_sin_alpha)
    length = diag * np.cos(alpha / 2)
    width = diag * np.sin(alpha / 2)
    bboxes_new[:, 3] = length
    bboxes_new[:, 4] = width
    z0 = bboxes[:,6]
    z1 = bboxes[:,7]
    bboxes_new[:, 2] = (z0+z1)/2
    bboxes_new[:, 5] = z1-z0

    abs_sin_theta = bboxes[:,4]
    sin2_theta = bboxes[:,5]
    is_pos = sin2_theta >= 0
    theta_abs = np.arcsin(abs_sin_theta)
    theta = theta_abs * is_pos - theta_abs * (1-is_pos)
    bboxes_new[:,6] = theta
    return bboxes_new

  def XYZLgWsHA_TO_XYDAsinAsinSin2Z0Z1(bboxes):
    n = bboxes.shape[0]
    bboxes_new = np.zeros([n, 8])
    bboxes_new[:,:2] = bboxes[:,:2]
    length = bboxes[:,3]
    width = bboxes[:,4]
    diag = np.sqrt(length **2 + width**2)
    bboxes_new[:,2] = diag
    alpha = np.arcsin(width / diag)*2
    abs_sin_alpha = np.sin(alpha)
    bboxes_new[:,3] = abs_sin_alpha
    theta = bboxes[:,6]
    bboxes_new[:,4] = np.abs(np.sin(theta))
    bboxes_new[:,5] = np.sin(theta*2)
    zc = bboxes[:,2]
    height = bboxes[:,5]
    bboxes_new[:,6] = zc - height/2
    bboxes_new[:,7] = zc + height/2
    return bboxes_new

  @staticmethod
  def XYZLgWsHA_to_XYLgWsAsinSin2Z0Z1(bboxes_csa):
    n = bboxes_csa.shape[0]
    theta = bboxes_csa[:,6]
    bboxes_ass2 = np.zeros([n, 8], dtype=bboxes_csa.dtype)
    bboxes_ass2[:,[0,1]] = bboxes_csa[:,[0,1]]
    bboxes_ass2[:,2] = bboxes_csa[:,[3,4]].max(axis=1)
    bboxes_ass2[:,3] = bboxes_csa[:,[3,4]].min(axis=1)
    bboxes_ass2[:,6] = bboxes_csa[:,2] - bboxes_csa[:,5] / 2.0
    bboxes_ass2[:,7] = bboxes_csa[:,2] + bboxes_csa[:,5] / 2.0
    bboxes_ass2[:,5] = np.sin(theta * 2)
    bboxes_ass2[:,4] = np.abs(np.sin(theta))
    return bboxes_ass2

  @staticmethod
  def XYZLgWsHA_to_XYLgWsSin2Sin4Z0Z1(bboxes_csa):
    n = bboxes_csa.shape[0]
    bboxes_s2s4 = np.zeros([n, 8], dtype=bboxes_csa.dtype)
    bboxes_s2s4[:,[0,1]] = bboxes_csa[:,[0,1]]
    bboxes_s2s4[:,2] = bboxes_csa[:,[3,4]].max(axis=1)
    bboxes_s2s4[:,3] = bboxes_csa[:,[3,4]].min(axis=1)
    bboxes_s2s4[:,6] = bboxes_csa[:,2] - bboxes_csa[:,5] / 2.0
    bboxes_s2s4[:,7] = bboxes_csa[:,2] + bboxes_csa[:,5] / 2.0
    theta = bboxes_csa[:,6]
    bboxes_s2s4[:,4] = np.sin(theta * 2)
    bboxes_s2s4[:,5] = np.sin(theta * 4)
    return bboxes_s2s4

  @staticmethod
  def XYLgWsAsinSin2Z0Z1_to_XYZLgWsHA(bboxes_ass2):
    n = bboxes_ass2.shape[0]
    bboxes_csa = np.zeros([n, 7], dtype=bboxes_ass2.dtype)
    bboxes_csa[:, [0,1]] = bboxes_ass2[:, [0,1]]
    bboxes_csa[:, 2] = bboxes_ass2[:, [6,7]].mean(axis=1)
    bboxes_csa[:, [3,4]] = bboxes_ass2[:, [2,3]]
    bboxes_csa[:, 5] = bboxes_ass2[:, 7] - bboxes_ass2[:, 6]
    abssin = bboxes_ass2[:, 4]
    sin2theta = bboxes_ass2[:, 5]
    theta_0 = np.arcsin(abssin)
    theta_1 = -theta_0
    flag = (sin2theta >= 0).astype(np.int)
    theta = theta_0 * flag + theta_1 * (1-flag)
    bboxes_csa[:, 6] = theta
    return bboxes_csa

  @staticmethod
  def XYLgWsAsinSin2Z0Z1_to_XYXYSin2(bboxes):
    XYZLgWsHA = OBJ_REPS_PARSE.encode_obj(bboxes, 'XYLgWsAsinSin2Z0Z1', 'XYZLgWsHA')
    XYXYSin2WZ0Z1 = OBJ_REPS_PARSE.encode_obj(XYZLgWsHA, 'XYZLgWsHA', 'XYXYSin2WZ0Z1')
    return XYXYSin2WZ0Z1[:,:5]

  @staticmethod
  def XYLgWsSin2Sin4Z0Z1_to_XYZLgWsHA(bboxes_s2s4):
    n = bboxes_s2s4.shape[0]
    bboxes_csa = np.zeros([n, 7], dtype=bboxes_s2s4.dtype)
    bboxes_csa[:, [0,1]] = bboxes_s2s4[:, [0,1]]
    bboxes_csa[:, 2] = bboxes_s2s4[:, [6,7]].mean(axis=1)
    bboxes_csa[:, [3,4]] = bboxes_s2s4[:, [2,3]]
    bboxes_csa[:, 5] = bboxes_s2s4[:, 7] - bboxes_s2s4[:, 6]
    sin2theta = bboxes_s2s4[:, 4]
    sin4theta = bboxes_s2s4[:, 5]
    assert np.all(sin2theta < 1.1)
    sin2theta = np.clip(sin2_theta, a_min=-1, a_max=1)
    theta_0 = np.arcsin(sin2theta) / 2
    theta_1 = limit_period_np( np.pi/2 - theta_0, 0.5, np.pi )
    flag = (sin2theta * sin4theta >= 0).astype(np.int)
    theta = theta_0 * flag + theta_1 * (1-flag)
    bboxes_csa[:, 6] = theta
    return bboxes_csa

  @staticmethod
  def XYLgWsSin2Sin4Z0Z1_to_RoLine2D_2p(bboxes):
    debug = 0
    if debug:
      from tools.visual_utils import _show_objs_ls_points_ls, _show_3d_points_objs_ls
      bboxes[:,3] = 1
      bboxes[:,-1] = 10
      _show_objs_ls_points_ls((512,512), objs_ls = [bboxes], obj_rep='XYLgWsSin2Sin4Z0Z1')
      #_show_3d_points_objs_ls(objs_ls=[bboxes], obj_rep='XYLgWsSin2Sin4Z0Z1')
    XYZLgWsHA = OBJ_REPS_PARSE.encode_obj(bboxes, 'XYLgWsSin2Sin4Z0Z1', 'XYZLgWsHA')
    XYXYSin2WZ0Z1 = OBJ_REPS_PARSE.encode_obj(XYZLgWsHA,  'XYZLgWsHA', 'XYXYSin2WZ0Z1')
    if debug:
      #_show_3d_points_objs_ls(objs_ls=[XYZLgWsHA], obj_rep='XYZLgWsHA')
      _show_3d_points_objs_ls(objs_ls=[XYXYSin2WZ0Z1], obj_rep='XYXYSin2WZ0Z1')
    XYXYSin2 = OBJ_REPS_PARSE.encode_obj(XYXYSin2WZ0Z1, 'XYXYSin2WZ0Z1', 'XYXYSin2')
    RoLine2D_2p = OBJ_REPS_PARSE.encode_obj( XYXYSin2, 'XYXYSin2',  'RoLine2D_2p' )

    if debug:
      _show_objs_ls_points_ls((512, 512), [RoLine2D_2p], obj_rep='RoLine2D_2p')
    return RoLine2D_2p

  @staticmethod
  def XYLgWsAsinSin2Z0Z1_to_RoLine2D_2p(bboxes):
    debug = 0
    if debug:
      from tools.visual_utils import _show_objs_ls_points_ls, _show_3d_points_objs_ls
      bboxes[:,3] = 1
      bboxes[:,-1] = 10
      _show_objs_ls_points_ls((512,512), objs_ls = [bboxes], obj_rep='XYLgWsAsinSin2Z0Z1')
      #_show_3d_points_objs_ls(objs_ls=[bboxes], obj_rep='XYLgWsAsinSin2Z0Z1')
    XYZLgWsHA = OBJ_REPS_PARSE.encode_obj(bboxes, 'XYLgWsAsinSin2Z0Z1', 'XYZLgWsHA')
    XYXYSin2WZ0Z1 = OBJ_REPS_PARSE.encode_obj(XYZLgWsHA,  'XYZLgWsHA', 'XYXYSin2WZ0Z1')
    if debug:
      #_show_3d_points_objs_ls(objs_ls=[XYZLgWsHA], obj_rep='XYZLgWsHA')
      _show_3d_points_objs_ls(objs_ls=[XYXYSin2WZ0Z1], obj_rep='XYXYSin2WZ0Z1')
    XYXYSin2 = OBJ_REPS_PARSE.encode_obj(XYXYSin2WZ0Z1, 'XYXYSin2WZ0Z1', 'XYXYSin2')
    RoLine2D_2p = OBJ_REPS_PARSE.encode_obj( XYXYSin2, 'XYXYSin2',  'RoLine2D_2p' )

    if debug:
      _show_objs_ls_points_ls((512, 512), [RoLine2D_2p], obj_rep='RoLine2D_2p')
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass
    return RoLine2D_2p

  @staticmethod
  def XYLgA_TO_RoLine2D_2p(bboxes):
    center = bboxes[:,:2]
    length = bboxes[:,2:3]
    angle = bboxes[:,3]
    vec = vec_from_angle_with_x_np(angle)
    corner0 = center - vec * length /2
    corner1 = center + vec * length /2
    line2d_2p = np.concatenate([corner0, corner1], axis=1)

    check=0
    if check:
      bboxes_c = OBJ_REPS_PARSE.RoLine2D_2p_TO_CenterLengthAngle(line2d_2p)
      err0 = bboxes - bboxes_c
      err0[:,3] = limit_period_np(err0[:,3] , 0.5, np.pi)
      err = np.max(np.abs(err0))
      if not (err.size==0 or np.abs(err).max() < 1e-3):
        print(err0)
        print(err)
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        pass
      pass
    return line2d_2p

  @staticmethod
  def RoLine2D_2p_TO_CenterLengthAngle(bboxes):
    corner0 = bboxes[:,0:2]
    corner1 = bboxes[:,2:4]
    center = bboxes.reshape(-1,2,2).mean(axis=1)
    vec = corner1 - corner0
    length = np.linalg.norm(vec, axis=1)[:,None]
    angle = angle_with_x_np(vec, scope_id=2)[:,None]
    bboxes_CLA = np.concatenate([center, length, angle], axis=1)
    return bboxes_CLA

  @staticmethod
  def RoLine2D_2p_TO_XYLgWsA(bboxes):
    corner0 = bboxes[:,0:2]
    corner1 = bboxes[:,2:4]
    center = bboxes.reshape(-1,2,2).mean(axis=1)
    vec = corner1 - corner0
    length = np.linalg.norm(vec, axis=1)[:,None]
    angle = angle_with_x_np(vec, scope_id=2)[:,None]
    # Because axis-y of img points to bottom for img, it is positive for
    # anti-clock wise. Change to positive for clock-wise
    width = np.zeros([bboxes.shape[0], 1], dtype=np.float32)
    XYLgWsA = np.concatenate([center, length, width, angle], axis=1)
    return XYLgWsA

  @staticmethod
  def XYLgWsA_TO_XYXYSin2W(bboxes):
    '''
    In the input , x either y can be the longer one.
    If y is the longer one, angle = angle + 90.
    To make x the longer one.
    '''
    center = bboxes[:,:2]
    size = bboxes[:,2:4]
    angle = bboxes[:,4:5]
    length = size.max(1)[:,None]
    thickness = size.min(1)[:,None]

    line2d_angle = np.concatenate([center, length, angle], axis=1)
    line2d_2p = OBJ_REPS_PARSE.XYLgA_TO_RoLine2D_2p(line2d_angle)
    line2d_sin2 = OBJ_REPS_PARSE.Line2p_TO_UpRight_xyxy_sin2a(line2d_2p)
    line2d_sin2tck = np.concatenate([line2d_sin2, thickness], axis=1)

    err = np.sin(angle.reshape(-1)*2) - line2d_sin2tck[:, -2]
    if not (err.size==0 or ( np.abs(err).max() < 1e-1 and np.abs(err).mean() < 1e-2)):
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass
    return line2d_sin2tck

  @staticmethod
  def XYXYSin2W_TO_XYLgWsA(bboxes, check_sin2=True):
    thickness = bboxes[:,5:6]
    lines_2p = OBJ_REPS_PARSE.XYXYSin2_TO_RoLine2D_2p(bboxes[:,:5])
    lines_CenLengthAngle = OBJ_REPS_PARSE.RoLine2D_2p_TO_CenterLengthAngle(lines_2p)
    boxes_csa = np.concatenate([lines_CenLengthAngle[:,[0,1,2]], thickness, lines_CenLengthAngle[:,[3]]], axis=1)
    check_sin2 = 0
    if check_sin2:
      err = np.sin(boxes_csa[:,-1]*2) - bboxes[:,4]
      if not (err.size==0 or max_err < 2e-1):
        i = np.abs(err).argmax()
        box_sin2_i = bboxes[i]
        box_csa_i = boxes_csa[i]
        print(f'box_sin2: {box_sin2_i}\nbox_csa_i: {box_csa_i}')
        assert False, "Something is wrong. 1) the obj encoding, 2) the input not right"
        pass
    return boxes_csa

  @staticmethod
  def Line2p_TO_UpRight_xyxy_sin2a(bboxes):
    '''
    From RoLine2D_2p to XYXYSin2
    '''
    bboxes = bboxes.reshape(-1,2,2)
    xy_min = bboxes.min(axis=1)
    xy_max = bboxes.max(axis=1)
    centroid = (xy_min + xy_max) / 2
    bboxes_0 = bboxes - centroid.reshape(-1,1,2)
    top_ids = bboxes_0[:,:,1].argmin(axis=-1)
    nb = bboxes_0.shape[0]

    tmp = np.arange(nb)
    top_points = bboxes_0[tmp, top_ids]
    vec_start = np.array([[0, -1]] * nb, dtype=np.float32).reshape(-1,2)
    istopleft = sin2theta_np( vec_start, top_points).reshape(-1,1)

    bboxes_out = np.concatenate([xy_min, xy_max, istopleft], axis=1)
    return bboxes_out

  @staticmethod
  def Line2p_TO_XYLgWsSin2Sin4Z0Z1(bboxes):
    UpRight_xyxy_sin2a = OBJ_REPS_PARSE.encode_obj(bboxes, 'RoLine2D_2p', 'XYXYSin2')
    n = UpRight_xyxy_sin2a.shape[0]
    tmp = np.zeros([n, 3], dtype=np.float32)
    XYXYSin2WZ0Z1 = np.concatenate([ UpRight_xyxy_sin2a, tmp  ], axis=1)
    XYZLgWsHA = OBJ_REPS_PARSE.encode_obj( XYXYSin2WZ0Z1, 'XYXYSin2WZ0Z1', 'XYZLgWsHA')
    XYLgWsSin2Sin4Z0Z1 = OBJ_REPS_PARSE.encode_obj( XYZLgWsHA, 'XYZLgWsHA', 'XYLgWsSin2Sin4Z0Z1' )

    show = 0
    if show:
      from tools.visual_utils import _show_3d_points_objs_ls
      XYLgWsSin2Sin4Z0Z1[:, 3] = 1
      XYLgWsSin2Sin4Z0Z1[:, -1] = 10
      _show_3d_points_objs_ls(objs_ls=[XYLgWsSin2Sin4Z0Z1], obj_rep='XYLgWsSin2Sin4Z0Z1')
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
    return XYLgWsSin2Sin4Z0Z1

  @staticmethod
  def Line2p_TO_XYLgWsAsinSin2Z0Z1(bboxes):
    UpRight_xyxy_sin2a = OBJ_REPS_PARSE.encode_obj(bboxes, 'RoLine2D_2p', 'XYXYSin2')
    n = UpRight_xyxy_sin2a.shape[0]
    tmp = np.zeros([n, 3], dtype=np.float32)
    XYXYSin2WZ0Z1 = np.concatenate([ UpRight_xyxy_sin2a, tmp  ], axis=1)
    XYZLgWsHA = OBJ_REPS_PARSE.encode_obj( XYXYSin2WZ0Z1, 'XYXYSin2WZ0Z1', 'XYZLgWsHA')
    XYLgWsAsinSin2Z0Z1 = OBJ_REPS_PARSE.encode_obj( XYZLgWsHA, 'XYZLgWsHA', 'XYLgWsAsinSin2Z0Z1' )

    show = 0
    if show:
      from tools.visual_utils import _show_3d_points_objs_ls
      XYLgWsAsinSin2Z0Z1[:, 3] = 1
      XYLgWsAsinSin2Z0Z1[:, -1] = 10
      _show_3d_points_objs_ls(objs_ls=[XYLgWsAsinSin2Z0Z1], obj_rep='XYLgWsAsinSin2Z0Z1')
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
    return XYLgWsAsinSin2Z0Z1

  @staticmethod
  def XYXYSin2_TO_RoLine2D_2p(lines):
    '''
    From XYXYSin2 to RoLine2D_2p
    '''
    istopleft = (lines[:,4:5] >= 0).astype(lines.dtype)
    lines_2p = lines[:,:4] * istopleft +  lines[:,[0,3,2,1]] * (1-istopleft)
    return lines_2p

  @staticmethod
  def UpRight_xyxy_sin2a_TO_XYLgWsAsinSin2Z0Z1(bboxes):
    XYLgWsA = OBJ_REPS_PARSE.encode_obj(bboxes, 'XYXYSin2', 'XYLgWsA')
    XYLgWsAsinSin2Z0Z1 = OBJ_REPS_PARSE.encode_obj(XYLgWsA, 'XYLgWsA', 'XYLgWsAsinSin2Z0Z1')
    return XYLgWsAsinSin2Z0Z1

  @staticmethod
  def XYLgWsA_TO_XYLgWsAsinSin2Z0Z1(bboxes):
    theta = bboxes[:,4:5]
    AbsSin = np.abs(np.sin(theta))
    sin2theta = np.sin(theta * 2)
    tmp = np.zeros([bboxes.shape[0], 2])
    return np.concatenate([ bboxes[:,:4], AbsSin, sin2theta, tmp  ], axis=1)

  @staticmethod
  def XYLgWsAbsSin2Z0Z1_TO_XYZLgWsHA(bboxes):
    nb = bboxes.shape[0]
    bboxes_new = np.zeros([nb, 7])
    bboxes_new[:,[0,1]] = bboxes[:,[0,1]]
    bboxes_new[:,[3,4]] = bboxes[:,[2,3]]
    abs_angle = bboxes[:,4]
    sin2 = bboxes[:,5]
    pos = sin2>0
    angle = abs_angle * pos - abs_angle * (1-pos)
    bboxes_new[:,6] = angle
    z0 = bboxes[:,6]
    z1 = bboxes[:,7]
    bboxes_new[:,2] = (z0+z1)/2
    bboxes_new[:,5] = z1 - z0
    return bboxes_new

  @staticmethod
  def XYLgWsSin2Cos2Z0Z1_TO_XYZLgWsHA(bboxes):
    nb = bboxes.shape[0]
    bboxes_new = np.zeros([nb, 7])
    bboxes_new[:,[0,1]] = bboxes[:,[0,1]]
    bboxes_new[:,[3,4]] = bboxes[:,[2,3]]
    sin2theta = bboxes[:,4]
    cos2theta = bboxes[:,5]

    tmp = 1/np.sqrt(sin2theta**2 + cos2theta**2)
    cos2theta *= tmp
    sin2theta *= tmp

    sin2theta = np.clip(sin2theta, a_min=-1, a_max=1)
    cos2theta = np.clip(cos2theta, a_min=-1, a_max=1)

    theta_0 = np.arcsin(sin2theta) / 2
    theta_1 = limit_period_np( np.pi/2 - theta_0, 0.5, np.pi )
    flag = (cos2theta > 0).astype(np.int)
    theta = theta_0 * flag + theta_1 * (1-flag)

    bboxes_new[:,6] = theta
    z0 = bboxes[:,6]
    z1 = bboxes[:,7]
    bboxes_new[:,2] = (z0+z1)/2
    bboxes_new[:,5] = z1 - z0
    return bboxes_new

  @staticmethod
  def XYDRSin2Cos2Z0Z1_TO_XYZLgWsHA(bboxes):
    nb = bboxes.shape[0]
    bboxes_new = np.zeros([nb, 7])
    bboxes_new[:,[0,1]] = bboxes[:,[0,1]]
    diags = bboxes[:,[2]]
    lwRate = bboxes[:,[3]]
    lg = np.sqrt( diags**2 / (1 + lwRate**2) )
    ws = lg * lwRate
    bboxes_new[:,[3]] =  lg
    bboxes_new[:,[4]] =  ws
    sin2theta = bboxes[:,4]
    cos2theta = bboxes[:,5]

    tmp = 1/np.sqrt(sin2theta**2 + cos2theta**2)
    cos2theta *= tmp
    sin2theta *= tmp

    sin2theta = np.clip(sin2theta, a_min=-1, a_max=1)
    cos2theta = np.clip(cos2theta, a_min=-1, a_max=1)

    theta_0 = np.arcsin(sin2theta) / 2
    theta_1 = limit_period_np( np.pi/2 - theta_0, 0.5, np.pi )
    flag = (cos2theta > 0).astype(np.int)
    theta = theta_0 * flag + theta_1 * (1-flag)

    bboxes_new[:,6] = theta
    z0 = bboxes[:,6]
    z1 = bboxes[:,7]
    bboxes_new[:,2] = (z0+z1)/2
    bboxes_new[:,5] = z1 - z0
    return bboxes_new

  @staticmethod
  def XYZLgWsHA_TO_XYLgWsAbsSin2Z0Z1(bboxes):
    nb = bboxes.shape[0]
    bboxes_new = np.zeros([nb, 8])
    bboxes_new[:,[0,1]] = bboxes[:,[0,1]]
    bboxes_new[:,[2,3]] = bboxes[:,[3,4]]
    bboxes_new[:,[4]] = np.abs(bboxes[:,[6]])
    bboxes_new[:,[5]] = np.sin(2*bboxes[:,[6]])
    zc = bboxes[:,2]
    h = bboxes[:,5]
    bboxes_new[:,6] = zc - h/2
    bboxes_new[:,7] = zc + h/2
    return bboxes_new

  @staticmethod
  def XYZLgWsHA_TO_XYDRSin2Cos2Z0Z1(bboxes):
    nb = bboxes.shape[0]
    bboxes_new = np.zeros([nb, 8])
    lg = bboxes[:,[3]]
    ws = bboxes[:,[4]]
    diag = np.sqrt( lg**2 + ws**2 )
    bboxes_new[:,[0,1]] = bboxes[:,[0,1]]
    bboxes_new[:,[2]] = diag
    bboxes_new[:,[3]] = ws / lg
    bboxes_new[:,[4]] = np.sin(2*bboxes[:,[6]])
    bboxes_new[:,[5]] = np.cos(2*bboxes[:,[6]])
    zc = bboxes[:,2]
    h = bboxes[:,5]
    bboxes_new[:,6] = zc - h/2
    bboxes_new[:,7] = zc + h/2
    return bboxes_new
  @staticmethod
  def XYZLgWsHA_TO_XYLgWsSin2Cos2Z0Z1(bboxes):
    nb = bboxes.shape[0]
    bboxes_new = np.zeros([nb, 8])
    bboxes_new[:,[0,1]] = bboxes[:,[0,1]]
    bboxes_new[:,[2,3]] = bboxes[:,[3,4]]
    bboxes_new[:,[4]] = np.sin(2*bboxes[:,[6]])
    bboxes_new[:,[5]] = np.cos(2*bboxes[:,[6]])
    zc = bboxes[:,2]
    h = bboxes[:,5]
    bboxes_new[:,6] = zc - h/2
    bboxes_new[:,7] = zc + h/2
    return bboxes_new

  @staticmethod
  def XYLgWsAsinSin2Z0Z1_TO_XYDAsinAsinSin2Z0Z1(bboxes):
    bboxes = bboxes.copy()
    length = bboxes[:,2]
    width = bboxes[:,3]
    diag_len = np.sqrt(length ** 2 + width ** 2)
    sin_half_alpha = width / diag_len
    cos_half_alpha = length / diag_len
    sin_alpha = 2 * sin_half_alpha * cos_half_alpha
    abs_sin_alpha = np.abs(sin_alpha)
    bboxes[:,2] = diag_len
    bboxes[:,3] = abs_sin_alpha
    return bboxes

  @staticmethod
  def XYDAsinAsinSin2Z0Z1_TO_XYLgWsAsinSin2Z0Z1(bboxes):
    bboxes = bboxes.copy()
    diag_len = bboxes[:,2]
    abs_sin_alpha = bboxes[:,3]
    half_alpha = np.arcsin(abs_sin_alpha) / 2
    length = diag_len * np.cos(half_alpha)
    width = diag_len * np.sin(half_alpha)
    bboxes[:,2] = length
    bboxes[:,3] = width
    return bboxes

  @staticmethod
  def XYDAsinAsinSin2Z0Z1_TO_XYXYSin2WZ0Z1(bboxes):
    XYLgWsA = OBJ_REPS_PARSE.encode_obj(bboxes, 'XYDAsinAsinSin2Z0Z1', 'XYLgWsA')
    XYXYSin2W = OBJ_REPS_PARSE.encode_obj(XYLgWsA, 'XYLgWsA','XYXYSin2W')
    return np.concatenate([XYXYSin2W, bboxes[:,6:8]], axis=1)

  @staticmethod
  def XYDAsinAsinSin2Z0Z1_TO_Rect4CornersZ0Z1(bboxes):
    '''
    The 4 corners are stored in clock-wise order in img coordinate system, starting from 0 to 2pi.
    The order is same with geometry_utils.py / sort_four_corners
    '''
    n = bboxes.shape[0]
    center = bboxes[:,:2]
    diag = bboxes[:,2:3]
    abs_sin_alpha = bboxes[:,3]
    abs_sin_theta = bboxes[:,4]
    sin2_theta = bboxes[:,5]

    assert np.all(abs_sin_alpha < 1.1)
    abs_sin_alpha = np.clip(abs_sin_alpha, a_min=0, a_max=1)
    alpha = np.arcsin(abs_sin_alpha)
    theta_abs = np.arcsin(abs_sin_theta)
    is_pos = sin2_theta>0
    theta = theta_abs * is_pos - theta_abs * (1-is_pos)

    beta1 = theta - alpha/2
    beta2 = theta + alpha/2

    vec_1 = vec_from_angle_with_x_np(beta1) * diag / 2
    vec_2 = vec_from_angle_with_x_np(beta2) * diag / 2
    corner0 = center - vec_1
    corner2 = center + vec_1
    corner1 = center - vec_2
    corner3 = center + vec_2

    rect_4corners = np.concatenate([corner0, corner1, corner2, corner3], axis=1)
    tmp = torch.from_numpy( rect_4corners.reshape(n, 4, 2) )
    rect_4corners, _ = sort_four_corners(tmp)
    rect_4corners = rect_4corners.reshape(n, 8).numpy()

    bboxes = np.concatenate([rect_4corners, bboxes[:,6:8]], -1)
    return bboxes

  @staticmethod
  def update_corners_order(bboxes, obj_rep):
    assert obj_rep == 'Rect4CornersZ0Z1'
    assert bboxes.shape[1] == 10
    n = bboxes.shape[0]
    tmp = torch.from_numpy( bboxes[:,:8].reshape(n, 4, 2) )
    rect_4corners, _ = sort_four_corners(tmp)
    rect_4corners = rect_4corners.reshape(n, 8).numpy()
    bboxes = np.concatenate([rect_4corners, bboxes[:,8:10]], -1)
    return bboxes

  @staticmethod
  def get_8_corners(bboxes, obj_rep):
    '''
    bboxes: [n,7]
    corners_3d: [n,8,3]
    '''
    from tools.visual_utils import _show_3d_points_objs_ls
    n = bboxes.shape[0]
    Rect4CornersZ0Z1 = OBJ_REPS_PARSE.encode_obj(bboxes, obj_rep, 'Rect4CornersZ0Z1')
    corners_2d = Rect4CornersZ0Z1[:,:8].reshape(n, 4, 2)
    z0 = np.repeat(Rect4CornersZ0Z1[:, None, 8:9], 4, axis=1)
    z1 = np.repeat(Rect4CornersZ0Z1[:, None, 9:10], 4, axis=1)
    corners_3d_bot = np.concatenate([corners_2d, z0], axis=2)
    corners_3d_top = np.concatenate([corners_2d, z1], axis=2)
    corners_3d = np.concatenate([corners_3d_bot, corners_3d_top], axis=1)

    #_show_3d_points_objs_ls([corners_3d.reshape(-1,3)], objs_ls=[bboxes], obj_rep=obj_rep)
    return corners_3d

  @staticmethod
  def get_12_line_cors(bboxes, obj_rep):
    '''
    bboxes: [n,7]
    lines_3d: [n,12,2,3]
    '''
    n = bboxes.shape[0]
    corners_3d = OBJ_REPS_PARSE.get_8_corners(bboxes, obj_rep) # [n,8,3]
    lines_vertical = np.concatenate([corners_3d[:,4:8,None,:] , corners_3d[:,:4,None,:]], axis=2)
    corners_moved = corners_3d.reshape(n,2,4,3)[:,:,[1,2,3,0],:].reshape(n,8,3)
    lines_hori = np.concatenate([ corners_3d[:,:,None,:], corners_moved[:,:,None,:]], axis=2)
    lines_3d = np.concatenate([lines_hori, lines_vertical], axis=1)
    return lines_3d

  @staticmethod
  def get_border_4_lines(bboxes, obj_rep):
    n = bboxes.shape[0]
    if n==0:
      return np.repeat( bboxes[:,None], 4 , 1)
    corners = OBJ_REPS_PARSE.encode_obj(bboxes, obj_rep, 'Rect4CornersZ0Z1')[:,:8].reshape(n,4,2)
    ids_ls = [ [0,1], [1,2], [2,3], [3,0] ]
    line_cors = [ corners[:, ids].reshape(n,1,4) for ids in ids_ls]
    line_cors = np.concatenate(line_cors, 1).reshape(n*4, -1)
    lines = OBJ_REPS_PARSE.encode_obj(line_cors, 'RoLine2D_2p', obj_rep).reshape(n,4,-1)
    return lines

  @staticmethod
  def normalized_bev(bboxes, obj_rep, size=512):
    corners = OBJ_REPS_PARSE.encode_obj(bboxes, obj_rep, 'RoLine2D_2p').reshape(-1,2)
    xyz_min = corners.min(0)
    xyz_max = corners.max(0)
    scope = max(xyz_max[:2] - xyz_min[:2])
    pad = 10
    corners_norm = (corners - xyz_min) * (size - pad*2) / scope + pad
    bboxes_norm = OBJ_REPS_PARSE.encode_obj(corners_norm.reshape(-1,4), 'RoLine2D_2p', obj_rep)
    return bboxes_norm, xyz_min

class GraphUtils:
  @staticmethod
  def optimize_wall_graph_after_room_opt(walls, scores_in=None, obj_rep_in='XYZLgWsHA',
                     opt_graph_cor_dis_thr=0, min_out_length=0):
    from mmdet.ops.nms.nms_wrapper import nms_rotated_np
    from tools.visual_utils import _show_objs_ls_points_ls, _show_3d_points_objs_ls, _show_2d_bboxes_ids, show_1by1
    obj_rep = 'XYZLgWsHA'
    if scores_in is None:
      scores = walls[:,-1:].copy()
      scores[:] = 1
    else:
      scores = scores_in

    #walls, xyz_min = OBJ_REPS_PARSE.normalized_bev(walls, obj_rep_in)
    walls = OBJ_REPS_PARSE.encode_obj(walls, obj_rep_in, obj_rep)

    ids_org = np.arange(walls.shape[0])
    all_ids = []


    walls, scores = GraphUtils.merge_wall_corners(walls, scores, obj_rep, opt_graph_cor_dis_thr)

    walls, scores = GraphUtils.crop_long_walls(walls, scores, obj_rep)

    walls, scores, ids = GraphUtils.rm_short_walls(walls, scores, obj_rep, 3)
    all_ids.append(ids)
    walls, scores, ids = GraphUtils.nms(walls, scores, obj_rep, iou_thr=0.5, min_width_length_ratio=0.2)

    walls, scores = GraphUtils.merge_parallel_con_walls(walls, scores, obj_rep, show=0)
    #_show_objs_ls_points_ls( (512,512), [walls], obj_rep, obj_thickness=5 )
    #show_1by1((512,512), walls, obj_rep)

    walls, scores, ids = GraphUtils.rm_short_walls(walls, scores, obj_rep, min_out_length)
    all_ids.append(ids)


    walls, scores, ids = GraphUtils.nms(walls, scores, obj_rep, iou_thr=0.2, min_width_length_ratio=0.2)
    all_ids.append(ids)
    #_show_objs_ls_points_ls( (512,512), [walls], obj_rep, obj_thickness=5 )

    ids_out = ids_org.copy()
    for ids in all_ids:
      if ids is not None:
        ids_out = ids_out[ids]

    walls = OBJ_REPS_PARSE.encode_obj(walls, obj_rep, obj_rep_in)
    if scores_in is None:
      scores = None

    if 1:
      if not check_duplicate( walls[:,:7], obj_rep, 0.3 ):
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        pass
    #_show_objs_ls_points_ls( (512,512), [walls], obj_rep, obj_thickness=5 )
    return walls, scores, ids_out

  @staticmethod
  def optimize_wall_graph(walls, scores_in=None, obj_rep_in='XYZLgWsHA',
                     opt_graph_cor_dis_thr=0, min_out_length=0):
    from mmdet.ops.nms.nms_wrapper import nms_rotated_np
    obj_rep = 'XYZLgWsHA'
    if scores_in is None:
      scores = walls[:,-1:].copy()
      scores[:] = 1
    else:
      scores = scores_in

    #walls, xyz_min = OBJ_REPS_PARSE.normalized_bev(walls, obj_rep_in)
    walls = OBJ_REPS_PARSE.encode_obj(walls, obj_rep_in, obj_rep)

    ids_org = np.arange(walls.shape[0])
    all_ids = []
    walls, scores, ids = GraphUtils.rm_short_walls(walls, scores, obj_rep, min_out_length)
    all_ids.append(ids)

    walls = GraphUtils.align_small_angle_walls(walls, obj_rep)

    walls, scores = GraphUtils.merge_wall_corners(walls, scores, obj_rep, opt_graph_cor_dis_thr)

    walls, scores, ids = GraphUtils.rm_short_walls(walls, scores, obj_rep, min_out_length)
    all_ids.append(ids)

    walls, scores, ids = GraphUtils.crop_intersec_walls(walls, scores, obj_rep, opt_graph_cor_dis_thr)
    all_ids.append(ids)

    walls, scores, ids = GraphUtils.nms(walls, scores, obj_rep, iou_thr=0.2, min_width_length_ratio=0.3)
    all_ids.append(ids)

    walls, scores = GraphUtils.crop_long_walls(walls, scores, obj_rep)

    walls, scores, ids = GraphUtils.nms(walls, scores, obj_rep, iou_thr=0.2, min_width_length_ratio=0.3)
    all_ids.append(ids)

    ids_out = ids_org.copy()
    for ids in all_ids:
      if ids is not None:
        ids_out = ids_out[ids]

    walls = OBJ_REPS_PARSE.encode_obj(walls, obj_rep, obj_rep_in)
    if scores_in is None:
      scores = None

    check_duplicate( walls[:,:7], obj_rep )
    return walls, scores, ids_out

  @staticmethod
  def align_small_angle_walls(walls, obj_rep):
    angles0  = walls[:,6].copy()
    steps = angles0 / (np.pi/2)
    angles1 = np.round(steps) * np.pi/2
    gaps = steps - np.round(steps)
    gaps = np.abs(gaps) * 180 / np.pi
    align_mask = gaps < MIN_ALIGN_ANGLE
    angles_aligned = angles0 *(1-align_mask) + angles1 * align_mask
    walls[:,6]  = angles_aligned
    return walls

  @staticmethod
  def crop_intersec_walls(walls, scores, obj_rep, opt_graph_cor_dis_thr):
    '''
    use corner_1-wall_0 to crop wall_1
    conditions:
    1. the degree of corner_1-wall-0 is 0
    2. angle between wall_0 and wall_1 > np.pi/4
    3. corner_1-wall-0 is on wall_1
    '''
    from obj_geo_utils.geometry_utils import  points_in_lines
    from tools.visual_utils import _show_objs_ls_points_ls, _show_3d_points_objs_ls
    cor_degrees, cors = get_cor_degree(walls, 1, obj_rep)
    # find degree 0 walls
    ids = np.array( np.where(cor_degrees==0)).T
    m = ids.shape[0]
    if m == 0:
      return walls, scores, None
    #  find corners on the other wall
    in_line_mask = points_in_lines( cors[ids[:,0], ids[:,1]], cors, opt_graph_cor_dis_thr/2 )
    ids_2 = np.array( np.where(in_line_mask==1)).T
    walls_new = walls.copy()
    walls_added = []
    scores_added = []
    ids_out = list(range(walls.shape[0]))
    for i in range(ids_2.shape[0]):
      j,k = ids_2[i]
      idx_w_0, idx_cor = ids[j]
      w_fix = walls[idx_w_0]
      w_croped = walls[k]
      angle_dif = limit_period_np( w_croped[-1] - w_fix[-1], 0.5, np.pi )
      angle_dif = np.abs(angle_dif)
      if angle_dif < np.pi/4:
        continue
      w_a, w_b = crop_two_intersect_overlaip_wall(w_croped, w_fix, obj_rep)
      if 0:
        _show_objs_ls_points_ls((512,512), objs_ls= [walls, w_croped[None], w_fix[None]], obj_rep=obj_rep, obj_colors=['white', 'random', 'random'])
        _show_objs_ls_points_ls((512,512), objs_ls= [walls, w_fix[None]], obj_rep=obj_rep, obj_colors=['white', 'random'])
        _show_objs_ls_points_ls((512,512), objs_ls= [walls, w_a[None], w_b[None], w_fix[None]], obj_rep=obj_rep, obj_colors=['white', 'random', 'random', 'random'])
      walls_new[k] = w_a
      walls_added.append(w_b[None])
      scores_added.append(scores[k])
      ids_out += [k]
      pass
    if len(walls_added) == 0:
      return walls, scores, None
    ids_out = np.array(ids_out).reshape(-1)
    walls_added = np.concatenate(walls_added, 0)
    walls_new = np.concatenate([walls_new, walls_added], 0)
    scores_added = np.array(scores_added).reshape(-1)
    scores_new = np.concatenate([scores, scores_added], 0)
    #_show_objs_ls_points_ls((512,512), objs_ls= [walls_new], obj_rep=obj_rep, obj_colors=[ 'random'])
    return walls_new, scores_new, ids_out

  @staticmethod
  def nms(walls, scores_in, obj_rep, iou_thr, min_width_length_ratio):
    from mmdet.ops.nms.nms_wrapper import nms_rotated_np
    if scores_in is None:
      scores = walls[:,-1:].copy()
      scores[:] = 1
    else:
      scores = scores_in
    walls = np.concatenate([walls, scores.reshape(-1,1)], axis=1)
    walls, ids = nms_rotated_np(walls, obj_rep, iou_thr, min_width_length_ratio)
    if scores_in is None:
      scores_out = None
    else:
      scores_out = walls[:,-1]
    return walls[:,:-1], scores_out, ids

  @staticmethod
  def crop_long_walls(walls_0, scores_0, obj_rep, iou_thres=0.2):
    '''
      Do not  change the number of walls.

      walls_0: [n,7]
      scores_0: [n]

      walls_1: [m,7]
      scores_1: [m]

      m <= n

      Merge condition:
        1. iou > 0.3
        2. angle dif < 1
    '''
    assert obj_rep == 'XYZLgWsHA'
    from mmdet.core.bbox.geometry import dsiou_rotated_3d_bbox_np
    from tools.visual_utils import _show_objs_ls_points_ls, _show_3d_points_objs_ls
    from obj_geo_utils.geometry_utils import points_in_lines

    debug = 0
    check_no_duplicate = 1

    if check_no_duplicate:
      tmp = walls_0.copy()
      tmp[:,3] *= 0.9
      ious = dsiou_rotated_3d_bbox_np(tmp, tmp, iou_w=1, size_rate_thres=0.3, ref='union')
      np.fill_diagonal(ious, 0)
      if ious.max() > 0.2:
        i = np.argmax(ious.max(0))
        j = np.argmax(ious[i])
        wi = walls_0[i:i+1]
        wj = walls_0[j:j+1]
        iou_ij = ious[i,j]
        print(f'Found a duplicate with big iou: {iou_ij}')
        iou_ij_ck = dsiou_rotated_3d_bbox_np(wi, wj, iou_w=1, size_rate_thres=0.3, ref='union')
        #_show_objs_ls_points_ls( (512,512), objs_ls=[walls_0, wi, wj], obj_rep=obj_rep, obj_thickness=[1,3,3], obj_colors=['white','lime','red'] )
        #_show_3d_points_objs_ls( objs_ls=[ wi, wj], obj_rep=obj_rep, obj_colors=['green','red'] )
        pass

    iofs = dsiou_rotated_3d_bbox_np(walls_0, walls_0, iou_w=1, size_rate_thres=0.3, ref='min')
    np.fill_diagonal(iofs, 0)
    mask = iofs > 0.1
    ids = np.where( mask.any(0))[0]
    walls_1 = walls_0.copy()
    for i in ids:
      ids_i = np.where(mask[i])[0]
      for j in ids_i:
        iof_ij = iofs[i,j]
        iou_ij = ious[i,j]
        wi = walls_0[i:i+1]
        wj = walls_0[j:j+1]

        angle_dif = limit_period_np( walls_0[i,-1] - walls_0[j,-1], 0.5, np.pi )
        angle_dif = np.abs(angle_dif)
        if angle_dif > np.pi/4:
          continue

        si = scores_0[i]
        sj = scores_0[j]

        # merge i j
        if wi[0, 3] > wj[0,3]:
          long_w, short_w = wi, wj
          long_w_idx = i
        else:
          long_w, short_w = wj, wi
          long_w_idx = j

        line_long = OBJ_REPS_PARSE.encode_obj(long_w, obj_rep, 'RoLine2D_2p').reshape(1,2,2)
        short_in_long = points_in_lines(short_w[:,:2], line_long, 5).reshape(-1)
        if not short_in_long:
          continue

        if debug:
          print(f'iou: {iou_ij}, iof_ij: {iof_ij} \nangle_dif: {angle_dif}')
          print(f'si:{si}, sj:{sj}')
          _show_objs_ls_points_ls( (512,512), objs_ls=[walls_0, wi, wj], obj_rep=obj_rep, obj_thickness=[1,3,3], obj_colors=['white','lime','red'] )


        wall_long_new = crop_two_parallel_overlaip_wall(long_w, short_w, obj_rep)
        walls_1[long_w_idx] = wall_long_new

        mask[j,i] = False

        if debug:
          _show_objs_ls_points_ls( (512,512), objs_ls=[wall_long_new, walls_1], obj_rep=obj_rep, obj_thickness=[4,1], obj_colors=['white','red'] )
        pass

    return walls_1, scores_0

  @staticmethod
  def merge_parallel_connected_walls( walls, scores, obj_rep ):
    return walls, scores
  @staticmethod
  def rm_short_walls(walls_0, scores_0, obj_rep, min_out_length):
    mask = walls_0[:,3] > min_out_length
    if 1:
      rm_walls = walls_0[mask==False]
    ids = np.where(mask)[0]
    scores_1 = scores_0[mask] if scores_0 is not None else scores_0
    return walls_0[mask], scores_1, ids

  @staticmethod
  def merge_parallel_con_walls(walls, scores, obj_rep, show=0):
    from obj_geo_utils.line_operations import gen_corners_from_lines_np
    from tools.visual_utils import _show_objs_ls_points_ls, _show_3d_points_objs_ls
    show_parallel_con_candidates = show
    show_parallel_con_candidates = 0

    walls_merged = walls.copy()
    rm_ids = []

    assert obj_rep == 'XYZLgWsHA'
    n = walls.shape[0]
    uq_corners, _, corIds_per_line, num_cor_uq, cor_degrees =  gen_corners_from_lines_np( walls, None, obj_rep, 1, get_degree=1, flag='deg_d' )
    corners = uq_corners[ corIds_per_line ]
    for i in range(n-1):
      cor_ids_i = corIds_per_line[i]
      cor_deg_i = cor_degrees[cor_ids_i]
      if cor_deg_i.min() >=2:
        continue

      angle_dif_i = np.abs( angle_dif_by_period_np(walls[i:i+1,-1], walls[i+1:,-1], 0) )
      dis_i_0 = np.linalg.norm( corners[i:i+1,None] - corners[i+1:,:,None] , axis=-1)
      dis_i = dis_i_0.reshape(-1,4).min(1)
      mask_0 = angle_dif_i < 5 * np.pi / 180
      mask_1 = dis_i < 1
      mask_i = mask_0 * mask_1
      ids_i = np.where(mask_i)[0] + i + 1

      #if show and i in [26, 30]:
      #    show_parallel_con_candidates = 1
      #    _show_objs_ls_points_ls( (512,512), [ walls, walls[i:i+1] ], obj_rep, obj_colors=['white', 'red'], obj_thickness=[1,4], points_ls=[corners[i]], point_thickness=6 )
      #    import pdb; pdb.set_trace()  # XXX BREAKPOINT
      #    pass


      for j in ids_i:
        cor_id_j, cor_id_i = np.where(dis_i_0[j-i-1] < 1)
        cdg_i = cor_degrees[ corIds_per_line[i, cor_id_i]  ]
        cdg_j = cor_degrees[ corIds_per_line[j, cor_id_j]  ]
        if cor_id_j.shape[0]>1:
          import pdb; pdb.set_trace()  # XXX BREAKPOINT
          pass


        if show_parallel_con_candidates:
            print(f'cdg_i: {cdg_i}, cdg_j:{cdg_j}')
            cor_i = corners[i, cor_id_i]
            cor_j = corners[j, cor_id_j]
            _show_objs_ls_points_ls( (512,512), [ walls, walls[i:i+1], walls[j:j+1] ], obj_rep, obj_colors=['white', 'red', 'green'], obj_thickness=[1,2, 4], points_ls=[ cor_i, cor_j ], point_thickness=6 )
            import pdb; pdb.set_trace()  # XXX BREAKPOINT
            pass

        if cdg_i < 2 and cdg_j < 2:
          # all requirements matched
          cor_i = corners[i, 1-cor_id_i]
          cor_j = corners[j, 1-cor_id_j]
          merged_cors = np.concatenate([cor_i, cor_j], axis=0).reshape(1,4)
          merged_wall = OBJ_REPS_PARSE.encode_obj(merged_cors, 'RoLine2D_2p', obj_rep)
          walls_merged[i] = merged_wall
          walls_merged[j] = merged_wall

          kp_ids = [ii for ii in range(n) if ii!=i ]
          scores = scores[kp_ids]
          walls_merged = walls_merged[kp_ids]
          return GraphUtils.merge_parallel_con_walls( walls_merged, scores, obj_rep, show )
          #rm_ids.append(i)
          pass

    return walls, scores

    #kp_ids = [i for i in range(n) if i not in rm_ids]
    #walls_merged = walls_merged[kp_ids]
    #if scores is not None:
    #  scores = scores[kp_ids]
    #return walls_merged, scores

  @staticmethod
  def merge_wall_corners(walls_0, scores_0, obj_rep, min_cor_dis_thr):
    '''
    Do not  change the number of walls
    '''
    from tools.visual_utils import _show_objs_ls_points_ls, _show_3d_points_objs_ls, show_connectivity
    n = walls_0.shape[0]
    corners_0 = OBJ_REPS_PARSE.encode_obj(walls_0, obj_rep, 'RoLine2D_2p')
    corners_0 = corners_0.reshape(-1,2)

    if scores_0 is None:
      scores_cor_0 = scores_1 = None
    else:
      scores_cor_0 = np.repeat(scores_0.reshape(-1,1), 2, 1).reshape(-1)
    corners_1, scores_cor_1 = merge_corners_1_cls(corners_0, min_cor_dis_thr, scores_cor_0)
    walls_1 = OBJ_REPS_PARSE.encode_obj( corners_1.reshape(n,4), 'RoLine2D_2p', obj_rep )
    if scores_0 is not None:
      scores_1 = scores_cor_1.reshape(n,2).mean(1)
    if 0:
      #_show_objs_ls_points_ls((512,512), objs_ls=[walls_0, walls_1], obj_colors=['red','lime'], obj_rep=obj_rep, obj_thickness=[2,1])
      _show_objs_ls_points_ls((512,512), objs_ls=[walls_0], obj_colors=['random'], obj_rep=obj_rep, obj_thickness=1)
      _show_objs_ls_points_ls((512,512), objs_ls=[walls_1], obj_colors=['random'], obj_rep=obj_rep, obj_thickness=1)

    if 0:
      min_cor_dis_thr = 4
      #geo_mask0 = find_wall_wall_connection(walls_0, min_cor_dis_thr, obj_rep)
      #show_connectivity(walls_0, walls_0, geo_mask0, obj_rep)

      geo_mask1 = find_wall_wall_connection(walls_1, min_cor_dis_thr, obj_rep)
      show_connectivity(walls_1, walls_1, geo_mask1, obj_rep)
      pass
    return  walls_1, scores_1


  @staticmethod
  def optimize_walls_by_relation(walls0, relations, max_ofs_by_rel, obj_rep, eval_dir=None, scene_name=None):
    '''
    connect two walls with condition:
      1. semantic relation is 1 , but geometric relation is 0
      2. no other walls connect them
      3. offset is below max_ofs_by_rel
    '''
    from tools.visual_utils import show_connectivity, _show_objs_ls_points_ls
    from configs.common import DEBUG_CFG
    debug = DEBUG_CFG.SAVE_REL_OPT

    if debug:
      rel_dir = os.path.join(eval_dir, 'rel')
      if not os.path.exists(rel_dir):
        os.makedirs(rel_dir)
      img_file_base = os.path.join(rel_dir, scene_name)

    n = walls0.shape[0]
    assert relations.shape == (n,n)
    min_cor_dis_thr = 1
    geo_mask, geo_cor_degrees, corners0 = find_wall_wall_connection(walls0[:,:-1], min_cor_dis_thr, obj_rep)
    np.fill_diagonal(relations, 0)
    rel_mask = relations > 0.3

    if debug and 0:
      show_connectivity(walls0[:,:-1], walls0[:,:-1], geo_mask, obj_rep)
      #show_connectivity(walls0[:,:-1], walls0[:,:-1], relations, obj_rep)
      pass

    walls1 = walls0.copy()
    missed_mask = rel_mask * (geo_mask==False)
    extra_walls = []
    if missed_mask.any():
      #show_connectivity(walls0[:,:-1], walls0[:,:-1], missed_mask, obj_rep)
      ids = np.where(missed_mask)
      ids = np.array(ids).T.reshape(-1,2)
      ids = [i for i in ids if i[0]<i[1]]
      n = len(ids)
      for i in range(n):
        j, k = ids[i]
        con_dis = connectivity_distance_1_pair(geo_mask, j, k, 2)
        rel = relations[j,k]

        if debug:
          img_file = img_file_base + f'_{i}-0rel_{rel:.3f}-con_dis_{con_dis:.3f}.png'
          _show_objs_ls_points_ls( (512,512), objs_ls=[walls0[:,:-1], walls0[[j,k]][:,:-1]], obj_rep=obj_rep, obj_colors=['white', 'random'], obj_thickness=[1,3], out_file=img_file, only_save=1)

        if con_dis <= 1:
          continue
        rel_score = relations[j,k]
        walls1[j], walls1[k], extra_w, modified = GraphUtils.connect_two_walls(walls0[j], walls0[k], corners0[j], corners0[k], geo_cor_degrees[j], geo_cor_degrees[k], obj_rep, max_ofs_by_rel, rel_score, walls0)
        if extra_w is not None:
          extra_walls.append(extra_w)

        if debug:
          #print(f'\nmodified: {modified}')
          img_file = img_file_base + f'_{i}-Opt_by_rel.png'
          if extra_w is None:
            _show_objs_ls_points_ls( (512,512), objs_ls=[walls1[:,:-1], walls1[[j,k]][:,:-1]], obj_rep=obj_rep, obj_colors=['white', 'random'], obj_thickness=[1,3], out_file=img_file, only_save=1)
          else:
            _show_objs_ls_points_ls( (512,512), objs_ls=[walls1[:,:-1], walls1[[j,k]][:,:-1], extra_w[:,:-1]], obj_rep=obj_rep, obj_colors=['white', 'red', 'blue'], obj_thickness=[1,3,2], out_file=img_file, only_save=1)
          pass
      pass
    num_e = len(extra_walls)
    if num_e > 0:
          n = walls0.shape[0]
          extra_walls = np.concatenate(extra_walls, 0)
          walls1 = np.concatenate([walls1, extra_walls], 0)
    walls1 = walls1[ np.isnan(walls1[:,0])==False ]
    return walls1

  @staticmethod
  def connect_two_walls(wall0, wall1, cor0, cor1, cor_degree0, cor_degree1, obj_rep, max_ofs_by_rel, rel_score=None, all_walls=None):
    '''
    Conditions to connect:
      1. angle between wall0 and wall1 is below np.pi/8
      2. have an intersection
      3. the offset for connection is small

      a) modify the two walls
      b) crop one wall
    '''
    from obj_geo_utils.geometry_utils import line_intersection_2d, vertical_dis_points_lines
    from tools.visual_utils import _show_objs_ls_points_ls
    from obj_geo_utils.geometry_utils import points_in_lines
    assert wall0.shape == (8,)
    assert wall1.shape == (8,)
    debug = 0
    skip_parallel = 1

    cor0 = cor0.reshape(2,2)
    cor1 = cor1.reshape(2,2)
    cor_degree0 = cor_degree0.reshape(2)
    cor_degree1 = cor_degree1.reshape(2)

    # 1. angle between wall0 and wall1 is below np.pi/8
    angle_dif = np.abs( angle_dif_by_period_np(wall0[6], wall1[6], 0) )
    if angle_dif < np.pi/8:
      ver_diss = vertical_dis_points_lines(cor0, cor1[None]).mean()
      if debug and 1:
        print('connect by merging')
        _show_objs_ls_points_ls((512,512), objs_ls=[wall0.reshape(1,-1)[:,:-1], wall1.reshape(1,-1)[:,:-1], all_walls[:,:-1]],
                                obj_colors=['red','lime','white'], obj_rep=obj_rep, obj_thickness=[4,4,1])
      if ver_diss > 5:
        print(f'The angle is {angle_dif}, vertical distance: {ver_diss}, abandon')
        return wall0, wall1, None, False
      else:
        print(f'Skip mergeing parallel')
        return wall0, wall1, None, False
        wall0_new, wall1_new =  unused_merge_two_parallel_walls(wall0, wall1, cor0, cor1, obj_rep, cor_degree0, cor_degree1, all_walls=all_walls)
        return wall0_new, wall1_new, None, True


    # 2. have an intersection
    intsec = line_intersection_2d( cor0, cor1, min_angle=np.pi/8 ).reshape(1,2)
    if np.isnan(intsec).any():
      print(f'There is no intersection, do not connect')
      return wall0, wall1, None, False

    dis0 = np.linalg.norm( cor0 - intsec, axis=-1)
    dis1 = np.linalg.norm( cor1 - intsec, axis=-1)

    # 3. the offset for connection is small
    dis_ave = (dis0.min() + dis1.min())/2
    is_small_dis = dis_ave < max_ofs_by_rel
    if not is_small_dis:
      print(f'The connection offset is too large: {dis0.min()}, {dis1.min()}')
      return wall0, wall1, None, False

    def update_wall_by_intsec(wall, cor, ints, cor_deg):
        ints = np.repeat(ints, 2, 0)
        ws = np.concatenate([ints, cor], 1)
        walls_new = OBJ_REPS_PARSE.encode_obj(ws, 'RoLine2D_2p', obj_rep)
        scores = np.repeat(wall[None, -1:], 2, 0)
        walls_new = np.concatenate([walls_new, scores], 1)
        # a) the intsection is close to one corner, replace the corner
        if walls_new[:,3].min() < 10:
          j = walls_new[:,3].argmax()
          return  walls_new[j], None
        # b) intersection is on wall, crop the wall
        on_wall = points_in_lines( ints[0:1], cor[None], 5 ).reshape(-1)
        if on_wall:
          out0, out1 = walls_new[0], walls_new[1]
        else:
          dis = np.linalg.norm( cor - ints, axis=-1)
          close_i = np.argmin(dis)
          if cor_deg[close_i]==0:
            # c) degree is 0, extent
            out0, out1 = walls_new[1-close_i], None
          else:
            # d) add the extented
            out0, out1 = wall,  walls_new[close_i]
        if debug and 0:
          _show_objs_ls_points_ls((512,512), objs_ls=[wall.reshape(1,-1)[:,:-1], out0.reshape(1,-1)[:,:-1], out1.reshape(1,-1)[:,:-1], all_walls[:,:-1]],
                                  obj_colors=['yellow','red','lime','white'], obj_rep=obj_rep, obj_thickness=[8,4,4,1])
        return out0, out1

    wall0_new, extra_w0 =  update_wall_by_intsec(wall0, cor0, intsec, cor_degree0)
    wall1_new, extra_w1 =  update_wall_by_intsec(wall1, cor1, intsec, cor_degree1)
    if extra_w0 is None and extra_w1 is None:
      extra_w = None
    elif extra_w0 is None and extra_w1 is not None:
      extra_w = extra_w1
    elif extra_w0 is not None and extra_w1 is None:
      extra_w = extra_w0
    else:
      extra_w = np.concatenate([extra_w0[None,:], extra_w1[None,:]], 0)
    if extra_w is not None:
      extra_w = extra_w.reshape(-1,8)
    return wall0_new, wall1_new, extra_w, True


    @staticmethod
    def old_optimize_wall_graph(bboxes_in, scores=None, labels=None,
                      obj_rep='XYXYSin2',
                      opt_graph_cor_dis_thr=0, min_out_length=0):
      lines_in = OBJ_REPS_PARSE.encode_obj(bboxes_in, obj_rep, 'XYXYSin2')
      lines_merged, line_scores_merged, line_labels_merged, valid_inds_final = \
        GraphUtils.optimize_graph_lines( lines_in, scores, labels,
                                        opt_graph_cor_dis_thr, min_out_length)
      bboxes_merged = OBJ_REPS_PARSE.encode_obj(lines_merged, 'XYXYSin2', obj_rep)
      if obj_rep == 'XYXYSin2':
        pass
      elif obj_rep == 'XYZLgWsHA':
        bboxes_merged[:,[2,4,5]] = bboxes_in[valid_inds_final][:,[2,4,5]]
      return bboxes_merged, line_scores_merged, line_labels_merged, valid_inds_final

    def optimize_graph_lines(lines_in, scores=None, labels=None,
                      opt_graph_cor_dis_thr=0, min_out_length=0):
      '''
        lines_in: [n,5]
        Before optimization, all lines with length < opt_graph_cor_dis_thr are deleted.
        After optimization, all lines with length < min_out_length are deleted.
      '''
      from tools.visual_utils import _show_objs_ls_points_ls, _show_3d_points_objs_ls
      from obj_geo_utils.line_operations import gen_corners_from_lines_np
      assert opt_graph_cor_dis_thr>0
      num_in = lines_in.shape[0]

      # filter short lines
      line_length_in = np.linalg.norm(lines_in[:,2:4] - lines_in[:,:2], axis=1)
      valid_line_mask = line_length_in > opt_graph_cor_dis_thr
      valid_inds_0 = np.where(valid_line_mask)[0]
      del_lines = lines_in[valid_line_mask==False]
      lines_in = lines_in[valid_line_mask]
      if scores is not None:
        scores = scores[valid_line_mask]
      if labels is not None:
        labels = labels[valid_line_mask]

      num_line = lines_in.shape[0]
      if scores is None and labels is None:
        lab_sco_lines = None
      else:
        lab_sco_lines = np.concatenate([labels.reshape(num_line,1), scores.reshape(num_line,1)], axis=1)
      #corners_in, lab_sco_cors, corIds_per_line, num_cor_uq_org = \
      #      gen_corners_from_lines_np(lines_in, lab_sco_lines, 'XYXYSin2', min_cor_dis_thr=1)

      corners_in = OBJ_REPS_PARSE.encode_obj(lines_in, 'XYXYSin2', 'RoLine2D_2p').reshape(-1,2)

      if scores is None and labels is None:
        labels_cor = None
        scores_cor = None
        corners_labs = corners_in
      else:
        labels_cor = np.stack( [labels, labels] ).reshape(-1,1)
        scores_cor = np.stack( [scores, scores] ).reshape(-1,1)
        corners_labs = np.concatenate([corners_in, labels_cor.reshape(-1,1)*100], axis=1)

        #labels_cor = lab_sco_cors[:,0]
        #scores_cor = lab_sco_cors[:,1]
        #corners_labs = np.concatenate([corners_in, labels_cor.reshape(-1,1)*100], axis=1)
      corners_merged, cor_scores_merged = merge_close_corners(
              corners_in, opt_graph_cor_dis_thr, labels_cor, scores_cor )
      #corners_merged = round_positions(corners_merged, 1000)
      corners_merged_per_line = corners_merged.reshape(-1,4)
      lines_merged = OBJ_REPS_PARSE.encode_obj(corners_merged_per_line, 'RoLine2D_2p', 'XYXYSin2')

      # remove short lines
      line_length_out = np.linalg.norm(lines_merged[:,2:4] - lines_merged[:,:2], axis=1)
      valid_mask_1 = line_length_out > min_out_length
      valid_inds = np.where(valid_mask_1)[0]
      rm_num = line_length_out.shape[0] - valid_inds.shape[0]
      lines_merged = lines_merged[valid_inds]

      valid_inds_final = valid_inds_0[valid_inds]

      if scores is None and labels is None:
        line_labels_merged = None
        line_scores_merged = None
      else:
        line_scores_merged = cor_scores_merged.reshape(-1,2).mean(axis=1)[:,None]
        line_labels_merged = labels[valid_inds]
        line_scores_merged = line_scores_merged[valid_inds]

      if valid_inds_final.shape[0] != lines_merged.shape[0]:
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        pass


      debug = 0
      if debug:
        corners_uq, unique_indices, inds_inverse = np.unique(corners_merged, axis=0, return_index=True, return_inverse=True)
        num_cor_org = corners_in.shape[0]
        num_cor_merged = corners_uq.shape[0]
        deleted_inds = [i for i in range(num_cor_org) if i not in unique_indices]
        #deleted_corners = corners_in[deleted_inds]
        deleted_corners = corners_merged[deleted_inds]

        dn = del_lines.shape[0]
        length_deled = line_length_in[ valid_line_mask==False ]
        print(f'\n\n\tcorner num: {num_cor_org} -> {num_cor_merged}\n')
        print(f'deleted input lines: {dn}')
        print(f'del length: {length_deled}')

        show = 1

        if show:
          data = ['2d', '3d'][0]
          if data=='2d':
            w, h = np.ceil(corners_in.max(0)+50).astype(np.int32)
            if rm_num > 0:
              _show_objs_ls_points_ls( (h,w), [lines_in, lines_in[line_merging_del_inds]], obj_colors=['green', 'red'], obj_thickness=[3,2],)

            import pdb; pdb.set_trace()  # XXX BREAKPOINT
            _show_objs_ls_points_ls( (h,w), [lines_in, lines_merged], obj_colors=['green', 'red'], obj_thickness=[3,2],
                        points_ls=[corners_in, corners_merged], point_colors=['green', 'red'], point_thickness=[3,2] )
          else:
            if dn>0:
              _show_3d_points_objs_ls(
                objs_ls = [lines_in, del_lines], obj_rep='XYXYSin2',
                obj_colors=['red','blue'], thickness=5,)

            print('\nCompare org and merged')
            _show_3d_points_objs_ls( points_ls=[deleted_corners],
              objs_ls = [lines_in, lines_merged], obj_rep='XYXYSin2',
              obj_colors=['blue', 'red'], thickness=[3,2],)

            print('\nMerged result')
            _show_3d_points_objs_ls( points_ls=[deleted_corners],
              objs_ls = [lines_merged], obj_rep='XYXYSin2',
              obj_colors='random', thickness=5,)

          pass

      return lines_merged, line_scores_merged, line_labels_merged, valid_inds_final

    @staticmethod
    def Unused_gen_corners_from_lines_np(lines, labels=None, obj_rep='XYXYSin2', flag=''):
        '''
        lines: [n,5]
        labels: [n,1/2]: 1 for label only, 2 for label and score

        corners: [m,2]
        labels_cor: [m, 1/2]
        corIds_per_line: [n,2]
        num_cor_uq: m
        '''
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        if lines.shape[0] == 0:
          if labels is None:
            labels_cor = None
          else:
            labels_cor = np.zeros([0,labels.shape[1]])
          return np.zeros([0,2]), labels_cor, np.zeros([0,2], dtype=np.int), 0

        lines0 = OBJ_REPS_PARSE.encode_obj(lines, obj_rep, 'RoLine2D_2p')
        if labels is not None:
          num_line = lines.shape[0]
          assert labels.shape[0] == num_line
          labels = labels.reshape(num_line, -1)
          lc = labels.shape[1]

          labels_1 = labels.reshape(-1,lc)
          lines1 = np.concatenate([lines0[:,0:2], labels_1, lines0[:,2:4], labels_1], axis=1)
          corners1 = lines1.reshape(-1,2+lc)
        else:
          corners1 = lines0.reshape(-1,2)
        #corners1 = round_positions(corners1, 1000)
        corners_uq, unique_indices, inds_inverse = np.unique(corners1, axis=0, return_index=True, return_inverse=True)
        num_cor_uq = corners_uq.shape[0]
        corners = corners_uq[:,:2]
        if labels is not None:
          labels_cor = corners_uq[:,2:].astype(labels.dtype)
        else:
          labels_cor = None
        corIds_per_line = inds_inverse.reshape(-1,2)

        lineIds_per_cor = get_lineIdsPerCor_from_corIdsPerLine(corIds_per_line, corners.shape[0])

        if flag=='debug':
          print('\n\n')
          print(lines[0:5])
          n0 = lines.shape[0]
          n1 = corners.shape[0]
          print(f'\n{n0} lines -> {n1} corners')
          _show_lines_ls_points_ls((512,512), [lines], [corners], 'random', 'random')
          #for i in range(corners.shape[0]):
          #  lids = lineIds_per_cor[i]
          #  _show_lines_ls_points_ls((512,512), [lines, lines[lids].reshape(-1, lines.shape[1])], [corners[i:i+1]], ['white', 'green'], ['red'], point_thickness=2)
          #for i in range(lines.shape[0]):
          #  cor_ids = corIds_per_line[i]
          #  _show_lines_ls_points_ls((512,512), [lines, lines[i:i+1]], [corners[cor_ids]], ['white', 'green'], ['red'], point_thickness=2)
          pass

        return corners, labels_cor, corIds_per_line, num_cor_uq

    @staticmethod
    def opti_wall_manually(lines, obj_rep, manual_merge_pairs):
      from tools.visual_utils import _show_objs_ls_points_ls, _show_3d_points_objs_ls, _show_3d_bboxes_ids
      #show_free_corners(lines, obj_rep)
      #_show_3d_bboxes_ids(lines, obj_rep)
      new_lines = merge_lines_intersect( lines, obj_rep, manual_merge_pairs )
      #show_free_corners(new_lines, obj_rep)
      return new_lines




def merge_2_parallel_walls(wall0, wall1, obj_rep):
  pass

def merge_two_parallel_walls(wall0, wall1, cor0, cor1, obj_rep, cor_degree0, cor_degree1, flag=0, all_walls=None):
  '''
  One wall must have a free corner
  Only change one corner of the four.
  If the two merged corners are both degree 0, merge as one instance.
  '''
  from tools.visual_utils import _show_objs_ls_points_ls, _show_3d_points_objs_ls, _show_3d_bboxes_ids
  assert flag <2
  if (cor_degree0==0).any():
    i = np.where(cor_degree0==0) [0].reshape(-1)[0]
    j = np.linalg.norm( cor0[i][None] - cor1, axis=-1).argmin()
    is_merge = cor_degree1[j] == 0
    cor0_new = cor0.copy()
    if is_merge:
      cor0_new[i] = cor1[1-j]
      wall1_new = None
    else:
      cor0_new[i] = cor1[j]
      cor1_new = cor1
      wall1_new = OBJ_REPS_PARSE.encode_obj( cor1_new.reshape(1,4), 'RoLine2D_2p', obj_rep ).reshape(-1)
      wall1_new = np.concatenate([wall1_new, wall1[-1:]])
    wall0_new = OBJ_REPS_PARSE.encode_obj( cor0_new.reshape(1,4), 'RoLine2D_2p', obj_rep ).reshape(-1)
    wall0_new = np.concatenate([wall0_new, wall0[-1:]])

    _show_objs_ls_points_ls((512,512), objs_ls=[wall0.reshape(1,-1)[:,:-1], wall1.reshape(1,-1)[:,:-1], all_walls[:,:-1]],
                                obj_colors=['red','lime','white'], obj_rep=obj_rep, obj_thickness=[4,4,1])
    if wall1_new is not None:
      _show_objs_ls_points_ls((512,512), objs_ls=[wall0_new.reshape(1,-1)[:,:-1], wall1_new.reshape(1,-1)[:,:-1], all_walls[:,:-1]],
                                obj_colors=['red','lime','white'], obj_rep=obj_rep, obj_thickness=[4,4,1])
    else:
      _show_objs_ls_points_ls((512,512), objs_ls=[wall0_new.reshape(1,-1)[:,:-1],  all_walls[:,:-1]],
                                obj_colors=['red','white'], obj_rep=obj_rep, obj_thickness=[4,1])
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    return wall0_new, wall1_new
  else:
    assert flag == 0
    flag += 1
    wall0_new, wall1_new = merge_two_parallel_walls(wall1, wall0, cor1, cor0, obj_rep, cor_degree1, cor_degree0, flag, all_walls)
    return wall0_new, wall1_new

def crop_two_intersect_overlaip_wall(w_croped, w_fix, obj_rep):
  assert w_croped.ndim == w_fix.ndim == 1
  cor_c = OBJ_REPS_PARSE.encode_obj(w_croped[None,:], obj_rep, 'RoLine2D_2p').reshape(2,2)
  cor_f = OBJ_REPS_PARSE.encode_obj(w_fix[None,:], obj_rep, 'RoLine2D_2p').reshape(2,2)
  intersect = line_intersection_2d(cor_c, cor_f, min_angle=np.pi/8).reshape(1,2)
  cor_a = np.concatenate([cor_c[0:1], intersect], 0).reshape(1,4)
  cor_b = np.concatenate([cor_c[1:2], intersect], 0).reshape(1,4)
  w_a = OBJ_REPS_PARSE.encode_obj(cor_a, 'RoLine2D_2p', obj_rep)[0]
  w_b = OBJ_REPS_PARSE.encode_obj(cor_b, 'RoLine2D_2p', obj_rep)[0]
  return w_a, w_b


def crop_two_parallel_overlaip_wall(long_w, short_w, obj_rep):
  corners_l = OBJ_REPS_PARSE.encode_obj(long_w, obj_rep, 'RoLine2D_2p').reshape(2,2)
  corners_s = OBJ_REPS_PARSE.encode_obj(short_w, obj_rep, 'RoLine2D_2p').reshape(2,2)
  dis = corners_s[:,None,:] - corners_l[None,:,:]
  dis = np.linalg.norm(dis, axis=-1)
  # idx_s is the idx of corners in short wall, that should crop long wall
  idx_s = np.argmax( dis.min(1) )
  idx_l = np.argmax( dis[1-idx_s] )
  cor_new_long = np.concatenate( [ corners_l[idx_l][None], corners_s[idx_s][None] ], 0 ).reshape(1,4)
  new_long_w = OBJ_REPS_PARSE.encode_obj(cor_new_long, 'RoLine2D_2p', obj_rep)

  if 0:
      from tools.visual_utils import _show_objs_ls_points_ls, _show_3d_points_objs_ls
      _show_objs_ls_points_ls( (512,512), objs_ls=[long_w, short_w, new_long_w], obj_rep=obj_rep, obj_thickness=[8, 4, 1], obj_colors=['white','lime','red'] )
  return new_long_w

def round_positions(data, scale=1000):
  return np.round(data*scale)/scale

def get_lineIdsPerCor_from_corIdsPerLine(corIds_per_line, num_corner):
  '''
  corIds_per_line: [num_line, 2]
  '''
  num_line = corIds_per_line.shape[0]
  lineIds_per_cor = [ None ] * num_corner
  for i in range(num_line):
    cj0, cj1 = corIds_per_line[i]
    if lineIds_per_cor[cj0] is None:
      lineIds_per_cor[cj0] = []
    if lineIds_per_cor[cj1] is None:
      lineIds_per_cor[cj1] = []
    lineIds_per_cor[cj0].append(i)
    lineIds_per_cor[cj1].append(i)
  #for i in range(num_corner):
  #  lineIds_per_cor[i] = np.array(lineIds_per_cor[i])
  return lineIds_per_cor

def merge_close_corners(corners_0, min_cor_dis_thr, labels_0=None, scores_0=None):
  if labels_0 is None:
    return merge_corners_1_cls(corners_0, min_cor_dis_thr, scores_0)
  else:
    assert corners_0.shape[0] == labels_0.shape[0]
    corners_1 = corners_0.copy()
    if scores_0 is not None:
      scores_1 = scores_0.copy()
    else:
      scores_1 = scores_0
    l0 = labels_0.min()
    l1 = labels_0.max()
    for l in range(l0,l1+1):
      mask = (labels_0 == l).reshape(-1)
      if scores_0 is not None:
        s0 = scores_0[mask]
      else:
        s0 = None
      c1, s1 = merge_corners_1_cls(corners_0[mask], min_cor_dis_thr, s0)
      #c1, s1 = merge_corners_1_cls(c1, min_cor_dis_thr, s1)
      corners_1[mask] = c1
      if scores_0 is not None:
        scores_1[mask] = s1
    return corners_1, scores_1

def merge_corners_1_cls(corners_0, min_cor_dis_thr, scores_0=None):
  '''
  In:
    corners_0: [n,2] [x,y]
    scores_0:[n]
  Out:
    corners_merged: [n,2]
    scores_merged: [n]
  '''
  assert corners_0.ndim == 2
  nc, nd = corners_0.shape
  assert nd == 2
  if scores_0 is not None:
    assert scores_0.shape[0] == nc
    scores_1 = scores_0.copy()
  else:
    scores_1 = scores_0

  corners_0 = corners_0.copy()
  for j in range(2):
    diss = corners_0[None,:,:] - corners_0[:,None,:]
    diss = np.linalg.norm(diss, axis=2)
    mask = diss < min_cor_dis_thr
    is_any_duplicate = mask.sum() > nc
    merging_ids = []
    for i in range(nc):
      ids_i = np.where(mask[i])[0]
      merging_ids.append(ids_i)
      if scores_0 is not None:
        weights = scores_0[ids_i] / scores_0[ids_i].sum()
        merged_sco = ( scores_0[ids_i] * weights).sum()
        scores_1[ids_i] = merged_sco
        merged_cor = ( corners_0[ids_i] * weights[:,None]).sum(axis=0)
      else:
        merged_cor = ( corners_0[ids_i] ).mean(axis=0)
      corners_0[ids_i] = merged_cor
      pass
  return corners_0, scores_1

def merge_lines_intersect(lines, obj_rep, pairs):
    assert obj_rep == 'XYXYSin2'
    lines = lines.copy()
    n = lines.shape[0]
    line_corners = OBJ_REPS_PARSE.encode_obj(lines, obj_rep, 'RoLine2D_2p').reshape(n,2,2)
    for i in range( len(pairs) ):
      j,k = pairs[i]
      inter_sec = line_intersection_2d( lines[j,:4].reshape(2,2), lines[k,:4].reshape(2,2) )
      replace_close_corner(line_corners[j], inter_sec)
      replace_close_corner(line_corners[k], inter_sec)

    new_lines = OBJ_REPS_PARSE.encode_obj(line_corners.reshape(n,4), 'RoLine2D_2p', obj_rep)
    return new_lines

def replace_close_corner(line_corners, point):
  assert line_corners.shape == (2,2)
  dis = np.linalg.norm( line_corners - point[None,:], axis=1)
  i = dis.argmin()
  line_corners[i] = point

def show_free_corners(bboxes, obj_rep, cor_connect_thre=0.15):
    from tools.visual_utils import _show_objs_ls_points_ls, _show_3d_points_objs_ls, _show_3d_bboxes_ids
    assert obj_rep == 'XYXYSin2'
    lines = OBJ_REPS_PARSE.encode_obj(bboxes, obj_rep, 'XYXYSin2')
    lines = lines.copy()
    n = lines.shape[0]
    corners = OBJ_REPS_PARSE.encode_obj(lines, obj_rep, 'RoLine2D_2p').reshape(-1,2)

    cor_diss = np.linalg.norm( corners[:,None,:] - corners[None,:,:], axis=-1 )
    np.fill_diagonal(cor_diss, 100)
    for i in range(n*2):
      if i%2==0:
        j = i+1
      else:
        j = i-1
      cor_diss[i,j] = 100

    mask = cor_diss < cor_connect_thre
    cor_degrees = mask.sum(axis=1)
    corners_new = corners.copy()

    free_mask = cor_degrees == 0
    cor_diss[:,free_mask] = 200

    free_corners = corners[free_mask]
    _show_3d_points_objs_ls( [free_corners], objs_ls=[lines], obj_rep='XYXYSin2' )

def find_wall_wd_connection(walls, windows, obj_rep):
  from obj_geo_utils.geometry_utils import  points_in_lines
  from tools.visual_utils import _show_objs_ls_points_ls, _show_3d_points_objs_ls, _show_3d_bboxes_ids
  #windows = windows[3:4]
  #walls = walls[-1:]
  #_show_objs_ls_points_ls((512,512), [walls, windows], obj_rep, obj_colors=['red', 'green'])
  nw = windows.shape[0]
  nwa = walls.shape[0]
  if nw==0 or nwa==0:
    return np.zeros([nw,nwa])==1
  walls_2p = OBJ_REPS_PARSE.encode_obj( walls, obj_rep, 'RoLine2D_2p' ).reshape(-1,2,2)
  window_centroids = OBJ_REPS_PARSE.encode_obj( windows, obj_rep, 'RoLine2D_2p' ).reshape(-1,2,2).mean(axis=1)
  win_in_wall_mask = points_in_lines(window_centroids, walls_2p, threshold_dis=10, one_point_in_max_1_line=True)
  win_ids, wall_ids_per_win = np.where(win_in_wall_mask)

  check_all_windows_mapped = 0
  if check_all_windows_mapped:
    if not (np.all(win_ids == np.arange(nw)) and win_ids.shape[0] == nw):
      print(f'win_ids: {win_ids}, nw={nw}')
      missed_win_ids = [i for i in range(windows.shape[0]) if i not in win_ids]
      _show_objs_ls_points_ls((512,512), [walls, windows[missed_win_ids] ], obj_rep, obj_colors=['white', 'green'])
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass
  if 0:
    for i,j in zip(win_ids, wall_ids_per_win):
      _show_objs_ls_points_ls((512,512), [walls, walls[j:j+1], windows[i:i+1] ], obj_rep, [window_centroids[i:i+1]], obj_colors=['white', 'green', 'red'])
      pass
  return win_in_wall_mask


def get_cor_degree(walls, connect_threshold, obj_rep):
  from tools.visual_utils import _show_objs_ls_points_ls, _show_3d_points_objs_ls
  n = walls.shape[0]
  cors = OBJ_REPS_PARSE.encode_obj(walls, obj_rep, 'RoLine2D_2p').reshape(-1,2)
  dis = np.linalg.norm( cors[:,None,:] - cors[None,:,:], axis=-1 )
  mask = dis < connect_threshold
  degrees = mask.sum(1).reshape(n,2) - 1
  assert degrees.min() >= 0
  cors = cors.reshape(-1,2,2)

  if 0:
    scores = degrees[:,0]*10 + degrees[:,1]
    _show_objs_ls_points_ls((512,512), objs_ls=[walls], obj_rep=obj_rep, obj_scores_ls=[scores] )
  return degrees, cors

def find_wall_wall_connection(bboxes, connect_threshold, obj_rep):
      corners_per_line = OBJ_REPS_PARSE.encode_obj(bboxes, obj_rep, 'RoLine2D_2p')
      n = bboxes.shape[0]
      corners_per_line = corners_per_line.reshape(n*2,2)
      # note: the order of two corners is not consistant
      corners_dif0 = corners_per_line[:,None,:] - corners_per_line[None,:,:]
      corners_dif1 = np.linalg.norm( corners_dif0, axis=-1 )
      corner_con_mask0 = corners_dif1 < connect_threshold
      np.fill_diagonal(corner_con_mask0, False)
      corner_degrees = corner_con_mask0.sum(1).reshape(n,2)
      wall_con_mask = corner_con_mask0.reshape(n,2,n,2).any(1).any(-1)
      return wall_con_mask, corner_degrees, corners_per_line.reshape(n,2,2)

      #xinds, yinds = np.where(connect_mask)
      #connection = np.concatenate([xinds[:,None], yinds[:,None]], axis=1).astype(np.uint8)
      #relations = []
      #for i in range(n):
      #  relations.append([])
      #for i in range(connection.shape[0]):
      #  x,y = connection[i]
      #  relations[x].append(y)
      #return connect_mask

def connectivity_distance_1_pair(relations, x, y, max_search=3):
  np.fill_diagonal(relations, 0)
  mask = relations > 0.5
  n = relations.shape[0]
  s = min(n, max_search)
  starts = [x]
  for i in range(s):
    ids_i = []
    #print(f'starts: {starts}')
    for j in starts:
      ids_j = np.where(mask[j])[0].tolist()
      ids_i += ids_j
    starts = ids_i
    if y in ids_i:
      return i
  return 1000

class OBJ_REPS_PARSE_TORCH():
  import torch

  @staticmethod
  def XYLgWsAsinSin2Z0Z1_TO_XYZLgWsHA(bboxes_ass2):
    n = bboxes_ass2.shape[0]
    bboxes_csa = torch.zeros_like(bboxes_ass2)[:,:7]
    bboxes_csa[:, [0,1]] = bboxes_ass2[:, [0,1]]
    bboxes_csa[:, 2] = bboxes_ass2[:, [6,7]].mean(axis=1)
    bboxes_csa[:, [3,4]] = bboxes_ass2[:, [2,3]]
    bboxes_csa[:, 5] = bboxes_ass2[:, 7] - bboxes_ass2[:, 6]
    abssin = bboxes_ass2[:, 4]
    sin2theta = bboxes_ass2[:, 5]
    theta_0 = torch.asin(abssin)
    theta_1 = -theta_0
    flag = (sin2theta >= 0).to(torch.int)
    theta = theta_0 * flag + theta_1 * (1-flag)
    bboxes_csa[:, 6] = theta
    return bboxes_csa

  @staticmethod
  def XYXYSin2WZ0Z1_TO_XYZLgWsHA(bboxes):
    lines_2p = OBJ_REPS_PARSE_TORCH.XYXYSin2_TO_RoLine2D_2p(bboxes)
    bboxes_CLA = OBJ_REPS_PARSE_TORCH.RoLine2D_2p_TO_CenterLengthAngle(lines_2p)
    zero_1 = torch.zeros_like(bboxes_xyxysin2[:,0:1])
    XYZLgWsHA = torch.cat([ bboxes_CLA[:,[0,1]], zero_1, bboxes_CLA[:,[2]], zero_1, zero_1, bboxes_CLA[:,[3]] ], dim=1)
    return XYZLgWsHA

  @staticmethod
  def XYXYSin2_TO_XYZLgWsHA(bboxes_xyxysin2):
    lines_2p = OBJ_REPS_PARSE_TORCH.XYXYSin2_TO_RoLine2D_2p(bboxes_xyxysin2)
    bboxes_CLA = OBJ_REPS_PARSE_TORCH.RoLine2D_2p_TO_CenterLengthAngle(lines_2p)
    zero_1 = torch.zeros_like(bboxes_xyxysin2[:,0:1])
    XYZLgWsHA = torch.cat([ bboxes_CLA[:,[0,1]], zero_1, bboxes_CLA[:,[2]], zero_1, zero_1, bboxes_CLA[:,[3]] ], dim=1)
    return XYZLgWsHA

  @staticmethod
  def RoLine2D_2p_TO_CenterLengthAngle(bboxes):
    assert bboxes.shape[1] == 4
    corner0 = bboxes[:,0:2]
    corner1 = bboxes[:,2:4]
    center = bboxes.reshape(-1,2,2).mean(axis=1)
    vec = corner1 - corner0
    length = vec.norm(dim=1)[:,None]
    angle = angle_with_x(vec, scope_id=2)[:,None]
    bboxes_CLA = torch.cat([center, length, angle], axis=1)
    return bboxes_CLA

  def XYXYSin2_TO_RoLine2D_2p(lines):
    '''
    From XYXYSin2 to RoLine2D_2p
    '''
    istopleft = (lines[:,4:5] >= 0).to(lines.dtype)
    lines_2p = lines[:,:4] * istopleft +  lines[:,[0,3,2,1]] * (1-istopleft)
    return lines_2p


def test_2d():
  from tools.visual_utils import _show_objs_ls_points_ls, _show_3d_points_objs_ls
  u = np.pi/180
  XYZLgWsHA = np.array([
    [200, 200, 0, 200, 30, 0, 45*u ],
  ])


  XYXYSin2WZ0Z1 = OBJ_REPS_PARSE.encode_obj(XYZLgWsHA, 'XYZLgWsHA', 'XYXYSin2WZ0Z1')
  XYDAsinAsinSin2Z0Z1 = OBJ_REPS_PARSE.encode_obj(XYXYSin2WZ0Z1, 'XYXYSin2WZ0Z1', 'XYDAsinAsinSin2Z0Z1')
  XYXYSin2WZ0Z1_c = OBJ_REPS_PARSE.encode_obj(XYDAsinAsinSin2Z0Z1, 'XYDAsinAsinSin2Z0Z1', 'XYXYSin2WZ0Z1')
  XYLgWsAsinSin2Z0Z1 = OBJ_REPS_PARSE.encode_obj(XYDAsinAsinSin2Z0Z1, 'XYDAsinAsinSin2Z0Z1', 'XYLgWsAsinSin2Z0Z1')
  XYLgWsA = OBJ_REPS_PARSE.encode_obj(XYDAsinAsinSin2Z0Z1, 'XYDAsinAsinSin2Z0Z1', 'XYLgWsA')

  err = XYXYSin2WZ0Z1 - XYXYSin2WZ0Z1_c
  err = np.max(np.abs(err))
  assert err < 1e-5
  print(XYXYSin2WZ0Z1)
  print(XYXYSin2WZ0Z1_c)
  print(XYDAsinAsinSin2Z0Z1)
  print(XYLgWsAsinSin2Z0Z1)
  print(XYLgWsA)


  _show_objs_ls_points_ls( (512,512), [XYZLgWsHA], 'XYZLgWsHA' )
  _show_objs_ls_points_ls( (512,512), [XYXYSin2WZ0Z1], 'XYXYSin2WZ0Z1' )
  _show_objs_ls_points_ls( (512,512), [XYDAsinAsinSin2Z0Z1], 'XYDAsinAsinSin2Z0Z1' )
  _show_objs_ls_points_ls((512,512), [XYLgWsAsinSin2Z0Z1], 'XYLgWsAsinSin2Z0Z1')

def test_2d_XYDAsinAsinSin2Z0Z1():
  from tools.visual_utils import _show_objs_ls_points_ls, _show_3d_points_objs_ls
  u = np.pi/180
  XYZLgWsHA = np.array([
    [200, 200, 0, 200, 30, 0, 45*u ],
  ])


  XYDAsinAsinSin2Z0Z1 = OBJ_REPS_PARSE.encode_obj(XYZLgWsHA, 'XYZLgWsHA', 'XYDAsinAsinSin2Z0Z1')
  XYZLgWsHA_c = OBJ_REPS_PARSE.encode_obj( XYDAsinAsinSin2Z0Z1, 'XYDAsinAsinSin2Z0Z1', 'XYZLgWsHA')
  corners = OBJ_REPS_PARSE.encode_obj(XYDAsinAsinSin2Z0Z1, 'XYDAsinAsinSin2Z0Z1', 'RoLine2D_2p').reshape(-1,2)

  err = XYZLgWsHA_c - XYZLgWsHA
  print('err', err)
  err = np.max(np.abs(err))
  assert err < 1e-5
  print('\nXYZLgWsHA', XYZLgWsHA )
  print('\nXYZLgWsHA_c', XYZLgWsHA_c)
  print('\nXYDAsinAsinSin2Z0Z1', XYDAsinAsinSin2Z0Z1)

  _show_objs_ls_points_ls( (512,512), [XYZLgWsHA], 'XYZLgWsHA' )
  _show_objs_ls_points_ls( (512,512), [XYDAsinAsinSin2Z0Z1], 'XYDAsinAsinSin2Z0Z1', points_ls=[corners])

def test_Rect4CornersZ0Z1():
  from obj_geo_utils.line_operations import rotate_bboxes_img
  from tools.visual_utils import _show_objs_ls_points_ls, _show_3d_points_objs_ls
  u = np.pi/180
  XYZLgWsHA = np.array([
    [200, 300, 0, 100, 0, 0, -45*u ],
    [200, 200, 0, 200, 30, 0, 45*u ],
    [300, 300, 0, 200, 200, 0, 80*u ],
  ])
  n = XYZLgWsHA.shape[0]
  Rect4CornersZ0Z1 = OBJ_REPS_PARSE.encode_obj(XYZLgWsHA, 'XYZLgWsHA', 'Rect4CornersZ0Z1')
  corners4 = Rect4CornersZ0Z1[:,:8].reshape(n,4,2)

  XYZLgWsHA_c = OBJ_REPS_PARSE.encode_obj( Rect4CornersZ0Z1, 'Rect4CornersZ0Z1', 'XYZLgWsHA' )
  err = XYZLgWsHA_c - XYZLgWsHA
  merr = np.max(np.abs(err))
  print(f'err: {merr}\n{err}')
  if not merr < 1e-5:
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    pass
  pass

  pids = np.repeat( np.arange(4).reshape([1,4]), n, 0 ).reshape(-1)
  _show_objs_ls_points_ls( (512,512), [Rect4CornersZ0Z1], 'Rect4CornersZ0Z1', points_ls=[corners4.reshape(-1, 2)], point_colors=['green', 'red'], point_scores_ls=[pids])

  img = np.zeros([512,512,4], dtype=np.uint8)
  for i in range(5):
    angle = np.random.rand() * 180
    print(f'angle: {angle}')
    bboxes_r, img_r = rotate_bboxes_img( Rect4CornersZ0Z1, img, angle, 'Rect4CornersZ0Z1' )
    corners4 = bboxes_r[:,:8].reshape(n,4,2)
    _show_objs_ls_points_ls( img_r[:,:,0:3], [bboxes_r], 'Rect4CornersZ0Z1', points_ls=[corners4.reshape(-1, 2)], point_colors=['green', 'red'], point_scores_ls=[pids])
  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  pass

def test_3d():
  from tools.visual_utils import _show_3d_points_objs_ls

  XYZLgWsHA = np.array([
                        [ 0,1,0, 5, 1, 0.5, np.pi/4 ],
                        [ 0,1,0, 5, 1, 0.5, 0 ],
                        ])
  XYLgWsSin2Sin4Z0Z1 = OBJ_REPS_PARSE.encode_obj(XYZLgWsHA, 'XYZLgWsHA', 'XYLgWsSin2Sin4Z0Z1')
  _XYZLgWsHA = OBJ_REPS_PARSE.encode_obj(XYLgWsSin2Sin4Z0Z1, 'XYLgWsSin2Sin4Z0Z1', 'XYZLgWsHA')
  err0 = XYZLgWsHA - _XYZLgWsHA
  err = np.abs(err0).max()
  print(f'\nerr: {err}')
  print(f'XYZLgWsHA: \n{XYZLgWsHA}')
  print(f'_XYZLgWsHA: \n{_XYZLgWsHA}')
  assert err < 1e-5
  _show_3d_points_objs_ls( objs_ls = [XYZLgWsHA], obj_rep='XYZLgWsHA', obj_colors='random' )
  _show_3d_points_objs_ls( objs_ls = [XYLgWsSin2Sin4Z0Z1], obj_rep='XYLgWsSin2Sin4Z0Z1', obj_colors='random'  )
  _show_3d_points_objs_ls( objs_ls = [XYZLgWsHA, _XYZLgWsHA], obj_rep='XYZLgWsHA' , obj_colors='random' )


if __name__ == '__main__':
  #test_2d_XYDAsinAsinSin2Z0Z1()
  test_Rect4CornersZ0Z1()
  pass


