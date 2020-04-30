# xyz 17 Apr
import numpy as np
import mmcv
from obj_geo_utils.geometry_utils import sin2theta_np, angle_with_x_np, \
      vec_from_angle_with_x_np, angle_with_x
import cv2
import torch
from obj_geo_utils.geometry_utils import limit_period_np


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
  def UpRight_xyxy_sin2a_TO_XYZLgWsHA(bboxes_xyxysin2):
    lines_2p = OBJ_REPS_PARSE_TORCH.UpRight_xyxy_sin2a_TO_RoLine2D_2p(bboxes_xyxysin2)
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
    # Because axis-y of img points to bottom for img, it is positive for
    # anti-clock wise. Change to positive for clock-wise
    angle = -angle
    bboxes_CLA = torch.cat([center, length, angle], axis=1)
    return bboxes_CLA

  def UpRight_xyxy_sin2a_TO_RoLine2D_2p(lines):
    '''
    From RoLine2D_UpRight_xyxy_sin2a to RoLine2D_2p
    '''
    istopleft = (lines[:,4:5] >= 0).to(lines.dtype)
    lines_2p = lines[:,:4] * istopleft +  lines[:,[0,3,2,1]] * (1-istopleft)
    return lines_2p

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

  (*) RoLine2D_UpRight_xyxy_sin2a
    x denotes the long axis.
    angle: [-90, 90)

    Convert:
      RoLine2D_2p <-> XYLgWsA
      RoLine2D_UpRight_xyxy_sin2a <-> XYLgWsA
      XYZLgWsHA <-> XYLgWsA
  '''
  _obj_dims = {
    'XYZLgWsHA': 7,
    'XYLgWsA': 5,

    'RoLine2D_UpRight_xyxy_sin2a': 5,
    'RoLine2D_2p': 4,
  }
  _obj_reps = _obj_dims.keys()

  @staticmethod
  def check_obj_dim(bboxes, obj_rep):
    assert bboxes.ndim == 2
    s = OBJ_REPS_PARSE._obj_dims[obj_rep]
    assert bboxes.shape[1] == s, \
      f'obj_rep={obj_rep}, input shape={bboxes.shape[1]}, correct shape={s}'

  @staticmethod
  def encode_obj(bboxes, obj_rep_in, obj_rep_out, check_sin2=1):
    '''
    '''
    assert obj_rep_in  in OBJ_REPS_PARSE._obj_reps, obj_rep_in
    assert obj_rep_out in OBJ_REPS_PARSE._obj_reps, obj_rep_out
    OBJ_REPS_PARSE.check_obj_dim(bboxes, obj_rep_in)

    bboxes = OBJ_REPS_PARSE.make_x_long_dim(bboxes, obj_rep_in)
    nbox = bboxes.shape[0]

    if obj_rep_in == obj_rep_out:
      return bboxes

    if obj_rep_in == 'XYZLgWsHA'  and obj_rep_out == 'XYLgWsA':
      return bboxes[:,[0,1,3,4,6]]

    if obj_rep_in == 'XYLgWsA'  and obj_rep_out == 'XYZLgWsHA':
      ze = np.zeros([nbox, 1])
      return np.concatenate([bboxes[:,[0,1]], ze, bboxes[:,[2,3]], ze, bboxes[:,[4]] ], axis=1)

    elif obj_rep_in == 'RoLine2D_2p'  and obj_rep_out == 'XYLgWsA':
      return OBJ_REPS_PARSE.RoLine2D_2p_TO_XYLgWsA(bboxes)

    elif obj_rep_in == 'XYLgWsA' and obj_rep_out == 'RoLine2D_2p':
      return OBJ_REPS_PARSE.XYLgWsA_TO_RoLine2D_2p(bboxes)

    elif obj_rep_in == 'RoLine2D_2p'  and obj_rep_out == 'XYZLgWsHA':
      XYLgWsA = OBJ_REPS_PARSE.RoLine2D_2p_TO_XYLgWsA(bboxes)
      return OBJ_REPS_PARSE.encode_obj(XYLgWsA, 'XYLgWsA', 'XYZLgWsHA' )

    elif obj_rep_in == 'XYZLgWsHA' and obj_rep_out == 'RoLine2D_2p':
      XYLgWsA = OBJ_REPS_PARSE.encode_obj(bboxes, 'XYZLgWsHA', 'XYLgWsA')
      return OBJ_REPS_PARSE.encode_obj(XYLgWsA, 'XYLgWsA', 'RoLine2D_2p')

    assert False, f'Not implemented:\nobj_rep_in: {obj_rep_in}\nobj_rep_out: {obj_rep_out}'

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

    return bboxes

  @staticmethod
  def RoLine2D_2p_TO_XYLgWsA(bboxes):
    assert bboxes.shape[1] == 4
    corner0 = bboxes[:,0:2]
    corner1 = bboxes[:,2:4]
    center = bboxes.reshape(-1,2,2).mean(axis=1)
    vec = corner1 - corner0
    length = np.linalg.norm(vec, axis=1)[:,None]
    angle = angle_with_x_np(vec, scope_id=2)[:,None]
    # Because axis-y of img points to bottom for img, it is positive for
    # anti-clock wise. Change to positive for clock-wise
    angle = -angle
    width = np.zeros([bboxes.shape[0], 1], dtype=np.float32)
    XYLgWsA = np.concatenate([center, length, width, angle], axis=1)
    return XYLgWsA

  @staticmethod
  def XYLgWsA_TO_RoLine2D_2p(bboxes):
    assert bboxes.shape[1] == 5
    center = bboxes[:,:2]
    length = bboxes[:,2:3]
    width = bboxes[:,3:4]
    angle = bboxes[:,4]
    vec = vec_from_angle_with_x_np(angle)
    # due to y points to bottom, to make clock-wise positive:
    vec[:,1] *= -1
    corner0 = center - vec * length /2
    corner1 = center + vec * length /2
    line2d_2p = np.concatenate([corner0, corner1], axis=1)

    assert width.min() == 0 and width.max() == 0
    return line2d_2p

class O_OBJ_REPS_PARSE():
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

  (*) RoLine2D_UpRight_xyxy_sin2a
    x denotes the long axis.
    angle: [-90, 90)
  '''
  _obj_dims = {
    'CenSizeA3D': 7,
    'CenSize2D': 5,

    'XYZLgWsHA': 7,
    'XYLgWsA': 5,

    'XYLgWsAsinSin2Z0Z1': 8,
    'XYLgWsAsinSin2': 6,

    'XYLgWsSin2Sin4Z0Z1': 8,
    'XYLgWsSin2Sin4': 6,


    'RoLine2D_UpRight_xyxy_sin2a': 5,
    'RoBox2D_UpRight_xyxy_sin2a_thick': 6,
    'RoLine2D_2p': 4,
    'RoLine2D_CenterLengthAngle': 4,

    'RoBox3D_UpRight_xyxy_sin2a_thick_Z0Z1': 8,
    'Bottom_Corners': 3+3,
  }
  _obj_reps = _obj_dims.keys()

  @staticmethod
  def check_obj_dim(bboxes, obj_rep):
    assert bboxes.ndim == 2
    s = OBJ_REPS_PARSE._obj_dims[obj_rep]
    assert bboxes.shape[1] == s, f'obj_rep={obj_rep}, input shape={bboxes.shape[1]}, correct shape={s}'

  @staticmethod
  def encode_obj(bboxes, obj_rep_in, obj_rep_out, check_sin2=1):
    '''
    bboxes: [n,4] or [n,2,2]
    bboxes_out : [n,4/5]
    '''
    assert obj_rep_in  in OBJ_REPS_PARSE._obj_reps, obj_rep_in
    assert obj_rep_out in OBJ_REPS_PARSE._obj_reps, obj_rep_out
    OBJ_REPS_PARSE.check_obj_dim(bboxes, obj_rep_in)

    bboxes = OBJ_REPS_PARSE.make_x_long_dim(bboxes, obj_rep_in)

    if obj_rep_in == obj_rep_out:
      return bboxes

    if obj_rep_in == 'XYZLgWsHA'  and obj_rep_out == 'XYLgWsA':
      return bboxes[:,[0,1,3,4,6]]

    elif obj_rep_in == 'XYZLgWsHA'  and obj_rep_out == 'XYLgWsSin2Sin4Z0Z1':
      return OBJ_REPS_PARSE.XYZLgWsHA_to_XYLgWsSin2Sin4Z0Z1(bboxes)

    elif obj_rep_in == 'XYZLgWsHA'  and obj_rep_out == 'XYLgWsAsinSin2Z0Z1':
      return OBJ_REPS_PARSE.XYZLgWsHA_to_XYLgWsAsinSin2Z0Z1(bboxes)

    elif obj_rep_in == 'XYLgWsSin2Sin4Z0Z1' and obj_rep_out == 'XYZLgWsHA':
      return OBJ_REPS_PARSE.XYLgWsSin2Sin4Z0Z1_to_XYZLgWsHA(bboxes)

    elif obj_rep_in == 'XYLgWsAsinSin2Z0Z1' and obj_rep_out == 'RoLine2D_UpRight_xyxy_sin2a':
      return OBJ_REPS_PARSE.XYLgWsAsinSin2Z0Z1_to_RoLine2D_UpRight_xyxy_sin2a(bboxes)

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

    elif obj_rep_in == 'RoLine2D_2p' and obj_rep_out == 'RoLine2D_UpRight_xyxy_sin2a':
      return OBJ_REPS_PARSE.Line2p_TO_UpRight_xyxy_sin2a(bboxes)

    elif obj_rep_in == 'RoLine2D_UpRight_xyxy_sin2a' and obj_rep_out == 'RoLine2D_2p':
      return OBJ_REPS_PARSE.UpRight_xyxy_sin2a_TO_RoLine2D_2p(bboxes)

    elif obj_rep_in == 'RoLine2D_UpRight_xyxy_sin2a' and obj_rep_out == 'XYLgWsAsinSin2Z0Z1':
      return OBJ_REPS_PARSE.UpRight_xyxy_sin2a_TO_XYLgWsAsinSin2Z0Z1(bboxes)

    elif obj_rep_in == 'XYLgWsA' and obj_rep_out == 'RoBox2D_UpRight_xyxy_sin2a_thick':
      bboxes_s2t = OBJ_REPS_PARSE.CenSizeAngle_TO_UpRight_xyxy_sin2a_thick(bboxes)
      check = 1
      if check:
        bboxes_c = OBJ_REPS_PARSE.encode_obj(bboxes_s2t, 'RoBox2D_UpRight_xyxy_sin2a_thick', 'XYLgWsA')
        err = bboxes_c - bboxes
        err[:,-1] = limit_period_np(err[:,-1], 0.5, np.pi)
        err = np.abs(err).max()
        if not (err.size==0 or np.abs(err).max() < 1e-3):
          import pdb; pdb.set_trace()  # XXX BREAKPOINT
          pass
      return bboxes_s2t

    elif obj_rep_in == 'RoLine2D_CenterLengthAngle' and obj_rep_out == 'RoLine2D_2p':
      return OBJ_REPS_PARSE.CenterLengthAngle_TO_RoLine2D_2p(bboxes)

    elif obj_rep_in == 'RoLine2D_2p' and obj_rep_out == 'RoLine2D_CenterLengthAngle':
      return OBJ_REPS_PARSE.RoLine2D_2p_TO_CenterLengthAngle(bboxes)

    elif obj_rep_in == 'RoLine2D_2p' and obj_rep_out == 'XYLgWsA':
      return OBJ_REPS_PARSE.RoLine2D_2p_TO_XYLgWsA(bboxes)

    elif obj_rep_in == 'RoLine2D_UpRight_xyxy_sin2a' and obj_rep_out == 'RoLine2D_CenterLengthAngle':
      lines_2p = OBJ_REPS_PARSE.encode_obj(bboxes, 'RoLine2D_UpRight_xyxy_sin2a', 'RoLine2D_2p')
      lines_cla = OBJ_REPS_PARSE.encode_obj(lines_2p, 'RoLine2D_2p', 'RoLine2D_CenterLengthAngle')
      return lines_cla

    elif obj_rep_out == 'XYLgWsA':
      if obj_rep_in == 'RoBox2D_UpRight_xyxy_sin2a_thick':
        return OBJ_REPS_PARSE.UpRight_xyxy_sin2a_thick_TO_CenSizeAngle(bboxes, check_sin2)
      if obj_rep_in == 'RoLine2D_UpRight_xyxy_sin2a':
        # to RoBox2D_UpRight_xyxy_sin2a_thick
        bboxes = np.concatenate([bboxes, bboxes[:,0:1]*0], axis=1)
        bboxes_csa =  OBJ_REPS_PARSE.UpRight_xyxy_sin2a_thick_TO_CenSizeAngle(bboxes, check_sin2)
        return bboxes_csa

    elif obj_rep_in == 'RoBox3D_UpRight_xyxy_sin2a_thick_Z0Z1' and obj_rep_out == 'XYZLgWsHA':
      box2d = OBJ_REPS_PARSE.encode_obj(bboxes[:,:6], 'RoBox2D_UpRight_xyxy_sin2a_thick', 'XYLgWsA')
      z0 = bboxes[:,6:7]
      z1 = bboxes[:,7:8]
      zc = bboxes[:,6:8].mean(1)[:,None]
      zs = np.abs(z1-z0)
      box3d = np.concatenate([box2d[:,:2], zc, box2d[:,2:4], zs, box2d[:,4:5]], axis=1)
      return box3d

    elif obj_rep_in == 'RoBox3D_UpRight_xyxy_sin2a_thick_Z0Z1' and obj_rep_out == 'RoLine2D_UpRight_xyxy_sin2a':
      return bboxes[:, :5]

    elif obj_rep_in == 'RoBox3D_UpRight_xyxy_sin2a_thick_Z0Z1' and obj_rep_out == 'Bottom_Corners':
      line_2p = OBJ_REPS_PARSE.encode_obj(bboxes[:,:5], 'RoLine2D_UpRight_xyxy_sin2a', 'RoLine2D_2p')
      thick = bboxes[:,5:6]
      z0 = bboxes[:,6:7]
      z1 = bboxes[:,7:8]
      height = z1-z0
      bottom_corners = np.concatenate([line_2p[:,:2], z0, line_2p[:,2:4], z0], axis=1)
      return bottom_corners

    elif obj_rep_in == 'XYZLgWsHA'  and obj_rep_out == 'RoBox3D_UpRight_xyxy_sin2a_thick_Z0Z1':
      # XYLgWsA
      XYLgWsA = bboxes[:, [0,1, 3,4, 6]]
      z0 = bboxes[:, 2:3] - bboxes[:, 5:6]/2
      z1 = bboxes[:, 2:3] + bboxes[:, 5:6]/2
      line2d = OBJ_REPS_PARSE.encode_obj(XYLgWsA, 'XYLgWsA', 'RoBox2D_UpRight_xyxy_sin2a_thick')
      line3d = np.concatenate([line2d, z0, z1], axis=1)
      return line3d

    assert False, f'Not implemented:\nobj_rep_in: {obj_rep_in}\nobj_rep_out: {obj_rep_out}'

  def lines2d_to_lines3d(lines2d, ):
    '''
      lines2d: [n,5]  RoLine2D_UpRight_xyxy_sin2a
      out: RoBox3D_UpRight_xyxy_sin2a_thick_Z0Z1
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
  def XYLgWsAsinSin2Z0Z1_to_RoLine2D_UpRight_xyxy_sin2a(bboxes):
    XYZLgWsHA = OBJ_REPS_PARSE.encode_obj(bboxes, 'XYLgWsAsinSin2Z0Z1', 'XYZLgWsHA')
    RoBox3D_UpRight_xyxy_sin2a_thick_Z0Z1 = OBJ_REPS_PARSE.encode_obj(XYZLgWsHA, 'XYZLgWsHA', 'RoBox3D_UpRight_xyxy_sin2a_thick_Z0Z1')
    return RoBox3D_UpRight_xyxy_sin2a_thick_Z0Z1[:,:5]

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
    RoBox3D_UpRight_xyxy_sin2a_thick_Z0Z1 = OBJ_REPS_PARSE.encode_obj(XYZLgWsHA,  'XYZLgWsHA', 'RoBox3D_UpRight_xyxy_sin2a_thick_Z0Z1')
    if debug:
      #_show_3d_points_objs_ls(objs_ls=[XYZLgWsHA], obj_rep='XYZLgWsHA')
      _show_3d_points_objs_ls(objs_ls=[RoBox3D_UpRight_xyxy_sin2a_thick_Z0Z1], obj_rep='RoBox3D_UpRight_xyxy_sin2a_thick_Z0Z1')
    RoLine2D_UpRight_xyxy_sin2a = OBJ_REPS_PARSE.encode_obj(RoBox3D_UpRight_xyxy_sin2a_thick_Z0Z1, 'RoBox3D_UpRight_xyxy_sin2a_thick_Z0Z1', 'RoLine2D_UpRight_xyxy_sin2a')
    RoLine2D_2p = OBJ_REPS_PARSE.encode_obj( RoLine2D_UpRight_xyxy_sin2a, 'RoLine2D_UpRight_xyxy_sin2a',  'RoLine2D_2p' )

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
    RoBox3D_UpRight_xyxy_sin2a_thick_Z0Z1 = OBJ_REPS_PARSE.encode_obj(XYZLgWsHA,  'XYZLgWsHA', 'RoBox3D_UpRight_xyxy_sin2a_thick_Z0Z1')
    if debug:
      #_show_3d_points_objs_ls(objs_ls=[XYZLgWsHA], obj_rep='XYZLgWsHA')
      _show_3d_points_objs_ls(objs_ls=[RoBox3D_UpRight_xyxy_sin2a_thick_Z0Z1], obj_rep='RoBox3D_UpRight_xyxy_sin2a_thick_Z0Z1')
    RoLine2D_UpRight_xyxy_sin2a = OBJ_REPS_PARSE.encode_obj(RoBox3D_UpRight_xyxy_sin2a_thick_Z0Z1, 'RoBox3D_UpRight_xyxy_sin2a_thick_Z0Z1', 'RoLine2D_UpRight_xyxy_sin2a')
    RoLine2D_2p = OBJ_REPS_PARSE.encode_obj( RoLine2D_UpRight_xyxy_sin2a, 'RoLine2D_UpRight_xyxy_sin2a',  'RoLine2D_2p' )

    if debug:
      _show_objs_ls_points_ls((512, 512), [RoLine2D_2p], obj_rep='RoLine2D_2p')
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass
    return RoLine2D_2p

  @staticmethod
  def CenterLengthAngle_TO_RoLine2D_2p(bboxes):
    center = bboxes[:,:2]
    length = bboxes[:,2:3]
    angle = bboxes[:,3]
    vec = vec_from_angle_with_x_np(angle)
    # due to y points to bottom, to make clock-wise positive:
    vec[:,1] *= -1
    corner0 = center - vec * length /2
    corner1 = center + vec * length /2
    line2d_2p = np.concatenate([corner0, corner1], axis=1)

    check=1
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
    # Because axis-y of img points to bottom for img, it is positive for
    # anti-clock wise. Change to positive for clock-wise
    angle = -angle
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
    angle = -angle
    width = np.zeros([bboxes.shape[0], 1], dtype=np.float32)
    XYLgWsA = np.concatenate([center, length, width, angle], axis=1)
    return XYLgWsA

  @staticmethod
  def CenSizeAngle_TO_UpRight_xyxy_sin2a_thick(bboxes):
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
    line2d_2p = OBJ_REPS_PARSE.CenterLengthAngle_TO_RoLine2D_2p(line2d_angle)
    line2d_sin2 = OBJ_REPS_PARSE.Line2p_TO_UpRight_xyxy_sin2a(line2d_2p)
    line2d_sin2tck = np.concatenate([line2d_sin2, thickness], axis=1)

    err = np.sin(angle.reshape(-1)*2) - line2d_sin2tck[:, -2]
    if not (err.size==0 or np.abs(err).max() < 1e-3):
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass
    return line2d_sin2tck

  @staticmethod
  def UpRight_xyxy_sin2a_thick_TO_CenSizeAngle(bboxes, check_sin2=True):
    thickness = bboxes[:,5:6]
    lines_2p = OBJ_REPS_PARSE.UpRight_xyxy_sin2a_TO_RoLine2D_2p(bboxes[:,:5])
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
    From RoLine2D_2p to RoLine2D_UpRight_xyxy_sin2a
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
    UpRight_xyxy_sin2a = OBJ_REPS_PARSE.encode_obj(bboxes, 'RoLine2D_2p', 'RoLine2D_UpRight_xyxy_sin2a')
    n = UpRight_xyxy_sin2a.shape[0]
    tmp = np.zeros([n, 3], dtype=np.float32)
    RoBox3D_UpRight_xyxy_sin2a_thick_Z0Z1 = np.concatenate([ UpRight_xyxy_sin2a, tmp  ], axis=1)
    XYZLgWsHA = OBJ_REPS_PARSE.encode_obj( RoBox3D_UpRight_xyxy_sin2a_thick_Z0Z1, 'RoBox3D_UpRight_xyxy_sin2a_thick_Z0Z1', 'XYZLgWsHA')
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
    UpRight_xyxy_sin2a = OBJ_REPS_PARSE.encode_obj(bboxes, 'RoLine2D_2p', 'RoLine2D_UpRight_xyxy_sin2a')
    n = UpRight_xyxy_sin2a.shape[0]
    tmp = np.zeros([n, 3], dtype=np.float32)
    RoBox3D_UpRight_xyxy_sin2a_thick_Z0Z1 = np.concatenate([ UpRight_xyxy_sin2a, tmp  ], axis=1)
    XYZLgWsHA = OBJ_REPS_PARSE.encode_obj( RoBox3D_UpRight_xyxy_sin2a_thick_Z0Z1, 'RoBox3D_UpRight_xyxy_sin2a_thick_Z0Z1', 'XYZLgWsHA')
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
  def UpRight_xyxy_sin2a_TO_RoLine2D_2p(lines):
    '''
    From RoLine2D_UpRight_xyxy_sin2a to RoLine2D_2p
    '''
    istopleft = (lines[:,4:5] >= 0).astype(lines.dtype)
    lines_2p = lines[:,:4] * istopleft +  lines[:,[0,3,2,1]] * (1-istopleft)
    return lines_2p

  @staticmethod
  def UpRight_xyxy_sin2a_TO_XYLgWsAsinSin2Z0Z1(bboxes):
    XYLgWsA = OBJ_REPS_PARSE.encode_obj(bboxes, 'RoLine2D_UpRight_xyxy_sin2a', 'XYLgWsA')
    XYLgWsAsinSin2Z0Z1 = OBJ_REPS_PARSE.encode_obj(XYLgWsA, 'XYLgWsA', 'XYLgWsAsinSin2Z0Z1')
    return XYLgWsAsinSin2Z0Z1

  @staticmethod
  def XYLgWsA_TO_XYLgWsAsinSin2Z0Z1(bboxes):
    theta = bboxes[:,4:5]
    AbsSin = np.abs(np.sin(theta))
    sin2theta = np.sin(theta * 2)
    tmp = np.zeros([bboxes.shape[0], 2])
    return np.concatenate([ bboxes[:,:4], AbsSin, sin2theta, tmp  ], axis=1)

class GraphUtils:
  @staticmethod
  def optimize_graph(lines_in, scores=None, labels=None,
                     obj_rep='RoLine2D_UpRight_xyxy_sin2a',
                     opt_graph_cor_dis_thr=0, min_out_length=0):
    '''
      lines_in: [n,5]
      Before optimization, all lines with length < opt_graph_cor_dis_thr are deleted.
      After optimization, all lines with length < min_out_length are deleted.
    '''
    from tools.visual_utils import _show_objs_ls_points_ls, _show_3d_points_objs_ls
    assert opt_graph_cor_dis_thr>0

    # filter short lines
    line_length_in = np.linalg.norm(lines_in[:,2:4] - lines_in[:,:2], axis=1)
    valid_line_mask = line_length_in > opt_graph_cor_dis_thr
    del_lines = lines_in[valid_line_mask==False]
    lines_in = lines_in[valid_line_mask]
    if scores is not None:
      scores = scores[valid_line_mask]
    if labels is not None:
      labels = labels[valid_line_mask]

    #
    if obj_rep != 'RoLine2D_UpRight_xyxy_sin2a':
      lines_in = OBJ_REPS_PARSE.encode_obj(lines_in, obj_rep_in, 'RoLine2D_UpRight_xyxy_sin2a')
    num_line = lines_in.shape[0]
    if scores is None and labels is None:
      lab_sco_lines = None
    else:
      lab_sco_lines = np.concatenate([labels.reshape(num_line,1), scores.reshape(num_line,1)], axis=1)
    corners_in, lab_sco_cors, corIds_per_line, num_cor_uq_org = \
          GraphUtils.gen_corners_from_lines_np(lines_in, lab_sco_lines, 'RoLine2D_UpRight_xyxy_sin2a')
    if scores is None and labels is None:
      labels_cor = None
      scores_cor = None
      corners_labs = corners_in
    else:
      labels_cor = lab_sco_cors[:,0]
      scores_cor = lab_sco_cors[:,1]
      corners_labs = np.concatenate([corners_in, labels_cor.reshape(-1,1)*100], axis=1)
    corners_merged, cor_scores_merged, cor_labels_merged = merge_corners(
            corners_labs, scores_cor, opt_graph_cor_dis_thr=opt_graph_cor_dis_thr)
    #corners_merged = round_positions(corners_merged, 1000)
    corners_merged_per_line = corners_merged[corIds_per_line]

    lines_merged = OBJ_REPS_PARSE.encode_obj(corners_merged_per_line.reshape(-1,4), 'RoLine2D_2p', obj_rep)


    # remove short lines
    line_length_out = np.linalg.norm(lines_merged[:,2:4] - lines_merged[:,:2], axis=1)
    valid_inds = np.where(line_length_out > min_out_length)[0]
    rm_num = line_length_out.shape[0] - valid_inds.shape[0]
    lines_merged = lines_merged[valid_inds]

    if scores is None and labels is None:
      line_labels_merged = None
      line_scores_merged = None
    else:
      line_labels_merged = (cor_labels_merged[corIds_per_line][:,0]/100).astype(np.int32)
      line_scores_merged = cor_scores_merged[corIds_per_line].mean(axis=1)[:,None]
      line_labels_merged = line_labels_merged[valid_inds]
      line_scores_merged = line_scores_merged[valid_inds]


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
              objs_ls = [lines_in, del_lines], obj_rep='RoLine2D_UpRight_xyxy_sin2a',
              obj_colors=['red','blue'], thickness=5,)

          print('\nCompare org and merged')
          _show_3d_points_objs_ls( points_ls=[deleted_corners],
            objs_ls = [lines_in, lines_merged], obj_rep='RoLine2D_UpRight_xyxy_sin2a',
            obj_colors=['blue', 'red'], thickness=[3,2],)

          print('\nMerged result')
          _show_3d_points_objs_ls( points_ls=[deleted_corners],
            objs_ls = [lines_merged], obj_rep='RoLine2D_UpRight_xyxy_sin2a',
            obj_colors='random', thickness=5,)

        pass

    return lines_merged, line_scores_merged, line_labels_merged, valid_line_mask

  @staticmethod
  def gen_corners_from_lines_np(lines, labels=None, obj_rep='RoLine2D_UpRight_xyxy_sin2a', flag=''):
      '''
      lines: [n,5]
      labels: [n,1/2]: 1 for label only, 2 for label and score

      corners: [m,2]
      labels_cor: [m, 1/2]
      corIds_per_line: [n,2]
      num_cor_uq: m
      '''
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

def merge_corners(corners_0, scores_0=None, opt_graph_cor_dis_thr=3):
  diss = corners_0[None,:,:] - corners_0[:,None,:]
  diss = np.linalg.norm(diss, axis=2)
  mask = diss < opt_graph_cor_dis_thr
  nc = corners_0.shape[0]
  merging_ids = []
  corners_1 = []
  scores_1 = []
  for i in range(nc):
    ids_i = np.where(mask[i])[0]
    merging_ids.append(ids_i)
    if scores_0 is not None:
      weights = scores_0[ids_i] / scores_0[ids_i].sum()
      merged_sco = ( scores_0[ids_i] * weights).sum()
      scores_1.append(merged_sco)
      merged_cor = ( corners_0[ids_i] * weights[:,None]).sum(axis=0)
    else:
      merged_cor = ( corners_0[ids_i] ).mean(axis=0)
    corners_1.append(merged_cor)
    pass
  corners_1 = np.array(corners_1).reshape((-1, corners_0.shape[1]))
  if scores_0 is not None:
    scores_merged = np.array(scores_1)
  else:
    scores_merged = None
  if corners_1.shape[1] == 2:
    labels_merged = None
  else:
    labels_merged = corners_1[:,2]
  corners_merged = corners_1[:,:2]
  return corners_merged, scores_merged, labels_merged



def test_2d():
  from tools.debug_utils import _show_lines_ls_points_ls
  bboxes_UpRight_xyxy_sin2a_thick = np.array([ [120.902, 344.609, 402.904, 345.328,  -0.51 ,   0. ]] )
  boxes_CenSizeAngle = np.array( [[ 261.903, 344.968, 282.002,   0.   ,  -0.003 ]] )
  _show_lines_ls_points_ls( (512,512),[ bboxes_UpRight_xyxy_sin2a_thick] )
  bboxes_csa = OBJ_REPS_PARSE.UpRight_xyxy_sin2a_thick_TO_CenSizeAngle( bboxes_UpRight_xyxy_sin2a_thick )
  bboxes_sin2 = OBJ_REPS_PARSE.CenSizeAngle_TO_UpRight_xyxy_sin2a_thick( boxes_CenSizeAngle )
  print( bboxes_csa )
  print(bboxes_sin2)

def test_3d():
  from tools.visual_utils import _show_3d_points_objs_ls

  XYZLgWsHA = np.array([
                        [ 0,1,0, 5, 1, 2, np.pi/4 ],
                        [ 0,1,0, 5, 1, 2, -np.pi/4 ],
                        [ 0,1,0, 5, 1, 2, np.pi/8 ],
                        ])
  XYLgWsSin2Sin4Z0Z1 = OBJ_REPS_PARSE.encode_obj(XYZLgWsHA, 'XYZLgWsHA', 'XYLgWsSin2Sin4Z0Z1')
  _XYZLgWsHA = OBJ_REPS_PARSE.encode_obj(XYLgWsSin2Sin4Z0Z1, 'XYLgWsSin2Sin4Z0Z1', 'XYZLgWsHA')
  err0 = XYZLgWsHA - _XYZLgWsHA
  err = np.abs(err0).max()
  print(f'\nerr: {err}')
  print(f'XYZLgWsHA: \n{XYZLgWsHA}')
  print(f'_XYZLgWsHA: \n{_XYZLgWsHA}')
  assert err < 1e-5
  _show_3d_points_objs_ls( objs_ls = [XYZLgWsHA], obj_rep='XYZLgWsHA' )
  _show_3d_points_objs_ls( objs_ls = [XYLgWsSin2Sin4Z0Z1], obj_rep='XYLgWsSin2Sin4Z0Z1' )
  _show_3d_points_objs_ls( objs_ls = [XYZLgWsHA, _XYZLgWsHA], obj_rep='XYZLgWsHA' )


if __name__ == '__main__':
  test_3d()



