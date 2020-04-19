# xyz 17 Apr
import numpy as np
import mmcv
from .geometry_utils import sin2theta_np, angle_with_x_np, vec_from_angle_with_x_np
import cv2
from tools import debug_utils
import torch
from obj_geo_utils.geometry_utils import limit_period_np

class OBJ_REPS_PARSE:
  '''
  For RoLine2D_UpRight_xyxy_sin2a, x responds to the long dimension.
  '''
  _obj_dims = {
    'AlBox2D_UpRight_xyxy': 4,
    'RoBox2D_CenSizeAngle': 5,
    'RoLine2D_UpRight_xyxy_sin2a': 5,
    'RoBox2D_UpRight_xyxy_sin2a_thick': 6,
    'RoLine2D_2p': 4,
    'RoLine2D_CenterLengthAngle': 4,

    'RoBox3D_CenSizeAngle': 7,
    'RoBox3D_UpRight_xyxy_sin2a_thick_Z0Z1': 8,
  }
  _obj_reps = _obj_dims.keys()

  @staticmethod
  def check_obj_dim(bboxes, obj_rep):
    assert bboxes.ndim == 2
    assert bboxes.shape[1] == OBJ_REPS_PARSE._obj_dims[obj_rep], f'obj_rep={obj_rep}, shape={bboxes.shape[1]}'

  @staticmethod
  def encode_obj(bboxes, obj_rep_in, obj_rep_out):
    '''
    bboxes: [n,4] or [n,2,2]
    bboxes_out : [n,4/5]
    '''
    assert obj_rep_in  in OBJ_REPS_PARSE._obj_reps, obj_rep_in
    assert obj_rep_out in OBJ_REPS_PARSE._obj_reps, obj_rep_out
    OBJ_REPS_PARSE.check_obj_dim(bboxes, obj_rep_in)

    if obj_rep_in == obj_rep_out:
      return bboxes

    if obj_rep_in == 'RoLine2D_2p' and obj_rep_out == 'RoLine2D_UpRight_xyxy_sin2a':
      return OBJ_REPS_PARSE.Line2p_TO_UpRight_xyxy_sin2a(bboxes)

    elif obj_rep_in == 'RoLine2D_UpRight_xyxy_sin2a' and obj_rep_out == 'RoLine2D_2p':
      return OBJ_REPS_PARSE.UpRight_xyxy_sin2a_TO_2p(bboxes)

    elif obj_rep_in == 'RoBox2D_CenSizeAngle' and obj_rep_out == 'RoBox2D_UpRight_xyxy_sin2a_thick':
      bboxes = OBJ_REPS_PARSE.make_x_long_dim(bboxes, 'RoBox2D_CenSizeAngle')
      bboxes_s2t = OBJ_REPS_PARSE.CenSizeAngle_TO_UpRight_xyxy_sin2a_thick(bboxes)
      check = 1
      if check:
        bboxes_c = OBJ_REPS_PARSE.encode_obj(bboxes_s2t, 'RoBox2D_UpRight_xyxy_sin2a_thick', 'RoBox2D_CenSizeAngle')
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

    elif obj_rep_out == 'RoBox2D_CenSizeAngle':
      if obj_rep_in == 'RoBox2D_UpRight_xyxy_sin2a_thick':
        return OBJ_REPS_PARSE.UpRight_xyxy_sin2a_thick_TO_CenSizeAngle(bboxes)
      if obj_rep_in == 'RoLine2D_UpRight_xyxy_sin2a':
        # to RoBox2D_UpRight_xyxy_sin2a_thick
        bboxes = np.concatenate([bboxes, bboxes[:,0:1]*0], axis=1)
        bboxes_csa =  OBJ_REPS_PARSE.UpRight_xyxy_sin2a_thick_TO_CenSizeAngle(bboxes)
        return bboxes_csa

    elif obj_rep_in == 'RoBox3D_UpRight_xyxy_sin2a_thick_Z0Z1' and obj_rep_out == 'RoBox3D_CenSizeAngle':
      box2d = OBJ_REPS_PARSE.encode_obj(bboxes[:,:6], 'RoBox2D_UpRight_xyxy_sin2a_thick', 'RoBox2D_CenSizeAngle')
      z0 = bboxes[:,6:7]
      z1 = bboxes[:,7:8]
      zc = bboxes[:,6:8].mean(1)[:,None]
      zs = np.abs(z1-z0)
      box3d = np.concatenate([box2d[:,:2], zc, box2d[:,2:4], zs, box2d[:,4:5]], axis=1)
      return box3d

    elif obj_rep_in == 'RoBox3D_CenSizeAngle'  and obj_rep_out == 'RoBox3D_UpRight_xyxy_sin2a_thick_Z0Z1':
      # RoBox2D_CenSizeAngle
      bboxes = OBJ_REPS_PARSE.make_x_long_dim(bboxes, 'RoBox3D_CenSizeAngle')
      bboxes = OBJ_REPS_PARSE.make_x_long_dim(bboxes, 'RoBox3D_CenSizeAngle')
      box2d = bboxes[:, [0,1, 3,4, 6]]
      z0 = bboxes[:, 2:3] - bboxes[:, 5:6]/2
      z1 = bboxes[:, 2:3] + bboxes[:, 5:6]/2
      line2d = OBJ_REPS_PARSE.encode_obj(box2d, 'RoBox2D_CenSizeAngle', 'RoBox2D_UpRight_xyxy_sin2a_thick')
      line3d = np.concatenate([line2d, z0, z1], axis=1)
      return line3d

    assert False, f'Not implemented:\nobj_rep_in: {obj_rep_in}\nobj_rep_out: {obj_rep_out}'


  @staticmethod
  def make_x_long_dim(bboxes, obj_rep):
    assert obj_rep in ['RoBox2D_CenSizeAngle', 'RoBox3D_CenSizeAngle']
    if obj_rep == 'RoBox2D_CenSizeAngle':
      xi, yi, ai = 2, 3, 4
    if obj_rep == 'RoBox3D_CenSizeAngle':
      xi, yi, ai = 3, 4, 6
    switch = (bboxes[:,yi] > bboxes[:,xi]).astype(bboxes.dtype)
    bboxes[:,ai] = limit_period_np( bboxes[:,ai] + switch * np.pi / 2, 0.5, np.pi )
    bboxes[:,[xi,yi]] = bboxes[:,[xi,yi]] * (1-switch) + bboxes[:,[yi,xi]] * switch
    return bboxes

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
  def CenSizeAngle_TO_UpRight_xyxy_sin2a_thick(bboxes):
    '''
    In the input , x either y can be the longer one.
    If y is the longer one, angle = angle + 90.
    To make x the longer one.
    '''
    bboxes = OBJ_REPS_PARSE.make_x_long_dim(bboxes, 'RoBox2D_CenSizeAngle')
    center = bboxes[:,:2]
    size = bboxes[:,2:4]
    angle = bboxes[:,4:5]
    length = size.max(1)[:,None]
    thickness = size.min(1)[:,None]

    line2d_angle = np.concatenate([center, length, angle], axis=1)
    line2d_2p = OBJ_REPS_PARSE.CenterLengthAngle_TO_RoLine2D_2p(line2d_angle)
    line2d_sin2 = OBJ_REPS_PARSE.Line2p_TO_UpRight_xyxy_sin2a(line2d_2p)
    line2d_sin2tck = np.concatenate([line2d_sin2, thickness], axis=1)

    err = np.sin(angle*2) - line2d_sin2tck[:, -2]
    if not (err.size==0 or np.abs(err).max() < 1e-3):
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass
    return line2d_sin2tck

  @staticmethod
  def UpRight_xyxy_sin2a_thick_TO_CenSizeAngle(bboxes):
    thickness = bboxes[:,5:6]
    lines_2p = OBJ_REPS_PARSE.UpRight_xyxy_sin2a_TO_2p(bboxes[:,:5])
    lines_CenLengthAngle = OBJ_REPS_PARSE.RoLine2D_2p_TO_CenterLengthAngle(lines_2p)
    boxes_csa = np.concatenate([lines_CenLengthAngle[:,[0,1,2]], thickness, lines_CenLengthAngle[:,[3]]], axis=1)
    err = np.sin(boxes_csa[:,-1]*2) - bboxes[:,4]*1.0
    if not (err.size==0 or np.abs(err).max() < 1e-3):
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
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
  def UpRight_xyxy_sin2a_TO_2p(lines):
    '''
    From RoLine2D_UpRight_xyxy_sin2a to RoLine2D_2p
    '''
    istopleft = (lines[:,4:5] >= 0).astype(lines.dtype)
    lines_2p = lines[:,:4] * istopleft +  lines[:,[0,3,2,1]] * (1-istopleft)
    return lines_2p

class GraphUtils:
  @staticmethod
  def optimize_graph(lines_in, scores=None, labels=None,
                     obj_rep='RoLine2D_UpRight_xyxy_sin2a',
                     opt_graph_cor_dis_thr=0):
    '''
      lines_in: [n,5]
    '''
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
    corners_merged = round_positions(corners_merged, 100)
    corners_merged_per_line = corners_merged[corIds_per_line]
    if scores is None and labels is None:
      line_labels_merged = None
      line_scores_merged = None
    else:
      line_labels_merged = cor_labels_merged[corIds_per_line][:,0].astype(np.int32)
      line_scores_merged = cor_scores_merged[corIds_per_line].mean(axis=1)[:,None]

    lines_merged = OBJ_REPS_PARSE.encode_obj(corners_merged_per_line.reshape(-1,4), 'RoLine2D_2p', obj_rep)

    #lines_merged = lines_merged[0:5]
    #line_scores_merged = line_scores_merged[0:5]
    #line_labels_merged = line_labels_merged[0:5]

    if 1:
      corners_uq, unique_indices, inds_inverse = np.unique(corners_merged, axis=0, return_index=True, return_inverse=True)
      num_cor_org = corners_in.shape[0]
      num_cor_merged = corners_uq.shape[0]
      print(f'\ncorner num: {num_cor_org} -> {num_cor_merged}')

      #_show_lines_ls_points_ls((512,512), [lines_in], [corners_merged], ['white'], 'random')
      #_show_lines_ls_points_ls((512,512), [lines_in, lines_merged], [], ['white','green'])

      #det_corners, cor_scores, det_cor_ids_per_line, num_cor_uq_merged_check = gen_corners_from_lines_np(lines_merged, line_labels_merged, 'lscope_istopleft', flag='debug')
      #print(f'num_cor_uq_merged_check: {num_cor_uq_merged_check}')

      check_lines = OBJ_REPS_PARSE.encode_obj(lines_merged, obj_rep, 'RoLine2D_2p')
      check_corners = check_lines.reshape(-1,2,2)
      cor_err = corners_merged_per_line - check_corners
      cor_loc_check = np.abs(cor_err).max() < 1e-3
      print(f'cor_loc_check: {cor_loc_check}')
      print(lines_merged[0:5])

    return lines_merged, line_scores_merged, line_labels_merged

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
      corners1 = round_positions(corners1, 1000)
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
