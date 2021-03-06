## April 2019 xyz
import torch, math
import numpy as np
import time
from collections import defaultdict
np.set_printoptions(precision=3,suppress=True)

'''
Angle order:

1. clock wise is positive in all my functions, same as opencv
2. cross: positive for anti-clock wise
3. opencv cv2.minAreaRect:  positive for anti-clock wise

All angle used in this file is in unit of radian
'''

def limit_period(val, offset, period):
  '''
    [0, pi]: offset=0, period=pi
    [-pi/2, pi/2]: offset=0.5, period=pi
    [-pi, 0]: offset=1, period=pi
  '''
  return val - torch.floor(val / period + offset) * period

def limit_period_np(val, offset, period):
  '''
    [0, pi]: offset=0, period=pi
    [-pi/2, pi/2]: offset=0.5, period=pi
    [-pi, 0]: offset=1, period=pi
  '''
  return val - np.floor(val / period + offset) * period

def mean_angles(angles0, angles1, offset, period, weights0=1, weights1=1):
    dif0 = angles0 - angles1
    dif1 = limit_period_np(dif0, 0.5, period)
    ofs = dif0 - dif1
    ave0 = (angles0 * weights0 + (angles1 + ofs) * weights1) / (weights0 + weights1)
    ave1 = limit_period_np(ave0, offset, period)
    return ave1

def is_align_angles(angles):
  steps = angles / (np.pi/2)
  gaps = steps - np.round(steps)
  gaps = np.abs(gaps) * 180 / np.pi
  align_mask = gaps < 5
  return align_mask

def angle_dif_by_period_np(val0, val1, aim_scope_id):
    dif = val1 - val0
    if aim_scope_id == 0:
      dif = limit_period_np(dif, 0.5, math.pi)
    else:
      raise NotImplementedError
    return dif

def angle_dif_by_period(val0, val1, aim_scope_id):
    '''
      aim_scope_id 0:[-pi/2, pi/2]
    '''
    dif = val1 - val0
    if aim_scope_id == 0:
      dif = limit_period(dif, 0.5, math.pi)
    else:
      raise NotImplementedError
    return dif

def angle_with_x(vec_start, scope_id=0,  debug=0):
  '''
   vec_start: [n,2/3]
   angle: [n]
  '''
  assert vec_start.dim() == 2
  vec_x = vec_start.clone().detach()
  vec_x[:,0] = 1
  vec_x[:,1:] = 0
  if torch.isnan(vec_start).any() or torch.isnan(vec_x).any():
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    pass
  return angle_from_vecs_to_vece(vec_x, vec_start, scope_id, debug)

def angle_with_x_np(vec_start, scope_id):
  # Note: for 3d coordinate, it is positive for clock wise.
  # But for img, it is positive for anti-clock wise, because y-axis of img
  # points to bottom
  angle = angle_with_x(torch.from_numpy(vec_start), scope_id)
  return angle.data.numpy()

def vec_from_angle_with_x_np(angle):
  '''
  angle: [n]
  vec: [n,2]
  '''
  assert angle.ndim == 1
  x = np.cos(angle)[:,None]
  y = np.sin(angle)[:,None]
  vec = np.concatenate([x,y], axis=1)

  check = 1
  if check:
    angle_c = angle_with_x_np(vec, 2)
    assert all(np.abs(angle - angle_c) < 1e-3)
  return vec

def sin2theta(vec_start_0, vec_end_0):
  '''
    vec_start: [n,2/3]
    vec_end: [n,2/3]
    zero as ref

   scope_id=0: [0,pi]
            1: (-pi/2, pi/2]
            2: (-pi, pi]
            3: (0, pi*2]

   clock wise is positive
   angle: [n]
  '''
  assert vec_start_0.dim() == 2 and  vec_end_0.dim() == 2
  assert (vec_start_0.shape[0] == vec_end_0.shape[0]) or vec_start_0.shape[0]==1 or vec_end_0.shape[0]==1
  assert vec_start_0.shape[1] == vec_end_0.shape[1] # 2 or 3

  vec_start_0 = vec_start_0.float()
  vec_end_0 = vec_end_0.float()

  norm_start = torch.norm(vec_start_0, dim=1, keepdim=True)
  norm_end = torch.norm(vec_end_0, dim=1, keepdim=True)
  #assert norm_start.min() > 1e-4 and norm_end.min() > 1e-4 # return nan
  vec_start = vec_start_0 / norm_start
  vec_end = vec_end_0 / norm_end
  #assert not torch.isnan(vec_end).any()
  if vec_start.dim() == 2:
    tmp = vec_start[:,0:1]*0
    vec_start = torch.cat([vec_start, tmp], 1)
    vec_end = torch.cat([vec_end, tmp], 1)
  cz = torch.cross( vec_start, vec_end, dim=1)[:,2]
  # sometimes abs(cz)>1 because of float drift. result in nan angle
  mask = (torch.abs(cz) > 1).to(vec_start.dtype)
  cz = cz * (1 - mask*1e-7)
  # cross is positive for anti-clock wise. change to clock-wise
  cz = -cz  # [-pi/2, pi/2]

  # check :angle or pi-angle
  cosa = torch.sum(vec_start * vec_end,dim=1)
  res = 2 * cz * cosa

  # get nan when the norm of vec_end is 0. set as 0 directly
  res[torch.isnan(res)] = 0
  assert not torch.isnan(res).any()
  sin_theta = cz
  return res, sin_theta

def sin2theta_np(vec_start, vec_end):
    vec_start_t = torch.from_numpy(vec_start)
    vec_end_t = torch.from_numpy(vec_end)
    res, _ = sin2theta(vec_start_t, vec_end_t)
    return res.cpu().data.numpy()

def angle_from_vecs_to_vece_np(vec_start, vec_end, scope_id, debug=0):
    vec_start_t = torch.from_numpy(vec_start)
    vec_end_t = torch.from_numpy(vec_end)
    angles = angle_from_vecs_to_vece(vec_start_t, vec_end_t, scope_id, debug)
    return angles.cpu().data.numpy()

def angle_from_vecs_to_vece(vec_start, vec_end, scope_id, debug=0):
  '''
    vec_start: [n,2/3]
    vec_end: [n,2/3]
    zero as ref

   scope_id=0: [0,pi]
            1: (-pi/2, pi/2]
            2: (-pi, pi]
            3: (0, pi*2]

   clock wise is positive from vec_start to vec_end
   angle: [n]
  '''
  assert vec_start.dim() == 2 and  vec_end.dim() == 2
  assert (vec_start.shape[0] == vec_end.shape[0]) or vec_start.shape[0]==1 or vec_end.shape[0]==1
  assert vec_start.shape[1] == vec_end.shape[1] # 2 or 3
  ns = vec_start.shape[0]
  ne = vec_end.shape[0]
  if not ns == ne:
    assert ns == 1 or ne == 1
    if ns == 1:
      vec_start = vec_start.repeat(ne, 1)
    if ne == 1:
      vec_end = vec_end.repeat(ns, 1)
    pass
  vec_start = vec_start.float()
  vec_end = vec_end.float()

  norm_start = torch.norm(vec_start, dim=1, keepdim=True)
  norm_end = torch.norm(vec_end, dim=1, keepdim=True)
  norm_start = norm_start.clamp(min=1e-4)
  norm_end = norm_end.clamp(min=1e-4)
  #assert norm_start.min() > 1e-4 and norm_end.min() > 1e-4 # return nan
  vec_start_nm = vec_start / norm_start
  vec_end_nm = vec_end / norm_end
  if vec_start_nm.dim() == 2:
    tmp = vec_start_nm[:,0:1]*0
    vec_start_nm = torch.cat([vec_start_nm, tmp], 1)
    vec_end_nm = torch.cat([vec_end_nm, tmp], 1)
  cz = torch.cross( vec_start_nm, vec_end_nm, dim=1)[:,2]
  cz = cz.clamp(min=-1, max=1)
  angle = torch.asin(cz)

  # check :angle or pi-angle
  cosa = torch.sum(vec_start_nm * vec_end_nm, dim=1)
  mask = (cosa >= 0).to(vec_end_nm.dtype)
  angle = angle * mask + (math.pi - angle)* (1-mask)
  # angle: [-pi/2, pi/2*3]

  if scope_id == 1:
    # [-pi/2, pi/2]: offset=0.5, period=pi
    angle = limit_period(angle, 0.5, math.pi)
  elif scope_id == 0:
    # [0, pi]: offset=0, period=pi
    angle = limit_period(angle, 0, math.pi)
  elif scope_id == 2:
    # [-pi, pi]: offset=0.5, period=pi*2
    angle = limit_period(angle, 0.5, math.pi*2)
  elif scope_id == 3:
    # [0, 2*pi]: offset=0, period=pi*2
    angle = limit_period(angle, 0, math.pi*2)
  else:
    raise NotImplementedError
  if torch.isnan(angle).any():
    ids = torch.nonzero(torch.isnan(angle))
    nan_vec_start = vec_start[ids]
    nan_vec_end = vec_end[ids]
    print(f'nan_vec_start:\n{nan_vec_start}')
    print(f'nan_vec_end:\n{nan_vec_end}')
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    #assert False
    pass
  return angle

def R_to_Euler(rMatrix):
  from scipy.spatial.transform import Rotation as R
  r = R.from_matrix(rMatrix)
  euler = r.as_euler('zyx', degrees=False)

  #r1 = R.from_euler('zyx', euler)
  #M1 = r1.as_matrix()
  return euler

def R_to_Vec(rMatrix):
  from scipy.spatial.transform import Rotation as R
  r = R.from_matrix(rMatrix)
  rotvec = r.as_rotvec()

  #r1 = R.from_rotvec(rotvec)
  #M1 = r1.as_matrix()
  return rotvec

def points_in_lines(points, lines, threshold_dis=0.03, one_point_in_max_1_line=False):
  '''
  points:[n,2/3]
  lines:[m,2,2/3]
  dis: [n,m]
  out: [n,m]

  (1)vertial dis=0
  (2) angle>90 OR corner dis=0
  '''
  assert points.ndim == 2
  assert lines.ndim == 3
  assert lines.shape[1] == 2
  assert points.shape[1] == lines.shape[2]
  epsion = points.max() * 1e-4

  num_p = points.shape[0]
  num_l = lines.shape[0]
  d = points.shape[1]

  pc_distances0 = points.reshape([num_p,1,1,d]) - lines.reshape([1,num_l,2,d])
  pc_distances = np.linalg.norm(pc_distances0, axis=-1).min(2)

  pl_distances = vertical_dis_points_lines(points, lines)

  tmp_l = np.tile( lines.reshape([1,num_l,2,d]), (num_p,1,1,1) )
  tmp_p = np.tile( points.reshape([num_p,1,1,d]), (1,num_l,1,1) )
  dirs0 = tmp_l - tmp_p
  dirs1 = dirs0.reshape([num_l*num_p, 2,d])
  angles0 = angle_of_2lines(dirs1[:,0,:], dirs1[:,1,:])
  angles = angles0.reshape([num_p, num_l])


  mask_pc = pc_distances <= threshold_dis
  mask_pl = pl_distances <= threshold_dis

  mask_a = angles > np.pi/2
  in_line_mask = mask_a  * mask_pl
  #in_line_mask = (mask_a + mask_pc) * mask_pl

  if one_point_in_max_1_line:
    tmp = pl_distances + (1-in_line_mask)*2000
    min_mask = tmp < tmp.min(1, keepdims=True) + 1e-3
    in_line_mask *= min_mask

  debug = 1
  if debug:
    final_pl_dis = pl_distances[in_line_mask]
    print(final_pl_dis)
  return in_line_mask

def vertical_dis_points_lines(points, lines):
  '''
  points:[n,3/2]
  lines:[m,2,3/2]
  dis: [n,m]
  '''
  dis = []
  pn = points.shape[0]
  for i in range(pn):
    dis.append( vertical_dis_1point_lines(points[i], lines).reshape([1,-1]) )
  dis = np.concatenate(dis, 0)
  return dis

def vertical_dis_1point_lines(point, lines, no_extend=False):
  '''
  point:[2/3]
  lines:[m,2,2/3]
  dis: [m]
  '''
  assert point.ndim == 1
  assert lines.ndim == 3
  assert lines.shape[1:] == (2,3) or lines.shape[1:] == (2,2)
  # use lines[:,0,:] as ref
  point = point.reshape([1,-1])
  ln = lines.shape[0]
  direc_p = point - lines[:,0,:]
  direc_l = lines[:,1,:] - lines[:,0,:]
  angles = angle_of_2lines(direc_p, direc_l, scope_id=0)
  dis = np.sin(angles) * np.linalg.norm(direc_p, axis=1)
  mask = np.isnan(dis)
  dis[mask] = 0

  if no_extend:
    diss_to_corner = np.linalg.norm( point[:,None,:] - lines, axis=-1 ).min(1)
    diss_to_center = np.linalg.norm( point[:,:] - lines.mean(1), axis=-1 )
    is_extend = diss_to_center > diss_to_corner
    dis += is_extend * 10000
  return dis

def angle_of_2lines(line0, line1, scope_id=0):
  '''
    line0: [n,2/3]
    line1: [n,2/3]
    zero as ref

   scope_id=0: [0,pi]
            1: (-pi/2, pi/2]

   angle: [n]
  '''
  assert line0.ndim == line1.ndim == 2
  assert (line0.shape[0] == line1.shape[0]) or line0.shape[0]==1 or line1.shape[0]==1
  assert line0.shape[1] == line1.shape[1] # 2 or 3

  norm0 = np.linalg.norm(line0, axis=1, keepdims=True)
  norm1 = np.linalg.norm(line1, axis=1, keepdims=True)
  #assert norm0.min() > 1e-4 and norm1.min() > 1e-4 # return nan
  line0 = line0 / norm0
  line1 = line1 / norm1
  cos = np.sum(line0 * line1, axis=1)
  # sometimes cos can be a bit > 1
  ref = np.clip(np.abs(cos), a_min=1, a_max=None)
  cos /= ref
  angle = np.arccos( cos )
  #if not np.all(np.abs(cos)<=1):
  #  mask = norm0 < 1e-5 or norm1 < 1e-5
  #  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  #  angle[mask] = 0
  angle[np.isnan(angle)]=0

  if scope_id == 0:
    pass
  elif scope_id == 1:
    # (-pi/2, pi/2]: offset=0.5, period=pi
    angle = limit_period_np(angle, 0.5, np.pi)
  else:
    raise NotImplementedError
  assert not np.any(np.isnan(angle))
  return angle

def align_pred_gt_bboxes( bboxes_pred, bboxes_gt, obj_rep, method='sum_loc_min' ):
  '''
  change bboxes_pred
  '''
  assert obj_rep == 'Rect4CornersZ0Z1' or obj_rep == 'Rect4Corners'
  c = 10 if obj_rep == 'Rect4CornersZ0Z1'  else 8
  n = bboxes_pred.shape[0]
  assert bboxes_pred.shape == bboxes_gt.shape == (n,c)
  pred_corners = bboxes_pred[:, :8].reshape(n,4,2)
  gt_corners = bboxes_gt[:,:8].reshape(n,4,2)

  if method == 'first_corner_angle_min':
    gt_center = gt_corners.mean(dim=1, keepdim=True)
    gt_corners = gt_corners - gt_center
    gt0_angles = angle_with_x(gt_corners[:,0,:], scope_id=3)[:,None]
    angles_dif = angles_s - gt0_angles
    # to [-pi, pi]
    angles_dif = limit_period( angles_dif, 0.5, 2*np.pi ).abs()
    start_ids = angles_dif.argmin(1)[:,None]

  elif method == 'sum_loc_min':
    device = pred_corners.device
    loc_difs = []
    for i in range(4):
      ids_i  = (torch.arange(4).to(device) + i) % 4
      cors_i = torch.index_select( pred_corners, 1, ids_i )
      dif_i = (cors_i - gt_corners).norm(dim=2).sum(1,keepdim=True)
      loc_difs.append( dif_i )
    loc_difs = torch.cat(loc_difs, dim=1)
    start_ids = loc_difs.argmin(1)[:,None]
  assert start_ids.shape == (n,1)
  tmp  = torch.arange(4).view(-1,4).to(start_ids.device)
  align_ids = start_ids + tmp
  align_ids = align_ids % 4
  corners_pred_aligned = torch.gather(  pred_corners, 1, align_ids[:,:,None].repeat(1,1,2) )
  bboxes_pred_aligned = torch.cat( [corners_pred_aligned.reshape(n,8), bboxes_pred[:,8:c]], dim=1 )
  return bboxes_pred_aligned

def align_four_corners( pred_corners,  pred_center=None, gt_corners=None ):
  pred_corners, rect_loss = sort_four_corners(pred_corners)
  pred_corners = align_pred_gt_bboxes(pred_corners.reshape(-1,8), gt_corners.reshape(-1,8), obj_rep='Rect4Corners')
  return pred_corners, rect_loss

def sort_four_corners_np( pred_corners, pred_center=None ):
  pred_corners = torch.from_numpy( pred_corners ).to(torch.float32)
  sorted_corners, _ =  sort_four_corners(pred_corners)
  return sorted_corners.numpy()

def sort_four_corners( pred_corners, pred_center=None ):
  '''
  pred_corners: [batch_size, 4, h, w,2] or  [n,4,2]
  '''
  from tools.visual_utils import _show_objs_ls_points_ls, _show_3d_points_objs_ls, _show_objs_ls_points_ls_torch
  input_ndim = pred_corners.ndim
  assert input_ndim == 5 or input_ndim == 3
  if input_ndim == 5:
    bs, npts, h, w, d = pred_corners.shape
    n = bs*h*w
    pred_corners = pred_corners.permute(0,2,3,1,4).reshape(n, npts, 2)
    if pred_center is not None:
      assert pred_center.shape == (bs, 1, h, w, d)
      pred_center = pred_center.permute(0,2,3,1,4).reshape(n, 1, 2)
  else:
    n, npts, d = pred_corners.shape
    if pred_center is not None:
      assert pred_center.shape == (n, 1, d)
  assert npts == 4
  assert d == 2

  if pred_center is not None:
    pred_cor_cen = torch.cat([pred_corners, pred_center], dim=1)
    #center = pred_cor_cen.mean(dim=1, keepdim=True)
    center = pred_corners.mean(dim=1, keepdim=True)
    pred_center_err = pred_center - center
    out_rect_loss_cen = (pred_center_err).abs().mean(dim=-1)
  else:
    center = pred_corners.mean(dim=1, keepdim=True)

  corners = pred_corners - center

  # 1. sort by angles, start from 0 to 2pi
  angles = angle_with_x(corners.detach().reshape(n*4,2), scope_id=3) # (0, pi*2]
  angles = angles.reshape(n,4)
  angles_s, sort_ids = angles.sort(dim=1)
  corners = torch.gather(corners, 1, sort_ids[:,:,None].repeat(1,1,2))

  # 2. the loss as a rect: same diagonal length, same alpha
  loss_diag_len = corners.norm(dim=-1).std(dim=-1,keepdim=True)
  dia_sum_0 = (corners[:,0]+corners[:,2]).abs().sum(-1, keepdim=True)
  dia_sum_1 = (corners[:,1]+corners[:,3]).abs().sum(-1, keepdim=True)
  loss_diag_sum = dia_sum_0 + dia_sum_1

  if pred_center is not None:
    rect_loss = loss_diag_len + loss_diag_sum
    #rect_loss = out_rect_loss_cen + loss_diag_len + loss_diag_sum
    #rect_loss = out_rect_loss_cen
  else:
    rect_loss = loss_diag_len + loss_diag_sum
    rect_loss = None

  corners = corners + center
  if input_ndim == 5:
    bboxes_out = corners.reshape(bs, h, w, 8).permute(0, 3, 1, 2)
    if rect_loss is not None:
      rect_loss = rect_loss.reshape(bs, h, w, 1).permute(0, 3, 1, 2)
  elif input_ndim == 3:
    bboxes_out = corners.reshape(n,8)
  return bboxes_out, rect_loss

def arg_sort_points_np( corners, center=None):
  '''
  corners: [n,m,2]
  sort_ids: [n,m]
  '''
  assert corners.ndim == 3
  assert corners.shape[2] == 2
  if center is not None:
    assert center.ndim == 3
  n, npts, _ = corners.shape
  if center is None:
    #center = corners.mean(axis=1, keepdims=True)
    min_xy = corners.min(axis=1, keepdims=True)
    max_xy = corners.max(axis=1, keepdims=True)
    center = (min_xy + max_xy) / 2
  corners_nm = corners - center

  # 1. sort by angles, start from 0 to 2pi
  angles = angle_with_x_np(corners_nm.reshape(n*npts,2), scope_id=3) # (0, pi*2]
  angles = angles.reshape(n,npts)
  sort_ids = angles.argsort(axis=1)

  #from tools.visual_utils import _show_objs_ls_points_ls
  #_show_objs_ls_points_ls( (512,512), points_ls=[corners[0][sort_ids[0]], center[0]], point_scores_ls=[range(corners[0].shape[0]), None], point_colors=['red', 'green'] )

  return sort_ids

def sort_corners_np( pred_corners, pred_center=None ):
  pred_corners = torch.from_numpy( pred_corners ).to(torch.float32)
  sorted_corners, _ =  sort_corners(pred_corners)
  return sorted_corners.numpy()
def sort_corners( pred_corners, pred_center=None ):
  '''
  pred_corners: [batch_size, 4, h, w,2] or  [n,4,2]
  '''
  from tools.visual_utils import _show_objs_ls_points_ls, _show_3d_points_objs_ls, _show_objs_ls_points_ls_torch
  input_ndim = pred_corners.ndim
  assert input_ndim == 5 or input_ndim == 3
  if input_ndim == 5:
    bs, npts, h, w, d = pred_corners.shape
    n = bs*h*w
    pred_corners = pred_corners.permute(0,2,3,1,4).reshape(n, npts, 2)
    if pred_center is not None:
      assert pred_center.shape == (bs, 1, h, w, d)
      pred_center = pred_center.permute(0,2,3,1,4).reshape(n, 1, 2)
  else:
    n, npts, d = pred_corners.shape
    if pred_center is not None:
      assert pred_center.shape == (n, 1, d)
  assert d == 2

  if pred_center is not None:
    pred_cor_cen = torch.cat([pred_corners, pred_center], dim=1)
    #center = pred_cor_cen.mean(dim=1, keepdim=True)
    center = pred_corners.mean(dim=1, keepdim=True)
    pred_center_err = pred_center - center
    out_rect_loss_cen = (pred_center_err).abs().mean(dim=-1)
  else:
    center = pred_corners.mean(dim=1, keepdim=True)

  corners = pred_corners - center

  # 1. sort by angles, start from 0 to 2pi
  angles = angle_with_x(corners.detach().reshape(n*npts,2), scope_id=3) # (0, pi*2]
  angles = angles.reshape(n,npts)
  angles_s, sort_ids = angles.sort(dim=1)
  corners = torch.gather(corners, 1, sort_ids[:,:,None].repeat(1,1,2))

  # 2. the loss as a rect: same diagonal length, same alpha
  loss_diag_len = corners.norm(dim=-1).std(dim=-1,keepdim=True)
  dia_sum_0 = (corners[:,0]+corners[:,2]).abs().sum(-1, keepdim=True)
  dia_sum_1 = (corners[:,1]+corners[:,3]).abs().sum(-1, keepdim=True)
  loss_diag_sum = dia_sum_0 + dia_sum_1

  if pred_center is not None:
    rect_loss = loss_diag_len + loss_diag_sum
    #rect_loss = out_rect_loss_cen + loss_diag_len + loss_diag_sum
    #rect_loss = out_rect_loss_cen
  else:
    rect_loss = loss_diag_len + loss_diag_sum
    rect_loss = None

  corners = corners + center
  if input_ndim == 5:
    bboxes_out = corners.reshape(bs, h, w, npts*2).permute(0, 3, 1, 2)
    if rect_loss is not None:
      rect_loss = rect_loss.reshape(bs, h, w, 1).permute(0, 3, 1, 2)
  elif input_ndim == 3:
    bboxes_out = corners.reshape(n,npts*2)
  return bboxes_out, rect_loss

def four_corners_to_box( rect_corners, rect_center=None,  stage=None,  bbox_weights=None, bbox_gt=None):
  '''
  rect_corners: [batch_size, 4, h, w,2] or  [n,4,2]
  rect_center: [batch_size, 1, h, w,2] or  [n,1,2]
  The first one is prediction of center, 1:5 are predictions of four corners
  bbox_weights: same size as box_out, used for reducing processing time by only processing positive

  box_out: [batch_size, 6, h, w],
  out_obj_rep: XYDAsinAsinSin2
  '''
  from tools.visual_utils import _show_objs_ls_points_ls, _show_3d_points_objs_ls, _show_objs_ls_points_ls_torch

  record_t = 0
  if record_t:
    t0 = time.time()
  input_ndim = rect_corners.ndim
  assert input_ndim == 5 or input_ndim == 3
  if input_ndim == 5:
    bs, npts, h, w, d = rect_corners.shape
    n = bs*h*w
    rect_corners = rect_corners.permute(0,2,3,1,4).reshape(n, npts, 2)
    if rect_center is not None:
      assert rect_center.shape == (bs, 1, h, w, d)
      rect_center = rect_center.permute(0,2,3,1,4).reshape(n, 1, 2)
  else:
    n, npts, d = rect_corners.shape
    if rect_center is not None:
      assert rect_center.shape == (n, 1, d)
  assert npts == 4
  assert d == 2

  if bbox_weights is not None:
    assert bbox_weights.shape[0] == n
    pos_inds = torch.nonzero(bbox_weights[:,0]).view(-1)
    rect_corners = rect_corners[pos_inds]
    bbox_gt = bbox_gt[pos_inds]
    if rect_center is not None:
      rect_center = rect_center[pos_inds]
    n0 = n
    n = pos_inds.numel()

  if rect_center is not None:
    rect_cor_cen = torch.cat([rect_corners, rect_center], dim=1)
    center = rect_cor_cen.mean(dim=1, keepdim=True)
    center_err = rect_center - center
    out_rect_loss_cen = (center_err).abs().mean(dim=-1)
  else:
    center = rect_corners.mean(dim=1, keepdim=True)

  corners = rect_corners - center

  # (1) get diagonal length
  diag_leng = corners.norm(dim=-1, keepdim=True)
  diag_leng = diag_leng.clamp(min=1e-4)
  out_diag_leng_ave = diag_leng.clamp(min = 1e-4).mean(dim=1) * 2
  corners_nm = corners / diag_leng # [16384, 4, 2]
  if torch.isnan(corners_nm).any():
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    pass
  out_rect_loss_diag_len = diag_leng.squeeze(2).std(1, keepdim=True)

  # (2) sort corners
  # sort corners by anges. As arcsin is nor differeniable, angles is only used
  # for sorting, not used for loss.
  angles = angle_with_x(corners_nm.detach().reshape(n*4,2), scope_id=3) # (0, pi*2]
  angles = angles.reshape(n,4)
  angles_s, sort_ids = angles.sort(dim=1)
  corners_nm = torch.gather(corners_nm, 1, sort_ids[:,:,None].repeat(1,1,2))

  # (3) get |cos(alpha)|, |sin(alpha)|
  #cos_corners = []
  sin_corners = []
  central_corners = []
  for i,j in [ (0,1), (1,2), (2,3), (3,0), (0,2), (1,3)]:
    cos_i = (corners_nm[:,i] * corners_nm[:,j]).sum(-1)
    cen_cor_i = (corners_nm[:,i] + corners_nm[:,j])/2
    #cos_corners.append(cos_i[:,None])
    central_corners.append(cen_cor_i[:,None,:])

    ze = torch.zeros_like(corners_nm[:,0,0:1])
    vi = torch.cat([corners_nm[:,i], ze], dim=1)
    vj = torch.cat([corners_nm[:,j], ze], dim=1)
    sin_i = torch.cross(vi,vj, dim=1)[:,2:3]
    sin_corners.append( sin_i )

  #cos_corners = torch.cat(cos_corners, dim=-1)
  central_corners = torch.cat(central_corners, dim=1)
  sin_corners = torch.cat(sin_corners, dim=-1)

  #out_cos_corners_abs_ave = cos_corners[:,:4].abs().mean(-1, keepdim=True)
  #out_rec_loss_hwratio = (cos_corners[:,4:6]+1).abs().mean(-1, keepdim=True)
  out_sin_corners_abs_ave = (sin_corners[:, :4]).abs().mean(-1, keepdim=True)
  out_rec_loss_hwratio = sin_corners[:, 4:6].abs().mean(-1, keepdim=True)

  # (4) get theta
  central_corners = central_corners[:,:4]
  cencor_norm = central_corners.norm(dim=-1)
  is_02_long = cencor_norm[:,[0,2]].mean(1) > cencor_norm[:,[1,3]].mean(1)
  is_02_long = is_02_long.to(torch.float32)[:,None, None]
  diacen_2pts_long_axis = central_corners[:,[0,2]] * is_02_long + central_corners[:,[1,3]] * (1-is_02_long)
  cencor_norm_long_axis = cencor_norm[:,[0,2],None] * is_02_long + cencor_norm[:,[1,3],None] * (1-is_02_long)
  diacen_2pts_long_axis = diacen_2pts_long_axis / cencor_norm_long_axis

  sin2_theta = 2*(diacen_2pts_long_axis[:,:,0] * diacen_2pts_long_axis[:,:,1]).mean(1, keepdim=True)
  sin_theta_abs = diacen_2pts_long_axis[:,:,1].abs().mean(1, keepdim=True)
  #cos_theta_abs = diacen_2pts_long_axis[:,:,0].abs().mean(1, keepdim=True)

  center = center.squeeze(1)
  rect_loss = torch.cat( [out_rect_loss_diag_len, out_rec_loss_hwratio ], dim = 1)
  if rect_center is not None:
    rect_loss = torch.cat([rect_loss, out_rect_loss_cen], dim=1)
  rect_loss = rect_loss.mean(1, keepdim=True)

  box_out = torch.cat([ center, out_diag_leng_ave, out_sin_corners_abs_ave, sin_theta_abs, sin2_theta  ], dim=1)

  # From  XYDAsinAsinSin2Z0Z1 to out_obj_rep
  #z0z1 = torch.zeros_like(box_out[:,:2])
  #box_out = torch.cat([box_out, z0z1], dim=1)
  #box_out = OBJ_REPS_PARSE.encode_obj( box_out, 'XYDAsinAsinSin2Z0Z1', out_obj_rep )

  if input_ndim == 5:
    box_out = box_out.reshape(bs, h, w, 6).permute(0, 3, 1, 2)
    rect_loss = rect_loss.reshape(bs, h, w, 1).permute(0, 3, 1, 2)
  if input_ndim == 3:
    pass

  if bbox_weights is not None:
    if n>0 and 0:
      rect_corners = (corners + center[:,None]).reshape(-1,2)
      _show_objs_ls_points_ls_torch( (512,512), [box_out, bbox_gt], 'XYDAsinAsinSin2Z0Z1', [rect_corners], obj_colors=['green','red'], point_colors='blue' )
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass
    box_out_all = torch.zeros([n0,8], dtype=box_out.dtype, device=box_out.device)
    box_out_all[pos_inds] = box_out
    box_out = box_out_all
    rect_loss_all = torch.zeros([n0,1], dtype=box_out.dtype, device=box_out.device)
    rect_loss_all[pos_inds] = rect_loss
    rect_loss = rect_loss_all
    pass


  if record_t:
    t = time.time() - t0
    print(f'4 corners to rect time: {n}, \t{t:.3f}')
    print(f'stage: {stage}')
  return box_out, rect_loss

def four_corners_to_box_np( rect_corners, rect_center=None,  stage=None,  bbox_weights=None, bbox_gt=None):
  '''
  rect_corners: [batch_size, 4, h, w,2] or  [n,4,2]
  rect_center: [batch_size, 1, h, w,2] or  [n,1,2]
  The first one is prediction of center, 1:5 are predictions of four corners
  bbox_weights: same size as box_out, used for reducing processing time by only processing positive

  box_out: [batch_size, 6, h, w],
  out_obj_rep: 'XYDAsinAsinSin2Z0Z1'
  '''
  rect_corners = torch.from_numpy(rect_corners).to(torch.float32)
  box, _ = four_corners_to_box(rect_corners)
  box = box.numpy()
  return box

def lines_intersection_2d(line0s, line1s, must_on0=False, must_on1=False,
          min_angle=0):
    '''
    line0s: [n,2,2]
    line1s: [m,2,2]
    return [n,m,2,2]
    '''
    shape0 = line0s.shape
    shape1 = line1s.shape
    if shape0[0] * shape1[0] == 0:
        return np.empty([shape0[0], shape1[0], 2, 2])
    assert len(shape0) == len(shape1) == 3
    assert shape0[1:] == shape1[1:] == (2,2)
    ints_all = []
    for line0 in line0s:
      ints_0 = []
      for line1 in line1s:
        ints = line_intersection_2d(line0, line1, must_on0, must_on1, min_angle)
        ints_0.append(ints.reshape(1,1,2))
      ints_0 = np.concatenate(ints_0, 1)
      ints_all.append(ints_0)
    ints_all = np.concatenate(ints_all, 0)
    return ints_all

def line_intersection_2d(line0, line1, must_on0=False, must_on1=False,
          min_angle=0):
    '''
      line0: [2,2]
      line1: [2,2]
      must_on0: must on the scope of line0, no extend
      must_on1: must on the scope of line1, no extend
      out: [2]

      v01 = p1 - p0
      v23 = p3 - p2
      intersection = p0 + v01*k0 = p2 + v23 * k1
      [v01, v23][k0;-k1] = p2 - p0
      intersection between p0 and p1: 1>=k0>=0
      intersection between p2 and p3: 1>=k1>=0

      return [2]
    '''

    assert (line0.shape == (2,2) and line1.shape == (2,2))
            #(line0.shape == (2,3) and line1.shape == (2,3))
    dim = line0.shape[1]
    p0,p1 = line0
    p2,p3 = line1

    v01 = p1-p0
    v23 = p3-p2
    v01v23 = np.concatenate([v01.reshape([2,1]), (-1)*v23.reshape([2,1])], 1)
    p2sp0 = (p2-p0).reshape([2,1])

    try:
      inv_vov1 = np.linalg.inv(v01v23)
      K = np.matmul(inv_vov1, p2sp0)

      if must_on0 and (K[0]>1 or K[0]<0):
        return np.array([np.nan]*2)
      if must_on1 and (K[1]>1 or K[1]<0):
        return np.array([np.nan]*2)

      intersec = p0 + v01 * K[0]
      intersec_ = p2 + v23 * K[1]
      if not np.linalg.norm(intersec - intersec_) < 1e-4:
        print(f'{intersec} \n{intersec_}')
        return np.array([np.nan]*2)
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        pass

      direc0 = (line0[1] - line0[0]).reshape([1,2])
      direc1 = (line1[1] - line1[0]).reshape([1,2])
      angle = angle_of_2lines(direc0, direc1, scope_id=1)[0]
      angle = np.abs(angle)

      show = 0
      if show:
        from tools.visual_utils import _show_objs_ls_points_ls, _show_3d_points_objs_ls
        from obj_geo_utils.obj_utils import OBJ_REPS_PARSE
        print(f'K:{K}\nangle:{angle}')
        lines_show = np.concatenate([np.expand_dims(line0,0), np.expand_dims(line1,0)],0).reshape(2,4)
        points_show = np.array([[intersec[0], intersec[1], 0]])
        _show_3d_points_objs_ls( [points_show], objs_ls=[lines_show], obj_rep='RoLine2D_2p' )
        import pdb; pdb.set_trace()  # XXX BREAKPOINT

      if angle > min_angle:
        return intersec
      else:
        return np.array([np.nan]*2)
    except np.linalg.LinAlgError:
      return np.array([np.nan]*2)

def unused_get_ceiling_floor_from_box_walls(ceiling_boxes, walls, obj_rep, cat_name):
  '''
  walls:  [m,7]
  ceiling_boxes: [n,7]

  ceiling_polygon: [n, k]
  '''
  from obj_geo_utils.obj_utils import OBJ_REPS_PARSE
  from tools.visual_utils import _show_polygon_surface, _show_3d_points_objs_ls

  if ceiling_boxes.shape[0] == 0:
    return ceiling_boxes

  ceiling_boxes = OBJ_REPS_PARSE.encode_obj(ceiling_boxes, obj_rep, 'XYZLgWsHA')
  nc = ceiling_boxes.shape[0]
  zc = ceiling_boxes[:,2]
  zs = ceiling_boxes[:,5]
  z0 = zc - zs/2
  z1 = zc + zs/2

  if cat_name == 'ceiling':
    cor_type = 'Top_Corners'
    z = z0
  elif cat_name == 'floor':
    cor_type = 'Bottom_Corners'
    z = z1

  wn = walls.shape[0]
  wall_corners = OBJ_REPS_PARSE.encode_obj(walls, obj_rep, cor_type).reshape(wn, 4, 3)
  wall_corners2d = sort_corners_np(wall_corners[:,:,:2].reshape(1,-1,2)).reshape(-1,2)
  wall_corners2d = np.repeat( wall_corners2d[None,:,:], nc, 0)
  cor_num = wall_corners2d.shape[1]

  zs = np.repeat(z0[:,None,None], cor_num, 1)
  wall_corners3d_0 = np.concatenate([wall_corners2d, zs],2)

  zs = np.repeat(z1[:,None,None], cor_num, 1)
  wall_corners3d_1 = np.concatenate([wall_corners2d, zs],2)

  wall_corners3d = np.concatenate([wall_corners3d_0, wall_corners3d_1], 1)

  #_show_3d_points_objs_ls([wall_corners3d.reshape(-1,3)], objs_ls=[walls], obj_rep=obj_rep, polygons_ls=[wall_corners3d])
  return wall_corners3d

def get_cf_from_wall(ceiling_boxes, walls, obj_rep, cat_name, check_valid=True):
  import open3d as o3d
  from obj_geo_utils.obj_utils import OBJ_REPS_PARSE
  from tools.visual_utils import _show_polygon_surface, _show_3d_points_objs_ls
  assert cat_name in ['ceiling', 'floor']
  #assert obj_rep == 'XYZLgWsHA'

  #if ceiling_boxes.shape[0] == 0:
  #  return ceiling_boxes

  walls = OBJ_REPS_PARSE.encode_obj(walls, obj_rep, 'XYZLgWsHA')
  ceiling_boxes = OBJ_REPS_PARSE.encode_obj(ceiling_boxes, obj_rep, 'XYZLgWsHA')
  obj_rep = 'XYZLgWsHA'

  nc = ceiling_boxes.shape[0]
  zc = ceiling_boxes[:,2]
  zs = ceiling_boxes[:,5]
  z0 = zc - zs/2
  z1 = zc + zs/2

  if cat_name == 'ceiling':
    cor_type = 'Top_Corners'
    z = z0
    walls_z =  walls[:,2] + walls[:,5]/2
  elif cat_name == 'floor':
    cor_type = 'Bottom_Corners'
    z = z1
    walls_z =  walls[:,2] - walls[:,5]/2

  z = walls_z[0]
  wall_points2d = OBJ_REPS_PARSE.encode_obj(walls, obj_rep, 'RoLine2D_2p').reshape(-1,2,2)
  #_show_3d_points_objs_ls(objs_ls=[walls], obj_rep=obj_rep, )
  mesh, _ =  corners2d_to_surface(wall_points2d)
  if check_valid:
    mesh = [m for m in mesh if is_valid_mesh(m[0], wall_points2d)]
  for i in range(len(mesh)):
    ki = mesh[i][0].shape[0]
    zs = np.ones([ki, 1]) * z
    triangles = np.concatenate([mesh[i][0], zs ], 1)
    mesh[i] = (triangles, mesh[i][1])
  if 0:
    _show_3d_points_objs_ls(objs_ls=[walls], obj_rep=obj_rep, polygons_ls=[mesh])
  if 0:
    for j in range(len(mesh)):
      _show_3d_points_objs_ls(objs_ls=[walls], obj_rep=obj_rep, polygons_ls=[mesh[j:j+1]])
  return mesh

def is_valid_mesh(points, edges):
  from tools.visual_utils import _show_polygon_surface, _show_3d_points_objs_ls
  cen = points.mean(0).reshape(1,2)
  leng = np.linalg.norm(points, axis=1).mean()
  line_A = np.repeat(cen, 2, 0)
  line_B = np.repeat(cen, 2, 0)
  line_A[0,0] -= leng/2
  line_A[1,0] += leng/2
  line_B[0,1] -= leng/2
  line_B[1,1] += leng/2
  intsec_A = lines_intersection_2d( line_A[None,...], edges, must_on1=True)
  intsec_B = lines_intersection_2d( line_B[None,...], edges, must_on1=True)
  its_A = intsec_A - cen
  its_B = intsec_B - cen
  A_ok = (its_A > 1).any() and (its_A<-1).any()
  B_ok = (its_B > 1).any() and (its_B<-1).any()
  num_A = np.sum( 1-np.isnan(its_A))/2
  num_B = np.sum( 1-np.isnan(its_B))/2
  if 0:
    print(f'intsec_A:\n{intsec_A}')
    print(f'intsec_B:\n{intsec_B}')
    print(f'A_ok: {A_ok}')
    print(f'B_ok: {B_ok}')
    _show_3d_points_objs_ls([points], objs_ls=[edges.reshape(-1,4)],  obj_rep='RoLine2D_2p')
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    pass
  valid =  A_ok and B_ok
  valid_extra = (num_A+num_B)>6 and (A_ok or B_ok)
  return valid or valid_extra

def corners2d_to_surface(corners, time=0):
  '''
  corners: [n,2,2]
  '''
  assert corners.shape[1:] == (2,2)
  time = time + 1
  from tools.visual_utils import _show_polygon_surface, _show_3d_points_objs_ls
  #_show_3d_points_objs_ls( objs_ls=[corners.reshape(-1,4)], obj_rep='RoLine2D_2p' )

  corners = corners.copy()
  n = corners.shape[0]
  if n==2:
    mesh = [convex_mesh(corners.reshape(-1,2))]
    return mesh, time

  dis = np.linalg.norm( corners[:,1] - corners[:,0], axis=1)
  ids = (-dis).argsort()
  min_ref_length = 30
  n1 = 0
  n2 = 0
  for j in range(n):
    i = ids[j]
    if dis[i] < min_ref_length:
      continue
    vec = corners.reshape(-1,2) - corners[i,0:1]
    ref = corners[i].reshape(1,4)
    cen = ref.reshape(2,2).mean(0, keepdims=True)
    ref_vec = cen - corners[i,0:1]
    angles_i = angle_from_vecs_to_vece_np( ref_vec, vec, 2)
    ver_dis = vertical_dis_points_lines( corners.reshape(-1,2), ref.reshape(1,2,2) ).reshape(-1)
    ver_dis *= np.sign(angles_i)
    mask_1 = ver_dis > 1
    mask_2 =  ver_dis < -1
    n1 = mask_1.sum()
    n2 = mask_2.sum()
    mask_1[[i*2, i*2+1]] = False
    mask_2[[i*2, i*2+1]] = False
    mask_on_ref = (mask_1+mask_2)==False
    #mask_on_ref[[i*2, i*2+1]] = False
    if n1 > 0 and n2 > 0:
      break
    pass
  print(f'{time} n={n}, j={j}, i={i}, n1={n1}, n2={n2}')
  if  n1 == 0 or n2 == 0:
    mesh = [convex_mesh(corners.reshape(-1,2))]
  else:
    if 0:
      points1 = corners.reshape(-1,2)[mask_1]
      points2 = corners.reshape(-1,2)[mask_2]
      _show_3d_points_objs_ls( [points1], objs_ls=[corners.reshape(-1,4), ref], obj_rep='RoLine2D_2p', obj_colors=['blue', 'red'] )
      _show_3d_points_objs_ls( [points2], objs_ls=[corners.reshape(-1,4), ref], obj_rep='RoLine2D_2p', obj_colors=['blue', 'red'] )
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass

    group1 = get_intsec_group(corners, ref[0], mask_1)
    group2 = get_intsec_group(corners, ref[0], mask_2)

    cor_on_ref = corners[ mask_on_ref.reshape(-1,2).all(1) ]
    group1 = np.concatenate([group1, cor_on_ref], 0)
    group2 = np.concatenate([group2, cor_on_ref], 0)
    if 0:
      _show_3d_points_objs_ls( [group1.reshape(-1,2)], objs_ls=[corners.reshape(-1,4), ref], obj_rep='RoLine2D_2p', obj_colors=['blue', 'red'] )
      _show_3d_points_objs_ls( [group2.reshape(-1,2)], objs_ls=[corners.reshape(-1,4), ref], obj_rep='RoLine2D_2p', obj_colors=['blue', 'red'] )

    if time > 100:
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass

    mesh1, time = corners2d_to_surface(group1, time)
    mesh2, time = corners2d_to_surface(group2, time)
    mesh = mesh1 + mesh2
    #time = time + 2
  return mesh, time

def get_intsec_group(corners, ref, mask_1 ):
      from tools.visual_utils import _show_polygon_surface, _show_3d_points_objs_ls
      corners = corners.copy()
      corners_1 = corners.copy()
      n = corners.shape[0]
      ref = ref.reshape(2,2)
      line_mask_1 = mask_1.reshape( n,2 )
      crop_mask_1 = line_mask_1.sum(1) == 1
      if not crop_mask_1.any():
        group1  = corners_1[line_mask_1.all(1)]
        return group1

      for k in range( crop_mask_1.shape[0] ):
        if crop_mask_1[k]:
          line_B = corners[ k ]
          intsec =  line_intersection_2d( ref, line_B)
          if line_mask_1[k][0]:
            k_i = 1
          else:
            k_i = 0

          if 0:
            group1  = corners_1[line_mask_1.any(1)]
            _show_3d_points_objs_ls([intsec.reshape(1,2)], objs_ls=[ref.reshape(1,4), line_B.reshape(1,4), group1.reshape(-1,4)], obj_rep='RoLine2D_2p', obj_colors=['red', 'blue', 'black'])
          corners_1[k, k_i] = intsec
          if 0:
            line_B = corners_1[ k ]
            group1  = corners_1[line_mask_1.any(1)]
            _show_3d_points_objs_ls([intsec.reshape(1,2)], objs_ls=[ref.reshape(1,4), line_B.reshape(1,4), group1.reshape(-1,4)], obj_rep='RoLine2D_2p', obj_colors=['red', 'blue', 'black'])
            import pdb; pdb.set_trace()  # XXX BREAKPOINT
            pass
      group1  = corners_1[line_mask_1.any(1)]
      return group1

def convex_mesh(points):
  assert points.shape[1:] == (2,)
  center = points.mean(0, keepdims=True)
  n = points.shape[0]
  vertices = np.concatenate([points, center], 0)
  ids = np.mgrid[:n,:n].reshape(2,-1)
  tmp = np.ones([1,n*n]) * n
  triangles = np.concatenate([ids, tmp], 0).T
  return (vertices, triangles)

def points_to_oriented_bbox(points, obj_rep_out='XYLgWsA'):
  '''
  generate 1 2d bbox
  points: [n,2]
  '''
  import cv2
  from obj_geo_utils.obj_utils import OBJ_REPS_PARSE
  box2d = cv2.minAreaRect(points)
  box2d = np.array( box2d[0]+box2d[1]+(box2d[2],) )[None, :]
  # The original angle from cv2.minAreaRect denotes the rotation from ref-x
  # to body-x. It is it positive for clock-wise.
  # make x the long dim
  if box2d[0,2] < box2d[0,3]:
    box2d[:,[2,3]] = box2d[:,[3,2]]
    box2d[:,-1] = 90+box2d[:,-1]
  box2d[:,-1] *= np.pi / 180
  box2d[:,-1] = limit_period_np(box2d[:,-1], 0.5, np.pi) # limit_period
  if obj_rep_out != 'XYLgWsA':
    box2d = OBJ_REPS_PARSE.encode_obj(box2d, 'XYLgWsA', obj_rep_out)
  return box2d

def draw_rooms_from_edges(edges, obj_rep, img_size=None, show=0):
  from scipy import ndimage
  from tools.visual_utils import draw_1_obj
  from obj_geo_utils.obj_utils import OBJ_REPS_PARSE
  from tools.visual_utils import _show_objs_ls_points_ls
  from tools.color import ColorValuesNp
  import mmcv

  edges = OBJ_REPS_PARSE.encode_obj(edges, obj_rep, 'XYLgWsA')
  if img_size is None:
    s = int(edges[:,:2].max() + 100)
    img_size = (s,s)

  img = np.zeros(img_size)
  obj_thickness = 1
  c = (255,255,255)

  for edge in edges:
    draw_1_obj(img, edge, c, obj_thickness, 'XYLgWsA')
  img = (img==0).astype(np.uint8)
  mask, num_rooms = ndimage.label(img)
  bg_label = mask[0,0]
  background = mask == bg_label
  mask[background] = 0


  img_mask = ColorValuesNp[mask]
  img_mask[background] = 255
  if show:
    mmcv.imshow(img_mask)

  return img_mask
  room_labels = [i for i in range(1, num_rooms+1) if i != bg_label]
  num_edge = edges.shape[0]
  room_ids_per_edge = np.zeros([num_edge, 2]) - 11

def get_rooms_from_edges(edges, obj_rep, gen_bbox=False, show_rooms=False, orth_dis=15):
  '''
  orth_dis: For each edge, from the center, along the orth direciton, extend {orth_dis} to get the rooom id
  '''
  from scipy import ndimage
  from tools.visual_utils import draw_1_obj
  from obj_geo_utils.obj_utils import OBJ_REPS_PARSE
  from tools.visual_utils import _show_objs_ls_points_ls
  debug = show_rooms

  edges = OBJ_REPS_PARSE.encode_obj(edges, obj_rep, 'XYLgWsA')
  s = int(edges[:,:2].max() + 100)
  img_size = (s,s)

  img = np.zeros(img_size)
  obj_thickness = 1
  c = (255,255,255)

  for edge in edges:
    draw_1_obj(img, edge, c, obj_thickness, 'XYLgWsA')
  img = (img==0).astype(np.uint8)
  mask, num_rooms = ndimage.label(img)
  bg_label = mask[0,0]
  background = mask == bg_label
  mask[background] = 0
  room_labels = [i for i in range(1, num_rooms+1) if i != bg_label]
  num_edge = edges.shape[0]
  room_ids_per_edge = np.zeros([num_edge, 2]) - 11

  orth_edges = edges.copy()
  orth_edges[:,2] = orth_dis
  orth_edges[:,-1] = orth_edges[:,-1] - np.pi/2
  orth_corners = OBJ_REPS_PARSE.encode_obj(orth_edges, 'XYLgWsA', 'RoLine2D_2p').reshape(-1,2,2).astype(np.int32)

  if debug:
    from tools.color import ColorValuesNp
    import mmcv
    img_mask = ColorValuesNp[mask]
    img_mask[background] = 255
    mmcv.imshow(img_mask)
    pass

  edge_ids_per_room = defaultdict(list)
  num_edges_inside_room = 0
  inside_edge_ids = []
  non_loop_edge_ids = []
  for i in range(num_edge):
    x1, y1 = orth_corners[i,0]
    x2, y2 = orth_corners[i,1]
    room_1 = mask[y1,x1]
    room_2 = mask[y2,x2]

    if room_1 in room_labels:
      edge_ids_per_room[room_1].append(i)
      room_ids_per_edge[i,0] = room_1
    if room_2 in room_labels:
      edge_ids_per_room[room_2].append(i)
      room_ids_per_edge[i,1] = room_2

    if debug:
      print(edge_ids_per_room)

    edge_type = ''
    if room_1 != room_2:
      if room_1 * room_2 ==0:
        edge_type = 'normal edge between a room and the background'
      else:
        edge_type = 'normal edge between two different rooms'
    if room_1 == room_2:
      if room_1 in room_labels:
        edge_type = 'this edge is inside of a room'
        inside_edge_ids.append(i)
        num_edges_inside_room += 1
      else:
        non_loop_edge_ids.append(i)
        edge_type = 'this edge belong to non_loop space, which is not a valid room'

    if debug and 1:
      cor_1 = img_mask[ y1, x1 ]
      cor_2 = img_mask[ y2, x2 ]
      print(f'\n\n{i}')
      print(f'Edge type: {edge_type}')
      print(f'room: {room_1}, {room_2}')
      print(f'room: {cor_1}, {cor_2}')
      _show_objs_ls_points_ls( img_mask, [edges[[i]]], obj_rep='XYLgWsA',
          points_ls=[ orth_corners[i] ], obj_colors='red', point_colors='green',  point_thickness=5 )
      pass

  edge_ids_per_room_out = [edge_ids_per_room[i] for i in room_labels]
  edge_ids_per_room_out = [ ids for ids in edge_ids_per_room_out  if len(ids)>0]

  if gen_bbox:
    edge_corners = OBJ_REPS_PARSE.encode_obj(edges, 'XYLgWsA', 'RoLine2D_2p')
    num_rooms = len(edge_ids_per_room_out)
    room_bboxes = []
    for i in range(num_rooms):
      ids_i = edge_ids_per_room_out[i]
      room_corners = edge_corners[ids_i]
      room_bbox = points_to_oriented_bbox(room_corners.reshape(-1,2), obj_rep)
      room_bboxes.append(room_bbox)
    if len(room_bboxes)>0:
      room_bboxes = np.concatenate(room_bboxes, 0)
    else:
      room_bboxes = np.zeros([0,7])
  else:
    room_bboxes = None

  if gen_bbox and 0:
    num_rooms = len(edge_ids_per_room_out)
    for i in range(num_rooms):
      eids = edge_ids_per_room_out[i]
      _show_objs_ls_points_ls( (512,512), [edges, edges[eids], room_bboxes[i:i+1, [0,1,3,4,6]] ],\
                              'XYLgWsA', obj_colors=['white','lime', 'red'] )

  return edge_ids_per_room_out, room_ids_per_edge, num_edges_inside_room, room_bboxes

def relation_mask_to_ids(relations):
  '''
  relations: [num_obj, num_wall]
  rel_ids: [ []*? ]* num_obj
  '''
  num_obj, num_wall = relations.shape
  rel_ids = []
  for i in range(num_obj):
    ids_i = np.where(relations[i])[0]
    rel_ids.append(ids_i)
  return rel_ids

def rel_ids_to_mask(rel_ids, num_wall):
  num_obj = len(rel_ids)
  mask = np.zeros( [num_obj, num_wall] ) == 1
  for i in range(num_obj):
    mask[i, rel_ids[i] ] = True
  return mask

def check_duplicate(objs, obj_rep, iou_thr=0.3):
  from mmdet.core.bbox.geometry import dsiou_rotated_3d_bbox_np
  from tools.visual_utils import _show_objs_ls_points_ls

  objs = objs[:,:7]
  ious = dsiou_rotated_3d_bbox_np( objs[:,:7], objs[:,:7], iou_w=1.0, size_rate_thres=0.3, ref='union', only_2d=True )
  mask = ious > iou_thr
  np.fill_diagonal(mask, False)
  duplicte = mask.sum() > 0
  if duplicte:
    ids_x, ids_y = np.where(mask)
    i, j = ids_x[0], ids_y[0]
    print(f'duplicate: {i}, {j}, iou={ious[i,j]}')
    print(objs[ [i,j] ])
    n = objs.shape[0]
    objs_du = objs[ [i,j] ]
    _show_objs_ls_points_ls((512,512), [objs, objs[[i]], objs[[j]] ], obj_rep, obj_colors=['white', 'red', 'lime'], obj_thickness=[1, 5, 2])
    return False
  return True


def ununsed_get_cf_from_wall(floors0, walls, obj_rep, cat_name):
  from obj_geo_utils.obj_utils import OBJ_REPS_PARSE
  from tools.visual_utils import _show_polygon_surface, _show_3d_points_objs_ls
  assert cat_name in ['ceiling', 'floor']
  assert obj_rep == 'XYZLgWsHA'
  scores = floors0[:,-1].copy()
  floors = floors0[:,:7].copy()
  walls = walls[:,:7].copy()

  if cat_name == 'ceiling':
    cor_type = 'Top_Corners'
  elif cat_name == 'floor':
    cor_type = 'Bottom_Corners'

  wall_corners_2d = OBJ_REPS_PARSE.encode_obj(walls, obj_rep, 'RoLine2D_2p')
  ver_walls = walls.copy()
  ver_walls[:,-1] += np.pi/2
  ver_wall_corners_2d = OBJ_REPS_PARSE.encode_obj(ver_walls, obj_rep, 'RoLine2D_2p')
  num_w = walls.shape[0]
  intsecs = lines_intersection_2d( ver_wall_corners_2d.reshape(num_w,2,2),  wall_corners_2d.reshape(num_w,2,2), must_on1=True)
  vertices = []
  for i in range(num_w):
    wall_i = wall_corners_2d[i].reshape(2,2)
    dis_i = wall_i - intsecs[i]
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    for j in range(num_w):
      if intsecs[i,j][i] != np.nan:
        #triangle = np.concatenate( [, intsecs[i,j][None,...] ], 0 )
        vertices.append( triangle )

  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  pass

class OBJ_DEF():
  @staticmethod
  def limit_yaw(yaws, yx_zb):
    '''
    standard: [0, pi]
    yx_zb: [-pi/2, pi/2]
    '''
    if yx_zb:
      yaws = limit_period(yaws, 0.5, math.pi)
    else:
      yaws = limit_period(yaws, 0, math.pi)
    return yaws

  @staticmethod
  def check_bboxes(bboxes, yx_zb, check_thickness=0):
    '''
    x_size > y_size
    '''
    ofs = 1e-6
    if bboxes.shape[0]==0:
      return
    if yx_zb:
      #if not torch.all(bboxes[:,3] <= bboxes[:,4]):
      #  xydif = bboxes[:,3] - bboxes[:,4]
      #  import pdb; pdb.set_trace()  # XXX BREAKPOINT
      #  pass
      #assert torch.all(bboxes[:,3] <= bboxes[:,4])


      #print(bboxes)
      #if not torch.max(torch.abs(bboxes[:,-1]))<=math.pi*0.5+ofs:
      #  import pdb; pdb.set_trace()  # XXX BREAKPOINT
      #  pass
      max_abs = torch.max(torch.abs(bboxes[:,-1]))
      if not max_abs<=math.pi*0.5+ofs:
        print(f'\n\nERROR in check_bboxes: max_abs={max_abs} \n\n')
        pass
    else:
      if check_thickness:
        assert torch.all(bboxes[:,3] >= bboxes[:,4])
      if not torch.max(bboxes[:,-1]) <= math.pi+ofs:
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        pass
      assert torch.max(bboxes[:,-1]) <= math.pi+ofs
      assert torch.min(bboxes[:,-1]) >= 0-ofs

def test_rotation_order():
  import cv2
  np.set_printoptions(precision=3, suppress=True)
  vec_start = np.array([[1,0]], dtype=np.int32) * 200
  vec_end = np.array([[1,1]], dtype=np.int32) * 200
  angle = angle_from_vecs_to_vece_np(vec_start, vec_end, scope_id=2)
  print(f'angle: {angle}')

  img = np.zeros([512,512,3], dtype=np.uint8)
  z = 256
  img = cv2.line(img, (z,z), (vec_start[0,0]+z, vec_start[0,1]+z), (0,255,0), thickness=1)
  img = cv2.line(img, (z,z), (vec_end[0,0]+z, vec_end[0,1]+z), (0,255,0), thickness=1)
  cv2.imshow('img', img)
  cv2.waitKey(0)


  pass


def test_4corners():
  from tools.visual_utils import _show_objs_ls_points_ls, _show_3d_points_objs_ls
  from obj_geo_utils.obj_utils import OBJ_REPS_PARSE
  u = np.pi/180
  #XYDAsinAsinSin2Z0Z1 = np.array([
  #  [200, 300, 100, 0.707, 1, 0, 0, 0 ],
  #])


  XYZLgWsHA = np.array([
    [200, 300, 0, 100, 20, 0, -45*u ],
  ])
  XYDAsinAsinSin2Z0Z1 = OBJ_REPS_PARSE.encode_obj( XYZLgWsHA, 'XYZLgWsHA', 'XYDAsinAsinSin2Z0Z1' )
  n = XYDAsinAsinSin2Z0Z1.shape[0]

  Rect4CornersZ0Z1 = OBJ_REPS_PARSE.encode_obj(XYDAsinAsinSin2Z0Z1, 'XYDAsinAsinSin2Z0Z1', 'Rect4CornersZ0Z1').reshape(n, 4, 2)
  print('Rect4CornersZ0Z1\n', Rect4CornersZ0Z1)
  corners = Rect4CornersZ0Z1

  #corners = np.array(
  # [[219.131, 346.195],
  #  [180.869, 346.195],
  #  [180.869, 253.805],
  #  [219.131, 253.805]])[None,:,:]

  XYDAsinAsinSin2Z0Z1_1, rect_loss = four_corners_to_box( torch.from_numpy( corners ) )
  XYDAsinAsinSin2Z0Z1_1 = XYDAsinAsinSin2Z0Z1_1.numpy()

  XYDAsinAsinSin2Z0Z1_2, rect_loss = four_corners_to_box( torch.from_numpy( corners[:,[2,1,0,3]] ) )
  XYDAsinAsinSin2Z0Z1_2 = XYDAsinAsinSin2Z0Z1_2.numpy()

  XYDAsinAsinSin2Z0Z1_3, rect_loss = four_corners_to_box( torch.from_numpy( corners[:,[2,0,3,1]] ) )
  XYDAsinAsinSin2Z0Z1_3 = XYDAsinAsinSin2Z0Z1_3.numpy()
  err = np.max(np.abs(XYDAsinAsinSin2Z0Z1_3 - XYDAsinAsinSin2Z0Z1))
  print(f'err:{err}')

  _show_objs_ls_points_ls( (512,512), [XYDAsinAsinSin2Z0Z1_1], 'XYDAsinAsinSin2Z0Z1', points_ls=[Rect4CornersZ0Z1.reshape(-1, 2)], point_colors=['green'])
  _show_objs_ls_points_ls( (512,512), [XYDAsinAsinSin2Z0Z1_2], 'XYDAsinAsinSin2Z0Z1', points_ls=[Rect4CornersZ0Z1.reshape(-1, 2)], point_colors=['green'])
  _show_objs_ls_points_ls( (512,512), [XYDAsinAsinSin2Z0Z1_3], 'XYDAsinAsinSin2Z0Z1', points_ls=[Rect4CornersZ0Z1.reshape(-1, 2)], point_colors=['green'])
  #for i in range(4):
  #  _show_objs_ls_points_ls( (512,512), [XYDAsinAsinSin2Z0Z1_1], 'XYDAsinAsinSin2Z0Z1', points_ls=[Rect4CornersZ0Z1.reshape(-1, 2), Rect4CornersZ0Z1[:,i]], point_colors=['green', 'red'])

def test_4corners_1():
  from tools.visual_utils import _show_objs_ls_points_ls, _show_3d_points_objs_ls
  from obj_geo_utils.obj_utils import OBJ_REPS_PARSE
  u = np.pi/180
  XYZLgWsHA = np.array([
    [200, 300, 0, 100, 20, 0, -45*u ],
    [200, 200, 0, 200, 30, 0, 45*u ],
    [300, 300, 0, 200, 200, 0, 80*u ],
  ])
  n = XYZLgWsHA.shape[0]
  Rect4CornersZ0Z1 = OBJ_REPS_PARSE.encode_obj(XYZLgWsHA, 'XYZLgWsHA', 'Rect4CornersZ0Z1').reshape(n, 4, 2)
  XYDAsinAsinSin2Z0Z1, rect_loss = four_corners_to_box( torch.from_numpy( Rect4CornersZ0Z1 ) )
  XYDAsinAsinSin2Z0Z1 = XYDAsinAsinSin2Z0Z1.numpy()
  print('XYDAsinAsinSin2Z0Z1\n', XYDAsinAsinSin2Z0Z1)
  rect_loss = rect_loss.numpy()
  assert np.max(np.abs(rect_loss)) < 1e-5

  XYDAsinAsinSin2Z0Z1_c = OBJ_REPS_PARSE.encode_obj(XYZLgWsHA, 'XYZLgWsHA', 'XYDAsinAsinSin2Z0Z1')
  err = XYDAsinAsinSin2Z0Z1_c - XYDAsinAsinSin2Z0Z1
  merr = np.max(np.abs(err))
  print(f'err: {merr}\n{err}')
  if not merr < 1e-7:
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    pass
  pass

  for i in range(4):
    #_show_objs_ls_points_ls( (512,512), [XYZLgWsHA], 'XYZLgWsHA', points_ls=[Rect4CornersZ0Z1.reshape(-1, 2), Rect4CornersZ0Z1[:,i]], point_colors=['green', 'red'])
    _show_objs_ls_points_ls( (512,512), [XYDAsinAsinSin2Z0Z1], 'XYDAsinAsinSin2Z0Z1', points_ls=[Rect4CornersZ0Z1.reshape(-1, 2), Rect4CornersZ0Z1[:,i]], point_colors=['green', 'red'])
    _show_objs_ls_points_ls( (512,512), [XYDAsinAsinSin2Z0Z1_c], 'XYDAsinAsinSin2Z0Z1', points_ls=[Rect4CornersZ0Z1.reshape(-1, 2), Rect4CornersZ0Z1[:,i]], point_colors=['green', 'red'])
  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  pass

def test_sort_corners():
  from tools.visual_utils import _show_objs_ls_points_ls, _show_3d_points_objs_ls
  from obj_geo_utils.obj_utils import OBJ_REPS_PARSE
  u = np.pi/180

  XYZLgWsHA = np.array([
    [300, 300, 0, 200, 20, 0, -40*u ],
    [120, 120, 0, 100, 20, 0, 15*u ],
  ])
  n = XYZLgWsHA.shape[0]
  Rect4CornersZ0Z1 = OBJ_REPS_PARSE.encode_obj(XYZLgWsHA, 'XYZLgWsHA', 'Rect4CornersZ0Z1')
  gt_corners = Rect4CornersZ0Z1[:,:8].reshape(n, 4, 2)
  c_XYZLgWsHA = OBJ_REPS_PARSE.encode_obj(Rect4CornersZ0Z1, 'Rect4CornersZ0Z1', 'XYZLgWsHA')
  err = np.sum(np.abs(c_XYZLgWsHA - XYZLgWsHA))
  print(f'err: {err}')
  assert err < 1e-7

  corners_raw = np.array([
    [
    [320, 200],
    [240, 406],
    [ 400, 200],
    [250, 400],
  ],
    [
    [50, 100],
    [190, 120],
    [220, 100],
    [50, 150]
  ]
  ]).astype(np.float32)
  #corners_raw = corners_raw[:,[1,3,0,2]]
  corners_raw_t = torch.from_numpy(corners_raw)
  gt_corners_t = torch.from_numpy(gt_corners)
  #corners_sort, rect_loss = align_four_corners(corners_raw_t)
  corners_sort, rect_loss = align_four_corners(corners_raw_t, gt_corners=gt_corners_t)
  corners_sort = corners_sort.numpy()
  print(f'rect_loss: {rect_loss}')

  #pts_raw = np.concatenate([corners_raw, corners_raw], -1).reshape(-1,4)
  #pts_sort = np.concatenate([corners_sort, corners_sort], -1).reshape(-1,4)
  ids_raw = np.repeat(np.arange(4).reshape(-1,4), n, 0).reshape(-1)
  _show_objs_ls_points_ls( (512,512),  [Rect4CornersZ0Z1 ], 'Rect4CornersZ0Z1',
                          points_ls = [corners_raw.reshape(-1,2) , gt_corners.reshape(-1,2) ], point_scores_ls=[ids_raw, ids_raw], point_thickness=3 )
  _show_objs_ls_points_ls( (512,512),  [Rect4CornersZ0Z1], 'Rect4CornersZ0Z1',
                          points_ls = [corners_sort.reshape(-1,2)  , gt_corners.reshape(-1,2)], point_scores_ls=[ids_raw, ids_raw], point_thickness=3 )
  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  pass

if __name__ == '__main__':
  #test_4corners()
  test_sort_corners()

