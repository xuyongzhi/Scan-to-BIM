## April 2019 xyz
import torch, math
import numpy as np

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

def angle_dif(val0, val1, aim_scope_id):
    '''
      aim_scope_id 0:[-pi/2, pi/2]
    '''
    dif = val1 - val0
    if aim_scope_id == 0:
      dif = limit_period(dif, 0.5, math.pi)
    else:
      raise NotImplementedError
    return dif

def angle_with_x(vec_start, scope_id=0, debug=0):
  '''
   vec_start: [n,2/3]
   scope_id=0: [0,pi]
            1: (-pi/2, pi/2]

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
  y = -np.sin(angle)[:,None]
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
  vec_start = vec_start.float()
  vec_end = vec_end.float()

  norm_start = torch.norm(vec_start, dim=1, keepdim=True)
  norm_end = torch.norm(vec_end, dim=1, keepdim=True)
  #assert norm_start.min() > 1e-4 and norm_end.min() > 1e-4 # return nan
  vec_start = vec_start / norm_start
  vec_end = vec_end / norm_end
  if vec_start.dim() == 2:
    tmp = vec_start[:,0:1]*0
    vec_start = torch.cat([vec_start, tmp], 1)
    vec_end = torch.cat([vec_end, tmp], 1)
  cz = torch.cross( vec_start, vec_end, dim=1)[:,2]
  # sometimes abs(cz)>1 because of float drift. result in nan angle
  mask = (torch.abs(cz) > 1).to(vec_start.dtype)
  cz = cz * (1 - mask*1e-7)
  angle = torch.asin(cz)
  # cross is positive for anti-clock wise. change to clock-wise
  angle = -angle  # [-pi/2, pi/2]

  # check :angle or pi-angle
  cosa = torch.sum(vec_start * vec_end,dim=1)
  mask = (cosa >= 0).to(vec_end.dtype)
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
    print(f'input vec norm=0\n norm_end={norm_end}')
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
  points:[n,3]
  lines:[m,2,3]
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
    angle = limit_period(angle, 0.5, np.pi)
  else:
    raise NotImplementedError
  assert not np.any(np.isnan(angle))
  return angle

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

def test():
  import cv2
  np.set_printoptions(precision=3, suppress=True)
  vec_start = np.array([[1,0]], dtype=np.int32) * 200
  vec_end = np.array([[0,1]], dtype=np.int32) * 200
  angle = angle_from_vecs_to_vece_np(vec_start, vec_end, scope_id=2)

  img = np.zeros([512,512,3], dtype=np.uint8)
  z = 256
  img = cv2.line(img, (z,z), (vec_start[0,0]+z, vec_start[0,1]+z), (0,255,0), thickness=1)
  img = cv2.line(img, (z,z), (vec_end[0,0]+z, vec_end[0,1]+z), (0,255,0), thickness=1)
  cv2.imshow('img', img)
  cv2.waitKey(0)


  pass


if __name__ == '__main__':
  test()

