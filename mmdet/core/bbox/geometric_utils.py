## April 2019 xyz
import torch, math
import numpy as np

def limit_period(val, offset, period):
  '''
    [0, pi]: offset=0, period=pi
    [-pi/2, pi/2]: offset=0.5, period=pi
    [-pi, 0]: offset=1, period=pi
  '''
  return val - torch.floor(val / period + offset) * period

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
  return angle_from_vecs_to_vece(vec_x, vec_start, scope_id, debug)


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

   clock wise is positive
   angle: [n]
  '''
  assert vec_start.dim() == 2 and  vec_end.dim() == 2
  assert (vec_start.shape[0] == vec_end.shape[0]) or vec_start.shape[0]==1 or vec_end.shape[0]==1
  assert vec_start.shape[1] == vec_end.shape[1] # 2 or 3
  vec_start = vec_start.float()
  vec_end = vec_end.float()

  norm0 = torch.norm(vec_start, dim=1, keepdim=True)
  norm1 = torch.norm(vec_end, dim=1, keepdim=True)
  #assert norm0.min() > 1e-4 and norm1.min() > 1e-4 # return nan
  vec_start = vec_start / norm0
  vec_end = vec_end / norm1
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

