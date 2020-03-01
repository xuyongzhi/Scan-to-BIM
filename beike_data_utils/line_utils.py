# xyz
import numpy as np
import mmcv
from .geometric_utils import sin2theta_np
import cv2
from mmdet.debug_tools import show_img_with_norm, show_img_lines
import torch

def encode_line_rep(lines, obj_rep):
  '''
  lines: [n,4] or [n,2,2]
  lines_out : [n,4/5]

  The input lines are in the standard format of lines, which are two end points.
  The output format is based on obj_rep.

  Used in mmdet/datasets/pipelines/transforms.py /RandomLineFlip/line_flip
  '''
  assert obj_rep in ['close_to_zero', 'box_scope', 'line_scope', 'lscope_istopleft']
  if lines.ndim == 2:
    assert lines.shape[1] == 4
    lines = lines.copy().reshape(-1,2,2)
  else:
    assert lines.ndim == 3
    assert lines.shape[1] == 2
    assert lines.shape[2] == 2

  if obj_rep == 'close_to_zero':
      # the point with smaller x^2 + y^2 is the first one
      flag = np.linalg.norm(lines, axis=-1)
      swap = (flag[:,1] - flag[:,0]) < 0
      n = lines.shape[0]
      for i in range(n):
        if swap[i]:
          lines[i] = lines[i,[1,0],:]
      lines_out = lines.reshape(-1,4)

  elif obj_rep == 'box_scope' or obj_rep == 'line_scope':
      xy_min = lines.min(axis=1)
      xy_max = lines.max(axis=1)
      lines_out = np.concatenate([xy_min, xy_max], axis=1)

  elif obj_rep == 'lscope_istopleft':
      xy_min = lines.min(axis=1)
      xy_max = lines.max(axis=1)
      centroid = (xy_min + xy_max) / 2
      lines_0 = lines - centroid.reshape(-1,1,2)
      top_ids = lines_0[:,:,1].argmin(axis=-1)
      nb = lines_0.shape[0]

      tmp = np.arange(nb)
      top_points = lines_0[tmp, top_ids]
      vec_start = np.array([[0, -1]] * nb, dtype=np.float32).reshape(-1,2)
      istopleft_0 = sin2theta_np( vec_start, top_points).reshape(-1,1)

      istopleft = istopleft_0

      lines_out = np.concatenate([xy_min, xy_max, istopleft], axis=1)
      pass
  else:
    raise NotImplemented
  return lines_out.astype(np.float32)
  pass

def decode_line_rep(lines, obj_rep):
  '''
  lines: [n,4/5]
  lines_out: [n,4]

  The input lines are in representation of obj_rep.
  The outout is standard lines in two end-points.
  '''
  assert obj_rep in ['close_to_zero', 'box_scope', 'line_scope',\
                     'lscope_istopleft']
  assert lines.ndim == 2
  if lines.shape[0] == 0:
    return lines
  if obj_rep == 'lscope_istopleft':
    assert lines.shape[1] == 5
    istopleft = (lines[:,4:5] >= 0).astype(lines.dtype)
    end_pts = lines[:,:4] * istopleft +  lines[:,[0,3,2,1]] * (1-istopleft)
  else:
    raise NotImplemented
  pass
  return end_pts

def decode_line_rep_th(lines, obj_rep):
  '''
  lines: [batch_size,4/5,w,h] or [n,4/5]   4:x,y,x,y    5: x,y,x,y,r
    lines.shape[1] is the channel
  lines_out: [batch_size,4,w,h]

  The input lines are in representation of obj_rep.
  The outout is standard lines in two end-points.
  '''
  assert obj_rep in ['close_to_zero', 'box_scope', 'line_scope',\
                     'lscope_istopleft']
  assert lines.dim() == 2 or lines.dim() == 4
  if lines.shape[0] == 0:
    return lines

  if obj_rep == 'lscope_istopleft':
    assert lines.shape[1] == 5
    istopleft = (lines[:,4:5,...] >= 0).type(lines.dtype)
    end_pts = lines[:,:4,...] * istopleft +  lines[:,[0,3,2,1],...] * (1-istopleft)
  else:
    raise NotImplemented
  pass
  return end_pts

def add_cross_in_img(img):
  h, w = img.shape[:2]
  assert h%2 == 0
  assert w%2 == 0
  img[h//2-1 : h//2+1, :, [1,2]] = 255
  img[:, w//2 - 1 : w//2+1, [1,2]] = 255

  #img[h//2-1 : h//2, :, [1,2]] = 255
  #img[:, w//2 - 1 : w//2, [1,2]] = 255

def add_cross_in_lines(lines, img_shape):
  h, w = img_shape
  cross0 = np.array([
                    [ 5, h//2 - 1, w-5, h//2-1, 0 ],
                    [w//2 - 1, 5, w//2 -1, h-5, 0 ],
                    ])
  cross1 = np.array([
                    [ 5, h//2, w-5, h//2, 0 ],
                    [w//2, 5, w//2, h-5, 0 ]
                    ])

  cross2 = np.array([
                    [ 5, h//2+2, w-5, h//2+2, 0 ],
                    [w//2+2, 5, w//2+2, h-5, 0 ]
                    ])
  #lines = cross
  lines = np.concatenate([lines, cross0, cross1], axis=0)
  return lines


def transfer_lines(lines, obj_rep, img_shape, angle, offset):
  '''
  angle: clock-wise is positive
  '''
  scale = 1
  h, w = img_shape
  assert h%2 == 0
  assert w%2 == 0
  center = ((w - 1) * 0.5, (h - 1) * 0.5 )
  matrix = cv2.getRotationMatrix2D(center, -angle, scale)
  n = lines.shape[0]

  lines_2endpts = decode_line_rep(lines, obj_rep).reshape(n,2,2)

  ones = np.ones([n,2,1], dtype=lines.dtype)
  tmp = np.concatenate([lines_2endpts, ones], axis=2).reshape([n*2, 3])
  lines_2pts_r = np.matmul( tmp, matrix.T ).reshape([n,2,2])
  lines_2pts_r[:,:,0] += offset[0]
  lines_2pts_r[:,:,1] += offset[1]
  lines_rotated = encode_line_rep(lines_2pts_r, obj_rep)
  return lines_rotated

def rotate_lines_img(lines, img, angle,  obj_rep, check_by_cross=False):
  assert img.ndim == 3
  assert lines.ndim == 2
  assert lines.shape[1] == 5

  img_shape = img.shape[:2]
  if check_by_cross:
    lines = add_cross_in_lines(lines, img_shape)

  n = lines.shape[0]
  if n == 0:
    return lines, img
  lines_2endpts = decode_line_rep(lines, obj_rep).reshape(n,2,2)

  h, w = img_shape
  assert h%2 == 0
  assert w%2 == 0
  center = ((w - 1) * 0.5, (h - 1) * 0.5)
  scale = 1.0
  matrix = cv2.getRotationMatrix2D(center, -angle, scale)

  ones = np.ones([n,2,1], dtype=lines.dtype)
  tmp = np.concatenate([lines_2endpts, ones], axis=2)
  lines_2pts_r = np.matmul( tmp, matrix.T )

  # (1) rotate the lines
  lines_rotated = encode_line_rep(lines_2pts_r, obj_rep)

  # (2) move the lines to center
  x_min =  lines_rotated[:,[0,2]].min()
  x_max =  lines_rotated[:,[0,2]].max()
  y_min =  lines_rotated[:,[1,3]].min()
  y_max =  lines_rotated[:,[1,3]].max()
  x_cen = (x_min + x_max) / 2
  y_cen = (y_min + y_max) / 2

  x_offset = np.floor( x_cen - (w-1)/2 ) * 1
  y_offset = np.floor( y_cen - (h-1)/2 ) * 1

  lines_rotated[:, [0,2]] -= x_offset
  lines_rotated[:, [1,3]] -= y_offset

  # (3) scale the lines to fit the image size
  # Move before scaling can increase the scale ratio
  x_min_ =  lines_rotated[:,[0,2]].min()
  x_max_ =  lines_rotated[:,[0,2]].max()
  y_min_ =  lines_rotated[:,[1,3]].min()
  y_max_ =  lines_rotated[:,[1,3]].max()

  gap_x0 = 0 - x_min_
  gap_y0 = 0 - y_min_
  gap_x1 = x_max_ - (w-1)
  gap_y1 = y_max_ - (h-1)
  gap_x = np.ceil(np.array([gap_x0, gap_x1, 0]).max())
  gap_y = np.ceil(np.array([gap_y0, gap_y1, 0]).max())
  scale_x = w / (w+gap_x*2.0)
  scale_y = h / (h+gap_y*2.0)
  scale = min(scale_x, scale_y)
  scale = np.floor(scale * 100)/100.0
  if scale < 1:
    scale = scale - 0.03
  #print(f'scale: {scale}')

  center = np.repeat(center, 2)
  lines_rotated[:,:4] = (lines_rotated[:,:4] - center) * scale + center

  # (4) rotate the image (do not scale at this stage)
  if check_by_cross:
    add_cross_in_img(img)


  img_big = np.zeros([h*2,w*2,img.shape[2]], dtype=lines.dtype)
  img_big[int(h/2):int(h/2*3), int(w/2):int(w/2*3)] = img
  img_r = mmcv.imrotate(img_big, angle, scale=scale)

  # (5) Move the image
  new_img = np.zeros([h,w], dtype=lines.dtype)
  h1,w1 = img_r.shape[:2]
  region = np.array([(h1-h)/2 + x_offset * scale,
                     (w1-w)/2 + y_offset * scale,
                     (h1-h)/2 + h -1 + x_offset * scale,
                     (w1-w)/2 + w -1 + y_offset * scale])
  region_int = region.astype(np.int32)
  new_img = mmcv.imcrop(img_r, region_int)
  assert new_img.shape[:2] == img_shape

  # rotate the surface normal
  new_img[:,:,[1,2]] = np.matmul( new_img[:,:,[1,2]], matrix[:,:2].T )

  #show_img_with_norm(img)
  #show_img_with_norm(new_img)

  lines_rotated = lines_rotated.astype(np.float32)
  new_img = new_img.astype(np.float32)
  #show_img_lines(img[:,:,:3]*255, lines)
  #show_img_lines(new_img[:,:,:3], lines_rotated)
  return  lines_rotated, new_img





