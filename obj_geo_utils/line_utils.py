# xyz
import numpy as np
import mmcv
from .geometric_utils import sin2theta_np, angle_with_x_np
import cv2
from tools.debug_utils import _show_img_with_norm, _show_lines_ls_points_ls, _show_3d_points_bboxes_ls, _draw_lines_ls_points_ls
import torch




#--------------------------------------------------------------------------------

def encode_line_rep(lines, obj_rep):
  '''
  lines: [n,4] or [n,2,2]
  lines_out : [n,4/5]

  The input lines are in the standard format of lines, which are two end points.
  The output format is based on obj_rep.

  Used in mmdet/datasets/pipelines/transforms.py /RandomLineFlip/line_flip
  '''
  assert obj_rep in ['std_2p' ,'close_to_zero', 'box_scope', 'line_scope', 'lscope_istopleft']
  if lines.ndim == 2:
    assert lines.shape[1] == 4
    lines = lines.copy().reshape(-1,2,2)
  else:
    assert lines.ndim == 3
    assert lines.shape[1] == 2
    assert lines.shape[2] == 2

  if obj_rep == 'std_2p':
    lines_out = lines
  elif obj_rep == 'close_to_zero':
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
  assert obj_rep in ['std_2p','close_to_zero', 'box_scope', 'line_scope',\
                     'lscope_istopleft']
  if lines.shape[0] == 0:
    return lines
  if obj_rep == 'std_2p':
    assert lines.ndim == 3
    assert lines.shape[1:] == (2,2)
    std_lines = lines
  elif obj_rep == 'lscope_istopleft':
    assert lines.ndim == 2
    assert lines.shape[1] == 5
    istopleft = (lines[:,4:5] >= 0).astype(lines.dtype)
    std_lines = lines[:,:4] * istopleft +  lines[:,[0,3,2,1]] * (1-istopleft)
  else:
    raise NotImplemented
  pass
  return std_lines

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


def m_transform_lines(lines, matrix, obj_rep):
  '''
  lines:  [n,5]
  matrix: [4,4]
  '''
  assert matrix.shape == (4,4)
  n = lines.shape[0]
  lines_2endpts = decode_line_rep(lines, obj_rep).reshape(n,2,2)

  ones = np.ones([n,2,2], dtype=lines.dtype)
  tmp = np.concatenate([lines_2endpts, ones], axis=2).reshape([n*2, 4])
  lines_2pts_r = (tmp @ matrix.T)[:,:2].reshape([n,2,2])
  lines_rotated = encode_line_rep(lines_2pts_r, obj_rep)
  return lines_rotated

def transfer_lines(lines, obj_rep, img_shape, angle, offset):
  '''
  lines: [n,5]
  angle: clock-wise is positive
  '''
  scale = 1
  h, w = img_shape
  #assert h%2 == 0
  #assert w%2 == 0
  center = ((w - 0) * 0.5, (h - 0) * 0.5 )
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
  scale = min(scale_x, scale_y) * 0.95
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

  #_show_img_with_norm(img)
  #_show_img_with_norm(new_img)

  lines_rotated = lines_rotated.astype(np.float32)
  new_img = new_img.astype(np.float32)
  #_show_lines_ls_points_ls(img[:,:,:3]*255, lines)
  #_show_lines_ls_points_ls(new_img[:,:,:3], lines_rotated)
  return  lines_rotated, new_img


def gen_corners_from_lines_th(lines, labels, obj_rep):
    lines0 = decode_line_rep_th(lines, obj_rep)
    labels_1 = labels.reshape(-1,1).to(lines.dtype)
    lines1 = torch.cat([lines0[:,0:2], labels_1, lines0[:,2:4], labels_1], dim=1)
    lines1 = lines1.reshape(-1,3)
    corners_uq = torch.unique(lines1, sorted=False, dim=0)
    lines_out = corners_uq[:,:2]
    labels_out = corners_uq[:,2].to(labels.dtype)

    if 0:
      n0 = lines.shape[0]
      n1 = lines_out.shape[0]
      print(f'{n0} -> {n1}')
      from mmdet.debug_tools import show_lines
      show_lines(lines.cpu().data.numpy(), (512,512), points=lines_out)
    return lines_out, labels_out

def gen_corners_from_lines_np(lines, labels, obj_rep, flag=''):
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

    lines0 = decode_line_rep(lines, obj_rep)
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
    corners1 = round_positions(corners1, 100)
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

def optimize_graph(lines_in, scores, labels, obj_rep, opt_graph_cor_dis_thr):
  '''
    lines_in: [n,5]
  '''
  num_line = lines_in.shape[0]
  lab_sco_lines = np.concatenate([labels.reshape(num_line,1), scores.reshape(num_line,1)], axis=1)
  corners_in, lab_sco_cors, corIds_per_line, num_cor_uq_org = gen_corners_from_lines_np(lines_in, lab_sco_lines, obj_rep)
  labels_cor = lab_sco_cors[:,0]
  scores_cor = lab_sco_cors[:,1]
  corners_labs = np.concatenate([corners_in, labels_cor.reshape(-1,1)*100], axis=1)
  corners_merged, cor_scores_merged, cor_labels_merged = merge_corners(corners_labs, scores_cor, opt_graph_cor_dis_thr=opt_graph_cor_dis_thr)
  corners_merged = round_positions(corners_merged, 100)
  corners_merged_per_line = corners_merged[corIds_per_line]
  line_labels_merged = cor_labels_merged[corIds_per_line][:,0].astype(np.int32)
  line_scores_merged = cor_scores_merged[corIds_per_line].mean(axis=1)[:,None]

  lines_merged = encode_line_rep(corners_merged_per_line, obj_rep)

  #lines_merged = lines_merged[0:5]
  #line_scores_merged = line_scores_merged[0:5]
  #line_labels_merged = line_labels_merged[0:5]

  if 0:
    corners_uq, unique_indices, inds_inverse = np.unique(corners_merged, axis=0, return_index=True, return_inverse=True)
    num_cor_org = corners_in.shape[0]
    num_cor_merged = corners_uq.shape[0]
    print(f'\ncorner num: {num_cor_org} -> {num_cor_merged}')

    #_show_lines_ls_points_ls((512,512), [lines_in], [corners_merged], ['white'], 'random')
    #_show_lines_ls_points_ls((512,512), [lines_in, lines_merged], [], ['white','green'])

    #det_corners, cor_scores, det_cor_ids_per_line, num_cor_uq_merged_check = gen_corners_from_lines_np(lines_merged, line_labels_merged, 'lscope_istopleft', flag='debug')
    #print(f'num_cor_uq_merged_check: {num_cor_uq_merged_check}')

    check_lines = decode_line_rep(lines_merged, obj_rep)
    check_corners = check_lines.reshape(-1,2,2)
    cor_err = corners_merged_per_line - check_corners
    cor_loc_check = np.abs(cor_err).max() < 1e-3
    print(f'cor_loc_check: {cor_loc_check}')
    print(lines_merged[0:5])

  return lines_merged, line_scores_merged, line_labels_merged

def merge_corners(corners_0, scores_0, opt_graph_cor_dis_thr=3):
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
    weights = scores_0[ids_i] / scores_0[ids_i].sum()
    merged_sco = ( scores_0[ids_i] * weights).sum()
    merged_cor = ( corners_0[ids_i] * weights[:,None]).sum(axis=0)
    corners_1.append(merged_cor)
    scores_1.append(merged_sco)
    pass
  corners_1 = np.array(corners_1).reshape((-1, corners_0.shape[1]))
  scores_merged = np.array(scores_1)
  labels_merged = corners_1[:,2]
  corners_merged = corners_1[:,:2]
  return corners_merged, scores_merged, labels_merged

def round_positions(data, scale=100):
  return np.round(data*scale)/scale

def lines2d_to_bboxes3d(lines, line_obj_rep='lscope_istopleft', height=60, thickness=1):
  '''
  lines:  [n,5]
  bboxes: [n,9] [center, size, angle]
  '''
  import open3d as o3d
  from  beike_data_utils.geometric_utils import angle_with_x_np
  assert line_obj_rep == 'lscope_istopleft'
  assert lines.ndim == 2
  assert lines.shape[1] == 5

  n = lines.shape[0]
  lines_std = decode_line_rep(lines, line_obj_rep).reshape(n,2,2)
  center2d = lines_std.mean(axis=1)
  vec_rotation = lines_std[:,0] - center2d
  z_angles = -angle_with_x_np(vec_rotation, scope_id=0)

  length = np.linalg.norm( lines_std[:,1] - lines_std[:,0] , axis=1)

  center = np.zeros([lines.shape[0], 3], dtype=lines.dtype)
  center[:,:2] = center2d
  center[:,2] = height/2
  extent = np.zeros([lines.shape[0], 3], dtype=lines.dtype)
  extent[:,0] = length
  extent[:,1] = thickness
  extent[:,2] = height
  angles = np.zeros([lines.shape[0], 3], dtype=lines.dtype)
  angles[:,2] = z_angles

  bboxes3d = np.concatenate([center, extent, angles], axis=1)
  #_show_3d_points_bboxes_ls(bboxes_ls = [bboxes3d], box_oriented=True)
  return bboxes3d

  matrixes = []
  for a in theta:
    axis_angle = np.array([0,0,a]).reshape([3,1])
    matrix = o3d.geometry.get_rotation_matrix_from_yxz(axis_angle)
    matrixes.append( matrix )
  matrixes = np.array(matrixes)
  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  pass


def getOrientedLineRectSubPix(img, line, obj_rep, length_aug=-5, thickness_aug=40):
  '''
  img: [h,w] [h,w,1/3]
  line: [5]
  '''
  debug  = 0
  assert line.shape == (5,)
  assert img.shape[:2] == (512,512)
  if debug:
    img = _draw_lines_ls_points_ls(img, [line.reshape(-1,5)])
  h, w = img.shape[:2]

  #angle = line[-1]
  line = decode_line_rep(line[None, :], obj_rep)[0]
  points = line[:4].reshape(2,2)
  x, y = points.mean(axis=0).astype(np.int)

  # move to center first
  xofs, yofs = w/2-x, h/2-y
  M_mc = np.array([[1,0, xofs], [0,1,yofs]])

  dir_v = points[1] - points[0]
  angle = angle_with_x_np( dir_v.reshape(1,2), 1 )[0]
  angle = -angle * 180 / np.pi

  M = cv2.getRotationMatrix2D( (w/2,h/2), angle, 1 )
  img_1 = cv2.warpAffine(img, M_mc, (w,h))
  img_rot = cv2.warpAffine(img_1, M, (w,h))
  tmp = np.concatenate([points, np.ones([2,1])], axis=1)
  points_1 = tmp @ M_mc.T
  tmp = np.concatenate([points_1, np.ones([2,1])], axis=1)
  points_rot = tmp @ M.T

  x,y = points_rot.mean(axis=0)
  center = (x,y)
  length = np.linalg.norm(points_rot[1] - points_rot[0]) + length_aug
  length = max(length, 5)
  thickness = thickness_aug
  size = (int(length), int(thickness))

  roi = cv2.getRectSubPix(img_rot, size, center)

  if debug:

    print(angle)
    _show_lines_ls_points_ls(img, points_ls=[points])
    _show_lines_ls_points_ls(img_rot, points_ls=[points_rot])
    _show_lines_ls_points_ls(roi)
    pass
  return roi

