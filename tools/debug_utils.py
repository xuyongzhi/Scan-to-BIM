import open3d as o3d
import torch
import numpy as np
import mmcv
from mmcv.image import imread, imwrite
import cv2
from MinkowskiEngine import SparseTensor

from .color import color_val, get_random_color, label2color

ADD_FRAME = 1

def _show_tensor_ls_shapes(tensor_ls, pre='', i=0):
  if isinstance(tensor_ls, torch.Tensor):
    shape = tensor_ls.shape
    print(f'{pre} {i} \t{shape}')
  else:
    assert isinstance(tensor_ls, list) or isinstance(tensor_ls, tuple)
    pre += '  '
    for i,tensor in enumerate(tensor_ls):
      _show_tensor_ls_shapes(tensor, pre, i)


def _show_sparse_ls_shapes(tensor_ls, pre='', i=0):
  if isinstance(tensor_ls, SparseTensor):
    tensor = tensor_ls
    coords = tensor.coords
    feats = tensor.feats
    c_shape = [*coords.shape]
    f_shape = [*feats.shape]
    t_stride = [*tensor.tensor_stride]
    min_coords = [*coords.min(dim=0)[0].numpy()]
    max_coords = [*coords.max(dim=0)[0].numpy()]
    print(f'{pre} {i} \tcoords:{c_shape} \tfeats:{f_shape} \tstride:{t_stride} \tcoords scope: {min_coords} - {max_coords}')
    pass
  else:
    assert isinstance(tensor_ls, list) or isinstance(tensor_ls, tuple)
    pre += '  '
    for i,tensor in enumerate(tensor_ls):
      _show_sparse_ls_shapes(tensor, pre, i)
  pass


#-3d------------------------------------------------------------------------------
def _show_3d_points_bboxes_ls(points_ls=None, point_feats=None,
             bboxes_ls=None, b_colors='random', box_oriented=False):
  show_ls = []
  if points_ls is not None:
    assert isinstance(points_ls, list)
    n = len(points_ls)
    if point_feats is not None:
      assert isinstance(point_feats, list)
    else:
      point_feats = [None] * n
    for points, feats in zip(points_ls, point_feats):
      pcd = _make_pcd(points, feats)
      show_ls.append(pcd)

  if bboxes_ls is not None:
    assert isinstance(bboxes_ls, list)
    if not isinstance(b_colors, list):
      b_colors = [b_colors] * len(bboxes_ls)
    for i,bboxes in enumerate(bboxes_ls):
      bboxes_o3d = _make_bboxes_o3d(bboxes, box_oriented, b_colors[i])
      show_ls = show_ls + bboxes_o3d

  if points_ls is not None:
    center = points_ls[0][:,:3].mean(axis=0)
  else:
    center = [0,0,0]
  center = [0,0,0]
  mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6 * 100, origin=center)
  if ADD_FRAME:
    show_ls.append(mesh_frame)

  o3d.visualization.draw_geometries( show_ls )

def _make_bboxes_o3d(bboxes, box_oriented, color):
  assert bboxes.ndim==2
  if box_oriented:
    assert bboxes.shape[1] == 9
  else:
    assert bboxes.shape[1] == 6
  bboxes_ = []
  for i,bbox in enumerate(bboxes):
    bboxes_.append( _make_bbox_o3d(bbox, box_oriented, color) )
  return bboxes_

def _make_bbox_o3d(bbox, box_oriented, color):
  assert bbox.ndim==1
  if not box_oriented:
    assert bbox.shape[0] == 6
    bbox_ = o3d.geometry.AxisAlignedBoundingBox(bbox[:3], bbox[3:6])
  else:
    assert bbox.shape[0] == 9
    center = bbox[:3]
    extent = bbox[3:6]
    axis_angle = bbox[6:9]
    R = o3d.geometry.get_rotation_matrix_from_yxz(axis_angle)
    bbox_ = o3d.geometry.OrientedBoundingBox( center, R, extent )
  c = _get_color(color)
  bbox_.color = c
  return bbox_

def _make_pcd(points, feats=None):
    if points.shape[1] == 6 and feats is None:
      feats = points[:,3:6]
      points = points[:,:3]
    assert points.shape[1] == 3
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if feats is not None:
      assert feats.shape[0] == points.shape[0]
      feats = feats.reshape(points.shape[0], -1)
      if  feats.shape[1] == 3:
        if feats.max() > 1:
          feats = feats / 255
        pcd.colors = o3d.utility.Vector3dVector(feats)
      elif feats.shape[1] == 1:
        labels = feats
        pcd.colors = o3d.utility.Vector3dVector( label2color(labels) / 255)
    return pcd


#-2d------------------------------------------------------------------------------
def _get_color(color_str):
  assert isinstance( color_str, str), print(color_str)
  if color_str == 'random':
    c = get_random_color()
  else:
    c = color_val(color_str)
  return c

def _read_img(img_in):
  '''
  img_in: [h,w,3] or [h,w,1], or [h,w] or (h_size, w_size)
  img_out: [h,w,3]
  '''
  if isinstance(img_in, tuple):
    assert len(img_in) == 2
    img_size = img_in + (3,)
    img_out = np.zeros(img_size, dtype=np.float32)
  else:
    assert isinstance(img_in, np.ndarray)
    if img_in.ndim == 2:
      img_in = np.expand_dims(img_in, 2)
    assert img_in.ndim == 3
    if img_in.shape[2] == 1:
      img_in = np.tile(img_in, (1,1,3))
    img_out = img_in.copy()

    if img_out.max() <= 1:
      img_out *= 255
    img_out = img_out.astype(np.uint8)
  return img_out

def _show_lines_ls_points_ls(img, lines_ls=None, points_ls=None,
    line_colors='green', point_colors='red', out_file=None, line_thickness=1,
    point_thickness=1, only_save=False, box=False):
  '''
  img: [h,w,3] or [h,w,1], or [h,w] or (h_size, w_size)
  '''
  img = _draw_lines_ls_points_ls(img, lines_ls, points_ls, line_colors, point_colors, out_file, line_thickness, point_thickness, box=box)
  if not only_save:
    mmcv.imshow(img)

def _draw_lines_ls_points_ls(img, lines_ls, points_ls=None, line_colors='green', point_colors='red', out_file=None, line_thickness=1, point_thickness=1, box=False):
  if lines_ls is not None:
    assert isinstance(lines_ls, list)
    if lines_ls is not None and isinstance(line_colors, str):
      line_colors = [line_colors] * len(lines_ls)
    if not isinstance(line_thickness, list):
      line_thickness = [line_thickness] * len(lines_ls)

  if points_ls is not None:
    assert isinstance(points_ls, list)
    if points_ls is not None and isinstance(point_colors, str):
      point_colors = [point_colors] * len(points_ls)
    if not isinstance(point_thickness, list):
      point_thickness = [point_thickness] * len(points_ls)

  img = _read_img(img)
  if lines_ls is not None:
    for i, lines in enumerate(lines_ls):
      img = _draw_lines(img, lines, line_colors[i], line_thickness[i], box=box)

  if points_ls is not None:
    for i, points in enumerate(points_ls):
      img = _draw_points(img, points, point_colors[i], point_thickness[i])
  if out_file is not None:
    mmcv.imwrite(img, out_file)
    print('\n',out_file)
  return img

def _draw_lines(img, lines, color, line_thickness=1, font_scale=0.5, text_color='green', box=False):
    '''
    img: [h,w,3]
    lines: [n,5/6]
    color: 'red'
    '''
    from configs.common import OBJ_LEGEND
    from beike_data_utils.line_utils import decode_line_rep
    assert lines.ndim == 2
    assert lines.shape[1]==5 or lines.shape[1]==6
    if lines.shape[1] == 6:
      scores = lines[:,5]
      lines = lines[:,:5]
    else:
      scores = None
    rotations = lines[:,4]
    text_color = _get_color(text_color)

    lines = decode_line_rep(lines, 'lscope_istopleft').reshape(-1,2,2)
    for i in range(lines.shape[0]):
        s, e = np.round(lines[i]).astype(np.int32)
        c = _get_color(color)
        if not box:
          cv2.line(img, (s[0], s[1]), (e[0], e[1]), c, thickness=line_thickness)
        else:
          cv2.rectangle(img, (s[0], s[1]), (e[0], e[1]), c, thickness=line_thickness)


        #label_text = class_names[label] if class_names is not None else 'cls {}'.format(label)
        label_text = ''
        if OBJ_LEGEND == 'score':
          if scores is not None:
            label_text += '{:.01f}'.format(scores[i]) # score
        else:
          label_text += '{:.01f}'.format(rotations[i]) # rotation
        m = np.round(((s+e)/2)).astype(np.int32)
        if label_text != '':
          cv2.putText(img, label_text, (m[0]-4, m[1] - 4),
                      cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
    return img

def _draw_points(img, points, color, point_thickness):
    points = np.round(points).astype(np.int32)
    for i in range(points.shape[0]):
      p = points[i]
      c = _get_color(color)
      cv2.circle(img, (p[0], p[1]), 2, c, thickness=point_thickness)
    return img

def _show_det_lines(img, lines, labels, class_names=None, score_thr=0,
                   line_color='green', text_color='green', thickness=2,
                   font_scale=0.5,show=True, win_name='', wait_time=0,
                   out_file=None, key_points=None, point_color='red', scores=None):
  '''
  img: [h,w,3]
  lines: [n,6]
  labels: [n]
  scores: [n] or None
  Use lines[:,-1] as score when scores is None, otherwise use scores as the score for filtering lines.
  Always show lines[:,-1] in the image.
  '''
  from configs.common import OBJ_LEGEND
  assert lines.ndim == 2
  assert lines.shape[1] == 6
  assert labels.ndim == 1
  assert labels.shape[0] == lines.shape[0]
  if key_points is not None:
    assert key_points.shape[0]== lines.shape[0]


  if score_thr > 0:
    scores = scores.reshape(-1)
    if scores is None:
      scores = lines[:,-1]
    inds = scores > score_thr
    lines = lines[inds, :]
    labels = labels[inds]
    if key_points is not None:
      key_points = key_points[inds, :]
  key_points_ls = [key_points] if key_points is not None else None
  _show_lines_ls_points_ls(img, [lines], key_points_ls, [line_color],
        [point_color], line_thickness=thickness, out_file=out_file, only_save=True)

#-------------------------------------------------------------------------------



def imshow_bboxes_random_colors(img, bboxes):
    color_lib = ['red', 'green', 'blue', 'cyan', 'yellow', 'magenta']
    ids = np.randint(len(color_lib))

def imshow_bboxes_ref(img, bboxes0, bboxes1):
    bboxes = np.concatenate([bboxes0, bboxes1], 0)
    n0 = bboxes0.shape[0]
    n1 = bboxes1.shape[0]
    colors = ['red']*n0 + ['green']*n1
    mmcv.imshow_bboxes(img, bboxes, colors)

def draw_points(img, points_list, colors_list='red'):
  if not isinstance(points_list, list):
    points_list = [points_list]
  if not isinstance(colors_list, list):
    colors_list = ['red'] * len(points_list)
  for k in range(len(points_list)):
    for i in range(points_list[k].shape[0]):
      p = points_list[k][i]
      c = color_val(colors_list[k])
      cv2.circle(img, (p[0], p[1]), 2, c, thickness=1)
  return img

def show_lines(lines, img_size, points=None, lines_ref=None, name='out.png'):
  img = np.zeros(img_size + (3,), dtype=np.uint8)
  if lines_ref is not None:
    colors = ['green'] * lines_ref.shape[0] + ['red'] * lines.shape[0]
    lines = np.concatenate([lines_ref, lines], axis=0)
  else:
    colors = ['random']
  show_img_lines(img, lines, points=points, colors=colors, name=name)

def imshow(img, win_name='', wait_time=0):
    """Show an image.
    Args:
        img (str or ndarray): The image to be displayed.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
    """
    cv2.imshow(win_name, imread(img))
    if wait_time == 0:  # prevent from hangning if windows was closed
        while True:
            ret = cv2.waitKey(1)

            closed = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1
            # if user closed window or if some key pressed
            if closed or ret != -1:
                break
    else:
        ret = cv2.waitKey(wait_time)

def show_det_lines_1by1(img, lines, labels, class_names=None, score_thr=0,
                   line_color='green', text_color='green', thickness=1,
                   font_scale=0.5,show=True, win_name='', wait_time=0,
                   out_file=None, key_points=None, point_color='red'):
  img = img.copy()
  if score_thr > 0:
    assert lines.shape[1] == 6
    scores = lines[:,-1]
    inds = lines[:,-1] > score_thr
    lines = lines[inds, :]
    labels = labels[inds]
    if key_points is not None:
      key_points = key_points[inds, :]

  size = img.shape[0:2]
  show = 0
  if out_file is not None:
    avi_out_file = out_file.replace('png', 'avi')
  else:
    avi_out_file = 'lines_detected.avi'
  out = cv2.VideoWriter(avi_out_file, cv2.VideoWriter_fourcc(*'DIVX'), 2, size)
  n = lines.shape[0]
  for i in range(n):
    img0 = img.copy()
    if key_points is not None:
      kpi = key_points[i:i+1]
    else:
      kpi = None
    show_det_lines(img0, lines[i:i+1], labels[i:i+1], class_names, score_thr,
                   line_color, text_color, thickness, font_scale, show, win_name,
                   wait_time, None, kpi, point_color)
    out.write(img0.astype(np.uint8))
    img0 = img.copy()

  show_det_lines(img, lines, labels, class_names, score_thr,
                  line_color, text_color, thickness, font_scale, show, win_name,
                  wait_time, out_file, key_points, point_color)
  out.write(img.astype(np.uint8))

  out.release()
  pass

def show_det_lines(img, lines, labels, class_names=None, score_thr=0,
                   line_color='green', text_color='green', thickness=1,
                   font_scale=0.5,show=True, win_name='', wait_time=0,
                   out_file=None, key_points=None, point_color='red', scores=None):
  '''
  img: [h,w,3]
  lines: [n,6]
  labels: [n]
  scores: [n] or None
  Use lines[:,-1] as score when scores is None, otherwise use scores as the score for filtering lines.
  Always show lines[:,-1] in the image.
  '''
  from configs.common import OBJ_LEGEND
  assert lines.ndim == 2
  assert lines.shape[1] == 6 or lines.shape[1] == 5
  assert labels.ndim == 1
  assert labels.shape[0] == lines.shape[0]
  if key_points is not None:
    assert key_points.shape[0]== lines.shape[0]

  img = imread(img.copy())

  if score_thr > 0:
    assert lines.shape[1] == 6
    if scores is None:
      inds = lines[:,-1] > score_thr
    else:
      inds = scores > score_thr
    lines = lines[inds, :]
    labels = labels[inds]
    if key_points is not None:
      key_points = key_points[inds, :]

  line_color = color_val(line_color)
  text_color = color_val(text_color)
  point_color = color_val(point_color)

  i = -1
  for line, label in zip(lines, labels):
    i += 1
    istopleft = line[4] >= 0
    if not istopleft:
      line[0], line[2] = line[2], line[0]
    line_int = line.astype(np.int32)
    s = line_int[0:2]
    e = line_int[2:4]

    line_color = get_random_color()
    cv2.line(img, (s[0], s[1]), (e[0], e[1]), line_color, thickness=thickness)

    if key_points is not None:
      for j in range(key_points.shape[1]):
        p = key_points[i][j].astype(np.int32)
        cv2.circle(img, (p[0], p[1]), 2, point_color, thickness=thickness)

    label_text = class_names[label] if class_names is not None else 'cls {}'.format(label)
    label_text = ''
    if len(line) == 6:
      if OBJ_LEGEND == 'score':
        label_text += '{:.01f}'.format(line[-1]) # score
      else:
        label_text += '{:.01f}'.format(line[-2]) # rotation

    m = ((s+e)/2).astype(np.int32)
    cv2.putText(img, label_text, (m[0]-2, m[1] - 2),
                cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)

  if show:
      imshow(img, win_name, wait_time)
  if out_file is not None:
      imwrite(img, out_file)
      print('\twrite {}'.format(out_file))
  pass


def show_points_3d(points):
  '''
  points: [n,3/6/9]    xyz - color - normal
  '''
  pcl = o3d.geometry.PointCloud()
  pcl.points = o3d.utility.Vector3dVector(points[:,0:3])
  if points.shape[1] >= 6:
    colors = points[:,3:6]
    if colors.max() > 200:
      colors = colors/255
    pcl.colors = o3d.utility.Vector3dVector( colors )
  else:
    color = (0,0,255)
    pcl.paint_uniform_color(color)
  if points.shape[1] >= 9:
    pcl.normals = o3d.utility.Vector3dVector(points[:,6:9])

  o3d.visualization.draw_geometries([pcl])


def show_img_with_norm(img):
  assert img.shape[2] == 4
  h,w = img.shape[:2]
  xs = np.repeat( np.arange(w).reshape(1,-1,1), h, axis=0).astype(np.float32)/h
  ys = np.repeat( np.arange(h).reshape(-1,1,1), w, axis=1).astype(np.float32)/w
  zs = np.zeros_like(ys)

  density = np.repeat(img[:,:,0:1], 3, axis=2)

  img3d = np.concatenate([xs,ys,zs, density,  img[:,:,1:] ], axis=2)
  assert img3d.shape[-1] == 9
  img3d = img3d.reshape(-1,9)

  show_points_3d(img3d)
  pass

