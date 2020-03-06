import open3d as o3d
import torch
import numpy as np
import mmcv
from mmcv.image import imread, imwrite
import cv2

from .color import color_val, get_random_color

def show_multi_ls_shapes(in_ls, names_ls, env):
  for i,ls in enumerate(in_ls):
    show_shapes(ls, env + ' - ' + names_ls[i])

def show_shapes(tensor_ls, flag=''):
  print(f'\n{flag}:')
  _show_tensor_ls_shapes(tensor_ls, i='', pre='')
  print(f'\n')

def _show_tensor_ls_shapes(tensor_ls, flag='', i='', pre=''):
  if isinstance(tensor_ls, torch.Tensor):
    shape = tensor_ls.shape
    print(f'{pre} {i} \t{shape}')
  else:
    pre += '  '
    for i,tensor in enumerate(tensor_ls):
      _show_tensor_ls_shapes(tensor, flag, i, pre)


#-------------------------------------------------------------------------------
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
    point_thickness=1, only_save=False):
  '''
  img: [h,w,3] or [h,w,1], or [h,w] or (h_size, w_size)
  '''
  img = _draw_lines_ls_points_ls(img, lines_ls, points_ls, line_colors, point_colors, out_file, line_thickness, point_thickness)
  if not only_save:
    mmcv.imshow(img)

def _draw_lines_ls_points_ls(img, lines_ls, points_ls=None, line_colors='green', point_colors='red', out_file=None, line_thickness=1, point_thickness=1):
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
      img = _draw_lines(img, lines, line_colors[i], line_thickness[i])

  if points_ls is not None:
    for i, points in enumerate(points_ls):
      img = _draw_points(img, points, point_colors[i], point_thickness[i])
  if out_file is not None:
    mmcv.imwrite(img, out_file)
    print(out_file)
  return img

def _draw_lines(img, lines, color, line_thickness=1, font_scale=0.5, text_color='green'):
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
        cv2.line(img, (s[0], s[1]), (e[0], e[1]), c, thickness=line_thickness)


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


#-------------------------------------------------------------------------------
def lines_to_2points_format(lines):
  lines = lines.copy()
  if lines.ndim == 3:
    assert lines.shape[1:] == (2,2)
    return lines
  assert lines.ndim == 2
  n = lines.shape[0]
  if lines.shape[1] == 4:
    return lines.reshape(n,2,2)
  assert lines.shape[1] == 5
  for i in range(lines.shape[0]):
    istopleft = lines[i,4] >= 0
    if not istopleft:
      lines[i,0], lines[i,2] = lines[i,2], lines[i,0]
  lines = lines[:,:4].reshape(n,2,2)
  return lines

def scale_img_touint8(img):
  #assert img.ndim == 2 or img.ndim == 3,  img.shape
  if img.max() <= 1:
    img = img*255
  img = img.astype(np.uint8)
  #if img.ndim == 2:
  #  img = np.expand_dims( img, 2)
  #if img.shape[2] == 1:
  #  img = np.tile(img, (1,1,3))
  return img

def show_img_lines(img, lines, points=None, colors=['random'], name=None, only_draw=False):
    '''
    img: [h,w, 3]
    lines: [n,4/5]
    '''
    lines = lines_to_2points_format(lines)
    img = img.copy()
    img = scale_img_touint8(img)
    if img.ndim == 2:
      img = np.expand_dims(img, 2)
    if img.shape[2] == 1:
      img = np.tile(img, (1,1,3))
    for i in range(lines.shape[0]):
        s, e = lines[i]
        if colors[0] == 'random':
          c = get_random_color()
        else:
          c = color_val(colors[i])
        cv2.line(img, (s[0], s[1]), (e[0], e[1]), c, 1)
        #cv2.putText(img, obj, (s[0], s[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
        #            (255, 255, 255), 1)
    if points is not None:
      for i in range(points.shape[0]):
        p = points[i]
        c = get_random_color()
        cv2.circle(img, (p[0], p[1]), 2, c, thickness=1)
    if not only_draw:
      mmcv.imshow(img)
    if name is not None:
      mmcv.imwrite(img, name)
    return img

def show_heatmap(scores, show_size=None, out_file=None, gt_lines=None, gt_corners=None, score_thr=None):
  '''
  scores: [h,w, 1]
  '''
  if isinstance(scores, torch.Tensor):
    img = scores.cpu().data.numpy().astype(np.float32)
  else:
    img = scores.astype(np.float32)
  assert img.ndim == 2

  if score_thr is not None:
    img = (img > score_thr).astype(np.float32)

  img = scale_img_touint8(img)
  if show_size is not None:
    img = mmcv.imresize(img, show_size)
  img = np.tile(np.expand_dims(img, 2),(1,1,3))
  #h,w = scores.shape[:2]
  #img = np.zeros((h,w), dtype=uint8)
  if gt_lines is not None:
    img = show_img_lines(img, gt_lines, colors=['random'], only_draw=True)
  if gt_corners is not None:
    img = draw_points(img, gt_corners, 'red')
  if out_file is None:
    mmcv.imshow(img)
  else:
    mmcv.imwrite(img, out_file)
    print(out_file)

def show_points(points0, img_size, points1, cor0='red', cor1='green', name=''):
  img = np.zeros(img_size + (3,), dtype=np.uint8)
  img = draw_points(img, [points0, points1], [cor0, cor1])
  mmcv.imshow(img)



