import torch
import numpy as np
import mmcv
from mmcv.image import imread, imwrite
import cv2

from .color import color_val

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


def imshow_bboxes_random_colors(img, bboxes):
    color_lib = ['red', 'green', 'blue', 'cyan', 'yellow', 'magenta']
    ids = np.randint(len(color_lib))


def imshow_bboxes_ref(img, bboxes0, bboxes1):
    bboxes = np.concatenate([bboxes0, bboxes1], 0)
    n0 = bboxes0.shape[0]
    n1 = bboxes1.shape[0]
    colors = ['red']*n0 + ['green']*n1
    mmcv.imshow_bboxes(img, bboxes, colors)

def draw_img_lines(img, lines, color='random', lines_ref=None):
    img = img.copy()
    for i in range(lines.shape[0]):
      s, e = lines[i]
      cv2.line(img, (s[0], s[1]), (e[0], e[1]), (255,0,0), 6)
      #cv2.putText(img, obj, (s[0], s[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
      #            (255, 255, 255), 1)
    mmcv.imshow(img)
    pass


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
  if score_thr > 0:
    assert lines.shape[1] == 6
    scores = lines[:,-1]
    inds = lines[:,-1] > score_thr
    lines = lines[inds, :]
    labels = labels[inds]
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
    show_det_lines(img0, lines[i:i+1], labels[i:i+1], class_names, score_thr,
                   line_color, text_color, thickness, font_scale, show, win_name,
                   wait_time, None, key_points[i:i+1], point_color)
    out.write(img0)
    img0 = img.copy()

  show_det_lines(img, lines, labels, class_names, score_thr,
                  line_color, text_color, thickness, font_scale, show, win_name,
                  wait_time, out_file, key_points, point_color)
  out.write(img)

  out.release()
  pass

def show_det_lines(img, lines, labels, class_names=None, score_thr=0,
                   line_color='green', text_color='green', thickness=1,
                   font_scale=0.5,show=True, win_name='', wait_time=0,
                   out_file=None, key_points=None, point_color='red'):
  assert lines.ndim == 2
  assert lines.shape[1] == 6 or lines.shape[1] == 5
  assert labels.ndim == 1
  assert labels.shape[0] == lines.shape[0]
  if key_points is not None:
    assert key_points.shape[0]== lines.shape[0]

  img = imread(img)

  if score_thr > 0:
    assert lines.shape[1] == 6
    scores = lines[:,-1]
    inds = lines[:,-1] > score_thr
    lines = lines[inds, :]
    labels = labels[inds]
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

    cv2.line(img, (s[0], s[1]), (e[0], e[1]), line_color, thickness=thickness)

    if key_points is not None:
      for j in range(key_points.shape[1]):
        p = key_points[i][j].astype(np.int32)
        cv2.circle(img, (p[0], p[1]), 2, point_color, thickness=thickness)

    label_text = class_names[label] if class_names is not None else 'cls {}'.format(label)
    label_text = ''
    if len(line) == 6:
      #label_text += '{:.01f}'.format(line[-1])
      label_text += '{:.01f}'.format(line[-2])

    m = ((s+e)/2).astype(np.int32)
    cv2.putText(img, label_text, (m[0]-2, m[1] - 2),
                cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)

  if show:
      imshow(img, win_name, wait_time)
  if out_file is not None:
      imwrite(img, out_file)
      print('\twrite {}'.format(out_file))
  pass
