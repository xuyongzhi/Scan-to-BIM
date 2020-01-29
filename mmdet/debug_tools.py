import torch
import numpy as np
import mmcv


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
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    pass
