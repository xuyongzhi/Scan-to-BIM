from MinkowskiEngine import SparseTensor
import numpy as np
import torch

from tools.debug_utils import _show_3d_points_bboxes_ls, _show_lines_ls_points_ls

def prepare_sparse_input(img, img_meta=None, gt_bboxes=None, gt_labels=None, rescale=None):
  coords_batch, feats_batch = img
  ## For some networks, making the network invariant to even, odd coords is important
  #coord_base = (torch.rand(3) * 100).type_as(coords_batch)
  #coords_batch[:, 1:4] += coord_base
  sinput = SparseTensor(feats_batch, coords_batch)

  if 0:
    batch_size = coords_batch[:,0].max()+1
    for i in range(batch_size):
      batch_mask = coords_batch[:,0] == i
      points = coords_batch[batch_mask][:, 1:].cpu().data.numpy()
      colors = feats_batch[batch_mask][:, :3].cpu().data.numpy()
      colors = colors+0.5

      img_meta_i = img_meta[i]
      voxel_size = img_meta_i['voxel_size']
      raw_dynamic_vox_size = img_meta_i['raw_dynamic_vox_size']

      mask_i = sinput.C[:,0] == i
      Ci = sinput.C[mask_i]

      lines2d = gt_bboxes[i].cpu().data.numpy()


      from beike_data_utils.line_utils import lines2d_to_bboxes3d
      from configs.common import OBJ_REP
      bboxes3d_pixel = lines2d_to_bboxes3d(lines2d, OBJ_REP, height=30, thickness=1)

      min_points = points.min(axis=0)
      max_points = points.max(axis=0)
      min_lines = lines2d[:,:4].reshape(-1,2).min(axis=0)
      max_lines = lines2d[:,:4].reshape(-1,2).max(axis=0)

      data_aug = img_meta_i['data_aug']
      print('\n\nfinal sparse input')
      print(img_meta[i]['filename'])
      print(f'points scope: {min_points} - {max_points}')
      print(f'lines scope: {min_lines} - {max_lines}')
      print(f'data aug:\n {data_aug}\n')

      scale = 3
      _show_lines_ls_points_ls((512,512), [lines2d*scale], [points*scale])

      #_show_3d_points_bboxes_ls([points], [colors], [ bboxes3d_pixel ],
      #            b_colors = 'red', box_oriented=True)
      pass
  return sinput


def update_img_shape_for_pcl(x, img_meta, point_strides):
  img_meta['feat_sizes'] = [np.array( [*xi.size()[2:]] ) for xi in x]
  img_meta['pad_shape'] = img_meta['feat_sizes'][0] * point_strides[0]
  img_meta['img_shape'] = img_meta['pad_shape']
