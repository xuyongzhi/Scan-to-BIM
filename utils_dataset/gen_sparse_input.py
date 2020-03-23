from MinkowskiEngine import SparseTensor
import numpy as np

from tools.debug_utils import _show_3d_points_bboxes_ls

def prepare_sparse_input(img, img_meta=None, gt_bboxes=None, gt_labels=None):
  coords_batch, feats_batch = img
  sinput = SparseTensor(feats_batch, coords_batch)

  if 0:
    batch_mask = coords_batch[:,0] == 0
    points = coords_batch[batch_mask][:, 1:].cpu().data.numpy()
    colors = feats_batch[batch_mask][:, :3].cpu().data.numpy()
    colors = colors+0.5

    img_meta_0 = img_meta[0]
    voxel_resolution = img_meta_0['voxel_resolution']
    voxel_size = img_meta_0['voxel_size']

    lines2d = gt_bboxes[0].cpu().data.numpy()
    from beike_data_utils.line_utils import lines2d_to_bboxes3d
    from configs.common import OBJ_REP
    bboxes3d_pixel = lines2d_to_bboxes3d(lines2d, OBJ_REP, height=80, thickness=1)

    min_points = points.min(axis=0)
    max_points = points.max(axis=0)
    min_lines = lines2d[:,:4].min(axis=0)
    max_lines = lines2d[:,:4].max(axis=0)
    print(img_meta[0]['filename'])
    print(f'points scope: {min_points} - {max_points}')
    print(f'lines scope: {min_lines} - {max_lines}')

    _show_3d_points_bboxes_ls([points], [colors], [ bboxes3d_pixel ],
                b_colors = 'red', box_oriented=True)
    pass
  return sinput

