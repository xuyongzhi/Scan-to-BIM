from plyfile import PlyData
from collections import defaultdict
import numpy as np
from tools.visual_utils import _show_3d_points_objs_ls
from tools.visual_utils import _show_objs_ls_points_ls, _show_3d_points_objs_ls
import glob

_classes = ['clutter', 'beam', 'board', 'bookcase', 'ceiling', 'chair', 'column',
            'door', 'floor', 'sofa', 'stairs', 'table', 'wall', 'window', 'room']
_category_ids_map = {cat:i for i,cat in enumerate(_classes)}

def load_1_ply(filepath):
    plydata = PlyData.read(filepath)
    data = plydata.elements[0].data
    coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
    feats = np.array([data['red'], data['green'], data['blue']], dtype=np.float32).T
    labels = np.array(data['label'], dtype=np.int32)
    instance = np.array(data['instance'], dtype=np.int32)
    #_show_3d_points_objs_ls([coords])
    return coords, feats, labels, None


def load_bboxes(pcl_file):
  anno_file = pcl_file.replace('.ply', '-boxes.npy')
  scope_file = pcl_file.replace('.ply', '-scope.txt')
  anno = defaultdict(list)

  bboxes_dict = np.load(anno_file, allow_pickle=True).tolist()
  bboxes = []
  bbox_cat_ids = []
  for cat in bboxes_dict:
    classes = ['door']
    classes = ['window']
    classes = ['beam', 'wall', 'column']
    if cat not in classes:
      continue
    bboxes.append( bboxes_dict[cat] )
    num_box = bboxes_dict[cat].shape[0]
    cat_ids = _category_ids_map[cat] * np.ones([num_box], dtype=np.int64)
    bbox_cat_ids.append( cat_ids )
  if len(bboxes) == 0:
    bboxes_3d = np.zeros([0,8])
    bboxes_2d = np.zeros([0,5])
  else:
    bboxes_3d = np.concatenate(bboxes, axis=0)

    scope = np.loadtxt(scope_file)
    anno['pcl_scope'] = scope

    #bboxes_3d[:, :2] -= scope[0:1,:2]
    #bboxes_3d[:, 2:4] -= scope[0:1,:2]

    # RoBox3D_UpRight_xyxy_sin2a_thick_Z0Z1  to   RoLine2D_UpRight_xyxy_sin2a
    bboxes_2d = bboxes_3d[:, :5]
    bbox_cat_ids = np.concatenate(bbox_cat_ids)

  filename = pcl_file
  anno['filename'] = filename
  anno['bboxes_3d'] = bboxes_3d
  anno['bboxes_2d'] = bboxes_2d
  anno['bbox_cat_ids'] = bbox_cat_ids

  #show_bboxes(bboxes_3d)
  return anno

def show_bboxes(bboxes_3d, points=None):
    #_show_3d_points_objs_ls(None, None, [bboxes_3d],  obj_rep='RoBox3D_UpRight_xyxy_sin2a_thick_Z0Z1')
    bboxes_show = bboxes_3d.copy()
    voxel_size = 0.02
    bboxes_show[:,:4] /= voxel_size
    bboxes_show[:,:4] += 10
    points_ls = [points] if points is not None else None
    #_show_objs_ls_points_ls( (512,512), [bboxes_show[:,:5]], obj_rep='RoLine2D_UpRight_xyxy_sin2a')
    #_show_objs_ls_points_ls( (512,512), [bboxes_show[:,:6]], obj_rep='RoBox2D_UpRight_xyxy_sin2a_thick' )
    _show_3d_points_objs_ls(points_ls, objs_ls=[bboxes_3d])

def main():
  path = '/DS/Stanford3D/aligned_processed_instance/Area_2/'
  files = glob.glob(path+'*.ply')
  for f in files:
    coords, feats, labels, _ = load_1_ply(f)
    anno = load_bboxes(f)
    show_bboxes(anno['bboxes_3d'], coords)
  pass

if __name__ == '__main__':
  main()
