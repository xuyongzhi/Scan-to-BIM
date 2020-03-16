import glob, os
import numpy as np
from collections import defaultdict
from tools.debug_utils import _show_lines_ls_points_ls
from beike_data_utils.beike_utils import meter_2_pixel

from configs.common import OBJ_DIM, OBJ_REP, IMAGE_SIZE

class STANFORD_PCL:
  _classes = ['clutter', 'beam', 'board', 'bookcase', 'ceiling', 'chair', 'column',
              'door', 'floor', 'sofa', 'stairs', 'table', 'wall', 'window', 'room']
  _category_ids_map = {cat:i for i,cat in enumerate(_classes)}
  _catid_2_cat = {i:cat for i,cat in enumerate(_classes)}

  def __init__(self, anno_folder='data/beike100/json/', img_prefix=None):
    self.anno_folder = anno_folder
    self.img_prefix = img_prefix
    self.load()
    pass

  def load(self):
    pcl_files = glob.glob(os.path.join(self.anno_folder, "*/*.ply"))
    pcl_files.sort()
    #ann_files = [f.replace('ply', 'npy') for f in pcl_files]
    n = len(pcl_files)
    self.img_infos = []
    for i in range(n):
      anno_3d = load_bboxes(pcl_files[i])
      anno_2d = anno3d_to_anno_topview(anno_3d)
      img_info = dict(filename=anno_2d['filename'],
                      ann_raw=anno_3d,
                      ann=anno_2d)
      self.img_infos.append(img_info)
    pass

  def getCatIds(self):
    return list(STANFORD_PCL._category_ids_map.values())

  def getImgIds(self):
    return list(range(len(self)))

  def __len__(self):
    return len(self.img_infos)

def load_bboxes(pcl_file):
  anno = defaultdict(list)

  anno_file = pcl_file.replace('ply', 'npy')
  bboxes_dict = np.load(anno_file, allow_pickle=True).tolist()
  bboxes = []
  bbox_cat_ids = []
  for cat in bboxes_dict:
    bboxes.append( bboxes_dict[cat] )
    num_box = bboxes_dict[cat].shape[0]
    cat_ids = STANFORD_PCL._category_ids_map[cat] * np.ones([num_box], dtype=np.int64)
    bbox_cat_ids.append( cat_ids )
  bboxes = np.concatenate(bboxes, axis=0)
  bbox_cat_ids = np.concatenate(bbox_cat_ids)

  filename = pcl_file
  anno['filename'] = filename
  anno['bboxes_3d'] = bboxes
  anno['bbox_cat_ids'] = bbox_cat_ids
  return anno

def normalize_bboxes3d(bboxes3d):
  room_scope = bboxes3d[-1]
  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  pass

def anno3d_to_anno_topview(anno_3d):

  bboxes3d = anno_3d['bboxes_3d']
  room_scope = bboxes3d[-1].reshape(2,3)
  bbox_cat_ids = anno_3d['bbox_cat_ids']
  assert STANFORD_PCL._catid_2_cat[bbox_cat_ids[-1]] == 'room'
  bboxes2d = bboxes3d[:,[0,1,3,4]].reshape(-1,2,2)
  _, bboxes2d_pt = meter_2_pixel(None, bboxes2d, room_scope)

  num_box = bboxes3d.shape[0]
  rotation = np.zeros([num_box,1], dtype=bboxes2d_pt.dtype)
  bboxes2d_pt = np.concatenate([bboxes2d_pt.reshape(-1,4), rotation], axis=1)

  if 0:
    _bboxes2d_pt, _bbox_cat_ids = remove_categories(bboxes2d_pt, bbox_cat_ids, ['room'])
    #_show_lines_ls_points_ls((IMAGE_SIZE,IMAGE_SIZE), [_bboxes2d_pt], line_colors='random', box=True)
    for cat in STANFORD_PCL._classes:
      print(cat)
      _bboxes2d_pt, _bbox_cat_ids = keep_categories(bboxes2d_pt, bbox_cat_ids, [cat])
      if _bboxes2d_pt.shape[0] == 0:
        continue
      _show_lines_ls_points_ls((IMAGE_SIZE,IMAGE_SIZE), [_bboxes2d_pt], line_colors='random', box=True)

  bboxes2d_pt, bbox_cat_ids = keep_categories(bboxes2d_pt, bbox_cat_ids, ['wall'])

  anno_2d = {}
  anno_2d['filename'] = anno_3d['filename']
  anno_2d['labels'] = bbox_cat_ids
  anno_2d['bboxes'] = bboxes2d_pt
  #_show_lines_ls_points_ls((IMAGE_SIZE,IMAGE_SIZE), [bboxes2d_pt], line_colors='random', box=True)
  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  return anno_2d



def remove_categories(bboxes, cat_ids, rm_cat_list):
  n = bboxes.shape[0]
  remain_mask = np.ones(n) == 1
  for cat in rm_cat_list:
    rm_id = STANFORD_PCL._category_ids_map[cat]
    mask = cat_ids == rm_id
    remain_mask[mask] = False
  return bboxes[remain_mask], cat_ids[remain_mask]

def keep_categories(bboxes, cat_ids, kp_cat_list):
  n = bboxes.shape[0]
  remain_mask = np.ones(n) == 0
  for cat in kp_cat_list:
    rm_id = STANFORD_PCL._category_ids_map[cat]
    mask = cat_ids == rm_id
    remain_mask[mask] = True
  return bboxes[remain_mask], cat_ids[remain_mask]

def load_ply(filepath):
  from plyfile import PlyData
  plydata = PlyData.read(filepath)
  data = plydata.elements[0].data
  positions = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
  colors = np.array([data['red'], data['green'], data['blue']], dtype=np.float32).T
  categories = np.array(data['label'], dtype=np.int32)
  instances = np.array(data['instance'], dtype=np.int32)
  labels = np.concatenate([categories[:,None], instances[:,None]], axis=1)
  return positions, colors, labels, None


if __name__ == '__main__':
  pass


