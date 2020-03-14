import glob, os
import numpy as np
from collections import defaultdict

class STANFORD_PCL:
  _classes = ['clutter', 'beam', 'board', 'bookcase', 'ceiling', 'chair', 'column',
              'door', 'floor', 'sofa', 'stairs', 'table', 'wall', 'window']
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
      anno = load_bboxes(pcl_files[i])
      img_info = dict(filename=anno['filename'], ann=anno)
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
    cat_ids = STANFORD_PCL._category_ids_map[cat] * np.ones([num_box], dtype=np.int32)
    bbox_cat_ids.append( cat_ids )
  bboxes = np.concatenate(bboxes, axis=0)
  bbox_cat_ids = np.concatenate(bbox_cat_ids)

  filename = pcl_file
  anno['filename'] = filename
  anno['bboxes'] = bboxes
  anno['bbox_cat_ids'] = bbox_cat_ids
  return anno

def load_ply(filepath):
  from plyfile import PlyData
  plydata = PlyData.read(filepath)
  data = plydata.elements[0].data
  positions = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
  colors = np.array([data['red'], data['green'], data['blue']], dtype=np.float32).T
  categories = np.array(data['label'], dtype=np.int32)
  instances = np.array(data['instance'], dtype=np.int32)
  return positions, colors, categories, instances

