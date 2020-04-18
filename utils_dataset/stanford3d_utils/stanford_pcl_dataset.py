from mmdet.datasets.custom_pcl import VoxelDatasetBase
from mmdet.datasets.registry import DATASETS
from plyfile import PlyData
import numpy as np
import glob
import os
from collections import defaultdict

from beike_data_utils.beike_utils import  raw_anno_to_img
from utils_dataset.beike_utils.beike_pcl_dataset import DataConfig
from configs.common import DEBUG_CFG
from tools.debug_utils import _show_lines_ls_points_ls
from tools.visual_utils import _show_objs_ls_points_ls, _show_3d_points_objs_ls
from tools import debug_utils

@DATASETS.register_module
class StanfordPclDataset(VoxelDatasetBase):
  _classes = ['clutter', 'beam', 'board', 'bookcase', 'ceiling', 'chair', 'column',
              'door', 'floor', 'sofa', 'stairs', 'table', 'wall', 'window', 'room']
  _category_ids_map = {cat:i for i,cat in enumerate(_classes)}
  _catid_2_cat = {i:cat for i,cat in enumerate(_classes)}

  CLASSES = _classes

  CLIP_SIZE = None
  LOCFEAT_IDX = 2
  ROTATION_AXIS = 'z'
  NUM_LABELS = 14
  IGNORE_LABELS = (10,)  # remove stairs, following SegCloud

  CLIP_BOUND = None  # [-N, N]
  TEST_CLIP_BOUND = None

  # Augmentation arguments
  ROTATION_AUGMENTATION_BOUND = \
      ((-np.pi * 0, np.pi *0), (-np.pi * 0, np.pi * 0), (-np.pi, np.pi))
  TRANSLATION_AUGMENTATION_RATIO_BOUND = ((0,0), (0,0), (0,0))

  AUGMENT_COORDS_TO_FEATS = True
  NUM_IN_CHANNEL = 9
  NORMALIZATION = True

  EASY = ['Area_1/office_16']
  UNALIGNED = ['Area_2/storage_9', 'Area_3/office_8', 'Area_2/storage_9',
               'Area_4/hallway_14', 'Area_3/office_7']

  def __init__(self,
               ann_file='data/stanford/',
               pipeline=None,
               img_prefix='data/stanford/ply/train.txt',
               voxel_size=0.05,
               auto_scale_vs = True,
               max_num_points = None,
               max_footprint_for_scale = None,
               augment_data = None,
               data_types = ['color', 'norm', 'xyz'],
               filter_edges = True,
               classes = ['wall'],
               ):
    self.save_sparse_input_for_debug = 0
    self.data_root = ann_file
    self.VOXEL_SIZE = voxel_size
    self.classes = classes
    self.max_num_points = max_num_points
    self.max_footprint_for_scale = max_footprint_for_scale
    self.max_voxel_footprint = max_footprint_for_scale / voxel_size / voxel_size
    self.load_voxlized_sparse = DEBUG_CFG.LOAD_VOXELIZED_SPARSE
    img_prefix = img_prefix.split('/')[-1].split('.txt')[0]
    assert img_prefix in ['train', 'test']
    self.area_list = [1,2,3,4,6] if img_prefix == 'train' else [5]
    #phase = DatasetPhase.Train if img_prefix == 'train' else DatasetPhase.Test
    phase = img_prefix
    self.data_config = DataConfig(phase, augment_data)
    self.load_anno()
    self._set_group_flag()
    print(f'\n Area {img_prefix}: load {len(self)} files for areas {self.area_list}\n')
    VoxelDatasetBase.__init__(self, phase, self.data_paths, self.data_config)

    self.data_types = data_types
    all_inds = dict(color=[0,1,2], norm=[3,4,5], xyz=[6,7,8])
    self.data_channel_inds = np.array([all_inds[dt] for dt in self.data_types]).reshape(-1)
    pass

  def _set_group_flag(self):
    self.flag = np.zeros(len(self), dtype=np.uint8)

  def load_anno(self):
    data_paths = glob.glob(os.path.join(self.data_root, "*/*.ply"))
    data_paths = [p for p in data_paths if int(p.split('Area_')[1][0]) in self.area_list]
    data_paths = [p.split(self.data_root)[1] for p in data_paths]

    #data_paths = [f+'.ply' for f in self.UNALIGNED]
    #data_paths = [f+'.ply' for f in self.EASY]

    data_paths.sort()
    self.data_paths = data_paths

    #data_roots = [f.replace('ply', 'npy') for f in pcl_files]
    n = len(self.data_paths)
    self.img_infos = []
    for i in range(n):
      pcl_file = os.path.join(self.data_root, self.data_paths[i])
      anno_3d = load_bboxes( pcl_file )
      #anno_2d = raw_anno_to_img(anno_3d, 'voxelization', {'voxel_size': self.VOXEL_SIZE})
      anno_2d = anno3d_to_anno_topview(anno_3d, self.classes)

      raw_dynamic_vox_size = (anno_3d['pcl_scope'][1] - anno_3d['pcl_scope'][0]) / self.VOXEL_SIZE
      raw_dynamic_vox_size = np.ceil(raw_dynamic_vox_size).astype(np.int32)
      raw_dynamic_vox_size = tuple(raw_dynamic_vox_size.tolist())

      img_meta = dict(filename = anno_3d['filename'],
                      pcl_scope = anno_3d['pcl_scope'],
                      input_style='pcl',
                      raw_dynamic_vox_size = raw_dynamic_vox_size,
                      voxel_size = self.VOXEL_SIZE,
                      classes = self.classes,
                      data_aug = {},
                      )

      img_info = dict(
        img_meta = img_meta,
        gt_bboxes_raw = anno_2d['bboxes'],
        gt_labels = anno_2d['labels'],
        gt_bboxes_3d = anno_3d['bboxes_3d'],
      )
      self.img_infos.append(img_info)
    pass

  def load_ply(self, index):
    filepath = self.data_root / self.data_paths[index]
    coords, feats, labels, _ = load_1_ply(filepath)
    normpath = str(filepath).replace('.ply', '-norm.npy')
    norm = np.load(normpath)
    feats = np.concatenate([feats, norm], axis=1)

    np0 = coords.shape[0]
    if self.max_num_points is not None and  np0 > self.max_num_points:
      inds = np.random.choice(np0, self.max_num_points, replace=False)
      inds.sort()
      coords = coords[inds]
      feats = feats[inds]
      labels = labels[inds]

    pcl_scope = self.img_infos[index]['img_meta']['pcl_scope']
    coords -= pcl_scope[0:1]

    bboxes_3d = self.img_infos[index]['gt_bboxes_3d']
    #_show_3d_points_objs_ls([coords], None, [bboxes_3d], 'RoBox3D_UpRight_xyxy_sin2a_thick_Z0Z1')
    return coords, feats, labels, None

  def _augment_coords_to_feats(self, coords, feats, labels=None):
    # Center x,y
    coords_center = coords.mean(0, keepdims=True)
    coords_center[0, 2] = 0
    norm_coords = coords - coords_center
    feats = np.concatenate((feats, norm_coords), 1)
    return coords, feats, labels

  def _normalization(self, feats):
    feats[:,:3] = feats[:,:3] / 255. - 0.5
    return feats

  def select_data_types(self, feats):
    '''
    do this at the last step
    '''
    assert feats.shape[1] == 9
    return feats[:, self.data_channel_inds]


def unused_aligned_3dbboxes_TO_oriented_line(bboxes_3d):
  '''
  bboxes_3d: [n,6] 6: x0,y0,z0, z1,y1,z1
  lines3d:  [n,8] 8: x2,y2,x3,y3, theta, thickness, z0,z1

  (x2,y2) and (x3,y3) are center of two corners
  '''
  size = np.abs(bboxes_3d[:,3:] - bboxes_3d[:,:3])

  lines3d = []
  n = bboxes_3d.shape[0]
  for i in range(n):
    bboxes_3d_i = bboxes_3d[i:i+1]
    z0 = bboxes_3d_i[:,2]
    z1 = bboxes_3d_i[:,5]
    zeros = z0 * 0
    if size[i][1] > size[i][0]:
      # x is the short axis
      x2 = x3 = bboxes_3d_i[:, [0,3]].mean(axis=1)
      y2 = bboxes_3d_i[:, [1, 4]].min(axis=1)
      y3 = bboxes_3d_i[:, [1, 4]].max(axis=1)
      thickness = bboxes_3d_i[:,3] - bboxes_3d_i[:,0]
      lines3d_i = np.concatenate([x2,y2,x3,y3,zeros, thickness,z0,z1], axis=0)[None,:]
    else:
      # y is the short axis
      y2 = y3 = bboxes_3d_i[:, [1,4]].mean(axis=1)
      x2 = bboxes_3d_i[:, [0, 3]].min(axis=1)
      x3 = bboxes_3d_i[:, [0, 3]].max(axis=1)
      thickness = bboxes_3d_i[:,4] - bboxes_3d_i[:,1]
      lines3d_i = np.concatenate([x2,y2,x3,y3,zeros, thickness,z0,z1], axis=0)[None,:]
    lines3d.append(lines3d_i)
  lines3d = np.concatenate( lines3d, axis=0 )
  return lines3d


def anno3d_to_anno_topview(anno_3d, classes):
  ##from beike_data_utils.beike_utils import meter_2_pixel
  #bboxes_3d = anno_3d['bboxes_3d']
  #room_scope = bboxes_3d[-1].reshape(2,3)
  bbox_cat_ids = anno_3d['bbox_cat_ids']
  #assert StanfordPclDataset._catid_2_cat[bbox_cat_ids[-1]] == 'room'

  #lines3d = aligned_3dbboxes_TO_oriented_line(bboxes_3d)
  #lines2d = lines3d[:,:5]

  #bboxes_2d_pt = lines2d

  ##debug_utils._show_3d_points_bboxes_ls(bboxes_ls = [bboxes_3d])
  #vs = 1/0.04*2
  ##debug_utils._show_lines_ls_points_ls( (512,512), lines_ls=[bboxes_2d_pt*vs], box=True, line_colors='random' )

  #if 0:
  #  _bboxes_2d_pt, _bbox_cat_ids = remove_categories(bboxes_2d_pt, bbox_cat_ids, ['room'])
  #  #_show_lines_ls_points_ls((IMAGE_SIZE,IMAGE_SIZE), [_bboxes_2d_pt], line_colors='random', box=True)
  #  for cat in StanfordPclDataset._classes:
  #    print(cat)
  #    _bboxes_2d_pt, _bbox_cat_ids = keep_categories(bboxes_2d_pt, bbox_cat_ids, [cat])
  #    if _bboxes_2d_pt.shape[0] == 0:
  #      continue
  #    _show_lines_ls_points_ls((512, 512), [_bboxes_2d_pt], line_colors='random', box=True)

  bboxes_2d = anno_3d['bboxes_2d']
  bboxes_2d, bbox_cat_ids = keep_categories(bboxes_2d, bbox_cat_ids, classes)

  anno_2d = {}
  anno_2d['filename'] = anno_3d['filename']
  anno_2d['labels'] = bbox_cat_ids
  anno_2d['bboxes'] = bboxes_2d

  #show_bboxes(anno_3d['bboxes_3d'])
  return anno_2d


def load_bboxes(pcl_file):
  anno_file = pcl_file.replace('.ply', '-boxes.npy')
  scope_file = pcl_file.replace('.ply', '-scope.txt')
  anno = defaultdict(list)

  bboxes_dict = np.load(anno_file, allow_pickle=True).tolist()
  bboxes = []
  bbox_cat_ids = []
  for cat in bboxes_dict:
    bboxes.append( bboxes_dict[cat] )
    num_box = bboxes_dict[cat].shape[0]
    cat_ids = StanfordPclDataset._category_ids_map[cat] * np.ones([num_box], dtype=np.int64)
    bbox_cat_ids.append( cat_ids )
  bboxes_3d = np.concatenate(bboxes, axis=0)

  scope = np.loadtxt(scope_file)
  anno['pcl_scope'] = scope

  bboxes_3d[:, :2] -= scope[0:1,:2]
  bboxes_3d[:, 2:4] -= scope[0:1,:2]

  # RoBox3D_UpRight_xyxy_sin2a_thick_Z0Z1  to   RoLine2D_UpRight_xyxy_sin2a
  bboxes_2d = bboxes_3d[:, :5]
  bbox_cat_ids = np.concatenate(bbox_cat_ids)

  filename = pcl_file
  anno['filename'] = filename
  anno['bboxes_3d'] = bboxes_3d
  anno['bboxes_2d'] = bboxes_2d
  anno['bbox_cat_ids'] = bbox_cat_ids

  show = 0
  if show:
    show_bboxes(bboxes_3d)
  return anno


def show_bboxes(bboxes_3d):
    #_show_3d_points_objs_ls(None, None, [bboxes_3d],  obj_rep='RoBox3D_UpRight_xyxy_sin2a_thick_Z0Z1')
    bboxes_show = bboxes_3d.copy()
    voxel_size = 0.02
    bboxes_show[:,:4] /= voxel_size
    bboxes_show[:,:4] += 10
    _show_objs_ls_points_ls( (512,512), [bboxes_show[:,:5]], obj_rep='RoLine2D_UpRight_xyxy_sin2a' )
    #_show_objs_ls_points_ls( (512,512), [bboxes_show[:,:6]], obj_rep='RoBox2D_UpRight_xyxy_sin2a_thick' )
    import pdb; pdb.set_trace()  # XXX BREAKPOINT


def remove_categories(bboxes, cat_ids, rm_cat_list):
  n = bboxes.shape[0]
  remain_mask = np.ones(n) == 1
  for cat in rm_cat_list:
    rm_id = StanfordPclDataset._category_ids_map[cat]
    mask = cat_ids == rm_id
    remain_mask[mask] = False
  return bboxes[remain_mask], cat_ids[remain_mask]


def keep_categories(bboxes, cat_ids, kp_cat_list):
  n = bboxes.shape[0]
  remain_mask = np.ones(n) == 0
  for cat in kp_cat_list:
    rm_id = StanfordPclDataset._category_ids_map[cat]
    mask = cat_ids == rm_id
    remain_mask[mask] = True
  return bboxes[remain_mask], cat_ids[remain_mask]


def load_1_ply(filepath):
    plydata = PlyData.read(filepath)
    data = plydata.elements[0].data
    coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
    feats = np.array([data['red'], data['green'], data['blue']], dtype=np.float32).T
    labels = np.array(data['label'], dtype=np.int32)
    instance = np.array(data['instance'], dtype=np.int32)
    return coords, feats, labels, None


