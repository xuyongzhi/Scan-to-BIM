from mmdet.datasets.custom_pcl import VoxelDatasetBase
from plyfile import PlyData
import numpy as np
import glob
import os
from collections import defaultdict
from tools.debug_utils import _show_3d_points_lines_ls
from beike_data_utils.beike_utils import BEIKE_CLSINFO, load_anno_1scene, raw_anno_to_img

DEBUG_INPUT = 0
from configs.common import DEBUG_CFG

class DataConfig:
    return_transformation=True
    data_aug_color_trans_ratio=0.05
    data_aug_color_jitter_std=0.005
    ignore_label=255
    limit_numpoints=0
    elastic_distortion = False

    def __init__(self, phase, augment_data):
      assert phase in ['train', 'test']
      self.phase = phase
      if phase == 'train':
        self.augment_data = augment_data
        self.shuffle = True
        self.repeat = True
      else:
        self.augment_data = augment_data
        self.shuffle = False
        self.repeat = False


class BeikePcl(VoxelDatasetBase, BEIKE_CLSINFO):
  #_classes = ['background', 'wall', 'door', 'window', 'other']
  #_category_ids_map = {cat:i for i,cat in enumerate(_classes)}
  #_catid_2_cat = {i:cat for i,cat in enumerate(_classes)}

  #CLASSES = ['wall']
  #CLASSES = ['window']
  #for i,cat in enumerate(_classes):
  #  if cat not in CLASSES:
  #    del _category_ids_map[cat]
  #    del _catid_2_cat[i]

  #cat_ids = list(_category_ids_map.values())

  CLIP_SIZE = None
  LOCFEAT_IDX = 2
  ROTATION_AXIS = 'z'
  #NUM_LABELS = len(CLASSES)
  IGNORE_LABELS = None

  CLIP_BOUND = None
  TEST_CLIP_BOUND = None

  # Augmentation arguments
  ROTATION_AUGMENTATION_BOUND = \
      ((-np.pi * 0, np.pi *0), (-np.pi * 0, np.pi * 0), (-np.pi, np.pi))
  TRANSLATION_AUGMENTATION_RATIO_BOUND = ((0,0), (0,0), (0,0))
  ELASTIC_DISTORT_PARAMS = None

  AUGMENT_COORDS_TO_FEATS = True
  NUM_IN_CHANNEL = 9
  NORMALIZATION = True
  USE_NORMAL = True

  def __init__(self,
               obj_rep,
               ann_file='data/beike/processed_512/',
               img_prefix='train',
               test_mode=False,
               voxel_size=None,
               auto_scale_vs = True,
               max_num_points = None,
               max_footprint_for_scale = None,
               augment_data = None,
               data_types = ['color', 'norm', 'xyz'],
               bev_pad_pixels = 0,
               filter_edges = True,
               classes = ['wall'],
               pipeline=None,):
    self.obj_rep = obj_rep
    self.save_sparse_input_for_debug = 0
    self.load_voxlized_sparse = DEBUG_CFG.LOAD_VOXELIZED_SPARSE
    BEIKE_CLSINFO.__init__(self, classes)

    assert voxel_size is not None
    self.bev_pad_pixels = bev_pad_pixels
    self.filter_edges = filter_edges
    self.ann_path = ann_file
    if ann_file[-1] == '/':
      self.data_root = os.path.dirname( ann_file[:-1] )
    else:
      self.data_root = os.path.dirname( ann_file )
    self.test_mode = test_mode
    self.VOXEL_SIZE = voxel_size
    self.max_num_points = max_num_points
    self.max_footprint_for_scale = max_footprint_for_scale
    self.max_voxel_footprint = max_footprint_for_scale / voxel_size / voxel_size
    self.data_types = data_types
    phase = os.path.basename(img_prefix).split('.')[0]
    assert phase in ['train', 'test']

    #if voxel_resolution[0] is not None:
    #  bdx, bdy, bdz = [s * voxel_size / 2 for s in voxel_resolution]
    #  clip_bound = ((-bdx, bdx), (-bdy, bdy), (-bdz, bdz))
    #  self.CLIP_BOUND = clip_bound
    #else:
    self.CLIP_BOUND = None

    self.area_list = [1,2,3,4,6] if phase == 'train' else [5]
    self.scene_list = np.loadtxt(img_prefix, 'str').tolist()

    #self.scene_list = ['wcSLwyAKZafnozTPsaQMyv']

    if not isinstance(self.scene_list, list):
      self.scene_list = [self.scene_list]
    self.scene_list = sorted(self.scene_list)
    if isinstance( self.scene_list, str ):
      self.scene_list = [self.scene_list]
    self.data_config = DataConfig(phase, augment_data)
    self.load_anno()
    self._set_group_flag()
    print(f'\n {img_prefix}: load {len(self)} files\n')
    VoxelDatasetBase.__init__(self, phase, self.data_paths, self.data_config)

    all_inds = dict(color=[0,1,2], norm=[3,4,5], xyz=[6,7,8])
    self.data_channel_inds = np.array([all_inds[dt] for dt in self.data_types]).reshape(-1)
    pass


  def _set_group_flag(self):
    self.flag = np.zeros(len(self), dtype=np.uint8)


  def load_anno(self):
    '''
      mean_pcl_scope: [10.841 10.851  3.392]
      max_pcl_scope: [20.041 15.847  6.531]
    '''

    dpaths = [f'ply/{s}.ply' for s in self.scene_list]
    self.data_paths = []
    for p in dpaths:
      if os.path.exists( os.path.join(self.data_root,p) ):
        self.data_paths.append( p )
    self.ann_files = [s+'.json' for s in self.scene_list]

    n = len(self.data_paths)
    self.img_infos = []
    self.anno_raws = []
    for i in range(n):
      anno_raw = load_anno_1scene(self.ann_path, self.ann_files[i],
                          self._classes,
                          filter_edges=self.filter_edges)

      self.anno_raws.append(anno_raw)
      anno_2d = raw_anno_to_img(obj_rep=self.obj_rep, anno_raw=anno_raw,
                                anno_style='voxelization',
                                pixel_config={'voxel_size': self.VOXEL_SIZE})
      raw_dynamic_vox_size = (anno_raw['pcl_scope'][1] - anno_raw['pcl_scope'][0]) / self.VOXEL_SIZE
      raw_dynamic_vox_size = np.ceil(raw_dynamic_vox_size).astype(np.int32)
      raw_dynamic_vox_size = tuple(raw_dynamic_vox_size.tolist())
      img_meta = dict(filename = anno_raw['filename'],
                      input_style='pcl',
                      is_pcl = True,
                      pcl_scope = anno_raw['pcl_scope'],
                      line_length_min_mean_max = anno_raw['line_length_min_mean_max'],
                      voxel_size = self.VOXEL_SIZE,
                      scale_factor = 1,
                      raw_dynamic_vox_size = raw_dynamic_vox_size,
                      classes = self._classes,
                      data_aug={})

      img_info = dict(
        img_meta = img_meta,)
      #if not self.test_mode:
      if True:
        gt_bboxes = anno_2d['bboxes']
        #img_info['gt_bboxes'] = gt_bboxes
        img_info['gt_bboxes_2d_raw'] =  gt_bboxes
        img_info['gt_labels'] = anno_2d['labels']
        img_info['gt_relations'] = anno_2d['relations']
      self.img_infos.append(img_info)

    pcl_scopes = np.array([x['img_meta']['pcl_scope'] for x in self.img_infos])
    pcl_scopes = np.array([s[1]-s[0] for s in pcl_scopes])
    self.max_pcl_scope = pcl_scopes.max(axis=0)
    self.min_pcl_scope = pcl_scopes.min(axis=0)
    self.mean_pcl_scope = pcl_scopes.mean(axis=0)
    print(f'mean_pcl_scope: {self.mean_pcl_scope}')
    print(f'max_pcl_scope: {self.max_pcl_scope}')
    pass


  def load_ply(self, index):
    filepath = self.data_root / self.data_paths[index]
    plydata = PlyData.read(filepath)
    points = np.array(plydata['vertex'].data.tolist()).astype(np.float32)
    np0 = points.shape[0]
    if self.max_num_points is not None and  np0 > self.max_num_points:
      inds = np.random.choice(np0, self.max_num_points, replace=False)
      inds.sort()
      points = points[inds]
    #print(f'num points raw: {np0/1000}K -> {self.max_num_points}')
    assert points.shape[1] == 9

    pcl_scope = self.img_infos[index]['img_meta']['pcl_scope']
    points[:,:3] = points[:,:3] - pcl_scope[0:1]
    assert abs(points[:,:3].min()) < 0.1

    coords = points[:,:3]
    if self.USE_NORMAL:
      feats = points[:,3:9]
    else:
      feats = points[:,3:6]
    point_labels = np.zeros([feats.shape[0]], dtype=np.int32)

    if DEBUG_INPUT:
      gt_bboxes = self.img_infos[index]['gt_bboxes_2d_raw']
      from configs.common import OBJ_REP
      from beike_data_utils.line_utils import lines2d_to_bboxes3d
      #bboxes3d = lines2d_to_bboxes3d(gt_bboxes, OBJ_REP, height=2.5, thickness=0.1)
      print(filepath)
      _show_3d_points_lines_ls([coords], [feats[:,:3]], [gt_bboxes], b_colors = 'red', height=2.5, thickness=0.1)
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
    return coords, feats, point_labels, None


  def _augment_coords_to_feats(self, coords, feats, labels=None):
    # Center x,y
    coords_center = coords.mean(0, keepdims=True)
    coords_center[0, 2] = 0
    norm_coords = coords - coords_center
    feats = np.concatenate((feats, norm_coords), 1)
    return coords, feats, labels


  def _normalization(self, feats):
    assert feats.shape[1] == 9
    feats[:,:3] = feats[:,:3] / 255. - 0.5
    return feats


  def select_data_types(self, feats):
    '''
    do this at the last step
    '''
    assert feats.shape[1] == 9
    return feats[:, self.data_channel_inds]


  def save_sparse_input(self, coords, feats, labels, img_info ):
    import pickle
    sparse_intput_dir = os.path.join(self.data_root, 'sparse_vox_inputs')
    if not os.path.exists(sparse_intput_dir):
      os.makedirs(sparse_intput_dir)
    filename = img_info['img_meta']['filename']

    is_rotate = len(img_info['img_meta']['data_aug']['rotate_angles']) == 3
    if is_rotate:
      angle = img_info['img_meta']['data_aug']['rotate_angles'][2]
      ang_str = str(int(abs(angle)*100))
      if angle<0:
        ang_str = 'n'+ang_str
      svi_file = os.path.join(sparse_intput_dir, filename.replace('.json', f'-{ang_str}.pickle'))
    else:
      svi_file = os.path.join(sparse_intput_dir, filename.replace('.json', '.pickle'))
    with open(svi_file, 'wb') as f:
      pickle.dump( (coords, feats, labels, img_info), f  )
    print(f'save sparse vox input: {svi_file}')
    pass


  def load_sparse_input(self, index, is_rotate=1):
    import pickle

    img_info = self.img_infos[index]
    filename =  img_info['img_meta']['filename']
    sparse_intput_dir = os.path.join(self.data_root, 'sparse_vox_inputs')
    if is_rotate:
      svi_file_tem = os.path.join(sparse_intput_dir, filename.replace('.json', '*.pickle'))
      svi_files = glob.glob(svi_file_tem)
      n = len(svi_files)
      i = np.random.choice(n)
      svi_file = svi_files[ i ]
      print(f'voxlized sparse input: {i}/{n}')
    else:
      svi_file = os.path.join(sparse_intput_dir, filename.replace('.json', '.pickle'))
      print(f'voxelized sparse input no rotate')
    with open(svi_file, 'rb') as f:
      return_args = pickle.load(f)
    return return_args


def test():
  beikepcl = BeikePcl(ann_file='/home/z/Research/mmdetection/data/beike/processed_512')
  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  pass

if __name__ == '__main__':
  test()

