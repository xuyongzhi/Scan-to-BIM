from mmdet.datasets.custom_pcl import VoxelDatasetBase
from mmdet.datasets.registry import DATASETS
from plyfile import PlyData
import numpy as np
import glob
import os
from collections import defaultdict
from tools.debug_utils import _show_3d_points_bboxes_ls

DEBUG_INPUT = 0

class DataConfig:
    return_transformation=False
    data_aug_color_trans_ratio=0.05
    data_aug_color_jitter_std=0.005
    ignore_label=255
    limit_numpoints=0
    elastic_distortion = False

    def __init__(self, phase):
      assert phase in ['train', 'test']
      self.phase = phase
      if phase == 'train':
        self.augment_data = True
        self.shuffle = True
        self.repeat = True

        self.augment_data = False
      else:
        self.augment_data = False
        self.shuffle = False
        self.repeat = False


@DATASETS.register_module
class BeikePclDataset(VoxelDatasetBase):
  _classes = ['background', 'wall', 'door', 'window', 'other']
  _category_ids_map = {cat:i for i,cat in enumerate(_classes)}
  _catid_2_cat = {i:cat for i,cat in enumerate(_classes)}

  CLASSES = ['wall']
  for i,cat in enumerate(_classes):
    if cat not in CLASSES:
      del _category_ids_map[cat]
      del _catid_2_cat[i]

  CLIP_SIZE = None
  LOCFEAT_IDX = 2
  ROTATION_AXIS = 'z'
  NUM_LABELS = len(CLASSES)
  IGNORE_LABELS = None

  CLIP_BOUND = None
  TEST_CLIP_BOUND = None

  # Augmentation arguments
  ROTATION_AUGMENTATION_BOUND = \
      ((-np.pi / 32, np.pi / 32), (-np.pi / 32, np.pi / 32), (-np.pi, np.pi))
  TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (-0.05, 0.05))

  AUGMENT_COORDS_TO_FEATS = True
  NUM_IN_CHANNEL = 9
  NORMALIZATION = True

  def __init__(self,
               ann_file='data/beike/processed_512/',
               pipeline=None,
               img_prefix='train',
               test_mode=False,
               voxel_size=0.05,
               voxel_resolution=[512,512,256]):
    self.data_root = ann_file
    self.test_mode = test_mode
    print(test_mode)
    self.VOXEL_SIZE = voxel_size
    self.voxel_resolution = voxel_resolution
    assert img_prefix in ['train', 'test']

    bdx, bdy, bdz = [s * voxel_size / 2 for s in voxel_resolution]
    clip_bound = ((-bdx, bdx), (-bdy, bdy), (-bdz, bdz))
    self.CLIP_BOUND = clip_bound


    self.area_list = [1,2,3,4,6] if img_prefix == 'train' else [5]
    self.scene_list = np.loadtxt(os.path.join(self.data_root, img_prefix+'.txt'), 'str').tolist()

    #self.scene_list = ['wcSLwyAKZafnozTPsaQMyv']

    if not isinstance(self.scene_list, list):
      self.scene_list = [self.scene_list]
    self.scene_list = sorted(self.scene_list)
    if isinstance( self.scene_list, str ):
      self.scene_list = [self.scene_list]
    #phase = DatasetPhase.Train if img_prefix == 'train' else DatasetPhase.Test
    phase = img_prefix
    self.data_config = DataConfig(phase)
    self.load_anno()
    self._set_group_flag()
    print(f'\n {img_prefix}: load {len(self)} files\n')
    VoxelDatasetBase.__init__(self, phase, self.data_paths, self.data_config)
    pass

  def _set_group_flag(self):
    self.flag = np.zeros(len(self), dtype=np.uint8)

  def load_anno(self):
    '''
      mean_pcl_scope: [10.841 10.851  3.392]
      max_pcl_scope: [20.041 15.847  6.531]
    '''
    from beike_data_utils.beike_utils import load_anno_1scene, raw_anno_to_img

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
      anno_raw = load_anno_1scene(os.path.join(self.data_root, 'json'), self.ann_files[i])
      self.anno_raws.append(anno_raw)
      anno_2d = raw_anno_to_img(anno_raw, 'voxelization', {'voxel_size': self.VOXEL_SIZE})
      img_meta = dict(filename = anno_raw['filename'],
                      input_style='pcl',
                      pad_shape=self.voxel_resolution,
                      pcl_scope = anno_raw['pcl_scope'],
                      line_length_min_mean_max = anno_raw['line_length_min_mean_max'],
                      voxel_resolution = self.voxel_resolution,
                      voxel_size = self.VOXEL_SIZE,
                      img_shape = self.voxel_resolution[:2]+[3,],
                      scale_factor = 1)

      img_info = dict(
        img_meta = img_meta,)
      if not self.test_mode:
        img_info['gt_bboxes'] = anno_2d['bboxes']
        img_info['gt_labels'] = anno_2d['labels']
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
    assert points.shape[1] == 9

    pcl_scope = self.img_infos[index]['img_meta']['pcl_scope']
    points[:,:3] = points[:,:3] - pcl_scope[0:1]
    if not abs(points[:,:3].min()) < 0.1:
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass

    coords = points[:,:3]
    feats = points[:,3:9]

    if DEBUG_INPUT:
      gt_bboxes = self.img_infos[index]['gt_bboxes'] * self.VOXEL_SIZE
      from configs.common import OBJ_REP
      from beike_data_utils.line_utils import lines2d_to_bboxes3d
      bboxes3d = lines2d_to_bboxes3d(gt_bboxes, OBJ_REP, height=2.5, thickness=0.1)
      print(filepath)
      _show_3d_points_bboxes_ls([coords], [feats[:,:3]], [bboxes3d], b_colors = 'red', box_oriented=True)
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
    return coords, feats, None, None

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


def test():
  beikepcl = BeikePclDataset(ann_file='/home/z/Research/mmdetection/data/beike/processed_512')
  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  pass

if __name__ == '__main__':
  test()

