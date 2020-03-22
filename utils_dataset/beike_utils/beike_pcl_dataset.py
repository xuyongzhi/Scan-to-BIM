from mmdet.datasets.custom_pcl import VoxelDatasetBase
from mmdet.datasets.registry import DATASETS
from plyfile import PlyData
import numpy as np
import glob
import os
from collections import defaultdict

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

  CLIP_BOUND = 4  # [-N, N]
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
               voxel_size=0.05,
               voxel_resolution=[512,512,256]):
    self.data_root = ann_file
    self.VOXEL_SIZE = voxel_size
    self.voxel_resolution = voxel_resolution
    assert img_prefix in ['train', 'test']
    self.area_list = [1,2,3,4,6] if img_prefix == 'train' else [5]
    self.scene_list = np.loadtxt(os.path.join(self.data_root, img_prefix+'.txt'), 'str').tolist()
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

    data_paths = glob.glob(os.path.join(self.data_root, "ply/*.ply"))
    data_paths.sort()
    data_paths = [p for p in data_paths if os.path.basename(p).split('.')[0]  in self.scene_list]
    self.data_paths = [p.split(self.data_root)[1] for p in data_paths]
    self.ann_files = [s+'.json' for s in self.scene_list]

    n = len(self.data_paths)
    self.img_infos = []
    for i in range(n):
      anno_raw = load_anno_1scene(os.path.join(self.data_root, 'json'), self.ann_files[i])
      anno_2d = raw_anno_to_img(anno_raw)
      img_meta = dict(filename = anno_raw['filename'],
                      input_style='pcl',
                      pad_shape=self.voxel_resolution,
                      pcl_scope = anno_raw['pcl_scope'],
                      line_length_min_mean_max = anno_raw['line_length_min_mean_max'],
                      voxel_resolution = self.voxel_resolution,
                      voxel_size = self.VOXEL_SIZE)

      img_info = dict(
        img_meta = img_meta,
        gt_bboxes = anno_2d['bboxes'],
        gt_labels = anno_2d['labels']
      )
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
    coords = points[:,:3]
    feats = points[:,3:9]
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

