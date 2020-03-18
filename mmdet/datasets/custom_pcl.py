import logging
import os
import sys
import numpy as np
from collections import defaultdict
from scipy import spatial
from plyfile import PlyData
import glob

from .registry import DATASETS

from utils_data3d.lib.utils import read_txt, fast_hist, per_class_iu
from utils_data3d.lib.dataset import VoxelizationDataset, DatasetPhase, str2datasetphase_type, cache
import utils_data3d.lib.transforms as t


class StanfordVoxelizationDatasetBase:
  CLIP_SIZE = None
  CLIP_BOUND = None
  LOCFEAT_IDX = 2
  ROTATION_AXIS = 'z'
  NUM_LABELS = 14
  IGNORE_LABELS = (10,)  # remove stairs, following SegCloud

  # CLASSES = [
  #     'clutter', 'beam', 'board', 'bookcase', 'ceiling', 'chair', 'column', 'door', 'floor', 'sofa',
  #     'table', 'wall', 'window'
  # ]

  IS_FULL_POINTCLOUD_EVAL = True

  DATA_PATH_FILE = {
      DatasetPhase.Train: 'train.txt',
      DatasetPhase.Val: 'val.txt',
      DatasetPhase.TrainVal: 'trainval.txt',
      DatasetPhase.Test: 'test.txt'
  }

  def test_pointcloud(self, pred_dir):
    print('Running full pointcloud evaluation.')
    # Join room by their area and room id.
    room_dict = defaultdict(list)
    for i, data_path in enumerate(self.data_paths):
      area, room = data_path.split(os.sep)
      room, _ = os.path.splitext(room)
      room_id = '_'.join(room.split('_')[:-1])
      room_dict[(area, room_id)].append(i)
    # Test independently for each room.
    sys.setrecursionlimit(100000)  # Increase recursion limit for k-d tree.
    pred_list = sorted(os.listdir(pred_dir))
    hist = np.zeros((self.NUM_LABELS, self.NUM_LABELS))
    for room_idx, room_list in enumerate(room_dict.values()):
      print(f'Evaluating room {room_idx} / {len(room_dict)}.')
      # Join all predictions and query pointclouds of split data.
      pred = np.zeros((0, 4))
      pointcloud = np.zeros((0, 7))
      for i in room_list:
        pred = np.vstack((pred, np.load(os.path.join(pred_dir, pred_list[i]))))
        pointcloud = np.vstack((pointcloud, self.load_ply(i)[0]))
      # Deduplicate all query pointclouds of split data.
      pointcloud = np.array(list(set(tuple(l) for l in pointcloud.tolist())))
      # Run test for each room.
      pred_tree = spatial.KDTree(pred[:, :3], leafsize=500)
      _, result = pred_tree.query(pointcloud[:, :3])
      ptc_pred = pred[result, 3].astype(int)
      ptc_gt = pointcloud[:, -1].astype(int)
      if self.IGNORE_LABELS:
        ptc_pred = self.label2masked[ptc_pred]
        ptc_gt = self.label2masked[ptc_gt]
      hist += fast_hist(ptc_pred, ptc_gt, self.NUM_LABELS)
      # Print results.
      ious = []
      print('Per class IoU:')
      for i, iou in enumerate(per_class_iu(hist) * 100):
        result_str = ''
        if hist.sum(1)[i]:
          result_str += f'{iou}'
          ious.append(iou)
        else:
          result_str += 'N/A'  # Do not print if data not in ground truth.
        print(result_str)
      print(f'Average IoU: {np.nanmean(ious)}')

  def _augment_coords_to_feats(self, coords, feats, labels=None):
    # Center x,y
    coords_center = coords.mean(0, keepdims=True)
    coords_center[0, 2] = 0
    norm_coords = coords - coords_center
    feats = np.concatenate((feats, norm_coords), 1)
    return coords, feats, labels


class StanfordDataset(StanfordVoxelizationDatasetBase, VoxelizationDataset):

  # Voxelization arguments
  VOXEL_SIZE = 0.05  # 5cm

  CLIP_BOUND = 4  # [-N, N]
  TEST_CLIP_BOUND = None

  # Augmentation arguments
  ROTATION_AUGMENTATION_BOUND = \
      ((-np.pi / 32, np.pi / 32), (-np.pi / 32, np.pi / 32), (-np.pi, np.pi))
  TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (-0.05, 0.05))

  AUGMENT_COORDS_TO_FEATS = True
  NUM_IN_CHANNEL = 6

  def __init__(self,
               config,
               prevoxel_transform=None,
               input_transform=None,
               target_transform=None,
               cache=False,
               augment_data=True,
               elastic_distortion=False,
               phase=DatasetPhase.Train):
    if isinstance(phase, str):
      phase = str2datasetphase_type(phase)
    if phase not in [DatasetPhase.Train, DatasetPhase.TrainVal]:
      self.CLIP_BOUND = self.TEST_CLIP_BOUND
    data_root = config.stanford3d_path
    if isinstance(self.DATA_PATH_FILE[phase], (list, tuple)):
      data_paths = []
      for split in self.DATA_PATH_FILE[phase]:
        data_paths += read_txt(os.path.join('splits/stanford', split))
    else:
      data_paths = read_txt(os.path.join('splits/stanford', self.DATA_PATH_FILE[phase]))

    logging.info('Loading {} {}: {}'.format(self.__class__.__name__, phase,
                                            self.DATA_PATH_FILE[phase]))

    VoxelizationDataset.__init__(
        self,
        data_paths,
        data_root=data_root,
        prevoxel_transform=prevoxel_transform,
        input_transform=input_transform,
        target_transform=target_transform,
        ignore_label=config.ignore_label,
        return_transformation=config.return_transformation,
        augment_data=augment_data,
        elastic_distortion=elastic_distortion,
        config=config)

  @cache
  def load_ply(self, index):
    filepath = self.data_root / self.data_paths[index]
    plydata = PlyData.read(filepath)
    data = plydata.elements[0].data
    coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
    feats = np.array([data['red'], data['green'], data['blue']], dtype=np.float32).T
    labels = np.array(data['label'], dtype=np.int32)
    return coords, feats, labels, None



@DATASETS.register_module
class VoxelDatasetBase(VoxelizationDataset):

  def __init__(self, phase, data_paths,  config, input_transform=None, target_transform=None
               ):
    if isinstance(phase, str):
      phase = str2datasetphase_type(phase)

    if config.return_transformation:
      collate_fn = t.cflt_collate_fn_factory(config.limit_numpoints)
    else:
      collate_fn = t.cfl_collate_fn_factory(config.limit_numpoints)

    prevoxel_transform_train = []
    if config.augment_data:
      prevoxel_transform_train.append(t.ElasticDistortion(self.ELASTIC_DISTORT_PARAMS))

    if len(prevoxel_transform_train) > 0:
      prevoxel_transforms = t.Compose(prevoxel_transform_train)
    else:
      prevoxel_transforms = None

    input_transforms = []
    if input_transform is not None:
      input_transforms += input_transform

    if config.augment_data:
      input_transforms += [
          t.RandomDropout(0.2),
          t.RandomHorizontalFlip(self.ROTATION_AXIS, self.IS_TEMPORAL),
          t.ChromaticAutoContrast(),
          t.ChromaticTranslation(config.data_aug_color_trans_ratio),
          t.ChromaticJitter(config.data_aug_color_jitter_std),
          # t.HueSaturationTranslation(config.data_aug_hue_max, config.data_aug_saturation_max),
      ]

    if len(input_transforms) > 0:
      input_transforms = t.Compose(input_transforms)
    else:
      input_transforms = None

    VoxelizationDataset.__init__(
        self,
        data_paths,
        data_root=self.data_root,
        prevoxel_transform=prevoxel_transforms,
        input_transform=input_transform,
        target_transform=target_transform,
        ignore_label=config.ignore_label,
        return_transformation=config.return_transformation,
        augment_data=config.augment_data,
        elastic_distortion=config.elastic_distortion,
        config=config)
    pass


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
      else:
        self.augment_data = False
        self.shuffle = False
        self.repeat = False


@DATASETS.register_module
class StanfordPclDataset(VoxelDatasetBase):
  _classes = ['clutter', 'beam', 'board', 'bookcase', 'ceiling', 'chair', 'column',
              'door', 'floor', 'sofa', 'stairs', 'table', 'wall', 'window', 'room']
  _category_ids_map = {cat:i for i,cat in enumerate(_classes)}
  _catid_2_cat = {i:cat for i,cat in enumerate(_classes)}

  CLASSES = _classes

  CLIP_SIZE = None
  CLIP_BOUND = None
  LOCFEAT_IDX = 2
  ROTATION_AXIS = 'z'
  NUM_LABELS = 14
  IGNORE_LABELS = (10,)  # remove stairs, following SegCloud


  def __init__(self, ann_file='data/stanford/', pipeline=None,  img_prefix=None):
    self.data_root = ann_file
    assert img_prefix in ['train', 'test']
    self.area_list = [1,2,3,4,6] if img_prefix == 'train' else [5]
    #phase = DatasetPhase.Train if img_prefix == 'train' else DatasetPhase.Test
    phase = img_prefix
    self.data_config = DataConfig(phase)
    self.load()
    self._set_group_flag()
    print(f'\n Area {img_prefix}: load {len(self)} files for areas {self.area_list}\n')
    VoxelDatasetBase.__init__(self, phase, self.data_paths, self.data_config)
    pass

  def _set_group_flag(self):
    self.flag = np.zeros(len(self), dtype=np.uint8)

  def load(self):
    data_paths = glob.glob(os.path.join(self.data_root, "*/*.ply"))
    data_paths.sort()
    data_paths = [p for p in data_paths if int(p.split('Area_')[1][0]) in self.area_list]
    self.data_paths = [p.split(self.data_root)[1] for p in data_paths]

    #data_roots = [f.replace('ply', 'npy') for f in pcl_files]
    n = len(self.data_paths)
    self.img_infos = []
    for i in range(n):
      ann_file = self.data_paths[i].replace('ply', 'npy')
      ann_file = os.path.join(self.data_root, ann_file )
      anno_3d = load_bboxes(ann_file)
      anno_2d = anno3d_to_anno_topview(anno_3d)

      img_meta = dict(filename = anno_3d['filename'], input_style='pcl')

      img_info = dict(
        img_meta = img_meta,
        gt_bboxes = anno_2d['bboxes'],
        gt_labels = anno_2d['labels']
      )
      self.img_infos.append(img_info)
    pass


def anno3d_to_anno_topview(anno_3d):
  from beike_data_utils.beike_utils import meter_2_pixel
  bboxes3d = anno_3d['bboxes_3d']
  room_scope = bboxes3d[-1].reshape(2,3)
  bbox_cat_ids = anno_3d['bbox_cat_ids']
  assert StanfordPclDataset._catid_2_cat[bbox_cat_ids[-1]] == 'room'
  bboxes2d = bboxes3d[:,[0,1,3,4]].reshape(-1,2,2)
  _, bboxes2d_pt = meter_2_pixel(None, bboxes2d, room_scope)

  num_box = bboxes3d.shape[0]
  rotation = np.zeros([num_box,1], dtype=bboxes2d_pt.dtype)
  bboxes2d_pt = np.concatenate([bboxes2d_pt.reshape(-1,4), rotation], axis=1)

  if 0:
    _bboxes2d_pt, _bbox_cat_ids = remove_categories(bboxes2d_pt, bbox_cat_ids, ['room'])
    #_show_lines_ls_points_ls((IMAGE_SIZE,IMAGE_SIZE), [_bboxes2d_pt], line_colors='random', box=True)
    for cat in StanfordPclDataset._classes:
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
  return anno_2d

def load_bboxes(pcl_file):
  anno = defaultdict(list)

  anno_file = pcl_file.replace('ply', 'npy')
  bboxes_dict = np.load(anno_file, allow_pickle=True).tolist()
  bboxes = []
  bbox_cat_ids = []
  for cat in bboxes_dict:
    bboxes.append( bboxes_dict[cat] )
    num_box = bboxes_dict[cat].shape[0]
    cat_ids = StanfordPclDataset._category_ids_map[cat] * np.ones([num_box], dtype=np.int64)
    bbox_cat_ids.append( cat_ids )
  bboxes = np.concatenate(bboxes, axis=0)
  bbox_cat_ids = np.concatenate(bbox_cat_ids)

  filename = pcl_file
  anno['filename'] = filename
  anno['bboxes_3d'] = bboxes
  anno['bbox_cat_ids'] = bbox_cat_ids
  return anno


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

def test(config):
  """Test point cloud data loader.
  """
  from torch.utils.data import DataLoader
  from utils_data3d.lib.utils import Timer
  import open3d as o3d

  def make_pcd(coords, feats):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords[:, :3].float().numpy())
    pcd.colors = o3d.utility.Vector3dVector(feats[:, :3].numpy() / 255)
    return pcd

  timer = Timer()
  DatasetClass = StanfordArea5Dataset
  transformations = [
      t.RandomHorizontalFlip(DatasetClass.ROTATION_AXIS, DatasetClass.IS_TEMPORAL),
      t.ChromaticAutoContrast(),
      t.ChromaticTranslation(config.data_aug_color_trans_ratio),
      t.ChromaticJitter(config.data_aug_color_jitter_std),
  ]

  dataset = DatasetClass(
      config,
      prevoxel_transform=t.ElasticDistortion(DatasetClass.ELASTIC_DISTORT_PARAMS),
      input_transform=t.Compose(transformations),
      augment_data=True,
      cache=True,
      elastic_distortion=True)

  data_loader = DataLoader(
      dataset=dataset,
      collate_fn=t.cfl_collate_fn_factory(limit_numpoints=False),
      batch_size=1,
      shuffle=True)

  # Start from index 1
  iter = data_loader.__iter__()
  for i in range(100):
    timer.tic()
    coords, feats, labels = iter.next()
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    pcd = make_pcd(coords, feats)
    o3d.visualization.draw_geometries([pcd])
    print(timer.toc())


if __name__ == '__main__':
  from utils_data3d.config import get_config
  config = get_config()

  test(config)
