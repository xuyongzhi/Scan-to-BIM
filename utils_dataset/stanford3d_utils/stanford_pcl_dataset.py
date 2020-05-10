from abc import abstractmethod
from mmdet.datasets.custom_pcl import VoxelDatasetBase
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
from obj_geo_utils.obj_utils import OBJ_REPS_PARSE

SMALL_DATA = 1
NO_LONG = 1

BAD_INSTANCES = ['Area_3/office_7', 'Area_2/storage_8']
DOOR_FAILED = ['office_4', 'Area_2/storage_5', 'Area_2/storage_7']

#_cat_2_color = {'wall':'blue', 'column': 'red','door':'green'}
_raw_pcl_classes_order = [ 'background', 'beam', 'board', 'bookcase', 'ceiling', 'chair', 'column',
                  'door', 'floor', 'sofa', 'stairs', 'table', 'wall', 'window', 'room']
_raw_pcl_category_ids_map = {_raw_pcl_classes_order[i]:i for i in range(len(_raw_pcl_classes_order))}
class Stanford_CLSINFO(object):
  classes_order = [ 'background', 'wall', 'beam', 'column',  'door', 'window', 'ceiling',
                   'floor', 'board', 'bookcase', 'chair', 'sofa', 'stairs', 'table', 'room']
  classes_order = [ 'background', 'wall', 'beam', 'column',  'door', 'window', 'ceiling',
                   'floor', 'board', 'bookcase', 'chair', 'sofa', 'table']
  def __init__(self, classes_in, always_load_walls=1):
      if classes_in == 'all':
        classes = self.classes_order
      else:
        classes = [c for c in self.classes_order if c in classes_in]
        if 'background' not in classes:
          classes = ['background']+ classes
      n = len(classes)
      self._classes = classes
      self.CLASSES = classes
      self.cat_ids = list(range(n))
      self._category_ids_map = {classes[i]:i for i in range(n)}
      self._catid_2_cat = {i:classes[i] for i in range(n)}
      self._point_labels = self.cat_ids

      if always_load_walls and 'wall' not in self._classes:
        # as current data augment always need walls, add walls if it is not
        # included, but set label as -1
        # remove all walls in pipelines/formating.py/Collect
        self._category_ids_map['wall'] = -1
        self._catid_2_cat[-1] = 'wall'
      pass

  def getCatIds(self):
      return list(self._category_ids_map.values())


  def getImgIds(self):
      return list(range(len(self)))


class Stanford_Ann():
  EASY = ['Area_1/office_16']
  LONG = ['Area_1/hallway_2', 'Area_1/hallway_6' , 'Area_3/hallway_1' , 'Area_3/hallway_4' , 'Area_5/hallway_2' , 'Area_5/hallway_1']
  UNALIGNED = ['Area_2/storage_9','Area_2/hallway_11',  'Area_2/auditorium_1', 'Area_2/auditorium_2',
               'Area_3/office_8',
               'Area_4/hallway_14', 'Area_3/office_7']
  SAMPLES1 = ['Area_2/auditorium_1']
  SAMPLES2 = ['Area_1/hallway_7']
  DIFFICULT = ['Area_2/auditorium_2']
  WithColum = ['Area_5/conferenceRoom_2']
  BadColum = ['Area_5/conferenceRoom_1']
  ALIGN_GOOD = ['Area_2/conferenceRoom_1']

  GoodSamples_Area5 = ['Area_5/conferenceRoom_2', 'Area_5/hallway_2', 'Area_5/office_21', 'Area_5/office_39', 'Area_5/office_40', 'Area_5/office_41']+\
                      ['Area_3/office_7']

  Area6_Aligned = ['Area_6/copyRoom_1', 'Area_6/office_10', 'Area_6/office_11', 'Area_6/office_2', 'Area_6/office_29', 'Area_6/office_30', 'Area_6/office_32', 'Area_6/office_8']
  Area3_Aligned = ['Area_3/conferenceRoom_1', 'Area_3/hallway_2', 'Area_3/lounge_2']
  Area3_Rotated_Good = ['Area_3/office_7','Area_3/office_5']
  Area3_Rotated_NotGood = ['Area_3/office_3']
  Area4_Aligned = ['Area_4/hallway_11', 'Area_4/hallway_3']
  Area1_Aligned=  ['Area_1/office_11', 'Area_1/office_16', 'Area_1/office_29', 'Area_1/office_8']
  Area2_Aligned = ['Area_2/hallway_11', 'Area_2/hallway_5', 'Area_2/office_14', 'office_4', 'Area_2/office_5']
  Area2_Rotated = ['Area_2/auditorium_1', 'Area_2/conferenceRoom_1', 'Area_2/storage_9']

  The_GoodSamples = ['Area_4/hallway_3', 'Area_4/lobby_2', 'Area_1/office_29', 'Area_2/auditorium_1', 'Area_2/conferenceRoom_1', 'Area_2/hallway_11', 'Area_2/hallway_5', 'Area_2/office_14', 'Area_2/storage_9']

  def __init__(self, input_style, data_root, phase, obj_rep, voxel_size=None):
    assert input_style in ['pcl', 'bev']
    self.obj_rep = obj_rep
    self.input_style = input_style
    self.phase = phase
    if phase in ['train', 'test']:
      self.area_list = [1,2,3,4,6] if phase == 'train' else [5]
      if phase == 'train':
        self.area_list = [1,2,3,4,5,6]
      else:
        self.area_list = [5,6]
    else:
      self.area_list = [int(a) for a in phase]

    self.data_root = data_root
    self.voxel_size = voxel_size
    self.load_annotation()
    n = len(self.data_paths)
    print(f'\nphase: {phase}. {n} scenes loaded\n')

  def load_annotation(self, ):
    data_paths = glob.glob(os.path.join(self.data_root, "*/*.ply"))
    data_paths = [p for p in data_paths if int(p.split('Area_')[1][0]) in self.area_list]
    data_paths = [p.split(self.data_root)[1] for p in data_paths]

    #data_paths = [f+'.ply' for f in self.UNALIGNED]
    if SMALL_DATA:
      data_paths = [f+'.ply' for f in self.DIFFICULT]
      #data_paths = [f+'.ply' for f in self.BadColum]
      #data_paths = [f+'.ply' for f in self.SAMPLES1]
      #data_paths = [f+'.ply' for f in self.UNALIGNED]
      #data_paths = [f+'.ply' for f in self.ALIGN_GOOD]

    if NO_LONG:
      data_paths_new = []
      for p in data_paths:
        ok = 1
        for l in self.LONG:
          if l in p:
            ok = 0
        if ok:
          data_paths_new.append(p)
      n_long = len(data_paths_new) - len(data_paths)
      print(f'{n_long} long scenes rmed')
      data_paths = data_paths_new
      pass

    data_paths.sort()
    self.data_paths = data_paths

    #data_roots = [f.replace('ply', 'npy') for f in pcl_files]
    n = len(self.data_paths)
    self.img_infos = []
    for i in range(n):
      pcl_file = os.path.join(self.data_root, self.data_paths[i])
      ann_filename = pcl_file.replace('.ply', '-boxes.npy')
      bev_filename = pcl_file.replace('.ply', '-topview.npy')
      anno = load_bboxes( pcl_file, self._classes, self._category_ids_map, self.obj_rep, self.input_style )
      #anno_2d = anno_raw_to_anno_topview(anno_raw, self._classes, self.input_style)

      if self.input_style == 'bev':
          img_info = {'filename': bev_filename,
                      'ann': anno,
                      }

      if self.input_style == 'pcl':
          raw_dynamic_vox_size = (anno['pcl_scope'][1] - anno['pcl_scope'][0]) / self.voxel_size
          raw_dynamic_vox_size = np.ceil(raw_dynamic_vox_size).astype(np.int32)
          raw_dynamic_vox_size = tuple(raw_dynamic_vox_size.tolist())

          img_meta = dict(filename = anno['filename'],
                          pcl_scope = anno['pcl_scope'],
                          input_style=self.input_style,
                          classes = self._classes,
                          scale_factor = 1,
                          data_aug = {},
                          )
          img_meta['voxel_size'] = self.voxel_size
          img_meta['raw_dynamic_vox_size'] = raw_dynamic_vox_size

          img_info = dict(
            img_meta = img_meta,
            gt_bboxes_2d_raw = anno['gt_bboxes'],
            gt_labels = anno['labels'],
            gt_bboxes_3d_raw = anno['bboxes_XYXYSin2WZ0Z1'],
          )
      self.img_infos.append(img_info)
    pass

  def __len__(self):
    return len(self.img_infos)


  def show_gt_bboxes(self,):
    for i in range(len(self)):
      if self.input_style == 'pcl':
        self.show_gt_bboxes_1scene_3d(i)
      if self.input_style == 'bev':
        self.show_gt_bboxes_1scene_bev(i)

  def show_gt_bboxes_1scene_bev(self, index):
    img_info = self.img_infos[index]
    print('\n\n', img_info['filename'])
    ann = img_info['ann']
    gt_bboxes_3d_raw = ann['bboxes_XYXYSin2WZ0Z1']
    gt_bboxes = ann['gt_bboxes']
    gt_labels = ann['labels']

    rm_cats = ['ceiling']
    #rm_cats = ['beam']
    kp_cats = ['wall', 'beam', 'column', 'door', 'window']

    # 2d
    gt_corners = OBJ_REPS_PARSE.encode_obj(gt_bboxes, self.obj_rep, 'RoLine2D_2p')
    gt_corners, _ = filter_categories('keep', gt_corners, gt_labels, kp_cats, self._category_ids_map)
    gt_corners = gt_corners.reshape(-1,2)
    w, h = gt_corners.max(0).astype(np.int) + 10
    gt_bboxes, gt_labels = filter_categories('keep', gt_bboxes, gt_labels, kp_cats, self._category_ids_map)
    obj_cats = np.array(kp_cats)[gt_labels]
    _show_objs_ls_points_ls((h,w), [gt_bboxes], self.obj_rep, points_ls=[gt_corners], obj_scores_ls=[obj_cats])


  def show_gt_bboxes_1scene_3d(self, index):
    img_info = self.img_infos[index]
    print('\n\n', img_info['img_meta']['filename'])
    gt_bboxes_3d_raw = img_info['gt_bboxes_3d_raw']
    gt_labels = img_info['gt_labels']
    coords, colors_norms, point_labels, _ = self.load_ply(index)

    kp_cats = ['wall', 'beam', 'column', 'door', 'window']
    rm_cats = ['ceiling']

    # 3d
    points = np.concatenate([coords, colors_norms], axis=1)
    points, point_labels = filter_categories('remove', points, point_labels, ['ceiling'], _raw_pcl_category_ids_map)
    #gt_bboxes_3d_raw, gt_labels = filter_categories('remove', gt_bboxes_3d_raw, gt_labels, ['ceiling', 'room', 'floor','door'], self._category_ids_map)
    gt_bboxes_3d_raw, gt_labels = filter_categories('keep', gt_bboxes_3d_raw, gt_labels, ['wall', 'beam', 'column', 'door', 'window'], self._category_ids_map)
    gt_cats = [self._catid_2_cat[l] for l in gt_labels]
    print(gt_cats)

    corners = OBJ_REPS_PARSE.encode_obj(gt_bboxes_3d_raw, 'XYXYSin2WZ0Z1', 'Bottom_Corners').reshape(-1,3)

    _show_3d_points_objs_ls([points[:,:3]], [points[:,3:6]], objs_ls=[gt_bboxes_3d_raw], obj_rep='XYXYSin2WZ0Z1')
    _show_3d_points_objs_ls([corners], objs_ls=[gt_bboxes_3d_raw], obj_rep='XYXYSin2WZ0Z1', obj_colors=[gt_labels])

    if self.obj_rep == 'XYXYSin2WZ0Z1':
      sin2 = gt_bboxes_3d_raw[:,4].copy()
      gt_bboxes_3d_raw /= 0.01
      gt_bboxes_3d_raw[:,4] = sin2
    elif self.obj_rep == 'XYZLgWsHA':
      theta = gt_bboxes_3d_raw[:,6].copy()
      gt_bboxes_3d_raw /= 0.01
      gt_bboxes_3d_raw[:,6] = theta
    else:
      raise NotImplementedError
    corners /= 0.01
    obj_cats = np.array(kp_cats)[gt_labels]
    _show_objs_ls_points_ls((1024, 1024), [gt_bboxes_3d_raw], 'XYXYSin2WZ0Z1', points_ls=[corners], obj_scores_ls=[obj_cats])
    pass

class StanfordPcl(VoxelDatasetBase, Stanford_CLSINFO, Stanford_Ann):
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


  def __init__(self,
               obj_rep,
               ann_file='data/stanford/',
               img_prefix='data/stanford/ply/train.txt',
               classes = ['wall'],
               pipeline=None,
               voxel_size=None,
               auto_scale_vs = True,
               max_num_points = None,
               max_footprint_for_scale = None,
               augment_data = None,
               data_types = ['color', 'norm', 'xyz'],
               bev_pad_pixels = 0,
               filter_edges = True,
               test_mode = False,
               ):
    assert voxel_size is not None
    self.obj_rep = obj_rep
    Stanford_CLSINFO.__init__(self, classes)
    data_root = ann_file
    phase = img_prefix = img_prefix.split('/')[-1].split('.txt')[0]
    Stanford_Ann.__init__(self, 'pcl', data_root, phase, obj_rep, voxel_size=voxel_size)

    self.save_sparse_input_for_debug = 0
    self.VOXEL_SIZE = self.voxel_size = voxel_size
    self.bev_pad_pixels = bev_pad_pixels
    if max_num_points is not None:
     max_num_points = int(max_num_points)
    self.max_num_points = max_num_points
    self.max_footprint_for_scale = max_footprint_for_scale
    self.max_voxel_footprint = max_footprint_for_scale / voxel_size / voxel_size if max_footprint_for_scale is not None else None
    self.load_voxlized_sparse = DEBUG_CFG.LOAD_VOXELIZED_SPARSE
    #phase = DatasetPhase.Train if img_prefix == 'train' else DatasetPhase.Test
    self.data_config = DataConfig(phase, augment_data)
    self._set_group_flag()
    print(f'\n Area {img_prefix}: load {len(self)} files for areas {self.area_list}\n')
    VoxelDatasetBase.__init__(self, phase, self.data_paths, self.data_config)

    self.data_types = data_types
    all_inds = dict(color=[0,1,2], norm=[3,4,5], xyz=[6,7,8])
    self.data_channel_inds = np.array([all_inds[dt] for dt in self.data_types]).reshape(-1)
    pass

  def _set_group_flag(self):
    self.flag = np.zeros(len(self), dtype=np.uint8)

  def load_ply(self, index):
    filepath = self.data_root / self.data_paths[index]
    coords, colors_norms, point_labels, _ = load_1_ply(filepath)
    normpath = str(filepath).replace('.ply', '-norm.npy')
    norm = np.load(normpath)
    colors_norms = np.concatenate([colors_norms, norm], axis=1)

    np0 = coords.shape[0]
    if self.max_num_points is not None and  np0 > self.max_num_points:
      inds = np.random.choice(np0, self.max_num_points, replace=False)
      inds.sort()
      coords = coords[inds]
      colors_norms = colors_norms[inds]
      point_labels = point_labels[inds]

    pcl_scope = self.img_infos[index]['img_meta']['pcl_scope']
    coords -= pcl_scope[0:1]

    #bboxes_3d = self.img_infos[index]['gt_bboxes_3d_raw']
    #_show_3d_points_objs_ls([coords], None, [bboxes_3d], 'XYXYSin2WZ0Z1')
    return coords, colors_norms, point_labels, None

  def _augment_coords_to_colors_norms(self, coords, colors_norms, point_labels=None):
    # Center x,y
    coords_center = coords.mean(0, keepdims=True)
    coords_center[0, 2] = 0
    norm_coords = coords - coords_center
    colors_norms = np.concatenate((colors_norms, norm_coords), 1)
    return coords, colors_norms, point_labels

  def _normalization(self, colors_norms):
    colors_norms[:,:3] = colors_norms[:,:3] / 255. - 0.5
    return colors_norms

  def select_data_types(self, colors_norms):
    '''
    do this at the last step
    '''
    assert colors_norms.shape[1] == 9
    return colors_norms[:, self.data_channel_inds]



  def show_topview_gts(self, voxel_size_prj=0.01):
    for i in range(len(self)):
      self.show_topview_gt_1scene(i, voxel_size_prj)

  def show_topview_gt_1scene(self, index, voxel_size_prj):
    ply_filename = self.img_infos[index]['img_meta']['filename']
    topview_file = ply_filename.replace('.ply', '-topview.npy')
    topview = np.load(topview_file, allow_pickle=True)
    gt_bboxes_2d_raw = self.img_infos[index]['gt_bboxes_2d_raw']
    gt_bboxes = gt_bboxes_2d_raw.copy()
    gt_bboxes[:,:4] /= voxel_size_prj

    density = topview[:,:,0].astype(np.uint8)
    _show_objs_ls_points_ls( density, [gt_bboxes], 'RoLine2D_UpRight_xyxy_sin2a' )
    _show_objs_ls_points_ls( (1024,1024), [gt_bboxes], 'RoLine2D_UpRight_xyxy_sin2a' )
    pass

  def gen_topviews(self, voxel_size_prj=0.01):
    for i in range(len(self)):
      self.gen_topview_1scene(i, voxel_size_prj)

  def gen_topview_1scene(self, index, voxel_size_prj):
    coords, colors_norms, point_labels, _ = self.load_ply(index)
    points = coords.copy()
    #points = np.concatenate([coords, colors_norms], axis=1)
    n = coords.shape[0]

    coords = np.round(coords / voxel_size_prj).astype(np.int32)[:,:2]
    inds, unique_inds, unique_inverse, density =  np.unique(coords, return_index=True, return_inverse=True, return_counts=True, axis=0)
    max_density = density.max()
    ref_full_density = max_density / 2
    m = inds.shape[0]

    norms_mean = np.zeros([m, 3])
    for i in range(n):
      j = unique_inverse[i]
      norms_mean[j]  += colors_norms[i, 3:6]

    norms_mean /= density[:,None]

    min_coords = coords.min(0)
    assert np.all(min_coords == np.array([0,0]))
    max_coords = coords.max(0)
    # the first dim is y, and the second is x
    img_size = (max_coords[1]+1, max_coords[0]+1, 4)
    topview = np.zeros(img_size, dtype=np.float32)
    topview[inds[:,1], inds[:,0], 0] = density / ref_full_density * 255
    topview[inds[:,1], inds[:,0], 1:4] =  norms_mean

    ply_filename = self.img_infos[index]['img_meta']['filename']
    out_file = ply_filename.replace('.ply', '-density.png')
    _show_objs_ls_points_ls(topview[:,:,0], out_file = out_file, only_save=True, obj_rep='RoLine2D_UpRight_xyxy_sin2a')
    norm_img = (np.abs(topview[:,:,1:4])*255).astype(np.uint8)
    out_file = ply_filename.replace('.ply', '-norm.png')
    _show_objs_ls_points_ls(norm_img, out_file = out_file, only_save=True, obj_rep='RoLine2D_UpRight_xyxy_sin2a' )


    topview_file = ply_filename.replace('.ply', '-topview.npy')
    np.save(topview_file, topview)


    #gt_bboxes_3d_raw = self.img_infos[index]['gt_bboxes_3d_raw']
    #_show_3d_points_objs_ls([points], objs_ls=[gt_bboxes_3d_raw], obj_rep='XYXYSin2WZ0Z1')

    gt_labels = self.img_infos[index]['gt_labels']
    gt_bboxes_2d_raw = self.img_infos[index]['gt_lines_thick_2d_raw']
    gt_bboxes_2d = gt_bboxes_2d_raw.copy()
    gt_bboxes_2d[:,:4] /= voxel_size_prj
    gt_bboxes_2d[:,5] /= voxel_size_prj
    out_file = ply_filename.replace('.ply', '-gt.png')
    _show_objs_ls_points_ls( topview[:,:,0], [gt_bboxes_2d], obj_colors=[gt_labels], obj_rep='RoBox2D_UpRight_xyxy_sin2a_thick', out_file=out_file, only_save=1 )

    print('\n\t', topview_file)
    pass


class Stanford_BEV(Stanford_CLSINFO, Stanford_Ann):
    def __init__(self,
                 obj_rep,
                 anno_folder='data/stanford/',
                 img_prefix='data/beike/processed_512/TopView_VerD/train.txt',
                 test_mode=False,
                 filter_edges=True,
                 classes = ['wall', ],
                 ):
        self.obj_rep = obj_rep
        Stanford_CLSINFO.__init__(self, classes, always_load_walls=1)
        phase = img_prefix
        Stanford_Ann.__init__(self, 'bev', anno_folder, phase)


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


def Unused_anno_raw_to_anno_topview(anno_raw, classes, input_style):
  anno_raw['classes'] = classes
  bbox_cat_ids = anno_raw['bbox_cat_ids']

  anno_2d = {}
  anno_2d['filename'] = anno_raw['filename']
  anno_2d['labels'] = bbox_cat_ids
  anno_2d['bboxes'] = anno_raw['lines_2d']
  anno_2d['classes'] = [c for c in classes if c!='background']
  #anno_2d['relations'] = None

  if input_style == 'bev':
    voxel_size_prj=0.01
    anno_2d['bboxes'][:, :4] /= voxel_size_prj

  #show_bboxes(anno_raw['bboxes_3d'])
  return anno_2d


def load_bboxes(pcl_file, classes, _category_ids_map, obj_rep, input_style):
  anno_file = pcl_file.replace('.ply', '.npy').replace('Area_', 'Boxes_Area_')
  scope_file = pcl_file.replace('.ply', '-scope.txt')

  bboxes_dict = np.load(anno_file, allow_pickle=True).tolist()
  if 'clutter' in bboxes_dict:
    bboxes_dict['background'] = bboxes_dict['clutter']
    del bboxes_dict['clutter']
  bboxes = []
  bbox_cat_ids = []
  for cat in bboxes_dict:
    if cat in _category_ids_map:
      bboxes.append( bboxes_dict[cat] )
      num_box = bboxes_dict[cat].shape[0]
      cat_ids = _category_ids_map[cat] * np.ones([num_box], dtype=np.int64)
      bbox_cat_ids.append( cat_ids )
  # the loaded format: XYXYSin2WZ0Z1
  bboxes_XYXYSin2WZ0Z1 = np.concatenate(bboxes, axis=0)
  bbox_cat_ids = np.concatenate(bbox_cat_ids)

  scope = np.loadtxt(scope_file)

  bboxes_XYXYSin2WZ0Z1[:, :2] -= scope[0:1,:2]
  bboxes_XYXYSin2WZ0Z1[:, 2:4] -= scope[0:1,:2]
  bboxes_XYXYSin2WZ0Z1[:, 6] -= scope[0:1,2]
  bboxes_XYXYSin2WZ0Z1[:, 7] -= scope[0:1,2]

  gt_bboxes = OBJ_REPS_PARSE.encode_obj(bboxes_XYXYSin2WZ0Z1, 'XYXYSin2WZ0Z1', obj_rep)
  anno = {}
  if input_style == 'bev':
    voxel_size_prj=0.01
    if obj_rep == 'XYXYSin2' or obj_rep == 'XYXYSin2WZ0Z1':
      sin2 = gt_bboxes[:,4].copy()
      gt_bboxes /= voxel_size_prj
      gt_bboxes[:,4] = sin2
    elif obj_rep == 'XYZLgWsHA':
      gt_bboxes[:,:6] /= voxel_size_prj
    elif obj_rep == 'XYLgWsAbsSin2Z0Z1':
      rotation = gt_bboxes[:,4:6].copy()
      gt_bboxes /= voxel_size_prj
      gt_bboxes[:,4:6] = rotation
    elif obj_rep == 'XYDAsinAsinSin2Z0Z1':
      rotation = gt_bboxes[:,3:6].copy()
      gt_bboxes /= voxel_size_prj
      gt_bboxes[:,3:6] = rotation
    elif obj_rep == 'Rect4CornersZ0Z1':
      gt_bboxes /= voxel_size_prj
    else:
      raise NotImplementedError
    anno['voxel_size_prj'] = voxel_size_prj

  filename = pcl_file
  # lines_2d: RoLine2D_UpRight_xyxy_sin2a
  anno['pcl_scope'] = scope
  anno['filename'] = filename
  anno['classes'] = [c for c in classes if c!='background']
  anno['bboxes_XYXYSin2WZ0Z1'] = bboxes_XYXYSin2WZ0Z1
  anno['gt_bboxes'] = gt_bboxes
  anno['labels'] = bbox_cat_ids

  show = 0
  if show:
    print(f'\n\n{pcl_file}\n\n')
    _show_3d_points_objs_ls(objs_ls=[bboxes_XYXYSin2WZ0Z1], obj_rep='XYXYSin2WZ0Z1')
    _show_3d_points_objs_ls(objs_ls=[gt_bboxes], obj_rep=obj_rep)
    w,h =  gt_bboxes[:,:4].reshape(-1,2).max(0).astype(np.int) + 10
    _show_objs_ls_points_ls( (h,w), [gt_bboxes], obj_rep=obj_rep )

  return anno


def show_bboxes(bboxes_3d):
    #_show_3d_points_objs_ls(None, None, [bboxes_3d],  obj_rep='XYXYSin2WZ0Z1')
    bboxes_show = bboxes_3d.copy()
    voxel_size = 0.02
    bboxes_show[:,:4] /= voxel_size
    bboxes_show[:,:4] += 10
    _show_objs_ls_points_ls( (512,512), [bboxes_show[:,:5]], obj_rep='RoLine2D_UpRight_xyxy_sin2a' )
    #_show_objs_ls_points_ls( (512,512), [bboxes_show[:,:6]], obj_rep='RoBox2D_UpRight_xyxy_sin2a_thick' )
    import pdb; pdb.set_trace()  # XXX BREAKPOINT


def filter_categories(event, bboxes, labels, filter_cats, category_ids_map):
  assert event in ['remove','keep']
  num0 = len(category_ids_map)
  num1 = len(filter_cats)
  labels_map = -np.ones([num0])
  filter_labels = [category_ids_map[c] for c in filter_cats]
  if event == 'remove':
    keep_labels = [category_ids_map[c] for c in category_ids_map if c not in filter_cats]
  else:
    keep_labels = filter_labels
  labels_map[keep_labels] = np.arange(num1)

  n = bboxes.shape[0]
  remain_mask = np.zeros(n)==1
  for l in keep_labels:
    mask = labels == l
    remain_mask[mask] = True
  new_bboxes =  bboxes[remain_mask]
  new_labels = labels_map[ labels[remain_mask] ].astype(np.int32)
  assert new_labels.min() >= 0
  return new_bboxes, new_labels


def load_1_ply(filepath):
    plydata = PlyData.read(filepath)
    data = plydata.elements[0].data
    coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
    colors_norms = np.array([data['red'], data['green'], data['blue']], dtype=np.float32).T
    point_labels = np.array(data['label'], dtype=np.int32)
    instance = np.array(data['instance'], dtype=np.int32)
    return coords, colors_norms, point_labels, None


def main3d():
  obj_rep = 'XYZLgWsHA'
  ann_file = '/home/z/Research/mmdetection/data/stanford/'
  img_prefix = './train.txt'
  #img_prefix = './test.txt'
  classes = Stanford_CLSINFO.classes_order
  max_num_points = 50 * 1e4
  max_num_points = None
  sfd_dataset = StanfordPcl(obj_rep, ann_file, img_prefix, voxel_size=0.02, classes = classes, max_num_points=max_num_points)
  #sfd_dataset.gen_topviews()
  #sfd_dataset.show_topview_gts()
  sfd_dataset.show_gt_bboxes()
  pass

def main2d():
  obj_rep = 'XYZLgWsHA'
  ann_file = '/home/z/Research/mmdetection/data/stanford/'
  img_prefix = './train.txt'
  img_prefix = 'test'
  classes = Stanford_CLSINFO.classes_order
  sfd_dataset = Stanford_BEV(obj_rep, ann_file, img_prefix,  classes = classes)
  #sfd_dataset.gen_topviews()
  #sfd_dataset.show_topview_gts()
  sfd_dataset.show_gt_bboxes()
  pass

if __name__ == '__main__':
   main3d()
   #main2d()


