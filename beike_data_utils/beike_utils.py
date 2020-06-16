# from floor-sp
import open3d as o3d
import os
import os.path as osp
import sys
import numpy as np
import cv2
from plyfile import PlyData, PlyElement
import random
import json
import math
import copy
from collections import defaultdict
import time
import mmcv
import glob
from multiprocessing import Pool
from functools import partial


from obj_geo_utils.obj_utils import OBJ_REPS_PARSE, find_wall_wall_connection,\
  find_wall_wd_connection
from obj_geo_utils.line_operations import rotate_bboxes_img, transfer_lines, gen_corners_from_lines_np
from configs.common import DIM_PARSE, DEBUG_CFG
from tools.visual_utils import _show_objs_ls_points_ls, _show_3d_points_objs_ls
np.set_printoptions(precision=3, suppress=True)


PCL_LINE_BOUND_METER = 1
PCL_LINE_BOUND_PIXEL = PCL_LINE_BOUND_METER / 0.04

#LOAD_CLASSES = ['wall', 'window', 'door']
#LOAD_CLASSES = ['wall']

DEBUG = True
#UNALIGNED_SCENES =  ['7w6zvVsOBAQK4h4Bne7caQ', 'IDZkUGse-74FIy2OqM2u_Y',
#                    'B9Abt6B78a0j2eRcygHjqC', 'Akkq4Ch_48pVUAum3ooSnK',
#                    'w2BaBfwjX0iN2cMjvpUNfa', 'yY5OzetjnLred7G8oOzZr1',
#                    'wIGxjDDGkPk3udZW4vo-ic', 'vZIjhovtYde9e2qUjMzvz3']
gt_out_pcl = ['vZIjhovtYde9e2qUjMzvz3', 'pMh3-KweDzzp_b_HI11eMA',
              'vYlCbx-H_v_uvacuiMq0no']
lost_gt_scenes = ['vYlCbx-H_v_uvacuiMq0no']
#BAD_SCENES = ['7w6zvVsOBAQK4h4Bne7caQ', 'IDZkUGse-74FIy2OqM2u_Y','B9Abt6B78a0j2eRcygHjqC']
BAD_SCENES = []
PRE_LOAD_ALL = False

BAD_SCENE_TRANSFERS_PCL  = {'7w6zvVsOBAQK4h4Bne7caQ': (-44, -2.071 - 0.2, -1.159 - 0.5),
                            'IDZkUGse-74FIy2OqM2u_Y': (30, -0.788 - 0.45, 0.681 + 0.08),
                            'B9Abt6B78a0j2eRcygHjqC': (44, -0.521 + 0.1, 0.928 + 0.1),
                            'Akkq4Ch_48pVUAum3ooSnK': (2.5, 0.108, 0.000),
                            'w2BaBfwjX0iN2cMjvpUNfa': (2, 0.110, 0.028+0.05),
                            'yY5OzetjnLred7G8oOzZr1': (-1.3, -0.1, 0.000),
                            'wIGxjDDGkPk3udZW4vo-ic': (-1, 0.074, 0.048),
                            'vZIjhovtYde9e2qUjMzvz3': (-1, 0.078, 0.000),
                            'wcSLwyAKZafnozTPsaQMyv': (-1,-0.1,-0.05),
                            }

class BEIKE_CLSINFO(object):
  def __init__(self, classes_in, always_load_walls=1):
      classes_order = ['background', 'wall', 'door', 'window', 'room', 'other']
      assert all([c in classes_order for c in classes_in])
      classes = [c for c in classes_order if c in classes_in]
      if 'background' not in classes:
        classes = ['background']+ classes
      n = len(classes)
      self._classes = classes
      self.CLASSES = classes
      self.cat_ids = list(range(n))
      self._category_ids_map = {classes[i]:i for i in range(n)}
      self._catid_2_cat = {i:classes[i] for i in range(n)}
      self._labels = self.cat_ids

      if always_load_walls and 'wall' not in self._classes:
        # as current data augment always need walls, add walls if it is not
        # included, but set label as -1
        # remove all walls in pipelines/formating.py/Collect
        self._category_ids_map['wall'] = -1
        self._catid_2_cat[-1] = 'wall'

      pass

class BEIKE(BEIKE_CLSINFO):
    edge_atts = ['thickness','curve', 'align', 'type', 'edgeComputed', 'thicknessComputed', 'offsetComputed', 'isLoadBearing']
    edge_atts = []
    edge_attributions =  ['e_'+a for a in edge_atts]

    def __init__(self,
                 obj_rep,
                 anno_folder='data/beike/processed_512/json/',
                 img_prefix='data/beike/processed_512/TopView_VerD/train.txt',
                 test_mode=False,
                 filter_edges=True,
                 classes = ['wall', ],
                 is_save_connection = False,
                 ):
        t0 = time.time()
        print(f'\nStart loading data: {img_prefix}')
        super().__init__(classes, always_load_walls=1)
        assert  anno_folder[-5:] == 'json/'
        self.obj_rep = obj_rep
        self.anno_folder = anno_folder
        self.test_mode = test_mode
        phase = os.path.basename(img_prefix)
        self.classes = classes
        self.is_save_connection = is_save_connection
        if is_save_connection:
          filter_edges = False
          phase = 'all.txt'
        self.filter_edges = filter_edges

        self.img_prefix = os.path.dirname( img_prefix )
        self.data_dir = os.path.dirname(os.path.dirname(anno_folder))
        self.split_file = os.path.join(self.data_dir, phase)
        raw_scene_list = np.loadtxt(self.split_file,str).tolist()
        if isinstance(raw_scene_list, str):
          raw_scene_list = [raw_scene_list]
        n0 = len(raw_scene_list)
        self.is_pcl = os.path.basename(self.img_prefix) == 'ply'
        data_format = '.ply' if self.is_pcl else '.npy'

        self.seperate_room_path = anno_folder.replace('json', 'seperate_room_data/test')
        #self.seperate_room_data_path = anno_folder.replace('json', 'seperate_room_data/test')
        base_path = os.path.dirname( os.path.dirname(anno_folder) )

        self.img_infos = []
        for scene_name in raw_scene_list:
            if scene_name in BAD_SCENES:
              continue
            json_file = self.get_json_file(scene_name)
            tv_file = self.get_topview_file(scene_name)
            scope_file = self.get_scope_file(scene_name)
            rel_file = self.get_relation_file(scene_name)
            valid = os.path.exists(json_file) and os.path.exists(tv_file) and os.path.exists(scope_file)
            if not valid:
              continue
            if not (is_save_connection or os.path.exists(rel_file)):
              continue
            filename = scene_name + data_format
            img_info = {'filename': filename,}
            self.img_infos.append(img_info)

        self.num_scenes = len(self.img_infos)
        if PRE_LOAD_ALL:
            for i in range(self.num_scenes):
              self.load_1_anno(i)

        t = time.time() - t0
        dstr = f'Data loading finished. {self.num_scenes} valid in {n0},  time : {t:.3f}s\n'
        assert self.num_scenes > 0, dstr
        print(dstr)
        pass

    def get_json_file(self, scene_name):
        return os.path.join(self.data_dir, 'json', scene_name + '.json')
    def get_topview_file(self, scene_name):
        return os.path.join(self.data_dir, 'TopView_VerD', scene_name + '.npy')
    def get_scope_file(self, scene_name):
        return os.path.join(self.data_dir, 'pcl_scopes', scene_name + '.txt')
    def get_relation_file(self, scene_name):
        return os.path.join(self.data_dir, 'relations', scene_name + '.npy')

    def show_summary(self, idx):
      img_info = self.img_infos[idx]
      anno = img_info['ann_raw']

      anno = self.img_infos[idx]['ann_raw']
      scene_name = anno['filename']
      corners =  anno['corners']
      corner_cat_ids = anno['corner_cat_ids']
      lines =  anno['lines']
      line_cat_ids = anno['line_cat_ids']
      print(f'\n{scene_name}')
      print(f'corner num = {corners.shape[0]}')
      print(f'line num = {lines.shape[0]}')

      line_length_min_mean_max = anno['line_length_min_mean_max']
      print(f'line_length_min_mean_max: {line_length_min_mean_max}')

    def __len__(self):
      return self.num_scenes

    def load_1_anno(self, idx):
      jfn = self.img_infos[idx]['filename'].split('.')[0] + '.json'
      anno_raw = load_anno_1scene(self.anno_folder, jfn,
                            self._classes, filter_edges=self.filter_edges,
                            is_save_connection = self.is_save_connection)

      if 'room' in self._category_ids_map:
        room_label = self._category_ids_map['room']
      else:
        room_label = None
      anno_img = raw_anno_to_img(self.classes, room_label, self.obj_rep, anno_raw, 'topview', {'img_size': DIM_PARSE.IMAGE_SIZE}, self.anno_folder)
      self.img_infos[idx]['ann'] = anno_img
      #self.img_infos[idx]['ann_raw'] = anno_raw
      return anno_img


    def getCatIds(self):
      return list(self._category_ids_map.values())

    def getImgIds(self):
      return list(range(len(self)))


    def get_scene_index(self, scene_name):
      for i,img_info in enumerate(self.img_infos):
        if img_info['filename'].split('.')[0] == scene_name:
          return i
      assert True, f'cannot fine scene {scene_name}'

    def load_data(self, scene_name):
      file_name = os.path.join(self.img_prefix, scene_name+'.npy')
      return load_topview_img(file_name)

    def load_data_by_idx(self, idx):
      filename = self.img_infos[idx]
      scene_name = filename.split('.')[0]
      return self.load_data(scene_name)

    @ staticmethod
    def add_anno_summary(anno):
      corners =  anno['corners']
      corner_cat_ids = anno['corner_cat_ids']
      lines =  anno['lines']
      line_cat_ids = anno['line_cat_ids']
      line_lengths = np.linalg.norm(lines[:,0] - lines[:,1], axis=1)
      min_leng = line_lengths.min()
      max_leng = line_lengths.max()
      n = lines.shape[0]
      print(f'{n} lines\nline_lengths: {line_lengths}, \nmin_leng: {min_leng}, max_leng: {max_leng}')

      ids = np.where( line_lengths == line_lengths.min() )[0]
      print(f'line with min length:\n {lines[ids]}')
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass


    def show_anno_img(self, idx,  with_img=True, rotate_angle=0, lines_transfer=(0,0,0), write=False):
      img_dir = self.img_prefix + '_Imgs'
      colors_line   = {'wall': (0,0,255), 'door': (0,255,255),
                       'window': (0,255,255), 'other':(100,100,0)}
      colors_corner = {'wall': (0,0,255), 'door': (0,255,0),
                       'window': (255,0,0), 'other':(255,255,255)}
      self.load_1_anno(idx)
      anno = self.img_infos[idx]['ann']
      bboxes = anno['gt_bboxes']
      labels = anno['labels']

      scene_name = self.img_infos[idx]['filename'].split('.')[0]
      anno_img_file = os.path.join(img_dir, scene_name+'-density.png')
      if os.path.exists(anno_img_file):
        return

      print(f'{scene_name}')

      if not with_img:
        img = np.zeros((DIM_PARSE.IMAGE_SIZE, DIM_PARSE.IMAGE_SIZE, 3), dtype=np.uint8)
        img = (DIM_PARSE.IMAGE_SIZE, DIM_PARSE.IMAGE_SIZE)
      else:
        img = self.load_data(scene_name)

      if (np.array(lines_transfer) != 0).any():
        angle, cx, cy = lines_transfer
        bboxes = transfer_lines(bboxes, DIM_PARSE.OBJ_REP, img.shape[:2], angle, (cx,cy))

      if rotate_angle != 0:
        bboxes, img = rotate_bboxes_img(bboxes, img, rotate_angle, self.obj_rep)

      room_mask = labels==4
      room_bboxes = bboxes[room_mask]
      bboxes = bboxes[room_mask==False]
      labels = labels[room_mask==False]
      cor_labels = np.vstack([labels, labels]).T.reshape(-1)


      corners = OBJ_REPS_PARSE.encode_obj(bboxes, self.obj_rep, 'RoLine2D_2p').reshape(-1,2)
      # draw rooms
      if write:
        anno_img_file = os.path.join(img_dir, scene_name+'-room.png')
      else:
        anno_img_file = None
      _show_objs_ls_points_ls(
        img, [bboxes], self.obj_rep, [corners],
                               obj_colors='random', point_colors=[cor_labels],
                               obj_thickness=2, point_thickness=2,
                               draw_rooms = True,
                               out_file=anno_img_file, only_save=write)
      if write:
        anno_img_file = os.path.join(img_dir, scene_name+'-room_box.png')
      else:
        anno_img_file = None
      _show_objs_ls_points_ls(
        img[:,:,0], [room_bboxes], self.obj_rep, [corners],
                               obj_colors='random', point_colors=[cor_labels],
                               obj_thickness=6, point_thickness=2,
                               out_file=anno_img_file, only_save=write)

      # draw density
      if write:
        anno_img_file = os.path.join(img_dir, scene_name+'-density.png')
      else:
        anno_img_file = None
      _show_objs_ls_points_ls(
                               img[:,:,0], [bboxes], self.obj_rep, [corners],
                               obj_colors='random', point_colors=[cor_labels],
                               obj_thickness=1, point_thickness=2,
                               out_file=anno_img_file, only_save=write)
      # draw normal
      img_norm = np.abs(img[:,:,1:]) * 255
      if write:
        anno_img_file = os.path.join(img_dir, scene_name+'-norm.png')
      _show_objs_ls_points_ls(
        img_norm, [bboxes], self.obj_rep, [corners],
                               obj_colors='white', point_colors=[cor_labels],
                               obj_thickness=1, point_thickness=2,
                               out_file=anno_img_file, only_save=write)

      show_1by1 = False
      if show_1by1:
        for k in range(bboxes.shape[0]):
            print(f'{k}')
            for ele in self.edge_attributions:
              if anno[ele].shape[0] > k:
                print(f'{ele}: {anno[ele][k]}')
            _show_lines_ls_points_ls(img, [bboxes[k:k+1]], [corners],
                                line_colors='random', point_colors='random',
                                line_thickness=1, point_thickness=1,
                                out_file=anno_img_file, only_save=0)
            pass
      return img


    def draw_anno_raw(self, idx,  with_img=True):
      self.show_summary(idx)
      anno = self.img_infos[idx]['ann_raw']
      if not with_img:
        img = None
      else:
        scene_name = anno['filename'].split('.')[0]
        img = self.load_data(scene_name)

        mask = img[:,:,1] == 0
        img[:,:,1] = mask * 30
        img[:,:,0] = mask * 30

      print('draw_anno in beike_data_utils/beike_utils.py')
      colors_corner = {'wall': (0,0,255), 'door': (0,255,0), 'window': (255,0,0), 'other':(255,255,255)}
      colors_line   = {'wall': (255,0,0), 'door': (0,255,255), 'window': (0,255,255), 'other':(100,100,0)}

      scene_name = anno['filename']
      corners =  anno['corners']
      corner_cat_ids = anno['corner_cat_ids']
      lines =  anno['lines']
      line_cat_ids = anno['line_cat_ids']
      #print(f'line_cat_ids: {line_cat_ids}')

      corners, lines, _ = meter_2_pixel('topview', {'img_size': DIM_PARSE.IMAGE_SIZE}, corners, lines, pcl_scope=anno['pcl_scope'], floor=True)

      if img is None:
        img = np.zeros([DIM_PARSE.IMAGE_SIZE, DIM_PARSE.IMAGE_SIZE, 3], dtype=np.uint8)
      for i in range(corners.shape[0]):
        obj = self._catid_2_cat[ corner_cat_ids[i] ]
        cv2.circle(img, (corners[i][0], corners[i][1]), 2, colors_corner[obj], -1)

      for i in range(lines.shape[0]):
        obj = self._catid_2_cat[ line_cat_ids[i] ]
        s, e = lines[i]
        cv2.line(img, (s[0], s[1]), (e[0], e[1]), colors_line[obj])
        if obj != 'wall':
          cv2.putText(img, obj, (s[0], s[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    (255, 255, 255), 1)

      #mmcv.imshow(img)
      return img

    def show_scene_anno(self, scene_name, with_img=True, rotate_angle=0, lines_transfer=(0,0,0), write=False):
      idx = None
      for i in range(len(self)):
        sn = self.img_infos[i]['filename'].split('.')[0]
        if sn == scene_name:
          idx = i
          break
      assert idx is not None, f'cannot find {scene_name}'
      self.show_anno_img(idx, with_img, rotate_angle, lines_transfer, write=write)


    def find_unscanned_edges(self,):
      self.line_density_sum_mean_dir = self.anno_folder.replace('json', 'line_density_sum_mean')
      if not os.path.exists(self.line_density_sum_mean_dir):
        os.makedirs( self.line_density_sum_mean_dir )
      for i in range(len(self)):
        self.find_unscanned_edges_1_scene(i)

    def find_unscanned_edges_1_scene(self, idx):
      from obj_geo_utils.line_operations import getOrientedLineRectSubPix
      self.load_1_anno(idx)
      anno = self.img_infos[idx]['ann']
      bboxes = anno['gt_bboxes']
      labels = anno['labels']
      corners,_,_,_ = gen_corners_from_lines_np(bboxes, None, self.obj_rep, 2)

      scene_name = self.img_infos[idx]['filename'].split('.')[0]
      print(f'{scene_name}')
      img = self.load_data(scene_name)[:,:,0]
      #_show_lines_ls_points_ls(img, [bboxes])

      density_sum_mean = []
      for i in range(0, bboxes.shape[0]):
        inside_i = getOrientedLineRectSubPix(img, bboxes[i], self.obj_rep)
        density_sum_mean_i = [inside_i.sum(), inside_i.mean()]
        density_sum_mean.append( density_sum_mean_i )
        if 0:
          print(density_sum_mean_i)
          if density_sum_mean_i[0] == 0:
            _show_lines_ls_points_ls(img, [bboxes, bboxes[i:i+1]], line_colors=['green', 'red'])
      density_sum_mean = np.array(density_sum_mean)

      line_density_sum_mean_file = os.path.join(self.line_density_sum_mean_dir, self.img_infos[idx]['filename'] )
      line_ids = np.array( self.img_infos[idx]['ann_raw']['line_ids'])
      line_cat_ids = self.img_infos[idx]['ann_raw']['line_cat_ids']

      res_cats = {}
      for label in range(  labels.min(),  labels.max()+1 ):
        cat = self._catid_2_cat[label]
        mask_l = labels == label
        res_cats[cat] = ( density_sum_mean[mask_l], line_ids[mask_l] )
      #res =  ( density_sum_mean[:,0], density_sum_mean[:,1], line_ids)
      np.save( line_density_sum_mean_file, res_cats, allow_pickle=True )
      print(line_density_sum_mean_file)
      return density_sum_mean
      pass

    def find_connection(self, save_connection_imgs=False, pool_num=0):
      assert self.filter_edges == False
      self.connection_dir = self.anno_folder.replace('json', 'relations')
      self.connection_img_dir = self.anno_folder.replace('json', 'relationImgs')
      if not os.path.exists(self.connection_dir):
        os.makedirs( self.connection_dir )
        os.makedirs( self.connection_img_dir )
      n = len(self)
      if pool_num == 0:
        for i in range(n):
          self.find_connection_1_scene(i, save_connection_imgs=save_connection_imgs)
      else:
          func = partial(self.find_connection_1_scene,  save_connection_imgs=save_connection_imgs)
          ids = list(range(n))
          with Pool(pool_num) as pool:
            pool.map(func, ids )

    def find_connection_1_scene(self, idx, connect_threshold = 3, save_connection_imgs=False):
      connection_file = os.path.join(self.connection_dir, self.img_infos[idx]['filename'] )
      if os.path.exists( connection_file ):
        return
      filename = self.img_infos[idx]['filename']
      print(f'Find connection of {filename}')
      self.load_1_anno(idx)
      anno = self.img_infos[idx]['ann']
      bboxes = anno['gt_bboxes']
      labels = anno['labels']
      assert self._classes == ['background', 'wall', 'door', 'window', 'room']
      walls = bboxes[self._category_ids_map['wall'] == labels]
      windows = bboxes[self._category_ids_map['window'] == labels]
      doors = bboxes[self._category_ids_map['door'] == labels]
      wall_connect_wall_mask,_,_ = find_wall_wall_connection(walls, connect_threshold, self.obj_rep)
      window_in_wall_mask = find_wall_wd_connection(walls, windows, self.obj_rep)
      door_in_wall_mask = find_wall_wd_connection(walls, doors, self.obj_rep)
      rooms_line_ids = anno['rooms_line_ids']
      num_walls = walls.shape[0]
      num_rooms = len(rooms_line_ids)
      room_wall_mask = np.zeros([num_rooms, num_walls]) == 1
      for i in range(num_rooms):
        room_wall_mask[i][ rooms_line_ids[i] ] = True
      relations = dict( wall = wall_connect_wall_mask,
                          window = window_in_wall_mask,
                          door = door_in_wall_mask,
                          room = room_wall_mask,
                       )

      #connect_mask = np.concatenate([wall_connect_wall_mask, door_in_wall_mask, window_in_wall_mask ], axis=0).astype(np.uint8)

      np.save(connection_file, relations, allow_pickle=True)
      print('\n\t', connection_file)

      if save_connection_imgs:
        scene = self.img_infos[idx]['filename'].split('.')[0]
        for cat in relations:
        #for cat in ['door']:
          img_file = os.path.join(self.connection_img_dir, scene + f'-{cat}.png' )
          #img_file = None
          objs = bboxes[self._category_ids_map[cat] == labels]
          show_connection_2(walls, objs, relations[cat], self.obj_rep, img_file)
          pass
      pass


def meter_2_pixel(anno_style, pixel_config, corners, lines, pcl_scope, floor=False, scene=None):
  '''
  corners: [n,2]
  lines: [m,2,2]
  pcl_scope: [2,3]
  '''
  assert lines.shape[1:] == (2,2)
  assert pcl_scope.shape == (2,3)
  assert anno_style in ['topview', 'voxelization']
  if anno_style == 'topview':
    img_size = pixel_config['img_size']
  elif anno_style == 'voxelization':
    voxel_size = pixel_config['voxel_size']

  if anno_style == 'topview':
    min_xy = pcl_scope[0,:2] * 0
    max_xy = pcl_scope[1,:2] - pcl_scope[0,:2]
    # leave a bit gap along the boundaries
    max_range = (max_xy - min_xy).max()
    padding = max_range * 0.05
    min_xy = (min_xy + max_xy) / 2 - max_range / 2 - padding
    max_range += padding * 2

    lines_norm = lines - min_xy[None, None, :]
    corners_norm = corners - min_xy[None, :]

    lines_pt = (lines_norm * img_size / max_range).astype(np.float32)
    lines_pt = np.clip(lines_pt, a_min=0, a_max=img_size-1)
  if anno_style == 'voxelization':
    # in voxelization of pcl: coords = floor( position / voxel_size)
    lines_pt = (lines ) / voxel_size
    #lines_pt = np.clip(lines_pt, a_min=0, a_max=None)

  #assert lines_pt.min() >=  0, f'lines_pt min<0: {lines_pt.min()}'
  if floor:
    lines_pt = np.floor(lines_pt).astype(np.uint32)

  #line_size = np.linalg.norm( lines[:,0] - lines[:,1], axis=1 )
  #line_size_pt = np.linalg.norm( lines_pt[:,0] - lines_pt[:,1], axis=1 )
  #assert line_size_pt.min() > 3

  if corners is None:
    corners_pt = None
  else:
    if anno_style == 'topview':
      corners_pt = ((corners - min_xy) * DIM_PARSE.IMAGE_SIZE / max_range).astype(np.float32)
    if anno_style == 'voxelization':
      corners_pt = (corners) / voxel_size

    if not( corners_pt.min() > -PCL_LINE_BOUND_PIXEL and corners_pt.max() < DIM_PARSE.IMAGE_SIZE+PCL_LINE_BOUND_PIXEL ):
        scene_name = scene.split('.')[0]
        print('meter_2_pixel corner scope error', scene)
        print(corners_pt.min())
        print(corners_pt.max())
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        pass
    corners_pt = np.clip(corners_pt, a_min=0, a_max=DIM_PARSE.IMAGE_SIZE-1)
    if floor:
      corners_pt = np.floor(corners_pt).astype(np.uint32)


  check = 0
  if check:
    json_file = os.path.join('/home/z/Research/mmdetection/data/beike/processed_512/TopView_VerD', scene.replace('json','npy'))
    img = load_topview_img(json_file)
    _show_objs_ls_points_ls(img[:,:,0], objs_ls=[lines_pt.reshape(-1,4)], obj_rep='RoLine2D_2p')

  return corners_pt, lines_pt

def old_meter_2_pixel(anno_style, pixel_config, corners, lines, pcl_scope, floor=False, scene=None):
  '''
  corners: [n,2]
  lines: [m,2,2]
  pcl_scope: [2,3]
  '''
  assert lines.shape[1:] == (2,2)
  assert pcl_scope.shape == (2,3)
  assert anno_style in ['topview', 'voxelization']
  if anno_style == 'topview':
    img_size = pixel_config['img_size']
  elif anno_style == 'voxelization':
    voxel_size = pixel_config['voxel_size']

  if pcl_scope is None:
    raise NotImplementedError
    min_xy = corners.min(axis=0)
    max_xy = corners.max(axis=0)
  else:
    min_xy = pcl_scope[0,0:2]
    max_xy = pcl_scope[1,0:2]


  if anno_style == 'topview':
    # leave a bit gap along the boundaries
    max_range = (max_xy - min_xy).max()
    padding = max_range * 0.05
    min_xy = (min_xy + max_xy) / 2 - max_range / 2 - padding
    max_range += padding * 2

    lines_pt = ((lines - min_xy) * img_size / max_range).astype(np.float32)
    lines_pt = np.clip(lines_pt, a_min=0, a_max=img_size-1)
  if anno_style == 'voxelization':
    lines_pt = (lines - min_xy - voxel_size / 2) / voxel_size
    lines_pt = np.clip(lines_pt, a_min=0, a_max=None)

  if floor:
    lines_pt = np.floor(lines_pt).astype(np.uint32)

  #line_size = np.linalg.norm( lines[:,0] - lines[:,1], axis=1 )
  #line_size_pt = np.linalg.norm( lines_pt[:,0] - lines_pt[:,1], axis=1 )
  #assert line_size_pt.min() > 3

  if corners is None:
    corners_pt = None
  else:
    if anno_style == 'topview':
      corners_pt = ((corners - min_xy) * DIM_PARSE.IMAGE_SIZE / max_range).astype(np.float32)
    if anno_style == 'voxelization':
      corners_pt = (corners - min_xy) / voxel_size
    if not( corners_pt.min() > -1 and corners_pt.max() < DIM_PARSE.IMAGE_SIZE ):
          scene_name = scene.split('.')[0]
          if scene_name not in UNALIGNED_SCENES:
            print(scene)
            print(corners_pt.min())
            print(corners_pt.max())
            import pdb; pdb.set_trace()  # XXX BREAKPOINT
            pass
    corners_pt = np.clip(corners_pt, a_min=0, a_max=DIM_PARSE.IMAGE_SIZE-1)
    if floor:
      corners_pt = np.floor(corners_pt).astype(np.uint32)

  return corners_pt, lines_pt

def raw_anno_to_img(classes, room_label, obj_rep, anno_raw, anno_style, pixel_config, anno_folder):
      anno_img = {}
      anno_img['classes'] = [c for c in classes if c!='background']
      if 'voxel_size' in pixel_config:
        corners_pt, lines_pt = anno_raw['corners'], anno_raw['lines']
      else:
        corners_pt, lines_pt = meter_2_pixel(anno_style, pixel_config, anno_raw['corners'], anno_raw['lines'],
                                           pcl_scope=anno_raw['pcl_scope'], scene=anno_raw['filename'])
      lines_pt_ordered = OBJ_REPS_PARSE.encode_obj(lines_pt.reshape(-1,4), 'RoLine2D_2p', obj_rep )
      line_sizes = np.linalg.norm(lines_pt_ordered[:,[2,3]] - lines_pt_ordered[:,[0,1]], axis=1)
      min_line_size = line_sizes.min()
      labels_line_corner = np.concatenate([anno_raw['line_cat_ids'], anno_raw['corner_cat_ids'] ], axis=0)

      anno_img['gt_bboxes'] = lines_pt_ordered
      anno_img['labels'] = anno_raw['line_cat_ids'].astype(np.int64)
      anno_img['relations'] = anno_raw['relations']

      anno_img['min_line_size'] = min_line_size

      obj_dim = OBJ_REPS_PARSE._obj_dims[obj_rep]
      anno_img['bboxes_ignore'] = np.empty([0,obj_dim], dtype=np.float32)
      anno_img['mask'] = []
      anno_img['seg_map'] = None
      gt_bboxes = anno_img['gt_bboxes'][:,:4]
      assert gt_bboxes.max() < DIM_PARSE.IMAGE_SIZE

      if 'room' in classes:
        anno_img['rooms_line_ids'] = anno_raw['rooms_line_ids']
        add_room_to_anno(room_label, anno_img, anno_raw, lines_pt_ordered, obj_rep, anno_folder)

      #assert gt_bboxes.min() >= 0
      if DEBUG_CFG.VISUAL_CONNECTIONS:
        walls = anno_img['gt_bboxes'][ anno_img['labels'] == 1 ]
        show_connection_2( walls, anno_img['gt_bboxes'], anno_img['relations'], obj_rep )
      return anno_img

def add_room_to_anno( room_label, anno_img, anno_raw, lines_pt_ordered, obj_rep, anno_folder):
      rooms_line_ids = anno_raw['rooms_line_ids']
      #rooms_line_ids = None
      # add room
      room_bboxes = load_room_bboxes(anno_folder, anno_raw['filename'], obj_rep)
      if room_bboxes is None:
        wall_mask = anno_img['labels'] == 1
        assert anno_img['classes'][0] == 'wall'
        room_bboxes = gen_room_bboxes_by_room_label(lines_pt_ordered[wall_mask], rooms_line_ids, obj_rep, anno_folder, anno_raw['filename'])
      #_show_objs_ls_points_ls( (512,512), [anno_img['gt_bboxes'], room_bboxes], obj_rep=obj_rep, obj_colors=[anno_img['labels'], 'black'], obj_thickness=[10,3] )
      n = room_bboxes.shape[0]
      anno_img['gt_bboxes'] = np.concatenate([anno_img['gt_bboxes'], room_bboxes], 0)
      room_labels = np.ones([n], dtype=np.int64) * room_label
      anno_img['labels'] = np.concatenate([anno_img['labels'], room_labels])

      if 0:
        walls = anno_img['gt_bboxes'][ anno_img['labels'] == 1 ]
        show_connection_2( walls, room_bboxes, rel_room_wall, obj_rep )
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        pass

def load_topview_img(file_name):
    data = np.load(file_name, allow_pickle=True).tolist()
    img = np.expand_dims(data['topview_image'],axis=2)
    normal_image = data['topview_mean_normal']
    img = np.concatenate([img, normal_image], axis=2)
    return  img

def unused_load_pcl_scope(anno_folder):
    base_path = anno_folder.split('json')[0]
    pcl_scopes_file = os.path.join(base_path, 'pcl_scopes.json')
    with open(pcl_scopes_file, 'r') as f:
      pcl_scopes = json.load(f)
    for fid in pcl_scopes.keys():
      pcl_scopes[fid] = np.array(pcl_scopes[fid])
      # augment the scope a bit

      #offset_aug = 0.1
      #pcl_scopes[fid][0] = np.floor(pcl_scopes[fid][0] * 10)/10 - offset_aug
      #pcl_scopes[fid][1] = np.ceil(pcl_scopes[fid][1] * 10)/10 + offset_aug
    return pcl_scopes

def get_line_valid_by_density(anno_folder, filename, line_ids_check, classes):
    line_density_dir = anno_folder.replace('json', 'line_density_sum_mean')
    file_path = os.path.join(line_density_dir, filename.replace('json', 'npy'))
    line_density_res_cats = np.load(file_path, allow_pickle=True).tolist()

    line_density_cats = []
    line_ids_cats = []
    for cat in classes:
      if cat == 'background':
        continue
      line_density_cats.append( line_density_res_cats[cat][0] )
      line_ids_cats.append( line_density_res_cats[cat][1] )
    line_density_cats = np.concatenate(line_density_cats, 0).astype(np.float32)
    line_ids_cats = np.concatenate(line_ids_cats, 0)
    #line_density_sum, line_density_mean, line_ids = np.load(file_path, allow_pickle=True).tolist()
    line_ids_check = np.array(line_ids_check)
    ids_mask = line_ids_check == line_ids_cats
    assert np.all(ids_mask)

    line_valid = line_density_cats[:,0] > 0
    return line_valid


def parse_wall(anno, points, lines, beike_clsinfo):
          point_dict = {}

          for point in points:
            xy = np.array([point['x'], point['y']]).reshape(1,2)
            anno['corners'].append( xy )
            anno['corner_ids'].append( point['id'] )
            #anno['corner_lines'].append( point['lines'] )
            anno['corner_cat_ids'].append( beike_clsinfo._category_ids_map['wall'] )
            #anno['corner_locked'].append( point['locked'] )
            point_dict[point['id']] = xy
            pass

          for line in lines:
            point_id_1, point_id_2 = line['points']
            xy1 = point_dict[point_id_1]
            xy2 = point_dict[point_id_2]
            line_xys = np.array([xy1, xy2]).reshape(1,2,2)
            anno['lines'].append( line_xys )
            anno['line_ids'].append( line['id']  )
            anno['line_ponit_ids'].append( line['points'] )
            anno['line_cat_ids'].append( beike_clsinfo._category_ids_map['wall'] )
            for ele in BEIKE.edge_atts:
              if ele in line:
                anno['e_'+ele].append( line[ele] )
              else:
                pass
                #rasie NotImplementedError
def parse_line_items(anno, line_items, classes, beike_clsinfo):
        for line_item in line_items:
          cat = line_item['is']
          if cat not in classes:
            continue
          start_pt = np.array([line_item['startPointAt']['x'], line_item['startPointAt']['y']]).reshape(1,2)
          end_pt = np.array([line_item['endPointAt']['x'], line_item['endPointAt']['y']]).reshape(1,2)
          cat_id = beike_clsinfo._category_ids_map[cat]
          line_xy = np.concatenate([start_pt, end_pt], 0).reshape(1,2,2)

          anno['corners'].append( start_pt )
          anno['corners'].append( end_pt )
          #anno['corner_ids'].append( line_item['line'] )
          #anno['corner_ids'].append( line_item['line'] )
          #anno['corner_lines'].append( line_item['id'] )
          #anno['corner_lines'].append( line_item['id'] )
          anno['corner_cat_ids'].append( cat_id )
          anno['corner_cat_ids'].append( cat_id )
          #anno['corner_locked'].append( False )
          #anno['corner_locked'].append( False )

          anno['line_ids'].append(line_item['id'])
          anno['lines'].append( line_xy )
          anno['line_ponit_ids'].append( [line_item['line'], line_item['line'] ] )
          anno['line_cat_ids'].append( cat_id )

          pass

def parse_areas(anno, areas):
    for area in areas:
      points = area['points']
      roomName = area['roomName']
      corner_ids = anno['corner_ids']
      line_ponit_ids = anno['line_ponit_ids']
      n = len(corner_ids)
      room_corner_inds = [i for i in range(n) if corner_ids[i] in points]
      m = len(line_ponit_ids)
      room_line_inds = [i for i in range(m) if line_ponit_ids[i][0] in points and line_ponit_ids[i][1] in points]

      anno['rooms_line_ids'].append( room_line_inds )

def load_anno_1scene(anno_folder, filename, classes,  filter_edges=False, is_save_connection=False):
      always_load_walls = 1

      beike_clsinfo = BEIKE_CLSINFO(classes, always_load_walls)
      if 'wall' not in classes and always_load_walls:
        classes =  ['wall'] + classes

      scene_name = os.path.splitext(filename)[0]
      data_dir = anno_folder.split('json')[0]

      file_path = os.path.join(anno_folder, filename)
      with open(file_path, 'r') as f:
        metadata = json.load(f)
        data = copy.deepcopy(metadata)
        points = data['points']
        lines = data['lines']
        line_items = data['lineItems']
        areas = data['areas']

        anno = defaultdict(list)
        anno['filename'] = filename
        anno['line_ids'] = []

        if 'wall' in classes:
          parse_wall(anno, points, lines, beike_clsinfo)

        parse_line_items(anno, line_items, classes, beike_clsinfo)
        parse_areas(anno, areas)

      for ele in ['corners', 'lines', ]:
              anno[ele] = np.concatenate(anno[ele], 0).astype(np.float32)
      for ele in ['line_cat_ids',  'corner_cat_ids', ]:
              anno[ele] = np.array(anno[ele]).astype(np.int32)
      for ele in ['line_ids', 'rooms_line_ids']:
              anno[ele] = np.array(anno[ele])

      lines_leng = np.linalg.norm(anno['lines'][:,0] - anno['lines'][:,1], axis=-1)
      anno['line_length_min_mean_max'] = [lines_leng.min(), lines_leng.mean(), lines_leng.max()]

      scope_file = os.path.join(data_dir, 'pcl_scopes', scene_name+'.txt')
      scope = np.loadtxt(scope_file, dtype=float)
      anno['pcl_scope'] = scope

      # normalize zero
      pcl_scope = anno['pcl_scope']
      anno['corners'] = anno['corners'] - pcl_scope[0:1,:2:]
      anno['lines'] = anno['lines'] - pcl_scope[0:1,None,:2:]

      # fix unaligned
      scene_size =  anno['pcl_scope'][1,:2] -  anno['pcl_scope'][0,:2]
      scene_size = scene_size[[1,0]]
      scene_name = filename.split('.json')[0]

      anno['lines'] = fix_1_unaligned_scene(scene_name, anno['lines'], scene_size, line_obj_rep='RoLine2D_2p')
      tmp = np.repeat( anno['corners'][:,None,:], 2, axis=1 )
      anno['corners'] = fix_1_unaligned_scene(scene_name, tmp, scene_size, line_obj_rep='RoLine2D_2p' )[:,0,:]

      # order by anno['line_cat_ids']
      line_cat_ids = anno['line_cat_ids']
      order = []
      for i in range( line_cat_ids.min(), line_cat_ids.max()+1 ):
        order.append( np.where( line_cat_ids == i )[0] )
      order = np.concatenate(order)
      for ele in ['line_ids', 'lines', 'line_cat_ids']:
            anno[ele] = anno[ele][order]

      if 0:
        if not( anno['corners'].min() > -PCL_LINE_BOUND_METER and anno['corners'].min() < 1 ):
          print(anno['corners'].min())
          import pdb; pdb.set_trace()  # XXX BREAKPOINT
          pass
        if not (anno['lines'].min() > -PCL_LINE_BOUND_METER and anno['lines'].min() < 1):
          print(  anno['lines'].min() )
          import pdb; pdb.set_trace()  # XXX BREAKPOINT
          pass
        if not (all(anno['lines'].max(axis=0).max(axis=0) <  pcl_scope[1,:2] - pcl_scope[0,:2]) + 0.1):
          print(  anno['lines'].max(axis=0).max(axis=0) )
          import pdb; pdb.set_trace()  # XXX BREAKPOINT
          pass

      if not is_save_connection:
        anno['relations'] = load_relations_1scene(anno_folder, filename, classes)

      if filter_edges:
        wall_num = np.sum(anno['line_cat_ids'] == beike_clsinfo._category_ids_map['wall'])
        line_valid = get_line_valid_by_density(anno_folder, filename, anno['line_ids'], classes)
        for ele in ['lines','line_cat_ids']:
          anno[ele] = anno[ele][line_valid]
        if not  is_save_connection:
          anno['relations'] = anno['relations'][line_valid][:, line_valid[:wall_num]]
        pass

      check_with_ply = 0
      if check_with_ply:
        show_ann_pcl(anno, file_path)
      return anno


def gen_room_bboxes_by_room_label(lines, rooms_line_ids, obj_rep_out, anno_folder, filename):
  from obj_geo_utils.geometry_utils import points_to_oriented_bbox, get_rooms_from_edges
  room_label_non_intact = ['uxMhs9XTA7txv6_kvoDjHv', 'isDTo3WSrPK99A14wmpcYg', '8Ej90dGb8mD7ykRTOsWVbV', '18H6WOCclkJY34-TVuOqX3']
  obj_rep = 'XYXYSin2WZ0Z1'
  rooms_line_ids_in = rooms_line_ids.copy()
  rooms_line_ids, room_ids_per_edge, num_walls_inside_room, rooms = get_rooms_from_edges(lines, obj_rep, gen_bbox=True)

  if len(rooms_line_ids)!=len(rooms_line_ids_in):
    print(f'\n\troom label not intact: {filename}')
    pass

  #line_corners = OBJ_REPS_PARSE.encode_obj(lines, obj_rep, 'RoLine2D_2p')
  #num_rooms = len(rooms_line_ids)
  #rooms = []
  #for i in range(num_rooms):
  #  ids_i = rooms_line_ids[i]
  #  room_corners = line_corners[ids_i]
  #  room_bbox = points_to_oriented_bbox(room_corners.reshape(-1,2), obj_rep)
  #  rooms.append(room_bbox)
  #  if 0:
  #    _show_objs_ls_points_ls( (1024, 1024),[room_bbox], obj_rep, [ room_corners.reshape(-1,2) ], point_thickness=5, obj_thickness=2 )
  #rooms = np.concatenate(rooms, 0)

  rooms_dir = anno_folder.replace('json', 'room_bboxes')
  filename = filename.replace('json', 'txt')
  rooms_file = os.path.join(rooms_dir, filename)

  if not os.path.exists(rooms_dir):
    os.makedirs(rooms_dir)
  np.savetxt(rooms_file, rooms, fmt='%.3f')
  print(f'save: {rooms_file}')

  if num_walls_inside_room >= 0:
    scene = os.path.splitext(filename)[0]
    summary_file = os.path.join(anno_folder.split('json')[0], 'num_walls_inside_room.txt')
    num_walls = lines.shape[0]
    with open( summary_file, 'a' ) as f:
      f.write( f'{scene}: {num_walls_inside_room} {num_walls}\n' )

  if obj_rep_out != obj_rep:
    rooms = OBJ_REPS_PARSE.encode_obj(rooms, obj_rep, obj_rep_out)
  return rooms

def load_room_bboxes(anno_folder, filename, obj_rep_out):
  obj_rep = 'XYXYSin2WZ0Z1'
  rooms_dir = anno_folder.replace('json', 'room_bboxes')
  filename = filename.replace('json', 'txt')
  rooms_file = os.path.join(rooms_dir, filename)
  if not os.path.exists(rooms_file):
    return None
  else:
    rooms = np.loadtxt(rooms_file)
    if obj_rep_out != obj_rep:
      rooms = OBJ_REPS_PARSE.encode_obj(rooms, obj_rep, obj_rep_out)
    return rooms


def show_ann_pcl(anno, json_path):
  ply_path = json_path.replace('json', 'ply')
  pcl = load_ply(ply_path)
  lines = anno['lines'].reshape(-1,4)
  line_scope = [ lines.reshape(-1,2).min(0), lines.reshape(-1,2).max(0)]
  corners = anno['corners']
  cor_scope = [corners.min(0), corners.max(0)]
  pcl_scope = anno['pcl_scope']


  lines_norm = lines.copy().reshape(-1,2) - pcl_scope[0:1, :2]
  lines_norm = lines_norm.reshape(-1,4)
  line_norm_scope = [ lines_norm.reshape(-1,2).min(0), lines_norm.reshape(-1,2).max(0)]

  pcl[:,:2] -= pcl_scope[0:1,:2]

  print(f'pcl_scope: \n{pcl_scope}')
  print(f'cor_scope: \n{cor_scope}')
  print(f'line_scope: \n{line_scope}')
  print(f'line_norm_scope: \n{line_norm_scope}')
  _show_3d_points_objs_ls( [pcl[:,:3]], [pcl[:,3:6]], objs_ls = [lines], obj_rep='RoLine2D_2p')
  _show_3d_points_objs_ls( [pcl[:,:3]], [pcl[:,3:6]], objs_ls = [lines_norm], obj_rep='RoLine2D_2p')

  pass

def show_connection_2(walls, bboxes, relations, obj_rep, img_file=None):
  n = bboxes.shape[0]
  for i in range(n):
    cids = np.where(relations[i])[0]
    #_show_objs_ls_points_ls( (512,512), [bboxes, bboxes[i:i+1], ], 'XYXYSin2', obj_colors=['white', 'green',])
    _show_objs_ls_points_ls( (512,512), [walls, walls[cids], bboxes[i:i+1]], obj_rep, obj_colors=['white', 'green', 'red'], obj_thickness=[1,5,2], out_file=img_file, only_save=img_file is not None)
    if img_file is not None:
      break
  pass

def load_relations_1scene(anno_folder, filename, classes):
  connection_dir = anno_folder.replace('json', 'relations')
  connect_file = os.path.join(connection_dir, filename.replace('.json', '.npy'))
  relations_dict = np.load(connect_file, allow_pickle=True).tolist()
  relations = [relations_dict[c] for c in classes if c!='background']
  relations = np.concatenate(relations, axis=0)
  #n_all, n_wall = relations.shape
  #n_others = n_all - n_wall
  #tmp = np.zeros([n_all, n_all-n_wall])==1
  #tmp[:n_wall, :] = relations[n_wall:, :n_wall].T
  #relations = np.concatenate([relations, tmp], axis=1)
  return relations

def unused_load_gt_lines_bk(img_meta, img, classes, filter_edges):
  beike_clsinfo = BEIKE_CLSINFO(classes)
  filename = img_meta['filename']
  scene_name = os.path.basename(filename).replace('.npy', '')
  processed_dir = os.path.dirname(os.path.dirname(filename))
  json_dir = os.path.join(processed_dir, 'json/')
  anno_raw = load_anno_1scene(json_dir, scene_name+'.json', beike_clsinfo._classes, filter_edges=filter_edges)
  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  anno_img = raw_anno_to_img(anno_raw,  'topview', {'img_size': DIM_PARSE.IMAGE_SIZE},)
  lines = anno_img['bboxes']
  labels = anno_img['labels']
  mask = labels >= 0

  lines = lines[mask]
  labels = labels[mask]
  #_show_lines_ls_points_ls(img[:,:,0], [lines])
  if 'rotate_angle' in img_meta:
    rotate_angle = img_meta['rotate_angle']
    lines, _ = rotate_bboxes_img(lines, img, rotate_angle, DIM_PARSE.OBJ_REP)
    return lines, labels
  else:
    return lines, labels

def fix_1_unaligned_scene(scene_name, lines_unaligned, image_size, line_obj_rep):
    '''
    lines_unaligned: [n,5]
    '''
    if scene_name in BAD_SCENE_TRANSFERS_PCL:
      #angle, cx, cy = BAD_SCENE_TRANSFERS_1024[scene_name]
      #if not isinstance(image_size, np.ndarray):
      #  image_size = np.array([image_size, image_size])
      #scale = image_size / 1024.0
      #cx = cx * scale[0]
      #cy = cy * scale[1]
      #print(scene_name, f'\t({angle}, {cx:.3f}, {cy:.3f})')

      angle, cx, cy = BAD_SCENE_TRANSFERS_PCL[scene_name]
      n = lines_unaligned.shape[0]
      lines_aligned = transfer_lines( lines_unaligned.reshape(n,4), line_obj_rep,
        image_size, angle, (cx, cy) ).reshape(n,2,2)
      pass
    else:
      lines_aligned = lines_unaligned
    return lines_aligned


def _UnUsed_gen_images_from_npy(data_path):
  npy_path = os.path.join(data_path, 'seperate_room_data/test')
  den_image_path = os.path.join(data_path, f'images/public100_{DIM_PARSE.IMAGE_SIZE}')
  norm_image_path = den_image_path

  if not os.path.exists(den_image_path):
    os.makedirs(den_image_path)
  if not os.path.exists(norm_image_path):
    os.makedirs(norm_image_path)

  file_names = os.listdir(npy_path)
  files = [os.path.join(npy_path, f) for f in file_names]
  den_images = [os.path.join(den_image_path, f.replace('.npy', '.density.png')) for f in file_names]
  norm_images = [os.path.join(norm_image_path, f.replace('.npy', '.norm.png')) for f in file_names]
  for i,fn in enumerate(files):
      data = np.load(fn, allow_pickle=True).tolist()
      img = data['topview_image']
      #lines = data['line_coords']
      #room_map = data['room_map']
      #bg_idx = data['bg_idx']
      normal_image = data['topview_mean_normal']
      cv2.imwrite(den_images[i], img)
      cv2.imwrite(norm_images[i], normal_image*255)
      print(den_images[i])
      pass

  pass

def Unused_get_scene_pcl_scopes(data_path):
  pcl_scopes_file = os.path.join(data_path, 'pcl_scopes.json')
  if os.path.exists(pcl_scopes_file):
    return
  ply_path = os.path.join(data_path, 'ply')
  pcl_files = os.listdir(ply_path)
  pcl_scopes = {}
  for pclf in pcl_files:
    pcl_file = os.path.join(ply_path, pclf)
    scene_name = pclf.split('.')[0]
    points_data = load_ply(pcl_file)
    xyz_min = points_data[:,0:3].min(0, keepdims=True)
    xyz_max = points_data[:,0:3].max(0, keepdims=True)
    xyz_min_max = np.concatenate([xyz_min, xyz_max], 0)
    pcl_scopes[scene_name] = xyz_min_max.tolist()
    print(f'{scene_name}: \n{xyz_min_max}\n')

  with open(pcl_scopes_file, 'w') as f:
    json.dump(pcl_scopes, f)
  print(f'save {pcl_scopes_file}')

def load_ply(pcl_file):
    with open(pcl_file, 'rb') as f:
      plydata = PlyData.read(f)
    points = np.array(plydata['vertex'].data.tolist()).astype(np.float32)
    points = points[:,:9]
    assert points.shape[1] == 9
    return points

def cal_topview_npy_mean_std(data_path, base, normnorm_method='raw'):
  '''
  TopView_All
      normnorm_method: raw
        mean: [ 4.753,  0.,     0.,    -0.015]
        std:  [16.158,  0.155,  0.153,  0.22 ]

      normnorm_method: abs255
        mean: [ 4.753, 11.142, 11.044, 25.969]
        std : [16.158, 36.841, 36.229, 46.637]

      normnorm_method: abs
        mean: [4.753, 0.044, 0.043, 0.102]
        std : [16.158,  0.144,  0.142,  0.183]


  TopView_VerD
      mean: [ 2.872  0.     0.    -0.015]
      std : [16.182  0.155  0.153  0.22 ]

    abs_norm
      mean: [2.872 0.044 0.043 0.102]
      std : [16.182  0.144  0.142  0.183]
  '''
  res_file = os.path.join( data_path, 'mean_std.txt' )
  if os.path.exists(res_file):
    return
  npy_path = os.path.join(data_path, base)
  files = glob.glob(npy_path + '/*.npy')
  n =  len(files)
  assert n>0

  topviews = []
  for fn in files:
    data = np.load(fn, allow_pickle=True).tolist()
    topview_image = np.expand_dims(data['topview_image'], axis=2)
    topview_mean_normal = data['topview_mean_normal']
    if normnorm_method == 'raw':
      pass
    elif normnorm_method == 'abs':
      topview_mean_normal = np.abs(topview_mean_normal)
    elif normnorm_method == 'abs255':
      topview_mean_normal = np.abs(topview_mean_normal) * 255

    topview = np.concatenate([topview_image, topview_mean_normal], axis=2)
    topviews.append( np.expand_dims( topview, 0) )
  topviews = np.concatenate(topviews, 0)
  imgs = topviews.reshape(-1, topviews.shape[-1])

  mean = imgs.mean(axis=0)
  tmp = imgs - mean
  std = tmp.std(axis=0)
  res = f'normnorm_method: {normnorm_method}\n'
  res += f'mean: {mean}\n'
  res += f'std : {std}\n'
  with open(res_file, 'w') as f:
    f.write(res)
  print(res)

def draw_img_lines(img, lines):
      img = img.copy()
      for i in range(lines.shape[0]):
        s, e = lines[i]
        cv2.line(img, (s[0], s[1]), (e[0], e[1]), (255,0,0), 6)
        #cv2.putText(img, obj, (s[0], s[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
        #            (255, 255, 255), 1)
      mmcv.imshow(img)
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass
# ------------------------------------------------------------------------------

def gen_images_from_npy(data_path):
  npy_path = os.path.join(data_path, 'seperate_room_data/test')
  den_image_path = os.path.join(data_path, f'images/public100_{DIM_PARSE.IMAGE_SIZE}')
  norm_image_path = den_image_path

  if not os.path.exists(den_image_path):
    os.makedirs(den_image_path)
  if not os.path.exists(norm_image_path):
    os.makedirs(norm_image_path)

  file_names = os.listdir(npy_path)
  files = [os.path.join(npy_path, f) for f in file_names]
  den_images = [os.path.join(den_image_path, f.replace('.npy', '.density.png')) for f in file_names]
  norm_images = [os.path.join(norm_image_path, f.replace('.npy', '.norm.png')) for f in file_names]
  for i,fn in enumerate(files):
      data = np.load(fn, allow_pickle=True).tolist()
      img = data['topview_image']
      #lines = data['line_coords']
      #room_map = data['room_map']
      #bg_idx = data['bg_idx']
      normal_image = data['topview_mean_normal']
      cv2.imwrite(den_images[i], img)
      cv2.imwrite(norm_images[i], np.abs(normal_image)*255)
      print(den_images[i])
      pass

  pass

def gen_gts(data_path):
  obj_rep = 'XYXYSin2'
  #obj_rep = 'XYZLgWsHA'
  obj_rep = 'XYXYSin2WZ0Z1'
  ANNO_PATH = os.path.join(data_path, 'json/')
  phase = 'all.txt'
  topview_path = os.path.join(data_path, 'TopView_VerD', phase)

  is_save_connection = 0
  classes = ['wall', 'door', 'window', 'room']
  beike = BEIKE(obj_rep, ANNO_PATH, topview_path,
                classes = classes,
                filter_edges= 0,
                is_save_connection=is_save_connection,
                )

  scene_list = os.path.join(data_path, 'all.txt')
  scenes = np.loadtxt(scene_list, str).tolist()

  rotate_angle = 30
  for s in scenes[:100]:
    beike.show_scene_anno(s, True, rotate_angle, write=True)
    pass


def save_unscanned_edges(data_path):
  obj_rep = 'XYXYSin2'
  ANNO_PATH = os.path.join(data_path, 'json/')
  phase = 'test.txt'

  topview_path = os.path.join(data_path, 'TopView_VerD', phase)

  is_save_connection = 1
  classes = ['wall', 'door', 'window', ]
  beike = BEIKE(obj_rep, ANNO_PATH, topview_path,
                classes = classes,
                filter_edges= 0,
                is_save_connection=is_save_connection,
                )
  beike.find_unscanned_edges()
  n = len(beike)
  print(f'find unscanned ok: {n}\n')

  pass

def gen_connections(data_path, pool_num):
  obj_rep = 'XYXYSin2'
  obj_rep = 'XYXYSin2WZ0Z1'
  ANNO_PATH = os.path.join(data_path, 'json/')
  phase = 'all.txt'
  topview_path = os.path.join(data_path, 'TopView_VerD', phase)

  is_save_connection = 1
  classes = ['wall', 'door', 'window', 'room',]
  beike = BEIKE(obj_rep, ANNO_PATH, topview_path,
                classes = classes,
                filter_edges= 0,
                is_save_connection=is_save_connection,
                )
  beike.find_connection(True, pool_num)
  n = len(beike)
  print(f'\nGenerate connections ok: {n}\n')

  pass


def gen_connection_gt(pool_num=3):
  cur_dir = os.path.dirname(os.path.realpath(__file__))
  root_dir = os.path.dirname(cur_dir)
  data_path = os.path.join(root_dir, f'data/beike/processed_{DIM_PARSE.IMAGE_SIZE}' )

  #save_unscanned_edges(data_path)

  cal_topview_npy_mean_std(data_path, base='TopView_VerD', normnorm_method='abs')
  gen_connections(data_path, pool_num)
  gen_gts(data_path)

def debug():
  cur_dir = os.path.dirname(os.path.realpath(__file__))
  root_dir = os.path.dirname(cur_dir)
  data_path = os.path.join(root_dir, f'data/beike/processed_{DIM_PARSE.IMAGE_SIZE}' )
  gen_gts(data_path)
  #gen_connections(data_path)

if __name__ == '__main__':
  debug()


