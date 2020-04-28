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


from obj_geo_utils.obj_utils import OBJ_REPS_PARSE
from configs.common import DIM_PARSE, DEBUG_CFG
from beike_data_utils.line_utils import encode_line_rep, rotate_lines_img, transfer_lines, gen_corners_from_lines_np
from tools.debug_utils import get_random_color, _show_img_with_norm, _show_lines_ls_points_ls
from tools.visual_utils import _show_objs_ls_points_ls
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
WRITE_ANNO_IMG = 0

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
      classes_order = ['background', 'wall', 'door', 'window', 'other']
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
        super().__init__(classes, always_load_walls=1)
        assert  anno_folder[-5:] == 'json/'
        self.obj_rep = obj_rep
        self.anno_folder = anno_folder
        self.test_mode = test_mode
        self.is_save_connection = is_save_connection
        if is_save_connection:
          filter_edges = False
        self.filter_edges = filter_edges
        self.split_file = img_prefix
        self.scene_list = np.loadtxt(self.split_file,str).tolist()

        #self.scene_list = [*BAD_SCENE_TRANSFERS_1024.keys()]
        #self.scene_list = ['wcSLwyAKZafnozTPsaQMyv']

        self.img_prefix = os.path.dirname( img_prefix )
        self.is_pcl = os.path.basename(self.img_prefix) == 'ply'
        data_format = '.ply' if self.is_pcl else '.npy'

        if WRITE_ANNO_IMG:
          self.anno_img_folder = self.anno_folder.replace('json', 'anno_imgs')
          if not os.path.exists(self.anno_img_folder):
            os.makedirs(self.anno_img_folder)
        self.seperate_room_path = anno_folder.replace('json', 'seperate_room_data/test')
        #self.seperate_room_data_path = anno_folder.replace('json', 'seperate_room_data/test')
        base_path = os.path.dirname( os.path.dirname(anno_folder) )

        json_files = os.listdir(anno_folder)
        assert len(json_files) > 0

        img_infos = []
        all_min_line_sizes = []
        for jfn in json_files:
          scene_name = jfn.split('.')[0]
          if scene_name not in self.scene_list:
            continue
          anno_raw = load_anno_1scene(self.anno_folder, jfn,
                                self._classes, filter_edges=filter_edges,
                                is_save_connection = self.is_save_connection)

          anno_img = raw_anno_to_img(self.obj_rep, anno_raw, 'topview', {'img_size': DIM_PARSE.IMAGE_SIZE}, )
          anno_img['classes'] = [c for c in classes if c!='background']
          filename = jfn.split('.')[0]+data_format
          img_info = {'filename': filename,
                      'ann': anno_img,
                      'ann_raw': anno_raw}
          img_infos.append(img_info)
          all_min_line_sizes.append(anno_img['min_line_size'])

        self.img_infos = img_infos
        self.all_min_line_sizes = np.array( all_min_line_sizes )
        print(f'min line size: {self.all_min_line_sizes.min()}')

        self.rm_bad_scenes()
        #self.fix_unaligned_scenes()
        #if self.img_prefix is not None:
        #  self.rm_anno_withno_data()

        n0 = len(self.img_infos)
        if WRITE_ANNO_IMG:
          for i in range(n0):
            #self.draw_anno_raw(i, with_img=1)
            self.show_anno_img(i, with_img=1)


    def unused_fix_unaligned_scenes(self):
      n0 = len(self.img_infos)
      for i in range(n0):
        sn = self.img_infos[i]['filename'].split('.')[0]
        self.img_infos[i]['ann']['bboxes'] = fix_1_unaligned_scene(sn, \
                                self.img_infos[i]['ann']['bboxes'], DIM_PARSE.IMAGE_SIZE)

    def rm_bad_scenes(self):
      valid_ids = []
      n0 = len(self.img_infos)
      for i in range(n0):
        sn = self.img_infos[i]['filename'].split('.')[0]
        if sn not in BAD_SCENES:
          valid_ids.append(i)
      self.img_infos = [self.img_infos[i] for i in valid_ids]
      n1 = len(self.img_infos)
      print(f'\n {self.split_file}\t load {n0} scenes with {n1} valid\n')

    def rm_anno_withno_data(self):
      n0 = len(self.img_infos)
      valid_inds = []
      valid_files = os.listdir(self.img_prefix)
      for i, img_info in enumerate(self.img_infos):
        filename = img_info['filename']
        if img_info['filename'] in valid_files:
          valid_inds.append(i)
      valid_img_infos = [self.img_infos[i] for i in valid_inds]
      self.img_infos = valid_img_infos
      n = len(self.img_infos)
      print(f'\n{n} valid scenes with annotation found in total {n0} in {self.img_prefix}\n')

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
      return len(self.img_infos)

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
      data = np.load(file_name, allow_pickle=True).tolist()
      img = np.expand_dims(data['topview_image'],axis=2)
      normal_image = data['topview_mean_normal']
      img = np.concatenate([img, normal_image], axis=2)
      return  img

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

    def show_anno_img(self, idx,  with_img=True, rotate_angle=0, lines_transfer=(0,0,0)):
      colors_line   = {'wall': (0,0,255), 'door': (0,255,255),
                       'window': (0,255,255), 'other':(100,100,0)}
      colors_corner = {'wall': (0,0,255), 'door': (0,255,0),
                       'window': (255,0,0), 'other':(255,255,255)}
      anno = self.img_infos[idx]['ann']
      bboxes = anno['bboxes']
      labels = anno['labels']
      corners,_,_,_ = gen_corners_from_lines_np(bboxes, None, DIM_PARSE.OBJ_REP)

      scene_name = self.img_infos[idx]['filename'].split('.')[0]
      print(f'{scene_name}')


      if not with_img:
        img = np.zeros((DIM_PARSE.IMAGE_SIZE, DIM_PARSE.IMAGE_SIZE, 3), dtype=np.uint8)
        img = (DIM_PARSE.IMAGE_SIZE, DIM_PARSE.IMAGE_SIZE)
      else:
        img = self.load_data(scene_name)[:,:,0]
        img = img *10


      if (np.array(lines_transfer) != 0).any():
        angle, cx, cy = lines_transfer
        bboxes = transfer_lines(bboxes, DIM_PARSE.OBJ_REP, img.shape[:2], angle, (cx,cy))

      if rotate_angle != 0:
        bboxes, img = rotate_lines_img(bboxes, img, rotate_angle,
                                      DIM_PARSE.OBJ_REP, check_by_cross=False)

      if WRITE_ANNO_IMG:
        anno_img_file = os.path.join(self.anno_img_folder, scene_name+'.png')
      else:
        anno_img_file = None
      _show_lines_ls_points_ls(img, [bboxes], [corners],
                               line_colors='random', point_colors='random',
                               line_thickness=1, point_thickness=1,
                               out_file=anno_img_file, only_save=0)

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
      #_show_img_with_norm(img)

      img_type = 'density'
      #img_type = 'norm'
      if img_type == 'density':
        img = np.repeat(img[:,:,0:1], 3, axis=2).astype(np.uint8)
      if img_type == 'norm':
        img = np.abs(img[:,:,1:]) * 255
        img = img.astype(np.uint8)

      lines = bboxes[:,:4].copy()

      if bboxes.shape[1] == 5:
        istopleft = bboxes[:,-1]
        print('istopleft:\n',istopleft)
        n = bboxes.shape[0]
        for i in range(n):
          if istopleft[i] < 0:
            lines[i] = lines[i,[2,1,0,3]]
      lines = lines.reshape(-1,2,2).astype(np.int32)
      line_sizes = np.linalg.norm(lines[:,0]-lines[:,1], axis=1)
      idx_min_size = line_sizes.argmin()
      min_line_size = line_sizes[idx_min_size]
      print(f'min line size: {min_line_size}')
      #mmcv.imshow_bboxes(img, bboxes)

      for i in range(lines.shape[0]):
        s, e = lines[i]
        if i != idx_min_size or 1:
          color = colors_line['wall']
          thickness = 1
        else:
          color = (0,255,255)
          thickness = 3
        color = get_random_color()
        cv2.line(img, (s[0], s[1]), (e[0], e[1]), color, thickness=thickness)

      if corners is not None:
        for i in range(corners.shape[0]):
          c = corners[i]
          cv2.circle(img, (c[0], c[1]), radius=3, color=colors_corner['wall'],
                    thickness=2)
      mmcv.imshow(img)
      if WRITE_ANNO_IMG:
        anno_img_file = os.path.join(self.anno_img_folder, scene_name+'.png')
        cv2.imwrite(anno_img_file, img)
        print(anno_img_file)
      return img
      pass


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

      if WRITE_ANNO_IMG:
        anno_img_file = os.path.join(self.anno_img_folder, scene_name+'.png')
        cv2.imwrite(anno_img_file, img)
        print(anno_img_file)
      #mmcv.imshow(img)
      return img

    def show_scene_anno(self, scene_name, with_img=True, rotate_angle=0, lines_transfer=(0,0,0)):
      idx = None
      for i in range(len(self)):
        sn = self.img_infos[i]['filename'].split('.')[0]
        if sn == scene_name:
          idx = i
          break
      assert idx is not None, f'cannot find {scene_name}'
      self.show_anno_img(idx, with_img, rotate_angle, lines_transfer)


    def find_unscanned_edges(self,):
      self.line_density_sum_mean_dir = self.anno_folder.replace('json', 'line_density_sum_mean')
      if not os.path.exists(self.line_density_sum_mean_dir):
        os.makedirs( self.line_density_sum_mean_dir )
      for i in range(len(self)):
        self.find_unscanned_edges_1_scene(i)

    def find_unscanned_edges_1_scene(self, idx):
      from beike_data_utils.line_utils import  getOrientedLineRectSubPix

      anno = self.img_infos[idx]['ann']
      bboxes = anno['bboxes']
      labels = anno['labels']
      corners,_,_,_ = gen_corners_from_lines_np(bboxes, None, DIM_PARSE.OBJ_REP)

      scene_name = self.img_infos[idx]['filename'].split('.')[0]
      print(f'{scene_name}')
      img = self.load_data(scene_name)[:,:,0]
      #_show_lines_ls_points_ls(img, [bboxes])

      density_sum_mean = []
      for i in range(0, bboxes.shape[0]):
        inside_i = getOrientedLineRectSubPix(img, bboxes[i], DIM_PARSE.OBJ_REP)
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

    def find_connection(self,):
      assert self.filter_edges == False
      self.connection_dir = self.anno_folder.replace('json', 'relations')
      if not os.path.exists(self.connection_dir):
        os.makedirs( self.connection_dir )
      for i in range(len(self)):
        self.find_connection_1_scene(i)

    def find_connection_1_scene(self, idx, connect_threshold = 3):
      print(self.img_infos[idx]['filename'])
      anno = self.img_infos[idx]['ann']
      bboxes = anno['bboxes']
      labels = anno['labels']
      assert self._classes == ['background', 'wall', 'door', 'window']
      walls = bboxes[self._category_ids_map['wall'] == labels]
      windows = bboxes[self._category_ids_map['window'] == labels]
      doors = bboxes[self._category_ids_map['door'] == labels]
      wall_connect_wall_mask = find_wall_wall_connection(walls, connect_threshold)
      window_in_wall_mask = find_wall_wd_connection(walls, windows)
      door_in_wall_mask = find_wall_wd_connection(walls, doors)
      relations = dict( wall = wall_connect_wall_mask,
                          window = window_in_wall_mask,
                          door = door_in_wall_mask )

      #connect_mask = np.concatenate([wall_connect_wall_mask, door_in_wall_mask, window_in_wall_mask ], axis=0).astype(np.uint8)

      connection_file = os.path.join(self.connection_dir, self.img_infos[idx]['filename'] )
      np.save(connection_file, relations, allow_pickle=True)
      print('\n\t', connection_file)
      pass


def find_wall_wd_connection(walls, windows):
  from obj_geo_utils.geometry_utils import  points_in_lines
  #windows = windows[3:4]
  #walls = walls[-1:]
  #_show_objs_ls_points_ls((512,512), [walls, windows], 'RoLine2D_UpRight_xyxy_sin2a', obj_colors=['red', 'green'])
  walls_2p = OBJ_REPS_PARSE.encode_obj( walls, 'RoLine2D_UpRight_xyxy_sin2a', 'RoLine2D_2p' ).reshape(-1,2,2)
  window_centroids = OBJ_REPS_PARSE.encode_obj( windows, 'RoLine2D_UpRight_xyxy_sin2a', 'RoLine2D_2p' ).reshape(-1,2,2).mean(axis=1)
  win_in_wall_mask = points_in_lines(window_centroids, walls_2p, threshold_dis=10, one_point_in_max_1_line=True)
  win_ids, wall_ids_per_win = np.where(win_in_wall_mask)

  nw = windows.shape[0]
  if not (np.all(win_ids == np.arange(nw)) and win_ids.shape[0] == nw):
    print(f'win_ids: {win_ids}, nw={nw}')
    missed_win_ids = [i for i in range(windows.shape[0]) if i not in win_ids]
    _show_objs_ls_points_ls((512,512), [walls, windows[missed_win_ids] ], 'RoLine2D_UpRight_xyxy_sin2a', obj_colors=['white', 'green'])
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    pass
  if 0:
    for i,j in zip(win_ids, wall_ids_per_win):
      _show_objs_ls_points_ls((512,512), [walls, walls[j:j+1], windows[i:i+1] ], 'RoLine2D_UpRight_xyxy_sin2a', [window_centroids[i:i+1]], obj_colors=['white', 'green', 'red'])
      pass
  return win_in_wall_mask

def find_wall_wall_connection(bboxes, connect_threshold):
      corners_per_line = OBJ_REPS_PARSE.encode_obj(bboxes, 'RoLine2D_UpRight_xyxy_sin2a', 'RoLine2D_2p')
      n = bboxes.shape[0]
      corners_per_line = corners_per_line.reshape(n,2,2)
      # note: the order of two corners is not consistant
      corners_dif0 = corners_per_line[:,None,:,None,:] - corners_per_line[None,:,None,:,:]
      corners_dif1 = np.linalg.norm( corners_dif0, axis=-1 )
      corners_dif = corners_dif1.min(axis=-1).min(axis=-1)
      np.fill_diagonal(corners_dif, 100)
      connect_mask = corners_dif < connect_threshold

      xinds, yinds = np.where(connect_mask)
      connection = np.concatenate([xinds[:,None], yinds[:,None]], axis=1).astype(np.uint8)
      relations = []
      for i in range(n):
        relations.append([])
      for i in range(connection.shape[0]):
        x,y = connection[i]
        relations[x].append(y)
      return connect_mask

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

    lines_pt = ((lines - min_xy) * img_size / max_range).astype(np.float32)
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
            pass
    corners_pt = np.clip(corners_pt, a_min=0, a_max=DIM_PARSE.IMAGE_SIZE-1)
    if floor:
      corners_pt = np.floor(corners_pt).astype(np.uint32)

  return corners_pt, lines_pt

def raw_anno_to_img(obj_rep, anno_raw, anno_style, pixel_config):
      anno_img = {}
      if 'voxel_size' in pixel_config:
        corners_pt, lines_pt = anno_raw['corners'], anno_raw['lines']
      else:
        corners_pt, lines_pt = meter_2_pixel(anno_style, pixel_config, anno_raw['corners'], anno_raw['lines'],
                                           pcl_scope=anno_raw['pcl_scope'], scene=anno_raw['filename'])
      lines_pt_ordered = OBJ_REPS_PARSE.encode_obj(lines_pt.reshape(-1,4), 'RoLine2D_2p', obj_rep )
      #lines_pt_ordered = encode_line_rep(lines_pt, DIM_PARSE.OBJ_REP)
      line_sizes = np.linalg.norm(lines_pt_ordered[:,[2,3]] - lines_pt_ordered[:,[0,1]], axis=1)
      min_line_size = line_sizes.min()
      labels_line_corner = np.concatenate([anno_raw['line_cat_ids'], anno_raw['corner_cat_ids'] ], axis=0)

      anno_img['gt_bboxes'] = lines_pt_ordered
      anno_img['labels'] = anno_raw['line_cat_ids']
      anno_img['relations'] = anno_raw['relations']

      anno_img['min_line_size'] = min_line_size

      obj_dim = OBJ_REPS_PARSE._obj_dims[obj_rep]
      anno_img['bboxes_ignore'] = np.empty([0,obj_dim], dtype=np.float32)
      anno_img['mask'] = []
      anno_img['seg_map'] = None
      gt_bboxes = anno_img['gt_bboxes'][:,:4]
      assert gt_bboxes.max() < DIM_PARSE.IMAGE_SIZE

      for ele in BEIKE.edge_attributions:
        anno_img[ele] = anno_raw[ele]
      #assert gt_bboxes.min() >= 0
      if DEBUG_CFG.VISUAL_CONNECTIONS:
        show_connection( anno_img['gt_bboxes'], anno_img['relations'] )
      return anno_img


def load_pcl_scope(anno_folder):
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

def load_anno_1scene(anno_folder, filename, classes,  filter_edges=True, is_save_connection=False):
      always_load_walls = 1

      beike_clsinfo = BEIKE_CLSINFO(classes, always_load_walls)
      if 'wall' not in classes and always_load_walls:
        classes =  ['wall'] + classes

      file_path = os.path.join(anno_folder, filename)
      with open(file_path, 'r') as f:
        metadata = json.load(f)
        data = copy.deepcopy(metadata)
        points = data['points']
        lines = data['lines']
        line_items = data['lineItems']

        anno = defaultdict(list)
        anno['filename'] = filename
        anno['line_ids'] = []

        if 'wall' in classes:
          point_dict = {}

          for point in points:
            xy = np.array([point['x'], point['y']]).reshape(1,2)
            anno['corners'].append( xy )
            #anno['corner_ids'].append( point['id'] )
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
            #anno['line_ponit_ids'].append( line['points'] )
            anno['line_cat_ids'].append( beike_clsinfo._category_ids_map['wall'] )
            for ele in BEIKE.edge_atts:
              if ele in line:
                anno['e_'+ele].append( line[ele] )
              else:
                pass
                #rasie NotImplementedError

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
          #anno['corner_ids'].append( line_item['line']+'_start_point' )
          #anno['corner_ids'].append( line_item['line']+'_end_point' )
          #anno['corner_lines'].append( line_item['id'] )
          #anno['corner_lines'].append( line_item['id'] )
          anno['corner_cat_ids'].append( cat_id )
          anno['corner_cat_ids'].append( cat_id )
          #anno['corner_locked'].append( False )
          #anno['corner_locked'].append( False )

          anno['line_ids'].append(line_item['id'])
          anno['lines'].append( line_xy )
          #anno['line_ponit_ids'].append( [line_item['line']+'_start_point', line_item['line']+'_end_point' ] )
          anno['line_cat_ids'].append( cat_id )
          #for ele in ['curve', 'align', 'type', 'edgeComputed', 'thicknessComputed']:
          #  if ele in line_item:
          #    anno['line_'+ele].append( line_item[ele] )
          #  else:
          #    rasie NotImplemented
          #    pass

          pass

        #for ele in ['corners', 'lines', 'corner_ids', 'corner_cat_ids', 'corner_locked', '']:
        for ele in anno:
          if len(anno[ele])>0 and (not isinstance(anno[ele][0], str)):
              #if isinstance(anno[ele][0], int):
              if isinstance(anno[ele][0], np.ndarray):
                anno[ele] = np.concatenate(anno[ele], 0)
              else:
                anno[ele] = np.array(anno[ele])
              #if ele in BEIKE.edge_attributions:
              #  import pdb; pdb.set_trace()  # XXX BREAKPOINT
              #  pass

      anno['corners'] = anno['corners'].astype(np.float32)
      anno['lines'] = anno['lines'].astype(np.float32)
      anno['line_ids'] = np.array(anno['line_ids'])

      lines_leng = np.linalg.norm(anno['lines'][:,0] - anno['lines'][:,1], axis=-1)
      anno['line_length_min_mean_max'] = [lines_leng.min(), lines_leng.mean(), lines_leng.max()]

      pcl_scopes_all = load_pcl_scope(anno_folder)
      anno['pcl_scope'] = pcl_scopes_all[filename.split('.')[0]]

      # normalize zero
      pcl_scope = anno['pcl_scope']
      anno['corners'] = anno['corners'] - pcl_scope[0:1,:2:]
      anno['lines'] = anno['lines'] - pcl_scope[0:1,None,:2:]

      # fix unaligned
      scene_size =  anno['pcl_scope'][1,:2] -  anno['pcl_scope'][0,:2]
      scene_size = scene_size[[1,0]]
      scene_name = filename.split('.json')[0]

      anno['lines'] = fix_1_unaligned_scene(scene_name, anno['lines'], scene_size, 'std_2p')
      tmp = np.repeat( anno['corners'][:,None,:], 2, axis=1 )
      anno['corners'] = fix_1_unaligned_scene(scene_name, tmp, scene_size, 'std_2p')[:,0,:]

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
      return anno

def show_connection(bboxes, relations):
  n = bboxes.shape[0]
  for i in range(n):
    cids = np.where(relations[i])[0]
    #_show_objs_ls_points_ls( (512,512), [bboxes, bboxes[i:i+1], ], 'RoLine2D_UpRight_xyxy_sin2a', obj_colors=['white', 'green',])
    _show_objs_ls_points_ls( (512,512), [bboxes, bboxes[i:i+1], bboxes[cids]], 'RoLine2D_UpRight_xyxy_sin2a', obj_colors=['white', 'green', 'red'])
  pass

def load_relations_1scene(anno_folder, filename, classes):
  connection_dir = anno_folder.replace('json', 'relations')
  connect_file = os.path.join(connection_dir, filename.replace('.json', '.npy'))
  relations_dict = np.load(connect_file, allow_pickle=True).tolist()
  relations = [relations_dict[c] for c in classes if c!='background']
  relations = np.concatenate(relations, axis=0)
  return relations

def load_gt_lines_bk(img_meta, img, classes, filter_edges):
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
    lines, _ = rotate_lines_img(lines, img, rotate_angle, DIM_PARSE.OBJ_REP)
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
      lines_aligned = transfer_lines( lines_unaligned, line_obj_rep,
        image_size, angle, (cx, cy) )
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

def get_scene_pcl_scopes(data_path):
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
  npy_path = os.path.join(data_path, base+'/test')
  files = glob.glob(npy_path + '/*.npy')

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
  print(f'normnorm_method: {normnorm_method}')
  print(f'mean: {mean}')
  print(f'std : {std}')
  pass

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

def main(data_path):
  ANNO_PATH = os.path.join(data_path, 'json/')
  topview_path = os.path.join(data_path, 'TopView_VerD/train.txt')
  #topview_path = os.path.join(data_path, 'TopView_VerD/test.txt')

  scenes = ['3sr-fOoghhC9kiOaGrvr7f', '3Q92imFGVI1hZ5b0sDFFC3', '0Kajc_nnyZ6K0cRGCQJW56', '0WzglyWg__6z55JLLEE1ll', 'Akkq4Ch_48pVUAum3ooSnK']
  #scenes = BAD_SCENE_TRANSFERS_PCL

  classes = ['wall', 'door', 'window', ]
  #classes = [ 'door',  ]
  #classes = [ 'window', 'door', ]
  #classes = [ 'window', ]
  #classes = [ 'wall', ]
  is_save_connection = 0
  if is_save_connection:
    classes = ['wall', 'door', 'window', ]
  beike = BEIKE(ANNO_PATH, topview_path,
                classes = classes,
                filter_edges= 0,
                is_save_connection=is_save_connection,
                )
  #beike.find_unscanned_edges()
  if is_save_connection:
    beike.find_connection()

  if 0:
    for s in scenes:
      beike.show_scene_anno(s, True, 0)


  #for s in UNALIGNED_SCENES:
  #  beike.show_scene_anno(s, True, 45)

  #for i in range(len(beike)):
  #  beike.show_anno_img( i, True, 45 )
  #pass



if __name__ == '__main__':
  data_path = f'/home/z/Research/mmdetection/data/beike/processed_{DIM_PARSE.IMAGE_SIZE}'
  main(data_path)
  #get_scene_pcl_scopes(data_path)
  #cal_topview_npy_mean_std(data_path, base='TopView_All', normnorm_method='abs')
  #gen_images_from_npy(data_path)

