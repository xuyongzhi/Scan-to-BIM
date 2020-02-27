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

from configs.common import OBJ_DIM, OBJ_REP, CORNER_FLAG, INCLUDE_CORNERS, IMAGE_SIZE
from beike_data_utils.line_utils import encode_line_rep, rotate_lines_img, transfer_lines
from mmdet.debug_tools import get_random_color, show_img_with_norm
np.set_printoptions(precision=3, suppress=True)

#LOAD_CLASSES = ['wall', 'window', 'door']
LOAD_CLASSES = ['wall']

DEBUG = True
UNALIGNED_SCENES =  ['7w6zvVsOBAQK4h4Bne7caQ', 'IDZkUGse-74FIy2OqM2u_Y',
                    'B9Abt6B78a0j2eRcygHjqC', 'Akkq4Ch_48pVUAum3ooSnK',
                    'w2BaBfwjX0iN2cMjvpUNfa', 'yY5OzetjnLred7G8oOzZr1',
                    'wIGxjDDGkPk3udZW4vo-ic', ]
gt_out_pcl = ['vZIjhovtYde9e2qUjMzvz3', 'pMh3-KweDzzp_b_HI11eMA',
              'vYlCbx-H_v_uvacuiMq0no']
lost_gt_scenes = ['vYlCbx-H_v_uvacuiMq0no']
#BAD_SCENES = ['7w6zvVsOBAQK4h4Bne7caQ', 'IDZkUGse-74FIy2OqM2u_Y','B9Abt6B78a0j2eRcygHjqC']
BAD_SCENES = []
WRITE_ANNO_IMG = 0

BAD_SCENE_TRANSFERS_1024 = {'7w6zvVsOBAQK4h4Bne7caQ': (-44,-208,-153),
                            'IDZkUGse-74FIy2OqM2u_Y': (30,-97,58),
                            'B9Abt6B78a0j2eRcygHjqC': (44,-52,93),
                            'Akkq4Ch_48pVUAum3ooSnK': (2,9,0),
                            'w2BaBfwjX0iN2cMjvpUNfa': (2,10,3),
                            'yY5OzetjnLred7G8oOzZr1': (-2,-7,0),
                            'wIGxjDDGkPk3udZW4vo-ic': (-1,7,5),
                            'vZIjhovtYde9e2qUjMzvz3': (-1, 5, 0)}

class BEIKE:
    _category_ids_map = {'wall':1, 'door':2, 'window':3, 'other':4}
    _catid_2_cat = {1:'wall', 2:'door', 3:'window', 4:'other'}

    def __init__(self, anno_folder='data/beike100/json/', topview_path=''):
        assert  anno_folder[-5:] == 'json/'
        self.anno_folder = anno_folder
        self.topview_path = topview_path
        if WRITE_ANNO_IMG:
          self.anno_img_folder = self.anno_folder.replace('json', 'anno_imgs')
          if not os.path.exists(self.anno_img_folder):
            os.makedirs(self.anno_img_folder)
        self.seperate_room_path = anno_folder.replace('json', 'seperate_room_data/test')
        #self.seperate_room_data_path = anno_folder.replace('json', 'seperate_room_data/test')
        base_path = os.path.dirname( os.path.dirname(anno_folder) )
        pcl_scopes_file = os.path.join(base_path, 'pcl_scopes.json')
        with open(pcl_scopes_file, 'r') as f:
          self.pcl_scopes = json.load(f)

        json_files = os.listdir(anno_folder)
        assert len(json_files) > 0

        img_infos = []
        all_min_line_sizes = []
        for jfn in json_files:
          scene_name = jfn.split('.')[0]
          anno_raw = load_anno_1scene(self.anno_folder, jfn)
          pcl_scope = np.array(self.pcl_scopes[anno_raw['filename'].split('.')[0]] )
          anno_raw['pcl_scope'] = pcl_scope
          anno_img = raw_anno_to_img(anno_raw)
          filename = jfn.split('.')[0]+'.npy'
          #filename = jfn.split('.')[0]+'.density.png'
          img_info = {'filename': filename,
                      'ann': anno_img,
                      'ann_raw': anno_raw}
          img_infos.append(img_info)
          all_min_line_sizes.append(anno_img['min_line_size'])

        self.img_infos = img_infos
        self.all_min_line_sizes = np.array( all_min_line_sizes )
        print(f'min line size: {self.all_min_line_sizes.min()}')

        self.rm_bad_scenes()
        self.fix_unaligned_scenes()

        n0 = len(self.img_infos)
        if WRITE_ANNO_IMG:
          for i in range(n0):
            #self.draw_anno_raw(i, with_img=1)
            self.show_anno_img(i, with_img=1)


    def fix_unaligned_scenes(self):
      n0 = len(self.img_infos)
      for i in range(n0):
        sn = self.img_infos[i]['filename'].split('.')[0]
        if sn in UNALIGNED_SCENES:
          angle, cx, cy = BAD_SCENE_TRANSFERS_1024[sn]
          scale = IMAGE_SIZE / 1024.0
          self.img_infos[i]['ann']['bboxes'] = transfer_lines(
            self.img_infos[i]['ann']['bboxes'], OBJ_REP,
            (IMAGE_SIZE, IMAGE_SIZE), angle, (cx*scale, cy*scale))
          pass

    def rm_bad_scenes(self):
      valid_ids = []
      n0 = len(self.img_infos)
      for i in range(n0):
        sn = self.img_infos[i]['filename'].split('.')[0]
        if sn not in BAD_SCENES:
          valid_ids.append(i)
      self.img_infos = [self.img_infos[i] for i in valid_ids]
      n1 = len(self.img_infos)
      print(f'\nload {n0} scenes with {n1} valid\n')

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
      return list(BEIKE._category_ids_map.values())

    def getImgIds(self):
      return list(range(len(self)))

    def unused_load_anno_1scene(self, filename):
      file_path = os.path.join(self.anno_folder, filename)
      with open(file_path, 'r') as f:
        metadata = json.load(f)
        data = copy.deepcopy(metadata)
        points = data['points']
        lines = data['lines']
        line_items = data['lineItems']

        anno = defaultdict(list)
        anno['filename'] = filename

        if 'wall' in LOAD_CLASSES:
          point_dict = {}

          for point in points:
            xy = np.array([point['x'], point['y']]).reshape(1,2)
            anno['corners'].append( xy )
            #anno['corner_ids'].append( point['id'] )
            #anno['corner_lines'].append( point['lines'] )
            anno['corner_cat_ids'].append( BEIKE._category_ids_map['wall'] )
            #anno['corner_locked'].append( point['locked'] )
            point_dict[point['id']] = xy
            pass

          for line in lines:
            point_id_1, point_id_2 = line['points']
            xy1 = point_dict[point_id_1]
            xy2 = point_dict[point_id_2]
            line_xys = np.array([xy1, xy2]).reshape(1,2,2)
            anno['lines'].append( line_xys )
            #anno['line_ids'].append( line['id']  )
            #anno['line_ponit_ids'].append( line['points'] )
            anno['line_cat_ids'].append( BEIKE._category_ids_map['wall'] )
            #for ele in ['curve', 'align', 'type', 'edgeComputed', 'thicknessComputed']:
            #  if ele in line:
            #    anno['line_'+ele].append( line[ele] )
            #  else:
            #    rasie NotImplemented
            if filename == '7w6zvVsOBAQK4h4Bne7caQ.json':
              pass
            pass

        for line_item in line_items:
          cat = line_item['is']
          if cat not in LOAD_CLASSES:
            continue
          start_pt = np.array([line_item['startPointAt']['x'], line_item['startPointAt']['y']]).reshape(1,2)
          end_pt = np.array([line_item['endPointAt']['x'], line_item['endPointAt']['y']]).reshape(1,2)
          cat_id = BEIKE._category_ids_map[cat]
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

          anno['lines'].append( line_xy )
          #anno['line_ids'].append( line_item['id']  )
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
            if isinstance(anno[ele][0], int):
              anno[ele] = np.array(anno[ele])
            else:
              anno[ele] = np.concatenate(anno[ele], 0)

      anno['corners'] = anno['corners'].astype(np.float32)
      anno['lines'] = anno['lines'].astype(np.float32)

      lines_leng = np.linalg.norm(anno['lines'][:,0] - anno['lines'][:,1], axis=-1)
      anno['line_length_min_mean_max'] = [lines_leng.min(), lines_leng.mean(), lines_leng.max()]
      return anno

    @staticmethod
    def unused_line_to_bbox(lines):
      '''
      input: lings: [n,2,2] [[x0,y0], [x1,y1]]
      output: bboxes: [n,4]

      x: width
      y: height
      original point (x=0,y=0) is left-top
      bbox format: [bbox_left, bbox_up, bbox_right, bbox_bottom]
      '''

      n = lines.shape[0]
      bboxes = np.empty([n,4], dtype=lines.dtype)
      bboxes[:,[0,1]] = lines.min(axis=1)
      bboxes[:,[2,3]] = lines.max(axis=1)
      # aug thickness for 2 pixels
      #mask = (bboxes[:,2:4] - bboxes[:,0:2]) < 2
      #bboxes[:,[0,1]] -= mask
      #bboxes[:,[2,3]] += mask

      bboxes = np.clip(bboxes, a_min=0, a_max=IMAGE_SIZE-1)

      #img = np.zeros([256,256,3])
      #mmcv.imshow_bboxes(img, bboxes)
      return bboxes

    @staticmethod
    def meter_2_pixel(corners, lines, pcl_scope=None, floor=False, scene=None):
      '''
      corners: [n,2]
      liens: [m,2,2]
      '''
      if pcl_scope is None:
        min_xy = corners.min(axis=0)
        max_xy = corners.max(axis=0)
      else:
        min_xy = pcl_scope[0,0:2]
        max_xy = pcl_scope[1,0:2]

      max_range = (max_xy - min_xy).max()
      padding = max_range * 0.05
      min_xy = (min_xy + max_xy) / 2 - max_range / 2 - padding
      max_range += padding * 2

      corners_pt = ((corners - min_xy) * IMAGE_SIZE / max_range).astype(np.float32)
      lines_pt = ((lines - min_xy) * IMAGE_SIZE / max_range).astype(np.float32)

      if not( corners_pt.min() > -1 and corners_pt.max() < IMAGE_SIZE ):
            scene_name = scene.split('.')[0]
            if scene_name not in UNALIGNED_SCENES:
              print(scene)
              print(corners_pt.min())
              print(corners_pt.max())
              import pdb; pdb.set_trace()  # XXX BREAKPOINT
              pass
      corners_pt = np.clip(corners_pt, a_min=0, a_max=IMAGE_SIZE-1)
      lines_pt = np.clip(lines_pt, a_min=0, a_max=IMAGE_SIZE-1)
      if floor:
        corners_pt = np.floor(corners_pt).astype(np.uint32)
        lines_pt = np.floor(lines_pt).astype(np.uint32)

      line_size = np.linalg.norm( lines[:,0] - lines[:,1], axis=1 )
      line_size_pt = np.linalg.norm( lines_pt[:,0] - lines_pt[:,1], axis=1 )
      #assert line_size_pt.min() > 3
      return corners_pt, lines_pt

    def get_scene_index(self, scene_name):
      for i,img_info in enumerate(self.img_infos):
        if img_info['filename'].split('.')[0] == scene_name:
          return i
      assert True, f'cannot fine scene {scene_name}'

    def load_data(self, scene_name):
      seperate_room_path = self.anno_folder.replace('json', f'{self.topview_path}/test')
      seperate_room_file = os.path.join(seperate_room_path, scene_name+'.npy')
      data = np.load(seperate_room_file, allow_pickle=True).tolist()
      img = np.expand_dims(data['topview_image'],axis=2)
      #lines = data['line_coords']
      #room_map = data['room_map']
      #bg_idx = data['bg_idx']
      normal_image = data['topview_mean_normal']
      #lines = data['lines']
      #point_dict = data['point_dict']

      #ann = self.load_anno_1scene(scene_name+'.json')
      #corners = ann['corners']

      #cv2.imwrite(f'./img_{scene_name}.png', img)
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
      bboxes, bbox_labels, corners, corner_labels = \
            split_line_corner(bboxes, labels)

      scene_name = self.img_infos[idx]['filename'].split('.')[0]
      print(f'{scene_name}')

      if not with_img:
        img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
      else:
        img = self.load_data(scene_name)


      if (np.array(lines_transfer) != 0).any():
        angle, cx, cy = lines_transfer
        bboxes = transfer_lines(bboxes, OBJ_REP, img.shape[:2], angle, (cx,cy))

      if rotate_angle != 0:
        bboxes, img = rotate_lines_img(bboxes, img, rotate_angle,
                                      OBJ_REP, check_by_cross=False)

      #show_img_with_norm(img)

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

      corners, lines = BEIKE.meter_2_pixel(corners, lines, pcl_scope=anno['pcl_scope'], floor=True)

      if img is None:
        img = np.zeros([IMAGE_SIZE, IMAGE_SIZE, 3], dtype=np.uint8)
      for i in range(corners.shape[0]):
        obj = BEIKE._catid_2_cat[ corner_cat_ids[i] ]
        cv2.circle(img, (corners[i][0], corners[i][1]), 2, colors_corner[obj], -1)

      for i in range(lines.shape[0]):
        obj = BEIKE._catid_2_cat[ line_cat_ids[i] ]
        s, e = lines[i]
        cv2.line(img, (s[0], s[1]), (e[0], e[1]), colors_line[obj])
        if obj != 'wall':
          cv2.putText(img, obj, (s[0], s[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    (255, 255, 255), 1)

      if WRITE_ANNO_IMG:
        anno_img_file = os.path.join(self.anno_img_folder, scene_name+'.png')
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
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


def raw_anno_to_img(anno_raw):
      anno_img = {}
      corners_pt, lines_pt = BEIKE.meter_2_pixel(anno_raw['corners'], anno_raw['lines'], pcl_scope=anno_raw['pcl_scope'], scene=anno_raw['filename'])
      lines_pt_ordered = encode_line_rep(lines_pt, OBJ_REP)
      line_sizes = np.linalg.norm(lines_pt_ordered[:,[2,3]] - lines_pt_ordered[:,[0,1]], axis=1)
      min_line_size = line_sizes.min()
      corners_pt_inbox = corners_as_boxformat(corners_pt)
      bboxes_line_corner = np.concatenate([lines_pt_ordered, corners_pt_inbox], axis=0)
      labels_line_corner = np.concatenate([anno_raw['line_cat_ids'], anno_raw['corner_cat_ids'] ], axis=0)

      if INCLUDE_CORNERS:
        anno_img['bboxes'] = bboxes_line_corner
        anno_img['labels'] = labels_line_corner
      else:
        anno_img['bboxes'] = lines_pt_ordered
        anno_img['labels'] = anno_raw['line_cat_ids']

      anno_img['min_line_size'] = min_line_size


      anno_img['bboxes_ignore'] = np.empty([0,OBJ_DIM], dtype=np.float32)
      anno_img['mask'] = []
      anno_img['seg_map'] = None
      bboxes = anno_img['bboxes'][:,:4]
      assert bboxes.max() < IMAGE_SIZE
      assert bboxes.min() >= 0
      return anno_img


def load_anno_1scene(anno_folder, filename):
      file_path = os.path.join(anno_folder, filename)
      with open(file_path, 'r') as f:
        metadata = json.load(f)
        data = copy.deepcopy(metadata)
        points = data['points']
        lines = data['lines']
        line_items = data['lineItems']

        anno = defaultdict(list)
        anno['filename'] = filename

        if 'wall' in LOAD_CLASSES:
          point_dict = {}

          for point in points:
            xy = np.array([point['x'], point['y']]).reshape(1,2)
            anno['corners'].append( xy )
            #anno['corner_ids'].append( point['id'] )
            #anno['corner_lines'].append( point['lines'] )
            anno['corner_cat_ids'].append( BEIKE._category_ids_map['wall'] )
            #anno['corner_locked'].append( point['locked'] )
            point_dict[point['id']] = xy
            pass

          for line in lines:
            point_id_1, point_id_2 = line['points']
            xy1 = point_dict[point_id_1]
            xy2 = point_dict[point_id_2]
            line_xys = np.array([xy1, xy2]).reshape(1,2,2)
            anno['lines'].append( line_xys )
            #anno['line_ids'].append( line['id']  )
            #anno['line_ponit_ids'].append( line['points'] )
            anno['line_cat_ids'].append( BEIKE._category_ids_map['wall'] )
            #for ele in ['curve', 'align', 'type', 'edgeComputed', 'thicknessComputed']:
            #  if ele in line:
            #    anno['line_'+ele].append( line[ele] )
            #  else:
            #    rasie NotImplemented
            if filename == '7w6zvVsOBAQK4h4Bne7caQ.json':
              pass
            pass

        for line_item in line_items:
          cat = line_item['is']
          if cat not in LOAD_CLASSES:
            continue
          start_pt = np.array([line_item['startPointAt']['x'], line_item['startPointAt']['y']]).reshape(1,2)
          end_pt = np.array([line_item['endPointAt']['x'], line_item['endPointAt']['y']]).reshape(1,2)
          cat_id = BEIKE._category_ids_map[cat]
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

          anno['lines'].append( line_xy )
          #anno['line_ids'].append( line_item['id']  )
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
            if isinstance(anno[ele][0], int):
              anno[ele] = np.array(anno[ele])
            else:
              anno[ele] = np.concatenate(anno[ele], 0)

      anno['corners'] = anno['corners'].astype(np.float32)
      anno['lines'] = anno['lines'].astype(np.float32)

      lines_leng = np.linalg.norm(anno['lines'][:,0] - anno['lines'][:,1], axis=-1)
      anno['line_length_min_mean_max'] = [lines_leng.min(), lines_leng.mean(), lines_leng.max()]
      return anno


def split_line_corner(bboxes, labels):
  assert bboxes.shape[1] == 5
  mask = np.abs(np.abs(bboxes[:,-1]) - CORNER_FLAG) < 1e-5
  lines = bboxes[mask==0]
  line_labels = labels[mask==0]
  corners = bboxes[mask]
  corner_labels = labels[mask]
  return lines, line_labels, corners, corner_labels

def corners_as_boxformat(corners_pt):
  n = corners_pt.shape[0]
  tmp0 = np.zeros((n,1), dtype = corners_pt.dtype) + CORNER_FLAG
  tmp1 = np.clip(corners_pt.copy() + 0, a_min=0, a_max=IMAGE_SIZE-1)
  corners_as_box = np.concatenate([corners_pt, tmp1, tmp0], axis=1)
  return corners_as_box


def _UnUsed_gen_images_from_npy(data_path):
  npy_path = os.path.join(data_path, 'seperate_room_data/test')
  den_image_path = os.path.join(data_path, f'images/public100_{IMAGE_SIZE}')
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
    with open(pcl_file, 'rb') as f:
      plydata = PlyData.read(f)
    points_data = np.array(plydata['vertex'].data.tolist())
    xyz_min = points_data[:,0:3].min(0, keepdims=True)
    xyz_max = points_data[:,0:3].max(0, keepdims=True)
    xyz_min_max = np.concatenate([xyz_min, xyz_max], 0)
    pcl_scopes[scene_name] = xyz_min_max.tolist()
    print(f'{scene_name}: \n{xyz_min_max}\n')

  with open(pcl_scopes_file, 'w') as f:
    json.dump(pcl_scopes, f)
  print(f'save {pcl_scopes_file}')

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
  den_image_path = os.path.join(data_path, f'images/public100_{IMAGE_SIZE}')
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
  topview_path = 'TopView_All'

  scenes = ['0Kajc_nnyZ6K0cRGCQJW56', '0WzglyWg__6z55JLLEE1ll', 'Akkq4Ch_48pVUAum3ooSnK']

  beike = BEIKE(ANNO_PATH, topview_path)

  for s in scenes:
    beike.show_scene_anno(s, True, 0)


  #for s in UNALIGNED_SCENES:
  #  beike.show_scene_anno(s, True, 45)

  #for i in range(len(beike)):
  #  beike.show_anno_img( i, True, 45 )
  #pass



if __name__ == '__main__':
  data_path = f'/home/z/Research/mmdetection/data/beike/processed_{IMAGE_SIZE}'
  main(data_path)
  #get_scene_pcl_scopes(DATA_PATH)
  #cal_topview_npy_mean_std(data_path, base='TopView_All', normnorm_method='abs')
  #gen_images_from_npy(data_path)

