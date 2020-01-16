# from floor-sp
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


IMAGE_SIZE = 256
LOAD_CLASSES = ['wall']

DEBUG = True
BAD_SCENES =  ['7w6zvVsOBAQK4h4Bne7caQ', 'IDZkUGse-74FIy2OqM2u_Y', 'B9Abt6B78a0j2eRcygHjqC']

class BEIKE:
    _category_ids_map = {'wall':0, 'door':1, 'window':2, 'other':3}
    _catid_2_cat = {0:'wall', 1:'door', 2:'window', 3:'other'}

    def __init__(self, anno_folder='data/beike100/json/'):
        assert  anno_folder[-5:] == 'json/'
        self.anno_folder = anno_folder
        self.seperate_room_path = anno_folder.replace('json', 'seperate_room_data/test')
        #self.seperate_room_data_path = anno_folder.replace('json', 'seperate_room_data/test')
        base_path = os.path.dirname( os.path.dirname(anno_folder) )
        scopes_file = os.path.join(base_path, 'scopes.json')
        with open(scopes_file, 'r') as f:
          self.scopes = json.load(f)

        json_files = os.listdir(anno_folder)
        assert len(json_files) > 0

        img_infos = []
        for jfn in json_files:
          scene_name = jfn.split('.')[0]
          if scene_name in BAD_SCENES:
            continue

          anno_raw = self.load_anno_1scene(jfn)
          scope = np.array(self.scopes[anno_raw['filename'].split('.')[0]] )
          anno_raw['scope'] = scope
          anno_img = BEIKE.raw_anno_to_img(anno_raw)
          filename = jfn.split('.')[0]+'.npy'
          img_info = {'filename': filename,
                      'ann': anno_img}
          if DEBUG and False:
            BEIKE.show_img_ann(img_info, self.seperate_room_path)
          img_infos.append(img_info)

        self.img_infos = img_infos
        n = len(self.img_infos)
        print(f'\nload {n} scenes\n')

    @staticmethod
    def show_img_ann(img_info, seperate_room_path):
      filename = img_info['filename']
      print(f'{filename}')
      file_path = os.path.join(seperate_room_path, filename)
      data = np.load(file_path, allow_pickle=True).tolist()
      img = data['topview_image']
      #mmcv.imshow(img)
      bboxes = img_info['ann']['bboxes']
      mmcv.imshow_bboxes(img, bboxes)
      pass

    def __len__(self):
      return len(self.img_infos)

    def getCatIds(self):
      return BEIKE._category_ids_map.values()

    def getImgIds(self):
      return list(range(len(self)))

    def load_anno_1scene(self, filename):
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
      #BEIKE.draw_anno(anno)
      return anno

    @staticmethod
    def raw_anno_to_img(anno_raw):
      anno_img = {}
      corners_pt, lines_pt = BEIKE.meter_2_pixel(anno_raw['corners'], anno_raw['lines'], scope=anno_raw['scope'], scene=anno_raw['filename'])
      anno_img['bboxes'] = BEIKE.line_to_bbox(lines_pt)
      anno_img['labels'] = anno_raw['line_cat_ids']
      anno_img['bboxes_ignore'] = np.empty([0,4], dtype=np.float32)
      anno_img['mask'] = []
      anno_img['seg_map'] = None
      return anno_img

    @staticmethod
    def line_to_bbox(lines):
      '''
      lings: [[x0,y0], [x1,y1]]
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
      mask = (bboxes[:,2:4] - bboxes[:,0:2]) < 2
      bboxes[:,[0,1]] -= mask
      bboxes[:,[2,3]] += mask

      bboxes = np.clip(bboxes, a_min=0, a_max=IMAGE_SIZE-1)

      #img = np.zeros([256,256,3])
      #mmcv.imshow_bboxes(img, bboxes)
      return bboxes

    @staticmethod
    def meter_2_pixel(corners, lines, scope=None, floor=False, scene=None):
      '''
      corners: [n,2]
      liens: [m,2,2]
      '''
      if scope is None:
        min_xy = corners.min(axis=0)
        max_xy = corners.max(axis=0)
      else:
        min_xy = scope[0,0:2]
        max_xy = scope[1,0:2]

      max_range = (max_xy - min_xy).max()
      padding = max_range * 0.05
      min_xy = (min_xy + max_xy) / 2 - max_range / 2 - padding
      max_range += padding * 2

      corners_pt = ((corners - min_xy) * IMAGE_SIZE / max_range).astype(np.float32)
      lines_pt = ((lines - min_xy) * IMAGE_SIZE / max_range).astype(np.float32)

      if not( corners_pt.min() > -1 and corners_pt.max() < IMAGE_SIZE ):
            print(scene)
            print(corners_pt.min())
            print(corners_pt.max())
            pass
      corners_pt = np.clip(corners_pt, a_min=0, a_max=IMAGE_SIZE-1)
      lines_pt = np.clip(lines_pt, a_min=0, a_max=IMAGE_SIZE-1)
      if floor:
        corners_pt = np.floor(corners_pt).astype(np.uint32)
        lines_pt = np.floor(lines_pt).astype(np.uint32)
      return corners_pt, lines_pt

    def get_scene_index(self, scene_name):
      for i,img_info in enumerate(self.img_infos):
        if img_info['filename'].split('.')[0] == scene_name:
          return i
      assert True, f'cannot fine scene {scene_name}'

    def load_data(self, scene_name):
      seperate_room_path = self.anno_folder.replace('json', 'seperate_room_data/test')
      seperate_room_file = os.path.join(seperate_room_path, scene_name+'.npy')
      data = np.load(seperate_room_file, allow_pickle=True).tolist()
      img = data['topview_image']
      #lines = data['line_coords']
      #room_map = data['room_map']
      #bg_idx = data['bg_idx']
      normal_image = data['topview_mean_normal']
      lines = data['lines']
      point_dict = data['point_dict']

      #ann = self.load_anno_1scene(scene_name+'.json')
      #corners = ann['corners']

      #cv2.imwrite(f'./img_{scene_name}.png', img)
      return img

    def draw_1scene_img(self, scene_name):
      anno = self.load_anno_1scene(scene_name+'.json')
      img = self.load_data(scene_name)
      anno['scope'] = np.array(self.scopes[scene_name])
      BEIKE.draw_anno(anno, img)

    def draw_1scene_pcl(self, scene_name):
      anno = self.load_anno_1scene(scene_name+'.json')
      img = self.load_data(scene_name)
      anno['scope'] = np.array(self.scopes[scene_name])
      BEIKE.draw_anno(anno, img)

    @ staticmethod
    def draw_anno(anno, img=None):
      print('draw_anno in beike_data_utils/beike_utils.py')
      colors_corner = {'wall': (0,0,255), 'door': (0,255,0), 'window': (255,0,0), 'other':(255,255,255)}
      colors_line   = {'wall': (255,0,0), 'door': (0,255,255), 'window': (0,255,255), 'other':(100,100,0)}

      scene_name = anno['filename']
      corners =  anno['corners']
      corner_cat_ids = anno['corner_cat_ids']
      lines =  anno['lines']
      line_cat_ids = anno['line_cat_ids']
      #print(f'line_cat_ids: {line_cat_ids}')

      corners, lines = BEIKE.meter_2_pixel(corners, lines, scope=anno['scope'], floor=True)

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
      cv2.imwrite(f'./anno_{scene_name}.png', img)
      pass


def get_scene_scopes(data_path):
  ply_path = os.path.join(data_path, 'ply')
  pcl_files = os.listdir(ply_path)
  scopes = {}
  for pclf in pcl_files:
    pcl_file = os.path.join(ply_path, pclf)
    scene_name = pclf.split('.')[0]
    with open(pcl_file, 'rb') as f:
      plydata = PlyData.read(f)
    points_data = np.array(plydata['vertex'].data.tolist())
    xyz_min = points_data[:,0:3].min(0, keepdims=True)
    xyz_max = points_data[:,0:3].max(0, keepdims=True)
    xyz_min_max = np.concatenate([xyz_min, xyz_max], 0)
    scopes[scene_name] = xyz_min_max.tolist()
    print(f'{scene_name}: \n{xyz_min_max}\n')

  scopes_file = os.path.join(data_path, 'scopes.json')
  with open(scopes_file, 'w') as f:
    json.dump(scopes, f)
  print(f'save {scopes_file}')
# ------------------------------------------------------------------------------

if __name__ == '__main__':
  DATA_PATH = '/home/z/Research/mmdetection/data/beike100'
  ANNO_PATH = os.path.join(DATA_PATH, 'json')

  #get_scene_scopes(DATA_PATH)


  scene = '7w6zvVsOBAQK4h4Bne7caQ'  # 10
  #scene = 'IDZkUGse-74FIy2OqM2u_Y' # 20
  #scene = 'B9Abt6B78a0j2eRcygHjqC' # 15

  beike = BEIKE(ANNO_PATH)
  beike.draw_1scene_img(scene_name = scene)

  pass

