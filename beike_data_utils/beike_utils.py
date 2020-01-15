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

DATA_PATH = '/home/z/Research/mmdetection/data/beike100'
ANNO_PATH = os.path.join(DATA_PATH, 'json')

IMAGE_SIZE = 256

DEBUG = True

class BEIKE:
    _category_ids_map = {'wall':0, 'door':1, 'window':2, 'other':3}
    _catid_2_cat = {0:'wall', 1:'door', 2:'window', 3:'other'}

    def __init__(self, anno_folder):
        self.anno_folder = anno_folder
        #self.seperate_room_data_path = anno_folder.replace('json', 'seperate_room_data/test')

        json_files = os.listdir(anno_folder)
        assert len(json_files) > 0

        img_infos = []
        for jfn in json_files:
          anno_raw = self.load_anno_1scene(jfn)
          anno_img = BEIKE.raw_anno_to_img(anno_raw)
          filename = jfn.split('.')[0]+'.npy'
          img_info = {'filename': filename,
                      'ann': anno_img}
          img_infos.append(img_info)

        self.img_infos = img_infos

    def __len__(self):
      return len(self.img_infos)
    def rm_anno_withno_data(self, img_prefix):
      valid_inds = []
      valid_files = os.listdir(img_prefix)
      for i, img_info in enumerate(self.img_infos):
        filename = img_info['filename']
        if img_info['filename'] in valid_files:
          valid_inds.append(i)
      valid_img_infos = [self.img_infos[i] for i in valid_inds]
      self.img_infos = valid_img_infos
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
          pass

        for line_item in line_items:
          start_pt = np.array([line_item['startPointAt']['x'], line_item['startPointAt']['y']]).reshape(1,2)
          end_pt = np.array([line_item['endPointAt']['x'], line_item['endPointAt']['y']]).reshape(1,2)
          cat = line_item['is']
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
      corners_pt, lines_pt = BEIKE.meter_2_pixel(anno_raw['corners'], anno_raw['lines'])
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
    def meter_2_pixel(corners, lines, floor=False):
      '''
      corners: [n,2]
      liens: [m,2,2]
      '''
      min_xy = corners.min(axis=0)
      max_xy = corners.max(axis=0)
      size_xy = (max_xy - min_xy) * 1.2
      corners_pt = ((corners - min_xy) * IMAGE_SIZE / size_xy).astype(np.float32)
      lines_pt = ((lines - min_xy) * IMAGE_SIZE / size_xy).astype(np.float32)
      if floor:
        corners_pt = np.floor(corners_pt).astype(np.uint32)
        lines_pt = np.floor(lines_pt).astype(np.uint32)
      return corners_pt, lines_pt

    def load_data(self, idx, data_prefix):
      scene_name = self.img_infos[idx]['filename']
      seperate_room_file = os.path.join(data_prefix, scene_name+'.npy')
      data = np.load(seperate_room_file, allow_pickle=True).tolist()
      img = data['topview_image']
      #lines = data['line_coords']
      #room_map = data['room_map']
      #bg_idx = data['bg_idx']
      normal_image = data['topview_mean_normal']
      #lines = data['lines']

      if DEBUG and 0:
        cv2.imwrite(f'./img_{scene_name}.png', img)
        self.draw_anno_1scene(idx)
      return img


    def draw_anno_1scene(self, idx=None, scene_name=None):
      if idx is None:
        for i,img_info in enumerate(self.img_infos):
          if img_info['filename'] == scene_name:
            idx = i
            break
        assert idx is not None, f'canot fine scene name == {scene_name}'
      anno = self.img_infos[idx]
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      BEIKE.draw_anno(anno)

    @ staticmethod
    def draw_anno(anno):
      print('draw_anno in beike_data_utils/beike_utils.py')
      colors_corner = {'wall': (0,0,255), 'door': (0,255,0), 'window': (255,0,0), 'other':(255,255,255)}
      colors_line   = {'wall': (255,0,0), 'door': (0,255,255), 'window': (0,255,255), 'other':(100,100,0)}

      scene_name = anno['filename']
      corners =  anno['corners']
      corner_cat_ids = anno['corner_cat_ids']
      lines =  anno['lines']
      line_cat_ids = anno['line_cat_ids']
      #print(f'line_cat_ids: {line_cat_ids}')

      corners, lines = BEIKE.meter_2_pixel(corners, lines, floor=True)

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


# ------------------------------------------------------------------------------

if __name__ == '__main__':
  scene = 'UNd46FX2vC-YEOIF-Wx1aZ'
  scene = '0Kajc_nnyZ6K0cRGCQJW56'
  beike = BEIKE(ANNO_PATH)
  beike.draw_anno_1scene(scene_name = scene)

  pass

