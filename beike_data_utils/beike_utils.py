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

DATA_PATH = '/home/z/Research/mmdetection/data/beike100'
ANNO_PATH = os.path.join(DATA_PATH, 'json')
SEPERATE_ROOM_PATH = os.path.join(DATA_PATH,'seperate_room_data/test')

TOP_DOWN_VIEW_PATH = './demo_nonm/map/point_evidence_visualize_0.jpg'
EXTRINSICS_PATH = './demo_nonm/final_extrinsics.txt'

IMAGE_SIZE = 256
ANNOT_OFFSET = 37500
ANNOT_SCALE = 1000


class BEIKE:
    _category_ids_map = {'wall':0, 'door':1, 'window':2, 'other':3}
    def __init__(self, anno_folder):
        self.anno_folder = anno_folder
        json_files = os.listdir(anno_folder)
        assert len(json_files) > 0

        img_infos = []
        for filename in json_files:
          info = {'filename': filename}
          info.update(self.load_anno_1scene(filename))
          img_infos.append(info)

        self.img_infos = img_infos
        self.img_num = len(img_infos)

    def getCatIds(self):
      return BEIKE._category_ids_map.values()

    def getImgIds(self):
      return list(range(self.img_num))

    def load_anno_1scene(self, filename):
      file_path = os.path.join(self.anno_folder, filename)
      with open(file_path, 'r') as f:
        metadata = json.load(f)
        data = copy.deepcopy(metadata)
        points = data['points']
        lines = data['lines']
        line_items = data['lineItems']

        anno = defaultdict(list)

        point_dict = {}

        for point in points:
          xy = np.array([point['x'], point['y']]).reshape(1,2)
          anno['corners'].append( xy )
          anno['corner_ids'].append( point['id'] )
          anno['corner_lines'].append( point['lines'] )
          anno['corner_cat_ids'].append( BEIKE._category_ids_map['wall'] )
          anno['corner_locked'].append( point['locked'] )
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
          anno['line_cat_ids'].append( BEIKE._category_ids_map['wall'] )
          for ele in ['curve', 'align', 'type', 'edgeComputed', 'thicknessComputed']:
            if ele in line:
              anno['line_'+ele].append( line[ele] )
          pass

        #for ele in ['corners', 'lines', 'corner_ids', 'corner_cat_ids', 'corner_locked', '']:
        for ele in anno:
          if len(anno[ele])>0 and (not isinstance(anno[ele][0], str)):
            if isinstance(anno[ele][0], int):
              anno[ele] = np.array(anno[ele])
            else:
              anno[ele] = np.concatenate(anno[ele], 0)

      #for ele in ['line_thicknessComputed', 'line_thicknessComputed']:
      #  print( ele, anno[ele] )
      return anno

def load_annot_all_scenes(anno_folder):
  room_json_files = os.listdir(anno_folder)
  room_json_files = [os.path.join(anno_folder, r) for r in room_json_files]
  annos = []
  for room in room_json_files:
    anno = load_annot_1scene(room)
    annos.append(anno)
  return annos

def load_annot_1scene(scene_anno_file):
  '''
  ['points', 'lines', 'lineItems', 'areas']
  '''
  #scene_anno_file = os.path.join(ANNO_PATH, f'{scene}.json')
  with open(scene_anno_file, 'r') as f:
    metadata = json.load(f)
  return load_annot(metadata)


def load_annot(annot):
    show = True
    data = copy.deepcopy(annot)
    points = data['points']
    lines = data['lines']
    line_items = data['lineItems']

    # wall points
    point_dict = dict()
    all_x = list()
    all_y = list()
    for point in points:
        point_dict[point['id']] = point
        all_x.append(point['x'])
        all_y.append(point['y'])

    wall_points = np.concatenate([  np.array(all_x).reshape(-1,1), np.array(all_y).reshape(-1,1) ], 1 )

    # wall lines
    wall_lines = []
    for line in lines:
        assert len(line['points']) == 2
        point_id_1, point_id_2 = line['points']
        xy1 = [point_dict[point_id_1]['x'], point_dict[point_id_1]['y']]
        xy2 = [point_dict[point_id_2]['x'], point_dict[point_id_2]['y']]
        line_xy = np.expand_dims( np.array([xy1, xy2]), 0)
        wall_lines.append( line_xy )

    wall_lines = np.concatenate(wall_lines, 0)

    all_corners   = {'wall': wall_points, 'door': [], 'window': [], 'other': []}
    all_lines = {'wall': wall_lines,  'door': [], 'window': [], 'other': []}

    # other classes: doors, windows
    for line_item in line_items:
        obj = line_item['is']
        start_pt = (line_item['startPointAt']['x'], line_item['startPointAt']['y'])
        end_pt = (line_item['endPointAt']['x'], line_item['endPointAt']['y'])

        obj_line = np.expand_dims(np.array([start_pt, end_pt]), 0)
        all_lines[obj].append( obj_line )

    # convert unit from meter to pixel
    min_corner_wall = all_corners['wall'].min(axis=0)
    max_corner_wall = all_corners['wall'].max(axis=0)
    size = (max_corner_wall - min_corner_wall) * 1.2

    all_lines_pt = {}
    all_corners_pt = {}
    for obj in all_lines:
      if obj != 'wall' and len(all_lines[obj]):
          all_lines[obj] = np.concatenate(all_lines[obj], 0)
          all_corners[obj] = all_lines[obj].reshape(-1,2)
          #print(f'{obj} corners: {all_corners[obj].shape}')
          #print(f'{obj} lines: {all_lines[obj].shape}')

          all_lines_pt[obj]   = np.floor((all_lines[obj]   - min_corner_wall) / size * IMAGE_SIZE).astype(np.uint16)
          all_corners_pt[obj] = np.floor((all_corners[obj] - min_corner_wall) / size * IMAGE_SIZE).astype(np.uint16)

    #draw_anno_dict(all_corners_pt, all_lines_pt)
    annotation_by_classes = {'all_corners': all_corners, 'all_lines': all_lines, 'all_corners_pt': all_corners_pt, 'all_lines_pt': all_lines_pt}
    anno = merge_anno_classes(annotation_by_classes)
    return anno

def merge_anno_classes(annotation_by_classes):
    corners = []
    anno_with_category_ids = {}
    for data_type in ['all_corners', 'all_lines', 'all_corners_pt', 'all_lines_pt']:
      gts = []
      category_ids = []
      for obj in annotation_by_classes[data_type]:
        if len(annotation_by_classes[data_type][obj])==0:
          continue
        gts.append( annotation_by_classes[data_type][obj] )
        label = np.ones([len(gts[-1])]) * BEIKE._category_ids_map[obj]
        category_ids.append(label)
      gts = np.concatenate(gts, 0)
      category_ids = np.concatenate(category_ids, 0)

      anno_with_category_ids[data_type] = {'gts': gts, 'category_ids':category_ids}
    return anno_with_category_ids

def draw_anno_dict(all_corners_pt, all_lines_pt):
    colors_corner = {'wall': (0,0,255), 'door': (0,255,0), 'window': (255,0,0), 'other':(255,255,255)}
    colors_line   = {'wall': (255,0,0), 'door': (0,255,255), 'window': (0,255,255), 'other':(100,100,0)}
    img = np.zeros([IMAGE_SIZE, IMAGE_SIZE, 3], dtype=np.uint8)
    for obj in all_corners_pt:
      #if obj != 'other':
      #  continue
      corners = all_corners_pt[obj]
      for i in range(corners.shape[0]):
        cv2.circle(img, (corners[i][0], corners[i][1]), 2, colors_corner[obj], -1)
        line = all_lines_pt[obj]

      lines = all_lines_pt[obj]
      for i in range(lines.shape[0]):
        s, e = lines[i]
        cv2.line(img, (s[0], s[1]), (e[0], e[1]), colors_line[obj])
    cv2.imwrite('./anno.png', img)


# ------------------------------------------------------------------------------

def visualize_annot(annot):
    data = copy.deepcopy(annot)
    points = data['points']
    lines = data['lines']
    line_items = data['lineItems']

    point_dict = dict()
    all_x = list()
    all_y = list()
    for point in points:
        point_dict[point['id']] = point
        all_x.append(point['x'])
        all_y.append(point['y'])

    img = np.zeros([IMAGE_SIZE, IMAGE_SIZE, 3], dtype=np.uint8)

    min_x = min(all_x)
    min_y = min(all_y)
    width = height = max(max(all_x) - min(all_x), max(all_y) - min(all_y)) * 1.2

    # draw all corners
    for point in points:
        img_x, img_y = draw_corner_with_scaling(img, (point['x'], point['y']), min_x, width, min_y, height,
                                                text=None)
        point_dict[point['id']]['img_x'] = img_x
        point_dict[point['id']]['img_y'] = img_y

    # draw all line segments
    for line in lines:
        assert len(line['points']) == 2
        point_id_1, point_id_2 = line['points']
        start_pt = (point_dict[point_id_1]['img_x'], point_dict[point_id_1]['img_y'])
        end_pt = (point_dict[point_id_2]['img_x'], point_dict[point_id_2]['img_y'])
        cv2.line(img, start_pt, end_pt, (255, 0, 0))

    # draw all line with category, such as doors, windows
    for line_item in line_items:
        start_pt = (line_item['startPointAt']['x'], line_item['startPointAt']['y'])
        end_pt = (line_item['endPointAt']['x'], line_item['endPointAt']['y'])
        img_start_pt = draw_corner_with_scaling(img, start_pt, min_x, width, min_y, height, color=(0, 255, 0))
        img_end_pt = draw_corner_with_scaling(img, end_pt, min_x, width, min_y, height, color=(0, 255, 0))
        line_item['img_start_pt'] = img_start_pt
        line_item['img_end_pt'] = img_end_pt
        if line_item['is'] == 'window':
          print(start_pt, img_start_pt)
          print(end_pt, img_end_pt)
        cv2.line(img, img_start_pt, img_end_pt, (0, 255, 255))
        cv2.putText(img, line_item['is'], (img_start_pt[0], img_start_pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    (255, 255, 255), 1)
        pass

    return img


def draw_corner_with_scaling(img, corner, min_x, width, min_y, height, color=(0, 0, 255), text=None):
    img_x = int(math.floor((corner[0] - min_x) * 1.0 / width * IMAGE_SIZE))
    img_y = int(math.floor((corner[1] - min_y) * 1.0 / height * IMAGE_SIZE))
    cv2.circle(img, (img_x, img_y), 2, color, -1)
    if text is not None:
        cv2.putText(img, text, (img_x, img_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1, 1)
    return img_x, img_y


def visualize_annot_scene(scene):
  '''
  ['points', 'lines', 'lineItems', 'areas']
  '''
  scene_anno_file = os.path.join(ANNO_PATH, f'{scene}.json')
  with open(scene_anno_file, 'r') as f:
    metadata = json.load(f)
  img = visualize_annot(metadata)
  img_path = f'./{scene}_anno.png'
  cv2.imwrite(img_path, img)
  #cv2.imshow('ano', img)
  pass


def visualize_annot_sep_room(scene):
    sep_room_file = os.path.join(SEPERATE_ROOM_PATH, f'{scene}.npy')
    file_path = sep_room_file
    data = np.load(file_path, encoding='latin1', allow_pickle=True).tolist()

    image = data['topview_image']
    room_annot = data['room_instances_annot']
    lines = data['line_coords']
    room_map = data['room_map']
    bg_idx = data['bg_idx']
    normal_image = data['topview_mean_normal']
    point_dict = data['point_dict']
    lines = data['lines']

    annot_image = np.zeros([image.shape[0], image.shape[1]], dtype=np.uint8)
    all_gt_masks = list()
    for i, room in enumerate(room_annot):
        mask = room['mask_large'].astype('uint8')
        all_gt_masks.append(mask)
        annot_image += mask * 200

    cv2.imwrite('./topview_image.png', image)
    cv2.imwrite('./topview_mean_normal.png', normal_image)
    cv2.imwrite('./sep_room_ano.png', annot_image)
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    pass


# ------------------------------------------------------------------------------


if __name__ == '__main__':
  scene = 'UNd46FX2vC-YEOIF-Wx1aZ'
  load_annot_scene(scene)
  #visualize_annot_scene(scene)
  #visualize_annot_sep_room(scene)


  pass

