import os
import numpy as np
from plyfile import PlyData, PlyElement
import json
import math
import cv2
from floorplan_utils import calcLineDirection, MAX_NUM_CORNERS, NUM_WALL_CORNERS, getRoomLabelMap, getLabelRoomMap
from utils_floorsp import getDensity, drawDensityImage
from multiprocessing import Pool
from functools import partial

IMAGE_SIZE = 512

def gen_top_view( pool_num=0 ):
  CUR_DIR = os.path.dirname(os.path.realpath(__file__))
  ROOT_DIR = os.path.dirname(CUR_DIR)
  base_dir = os.path.join(ROOT_DIR, f'data/beike/processed_{IMAGE_SIZE}' )

  scene_file = os.path.join(base_dir, 'all.txt')
  scenes = np.loadtxt(scene_file, dtype = str).tolist()
  vertical_density = True
  num_points=50000*2

  topview_write_base = os.path.join(base_dir, 'TopView')
  if vertical_density:
    topview_write_base += '_VerD'
  else:
    topview_write_base += '_All'
  topview_write_base_imgs = os.path.join(base_dir, 'TopViewImgs')
  if not os.path.exists(topview_write_base):
      os.mkdir(topview_write_base)
  if not os.path.exists(topview_write_base_imgs):
      os.mkdir(topview_write_base_imgs)

  if pool_num == 0:
    for i, scene in enumerate(scenes):
      write_example(base_dir, topview_write_base, topview_write_base_imgs, num_points, vertical_density, scene)
  else:
    func = partial( write_example, base_dir, topview_write_base, topview_write_base_imgs, num_points, vertical_density)
    with Pool(pool_num) as pool:
      pool.map(func, scenes)
  n = len(scenes)
  print(f'\n\ngenerate topview finished: {n}\n')

def read_scene_pc(file_path):
    with open(file_path, 'rb') as f:
        plydata = PlyData.read(f)
        dtype = plydata['vertex'].data.dtype
    #print('dtype of file{}: {}'.format(file_path, dtype))

    points_data = np.array(plydata['vertex'].data.tolist())

    return points_data

def get_topview_mean_normal(points):
    topview_normal = np.zeros([IMAGE_SIZE, IMAGE_SIZE, 3])
    count = np.ones([IMAGE_SIZE, IMAGE_SIZE])
    full_2d_coordinates = np.clip(np.round(points[:, :2] * \
                  IMAGE_SIZE).astype(np.int32), 0, IMAGE_SIZE - 1)
    for point, coord in zip(points, full_2d_coordinates):
        topview_normal[coord[1], coord[0], :] += point[6:9]
        count[coord[1], coord[0]] += 1
    count = np.stack([count, count, count], 2)
    topview_normal /= count
    return topview_normal

def get_topview_data(points):
    """
    Add one more channel for counting the density
    :param points: full point cloud data
    :return: top-view of full points
    """
    topview_image = np.zeros([IMAGE_SIZE, IMAGE_SIZE, points.shape[-1] + 1])
    full_2d_coordinates = np.clip(np.round(points[:, :2] * IMAGE_SIZE).astype(np.int32), 0, IMAGE_SIZE - 1)
    for point, coord in zip(points, full_2d_coordinates):
        topview_image[coord[1], coord[0], :-1] += point
        topview_image[coord[1], coord[0], -1] += 1
    return topview_image

def _draw_corner_with_scaling(img, corner, min_x, width, min_y, height, color=(0, 0, 255), text=None):
    img_x = int(math.floor((corner[0] - min_x) * 1.0 / width * IMAGE_SIZE))
    img_y = int(math.floor((corner[1] - min_y) * 1.0 / height * IMAGE_SIZE))
    cv2.circle(img, (img_x, img_y), 2, color, -1)
    if text is not None:
        cv2.putText(img, text, (img_x, img_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, 1)
    return img_x, img_y

def parse_annot(file_path, mins, max_range, draw_img=False, img=None):
    with open(file_path, 'r') as f:
        data = json.load(f)

    if draw_img:
        assert img is not None

    points = data['points']
    lines = data['lines']
    line_items = data['lineItems']
    areas = data['areas']

    point_dict = dict()
    for point in points:
        point_dict[point['id']] = point

    # img = np.zeros([self.im_size, self.im_size, 3], dtype=np.uint8)

    min_x = mins[0][0]
    min_y = mins[0][1]
    width = height = max_range

    adjacency_dict = dict()
    for point in points:
        adjacency_dict[point['id']] = set()

    for line in lines:
        pt1, pt2 = line['points']
        adjacency_dict[pt1].add(pt2)
        adjacency_dict[pt2].add(pt1)

    # draw all corners
    point_img_coord_set = set()
    for point in points:
        img_x, img_y = _draw_corner_with_scaling(img, (point['x'], point['y']), min_x, width, min_y, height)
        point_dict[point['id']]['img_x'] = img_x
        point_dict[point['id']]['img_y'] = img_y
        point_img_coord_set.add((img_x, img_y))

    # draw all line segments
    for line in lines:
        assert len(line['points']) == 2
        point_id_1, point_id_2 = line['points']
        start_pt = (point_dict[point_id_1]['img_x'], point_dict[point_id_1]['img_y'])
        end_pt = (point_dict[point_id_2]['img_x'], point_dict[point_id_2]['img_y'])
        cv2.line(img, start_pt, end_pt, (255, 0, 0))

    # draw all line with labels, such as doors, windows
    for line_item in line_items:
        start_pt = (line_item['startPointAt']['x'], line_item['startPointAt']['y'])
        end_pt = (line_item['endPointAt']['x'], line_item['endPointAt']['y'])
        line_direction = calcLineDirection((start_pt, end_pt))  # 0 means horizontal and 1 means vertical
        img_start_pt = _draw_corner_with_scaling(img, start_pt, min_x, width, min_y, height, color=(0, 255, 0))
        img_end_pt = _draw_corner_with_scaling(img, end_pt, min_x, width, min_y, height, color=(0, 255, 0))

        # manually prevent opening corners to be exactly overlapping with other wall corners
        if img_start_pt in point_img_coord_set:
            if line_direction == 0:
                img_start_pt = (img_start_pt[0] + int(np.sign(img_end_pt[0] - img_start_pt[0])), img_start_pt[1])
            else:
                img_start_pt = (img_start_pt[0], img_start_pt[1] + int(np.sign(img_end_pt[1] - img_start_pt[1])))
        if img_end_pt in point_img_coord_set:
            if line_direction == 0:
                img_end_pt = (img_end_pt[0] + int(np.sign(img_start_pt[0] - img_end_pt[0])), img_end_pt[1])
            else:
                img_end_pt = (img_end_pt[0], img_end_pt[1] + int(np.sign(img_start_pt[1] - img_end_pt[1])))

        line_item['img_start_pt'] = img_start_pt
        line_item['img_end_pt'] = img_end_pt
        cv2.line(img, img_start_pt, img_end_pt, (0, 255, 255))
        cv2.putText(img, line_item['is'], (img_start_pt[0], img_start_pt[1] - 5), cv2.FONT_HERSHEY_PLAIN, 0.3,
                    (255, 255, 255))

    data['point_dict'] = point_dict

    if draw_img:
        return data, img
    else:
        return data

def write_example(base_dir, topview_write_base, topview_write_base_imgs, num_points,  vertical_density, scene):
    ply_path = os.path.join(base_dir, 'ply', scene+'.ply')
    annot_path = os.path.join(base_dir, 'json', scene+'.json')
    if not os.path.exists(ply_path) and os.path.exists(annot_path):
      return
    output_path = os.path.join(topview_write_base, scene + '.npy')
    if os.path.exists(output_path):
      return
    print(f'writing: {scene}')

    points = read_scene_pc(ply_path)
    n0 = points.shape[0]
    n0k = int(n0/1000)
    n1k = int(num_points/1000)
    print(f'\t{n0k}K -> {n1k}K')

    xyz = points[:, :3]

    mins = xyz.min(0, keepdims=True)
    maxs = xyz.max(0, keepdims=True)

    max_range = (maxs - mins)[:, :2].max()
    padding = max_range * 0.05

    mins = (maxs + mins) / 2 - max_range / 2
    mins -= padding
    max_range += padding * 2

    xyz = (xyz - mins) / max_range  # re-scale coords into [0.0, 1.0]

    new_points = np.concatenate([xyz, points[:, 3:9]], axis=1)
    points = new_points

    # down-sampling points to get a subset with size 50,000
    topview_mean_normal = get_topview_mean_normal(points).astype(np.float32)
    if not vertical_density:
        indices = np.arange(points.shape[0])
        if num_points < points.shape[0]:
          points = points[ np.random.choice(points.shape[0], num_points, replace=False) ]
        topview_points = get_topview_data(points)
    else:
        normal_z = points[:, 8] # vertical = z

        horizontal_points = np.array([point for p_i, point in enumerate(points) if abs(normal_z[p_i]) >= 0.5])
        vertical_points = np.array([point for p_i, point in enumerate(points) if abs(normal_z[p_i]) < 0.5])
        # note: only use vertical points for the full density map. Otherwise all fine details are ruined
        topview_points = get_topview_data(vertical_points)
        #show_points(vertical_points)
        #show_points(horizontal_points)

        point_subsets = [horizontal_points, vertical_points]
        subset_ratio = [0.3, 0.7]
        sampled_points = list()
        for point_subset, ratio in zip(point_subsets, subset_ratio):
            sampled_indices = np.arange(point_subset.shape[0])
            np.random.shuffle(sampled_indices)
            sampled_points.append(point_subset[sampled_indices[:int(num_points * ratio)]])
        points = np.concatenate(sampled_points, axis=0)

    annot = parse_annot(annot_path, mins, max_range)
    scene_id, _ = os.path.splitext(os.path.basename(annot_path))

    points[:, 3:6] = points[:, 3:6] / 255 - 0.5  # normalize color


    topview_image = drawDensityImage(topview_points[:, :, -1], nChannels=1).astype(np.float32)
    room_data = {
        'scene_id': scene_id,
        'topview_image': topview_image,
        'topview_mean_normal': topview_mean_normal,}
    #    'room_instances_annot': room_instances,
    #    'line_coords': line_coords,
    #    'room_map': rooms_large,
    #    'bg_idx': bg_idx,
    #    'point_dict': point_dict,
    #    'lines': lines,
    #}


    with open(output_path, 'wb') as f:
        np.save(f, room_data)

    if 0:
      output_path_d = os.path.join(topview_write_base_imgs, scene + '-density.png')
      cv2.imwrite(output_path_d, topview_image)
      output_path_n = os.path.join(topview_write_base_imgs, scene + '-norm.png')
      cv2.imwrite(output_path_n, np.abs(topview_mean_normal)*255)

    #num_non_manhattan_room = 0
    #for room_instance in room_instances:
    #    manhattan = self.check_manhattan(room_instance)
    #    if not manhattan:
    #        num_non_manhattan_room += 1

    stats = {
        'id': scene_id,
        'num_points': points.shape[0],
    }

    return True, stats

if __name__ == '__main__':
    gen_top_view( pool_num = 3 )


