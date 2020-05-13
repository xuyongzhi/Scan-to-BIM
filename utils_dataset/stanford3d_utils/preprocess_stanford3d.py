import glob
import numpy as np
import os

from tqdm import tqdm

from obj_geo_utils.geometry_utils import limit_period_np, vertical_dis_1point_lines, points_in_lines
from obj_geo_utils.obj_utils import OBJ_REPS_PARSE, GraphUtils
from utils_dataset.lib.pc_utils import save_point_cloud
from tools.debug_utils import _show_3d_points_bboxes_ls, _show_3d_points_lines_ls, _show_lines_ls_points_ls
from tools import debug_utils
from tools.visual_utils import _show_objs_ls_points_ls, _show_3d_points_objs_ls, _show_3d_as_img

import MinkowskiEngine as ME

from collections import defaultdict
import open3d as o3d

STANFORD_3D_IN_PATH = '/cvgl/group/Stanford3dDataset_v1.2/'
STANFORD_3D_IN_PATH = '/DS/Stanford3D/Stanford3dDataset_v1.2_Aligned_Version/'
STANFORD_3D_OUT_PATH = '/DS/Stanford3D/aligned_processed_instance'

STANFORD_3D_TO_SEGCLOUD_LABEL = {
    4: 0,
    8: 1,
    12: 2,
    1: 3,
    6: 4,
    13: 5,
    7: 6,
    5: 7,
    11: 8,
    3: 9,
    9: 10,
    2: 11,
    0: 12,
}
THICK_MAP = {'door': 0.2, 'window': 0.25}

DEBUG=0

Instance_Bad_Scenes = ['Area_1/hallway_7', 'Area_6/office_9', 'Area_4/lobby_1']

def get_manual_merge_pairs(scene_name):
  if scene_name == 'Area_3/office_4':
    return   [ (0,1), (0,6), (1,3) ]
  else:
    return None

class Stanford3DDatasetConverter:

  CLASSES = [
      'clutter', 'beam', 'board', 'bookcase', 'ceiling', 'chair', 'column', 'door', 'floor', 'sofa',
      'stairs', 'table', 'wall', 'window'
  ]
  num_cat = len(CLASSES)
  Cat2Id = {}
  for i in range(num_cat):
    Cat2Id[CLASSES[i]] = i
  wall_id = Cat2Id['wall']
  TRAIN_TEXT = 'train'
  VAL_TEXT = 'val'
  TEST_TEXT = 'test'

  @classmethod
  def read_txt(cls, txtfile):
    # Read txt file and parse its content.
    with open(txtfile) as f:
      pointcloud = [l.split() for l in f]
    # Load point cloud to named numpy array.
    invalid_ids = [i for i in range(len(pointcloud)) if len(pointcloud[i]) != 6]
    if len(invalid_ids)>0:
      print('\n\n\n Found invalid points with len!=6')
      print(invalid_ids)
      print('\n\n\n')
    pointcloud = [p for p in pointcloud if len(p) == 6]
    pointcloud = np.array(pointcloud).astype(np.float32)
    assert pointcloud.shape[1] == 6
    xyz = pointcloud[:, :3].astype(np.float32)
    rgb = pointcloud[:, 3:].astype(np.uint8)
    return xyz, rgb

  @classmethod
  def convert_to_ply(cls, root_path, out_path):
    """Convert Stanford3DDataset to PLY format that is compatible with
    Synthia dataset. Assumes file structure as given by the dataset.
    Outputs the processed PLY files to `STANFORD_3D_OUT_PATH`.
    """

    txtfiles = glob.glob(os.path.join(root_path, '*/*/*.txt'))
    for txtfile in tqdm(txtfiles):
      file_sp = os.path.normpath(txtfile).split(os.path.sep)
      target_path = os.path.join(out_path, file_sp[-3])
      out_file = os.path.join(target_path, file_sp[-2] + '.ply')

      if os.path.exists(out_file):
        print(out_file, ' exists')
        continue

      annotation, _ = os.path.split(txtfile)
      subclouds = glob.glob(os.path.join(annotation, 'Annotations/*.txt'))
      coords, feats, labels = [], [], []
      all_instance_ids = defaultdict(list)
      for inst, subcloud in enumerate(subclouds):
        # Read ply file and parse its rgb values.
        xyz, rgb = cls.read_txt(subcloud)
        _, annotation_subfile = os.path.split(subcloud)
        clsidx = cls.CLASSES.index(annotation_subfile.split('_')[0])
        instance_id = int(os.path.splitext( os.path.basename(subcloud) )[0].split('_')[1])

        coords.append(xyz)
        feats.append(rgb)
        category = np.ones((len(xyz), 1), dtype=np.int32) * clsidx
        instance_ids = np.ones((len(xyz), 1), dtype=np.int32) * instance_id
        label = np.concatenate([category, instance_ids], 1)
        labels.append(label)

        all_instance_ids[clsidx].append(instance_id)
        pass

      for clsidx in all_instance_ids:
        if not len(all_instance_ids[clsidx]) == max(all_instance_ids[clsidx]):
          pass

      if len(coords) == 0:
        print(txtfile, ' has 0 files.')
      else:
        # Concat
        coords = np.concatenate(coords, 0)
        feats = np.concatenate(feats, 0)
        labels = np.concatenate(labels, 0)
        inds, collabels = ME.utils.sparse_quantize(
            coords,
            feats,
            labels[:,0],
            return_index=True,
            ignore_label=255,
            quantization_size=0.01  # 1cm
        )
        pointcloud = np.concatenate((coords[inds], feats[inds], labels[inds]), axis=1)

        # Write ply file.
        if not os.path.exists(target_path):
          os.makedirs(target_path)
        save_point_cloud(pointcloud, out_file, with_label=True, verbose=False)


def generate_splits(stanford_out_path):
  """Takes preprocessed out path and generate txt files"""
  split_path = './splits/stanford'
  if not os.path.exists(split_path):
    os.makedirs(split_path)
  for i in range(1, 7):
    curr_path = os.path.join(stanford_out_path, f'Area_{i}')
    files = glob.glob(os.path.join(curr_path, '*.ply'))
    files = [os.path.relpath(full_path, stanford_out_path) for full_path in files]
    out_txt = os.path.join(split_path, f'area{i}.txt')
    with open(out_txt, 'w') as f:
      f.write('\n'.join(files))



def aligned_points_to_bbox(points, out_np=False):
  assert points.ndim == 2
  assert points.shape[1] == 3
  points_ = o3d.utility.Vector3dVector(points)
  bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(points_)
  #rMatrix = bbox.R
  #print(rMatrix)

  #rotvec = R_to_Vec(rMatrix)
  #rMatrix_Check_v = o3d.geometry.get_rotation_matrix_from_axis_angle(rotvec)
  #errM = np.matmul(rMatrix, rMatrix_Check_v.transpose())
  #rotvec_c = R_to_Vec(rMatrix_Check_v)
  #if not  np.max(np.abs(( rotvec - rotvec_c ))) < 1e-6:
  #  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  #  pass

  #euler = R_to_Euler(rMatrix)
  #rMatrix_Check_e = o3d.geometry.get_rotation_matrix_from_zyx(euler)

  if 1:
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=bbox.get_center())
    pcd = o3d.geometry.PointCloud()
    pcd.points = points_
    o3d.visualization.draw_geometries([bbox, pcd, mesh_frame])

  if not out_np:
    return bbox
  else:
    min_bound = bbox.min_bound
    max_bound = bbox.max_bound
    bbox_np = np.concatenate([min_bound, max_bound], axis=0)
  return bbox_np

def points_to_oriented_bbox(points, bboxes_wall0, cat_name,  voxel_size=0.005):
  '''
  points: [n,3]
  boxes3d: [n,8] XYXYSin2WZ0Z1
  '''
  import cv2
  assert points.ndim == 2
  assert points.shape[1] == 3
  if cat_name!='wall':
    assert bboxes_wall0 is not None

  #if bboxes_wall0 is not None:
  #  _show_3d_points_objs_ls([points], objs_ls=[bboxes_wall0], obj_rep='XYXYSin2WZ0Z1')
  points = points.copy()

  xyz_min = points.min(0)
  xyz_max = points.max(0)
  if bboxes_wall0 is not None:
    xy_min_w = bboxes_wall0[:,:4].reshape(-1,2).min(0)
    z_mi_w = bboxes_wall0[:,-2].min()
    xyz_min[0] = min(xyz_min[0], xy_min_w[0])
    xyz_min[1] = min(xyz_min[1], xy_min_w[1])
  points = points - xyz_min

  point_inds = points / voxel_size
  point_inds = np.round(point_inds).astype(np.int)[:,:2]

  if bboxes_wall0 is  not None:
    bboxes_wall = bboxes_wall0.copy()
    bboxes_wall[:,:2] -= xyz_min[:2]
    bboxes_wall[:,2:4] -= xyz_min[:2]
    walls_inds = bboxes_wall / voxel_size
    walls_inds = np.round(walls_inds).astype(np.float32)
    walls_inds[:,:4]
    walls_inds[:,4] = bboxes_wall0[:,4]
    #_show_objs_ls_points_ls( (512,512), [walls_inds], obj_rep='XYXYSin2WZ0Z1' )
    pass

  if cat_name in ['wall', 'window']:
    rotate_box = True
  else:
    rotate_box = False

  if cat_name == 'wall':
    # ( (cx,cy), (sx,sy), angle )
    box2d = cv2.minAreaRect(point_inds)
  elif cat_name in ['door', 'window']:
    box2d = aligned_box(point_inds)
    size_a = min(box2d[1]) * voxel_size
    if size_a > 0.3:
      print(f'size_align: {size_a}')
      box2d = points_to_box_align_with_wall(point_inds, walls_inds, cat_name, voxel_size)
      #_show_3d_points_objs_ls([points+xyz_min], objs_ls=[bboxes_wall0], obj_rep='XYXYSin2WZ0Z1')
      if box2d is None:
        return None
  else:
    box2d = aligned_box(point_inds)
  box2d = np.array( box2d[0]+box2d[1]+(box2d[2],) )[None, :]
  # The original angle from cv2.minAreaRect denotes the rotation from ref-x
  # to body-x. It is it positive for clock-wise.
  # make x the long dim
  if box2d[0,2] < box2d[0,3]:
    box2d[:,[2,3]] = box2d[:,[3,2]]
    box2d[:,-1] = 90+box2d[:,-1]
  box2d[:,-1] *= np.pi / 180
  box2d[:,-1] = limit_period_np(box2d[:,-1], 0.5, np.pi) # limit_period

  box2d_st = OBJ_REPS_PARSE.encode_obj(box2d, obj_rep_in = 'XYLgWsA',
                     obj_rep_out = 'XYXYSin2W')

  box3d = np.concatenate([box2d_st, xyz_min[None,2:], xyz_max[None,2:]], axis=1)
  box3d[:, [0,1,2,3, 5,]] *= voxel_size
  box3d[:, [0,1]] += xyz_min[None, [0,1]]
  box3d[:, [2,3]] += xyz_min[None, [0,1]]

  if 0 and cat_name in ['door']:
    img_size = point_inds.max(0)[[1,0]]+1 + 100
    img = np.zeros(img_size, dtype=np.uint8)

    if bboxes_wall0 is not None:
      _show_objs_ls_points_ls(img, [box2d_st, walls_inds[:,:6]], obj_rep='XYXYSin2W',
                            points_ls=[point_inds], obj_colors=['green', 'blue'])
    else:
      _show_objs_ls_points_ls(img, [box2d_st], obj_rep='XYXYSin2W',
                            points_ls=[point_inds], obj_colors=['green', 'blue'])
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    pass
  return box3d

def aligned_box(point_inds):
    min_xy = point_inds.min(0)
    max_xy = point_inds.max(0)
    cx, cy = (min_xy + max_xy) / 2
    sx, sy = max_xy - min_xy
    box2d = ( (cx,cy), (sx,sy), 0 )
    return box2d

def points_to_box_align_with_wall(points, walls, cat_name, voxel_size):
  for i in range(4):
    box2d, wall_id = points_to_box_align_with_wall_1ite(points, walls, cat_name, voxel_size)
    if box2d is None:
      return None
    length = max(box2d[1]) * voxel_size
    thick = min(box2d[1]) * voxel_size

    print(f'{cat_name} length={length}, thick={thick}')
    if length > 0.5:
      return box2d
    mask = np.arange(walls.shape[0]) != wall_id
    walls = walls[mask]
    pass
  return None

def points_to_box_align_with_wall_1ite(points, walls, cat_name, voxel_size, max_diss_meter=1):
  from obj_geo_utils.line_operations import transfer_lines_points
  thick = THICK_MAP[cat_name] / voxel_size
  max_diss = max_diss_meter / voxel_size
  n = walls.shape[0]

  meanp = points.mean(0)
  wall_lines = OBJ_REPS_PARSE.encode_obj( walls, 'XYXYSin2WZ0Z1', 'RoLine2D_2p' ).reshape(-1,2,2)
  walls_ = OBJ_REPS_PARSE.encode_obj(walls, 'XYXYSin2WZ0Z1', 'XYLgWsA')
  diss = vertical_dis_1point_lines(meanp, wall_lines, no_extend=False)
  diss *= voxel_size
  if np.min(diss) > max_diss:
    return None, None

  inside_rates = np.zeros([n])
  for i in range(n):
    if diss[i] > 0.8:
      continue
    thres0 = 0.05 / voxel_size
    wall_i_aug = walls_[i:i+1].copy()
    thres = wall_i_aug[:,3] * 0.4
    thres = np.clip(thres, a_min=thres0, a_max=None)
    wall_i_aug[:,2] += 0.5 / voxel_size
    the_line2d = OBJ_REPS_PARSE.encode_obj( wall_i_aug, 'XYLgWsA', 'RoLine2D_2p' ).reshape(-1,2,2)
    num_inside = points_in_lines( points, the_line2d, thres ).sum()
    inside_rate = 1.0 * num_inside / points.shape[0]
    inside_rates[i] = inside_rate
  if np.max(inside_rates) == 0:
    _show_objs_ls_points_ls( (1324,1324), [walls_], obj_rep='XYLgWsA', points_ls = [points])
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    return None, None
  inside_rates_nm = inside_rates / sum(inside_rates)

  fused_diss = diss * (1-inside_rates_nm)
  wall_id = fused_diss.argmin()
  the_wall = walls_[wall_id][None,:]
  #_show_objs_ls_points_ls( (1024,1024), [the_wall], obj_rep='XYLgWsA', obj_colors=['green','blue'], points_ls = [points])

  #for i in range(5):
  #  wall_id = diss.argmin()
  #  min_diss = diss[wall_id]
  #  if min_diss > max_diss:
  #    return None, None
  #  print(f'\n{cat_name} and wall, min diss: {min_diss}')
  #  the_wall = walls_[wall_id][None,:]
  #  #_show_objs_ls_points_ls( (512,512), [walls_, the_wall], obj_rep='XYLgWsA', obj_colors=['green','blue'], points_ls = [points])
  #  #_show_objs_ls_points_ls( (512,512), [walls_], obj_rep='XYLgWsA' , points_ls = [points])

  #  thres = 0.2 / voxel_size
  #  the_wall_aug = the_wall.copy()
  #  the_wall_aug[:,2:4] += 0.2 / voxel_size
  #  the_line2d = OBJ_REPS_PARSE.encode_obj( the_wall_aug, 'XYLgWsA', 'RoLine2D_2p' ).reshape(-1,2,2)
  #  num_inside = points_in_lines( points, the_line2d, thres ).sum()
  #  inside_rate = 1.0 * num_inside / points.shape[0]
  #  print(f'inside_rate: {inside_rate}')
  #  if inside_rate > 0.2:
  #    break
  #  else:
  #    diss[wall_id] += 10000
  #if inside_rate <0.2:
  #  return None, None

  angle = -the_wall[0,4]
  center = (the_wall[0,0], the_wall[0,1])
  walls_r, points_r = transfer_lines_points( walls_, 'XYLgWsA', points, center, angle, (0,0) )
  the_wall_r = walls_r[wall_id]
  #_show_objs_ls_points_ls( (512,512), [walls_r], obj_rep='XYLgWsA', obj_colors=['green','blue'], points_ls = [points_r])

  the_wall_xmin = the_wall_r[0] - the_wall_r[2]/2
  the_wall_xmax = the_wall_r[0] + the_wall_r[2]/2

  min_xy = points_r.min(0)
  max_xy = points_r.max(0)

  #min_xy[0] = max(min_xy[0], the_wall_xmin)
  #max_xy[0] = min(max_xy[0], the_wall_xmax)

  xc, yc = (min_xy + max_xy) / 2
  xs, ys = max_xy - min_xy
  ys_meter = ys * voxel_size
  box_r = the_wall_r.copy()
  box_r[0]  = xc
  box_r[2] = xs

  wall_yc = box_r[1]
  #thick = box_r[3] * 2
  y_max = max_xy[1]
  y_min = min_xy[1]
  if wall_yc > y_max - thick/2:
    y_min = y_max - thick
    yc = (y_min + y_max) / 2
  elif wall_yc < y_min + thick/2:
    y_max = y_min + thick
    yc = (y_min + y_max) / 2
  else:
    yc = wall_yc

  box_r[1] = yc
  box_r[3] = thick

  box_r = box_r[None,:]
  #_show_objs_ls_points_ls( (512,512), [walls_r, box_r], obj_rep='XYLgWsA', obj_colors=['green','blue'], points_ls = [points_r])

  box_out, points_out = transfer_lines_points( box_r, 'XYLgWsA', points_r, center, -angle, (0,0) )


  wall_view = walls_[wall_id:wall_id+1]
  wall_view = walls_

  #_show_objs_ls_points_ls( (1024,1024), [wall_view, box_out], obj_rep='XYLgWsA', obj_colors=['green','blue'], points_ls = [points_out, points], point_colors=['yellow','red'])

  cx, cy, l, w, a = box_out[0]
  a = a*180/np.pi
  box_2d = ( (cx,cy), (l,w), a)
  return box_2d, wall_id

def unused_points_to_bbox(points, out_np=False):
  from beike_data_utils.geometric_utils import R_to_Vec, R_to_Euler
  assert points.ndim == 2
  assert points.shape[1] == 3
  zmax = points[:,2].max()
  zmin = points[:,2].min()
  zmean = (zmax+zmin)/2
  bev_points = points.copy()
  bevz = bev_points[:,2].copy()
  bev_points[:,2] = zmean
  bev_points[:,2][bevz>zmean+0.5] = zmax
  bev_points[:,2][bevz<=zmean-0.5] = zmin
  pvec = o3d.utility.Vector3dVector(bev_points)
  bbox = o3d.geometry.OrientedBoundingBox.create_from_points(pvec)
  rMatrix = bbox.R
  print(rMatrix)

  rotvec = R_to_Vec(rMatrix)
  rMatrix_Check_v = o3d.geometry.get_rotation_matrix_from_axis_angle(rotvec)
  errM = np.matmul(rMatrix, rMatrix_Check_v.transpose())
  rotvec_c = R_to_Vec(rMatrix_Check_v)
  if not  np.max(np.abs(( rotvec - rotvec_c ))) < 1e-6:
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    pass

  euler = R_to_Euler(rMatrix)
  rMatrix_Check_e = o3d.geometry.get_rotation_matrix_from_zyx(euler)

  if 1:
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=bbox.get_center())
    pcd = o3d.geometry.PointCloud()
    pcd.points = pvec
    o3d.visualization.draw_geometries([bbox, pcd, mesh_frame])

  if not out_np:
    return bbox
  else:
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    min_bound = bbox.min_bound
    max_bound = bbox.max_bound
    bbox_np = np.concatenate([min_bound, max_bound], axis=0)
  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  return bbox_np


def sampling_points(max_num, points, *args):
    max_num = int(max_num)
    n0 = points.shape[0]
    if n0 > max_num:
      inds = np.random.choice(n0, max_num, replace=False)
      points = points[inds]
      new_args = []
      for i in range(len(args)):
        new_args.append(  args[i][inds] )
      new_args = tuple(new_args)
    else:
      new_args = args
    return ( points,) + new_args

def get_scene_name(file_path):
  s0, s1 = file_path.split('/')[-2:]
  s = s0 + '/' + s1.split('.')[0]
  return s

def gen_bboxes(max_num_points=1e5):
  from plyfile import PlyData
  ply_files = glob.glob(STANFORD_3D_OUT_PATH + '/*/*.ply')
  #ply_files = [os.path.join(STANFORD_3D_OUT_PATH,  'Area_1/hallway_8.ply' )]
  UNALIGNED = ['Area_2/auditorium_1', 'Area_3/office_8',
               'Area_4/hallway_14', 'Area_3/office_7']
  DOOR_HAD = ['Area_1/hallway_4','Area_2/auditorium_1','Area_1/hallway_8']
  ROTAE_WINDOW=['Area_1/office_11']
  IntroSample = ['Area_3/office_4']
  scenes = IntroSample
  ply_files = [os.path.join(STANFORD_3D_OUT_PATH,  f'{s}.ply' ) for s in scenes]

  # The first 72 is checked
  for l, plyf in enumerate( ply_files ):
      scene_name_l = get_scene_name(plyf)
      bbox_file = plyf.replace('.ply', '.npy').replace('Area_', 'Boxes_Area_')
      topview_file = plyf.replace('.ply', '.png').replace('Area_', 'Boxes_Area_')
      bbox_dir = os.path.dirname(bbox_file)
      if not os.path.exists(bbox_dir):
        os.makedirs(bbox_dir)
      print(f'\n\nStart processing \t{bbox_file} \n\t\t{l}\n')
      if os.path.exists(bbox_file):
        pass
        continue

      plydata = PlyData.read(plyf)
      data = plydata.elements[0].data
      coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
      feats = np.array([data['red'], data['green'], data['blue']], dtype=np.float32).T
      categories = np.array(data['label'], dtype=np.int32)
      instances = np.array(data['instance'], dtype=np.int32)

      coords, feats, categories, instances = sampling_points(max_num_points, coords, feats, categories, instances)

      #show_pcd(coords)

      cat_min = categories.min()
      cat_max = categories.max()
      cat_ids = [Stanford3DDatasetConverter.wall_id, ] + [i for i in range(cat_max+1) if i != Stanford3DDatasetConverter.wall_id]

      bboxes_wall = None
      abandon_num = 0

      bboxes = defaultdict(list)
      for cat in cat_ids:
        mask_cat = categories == cat
        cat_name = Stanford3DDatasetConverter.CLASSES[cat]
        num_cat = mask_cat.sum()
        print(f'\n{cat_name} has {num_cat} points')
        if num_cat == 0:
          continue
        coords_cat, instances_cat = coords[mask_cat], instances[mask_cat]

        ins_min = instances_cat.min()
        ins_max = instances_cat.max()
        for ins in range(ins_min, ins_max+1):
          mask_ins = instances_cat == ins
          num_ins = mask_ins.sum()
          #print(f'\t{ins}th instance has {num_ins} points')
          if num_ins == 0:
            continue
          coords_cat_ins = coords_cat[mask_ins]
          if cat_name != 'clutter':
            #show_pcd(coords_cat_ins)
            bbox = points_to_oriented_bbox(coords_cat_ins, bboxes_wall, cat_name)
            if bbox is not None:
              bboxes[cat_name].append(bbox)
            else:
              abandon_num += 1
              print(f'\n\nAbandon one {cat_name}\n\n')
        if cat_name == 'wall':
          bboxes_wall = np.concatenate(bboxes['wall'],0)
        pass
      for cat in bboxes:
        bboxes[cat] = np.concatenate(bboxes[cat], 0)


      # cal pcl scope
      min_pcl = coords.min(axis=0)
      min_pcl = np.floor(min_pcl*10)/10
      max_pcl = coords.max(axis=0)
      max_pcl = np.ceil(max_pcl*10)/10
      mean_pcl = (min_pcl + max_pcl) / 2
      pcl_scope = np.concatenate([min_pcl, max_pcl], axis=0)
      room = np.concatenate([ mean_pcl, max_pcl-min_pcl, np.array([0]) ], axis=0)

      bboxes['room'] = OBJ_REPS_PARSE.encode_obj( room[None,:], 'XYZLgWsHA', 'XYXYSin2WZ0Z1' )

      bboxes['wall'] = optimize_walls(bboxes['wall'], get_manual_merge_pairs(scene_name_l))

      np.save(bbox_file, bboxes)
      print(f'\n save {bbox_file}')

      colors = instances.astype(np.int32)
      colors = feats

      if 0:
        _show_3d_points_objs_ls([coords], [colors], [bboxes['wall']],  obj_rep='XYXYSin2WZ0Z1')
      if 1:
        print(f'abandon_num: {abandon_num}')
        all_bboxes = []
        all_cats = []
        view_cats = ['wall', 'beam', 'window', 'column', 'door']
        #view_cats = ['wall']
        for cat in view_cats:
          if cat not in bboxes:
            continue
          all_bboxes.append( bboxes[cat] )
          all_cats += [cat] * bboxes[cat].shape[0]
        all_bboxes = np.concatenate(all_bboxes, 0)
        all_cats = np.array(all_cats)

        #_show_3d_points_objs_ls([coords], [colors], [all_bboxes],  obj_rep='XYXYSin2WZ0Z1', obj_colors='random')

        scope = all_bboxes[:,:4].reshape(-1,2).max(0) - all_bboxes[:,:4].reshape(-1,2).min(0)
        voxel_size =  max(scope) / 1000
        all_bboxes_2d = all_bboxes[:,:6].copy() / voxel_size
        org = all_bboxes_2d[:,:4].reshape(-1,2).min(0)[None,:] - 50
        all_bboxes_2d[:,:2] -=  org
        all_bboxes_2d[:,2:4] -=  org
        w, h = all_bboxes_2d[:,:4].reshape(-1,2).max(0).astype(np.int)+100
        _show_objs_ls_points_ls( (h,w), [all_bboxes_2d], obj_rep='XYXYSin2W',
                                obj_scores_ls=[all_cats], out_file=topview_file,
                                only_save=1)
        pass
      if 0:
        view_cats = ['column', 'beam']
        view_cats = ['door',]
        for cat in view_cats:
          if cat not in bboxes:
            continue
          print(cat)
          _show_3d_points_objs_ls([coords], [colors], [bboxes[cat]],  obj_rep='XYXYSin2WZ0Z1')
          pass
  pass

def optimize_walls(walls_3d_line, manual_merge_pairs=None):
    '''
    XYXYSin2WZ0Z1
    '''
    wall_bottom = walls_3d_line[:,-2].min()
    walls_3d_line[:,-2] = wall_bottom

    bottom_corners = OBJ_REPS_PARSE.encode_obj(walls_3d_line, 'XYXYSin2WZ0Z1', 'Bottom_Corners').reshape(-1,3)
    #_show_3d_points_objs_ls([bottom_corners], objs_ls = [walls_3d_line],  obj_rep='XYXYSin2WZ0Z1')

    walls_2d_line = walls_3d_line[:,:5]

    #_show_3d_points_objs_ls(objs_ls=[walls_2d_line], obj_rep='RoLine2D_UpRight_xyxy_sin2a')
    walls_2d_line_new, _, _, valid_mask = GraphUtils.optimize_graph(walls_2d_line, obj_rep='XYXYSin2', opt_graph_cor_dis_thr=0.15, min_out_length=0.22)

    if manual_merge_pairs is not None:
      walls_2d_line_new = GraphUtils.opti_wall_manually(walls_2d_line_new, 'XYXYSin2', manual_merge_pairs)
    #_show_3d_points_objs_ls(objs_ls=[walls_2d_line_new], obj_rep='RoLine2D_UpRight_xyxy_sin2a')

    try:
      walls_3d_line_new = np.concatenate( [walls_2d_line_new, walls_3d_line[valid_mask][:,5:8]], axis=1 )
    except:
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass
    bottom_corners_new = OBJ_REPS_PARSE.encode_obj(walls_3d_line_new, 'XYXYSin2WZ0Z1', 'Bottom_Corners').reshape(-1,3)
    #_show_3d_points_objs_ls([bottom_corners_new], objs_ls = [walls_3d_line_new],  obj_rep='XYXYSin2WZ0Z1')
    return walls_3d_line_new
    pass

def load_1_ply(filepath):
    from plyfile import PlyData
    plydata = PlyData.read(filepath)
    data = plydata.elements[0].data
    coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
    feats = np.array([data['red'], data['green'], data['blue']], dtype=np.float32).T
    labels = np.array(data['label'], dtype=np.int32)
    instance = np.array(data['instance'], dtype=np.int32)
    return coords, feats, labels, None


def get_surface_normal():
  from tools.debug_utils import _make_pcd
  path = '/home/z/Research/mmdetection/data/stanford'
  files = glob.glob(path+'/*/*.ply')
  for fn in files:
    points, _, _, _ = load_1_ply(fn)
    pcd = _make_pcd(points)
    pcd.estimate_normals(fast_normal_computation = True)
    norm = np.asarray(pcd.normals)
    #o3d.visualization.draw_geometries( [pcd] )
    norm_fn = fn.replace('.ply', '-norm.npy')
    np.save( norm_fn, norm )
    print('save:  ', norm_fn)

def get_scene_pcl_scopes():
  path = '/home/z/Research/mmdetection/data/stanford'
  files = glob.glob(path+'/*/*.ply')
  for fn in files:
    points, _, _, _ = load_1_ply(fn)
    min_points = points.min(0)
    max_points = points.max(0)
    scope_i = np.vstack([min_points, max_points])
    scope_fn = fn.replace('.ply', '-scope.txt')
    np.savetxt( scope_fn, scope_i)
    print('save:  ', scope_fn)
  pass



if __name__ == '__main__':
  #Stanford3DDatasetConverter.convert_to_ply(STANFORD_3D_IN_PATH, STANFORD_3D_OUT_PATH)
  #generate_splits(STANFORD_3D_OUT_PATH)
  gen_bboxes()
  #get_scene_pcl_scopes()
  #get_surface_normal()

