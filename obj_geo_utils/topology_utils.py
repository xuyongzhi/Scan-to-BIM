# xyz 16 June
import os
import numpy as np
import mmcv
from obj_geo_utils.geometry_utils import sin2theta_np, angle_with_x_np, \
      vec_from_angle_with_x_np, angle_with_x
import cv2
import torch
from obj_geo_utils.geometry_utils import limit_period_np, four_corners_to_box,\
  sort_four_corners, line_intersection_2d, mean_angles, angle_dif_by_period_np
from obj_geo_utils.obj_utils import OBJ_REPS_PARSE, GraphUtils
from obj_geo_utils.obj_utils import find_wall_wall_connection
from obj_geo_utils.line_operations import gen_corners_from_lines_np, get_lineIdsPerCor_from_corIdsPerLine
from collections import defaultdict
from tools.visual_utils import _show_objs_ls_points_ls, _draw_objs_ls_points_ls


def geometrically_opti_walls(det_lines, obj_rep, opt_graph_cor_dis_thr, min_out_length):
      obj_dim = OBJ_REPS_PARSE._obj_dims[obj_rep]
      scores_i = det_lines[:,-1]
      det_lines_merged, scores_merged, ids = \
        GraphUtils.optimize_wall_graph(det_lines[:,:obj_dim], scores_i, obj_rep_in=obj_rep,
          opt_graph_cor_dis_thr=opt_graph_cor_dis_thr, min_out_length=min_out_length )

      det_lines_merged = np.concatenate([det_lines_merged, scores_merged.reshape(-1,1)], axis=1)
      return det_lines_merged, ids

def optimize_walls_by_rooms_main(walls, rooms, obj_rep ):
  '''
  1. Find all the rooms already detected by edge loops.
  2. Find all the failing walls and failing rooms.
  3. Fix failing rooms.
  '''
  from obj_geo_utils.geometry_utils import points_to_oriented_bbox, get_rooms_from_edges
  from mmdet.core.bbox.geometry import dsiou_rotated_3d_bbox_np
  debug = 1
  #rooms = rooms[:,:7]
  #walls = walls[:,:7]

  num_rooms = rooms.shape[0]
  wall_ids_per_room, room_ids_per_wall, num_walls_inside_room, rooms_from_w = get_rooms_from_edges(walls[:,:7], obj_rep, gen_bbox=True)
  non_full_mask = (room_ids_per_wall<0).any(1)
  full_w_ids = np.where(non_full_mask==False)[0]
  non_full_w_ids = np.where(non_full_mask)[0]
  ious = dsiou_rotated_3d_bbox_np( rooms[:,:7], rooms_from_w, iou_w=1.0, size_rate_thres=None, ref='union', only_2d=True )
  succ_r_ids = ious.argmax(0)
  succ_ious = ious.max(0)
  fail_r_ids = [i for i in range(num_rooms) if i not in succ_r_ids]

  fail_rooms = rooms[fail_r_ids]
  fail_walls = walls[non_full_w_ids]
  if 0:
    _show_objs_ls_points_ls( (512,512), [rooms_from_w, rooms[succ_r_ids][:,:7], rooms[fail_r_ids][:,:7] ], obj_rep, obj_colors=['white', 'red', 'green'] )
    _show_objs_ls_points_ls( (512,512), [walls[:,:7], rooms[fail_r_ids][:,:7] ], obj_rep, obj_colors=['white', 'red'] )
    _show_objs_ls_points_ls( (512,512), [ fail_walls[:,:7], fail_rooms[:,:7] ], obj_rep, obj_colors=['white', 'red'] )
    #_show_objs_ls_points_ls( (512,512), [walls, walls[ wall_ids_per_room[0] ], rooms_from_w[0:1]], obj_rep, obj_colors=['white', 'green','red'] )

  room_ids_per_fail_wall = room_ids_per_wall[non_full_w_ids]
  num_rooms_per_fail_wall = (room_ids_per_fail_wall>=0).sum(1)
  walls_fail_fixed = fix_failed_rooms_walls(fail_walls, fail_rooms, num_rooms_per_fail_wall, obj_rep)
  walls_fixed = np.concatenate([walls[full_w_ids], walls_fail_fixed ], 0)

  if 0:
    _show_objs_ls_points_ls( (512,512), [rooms[:,:7], walls[:,:7], walls_fixed[:,:7] ],
                            obj_rep, obj_colors=['white', 'red', 'lime'], obj_thickness=[1, 5, 1] )
    _show_objs_ls_points_ls( (512,512), [rooms[:,:7],  walls_fixed[:,:7] ],
                            obj_rep, obj_colors=['white', 'lime'], obj_thickness=[1, 2] )


  #walls_final, ids_geo_opt = geometrically_opti_walls( walls_fixed, obj_rep, 2, 5 )
  if 0:
    _show_objs_ls_points_ls( (512,512), [walls_final[:,:7],  walls_fixed[:,:7] ],
                            obj_rep, obj_colors=['red', 'lime'], obj_thickness=[4, 1] )
    _show_objs_ls_points_ls( (512,512), [walls_final[:,:7] ], obj_rep, obj_colors=['random'], obj_thickness=4 )
  return walls_fixed

def fix_failed_rooms_walls(walls, rooms, num_rooms_per_fail_wall, obj_rep):
  '''
  1. For each wall, find the one or two matched rooms.
  2. Get the matched walls for each room.
  3. Fix room by room
  '''
  show_room_ids_per_wall = 0

  score_th = 0.4
  num_rooms = rooms.shape[0]
  num_walls = walls.shape[0]
  walls_aug = walls.copy()
  walls_aug[:,3] *= 0.8
  walls_aug[:,4] = 3
  rooms_aug = rooms.copy()
  rooms_aug[:,3:5] *= 1.2

  #_show_objs_ls_points_ls( (512,512), [walls[:,:-1], rooms[:,:-1]], obj_rep, obj_colors=['red', 'white'], obj_thickness=[3,1] )
  #_show_objs_ls_points_ls( (512,512), [walls[:,:-1], rooms_aug[:,:-1]], obj_rep, obj_colors=['red', 'white'], obj_thickness=[3,1] )

  w_in_r_scores = cal_edge_in_room_scores(walls_aug[:,:7], rooms_aug[:,:7], obj_rep)
  room_qua_scores = get_room_quality_scores(rooms, obj_rep)
  w_in_r_scores = w_in_r_scores *  room_qua_scores[None,:]
  sort_room_ids = (-w_in_r_scores).argsort(1)

  room_ids_per_w = []
  wall_ids_per_r = [ [] for i in range(num_rooms)]
  for i in range(num_walls):
    n_room_0 = num_rooms_per_fail_wall[i]
    room_scores_i = w_in_r_scores[i][ sort_room_ids[i] ]
    mask_i = room_scores_i > score_th
    mask_i[2-n_room_0:] = False
    room_ids_wi = sort_room_ids[i][mask_i].tolist()
    room_ids_per_w.append( room_ids_wi )

    for j in room_ids_wi:
      wall_ids_per_r[j].append(i)

    if show_room_ids_per_wall:
      room_scores_wi = room_scores_i[mask_i]
      max_out_score = room_scores_i[mask_i==False][0]
      print(f'\nwall {i}')
      print(f'room_scores_i: {room_scores_wi}')
      print(f'max_out_score: {max_out_score:.3f}')
      _show_objs_ls_points_ls( (512,512), [walls[:,:-1], walls[i:i+1,:-1], rooms[room_ids_wi,:-1] ], obj_rep, obj_colors=['green', 'red', 'white'], obj_thickness=[1,3,1] )
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass

  walls_fixed = fix_failed_room_by_room( walls, rooms, wall_ids_per_r, obj_rep )

  return walls_fixed

def cal_edge_in_room_scores(walls, rooms, obj_rep):
  from mmdet.core.bbox.geometry import dsiou_rotated_3d_bbox_np
  ious = dsiou_rotated_3d_bbox_np( walls, rooms, iou_w=1.0, size_rate_thres=None, ref='bboxes1', only_2d=True )
  return ious

def fix_failed_room_by_room(walls, rooms, wall_ids_per_r, obj_rep):
  num_rooms = rooms.shape[0]
  walls_new = []
  for i in range(num_rooms):
    wids = wall_ids_per_r[i]
    walls_fixed_i, walls_new_i =  fix_walls_1_room(walls[wids], rooms[i], obj_rep)
    walls[wids] = walls_fixed_i
    walls_new.append( walls_new_i )
  walls_fixed = walls
  if len(walls_new)>0:
    walls_new = np.concatenate(walls_new, 0)
    walls_fixed = np.concatenate([walls_fixed, walls_new], 0)
  return walls_fixed

def fix_walls_1_room(walls, room, obj_rep):
  '''
  1. Sort the corners by angle.
  2. Find corner pairs needed to be fixed.
  3. Connect two corners pair by pair. (i) Direct connect by adding an edge. (ii) Modify one corner by the intersection.
  '''
  from obj_geo_utils.geometry_utils import arg_sort_points_np
  from obj_geo_utils.obj_utils import find_wall_wall_connection
  from obj_geo_utils.topology_utils import connect_two_walls

  show_fix_process_per_room = 1

  n_wall = walls.shape[0]
  corners, _, corIds_per_w0, n_cor = gen_corners_from_lines_np(walls[:,:7], None, obj_rep, min_cor_dis_thr=2)
  sort_ids = arg_sort_points_np( corners[None] ).reshape(-1)
  corners_sorted = corners[sort_ids]
  ids0_to_ids1 = np.arange(n_cor)
  ids0_to_ids1[sort_ids] = np.arange(n_cor)
  corIds_per_w1 = ids0_to_ids1[ corIds_per_w0 ]

  wIds_per_cor = defaultdict(list)
  con_cor_ids = defaultdict(list)
  for i in range(n_wall):
    c0, c1 = corIds_per_w1[i]
    con_cor_ids[c0].append(c1)
    con_cor_ids[c1].append(c0)
    wIds_per_cor[c0].append(i)
    wIds_per_cor[c1].append(i)
  con_cor_ids = dict(con_cor_ids)
  wIds_per_cor = dict(wIds_per_cor)

  fix_pairs = []
  fix_wall_ids = []
  for i in range(n_cor):
    ids_i = con_cor_ids[i]
    if len(ids_i) <= 1:
      j_a = (i-1) % n_cor
      j_b = (i+1) % n_cor
      if ids_i[0] == j_a:
        j = j_b
      elif ids_i[0] == j_b:
        j = j_a
      else:
        continue
        _show_objs_ls_points_ls( (512,512), [walls[:,:7]], obj_rep,
              obj_colors=['white'], points_ls=[corners_sorted[[i]], corners_sorted[ ids_i ] ], point_thickness=3 )
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        raise ValueError
      if j>i:
        fix_pairs.append( [i, j] )
        wid_i = wIds_per_cor[i]
        wid_j = wIds_per_cor[j]

        if len(wid_i)!=1:
          _show_objs_ls_points_ls( (512,512), [walls[:,:7], walls[wid_i,:7]], obj_rep,
              obj_colors=['white', 'lime'], points_ls=[corners_sorted[[i]] ], point_thickness=3 )
        if len(wid_j)!=1:
          _show_objs_ls_points_ls( (512,512), [walls[:,:7], walls[wid_j,:7]], obj_rep,
              obj_colors=['white', 'lime'], points_ls=[corners_sorted[[j]] ], point_thickness=3 )
        pass
        if not( len(wid_i) == len(wid_j) == 1):
          print( "The corners needed to be fixed should only belong to one wall.")
          assert False
        fix_wall_ids.append( [wid_i[0], wid_j[0]] )

  walls_fixed = walls.copy()
  walls_new = []
  for  i in range( len(fix_pairs) ):
    cors_i = corners_sorted[fix_pairs[i]]
    walls_i = walls[fix_wall_ids[i]]
    walls1, new_wall = connect_two_corner(cors_i, walls_i, obj_rep)
    walls_fixed[fix_wall_ids[i],:7] = walls1
    walls_new.append( new_wall )

    if show_fix_process_per_room:
      _show_objs_ls_points_ls( (512,512), [walls[:,:7], room[None,:7], walls_i[:,:7]], obj_rep,
          obj_colors=['red', 'white', 'lime'], points_ls=[cors_i], point_thickness=3 )
      wn= np.concatenate(walls_new, 0)
      _show_objs_ls_points_ls( (512,512), [walls[:,:7], walls_fixed[:,:7], wn[:,:7] ], obj_rep,
          obj_colors=['green', 'red', 'yellow'], obj_thickness=[5, 2, 2] )
  walls_new = np.concatenate(walls_new, 0)
  return walls_fixed, walls_new

def connect_two_corner(corners, walls0, obj_rep):
  assert corners.shape == (2,2)
  assert walls0.shape[0] == 2
  show = 0

  c = walls0.shape[1]
  angle_dif = limit_period_np( walls0[0,6] - walls0[1,6], 0.5, np.pi)
  angle_dif = abs(angle_dif)
  walls1 = None
  if angle_dif > np.pi / 4:
      walls1 = connect_two_edges_by_intersect( walls0[:,:7], obj_rep, corners )
      new_wall = walls0[0:0,:c]
  if walls1 is None:
      new_wall = OBJ_REPS_PARSE.encode_obj(corners.reshape(1,4), 'RoLine2D_2p', obj_rep)
      if c==8:
        score = walls0[:, 7:8].mean().reshape(1,1)
        new_wall = np.concatenate([new_wall, score ], 1)
      walls1 = walls0[:,:7]
  if show:
    _show_objs_ls_points_ls( (512, 512), [walls0[:,:7], walls1[:,:7], new_wall[:,:7]], obj_rep,
                            obj_colors=['white', 'green', 'red'], obj_thickness=[4,1,1] )
    pass
  return walls1, new_wall

def get_room_quality_scores(rooms, obj_rep):
  from obj_geo_utils.geometry_utils import limit_period_np
  assert obj_rep == 'XYZLgWsHA'
  assert rooms.shape[1] == 8
  areas = rooms[:,3] * rooms[:,4]
  # smaller is better
  area_scores = 1 - areas / areas.mean()
  area_scores = np.clip(area_scores, a_min=0, a_max=None)
  rotations = limit_period_np( rooms[:,6], 0.5, np.pi/2) * 180 / np.pi
  # smaller is better
  align_scores = 1 - np.abs(rotations) / 45
  det_scores = rooms[:,-1]
  final_scores = area_scores * 0.2 + align_scores * 0.5 + det_scores * 0.3
  #_show_objs_ls_points_ls((512,512), [rooms[:,:-1]], obj_rep, obj_scores_ls=[final_scores])
  return final_scores

def sort_rooms(rooms, obj_rep):
  '''
  Small first, Align first
  '''
  from obj_geo_utils.geometry_utils import limit_period_np
  assert rooms.shape[1] == 7+1
  assert obj_rep == 'XYZLgWsHA'
  areas = rooms[:,3] * rooms[:,4]
  # smaller is better
  area_scores = 1 - areas / areas.mean()
  rotations = limit_period_np( rooms[:,6], 0.5, np.pi/2) * 180 / np.pi
  # smaller is better
  align_scores = 1 - np.abs(rotations) / 45
  det_scores = rooms[:,-1]
  final_scores = area_scores * 0.4 + align_scores * 0.3 + det_scores * 0.3
  ids = np.argsort(-final_scores)
  rooms_new = rooms[ids]
  return rooms_new

def connect_two_walls(walls0, obj_rep, walls_all=None):
  assert obj_rep == 'XYZLgWsHA'
  connection, corner_degrees, corners_per_line = find_wall_wall_connection(walls0, 3, obj_rep)
  con_num = connection[0].sum()
  if con_num == 1:
    # (1) One corner of the two walls is already connected. Connect the other
    # corner to a triangle. Add one wall.
    walls1 = connect_two_edges_to_a_triangle( corner_degrees, corners_per_line, obj_rep )
  else:
    angle_dif = limit_period_np( walls0[0,6] - walls0[1,6], 0.5, np.pi)
    angle_dif = abs(angle_dif)
    if angle_dif > np.pi / 4:
      # (2) Only connect one closer point by the intersected point. Do not add
      # walls.
      walls1 = connect_two_edges_by_intersect( walls0[:,:7], obj_rep )
    else:
      # (3) Only connect one closer point by adding a wall.
      pass
  _show_objs_ls_points_ls( (512,512), [  walls_all[:,:7], walls0[:,:7], walls1[:,:7] ], obj_rep, obj_colors=['white', 'red', 'lime'], obj_thickness=[1, 3,1] )
  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  pass

def connect_two_edges_to_a_triangle(corner_degrees, corners_per_line, obj_rep):
  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  pass

def connect_two_edges_by_intersect(walls0, obj_rep, corners_fixed=None):
  '''
  corners_fixed: if this is not None, this is the corners that should be fixed.
  '''
  assert walls0.shape[0] == 2
  w0, w1 = walls0[0], walls0[1]
  cor_0 = OBJ_REPS_PARSE.encode_obj(w0[None,:], obj_rep, 'RoLine2D_2p').reshape(2,2)
  cor_1 = OBJ_REPS_PARSE.encode_obj(w1[None,:], obj_rep, 'RoLine2D_2p').reshape(2,2)
  intersect = line_intersection_2d(cor_0, cor_1, min_angle=np.pi/8).reshape(1,2)
  cor_0_new = replace_1cor_of_edge(cor_0, intersect).reshape(1,4)
  cor_1_new = replace_1cor_of_edge(cor_1, intersect).reshape(1,4)
  if corners_fixed is not None:
    dif0 = corners_fixed[0:1] - cor_0_new.reshape(2,2)
    dis0 = np.linalg.norm( dif0, axis=1 ).min()
    if dis0 < 2:
      # The corners_fixed should be replaced. But it is still here. Invalid
      # connection
      return None
  w_0_new = OBJ_REPS_PARSE.encode_obj(cor_0_new, 'RoLine2D_2p', obj_rep)[0]
  w_1_new = OBJ_REPS_PARSE.encode_obj(cor_1_new, 'RoLine2D_2p', obj_rep)[0]
  walls_new = np.concatenate([w_0_new[None], w_1_new[None]], 0)
  return walls_new

def replace_1cor_of_edge(edge, cor):
  assert edge.shape == (2,2)
  assert cor.shape == (1,2)
  dis = np.linalg.norm( edge - cor, axis=1 )
  i = dis.argmin()
  edge_new = edge.copy()
  edge_new[i] = cor
  return edge_new

