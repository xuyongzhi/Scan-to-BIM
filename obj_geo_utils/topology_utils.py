# xyz 16 June
import os
import numpy as np
import mmcv
from obj_geo_utils.geometry_utils import sin2theta_np, angle_with_x_np, \
      vec_from_angle_with_x_np, angle_with_x
import cv2
import torch
from obj_geo_utils.geometry_utils import limit_period_np, four_corners_to_box,\
  sort_four_corners, line_intersection_2d, mean_angles, angle_dif_by_period_np,\
  lines_intersection_2d
from obj_geo_utils.obj_utils import OBJ_REPS_PARSE, GraphUtils
from obj_geo_utils.obj_utils import find_wall_wall_connection
from obj_geo_utils.line_operations import gen_corners_from_lines_np, get_lineIdsPerCor_from_corIdsPerLine
from collections import defaultdict
from tools.visual_utils import _show_objs_ls_points_ls, _draw_objs_ls_points_ls, show_1by1
from obj_geo_utils.geometry_utils import arg_sort_points_np, check_duplicate


def geometrically_opti_walls(det_lines, obj_rep, opt_graph_cor_dis_thr=None, min_out_length=None):
      from utils_dataset.graph_eval_utils import GraphEval
      if opt_graph_cor_dis_thr is None:
        opt_graph_cor_dis_thr = GraphEval._opt_graph_cor_dis_thr
        min_out_length = GraphEval._min_out_length
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
  show_fail_rooms = 0
  show_fixed_res = 0

  num_rooms = rooms.shape[0]
  wall_ids_per_room, room_ids_per_wall, num_walls_inside_room, rooms_from_w = get_rooms_from_edges(walls[:,:7], obj_rep, gen_bbox=True, show_rooms=False)
  non_full_mask = (room_ids_per_wall<0).any(1)
  full_w_ids = np.where(non_full_mask==False)[0]
  non_full_w_ids = np.where(non_full_mask)[0]
  ious = dsiou_rotated_3d_bbox_np( rooms[:,:7], rooms_from_w[:,:7], iou_w=1.0, size_rate_thres=None, ref='union', only_2d=True )
  succ_r_ids = ious.argmax(0)
  succ_ious = ious.max(0)
  fail_r_ids = [i for i in range(num_rooms) if i not in succ_r_ids]

  fail_rooms = rooms[fail_r_ids]
  fail_walls = walls[non_full_w_ids]
  if show_fail_rooms:
    print(f'detected rooms')
    _show_objs_ls_points_ls( (512,512), [walls[:,:7], rooms[:,:7] ], obj_rep, obj_colors=['white', 'red'] )
    print(f'rooms from wall')
    _show_objs_ls_points_ls( (512,512), [walls[:,:7], rooms_from_w[:,:7] ], obj_rep, obj_colors=['white', 'red'] )
    print(f'success rooms')
    _show_objs_ls_points_ls( (512,512), [rooms_from_w[:,:7], rooms[succ_r_ids][:,:7], ], obj_rep, obj_colors=['white', 'red', ], obj_thickness=[6,2,2])
    print(f'fail rooms')
    _show_objs_ls_points_ls( (512,512), [walls[:,:7], rooms[fail_r_ids][:,:7] ], obj_rep, obj_colors=['white', 'red'] )
    print(f'faiil walls')
    _show_objs_ls_points_ls( (512,512), [ fail_walls[:,:7], fail_rooms[:,:7] ], obj_rep, obj_colors=['white', 'red'] )

  _, cor_degress, _ = find_wall_wall_connection(walls[:,:7], 2, obj_rep)

  room_ids_per_fail_wall = room_ids_per_wall[non_full_w_ids]
  num_rooms_per_fail_wall = (room_ids_per_fail_wall>=0).sum(1)
  walls_fail_fixed = fix_failed_rooms_walls(fail_walls, fail_rooms, obj_rep, num_rooms_per_fail_wall)
  walls_fixed = np.concatenate([walls[full_w_ids], walls_fail_fixed ], 0)

  if show_fixed_res:
    _show_objs_ls_points_ls( (512,512), [rooms[:,:7], walls[:,:7], walls_fixed[:,:7] ],
                            obj_rep, obj_colors=['white', 'red', 'lime'], obj_thickness=[1, 6, 2] )
    _show_objs_ls_points_ls( (512,512), [rooms[:,:7],  walls_fixed[:,:7] ],
                            obj_rep, obj_colors=['white', 'lime'], obj_thickness=[1, 2] )


  if 0:
    _show_objs_ls_points_ls( (512,512), [walls_final[:,:7],  walls_fixed[:,:7] ],
                            obj_rep, obj_colors=['red', 'lime'], obj_thickness=[4, 1] )
    _show_objs_ls_points_ls( (512,512), [walls_final[:,:7] ], obj_rep, obj_colors=['random'], obj_thickness=4 )
  #return walls_fixed
  walls_final, ids_geo_opt = geometrically_opti_walls( walls_fixed, obj_rep)
  return walls_final

def fix_failed_rooms_walls(walls, rooms, obj_rep, num_rooms_per_fail_wall=None):
  '''
  1. For each wall, find the one or two matched rooms.
  2. Get the matched walls for each room.
  3. Fix room by room
  '''
  show_room_ids_per_wall = 0
  show_walls_per_room = 0
  show_fail_fixed = 0

  score_th = 0.5
  num_rooms = rooms.shape[0]
  num_walls = walls.shape[0]
  walls_aug = walls.copy()
  walls_aug[:,3] *= 0.8
  walls_aug[:,4] = 10
  rooms_aug = rooms.copy()
  rooms_aug[:,3:5] *= 1.2

  #_show_objs_ls_points_ls( (512,512), [walls[:,:-1], rooms[:,:-1]], obj_rep, obj_colors=['red', 'white'], obj_thickness=[3,1] )
  #_show_objs_ls_points_ls( (512,512), [walls[:,:-1], rooms_aug[:,:-1]], obj_rep, obj_colors=['red', 'white'], obj_thickness=[3,1] )

  w_in_r_scores = cal_edge_in_room_scores(walls_aug[:,:7], rooms_aug[:,:7], obj_rep)
  room_qua_scores = get_room_quality_scores(rooms, obj_rep)
  #w_in_r_scores = w_in_r_scores *  room_qua_scores[None,:]
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

    if show_room_ids_per_wall and i ==16:
      room_scores_wi = room_scores_i[mask_i]
      print(f'\nwall {i}')
      print(f'room_scores_i: {room_scores_wi}')
      max_out_score = room_scores_i[mask_i==False]
      if len(max_out_score)>0:
        max_out_score = max_out_score[0]
        print(f'max_out_score: {max_out_score:.3f}')
      _show_objs_ls_points_ls( (512,512), [walls[:,:-1], walls[i:i+1,:-1], rooms_aug[room_ids_wi,:-1] ], obj_rep, obj_colors=['green', 'red', 'white'], obj_thickness=[1,3,1] )
      _show_objs_ls_points_ls( (512,512), [walls_aug[:,:-1], walls[i:i+1,:-1], rooms_aug[:,:-1] ], obj_rep, obj_colors=['green', 'red', 'white'], obj_thickness=[1,3,1] )
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass

  num_r = rooms.shape[0]
  for i in range(num_r):
    wids = wall_ids_per_r[i]
    valid_ids_i =  clean_outer_false_walls_of_1_room(rooms[i], walls[wids], obj_rep)
    wids_valid = [wids[vi] for vi in  valid_ids_i]
    valid_ids_i =  clean_inner_false_walls_of_1_room(rooms[i], walls[wids_valid], obj_rep)
    wids_valid = [wids_valid[vi] for vi in  valid_ids_i]
    wall_ids_per_r[i] = wids_valid
    if show_walls_per_room:
      _show_objs_ls_points_ls( (512,512), [walls[:,:-1], walls[wids,:-1], rooms[i:i+1,:-1] ], obj_rep, obj_colors=['green', 'red', 'white'], obj_thickness=[1,3,1] )
      _show_objs_ls_points_ls( (512,512), [walls[:,:-1], walls[wids_valid,:-1], rooms[i:i+1,:-1] ], obj_rep, obj_colors=['green', 'red', 'white'], obj_thickness=[1,3,1] )

  walls_fixed, walls_new = fix_failed_room_by_room( walls.copy(), rooms, wall_ids_per_r, obj_rep )
  num_new = walls_new.shape[0]
  if show_fail_fixed:
    print(f'\nadd {num_new} walls\n')
    _show_objs_ls_points_ls( (512,512), [walls[:,:7], walls_fixed[:,:7], walls_new[:,:7]], obj_rep, obj_colors=['white', 'blue', 'red'], obj_thickness=[8,2,2] )
  walls_fixed = np.concatenate([walls_fixed, walls_new], 0)

  return walls_fixed

def clean_inner_false_walls_of_1_room(room, walls, obj_rep):
  if walls.shape[0] == 0:
    return []
  walls = walls[:,:7].copy()
  corners, _, corIds_per_line, num_cor_uq, cor_degrees = gen_corners_from_lines_np( walls, None, obj_rep, 2, get_degree=1 )
  remove_wall_ids = []
  for i in range(num_cor_uq):
    if cor_degrees[i] > 1:
      wids_i = np.where( (corIds_per_line == i).any(1) )[0]
      #_show_objs_ls_points_ls( (512,512), [room[None,:7], walls, walls [wids_i] ], obj_rep, obj_colors=['red', 'white','blue'], obj_thickness=[8,4,2] )
      check_duplicate(walls, obj_rep)

      rm_wall_ids = []
      for j in wids_i:
        cor_js = [k for k in corIds_per_line[j] if k !=i]
        if len(cor_js) == 0:
          continue
        cor_j = cor_js[0]
        if cor_degrees[cor_j] == 0 or cor_degrees[cor_j] > 1:
          rm_wall_ids.append(j)
        pass
      if len(rm_wall_ids)>0:
        walls_i = walls[rm_wall_ids]
        rm_wall_id = rm_wall_ids[ walls_i[:,3].argmin() ]
      else:
        print(f'A corner connect three walls, but do not know which one to remove')
        continue
        _show_objs_ls_points_ls( (512,512), [room[None,:7], walls, walls [wids_i] ], obj_rep, obj_colors=['red', 'white','blue'], obj_thickness=[8,4,2] )
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        pass
      remove_wall_ids.append( rm_wall_id )
      pass
  n = walls.shape[0]
  valid_ids = [i for i in range(n) if i not in remove_wall_ids]
  return valid_ids

def clean_outer_false_walls_of_1_room(room, walls, obj_rep):
  from mmdet.core.bbox.geometry import dsiou_rotated_3d_bbox_np
  assert obj_rep == 'XYZLgWsHA'
  nw = walls.shape[0]
  cen_room = room[None, :2]
  w_lines = OBJ_REPS_PARSE.encode_obj(walls[:,:7], obj_rep, 'RoLine2D_2p').reshape(-1,2,2)
  valid_ids = []
  for i in range(nw):
    cen_i = walls[i:i+1, :2]
    box = np.concatenate([cen_i, cen_room], 1)
    box = OBJ_REPS_PARSE.encode_obj(box, 'RoLine2D_2p', obj_rep)
    box[:, 3:5] *= 0.95

    line_i = OBJ_REPS_PARSE.encode_obj(box, obj_rep, 'RoLine2D_2p').reshape(1,2,2)
    intsects = lines_intersection_2d( line_i, w_lines, True, True )
    valid_i = np.isnan(intsects).all()

    if valid_i:
      valid_ids.append( i )
    if not valid_i and 0:
      print(valid_i)
      print(intsects)
      _show_objs_ls_points_ls( (512,512), [walls[:,:7], walls[i:i+1,:7], box], obj_rep, obj_colors=['white', 'lime', 'red'] )
  #_show_objs_ls_points_ls( (512,512), [walls[:,:7], walls[valid_ids,:7], box], obj_rep, obj_colors=['white', 'lime', 'red'] )
  return valid_ids

def cal_edge_in_room_scores(walls, rooms, obj_rep):
  from mmdet.core.bbox.geometry import dsiou_rotated_3d_bbox_np
  ious = dsiou_rotated_3d_bbox_np( walls, rooms, iou_w=1.0, size_rate_thres=None, ref='bboxes1', only_2d=True )
  #print(ious)
  #_show_objs_ls_points_ls( (512,512), [walls, rooms], obj_rep, obj_colors=['green', 'red' ], obj_thickness=[1,3] )
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
  if len(walls_new) > 0:
    walls_new = np.concatenate(walls_new, 0)
  else:
    walls_new = walls_fixed[0:0]

  #_show_objs_ls_points_ls( (512,512), [walls[:,:7], walls_new[:,:7], walls_fixed[:,:7]], obj_rep,
  #          obj_colors=['white', 'lime', 'red'], obj_thickness=[1,1,1])
  return walls_fixed, walls_new

def fulfil_a_rectangle_room(corners_sorted, walls, room, obj_rep, show_fix_process_per_room=0):
  '''
  There are only two walls for one room. Add these two walls are connected.
  Add one extra corner and two extra walls to fulfil a rectangle.
  '''
  cors = corners_sorted
  cor3 = cors[0] - cors[1] + cors[2]
  tmp = np.repeat(cor3[None], 2, 0)
  wall_cors_new = np.concatenate([tmp, cors[[0,2]]], 1).reshape(2,4)
  walls_new = OBJ_REPS_PARSE.encode_obj(wall_cors_new, 'RoLine2D_2p', obj_rep)
  walls_new = np.concatenate([walls_new, walls[:,7:8]], 1)
  if show_fix_process_per_room:
    _show_objs_ls_points_ls( (512,512), [ walls[:,:7], room[None,:7], walls_new[:,:7] ], obj_rep, obj_colors=['red', 'white', 'lime'])
  return walls, walls_new

def insert_with_order(groups, exist_i, new_i):
  '''
  groups: []
  exist_i: alread in groups
  new_i: not in groups, connected with exist_i

  insert new_i into groups. make exist_i and new_i connected
  exist_i should at the start or end of groups
  '''
  if exist_i == groups[0]:
    groups = [new_i] +  groups
  elif exist_i == groups[-1]:
    groups = groups + [new_i]
  else:
    print("Not a loop")
    #raise NotImplementedError
  return groups

def merge_groups( group_ids, w_in_g_ids ):
  gi0 = w_in_g_ids[0]
  for i in w_in_g_ids[1:]:
    group_ids[gi0] = merge_two_groups( group_ids[gi0], group_ids[i] )
  group_ids = [group_ids[ i ] for i in range(len(group_ids)) if i not in w_in_g_ids[1:] ]
  return group_ids

def merge_two_groups(group0, group1):
  if group0[0] == group1[-1]:
    return group1 + group0[1:]
  if group1[0] == group0[-1]:
    return group0 + group1[1:]
  if group0[0] == group1[0]:
    group0.reverse()
    return group0 + group1[1:]
  if group0[-1] == group1[-1]:
    group1.reverse()
    return group0 + group1[1:]
  raise NotImplementedError

def sort_connected_corners(walls, obj_rep):
  '''
  Split the corners into several group_ids by connection. Then sort the splited group_ids.
  '''
  #_show_objs_ls_points_ls( (512,512), [walls[:,:7]], obj_rep, )

  corners, _, corIds_per_w0, n_cor = gen_corners_from_lines_np(walls[:,:7], None, obj_rep, min_cor_dis_thr=2)
  group_ids = []
  n_wall = walls.shape[0]
  for i in range(n_wall):
    c0, c1 = corIds_per_w0[i]
    w_in_g_ids = []
    for j in range( len(group_ids) ):
      if c0 in group_ids[j] or c1 in group_ids[j]:
        if len(w_in_g_ids) == 0:
            if c0 not in group_ids[j]:
              group_ids[j] = insert_with_order( group_ids[j], c1, c0 )
            if c1 not in group_ids[j]:
              group_ids[j] = insert_with_order( group_ids[j], c0, c1 )
        w_in_g_ids.append(j)
    if len(w_in_g_ids) >= 2:
      # merge the groups of w_in_g_ids
      group_ids = merge_groups(group_ids, w_in_g_ids)
    if len(w_in_g_ids) == 0:
      group_ids.append( [c0, c1] )
    #print(i, group_ids)
  group_ids = [np.array(ids) for ids in group_ids]
  cor_nums_per_g = [len(g) for g in group_ids]
  n_cor_1 = sum(cor_nums_per_g)
  if not n_cor_1 == n_cor:
    _show_objs_ls_points_ls( (512,512), [walls[:,:7]], obj_rep, )
    assert False, f"{n_cor} != {n_cor_1}"

  min_xy = corners.min(axis=0, keepdims=True)
  max_xy = corners.max(axis=0, keepdims=True)
  center = (min_xy + max_xy) / 2

  group_centers = [ corners[ids].mean(0)[None] for ids in group_ids]
  group_centers = np.concatenate(group_centers, 0)
  group_sort_ids = arg_sort_points_np( group_centers[None], center[None] ).reshape(-1)
  group_ids = [ group_ids[i] for i in group_sort_ids ]

  if 0:
    for ids in group_ids:
      _show_objs_ls_points_ls( (512,512), [walls[:,:7]], obj_rep, [corners[ids], center], point_scores_ls=[range(len(ids)), None] )

  num_groups = len(group_ids)
  for i in  range(1, num_groups):
    anchor_last = corners[ group_ids[i-1][-1] ][None]
    s, e = group_ids[i][0], group_ids[i][-1]
    cor_cur_s_e = corners[ [s, e] ]
    dis_s_e = np.linalg.norm( anchor_last - cor_cur_s_e, axis=1 )
    if dis_s_e[0] > dis_s_e[1]:
      group_ids[i] = group_ids[i][::-1]
    if 0:
      print(dis_s_e)
      _show_objs_ls_points_ls( (512,512), [walls[:,:7]], obj_rep, [anchor_last, cor_cur_s_e[0:1], cor_cur_s_e[1:2] ], point_scores_ls=[[0],[1],[2]] )
    pass

  cor_sort_ids = np.concatenate(group_ids, 0)
  corners_sorted = corners[ cor_sort_ids ]

  ids0_to_ids1 = np.arange(n_cor)
  ids0_to_ids1[cor_sort_ids] = np.arange(n_cor)
  corIds_per_w1 = ids0_to_ids1[ corIds_per_w0 ]

  #_show_objs_ls_points_ls( (512,512), [walls[:,:7]], obj_rep, [corners_sorted], point_scores_ls=[range(n_cor)] )
  return corners_sorted, corIds_per_w1

def fix_walls_1_room(walls, room, obj_rep):
  '''
  1. Sort the corners by angle.
  2. Find corner pairs needed to be fixed.
  3. Connect two corners pair by pair. (i) Direct connect by adding an edge. (ii) Modify one corner by the intersection.
  '''

  if walls.shape[0] == 0:
    return walls, walls

  show_fix_process_per_room = 0
  show_fix_res_per_room = 0

  n_wall = walls.shape[0]
  corners_sorted, corIds_per_w1 = sort_connected_corners(walls, obj_rep)
  n_cor = corners_sorted.shape[0]


  if walls.shape[0] == 2 and corners_sorted.shape[0] == 3:
    #room_cors = OBJ_REPS_PARSE.encode_obj(room[None,:7], obj_rep, 'Rect4CornersZ0Z1')[:,:8].reshape(4,2)
    return fulfil_a_rectangle_room(corners_sorted, walls, room, obj_rep, show_fix_res_per_room)

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

        if 0:
          if len(wid_i)!=1:
            _show_objs_ls_points_ls( (512,512), [walls[:,:7], walls[wid_i,:7]], obj_rep,
                obj_colors=['white', 'lime'], points_ls=[corners_sorted[[i]] ], point_thickness=3 )
          if len(wid_j)!=1:
            _show_objs_ls_points_ls( (512,512), [walls[:,:7], walls[wid_j,:7]], obj_rep,
                obj_colors=['white', 'lime'], points_ls=[corners_sorted[[j]] ], point_thickness=3 )
        pass
        if not( len(wid_i) == len(wid_j) == 1):
          print( "The corners needed to be fixed should only belong to one wall.")
          #assert False
        fix_wall_ids.append( [wid_i[0], wid_j[0]] )

  walls_fixed = walls.copy()
  walls_new = []
  fixed_ids = []
  for  i in range( len(fix_pairs) ):
    cors_i = corners_sorted[fix_pairs[i]]
    walls_i = walls_fixed[fix_wall_ids[i]]
    walls1, new_wall = connect_two_corner(cors_i, walls_i, obj_rep)
    walls_fixed[fix_wall_ids[i],:7] = walls1
    fixed_ids.append( fix_wall_ids[i] )
    walls_new.append( new_wall )

    if show_fix_process_per_room:
      _show_objs_ls_points_ls( (512,512), [walls[:,:7], room[None,:7], walls_i[:,:7]], obj_rep,
          obj_colors=['red', 'white', 'lime'], points_ls=[cors_i], point_thickness=3 )
      wn= np.concatenate(walls_new, 0)
      _show_objs_ls_points_ls( (512,512), [walls[:,:7], walls_fixed[:,:7], wn[:,:7] ], obj_rep,
          obj_colors=['green', 'red', 'yellow'], obj_thickness=[5, 2, 2] )
      pass
  if len(walls_new) > 0:
    walls_new = np.concatenate(walls_new, 0)
  else:
    walls_new = walls_fixed[0:0].copy()

  if show_fix_res_per_room:
    walls_final = np.concatenate([walls_fixed, walls_new], 0)
    _show_objs_ls_points_ls( (512,512), [walls[:,:7], room[None,:7], walls_final[:,:7]], obj_rep,
            obj_colors=['red', 'white', 'lime'], obj_thickness=[5,1,1])
  #fixed_ids = np.concatenate( fixed_ids, 0 )
  return walls_fixed, walls_new

def connect_two_corner(corners, walls0, obj_rep):
  assert corners.shape == (2,2)
  assert walls0.shape[0] == 2
  show = 0

  c = walls0.shape[1]
  angle_dif = limit_period_np( walls0[0,6] - walls0[1,6], 0.5, np.pi)
  angle_dif = abs(angle_dif)
  if angle_dif > np.pi / 4:
      new_wall = intersect_2edges_by_add_new_wall( walls0[:,:7], obj_rep, corners )
      walls1 = walls0[:,:7]
  else:
      new_wall = OBJ_REPS_PARSE.encode_obj(corners.reshape(1,4), 'RoLine2D_2p', obj_rep)
      walls1 = walls0[:,:7]
  if c==8:
    score =  new_wall[:,0:1].copy()
    score[:]  =1
    new_wall = np.concatenate([new_wall, score ], 1)
  if show:
    _show_objs_ls_points_ls( (512, 512), [walls0[:,:7], walls1[:,:7], new_wall[:,:7]], obj_rep,
                            obj_colors=['white', 'green', 'red'], obj_thickness=[4,1,1] )
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
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


def get_rooms_walls_rel(walls, rooms, obj_rep, num_rooms_per_fail_wall=None):
  show_room_ids_per_wall = 0
  show_walls_per_room = 0
  show_fail_fixed = 0

  score_th = 0.5
  num_rooms = rooms.shape[0]
  num_walls = walls.shape[0]
  walls_aug = walls.copy()
  walls_aug[:,3] *= 0.8
  walls_aug[:,4] = 10
  rooms_aug = rooms.copy()
  rooms_aug[:,3:5] *= 1.2

  #_show_objs_ls_points_ls( (512,512), [walls[:,:-1], rooms[:,:-1]], obj_rep, obj_colors=['red', 'white'], obj_thickness=[3,1] )
  #_show_objs_ls_points_ls( (512,512), [walls[:,:-1], rooms_aug[:,:-1]], obj_rep, obj_colors=['red', 'white'], obj_thickness=[3,1] )

  w_in_r_scores = cal_edge_in_room_scores(walls_aug[:,:7], rooms_aug[:,:7], obj_rep)
  room_qua_scores = get_room_quality_scores(rooms, obj_rep)
  #w_in_r_scores = w_in_r_scores *  room_qua_scores[None,:]
  sort_room_ids = (-w_in_r_scores).argsort(1)

  room_ids_per_w = []
  wall_ids_per_r = [ [] for i in range(num_rooms)]
  for i in range(num_walls):
    room_scores_i = w_in_r_scores[i][ sort_room_ids[i] ]
    mask_i = room_scores_i > score_th
    if num_rooms_per_fail_wall is not None:
      n_room_0 = num_rooms_per_fail_wall[i]
      mask_i[2-n_room_0:] = False
    room_ids_wi = sort_room_ids[i][mask_i].tolist()
    room_ids_per_w.append( room_ids_wi )

    for j in room_ids_wi:
      wall_ids_per_r[j].append(i)

    if show_room_ids_per_wall and i ==16:
      room_scores_wi = room_scores_i[mask_i]
      print(f'\nwall {i}')
      print(f'room_scores_i: {room_scores_wi}')
      max_out_score = room_scores_i[mask_i==False]
      if len(max_out_score)>0:
        max_out_score = max_out_score[0]
        print(f'max_out_score: {max_out_score:.3f}')
      _show_objs_ls_points_ls( (512,512), [walls[:,:-1], walls[i:i+1,:-1], rooms_aug[room_ids_wi,:-1] ], obj_rep, obj_colors=['green', 'red', 'white'], obj_thickness=[1,3,1] )
      _show_objs_ls_points_ls( (512,512), [walls_aug[:,:-1], walls[i:i+1,:-1], rooms_aug[:,:-1] ], obj_rep, obj_colors=['green', 'red', 'white'], obj_thickness=[1,3,1] )
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass

  num_r = rooms.shape[0]
  for i in range(num_r):
    wids = wall_ids_per_r[i]
    valid_ids_i =  clean_outer_false_walls_of_1_room(rooms[i], walls[wids], obj_rep)
    wids_valid = [wids[vi] for vi in  valid_ids_i]
    valid_ids_i =  clean_inner_false_walls_of_1_room(rooms[i], walls[wids_valid], obj_rep)
    wids_valid = [wids_valid[vi] for vi in  valid_ids_i]
    wall_ids_per_r[i] = wids_valid
    if show_walls_per_room:
      _show_objs_ls_points_ls( (512,512), [walls[:,:-1], walls[wids,:-1], rooms[i:i+1,:-1] ], obj_rep, obj_colors=['green', 'red', 'white'], obj_thickness=[1,3,1] )
      _show_objs_ls_points_ls( (512,512), [walls[:,:-1], walls[wids_valid,:-1], rooms[i:i+1,:-1] ], obj_rep, obj_colors=['green', 'red', 'white'], obj_thickness=[1,3,1] )
  return wall_ids_per_r

def draw_walls_rooms_rel(img, walls, rooms, obj_rep):
  from tools.color import COLOR_MAP_2D, ColorList, ColorValuesNp

  check_duplicate(walls, obj_rep)
  wall_ids_per_r = get_rooms_walls_rel(walls, rooms, obj_rep)
  num_rooms = rooms.shape[0]
  for i in range(num_rooms):
    ci = ColorList[i]
    img = _show_objs_ls_points_ls( img, points_ls = [rooms[i:i+1,:2]], point_colors=ci, point_thickness=8, only_save=1 )
    walls_i  = walls[ wall_ids_per_r[i] ]
    ni = walls_i.shape[0]
    tmp = np.repeat( rooms[i:i+1], ni, 0 )
    rels_i = np.concatenate( [walls_i[:,:2], tmp[:,:2]], 1 )
    img = _show_objs_ls_points_ls( img, [rels_i], 'RoLine2D_2p', obj_colors=ci, obj_thickness=2, only_save=1 )
    pass
  pass
  #mmcv.imshow( img )
  return img

def connect_two_edges_to_a_triangle(corner_degrees, corners_per_line, obj_rep):
  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  pass

def intersect_2edges_by_modification(walls0, obj_rep, corners_bad=None):
  '''
  corners_bad: if this is not None, this is the corners that should be fixed.
  '''
  assert walls0.shape[0] == 2
  #_show_objs_ls_points_ls( (512,512), [walls0], obj_rep, points_ls=[corners_bad] )
  w0, w1 = walls0[0], walls0[1]
  cor_0 = OBJ_REPS_PARSE.encode_obj(w0[None,:], obj_rep, 'RoLine2D_2p').reshape(2,2)
  cor_1 = OBJ_REPS_PARSE.encode_obj(w1[None,:], obj_rep, 'RoLine2D_2p').reshape(2,2)
  intersect = line_intersection_2d(cor_0, cor_1, min_angle=np.pi/8).reshape(1,2)
  cor_0_new = replace_1cor_of_edge_to_midify(cor_0, intersect).reshape(1,4)
  cor_1_new = replace_1cor_of_edge_to_midify(cor_1, intersect).reshape(1,4)
  if corners_bad is not None:
    dis_to_bad = np.linalg.norm( corners_bad - intersect, axis=1 )
    dis0 = np.linalg.norm( cor_0 - intersect, axis=1 )
    dis1 = np.linalg.norm( cor_1 - intersect, axis=1 )
    check0 =  abs(dis_to_bad[0] - dis0.min()) < 1
    check1 =  abs(dis_to_bad[1] - dis1.min()) < 1
    check = check0 * check1
    if not check:
      # The corners_bad should be replaced. But it is still here. Invalid
      # connection
      return None
  w_0_new = OBJ_REPS_PARSE.encode_obj(cor_0_new, 'RoLine2D_2p', obj_rep)[0]
  w_1_new = OBJ_REPS_PARSE.encode_obj(cor_1_new, 'RoLine2D_2p', obj_rep)[0]
  walls_new = np.concatenate([w_0_new[None], w_1_new[None]], 0)
  return walls_new

def intersect_2edges_by_add_new_wall(walls0, obj_rep, corners_bad=None):
  '''
  corners_bad: if this is not None, this is the corners that should be fixed.
  '''
  assert walls0.shape[0] == 2
  #_show_objs_ls_points_ls( (512,512), [walls0], obj_rep, points_ls=[corners_bad] )
  w0, w1 = walls0[0], walls0[1]
  cor_0 = OBJ_REPS_PARSE.encode_obj(w0[None,:], obj_rep, 'RoLine2D_2p').reshape(2,2)
  cor_1 = OBJ_REPS_PARSE.encode_obj(w1[None,:], obj_rep, 'RoLine2D_2p').reshape(2,2)
  intersect = line_intersection_2d(cor_0, cor_1, min_angle=np.pi/8).reshape(1,2)
  intersect = np.repeat( intersect, 2, 0 )
  new_walls = np.concatenate( [intersect, corners_bad], 1 )
  new_walls = OBJ_REPS_PARSE.encode_obj(new_walls, 'RoLine2D_2p', obj_rep)
  return new_walls

def replace_1cor_of_edge_to_add(edge, cor):
  assert edge.shape == (2,2)
  assert cor.shape == (1,2)
  dis = np.linalg.norm( edge - cor, axis=1 )
  i = dis.argmax()
  edge_new = edge.copy()
  edge_new[i] = cor
  return edge_new

def replace_1cor_of_edge_to_midify(edge, cor):
  assert edge.shape == (2,2)
  assert cor.shape == (1,2)
  dis = np.linalg.norm( edge - cor, axis=1 )
  i = dis.argmin()
  edge_new = edge.copy()
  edge_new[i] = cor
  return edge_new

