import numpy as np
from obj_geo_utils.geometry_utils import limit_period_np, vertical_dis_1point_lines, points_in_lines
from tools.visual_utils import _show_objs_ls_points_ls, _show_3d_points_objs_ls, _show_3d_as_img
from mmdet.core.bbox.geometry import dsiou_rotated_3d_bbox_np
from obj_geo_utils.line_operations import transfer_lines_points

def align_bboxes_with_wall(dets, walls, cat, obj_rep):
  assert cat in ['door', 'window']
  assert obj_rep == 'XYZLgWsHA'
  new_dets = dets.copy()
  det_scores = dets[:,-1:]
  dets = dets[:,:-1]
  wall_scores = walls[:,-1:]
  walls = walls[:,:-1]
  ious = dsiou_rotated_3d_bbox_np(dets, walls, iou_w=1, size_rate_thres=0.2, ref='bboxes1')
  the_wall_ids = ious.argmax(1)
  max_ious = ious.max(1)
  ndet = dets.shape[0]

  #print(f'max_ious: {max_ious}')
  #_show_objs_ls_points_ls( (512,512), [dets, walls], obj_rep=obj_rep, obj_colors=['red', 'blue'])
  matched_ids = []
  for i in range(ndet):
    if max_ious[i] > 0.5:
      matched_ids.append(i)
      the_wall = walls[the_wall_ids[i]]
      #_show_objs_ls_points_ls( (512,512), [dets[i:i+1], the_wall[None,:]], obj_rep=obj_rep, obj_colors=['red', 'blue'])
      angle_dif = np.abs( limit_period_np( dets[i,-1] - the_wall[-1] , 0.5, np.pi))
      if angle_dif < np.pi/4:
        #new_dets[i, -1] = the_wall[-1]
        #new_dets[i, 4]  = max(dets[i,4], the_wall[4]*1.5)
        new_dets[i,:7] = align_1_box_with_wall(new_dets[i,:7], the_wall[:7], obj_rep)
        #_show_objs_ls_points_ls( (512,512), [new_dets[i:i+1,:7], the_wall[None,:]], obj_rep=obj_rep, obj_colors=['red', 'blue'])
      pass

  pass
  new_dets = new_dets[matched_ids]

  from mmdet.ops.nms.nms_wrapper import nms_rotated_np
  iou_thr = 0.2
  min_width_length_ratio = 0.2
  new_dets_nms, ids = nms_rotated_np(new_dets, obj_rep, iou_thr, min_width_length_ratio)
  return   new_dets_nms

def align_1_box_with_wall(box, wall, obj_rep):
  box = box.reshape(1,7)
  wall = wall.reshape(1,7)
  center = (wall[0,0], wall[0,1])
  angle = -wall[0,-1]
  bw = np.concatenate([box, wall], 0)
  bw_r, _ = transfer_lines_points( bw, obj_rep, None, center, angle, (0,0) )
  box_r = bw_r[0]
  wall_r = bw_r[1]

  box_r[-1] = 0
  box_r[1] = wall_r[1]
  box_r[4] = wall_r[4]*2
  #center = (-wall[0,0], -wall[0,1])
  box_new,_ = transfer_lines_points( box_r[None,:], obj_rep, None, center, -angle, (0,0) )
  return box_new[0]

def points_to_box_align_with_wall(points, walls, cat_name, voxel_size):
  for i in range(4):
    box2d, wall_id = points_to_box_align_with_wall_1ite(points, walls, cat_name, voxel_size)
    if box2d is None:
      return None
    length = box2d[0,2] * voxel_size
    thick =  box2d[0,3] * voxel_size

    print(f'{cat_name} length={length}, thick={thick}')
    if length > 0.5:
      return box2d
    mask = np.arange(walls.shape[0]) != wall_id
    walls = walls[mask]
    pass
  return None

def box_align_with_wall_1ite(bboxes, walls, cat_name, voxel_size, max_diss_meter=1):
  from obj_geo_utils.line_operations import transfer_lines_points
  max_thick = MAX_THICK_MAP[cat_name] / voxel_size
  thick_add_on_wall = int(THICK_GREATER_THAN_WALL[cat_name] / voxel_size)
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
  walls_r, points_r = transfer_lines_points( walls_.copy(), 'XYLgWsA', points, center, angle, (0,0) )
  the_wall_r = walls_r[wall_id]

  #_show_3d_points_objs_ls( [points_r], objs_ls = [walls_r, walls_r[wall_id][None] ], obj_rep='XYLgWsA', obj_colors=['green', 'red'] )
  #_show_objs_ls_points_ls( (1512,1512), [walls_r], obj_rep='XYLgWsA', obj_colors=['green','blue'], points_ls = [points_r])

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
  thick = min(box_r[3] + thick_add_on_wall, max_thick)
  y_max = max_xy[1]
  y_min = min_xy[1]
  y_mean = (y_max + y_min)/2
  move = wall_yc - y_mean
  yc = y_mean + move

  y_min_new = y_min + move
  y_max_new = y_max + move

  if abs(y_max_new) < y_min:
    # keep y_min, because y_min close to wall
    y_min_new = y_min
    yc = y_min_new + thick/2
  elif abs(y_min_new) > y_max:
    # keep y_max, becasue y_max close to wall
    y_max_new = y_max
    yc = y_max_new - thick/2

  #if wall_yc > y_max - thick/2:
  #  # keep y_max
  #  y_min = y_max - thick
  #  yc = (y_min + y_max) / 2
  #elif wall_yc < y_min + thick/2:
  #  # keep y_min
  #  y_max = y_min + thick
  #  yc = (y_min + y_max) / 2
  #else:
  #  # use yc of wall
  #  yc = wall_yc


  box_r[1] = yc
  box_r[3] = thick

  box_r = box_r[None,:]

  box_out, points_out = transfer_lines_points( box_r, 'XYLgWsA', points_r, center, -angle, (0,0) )


  wall_view = walls_[wall_id:wall_id+1]
  wall_view = walls_

  #_show_objs_ls_points_ls( (2024,2024), [wall_view, box_out], obj_rep='XYLgWsA', obj_colors=['green','blue'], points_ls = [points_out, points], point_colors=['yellow','red'])

  cx, cy, l, w, a = box_out[0]
  box_2d = np.array([[cx,cy, l,w, a]])
  return box_2d, wall_id

