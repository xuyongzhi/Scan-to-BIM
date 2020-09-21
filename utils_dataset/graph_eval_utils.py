import pickle
from configs.common import DIM_PARSE
from obj_geo_utils.line_operations import gen_corners_from_lines_np, get_lineIdsPerCor_from_corIdsPerLine
from obj_geo_utils.obj_utils import GraphUtils, OBJ_REPS_PARSE, find_wall_wall_connection
from obj_geo_utils.geometry_utils import get_cf_from_wall
from collections import defaultdict
import os
import numpy as np
from tools.visual_utils import _show_objs_ls_points_ls, _draw_objs_ls_points_ls,\
  _show_3d_points_objs_ls, _show_3d_bboxes_ids, show_connectivity, show_1by1
from utils_dataset.stanford3d_utils.post_processing import align_bboxes_with_wall
from obj_geo_utils.topology_utils import optimize_walls_by_rooms_main, draw_walls_rooms_rel, \
      _show_2dlines_as_3d
from obj_geo_utils.geometry_utils import points_to_oriented_bbox, get_rooms_from_edges, draw_rooms_from_edges, relation_mask_to_ids, rel_ids_to_mask, check_duplicate
from tools.color import COLOR_MAP_3D, ColorList
import torch
import time


SHOW_EACH_CLASS = False
SET_DET_Z_AS_GT = 1
SHOW_3D = 0
DET_MID = 1
MAX_Draw_Num = 100

DEBUG = []
#DEBUG.append('_D_show_gt')
#DEBUG.append('_D_show_det_graph_2D')
DEBUG.append('_D_show_det_graph_3D')

_scene_list = ['Area_5/conferenceRoom_2', 'Area_5/hallway_2', 'Area_5/office_21', 'Area_5/office_39', 'Area_5/office_40', 'Area_5/office_41']
_scene_list = ['OI2dE1xgN090iaEGc0BpEZ']
_scene_list_Good_in_paper = ['1Kc4s2I4OuEriA-vURDlpH', '3Q92imFGVI1hZ5b0sDFFC3', '5pJn2a9zfRzInJ7IwpIzQd', '6rZTHkgA4fppApEyJjujeg', '7AbkTGB2HBQoCKPAGIHaio', '18H6WOCclkJY34-TVuOqX3', 'bXuAcmEeGjQgzZd1nSxHFD', 'LlljJo9Y5n4sMGbXZiIApj']
_scene_list = _scene_list_Good_in_paper
#_scene_list = None

def change_result_rep(results, classes, obj_rep_pred, obj_rep_gt, obj_rep_out='XYZLgWsHA'):
    dim_parse = DIM_PARSE(obj_rep_pred, len(classes)+1)
    dim_parse_out = DIM_PARSE(obj_rep_out, len(classes)+1)
    re_s, re_e = dim_parse.OUT_ORDER['bbox_refine']
    in_s, in_e = dim_parse.OUT_ORDER['bbox_init']

    num_img = len(results)
    for i in range(num_img):
      det_bboxes = results[i]['det_bboxes']
      gt_bboxes = results[i]['gt_bboxes']
      gt_labels = results[i]['gt_labels']
      assert len(gt_bboxes) ==1
      if isinstance(gt_bboxes[0], list):
        assert len(gt_bboxes[0]) ==1
        gt_bboxes = gt_bboxes[0][0]
        gt_labels = gt_labels[0][0]
      elif isinstance(gt_bboxes[0], torch.Tensor):
        gt_bboxes = gt_bboxes[0]
        gt_labels = gt_labels[0]
      else:
        raise NotImplementedError
      gt_bboxes = OBJ_REPS_PARSE.encode_obj(gt_bboxes, obj_rep_gt, obj_rep_out)
      #_show_objs_ls_points_ls( (512,512), [gt_bboxes.cpu().data.numpy()], obj_rep_out )
      num_level = len(det_bboxes)
      for l in range(num_level):
          assert det_bboxes[l].shape[1] == dim_parse.OUT_DIM_FINAL
          #_show_objs_ls_points_ls( (512,512), [det_bboxes[l][:,re_s:re_e]], obj_rep_pred )
          bbox_refine = OBJ_REPS_PARSE.encode_obj(det_bboxes[l][:, re_s:re_e], obj_rep_pred, obj_rep_out)
          bbox_init = OBJ_REPS_PARSE.encode_obj(det_bboxes[l][:, in_s:in_e], obj_rep_pred, obj_rep_out)
          det_bboxes_new = np.concatenate([ bbox_refine, bbox_init, det_bboxes[l][:,in_e:] ], axis=1)
          #_show_objs_ls_points_ls( (512,512), [ bbox_refine ], obj_rep_out )
          if not det_bboxes_new.shape[1] == dim_parse_out.OUT_DIM_FINAL:
            import pdb; pdb.set_trace()  # XXX BREAKPOINT
            pass
          det_bboxes[l] = det_bboxes_new

      results[i]['det_bboxes'] = det_bboxes
      results[i]['gt_bboxes'] = gt_bboxes
      results[i]['gt_labels'] = gt_labels
    return results, obj_rep_out

def save_res_graph(dataset, data_loader, results, out_file, data_test_cfg):
    filter_edges = data_test_cfg['filter_edges']
    classes = data_test_cfg['classes']
    obj_rep_gt = data_test_cfg['obj_rep']
    if 'obj_rep_out' in data_test_cfg:
      obj_rep_pred = data_test_cfg['obj_rep_out']
    else:
      obj_rep_pred = obj_rep_gt
    results, obj_rep = change_result_rep(results, classes, obj_rep_pred, obj_rep_gt, 'XYZLgWsHA')
    dim_parse = DIM_PARSE(obj_rep, len(classes)+1)
    num_imgs = len(results)
    assert len(data_loader) == num_imgs
    results_datas = []
    catid_2_cat = dataset._catid_2_cat
    for i_img, data in enumerate(data_loader):
        #is_pcl = 'input_style' in data['img_meta'][0] and data['img_meta'][0]['input_style'] == 'pcl'
        is_image = data['img_meta'][0].__class__.__name__ == 'DataContainer'

        is_pcl = not is_image
        if not is_pcl:
          #img_i = data['img'][0][0].permute(1,2,0).cpu().data.numpy()
          img_meta_i = data['img_meta'][0].data[0][0]
          mean = img_meta_i['img_norm_cfg']['std']
          std = img_meta_i['img_norm_cfg']['std']

          img_i = results[i_img]['img'][0].permute(1,2,0).cpu().data.numpy()
          img_i = (img_i * std) + mean
        else:
          img_meta_i = data['img_meta'][0]
          img_shape = img_meta_i['dynamic_vox_size_aug'][[1,0,2]]
          img_shape[2] = 3
          img_i = img_shape

          #img_i = np.zeros(img_shape, dtype=np.int8)
        res_data = dict(  img_id = i_img,
                          img = img_i,
                          img_meta = img_meta_i,
                          catid_2_cat = catid_2_cat,
                          filter_edges = filter_edges,
                          obj_rep = obj_rep,
                        )
        #img_id = dataset.img_ids[i_img]
        #assert img_id == i_img
        result = results[i_img]
        det_result = result['det_bboxes']
        num_cat = len(det_result)
        if result['gt_bboxes'] is not None:
            # test mode
          gt_bboxes = result['gt_bboxes']
          gt_labels = result['gt_labels']
          if isinstance(gt_bboxes, list):
            assert len(gt_bboxes) == 1
            gt_bboxes = gt_bboxes[0]
            gt_labels = gt_labels[0]
          if isinstance(gt_bboxes, list):
            assert len(gt_bboxes) == 1
            gt_bboxes = gt_bboxes[0]
            gt_labels = gt_labels[0]
          res_data['gt_bboxes'] = gt_bboxes.cpu().numpy()
          res_data['gt_labels'] = gt_labels.cpu().numpy()
          gt_relations = result['gt_relations'][0][0]
          res_data['gt_relations'] = gt_relations.cpu().numpy()

          #_show_objs_ls_points_ls(img_i, [res_data['gt_bboxes'][0]], obj_rep='XYXYSin2')
          pass

        detections_all_labels = []
        for label in range(1, num_cat+1):
          det_lines_multi_stages = det_result[label-1]
          det_lines = det_lines_multi_stages
          category_id = dataset.cat_ids[label]
          cat = catid_2_cat[category_id]
          assert det_lines.shape[1] == dim_parse.OUT_DIM_FINAL
          detection_bRefine_sAve = dim_parse.clean_bboxes_out(det_lines, stage='final', out_type='bRefine_sAve')
          detection_bInit_sRefine = dim_parse.clean_bboxes_out(det_lines, stage='final', out_type='bInit_sRefine')
          #_show_objs_ls_points_ls(img_i, [detection_bInit_sRefine[:,:-1]], obj_rep='XYZLgWsHA')
          s, e = dim_parse.OUT_ORDER['points_refine']
          points_refine = det_lines[:, s:e]
          s, e = dim_parse.OUT_ORDER['points_init']
          points_init = det_lines[:, s:e]

          if not check_duplicate(detection_bRefine_sAve, obj_rep, 0.3):
            import pdb; pdb.set_trace()  # XXX BREAKPOINT
            pass
          detection_l = {'det_lines': det_lines, 'category_id': category_id, 'cat': cat,
                         'detection_bRefine_sAve': detection_bRefine_sAve,
                         'detection_bInit_sRefine': detection_bInit_sRefine,
                         'points_init': points_init,
                         'points_refine': points_refine,
                         }

          detections_all_labels.append(detection_l)
        res_data['detections'] = detections_all_labels

        # parse det_relations by classes
        if 'det_relations' in result and result['det_relations'] is not None:
          det_relations_0 = result['det_relations']
          res_data['det_relations'] = split_relations(det_relations_0, det_result, catid_2_cat, dataset)

        # optimize
        results_datas.append( res_data )

    base_dir = os.path.dirname(out_file)
    #base_dir = os.path.join(base_dir, f'eval_res')
    if not os.path.exists(base_dir):
      os.makedirs(base_dir)
    out_file = os.path.join( base_dir,  f'detection_{num_imgs}_Imgs.pickle')
    with open(out_file, 'wb') as f:
      pickle.dump(results_datas, f)
      print(f'\nsave: {out_file}')

    eval_graph( out_file )

def split_relations(det_relations_0, det_result, catid_2_cat, dataset):
    det_relations = {}
    num_det_cats = [d.shape[0] for d in det_result]
    assert catid_2_cat[1] == 'wall'
    num_wall = num_det_cats[0]
    s = 0
    num_cat = len(det_result)
    for label in range(1, num_cat+1):
      category_id = dataset.cat_ids[label]
      cat = catid_2_cat[category_id]
      e = s+num_det_cats[label-1]
      rel_0 = det_relations_0[s:e, 0:num_wall]
      rel_1 = det_relations_0[0:num_wall, s:e].T
      dif_01 = np.abs( rel_0 - rel_1 )
      rel = (rel_0 + rel_1) /2
      det_relations[cat] = rel
      s = e
    return det_relations

def post_process_bboxes_1cls(det_lines, score_threshold, label, cat,
              opt_graph_cor_dis_thr, obj_rep, min_out_length, walls=None):
  '''
  The input detection belong to one class
  det_lines: [n,6]
  score_threshold: 0.4
  label: 1
  opt_graph_cor_dis_thr: 10
  obj_rep: 'XYXYSin2'

  det_lines_merged: [m,6]
  '''
  from utils_dataset.stanford3d_utils.post_processing import align_bboxes_with_wall

  t0 = time.time()
  obj_dim = OBJ_REPS_PARSE._obj_dims[obj_rep]
  assert det_lines.shape[1] == obj_dim+1
  det_lines, ids0 = filter_low_score_det(det_lines, score_threshold)
  labels_i = np.ones(det_lines.shape[0], dtype=np.int)*label
  #show_connectivity(walls[:,:-1], det_lines[:,:-1], det_relations[cat], self.obj_rep)
  if cat == 'wall':
      scores_i = det_lines[:,-1]
      det_lines_merged, scores_merged, ids = \
        GraphUtils.optimize_wall_graph(det_lines[:,:obj_dim], scores_i, obj_rep_in=obj_rep,
          opt_graph_cor_dis_thr=opt_graph_cor_dis_thr, min_out_length=min_out_length )

      det_lines_merged = np.concatenate([det_lines_merged, scores_merged.reshape(-1,1)], axis=1)
      m = det_lines_merged.shape[0]
      labels_merged = labels_i[:m]
      ids = ids0[ids]
  elif walls is not None:
    ids = ids0
    if cat in ['door', 'window']:
      det_lines_merged = align_bboxes_with_wall(det_lines, walls, cat, obj_rep)
    else:
      det_lines_merged = det_lines
    labels_merged = labels_i
  else:
    det_lines_merged = det_lines
    labels_merged = labels_i
    ids = ids0
  t1 = time.time()
  t = t1 - t0
  return det_lines_merged, labels_merged, ids, t


def _optimize_wall_by_room(walls, rooms, obj_rep ):
  score_th = 0.5
  #_show_objs_ls_points_ls( (512,512), [walls[:,:-1], rooms[:,:-1]], obj_rep, obj_colors=['red', 'white'], obj_thickness=[3,1] )
  rooms = sort_rooms(rooms, obj_rep)
  num_rooms = rooms.shape[0]
  num_walls = walls.shape[0]
  walls_aug = walls.copy()
  walls_aug[:,4] = 3
  room_ids_per_wall = np.zeros([num_walls, 2], dtype=np.int32) - 1
  room_nums_per_wall = np.zeros([num_walls], dtype=np.int32)
  candi_w_mask = room_nums_per_wall != 2
  for i in range(num_rooms):
    room_i_raw = rooms[i:i+1,:-1].copy()
    room_i_aug = room_i_raw.copy()
    room_i_aug[:,3:5] += 15
    candi_ids = np.where(candi_w_mask)[0]
    w_in_r_scores = cal_edge_in_room_scores(walls_aug[candi_ids,:-1], room_i_aug, obj_rep)
    mask = w_in_r_scores > score_th
    inside_ids = np.where(mask)[0]
    inside_ids = candi_ids[inside_ids]
    in_scores = w_in_r_scores[mask]
    max_out_score = w_in_r_scores[mask==False].max()

    for j in inside_ids:
      try:
        room_ids_per_wall[j, room_nums_per_wall[j]] = i
      except:
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        pass
    room_nums_per_wall[inside_ids] += 1

    if 1:
      print(f'in_scores: {in_scores}\n max_out_score:{max_out_score:.3f}')
      _show_objs_ls_points_ls( (512,512), [walls_aug[candi_w_mask,:-1], room_i_raw, room_i_aug ], obj_rep, obj_colors=['red', 'white', 'white'], obj_thickness=1 )
      _show_objs_ls_points_ls( (512,512), [walls[inside_ids,:-1], room_i_raw ], obj_rep, obj_colors=['red', 'white'], obj_thickness=[3,1] )
    candi_w_mask = room_nums_per_wall != 2
  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  return walls


def eval_graph(res_file, eval_method='corner'):
  with open(res_file, 'rb') as f:
    results_datas = pickle.load(f)
  img_meta = results_datas[0]['img_meta']
  classes = img_meta['classes']
  if classes == ['room']:
    eval_method = 'iou'
  else:
    eval_method = 'corner'
  filter_edges =  results_datas[0]['filter_edges']
  obj_rep =  results_datas[0]['obj_rep']
  graph_eval = GraphEval(obj_rep, classes, filter_edges, eval_method)
  graph_eval(results_datas, res_file)

def eval_graph_multi_files(res_files):
  with open(res_files[0], 'rb') as f:
    wall_results_datas = pickle.load(f)
  with open(res_files[1], 'rb') as f:
    room_results_datas = pickle.load(f)
  results_datas = merge_two_results(wall_results_datas, room_results_datas)

  if len(res_files) == 3:
    with open(res_files[2], 'rb') as f:
      win_results_datas = pickle.load(f)
    #win_results_datas = filter_res_classes( win_results_datas, 'window' )
    results_datas = merge_two_results(results_datas, win_results_datas, kp_cls_in2='window')
  img_meta = results_datas[0]['img_meta']
  classes = img_meta['classes']
  filter_edges =  results_datas[0]['filter_edges']
  obj_rep =  results_datas[0]['obj_rep']
  graph_eval = GraphEval(obj_rep, classes, filter_edges, 'corner')
  base_dir = os.path.dirname( os.path.dirname(res_files[0]) )
  combine_dir = os.path.join( base_dir, 'combined_test' )
  if not os.path.exists(combine_dir):
    os.makedirs(combine_dir)
  combine_file = os.path.join( combine_dir, 'detection.pickle' )
  graph_eval(results_datas, combine_file)

def filter_res_classes(results_datas, keep_cls):
  for res in results_datas:
    catid_2_cat = res['catid_2_cat']
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    pass

def merge_two_results(results_datas_1, results_datas_2, kp_cls_in2=None):
  results_datas_3 = []
  same_eles = ['img_id', 'img', 'img_meta', 'catid_2_cat', 'filter_edges', 'obj_rep']
  concat_eles = ['gt_bboxes', 'gt_labels']
  for res1, res2 in zip(results_datas_1, results_datas_2):
    res3 = {}
    assert res1['img_id'] == res2['img_id']
    assert res1['img_meta']['filename'] == res2['img_meta']['filename']
    assert res1['obj_rep'] == res2['obj_rep']
    for e in same_eles:
      res3[e] = res1[e]
    classes1 = res1['img_meta']['classes']
    n1 = len(classes1)
    classes2 = res2['img_meta']['classes']

    #if classes2 == ['room']:
    if 1:
      res3['img_meta']['classes'] = classes1 + classes2
      for i, cat in enumerate(classes2):
        res3['catid_2_cat'][n1+1+i] = cat
      res2['gt_labels'] +=n1
    if classes2 == ['room']:
      res3['gt_relations_room_wall'] = res2['gt_relations']
    else:
      res3['gt_relations_room_wall'] = res1['gt_relations_room_wall']

    #elif classes2 == ['door', 'window']:
    #  res3['img_meta']['classes'] = classes1 + ['window']
    #  res3['catid_2_cat'][n1+1] = 'window'
    #  assert res2['catid_2_cat'][2] == 'window'
    #  win_mask = res2['gt_labels'] == 2

    #  res2['gt_labels'] = res2['gt_labels'][win_mask]
    #  res2['gt_bboxes'] = res2['gt_bboxes'][win_mask]
    #  res2['gt_labels'] += n1 - 1
    #  res2['detections'] = [ res2['detections'][1] ]

    #  res3['gt_relations_room_wall'] = res1['gt_relations_room_wall']
    #  pass
    #else:
    #  raise NotImplementedError

    for e in concat_eles:
      res3[e] = np.concatenate( [res1[e], res2[e]], 0 )
    res3['detections'] = res1['detections'] + res2['detections']
    res3['det_relations'] = res1['det_relations']

    results_datas_3.append(res3)
  return results_datas_3


class GraphEval():
  #_all_out_types = [ 'composite', 'bInit_sRefine', 'bRefine_sAve' ]

  _img_ids_debuging = list(range( 2 ))
  _img_ids_debuging = None
  _opti_room = 0


  if 1:
    _all_out_types = [ 'bRefine_sAve' ]
    _opti_graph = [1]
    _opti_by_rel = [1]

    _opti_room = 1

  if 0:
    _all_out_types = [ 'bInit_sRefine' ]
    _opti_graph = [0]
    _opti_by_rel = [0]

  if 0:
    _all_out_types = [ 'bInit_sRefine', 'bRefine_sAve', 'bRefine_sAve' ]
    _opti_graph = [0, 0, 1]

  if _opti_graph[0] == 0:
    assert _opti_room == 0
  _draw_pts = _opti_room != 1

  _score_threshold  = 0.4
  _corner_dis_threshold = 15
  _opt_graph_cor_dis_thr = 15
  _min_out_length = 10

  _eval_img_scale_ratio = 1.0
  _eval_img_size_aug = 0
  _max_ofs_by_rel = 30

  _iou_threshold = 0.7
  del_alone_walls = 1

  scene_list = _scene_list


  def __init__(self, obj_rep, classes, filter_edges, eval_method):
    self.obj_rep = obj_rep
    self.eval_method = eval_method
    self.obj_dim = OBJ_REPS_PARSE._obj_dims[obj_rep]
    self.classes = classes
    self.filter_edges = filter_edges
    self.dim_parse = DIM_PARSE(obj_rep, len(classes)+1)
    pass

  def __str__(self):
    s = self._score_threshold
    par_str =  f'Eval parameters:\n'
    if self.is_pcl:
      par_str += f'input: pcl\n'
    else:
      par_str += f'input: image\n'
    par_str += f'Out type:{self.out_type}\n'
    par_str += f'Graph optimization corner distance threshold:{self._opt_graph_cor_dis_thr}\n'
    par_str += f'Positive score threshold:{s}\n'
    par_str += f'Min out length:{self._min_out_length}\n'
    par_str += f'Positive corner distance threshold:{self._corner_dis_threshold}\n'
    par_str += '\n'
    return par_str

  def update_path(self, out_file):
    self.work_dir = os.path.join( os.path.dirname(out_file), 'eval_res')
    s = int(self._score_threshold*10)
    self.par_nice = f'Sc{s}_mL{self._min_out_length}_{self.out_type}'
    if self.eval_method == 'corner':
      self.par_nice += f'_corD{self._corner_dis_threshold}'
    if self.eval_method == 'iou':
      self.par_nice += f'IoU{self._iou_threshold}'

    if self.optimize_graph:
      self.par_nice += f'-Gh{self._opt_graph_cor_dis_thr}'
    if self.optimize_graph_by_relation:
      self.par_nice += f'-wRel'
    if self._opti_room:
      self.par_nice += '-Room'
    self.eval_dir = os.path.join(self.work_dir, self.par_nice + f'_{self.num_img}Imgs/PerClass')
    self.eval_dir_all_cls = os.path.join(self.work_dir, self.par_nice + f'_{self.num_img}Imgs/AllClasses')

    if not os.path.exists(self.eval_dir):
      os.makedirs(self.eval_dir)
    if not os.path.exists(self.eval_dir_all_cls):
      os.makedirs(self.eval_dir_all_cls)

  def __call__(self, results_datas, out_file):
    self.num_img = len(results_datas)
    if self._img_ids_debuging is not None:
      self.num_img = len(self._img_ids_debuging)
    if self.scene_list is not None:
      self.num_img = len(self.scene_list)
    if self.eval_method == 'corner':
      eval_fn = self.evaluate_by_corner
    if self.eval_method == 'iou':
      eval_fn = self.evaluate_by_iou
    for out_type, opt_g, opt_rel in zip(self._all_out_types, self._opti_graph, self._opti_by_rel):
      self.out_type = out_type
      eval_fn(results_datas, out_file, out_type, opt_g, opt_rel)

  def evaluate_by_corner(self, results_datas, out_file, out_type, optimize_graph=True, optimize_graph_by_relation=False):
    assert self.obj_rep  == 'XYZLgWsHA'
    self.optimize_graph = optimize_graph
    self.optimize_graph_by_relation = optimize_graph_by_relation
    debug = 1

    time_post = 0
    with_rel = 'det_relations' in results_datas[0]
    if optimize_graph_by_relation and not with_rel:
      return
    self.update_path(out_file)
    all_cor_nums_gt_dt_tp = defaultdict(list)
    all_line_nums_gt_dt_tp = defaultdict(list)
    rooms_gt_dt_tp_rel_ls = []
    all_ious = defaultdict(list)
    catid_2_cat = results_datas[0]['catid_2_cat']
    scene_list = []

    self.is_pcl = 'input_style' in results_datas[0]['img_meta'] and results_datas[0]['img_meta']['input_style'] == 'pcl'
    if self.is_pcl:
      self._eval_img_size_aug = 20
      self._eval_img_scale_ratio = 1.5

    for i_img, res_data in enumerate(results_datas):
        if self._img_ids_debuging is not None:
          if i_img not in self._img_ids_debuging:
            continue

        detections = res_data['detections']
        if with_rel:
          det_relations = res_data['det_relations'].copy()
        else:
          det_relations = None
        img_meta = res_data['img_meta']
        is_pcl = 'input_style' in img_meta and img_meta['input_style'] == 'pcl'
        if not is_pcl:
          img = res_data['img']
          p = self._eval_img_size_aug
          if p>0:
            img = np.pad(img, (p,p,p,p,0,0), 'constant', constant_values=0)
        else:
          img_shape = res_data['img']
          img_shape[:2] = img_shape[:2] * self._eval_img_scale_ratio + self._eval_img_size_aug * 2
          img = np.zeros(img_shape, dtype=np.int8)
          pass

        filename =  img_meta['filename']
        scene_name = os.path.splitext(os.path.basename(filename))[0]
        if 'Area_' in filename:
          area_id = filename.split('Area_')[1][0]
          scene_name = scene_name.split('-topview')[0]
          scene_name = 'Area_' + area_id + '/' + scene_name

        if self.scene_list is not None:
          if scene_name not in self.scene_list:
            continue
        scene_list.append(scene_name)

        print(f'\n\n\n\n{i_img}th file: {filename}')

        gt_lines = results_datas[i_img]['gt_bboxes'].copy()
        gt_labels = results_datas[i_img]['gt_labels'].copy()
        if 'gt_relations' in results_datas[i_img]:
          gt_relations = results_datas[i_img]['gt_relations'].copy()
        else:
          gt_relations = None
        if 'gt_relations_room_wall' in results_datas[i_img]:
          gt_relations_room_wall = results_datas[i_img]['gt_relations_room_wall'].copy()
        else:
          gt_relations_room_wall = None
        if gt_lines.ndim == 1:
          gt_lines = gt_lines[None,:]
        gt_lines[:,:4] = gt_lines[:,:4] * self._eval_img_scale_ratio + self._eval_img_size_aug
        pass

        if '_D_show_gt' in DEBUG:
          print('gt')
          _show_objs_ls_points_ls(img, [gt_lines], obj_rep=self.obj_rep)
        pass

        num_labels = len(detections)
        eval_draws_ls = []

        cat_ls, det_lines_merged_ls, gt_lines_ls, det_points_ls = self.geo_opti_per_cls(num_labels, catid_2_cat, gt_labels, gt_lines, detections, out_type, optimize_graph, optimize_graph_by_relation, with_rel, det_relations)
        if self._opti_room:
          wall_ids_per_room = self.wall_room_opti(cat_ls, det_lines_merged_ls)
          if 'room' in cat_ls:
            rooms_gt_dt_tp_rel = self.eval_rooms_with_rel(det_lines_merged_ls, gt_lines_ls, cat_ls, wall_ids_per_room, gt_relations_room_wall)
            rooms_gt_dt_tp_rel_ls.append( rooms_gt_dt_tp_rel )

        if '_D_show_det_graph_3D' in DEBUG:
              _show_2dlines_as_3d([det_lines_merged_ls[di][:,:-1] for di in [0, 2, 3]], obj_rep=self.obj_rep, filename=filename)
        if '_D_show_det_graph_2D' in DEBUG:
              _show_objs_ls_points_ls( img.shape[:2],  [det_lines_merged_ls[0][:,:-1]], obj_rep=self.obj_rep)
              pass

        for i in range(num_labels):
            label = i+1
            cat = cat_ls[i]
            cor_nums_gt_dt_tp, line_nums_gt_dt_tp, eval_draws = self.eval_1img_1cls_by_corner(img, det_lines_merged_ls[i], gt_lines_ls[i], scene_name, cat, det_points_ls[i])
            ious = self.cal_iou_1img_1cls(img, det_lines_merged_ls[i], gt_lines_ls[i], scene_name, cat, det_points_ls[i])
            all_cor_nums_gt_dt_tp[label].append(cor_nums_gt_dt_tp)
            all_line_nums_gt_dt_tp[label].append(line_nums_gt_dt_tp)
            all_ious[label].append(ious)
            if i_img < MAX_Draw_Num:
              eval_draws_ls.append(eval_draws)

            if debug and 0:
              print(f'optimize graph with self._opt_graph_cor_dis_thr= {self._opt_graph_cor_dis_thr}')
              _show_objs_ls_points_ls(img.shape[:2], [det_lines_merged[:,:-1], det_lines[:,:-1]], obj_colors=['green','red'], obj_rep=self.obj_rep, obj_thickness=[4,2])
              _show_objs_ls_points_ls(img.shape[:2], [det_lines_merged[:,:-1], gt_lines_l], obj_colors=['green','red'], obj_rep=self.obj_rep, obj_thickness=[4,2])

            pass
        if i_img < MAX_Draw_Num:
          draw_eval_all_classes_1_scene(eval_draws_ls, self.obj_rep, self._draw_pts)
        pass

    corner_recall_precision_perimg = defaultdict(list)
    line_recall_precision_perimg = defaultdict(list)

    corner_recall_precision = {}
    line_recall_precision = {}
    cor_nums_sum = {}
    line_nums_sum = {}
    ave_ious = {}
    iou_thres = 0.3

    for label in all_cor_nums_gt_dt_tp:
      cat =  catid_2_cat[label]
      cor_nums = np.array(all_cor_nums_gt_dt_tp[label])
      line_nums = np.array(all_line_nums_gt_dt_tp[label])
      corner_recall_precision_perimg[cat].append( (cor_nums[:,2] / cor_nums[:,0], cor_nums[:,2] / cor_nums[:,1] ))
      line_recall_precision_perimg[cat].append( (line_nums[:,2] / line_nums[:,0], line_nums[:,2] / line_nums[:,1] ))

      cor = cor_nums.sum(axis=0)
      line = line_nums.sum(axis=0)

      corner_recall_precision[cat] = [cor[2]/cor[0], cor[2]/cor[1]]
      line_recall_precision[cat] = [line[2]/line[0], line[2]/line[1]]
      cor_nums_sum[cat] = cor
      line_nums_sum[cat] = line

      # cal iou
      ious_l = np.concatenate(all_ious[label])
      ave_iou = ious_l[ious_l > iou_thres].mean()
      ave_ious[cat] = ave_iou
      pass

    if len(rooms_gt_dt_tp_rel_ls) > 0:
      all_rooms_gt_dt_tp_rel = np.array(rooms_gt_dt_tp_rel_ls).sum(0)
      iou_rec = all_rooms_gt_dt_tp_rel[2]/  all_rooms_gt_dt_tp_rel[0]
      iou_prec = all_rooms_gt_dt_tp_rel[2]/  all_rooms_gt_dt_tp_rel[1]
      rel_rec = all_rooms_gt_dt_tp_rel[3]/  all_rooms_gt_dt_tp_rel[0]
      rel_prec = all_rooms_gt_dt_tp_rel[3]/  all_rooms_gt_dt_tp_rel[1]
      line_recall_precision['room_iou'] = [iou_rec, iou_prec]
      line_recall_precision['room_rel'] = [rel_rec, rel_prec]

    eval_res_str = self.get_eval_res_str(corner_recall_precision, line_recall_precision, img_meta, line_nums_sum, cor_nums_sum, ave_ious, time_post)
    path = os.path.dirname(out_file)
    path = os.path.join(path, 'eval_res')
    if not os.path.exists(path):
      os.makedirs(path)
    num_cat = len(catid_2_cat)-1
    eval_path = os.path.join(path, f'_eval_res_{num_cat}_cats.txt')
    with open(eval_path, 'a') as f:
      f.write(eval_res_str)
    #print(eval_res_str)


    acc_per_img_str = ''
    ns = len(scene_list)
    for i in range( min( ns, MAX_Draw_Num * 2 ) ):
      s  = scene_list[i]
      acc_per_img_str += f'\n{s} :\t'
      for cat in corner_recall_precision_perimg.keys():
        cor_rec = corner_recall_precision_perimg[cat][0][0][i]
        cor_pre = corner_recall_precision_perimg[cat][0][1][i]
        line_rec = line_recall_precision_perimg[cat][0][0][i]
        line_pre = line_recall_precision_perimg[cat][0][1][i]
        acc_per_img_str += f'{cat}: cor= {cor_pre:.3f}-{cor_rec:.3f}, line= {line_pre:.3f}-{line_rec:.3f}'
        pass

    eval_path = os.path.join(os.path.dirname(self.eval_dir_all_cls), 'per_scene_eval_res.txt')
    with open(eval_path, 'a') as f:
      f.write(acc_per_img_str)


    # save eval res
    eval_res = dict( corner_recall_precision = corner_recall_precision,
                    line_recall_precision = line_recall_precision,
                    corner_recall_precision_perimg = corner_recall_precision_perimg,
                    line_recall_precision_perimg = line_recall_precision_perimg,
                    )
    s = 'OptimizeGraph' if self.optimize_graph else 'NoOptmizeGraph'
    eval_res_file = out_file.replace('.pickle', f'_EvalRes{s}.npy')
    #np.save(eval_res_file, eval_res)
    return eval_res_str

  def geo_opti_per_cls(self, num_labels, catid_2_cat, gt_labels, gt_lines, detections, out_type, optimize_graph, optimize_graph_by_relation, with_rel, det_relations):
        debug = 0
        walls = None
        time_post = 0
        det_lines_merged_ls = []
        gt_lines_ls = []
        det_points_ls = []
        cat_ls = []
        for label in range(1, num_labels+1):
            cat = catid_2_cat[label]
            cat_ls.append(cat)
            label_mask = (gt_labels == label).reshape(-1)
            gt_lines_l = gt_lines[label_mask]
            det_lines = detections[label-1][f'detection_{out_type}'].copy()

            if out_type == 'bInit_sRefine':
              det_points = detections[label-1]['points_init']
            elif out_type == 'bRefine_sAve':
              det_points = detections[label-1]['points_refine']

            det_lines[:,:2] = det_lines[:,:2] * self._eval_img_scale_ratio + self._eval_img_size_aug
            det_lines[:,3] = det_lines[:,3] * self._eval_img_scale_ratio


            if optimize_graph:
              det_lines_merged, _, ids_merged, t = post_process_bboxes_1cls(det_lines,
                  self._score_threshold, label, cat, self._opt_graph_cor_dis_thr,
                  self.dim_parse.OBJ_REP, self._min_out_length, walls=walls,)
              time_post += t

              if cat == 'wall' and with_rel:
                  for c in det_relations:
                    if c == 'wall':
                      det_relations[c] = det_relations[c][ids_merged, :][:, ids_merged]
                    else:
                      det_relations[c] = det_relations[c][:, ids_merged]
                  if optimize_graph_by_relation:
                    det_lines_merged = GraphUtils.optimize_walls_by_relation(det_lines_merged,
                              det_relations[cat], self._max_ofs_by_rel, self.obj_rep, self.eval_dir_all_cls, scene_name='')
                    det_lines_merged, _, ids_merged, t = post_process_bboxes_1cls(det_lines_merged,
                      self._score_threshold, label, cat, self._opt_graph_cor_dis_thr,
                      self.dim_parse.OBJ_REP, self._min_out_length, walls=walls,)

            else:
              det_lines_merged, ids_high_scores = filter_low_score_det(det_lines, self._score_threshold)
            if cat == 'wall':
              walls = det_lines_merged
            if not check_duplicate(det_lines_merged, self.obj_rep, 0.4):
              import pdb; pdb.set_trace()  # XXX BREAKPOINT
              pass
            #show_connectivity(walls[:,:-1], det_lines_merged[:,:-1], det_relations[cat], self.obj_rep)
            det_lines_merged_ls.append(det_lines_merged)
            gt_lines_ls.append( gt_lines_l )
            det_points_ls.append( det_points )

            if debug and 1:
              print('raw prediction')
              _show_objs_ls_points_ls((512,512), [det_lines[:,:-1], gt_lines_l], obj_colors=['green','red'], obj_rep=self.obj_rep)
              #check_duplicate( det_lines, self.obj_rep )
              pass

        return cat_ls, det_lines_merged_ls, gt_lines_ls, det_points_ls

  def wall_room_opti(self, cat_ls, det_lines_merged_ls):
    if not ('wall' in cat_ls and 'room' in cat_ls):
      return
    n = len(cat_ls)
    wall_i = [i for i in range(n) if cat_ls[i]=='wall'][0]
    room_i = [i for i in range(n) if cat_ls[i]=='room'][0]
    walls_org = det_lines_merged_ls[wall_i]
    rooms_org = det_lines_merged_ls[room_i]

    if not check_duplicate(walls_org, self.obj_rep, 0.3):
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass

    new_walls = optimize_walls_by_rooms_main( walls_org, rooms_org, self.obj_rep )
    wall_ids_per_room,  room_ids_per_wall ,_, room_bboxes = get_rooms_from_edges(new_walls[:,:7], self.obj_rep, gen_bbox=True)
    scores_r = room_bboxes[:,0:1].copy()
    scores_r[:] = 1
    room_bboxes = np.concatenate([room_bboxes, scores_r], 1)

    if not check_duplicate(new_walls, self.obj_rep, 0.3):
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass

    # delete alone walls
    if self.del_alone_walls:
      mask = (room_ids_per_wall>=0).any(1)
      valid_ids = np.where(mask)[0]
      new_walls = new_walls[valid_ids]
      det_lines_merged_ls[wall_i] = new_walls
      nw = len(mask)
      ids_map = np.zeros([nw], dtype=np.int32)-1
      ids_map[ valid_ids ] = np.arange( len(valid_ids) )
      wall_ids_per_room = [ ids_map[ids] for ids in wall_ids_per_room ]

    det_lines_merged_ls[room_i] = room_bboxes
    det_lines_merged_ls[wall_i] = new_walls
    return wall_ids_per_room

  def evaluate_by_iou(self, results_datas, out_file, out_type, optimize_graph=True, optimize_graph_by_relation=False):
    assert self.obj_rep  == 'XYZLgWsHA'
    self.optimize_graph = optimize_graph
    self.optimize_graph_by_relation = optimize_graph_by_relation
    debug = 0

    self.update_path(out_file)
    all_line_nums_gt_dt_tp = defaultdict(list)
    all_ious = defaultdict(list)
    catid_2_cat = results_datas[0]['catid_2_cat']

    self.is_pcl = 'input_style' in results_datas[0]['img_meta'] and results_datas[0]['img_meta']['input_style'] == 'pcl'
    if self.is_pcl:
      self._eval_img_size_aug = 20
      self._eval_img_scale_ratio = 1.5

    for i_img, res_data in enumerate(results_datas):
        if self._img_ids_debuging is not None:
          if i_img not in self._img_ids_debuging:
            continue

        detections = res_data['detections']
        img_meta = res_data['img_meta']
        is_pcl = 'input_style' in img_meta and img_meta['input_style'] == 'pcl'
        if not is_pcl:
          img = res_data['img']
          p = self._eval_img_size_aug
          if p>0:
            img = np.pad(img, (p,p,p,p,0,0), 'constant', constant_values=0)
        else:
          img_shape = res_data['img']
          img_shape[:2] = img_shape[:2] * self._eval_img_scale_ratio + self._eval_img_size_aug * 2
          img = np.zeros(img_shape, dtype=np.int8)
          pass

        filename =  img_meta['filename']
        scene_name = os.path.splitext(os.path.basename(filename))[0]
        if 'Area_' in filename:
          area_id = filename.split('Area_')[1][0]
          scene_name = scene_name.split('-topview')[0]
          scene_name = 'Area_' + area_id + '/' + scene_name

        if self.scene_list is not None:
          if scene_name not in self.scene_list:
            continue

        print(f'\n\n\n\n{i_img}th file: {filename}')

        gt_lines = results_datas[i_img]['gt_bboxes'].copy()
        gt_labels = results_datas[i_img]['gt_labels'].copy()
        if gt_lines.ndim == 1:
          gt_lines = gt_lines[None,:]
        gt_lines[:,:4] = gt_lines[:,:4] * self._eval_img_scale_ratio + self._eval_img_size_aug
        pass

        if debug and 0:
          print('gt')
          _show_objs_ls_points_ls(img, [gt_lines], obj_rep=self.obj_rep)
        pass

        num_labels = len(detections)
        eval_draws_ls = []
        walls = None
        time_post = 0
        dets_1s = {}
        gts_1s = {}
        det_points_1s = {}
        ious_1s = {}
        labels_to_cats = {}
        for label in range(1, num_labels+1):
            cat = catid_2_cat[label]
            if cat in ['ceiling', 'floor']:
              continue
            labels_to_cats[label] = cat
            label_mask = (gt_labels == label).reshape(-1)
            gt_lines_l = gt_lines[label_mask]
            det_lines = detections[label-1][f'detection_{out_type}'].copy()

            gts_1s[label] = gt_lines_l

            if out_type == 'bInit_sRefine':
              det_points = detections[label-1]['points_init']
            elif out_type == 'bRefine_sAve':
              det_points = detections[label-1]['points_refine']

            det_lines[:,:2] = det_lines[:,:2] * self._eval_img_scale_ratio + self._eval_img_size_aug
            det_lines[:,3] = det_lines[:,3] * self._eval_img_scale_ratio
            if optimize_graph:
              det_lines_merged, _, ids_merged, t = post_process_bboxes_1cls(det_lines,
                  self._score_threshold, label, cat, self._opt_graph_cor_dis_thr,
                  self.dim_parse.OBJ_REP, self._min_out_length, walls=walls)
              time_post += t
            else:
              det_lines_merged, ids_merged = filter_low_score_det(det_lines, self._score_threshold)
            if cat == 'wall':
              walls = det_lines_merged
            dets_1s[label] = det_lines_merged

            det_points_1s[label] = det_points[ids_merged]

            if debug and 1:
              print('raw prediction')
              _show_objs_ls_points_ls(img[:,:,0], [det_lines[:,:-1], gt_lines_l], obj_rep=self.obj_rep, obj_colors=['green','red'])


            det_category_id = detections[label-1]['category_id']
            if det_category_id != 1:
              pass
              #raise NotImplementedError
            line_nums_gt_dt_tp, ious = self.eval_1img_1cls_by_iou(img, det_lines_merged, gt_lines_l, scene_name, cat, det_points)
            all_line_nums_gt_dt_tp[label].append(line_nums_gt_dt_tp)
            if ious.shape[0] > 0:
              ious_of_dets = ious.max(1)
            else:
              ious_of_dets = ious
            all_ious[label].append( ious_of_dets )
            ious_1s[label] = ious
            #eval_draws_ls.append(eval_draws)

            if debug or 0:
              print(f'optimize graph with self._opt_graph_cor_dis_thr= {self._opt_graph_cor_dis_thr}')
              _show_objs_ls_points_ls(img[:,:,0], [det_lines[:,:5], gt_lines_l], obj_colors=['green','red'])
              _show_objs_ls_points_ls(img[:,:,0], [det_lines_merged[:,:5], gt_lines_l], obj_colors=['green','red'])

            pass
        res_filename = os.path.join( self.eval_dir_all_cls, scene_name)
        if i_img < MAX_Draw_Num:
          draw_1_scene(img, gts_1s, dets_1s,  ious_1s, labels_to_cats, self.obj_rep, self._iou_threshold, res_filename, det_points_1s)
        pass

    line_recall_precision_perimg = defaultdict(list)

    line_recall_precision = {}
    line_nums_sum = {}
    ave_ious = {}
    iou_thres = 0.3

    for label in all_line_nums_gt_dt_tp:
      cat =  catid_2_cat[label]
      line_nums = np.array(all_line_nums_gt_dt_tp[label])
      line_recall_precision_perimg[cat].append( (line_nums[:,2] / line_nums[:,0], line_nums[:,2] / line_nums[:,1] ))

      line = line_nums.sum(axis=0)

      line_recall_precision[cat] = [line[2]/line[0], line[2]/line[1]]
      line_nums_sum[cat] = line

      # cal iou
      ious_l = np.concatenate(all_ious[label])
      ave_iou = ious_l.mean()
      #ave_iou = ious_l[ious_l > iou_thres].mean()
      ave_ious[cat] = ave_iou


    eval_res_str = self.get_eval_res_str_iou(line_recall_precision, img_meta, line_nums_sum, ave_ious)
    path = os.path.dirname(out_file)
    #path = os.path.join(path, 'eval_res')
    if not os.path.exists(path):
      os.makedirs(path)
    eval_path = os.path.join(path, 'eval_res.txt')
    with open(eval_path, 'a') as f:
      f.write(eval_res_str)
    #print(eval_res_str)
    #print(f'post time: {time_post}')

    # save eval res
    eval_res = dict(
                    line_recall_precision = line_recall_precision,
                    line_recall_precision_perimg = line_recall_precision_perimg,
                    all_ious = all_ious,
                    )
    s = 'OptimizeGraph' if self.optimize_graph else 'NoOptmizeGraph'
    eval_res_file = out_file.replace('.pickle', f'_EvalRes{s}.npy')
    #np.save(eval_res_file, eval_res)
    return eval_res_str

  def eval_rooms_with_rel(self, dets_ls, gts_ls, cat_ls, det_wall_ids_per_room, gt_relations_room_wall):
    from mmdet.core.bbox.geometry import dsiou_rotated_3d_bbox_np
    show_gt_dt_compare = 0
    show_in_relations = 0
    show_fail_rooms = 0
    show_per_room = 0


    num_cats = len(cat_ls)
    cats_to_label = {cat_ls[i]: i for i in range(num_cats)}
    gt_rooms = gts_ls[cats_to_label['room']]
    gt_walls = gts_ls[cats_to_label['wall']]
    dt_rooms = dets_ls[cats_to_label['room']]
    dt_walls = dets_ls[cats_to_label['wall']]
    num_dt_w = dt_walls.shape[0]


    #dt_rooms = dt_rooms0[dt_rooms0[:,-1] > self._score_threshold][:,:7]
    #dt_walls = dt_walls0[dt_walls0[:,-1] > self._score_threshold][:,:7]
    num_gt_r = gt_rooms.shape[0]
    num_gt_w = gt_walls.shape[0]
    num_dt_r = dt_rooms.shape[0]
    assert gt_relations_room_wall.shape == ( num_gt_r, num_gt_w)

    if show_in_relations:
      det_relations = rel_ids_to_mask(det_wall_ids_per_room, num_dt_w)
      show_connectivity( gt_walls, gt_rooms, gt_relations_room_wall, self.obj_rep)
      show_connectivity( dt_walls[:,:7], dt_rooms[:,:7], det_relations, self.obj_rep)

    ious = dsiou_rotated_3d_bbox_np( gt_rooms, dt_rooms[:,:7], iou_w=1, size_rate_thres=None )
    dt_id_per_gt = ious.argmax(1)
    iou_per_gt = ious.max(1)
    gt_true_mask = iou_per_gt >= self._iou_threshold

    gt_true_ids = np.where(gt_true_mask)[0]
    dt_pos_ids = dt_id_per_gt[gt_true_mask]
    num_tp = gt_true_ids.shape[0]

    room_nums_gt_dt_tp = [num_gt_r, num_dt_r, num_tp]

    # analye walls of per room
    gt_wids_per_room = relation_mask_to_ids(gt_relations_room_wall)

    if show_gt_dt_compare:
      _show_objs_ls_points_ls( (512,512), [gt_walls, dt_walls[:,:7]],
                  self.obj_rep, obj_colors=['white', 'red', ], obj_thickness = [6,1] )

    if show_fail_rooms:
      gt_false_ids = np.where(gt_true_mask==False)[0]
      fail_wids = [gt_wids_per_room[i] for i in  gt_false_ids]
      fail_wids = np.concatenate(fail_wids)
      _show_objs_ls_points_ls( (512,512), [gt_walls, gt_walls[fail_wids], gt_rooms[gt_false_ids]],
                              self.obj_rep, obj_colors=['white', 'red', 'lime' ] )
      _show_objs_ls_points_ls( (512,512), [dt_walls[:,:7], ],
                              self.obj_rep, obj_colors=['lime' ] )
      pass

    succ_room_ids = []
    fail_room_ids = []
    for i in range(num_tp):
      gt_i = gt_true_ids[i]
      dt_i = dt_id_per_gt[gt_i]
      wids_gt_i = gt_wids_per_room[gt_i]
      wids_dt_i = det_wall_ids_per_room[dt_i]
      gtws_i = gt_walls[wids_gt_i]
      dtws_i = dt_walls[wids_dt_i]
      gtn = gtws_i.shape[0]
      dtn = dtws_i.shape[0]
      if gtn == dtn:
        ious_i = dsiou_rotated_3d_bbox_np(gtws_i, dtws_i[:,:7], 0.7, size_rate_thres=0.3).max(0)
        miou = ious_i.mean()
        if miou > 0.7:
          succ_room_ids.append( gt_i )
          if show_per_room:
            print(f'success wall rel')
        else:
          fail_room_ids.append( gt_i )
          if show_per_room:
            print(f'fail wall rel')
        if show_per_room:
          print(f'{i} ious: {ious_i}, {miou:.3f}')

        #ni = gtws_i.shape[0]
        #dtws_i = OBJ_REPS_PARSE.encode_obj(dtws_i[:,:7], self.obj_rep, 'RoLine2D_2p').reshape(1,ni, 2,1, 2)
        #gtws_i = OBJ_REPS_PARSE.encode_obj(gtws_i, self.obj_rep, 'RoLine2D_2p').reshape(ni,1, 1,2, 2)
        #dif_i = dtws_i - gtws_i
        #dif_i = np.linalg.norm(dif_i, axis=-1)
        #dif_i = dif_i.max(-1).min(-1)
      else:
        if show_per_room:
          print(f'fail room')
          print(f'{i} gtn:{gtn}, dtn:{dtn}, iou:{iou_per_gt[gt_i]:.3f}')
        pass
      if show_per_room:
        #_show_objs_ls_points_ls( (512,512), [gt_rooms[gt_i,None], dt_rooms[dt_i,None][:,:7]], self.obj_rep, obj_colors=['red','lime'] )
        _show_objs_ls_points_ls( (512,512), [gt_walls, gtws_i, dtws_i[:,:7]], self.obj_rep, obj_colors=['white', 'red','lime'], obj_thickness=[1,2,2] )
        #show_1by1((512,512), gtws_i, self.obj_rep, gt_walls)
        #show_1by1((512,512), dtws_i[:,:7], self.obj_rep, dt_walls[:,:7])
        print(f'\n')
      pass

    num_rel_tp = len(succ_room_ids)
    rooms_gt_dt_tp_rel = room_nums_gt_dt_tp + [num_rel_tp]
    return rooms_gt_dt_tp_rel

  def get_eval_res_str_iou(self, line_recall_precision, img_meta, line_nums_sum, ave_ious ):
    rotate = False
    eval_str = '\n\n--------------------------------------\n\n' + \
                str(self) + f'num_img: {self.num_img}\n'
    eval_str += f'optimize_graph: {self.optimize_graph}\n'
    eval_str += f'IoU threshold: {self._iou_threshold}\n'

    eval_str += 'Precision-Recall\n\n'
    cats = line_recall_precision.keys()
    eval_str += '| split |'
    for cat in cats:
      str_e = f'{cat} edge '
      eval_str += f'{str_e:14}|'
    eval_str += '\n|-|'
    for cat in cats:
      eval_str += '-|-|'

    eval_str += '\n|pre-rec|'
    for cat in cats:
      line_rec, line_prec = line_recall_precision[cat]
      line_str = f'{line_prec:.3} - {line_rec:.3}'
      eval_str += f'{line_str:14}|'
      pass
    eval_str += '\n'

    eval_str += '| iou   |'
    for cat in cats:
      iou = ave_ious[cat]
      iou_str = f'{iou:.3}'
      s=''
      eval_str += f'{iou_str:14}|'
      pass
    eval_str += '\n'

    eval_str += '|gt num |'
    for cat in cats:
      line_num = line_nums_sum[cat][0]
      eval_str += f'{line_num:14}|'
      pass
    eval_str += '\n'

    return eval_str

  def get_eval_res_str(self, corner_recall_precision, line_recall_precision, img_meta, line_nums_sum, cor_nums_sum, ave_ious, time_post):
    rotate = False
    eval_str = '\n\n--------------------------------------\n\n' + \
                str(self) + f'num_img: {self.num_img}\n'
    eval_str += f'optimize graph geometrically: {self.optimize_graph}\n'
    eval_str += f'optimize graph semantically: {self.optimize_graph_by_relation}\n'
    eval_str += f'optimize walls by rooms: {self._opti_room}\n'

    eval_str += 'Precision-Recall\n\n'
    cats = line_recall_precision.keys()
    eval_str += '| split |'
    for cat in cats:
      str_c = f'{cat} corner'
      str_e = f'{cat} edge '
      eval_str += f'{str_c:14}|{str_e:14}|'
    eval_str += '\n|-|'
    for cat in cats:
      eval_str += '-|-|'

    eval_str += '\n|pre-rec|'
    for cat in cats:
      if cat in corner_recall_precision:
        cor_rec, cor_prec = corner_recall_precision[cat]
        cor_str = f'{cor_prec:.3} - {cor_rec:.3}'
      else:
        cor_str = ''

      line_rec, line_prec = line_recall_precision[cat]
      line_str = f'{line_prec:.3} - {line_rec:.3}'
      eval_str += f'{cor_str:14}|{line_str:14}|'
      pass
    eval_str += '\n'

    eval_str += '| iou   |'
    for cat in corner_recall_precision:
      iou = ave_ious[cat]
      iou_str = f'{iou:.3}'
      s=''
      eval_str += f'{s:14}|{iou_str:14}|'
      pass
    eval_str += '\n'

    eval_str += '|gt num |'
    for cat in cats:
      if cat in cor_nums_sum:
        cor_num = cor_nums_sum[cat][0]
        line_num = line_nums_sum[cat][0]
      else:
        cor_num = 0
        line_num = 0
      eval_str += f'{cor_num:14}|{line_num:14}|'
      pass
    eval_str += '\n'
    eval_str += f'post time: {time_post}'
    eval_str += '\n'

    return eval_str

  def eval_1img_1cls_by_iou(self, img, det_lines, gt_lines, scene_name, cat, det_points):
    from mmdet.core.bbox.geometry import dsiou_rotated_3d_bbox_np
    num_det = det_lines.shape[0]
    num_gt = gt_lines.shape[0]
    if num_det == 0:
      return [num_gt, num_det, 0], np.array([])
    # [num_det, num_gt]
    iou_matrix = dsiou_rotated_3d_bbox_np(det_lines[:,:-1], gt_lines, iou_w=1, size_rate_thres=0.07)
    ious = iou_matrix.max(0)
    mask = ious > self._iou_threshold
    num_tp = sum(mask)
    obj_nums_gt_dt_tp = [num_gt, num_det, num_tp]
    return obj_nums_gt_dt_tp, iou_matrix

  def cal_iou_1img_1cls(self, img, det_lines, gt_lines, scene_name, det_cat, det_points):
    from mmdet.core.bbox.geometry import dsiou_rotated_3d_bbox_np
    if det_lines.shape[0] == 0:
      return np.array([])
    iou_matrix = dsiou_rotated_3d_bbox_np(det_lines[:,:7], gt_lines, iou_w=1, size_rate_thres=0.07)
    ious = iou_matrix.max(1)
    return ious

  def eval_1img_1cls_by_corner(self, img, det_lines, gt_lines, scene_name, det_cat, det_points):
    show_missed_gt = 0
    show_all_matching = 0

    num_gt = gt_lines.shape[0]
    det_corners, cor_scores, det_cor_ids_per_line,_ = gen_corners_from_lines_np(det_lines[:,:self.obj_dim],\
                                          None, self.obj_rep, self._opt_graph_cor_dis_thr//2)
    gt_corners, _, gt_corIds_per_line,_ = gen_corners_from_lines_np(gt_lines, None, self.obj_rep, self._opt_graph_cor_dis_thr//2)

    cor_nums_gt_dt_tp, cor_detIds_per_gt = self.eval_corners(gt_corners, det_corners)

    # cal det_lineIds_per_cor: [num_det_corner]
    det_lineIds_per_cor = get_lineIdsPerCor_from_corIdsPerLine(det_cor_ids_per_line, det_corners.shape[0])

    # detCorIds_per_gtLine: [num_gt_line, 2]
    # line_detIds_per_gt: [num_gt_line]
    detCorIds_per_gtLine = cor_detIds_per_gt[ gt_corIds_per_line ]
    line_detIds_per_gt = []
    for i in range(detCorIds_per_gtLine.shape[0]):
      a,b = detCorIds_per_gtLine[i]
      det_lineIds = -1
      if a>=0 and b>=0:
        lineIds_a = det_lineIds_per_cor[a]
        lineIds_b = det_lineIds_per_cor[b]
        # find if the two corners map the same line
        for ai in lineIds_a:
          for bi in lineIds_b:
            if ai == bi:
              det_lineIds = ai

        if show_missed_gt and det_lineIds == -1:
          print(f'A gt match two det corners, but no det line')
          det_ids = lineIds_a + lineIds_b
          _show_objs_ls_points_ls(img[:,:,0], [det_lines[det_ids,:-1], gt_lines[i:i+1]], obj_colors=['red', 'lime'], obj_rep=self.obj_rep, obj_thickness=[2,1])
          import pdb; pdb.set_trace()  # XXX BREAKPOINT
          pass
      if det_lineIds != -1 and det_lineIds in line_detIds_per_gt:
        print(f'a det match two gts')
        #_show_objs_ls_points_ls(img[:,:,0], [det_lines[[det_lineIds]][:,:-1], gt_lines[i:i+1]], obj_colors=['red', 'lime'], obj_rep=self.obj_rep, obj_thickness=[2,1])
        det_lineIds = -1
      line_detIds_per_gt.append(det_lineIds)
    line_detIds_per_gt = np.array(line_detIds_per_gt)

    num_ture_pos_line = (line_detIds_per_gt>=0).sum()
    line_nums_gt_dt_tp = [gt_lines.shape[0], det_lines.shape[0], num_ture_pos_line]

    if show_all_matching:
      #_show_objs_ls_points_ls(img[:,:,0], [det_lines[:,:-1], gt_lines], obj_colors=['random', 'white'], obj_rep=self.obj_rep, obj_thickness=[3,1])
      n = gt_lines.shape[0]
      for i in range(n):
        j = line_detIds_per_gt[i]
        if j>=0:
          _show_objs_ls_points_ls(img[:,:,0], [det_lines[j:j+1,:-1], gt_lines[i:i+1], gt_lines], obj_colors=['red', 'lime', 'white'], obj_rep=self.obj_rep, obj_thickness=[8, 4, 1])
        else:
          _show_objs_ls_points_ls(img[:,:,0], [gt_lines[i:i+1], det_lines[:,:-1], gt_lines], obj_colors=['red', 'lime', 'white'], obj_rep=self.obj_rep, obj_thickness=[8, 4, 1])
          import pdb; pdb.set_trace()  # XXX BREAKPOINT
          pass
        pass


    if 1:
      #det_lines_pos, det_lines_neg, det_corners_pos, det_corners_neg, gt_lines_true, gt_lines_false
      eval_draws = self.save_eval_res_img_1cls(img, det_lines, gt_lines, det_corners, gt_corners,
                            cor_detIds_per_gt, line_detIds_per_gt,
                            cor_nums_gt_dt_tp, scene_name, det_cat, det_points)
      #self.debug_line_eval(det_lines, gt_lines, line_detIds_per_gt)
    return cor_nums_gt_dt_tp, line_nums_gt_dt_tp, eval_draws

  def eval_corners(self, gt_corners, det_corners):
    '''
    gt_corners: [n,2]
    det_corners: [m,2]
    cor_scores: [m]

    A gt corner gt_i is successfully detected by det_j, when both of follownig two matches:
      1. det_j is the cloest to gt_i, and the distance is below corner_dis_threshold
      2. gt_i is the cloest to det_j

    nums_gt_dt_tp: [3]
    detIds_per_gt_2: [n]
    '''
    if det_corners.shape[0]==0:
      gt_num = gt_corners.shape[0]
      return [gt_num, 0, 0], -np.ones(gt_num, dtype=np.int)

    diss = np.linalg.norm(gt_corners[:,None,:] - det_corners[None,:,:], axis=2)
    detIds_per_gt_0 = np.argmin(diss, axis=1)
    mindis_per_gt = diss.min(axis=1)
    dis_valid_mask_gt = (mindis_per_gt < self._corner_dis_threshold).astype(np.float32)
    detIds_per_gt_1 = apply_mask_on_ids(detIds_per_gt_0, dis_valid_mask_gt).astype(np.int32)

    # check if some gts are matched by multiple detections
    #mindis_per_det = diss.min(axis=0)
    if diss.shape[0]>0:
      gt_ids_per_det = np.argmin(diss, axis=0)
    else:
      gt_ids_per_det = np.zeros([0], dtype=np.int)
    gt_ids_check = apply_mask_on_ids(gt_ids_per_det[detIds_per_gt_1], dis_valid_mask_gt)
    num_corner = gt_corners.shape[0]
    only_match_one_mask = gt_ids_check == np.arange(num_corner)

    detIds_per_gt_2 = apply_mask_on_ids(detIds_per_gt_1, only_match_one_mask)
    num_ture_pos = (detIds_per_gt_2 >=0).sum()
    num_gt = gt_corners.shape[0]
    num_pos = det_corners.shape[0]
    nums_gt_dt_tp = [num_gt, num_pos, num_ture_pos]
    return nums_gt_dt_tp, detIds_per_gt_2

  def debug_line_eval(self, det_lines, gt_lines, line_detIds_per_gt, obj_wise=1):
    pos_line_ids = line_detIds_per_gt[line_detIds_per_gt>=0]
    det_lines_pos = det_lines[pos_line_ids]
    #_show_objs_ls_points_ls((512,512), [gt_lines, det_lines[:,:5], det_lines_pos[:,:5]], obj_colors=['white','yellow', 'green'], line_thickness=[1,1,2])
    if obj_wise:
      for i in range(line_detIds_per_gt.shape[0]):
        j = line_detIds_per_gt[i]
        if j>=0:
          _show_objs_ls_points_ls((512,512), [gt_lines[i:i+1], det_lines[j:j+1,:5]], obj_colors=['white', 'green'])
    pass

  def save_eval_res_img_1cls(self, img, det_lines, gt_lines, det_corners, gt_corners,
                        cor_detIds_per_gt, line_detIds_per_gt,
                        cor_nums_gt_dt_tp,  scene_name, det_cat,
                        det_points, obj_wise=0):
    if cor_nums_gt_dt_tp[0] == 0:
      cor_recall = 1
    else:
      cor_recall = cor_nums_gt_dt_tp[2]/cor_nums_gt_dt_tp[0]
    if cor_nums_gt_dt_tp[1] == 0:
      cor_precision=1
    else:
      cor_precision = cor_nums_gt_dt_tp[2]/cor_nums_gt_dt_tp[1]
    print(f'\ncor_nums_gt_dt_tp: {cor_nums_gt_dt_tp}')
    print(f'\ncor recall: {cor_recall}\ncor precision: {cor_precision}')

    # parse corner detection
    num_det_cors = det_corners.shape[0]
    pos_det_cor_ids = cor_detIds_per_gt[cor_detIds_per_gt>=0]
    neg_det_cor_ids = np.array([i for i in range(num_det_cors) if i not in pos_det_cor_ids], dtype=np.int32)
    det_corners_pos = det_corners[pos_det_cor_ids]
    det_corners_neg = det_corners[neg_det_cor_ids]
    gt_corners_true = gt_corners[cor_detIds_per_gt >= 0]
    gt_corners_false = gt_corners[cor_detIds_per_gt < 0]

    # parse line detection
    pos_line_ids = line_detIds_per_gt[line_detIds_per_gt>=0].astype(np.int)
    neg_line_ids = np.array([ i for i in range(det_lines.shape[0] ) if i not in pos_line_ids])
    det_lines_pos = det_lines[pos_line_ids]
    # sometimes merging generate extra lines, but points not
    nline = det_lines.shape[0]
    npts = det_points.shape[0]
    if nline > npts:
      det_points = np.concatenate([det_points, det_points[:nline-npts]], 0)
    det_points_pos = det_points[pos_line_ids]
    if len(neg_line_ids) == 0:
      det_lines_neg = det_lines[0:0]
      det_points_neg = det_points[0:0]
    else:
      det_lines_neg = det_lines[neg_line_ids]
      if self._draw_pts:
        det_points_neg = det_points[neg_line_ids]
      else:
        det_points_neg = None


    gt_line_true_ids = np.where(line_detIds_per_gt>=0)[0]
    gt_line_false_ids = np.where(line_detIds_per_gt<0)[0]
    gt_lines_true = gt_lines[gt_line_true_ids]
    gt_lines_false = gt_lines[gt_line_false_ids]

    if SET_DET_Z_AS_GT:
      gt_ids_per_pos = np.where( line_detIds_per_gt>=0 )[0]
      #_show_3d_points_objs_ls(objs_ls=[det_lines_pos[:,:7]], obj_rep='XYZLgWsHA')
      det_lines_pos[:,[2,5]] = gt_lines[gt_ids_per_pos][:,[2,5]]
      #_show_3d_points_objs_ls(objs_ls=[det_lines_pos[:,:7]], obj_rep='XYZLgWsHA')

    r = int(cor_recall*100)
    p = int(cor_precision*100)
    cat = det_cat

    img_name = f'{scene_name}_{cat}_Recall_0d{r}_Precision_0d{p}_EvalDet.png'
    img_file = os.path.join(self.eval_dir, img_name)
    #print('det corners. green: true pos, red: false pos')
    img_size = img.shape[:2]
    if SHOW_EACH_CLASS:
      _show_objs_ls_points_ls(
        img_size, [det_lines_pos[:,:5], det_lines_neg[:,:5]],
        obj_scores_ls=[det_lines_pos[:,-1:], det_lines_neg[:,-1:]],
                              points_ls=[det_corners_pos, det_corners_neg],
                              obj_colors=['green', 'red'], obj_thickness=1,
                              point_colors=['blue', 'yellow'], point_thickness=2,
                              out_file=img_file, only_save=1)

    #print('gt  corners. green: true pos, red: false neg')
    img_name = f'{scene_name}_{cat}_Recall_0d{r}_Precision_0d{p}_EvalGt.png'
    img_file = os.path.join(self.eval_dir, img_name)
    if SHOW_EACH_CLASS:
      _show_objs_ls_points_ls(img_size, [gt_lines_true, gt_lines_false],
                            points_ls=[gt_corners_true, gt_corners_false],
                            obj_colors=['green','red'],
                            line_thickness=1,
                            point_colors=['blue', 'yellow'],
                            point_thickness=2, out_file=img_file, only_save=1)

    img_name = f'{scene_name}_{cat}_Recall_0d{r}_Precision_0d{p}_Det.png'
    img_file = os.path.join(self.eval_dir, img_name)
    if SHOW_EACH_CLASS:
      _show_objs_ls_points_ls(img_size, [det_lines], [det_corners],
                             obj_colors='random', point_colors='random',
                             line_thickness=1, point_thickness=2,
                             out_file=img_file, only_save=1)

    # with input
    if  not self.is_pcl:
      img_name = f'{scene_name}_{cat}_Recall_0d{r}_Precision_0d{p}_EvalGt_wiht_input.png'
      img_file = os.path.join(self.eval_dir, img_name)
      if SHOW_EACH_CLASS:
        _show_objs_ls_points_ls(img[:,:,0], [gt_lines_true, gt_lines_false],
                              points_ls=[gt_corners_true, gt_corners_false],
                              obj_colors=['green','red'],
                              line_thickness=1,
                              point_colors=['blue', 'yellow'],
                              point_thickness=2, out_file=img_file, only_save=1)

      img_name = f'{scene_name}_{cat}_Recall_0d{r}_Precision_0d{p}_EvalDet_with_input.png'
      img_file = os.path.join(self.eval_dir, img_name)
      if SHOW_EACH_CLASS:
        _show_objs_ls_points_ls(img[:,:,0], [det_lines_pos, det_lines_neg],
                                points_ls=[det_corners_pos, det_corners_neg],
                              obj_colors=['green', 'red'], line_thickness=1,
                              point_colors=['blue', 'yellow'], point_thickness=2,
                              out_file=img_file, only_save=1)

    img_name = f'{scene_name}_{cat}_Recall_0d{r}_Precision_0d{p}'
    img_file_base_all_cls = os.path.join(self.eval_dir_all_cls, img_name)
    pass

    if obj_wise:
      for i in range(gt_corners.shape[0]):
        j = cor_detIds_per_gt[i]
        if j >= 0:
          _show_objs_ls_points_ls((512,512), [gt_lines], [gt_corners[i:i+1], det_corners[j:j+1]], obj_colors=['white'], point_colors=['green', 'red'], point_thickness=2)
    pass
    if not check_duplicate(det_lines_pos, self.obj_rep, 0.5):
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass
    return cat, img, img_file_base_all_cls, det_lines_pos, det_lines_neg, det_corners_pos, det_corners_neg, gt_lines_true, gt_lines_false, gt_corners_true, gt_corners_false, det_points_pos, det_points_neg

def get_z_by_iou(dets, det_cats, gts, gt_cats, obj_rep):
  from mmdet.core.bbox.geometry import dsiou_rotated_3d_bbox
  import torch
  assert obj_rep ==  'XYZLgWsHA'
  dets_t = torch.from_numpy( dets[:,:7] ).to(torch.float32)
  gts_t = torch.from_numpy(gts).to(torch.float32)
  ious = dsiou_rotated_3d_bbox( dets_t, gts_t, 0.8, only_2d=True ).numpy()
  for i in range(dets.shape[0]):
    mask_i = [det_cats[i] == c for c in gt_cats]
    ids_i = np.where(mask_i)[0]
    j = np.argmax(ious[i][ids_i])
    k = ids_i[j]
    #print(ious[i,k])
    dets[i, [2,5]] = gts[k, [2,5]]
  return dets

def draw_1_scene(img, all_gts, all_dets,  all_ious, labels_to_cats, obj_rep, iou_threshold, res_filename, all_det_points):
  import mmcv
  colors_map = COLOR_MAP_3D
  num_cats = len(labels_to_cats)
  img_det_score = None
  img_det_pts = None
  img_gt = None
  img_det_iou = None
  obj_dim = OBJ_REPS_PARSE._obj_dims[obj_rep]

  det_bboxes = []
  gt_bboxes = []
  det_cats = []
  gt_cats = []

  walls = None
  labels = all_gts.keys()
  for l in labels:
      cat = labels_to_cats[l]
      #if DEBUG:
      #  if cat == 'window':
      #    continue
      c = colors_map[cat]
      dets = all_dets[l]
      num_det = dets.shape[0]
      pts = all_det_points[l].reshape(num_det, 9,2)

      # ------ gt ------
      gts = all_gts[l]
      iou_matrix = all_ious[l]
      if iou_matrix.shape[0]==0:
        gt_lines_true = gts
        gt_lines_false = gts[0:0]
      else:
        ious_gt = iou_matrix.max(0)
        gt_pos_mask = ious_gt > iou_threshold
        gt_lines_true = gts[gt_pos_mask]
        gt_lines_false = gts[gt_pos_mask==False]

      if img_gt is None:
        img_gt = img.shape[:2]
        img_gt = img[:,:,0]
      gt_file = res_filename + '_Gt.png'
      img_gt = _draw_objs_ls_points_ls(img_gt,
              [gt_lines_true[:,:obj_dim], gt_lines_false[:,:obj_dim]],
              obj_rep,
              obj_colors=c,
              obj_cats_ls = ['', 'F'],
              point_colors=['blue', 'yellow'],
              obj_thickness=[2,2],
              point_thickness=[3,3],
              out_file=None,
              text_colors_ls=['green', 'red'])
      #mmcv.imshow(img_gt)

      if dets.shape[0]==0:
        continue
      ious_det = iou_matrix.max(1)

      det_pos_mask = ious_det > iou_threshold
      det_lines_pos = dets[det_pos_mask]
      det_lines_neg = dets[det_pos_mask==False]


      if cat == 'wall':
        walls = np.concatenate([det_lines_pos, det_lines_neg], 0)
      if cat in ['window', 'door'] and walls is not None:
        det_lines_pos = align_bboxes_with_wall(det_lines_pos, walls, cat, obj_rep)
        det_lines_neg = align_bboxes_with_wall(det_lines_neg, walls, cat, obj_rep)

      det_bboxes.append( det_lines_pos )
      det_bboxes.append( det_lines_neg )
      gt_bboxes.append( gt_lines_true )
      gt_bboxes.append( gt_lines_false )
      dn = det_lines_pos.shape[0] + det_lines_neg.shape[0]
      gn = gt_lines_true.shape[0] + gt_lines_false.shape[0]
      det_cats += [cat] * dn
      gt_cats += [cat] * gn

      if img_det_score is None:
        h, w = img.shape[:2]
        h+= 100
        w+=100
        img_det_iou =  (h,w)
        img_det_score =  (h,w)
        img_det_pts =  (h,w)
        img_det_pts =  img[:,:,0]
        det_iou_file = res_filename + '_Det_IoU.png'
        det_score_file = res_filename + '_Det_Score.png'
        det_pts_file = res_filename + '_DetPts.png'

      obj_scores_ls = [det_lines_pos[:,obj_dim], det_lines_neg[:,obj_dim]]
      img_det_score = _draw_objs_ls_points_ls(img_det_score,
              [det_lines_pos[:,:obj_dim], det_lines_neg[:,:obj_dim]],
              obj_rep,
              obj_colors=c,
              obj_scores_ls = obj_scores_ls,
              obj_cats_ls = ['', 'F'],
              point_colors=['blue', 'yellow'],
              obj_thickness=[2,2],
              point_thickness=[3,3],
              out_file=None,
              text_colors_ls=['green', 'red'])

      obj_scores_ls = [ious_det[det_pos_mask], ious_det[det_pos_mask==False]]
      img_det_iou = _draw_objs_ls_points_ls(img_det_iou,
              [det_lines_pos[:,:obj_dim], det_lines_neg[:,:obj_dim]],
              obj_rep,
              obj_colors=c,
              obj_scores_ls = obj_scores_ls,
              obj_cats_ls = ['', 'F'],
              point_colors=['blue', 'yellow'],
              obj_thickness=[2,2],
              point_thickness=[3,3],
              out_file=None,
              text_colors_ls=['green', 'red'])

      for di in range(num_det):
        ci = ColorList[di]
        img_det_pts = _draw_objs_ls_points_ls(img_det_pts,
              [dets[[di]][:,:obj_dim]],
              obj_rep,
              points_ls = [pts[di]],
              obj_colors=ci,
              point_colors=ci,
              obj_thickness=2,
              point_thickness=5,
              out_file=None,)

      pass

  write_img(img_gt, gt_file)
  if img_det_iou is None:
    return
  write_img(img_det_iou, det_iou_file)
  write_img(img_det_score, det_score_file)
  write_img(img_det_pts, det_pts_file)

  det_bboxes = np.concatenate( det_bboxes, axis=0 )
  gt_bboxes = np.concatenate( gt_bboxes, axis=0 )

  det_colors = [colors_map[c] for c in det_cats]
  gt_colors = [colors_map[c] for c in gt_cats]

  img_detgt = img.shape[:2]
  detgt_file = res_filename + '_DetGt.png'
  img_detgt = _draw_objs_ls_points_ls(img_detgt,
          [det_bboxes[:,:obj_dim], gt_bboxes[:,:obj_dim]],
          obj_rep,
          obj_colors=['blue', 'red'],
          obj_thickness=[3,1],
          out_file=None,)
  write_img(img_detgt, detgt_file)
  #mmcv.imshow(img_detgt)

  if SHOW_3D:
    gt_bboxes = aug_thickness( gt_bboxes, gt_cats )
    det_bboxes = aug_thickness( det_bboxes, det_cats )

    wall_mask = [c in ['wall', 'beam', 'window', 'door'] for c in gt_cats]
    floor_mask = [c=='floor' for c in gt_cats]
    gt_walls = gt_bboxes[wall_mask]
    gt_floors0 = gt_bboxes[floor_mask]
    gt_floors_mesh = get_cf_from_wall(gt_floors0, gt_walls, obj_rep, 'floor')

    non_floor_mask = [c!='floor' for c in gt_cats]
    gts = gt_bboxes[non_floor_mask]
    m = len(gt_colors)
    gt_colors = [gt_colors[j] for j in range(m) if non_floor_mask[j] ]

    det_bboxes = get_z_by_iou(det_bboxes, det_cats, gt_bboxes, gt_cats, obj_rep)

    wall_mask = [c in ['wall', 'beam', 'window', 'door'] for c in det_cats]
    floor_mask = [c=='floor' for c in det_cats]
    det_walls = det_bboxes[wall_mask][:,:-1]
    det_floors0 = det_bboxes[floor_mask][:,:-1]
    det_floors_mesh = get_cf_from_wall(det_floors0, det_walls, obj_rep, 'floor')

    non_floor_mask = [c!='floor' for c in det_cats]
    dets = det_bboxes[non_floor_mask][:,:-1]
    n = len(det_colors)
    det_colors = [det_colors[j] for j in range(n) if non_floor_mask[j] ]

    #_show_3d_points_objs_ls( objs_ls=[gts, gts], obj_rep=obj_rep, obj_colors=[gt_colors, 'navy'], box_types= ['surface_mesh', 'line_mesh'], polygons_ls=[gt_floors_mesh], polygon_colors='silver' )
    #dets, det_colors = manual_add(dets, det_colors)
    #if DEBUG:
    #  dets, det_colors= rm_short(dets, det_colors)
    _show_3d_points_objs_ls( objs_ls=[dets, dets], obj_rep=obj_rep, obj_colors=[det_colors, 'navy'], box_types=['surface_mesh', 'line_mesh'], polygons_ls=[gt_floors_mesh], polygon_colors=['silver'] )
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    pass

  pass

def rm_short(bboxes, colors):
  mask = bboxes[:,3] > 0
  mask[[12,14, 17]] = False
  # [10, 12, 13, 14, 17]
  colors = [ colors[i] for i in range(len(mask)) if mask[i] ]
  return bboxes[mask], colors


def lighten_density_img(img):
  mask = img > 50
  rate = mask * 2
  img = img  * rate
  img = np.clip(img, a_min=0, a_max=255)
  img = img.astype(np.uint8)
  return img

def draw_eval_all_classes_1_scene(eval_draws_ls, obj_rep, _draw_pts ):
  import mmcv
  from tools.color import COLOR_MAP_2D, ColorList, ColorValuesNp
  colors_map = COLOR_MAP_2D

  num_cats = len(eval_draws_ls)
  img_det = None
  img_det_mesh = None
  img_det_pts = None
  img_gt = None
  img_gt_mesh = None
  obj_dim = OBJ_REPS_PARSE._obj_dims[obj_rep]

  det_bboxes = []
  gt_bboxes = []
  det_cats = []
  gt_cats = []
  cat_ls = []
  rooms = None
  walls = None
  gt_walls = None
  for i in range(num_cats):
      cat, img, img_file_base_all_cls, det_lines_pos, det_lines_neg, det_corners_pos, \
        det_corners_neg, gt_lines_true, gt_lines_false, gt_corners_true, gt_corners_false,\
        det_points_pos, det_points_neg = eval_draws_ls[i]
      cat_ls.append(cat)
      det_lines = np.concatenate([det_lines_pos, det_lines_neg], 0)
      gt_lines = np.concatenate([gt_lines_true, gt_lines_false], 0)
      num_pos = det_lines_pos.shape[0]
      num_neg = det_lines_neg.shape[0]
      num_dets = num_pos + num_neg
      c = colors_map[cat]
      c = 'black'
      img_size = img.shape[:2]

      if cat == 'wall':
        gt_walls = gt_lines

      if _draw_pts:
        det_points = np.concatenate([det_points_pos, det_points_neg], 0).reshape(-1,9,2)
        det_points_pos = det_points_pos.reshape(-1, 2)
        det_points_neg = det_points_neg.reshape(-1, 2)

      if img_det is None:
        h,w = img.shape[:2]
        img_det_no_cor = img_det = img_det_mesh = img_gt = img_gt_mesh = np.ones([h,w,3], dtype=np.uint8)*0
        det_file = img_file_base_all_cls + '_Det.png'
        det_mesh_file = img_file_base_all_cls + '_DetMesh.png'

        img_det_pts = lighten_density_img( img[:,:,0] )
        det_pts_file = img_file_base_all_cls + '_DetPts.png'

        #img_gt = lighten_density_img( img[:,:,0])
        gt_file = img_file_base_all_cls + '_Gt.png'
        gt_mesh_file = img_file_base_all_cls + '_GtMesh.png'

      if cat == 'room':
        rooms = det_lines
        img_det_rooms = lighten_density_img( img[:,:,0])
        img_det_rooms_pts = lighten_density_img( img[:,:,0])

        det_rooms_file = img_file_base_all_cls + '_DetRoomScores.png'
        det_rooms_pts_file = img_file_base_all_cls + '_DetRoomsPts.png'
        for di in range(num_dets):
          ci = ColorList[di]
          img_det_rooms = _draw_objs_ls_points_ls(img_det_rooms,
                [det_lines[di:di+1,:obj_dim]],
                obj_rep,
                [det_lines[di:di+1,:2]],
                obj_colors=ci,
                obj_thickness=5,
                point_colors=ci,
                point_thickness=15,
                out_file=None,)

        if _draw_pts:
          for di in range(num_dets):
            ci = ColorList[di]
            img_det_rooms_pts = _draw_objs_ls_points_ls(img_det_rooms_pts,
                    [det_lines[di:di+1,:obj_dim]],
                    obj_rep,
                    [det_points[di]],
                    obj_colors=c,
                    obj_scores_ls = [det_lines[di:di+1,obj_dim]],
                    point_colors=ci,
                    obj_thickness=2,
                    point_thickness=12,
                    out_file=None,)
        continue

      if cat == 'wall':
        walls = np.concatenate([det_lines_pos, det_lines_neg], 0)
        if not check_duplicate(walls, obj_rep, 0.3):
          import pdb; pdb.set_trace()  # XXX BREAKPOINT
          pass
        img_det_mesh = draw_rooms_from_edges(walls[:,:7], obj_rep, img_size, 0)
        img_gt_mesh = draw_rooms_from_edges(gt_lines[:,:7], obj_rep, img_size)
      if cat in ['window', 'door'] and walls is not None:
        det_lines_pos = align_bboxes_with_wall(det_lines_pos, walls, cat, obj_rep)
        det_lines_neg = align_bboxes_with_wall(det_lines_neg, walls, cat, obj_rep)

      det_bboxes.append( det_lines_pos )
      det_bboxes.append( det_lines_neg )
      gt_bboxes.append( gt_lines_true )
      gt_bboxes.append( gt_lines_false )
      dn = det_lines_pos.shape[0] + det_lines_neg.shape[0]
      gn = gt_lines_true.shape[0] + gt_lines_false.shape[0]
      det_cats += [cat] * dn
      gt_cats += [cat] * gn

      img_det_mesh = draw_building_objs('mesh', cat, img_det_mesh, det_lines_pos, det_lines_neg, obj_rep,\
            obj_dim, det_corners_pos, det_corners_neg)

      img_det_no_cor = draw_building_objs('edge', cat, img_det_no_cor, det_lines_pos, det_lines_neg, obj_rep,\
            obj_dim, det_corners_pos, det_corners_neg, with_cor=0)

      img_det = draw_building_objs('edge', cat, img_det, det_lines_pos, det_lines_neg, obj_rep,\
            obj_dim, det_corners_pos, det_corners_neg)


      if _draw_pts:
        for di in range(num_dets):
          ci = ColorList[di]
          if det_points.shape[0] <= di:
            continue
          dets_ls = [det_lines[:,:obj_dim]]
          dets_ls = None
          img_det_pts = _draw_objs_ls_points_ls(img_det_pts,
                dets_ls,
                obj_rep,
                [det_points[di]],
                obj_colors='white',
                obj_scores_ls = [det_lines[:,obj_dim]],
                point_colors=ci,
                obj_thickness=1,
                point_thickness=12,
                out_file=None,)

        #mmcv.imshow(img_det_pts)
      img_gt_mesh = draw_building_objs('mesh', cat, img_gt_mesh, gt_lines_true, gt_lines_false, obj_rep,\
            obj_dim, gt_corners_true, gt_corners_false)
      img_gt = draw_building_objs('edge', cat, img_gt, gt_lines_true, gt_lines_false, obj_rep,\
            obj_dim, gt_corners_true, gt_corners_false)

  #--------------------------------
  # crop image size
  auto_crop_img = 1
  if auto_crop_img:
    d_wall_corners = OBJ_REPS_PARSE.encode_obj(walls[:,:7], obj_rep, 'RoLine2D_2p').reshape(-1,2)
    g_wall_corners = OBJ_REPS_PARSE.encode_obj(gt_walls, obj_rep, 'RoLine2D_2p').reshape(-1,2)
    wall_corners = np.concatenate([d_wall_corners, g_wall_corners], 0)
    min_xy = wall_corners.min(0).astype(np.int) - 10
    y0, x0 = min_xy.clip(min=0)
    max_xy = wall_corners.max(0).astype(np.int) + 10
    y1, x1 = max_xy.clip(max=img_det.shape[0])

  # ----------------------------------------
  # room related
  if 'room' in cat_ls:
    if walls is not None:
      img_det_room_rel = draw_walls_rooms_rel(img_det, walls, rooms, obj_rep)
      det_room_rel_file = det_file.replace('.png','_room_rels.png')
      if auto_crop_img:
        img_det_room_rel = img_det_room_rel[ x0:x1, y0:y1,: ]
      write_img( img_det_room_rel, det_room_rel_file )
    if auto_crop_img:
      img_det_rooms = img_det_rooms[ x0:x1, y0:y1,: ]
      img_det_rooms_pts = img_det_rooms_pts[ x0:x1, y0:y1]
    write_img(img_det_rooms, det_rooms_file)
    write_img(img_det_rooms_pts, det_rooms_pts_file)

  # ----------------------------------------
  if auto_crop_img:
    img_det = img_det[ x0:x1, y0:y1,: ]
    img_det_no_cor = img_det_no_cor[ x0:x1, y0:y1,: ]
    img_det_mesh = img_det_mesh[ x0:x1, y0:y1,: ]
    img_gt = img_gt[ x0:x1, y0:y1,: ]
    img_gt_mesh = img_gt_mesh[ x0:x1, y0:y1,: ]
    if _draw_pts:
      img_det_pts = img_det_pts[ x0:x1, y0:y1,: ]

  write_img(img_det, det_file)
  write_img(img_det_no_cor, det_file.replace('.png', '_NoCor.png'))
  write_img(img_det_mesh, det_mesh_file)
  if _draw_pts:
    write_img(img_det_pts, det_pts_file)
  write_img(img_gt, gt_file)
  write_img(img_gt_mesh, gt_mesh_file)


  det_bboxes = np.concatenate( det_bboxes, axis=0 )
  gt_bboxes = np.concatenate( gt_bboxes, axis=0 )

  det_colors = [colors_map[c] for c in det_cats]
  gt_colors = [colors_map[c] for c in gt_cats]

  img_detgt = img.shape[:2]
  detgt_file = img_file_base_all_cls + '_DetGt.png'
  img_detgt = _draw_objs_ls_points_ls(img_detgt,
          [det_bboxes[:,:obj_dim], gt_bboxes[:,:obj_dim]],
          obj_rep,
          obj_colors=['blue', 'red'],
          obj_thickness=[3,1],
          out_file=None,)
  write_img(img_detgt, detgt_file)
  #mmcv.imshow(img_detgt)

  if SHOW_3D:
    gt_bboxes = aug_thickness( gt_bboxes, gt_cats )
    det_bboxes = aug_thickness( det_bboxes, det_cats )

    wall_mask = [c in ['wall', 'beam', 'window', 'door'] for c in gt_cats]
    floor_mask = [c=='floor' for c in gt_cats]
    gt_walls = gt_bboxes[wall_mask]
    gt_floors0 = gt_bboxes[floor_mask]
    gt_floors_mesh = get_cf_from_wall(gt_floors0, gt_walls, obj_rep, 'floor')

    non_floor_mask = [c!='floor' for c in gt_cats]
    gts = gt_bboxes[non_floor_mask]
    m = len(gt_colors)
    gt_colors = [gt_colors[j] for j in range(m) if non_floor_mask[j] ]

    det_bboxes = get_z_by_iou(det_bboxes, det_cats, gt_bboxes, gt_cats, obj_rep)

    wall_mask = [c in ['wall', 'beam', 'window', 'door'] for c in det_cats]
    floor_mask = [c=='floor' for c in det_cats]
    det_walls = det_bboxes[wall_mask][:,:-1]
    det_floors0 = det_bboxes[floor_mask][:,:-1]
    det_floors_mesh = get_cf_from_wall(det_floors0, det_walls, obj_rep, 'floor')

    non_floor_mask = [c!='floor' for c in det_cats]
    dets = det_bboxes[non_floor_mask][:,:-1]
    n = len(det_colors)
    det_colors = [det_colors[j] for j in range(n) if non_floor_mask[j] ]

    _show_3d_points_objs_ls( objs_ls=[gts, gts], obj_rep=obj_rep, obj_colors=[gt_colors, 'navy'], box_types= ['surface_mesh', 'line_mesh'], polygons_ls=[gt_floors_mesh], polygon_colors='silver' )
    #dets, det_colors = manual_add(dets, det_colors)
    #_show_3d_points_objs_ls( objs_ls=[dets, dets], obj_rep=obj_rep, obj_colors=[det_colors, 'navy'], box_types=['surface_mesh', 'line_mesh'], polygons_ls=[gt_floors_mesh], polygon_colors=['silver'] )
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    pass

  pass

def write_img(img, path):
  import mmcv
  #img = mmcv.imflip(img)
  mmcv.imwrite(img, path)

def aug_min_length(objs, obj_rep):
  #objs[:,3] = np.clip(objs[:,3], a_min=70, a_max=None)
  pass

def draw_building_objs(style, cat, img_det, det_lines_pos, det_lines_neg, obj_rep,
            obj_dim, det_corners_pos, det_corners_neg, with_cor=True):
      if style == 'mesh':
        c = 'black'
      if style == 'edge':
        c = 'random'

      if cat=='wall':
        aug_min_length(det_lines_pos, obj_rep)

      det_lines_pos = det_lines_pos.copy()
      det_lines_neg = det_lines_neg.copy()
      #obj_scores_ls = [det_lines_pos[:,obj_dim], det_lines_neg[:,obj_dim]]
      obj_scores_ls = None
      if with_cor:
        corners =  [det_corners_neg, det_corners_pos]
      else:
        corners = None
      obj_cats_ls = ['', 'F']
      obj_cats_ls = None
      if style == 'mesh':
        tk = 3
      else:
        tk = 8
      if cat in ['door', 'window']:
        c = {'window':'gray', 'door':'lime'} [cat]
        if style == 'mesh':
          tk = 6
        else:
          tk = 8
        rds = tk

        det_lines_pos[:,3] -= rds+2
        det_lines_neg[:,3] -= rds+2
        corners = None
      point_colors=['red', 'red']
      img_det = _draw_objs_ls_points_ls(img_det,
              [det_lines_pos[:,:obj_dim], det_lines_neg[:,:obj_dim]],
              obj_rep,
              corners,
              obj_colors=c,
              obj_scores_ls = obj_scores_ls,
              obj_cats_ls = obj_cats_ls,
              point_colors = point_colors,
              obj_thickness=tk,
              point_thickness=tk,
              out_file=None,
              text_colors_ls=['green', 'red'])

      if cat in ['door', 'window']:
        if style == 'mesh':
          c = 'black'
          tk = 2
          width = 8
        else:
          c = 'white'
          tk = 2
          width = 8
          #return img_det
        det_lines_pos[:,3] += rds-2
        det_lines_neg[:,3] += rds-2
        det_lines_pos[:,4] = width
        det_lines_neg[:,4] = width
        img_det = _draw_objs_ls_points_ls(img_det,
                [det_lines_pos[:,:obj_dim], det_lines_neg[:,:obj_dim]],
                obj_rep,
                obj_colors=c,
                obj_scores_ls = obj_scores_ls,
                obj_cats_ls = obj_cats_ls,
                obj_thickness=tk,
                out_file=None,)
      #mmcv.imshow(img_det)
      return img_det

def manual_add(dets, det_colors):
  w = np.array([[504.686, 266.188, 117.   , 322.194,   4.671, 208.2  ,   1.56 ]], dtype=np.float32)
  dets = np.concatenate([dets, w], 0)
  det_colors += ['gray']
  return dets, det_colors

def aug_thickness(bboxes, cats, min_thick=12):
  mask = np.array([c in ['door', 'window'] for c in cats])
  bboxes[mask,4] = np.clip(bboxes[mask][:,4], a_min=min_thick, a_max=None)
  return bboxes


def filter_low_score_det(det_lines, score_threshold=0.5):
  mask = det_lines[:,-1] > score_threshold
  det_lines_ = det_lines[mask]
  ids = np.where(mask)[0]
  return det_lines_, ids

def apply_mask_on_ids(ids, mask):
  return (ids + 1) * mask - 1


def main():
  workdir = '/home/z/Research/mmdetection/work_dirs/'
  workdir = '/home/z/Research/mmdetection/work_dirs/TrL18K_S_July3/'

  dirs = [
  'bTPV_r50_fpn_XYXYSin2_beike2d_wa_bs7_lr10_LsW510R2P1N1_Rfiou631_Fpn44_Pbs1_Bp32_Rel',
  'bTPV_r50_fpn_XYXYSin2WZ0Z1_Std__beike2d_ro_bs7_lr1_LsW510R2P1N1_Rfiou832_Fpn44_Pbs1_Bp32',
  'bTPV_r50_fpn_XYXYSin2_beike2d_dowi_bs7_lr10_LsW510R2P1N1_Rfiou631_Fpn44_Pbs1_Bp32',
  ]

  filename = 'detection_11_Imgs.pickle'
  filename = 'detection_111_Imgs.pickle'

  res_files = [os.path.join( os.path.join(workdir, d), filename) for d in dirs]
  #eval_graph(res_files[0])

  eval_graph_multi_files(res_files)

if __name__ == '__main__'  :
  main()




