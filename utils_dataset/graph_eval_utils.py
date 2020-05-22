import pickle
from beike_data_utils.beike_utils import load_gt_lines_bk
from tools.debug_utils import _show_lines_ls_points_ls
from configs.common import DIM_PARSE
from obj_geo_utils.line_operations import gen_corners_from_lines_np, get_lineIdsPerCor_from_corIdsPerLine
from obj_geo_utils.obj_utils import GraphUtils, OBJ_REPS_PARSE
from obj_geo_utils.geometry_utils import get_cf_from_wall
from collections import defaultdict
import os
import numpy as np
from tools.visual_utils import _show_objs_ls_points_ls, _draw_objs_ls_points_ls, _show_3d_points_objs_ls
from utils_dataset.stanford3d_utils.post_processing import align_bboxes_with_wall

SHOW_EACH_CLASS = False
SET_DET_Z_AS_GT = 1
SHOW_3D = 0

def change_result_rep(results, classes, obj_rep_org, obj_rep_out='XYZLgWsHA'):
    dim_parse = DIM_PARSE(obj_rep_org, len(classes)+1)
    dim_parse_out = DIM_PARSE(obj_rep_out, len(classes)+1)
    re_s, re_e = dim_parse.OUT_ORDER['bbox_refine']
    in_s, in_e = dim_parse.OUT_ORDER['bbox_init']

    num_img = len(results)
    for i in range(num_img):
      det_bboxes = results[i]['det_bboxes']
      gt_bboxes = results[i]['gt_bboxes']
      gt_bboxes = [[ OBJ_REPS_PARSE.encode_obj(gt_bboxes[0][0], obj_rep_org, obj_rep_out) ]]
      num_level = len(det_bboxes)
      for l in range(num_level):
          assert det_bboxes[l].shape[1] == dim_parse.OUT_DIM_FINAL
          #_show_objs_ls_points_ls( (512,512), [det_bboxes[l][:,re_s:re_e]], obj_rep_org )
          bbox_refine = OBJ_REPS_PARSE.encode_obj(det_bboxes[l][:, re_s:re_e], obj_rep_org, obj_rep_out)
          bbox_init = OBJ_REPS_PARSE.encode_obj(det_bboxes[l][:, in_s:in_e], obj_rep_org, obj_rep_out)
          det_bboxes_new = np.concatenate([ bbox_refine, bbox_init, det_bboxes[l][:,in_e:] ], axis=1)
          #_show_objs_ls_points_ls( (512,512), [ bbox_refine ], obj_rep_out )
          if not det_bboxes_new.shape[1] == dim_parse_out.OUT_DIM_FINAL:
            import pdb; pdb.set_trace()  # XXX BREAKPOINT
            pass
          det_bboxes[l] = det_bboxes_new

      results[i]['det_bboxes'] = det_bboxes
      results[i]['gt_bboxes'] = gt_bboxes
    return results, obj_rep_out

def save_res_graph(dataset, data_loader, results, out_file, data_test_cfg):
    filter_edges = data_test_cfg['filter_edges']
    classes = data_test_cfg['classes']
    obj_rep = data_test_cfg['obj_rep']
    results, obj_rep = change_result_rep(results, classes, obj_rep, 'XYZLgWsHA')
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
        det_relations = result['det_relations']
        if result['gt_bboxes']:
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

          #_show_objs_ls_points_ls(img_i, [res_data['gt_bboxes'][0]], obj_rep='XYXYSin2')
          pass

        detections_all_labels = []
        for label in range(1, len(det_result)+1):
          det_lines_multi_stages = det_result[label-1]
          det_lines = det_lines_multi_stages
          category_id = dataset.cat_ids[label]
          cat = catid_2_cat[category_id]
          assert det_lines.shape[1] == dim_parse.OUT_DIM_FINAL
          detection_bRefine_sAve = dim_parse.clean_bboxes_out(det_lines, stage='final', out_type='bRefine_sAve')
          detection_bInit_sRefine = dim_parse.clean_bboxes_out(det_lines, stage='final', out_type='bInit_sRefine')
          s, e = dim_parse.OUT_ORDER['points_refine']
          points_refine = det_lines[:, s:e]
          s, e = dim_parse.OUT_ORDER['points_init']
          points_init = det_lines[:, s:e]

          detection_l = {'det_lines': det_lines, 'category_id': category_id, 'cat': cat,
                         'detection_bRefine_sAve': detection_bRefine_sAve,
                         'detection_bInit_sRefine': detection_bInit_sRefine,
                         'points_init': points_init,
                         'points_refine': points_refine,
                         }

          detections_all_labels.append(detection_l)
        res_data['detections'] = detections_all_labels
        res_data['det_relations'] = det_relations

        # optimize
        results_datas.append( res_data )

    out_file = out_file.replace('.pickle', f'_{num_imgs}_Imgs.pickle')
    with open(out_file, 'wb') as f:
      pickle.dump(results_datas, f)
      print(f'\nsave: {out_file}')

    eval_graph( out_file )


def post_process_bboxes_1cls(det_lines, score_threshold, label, cat, opt_graph_cor_dis_thr, obj_rep, min_out_length):
  '''
  The input detection belong to one class
  det_lines: [n,6]
  score_threshold: 0.4
  label: 1
  opt_graph_cor_dis_thr: 10
  obj_rep: 'XYXYSin2'

  det_lines_merged: [m,6]
  '''
  obj_dim = OBJ_REPS_PARSE._obj_dims[obj_rep]
  assert det_lines.shape[1] == obj_dim+1
  det_lines = filter_low_score_det(det_lines, score_threshold)
  labels_i = np.ones(det_lines.shape[0], dtype=np.int)*label
  scores_i = det_lines[:,-1]
  if cat == 'wall':
    det_lines_merged, scores_merged, labels_merged, _ = \
      GraphUtils.optimize_graph(det_lines[:,:obj_dim], scores_i, labels_i, obj_rep=obj_rep,
      opt_graph_cor_dis_thr=opt_graph_cor_dis_thr, min_out_length=min_out_length)

    check_length = 0
    if check_length:
      # check length
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      det_lines_length = OBJ_REPS_PARSE.encode_obj( det_lines[:,:5], 'XYXYSin2', 'RoLine2D_CenterLengthAngle' )[:,2]
      merged_lines_length = OBJ_REPS_PARSE.encode_obj( det_lines_merged[:,:5], 'XYXYSin2', 'RoLine2D_CenterLengthAngle' )[:,2]

    det_lines_merged = np.concatenate([det_lines_merged, scores_merged], axis=1)
  else:
    det_lines_merged = det_lines
    labels_merged = labels_i
  return det_lines_merged, labels_merged

def eval_graph(res_file):
  with open(res_file, 'rb') as f:
    results_datas = pickle.load(f)
  img_meta = results_datas[0]['img_meta']
  classes = img_meta['classes']
  filter_edges =  results_datas[0]['filter_edges']
  obj_rep =  results_datas[0]['obj_rep']
  graph_eval = GraphEval(obj_rep, classes, filter_edges)
  graph_eval(results_datas, res_file)


class GraphEval():
  _all_out_types = [ 'composite', 'bInit_sRefine', 'bRefine_sAve' ][1:3]
  _all_out_types = [ 'bRefine_sAve' ]
  #_all_out_types = [ 'bInit_sRefine' ]
  _score_threshold  = 0.4
  _corner_dis_threshold = 15
  _opt_graph_cor_dis_thr = 10
  #_opt_graph_cor_dis_thr = 15
  _min_out_length = 5

  _eval_img_scale_ratio = 1.0
  _eval_img_size_aug = 0

  scene_list = ['Area_5/conferenceRoom_2', 'Area_5/hallway_2', 'Area_5/office_21', 'Area_5/office_39', 'Area_5/office_40', 'Area_5/office_41']
  scene_list = None

  def __init__(self, obj_rep, classes, filter_edges, score_threshold=0.4,):
    self.obj_rep = obj_rep
    self.obj_dim = OBJ_REPS_PARSE._obj_dims[obj_rep]
    self.classes = classes
    self.filter_edges = filter_edges
    self._score_threshold = score_threshold
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
    self.work_dir = os.path.dirname(out_file)
    s = int(self._score_threshold*10)
    if self.optimize_graph:
      self.par_nice = f'Score{s}_optGraph{self._opt_graph_cor_dis_thr}_minLen{self._min_out_length}_{self.out_type}_corDis{self._corner_dis_threshold}'
    else:
      self.par_nice = f'Score{s}_NoOptGraph_minLen{self._min_out_length}_{self.out_type}_corDis{self._corner_dis_threshold}'
    self.eval_dir = os.path.join(self.work_dir, 'Eval_' + self.par_nice + f'_{self.num_img}Imgs/PerClass')
    self.eval_dir_all_cls = os.path.join(self.work_dir, 'Eval_' + self.par_nice + f'_{self.num_img}Imgs/AllClasses')
    if not os.path.exists(self.eval_dir):
      os.makedirs(self.eval_dir)
    if not os.path.exists(self.eval_dir_all_cls):
      os.makedirs(self.eval_dir_all_cls)

  def __call__(self, results_datas, out_file):
    for out_type in self._all_out_types:
      self.out_type = out_type
      self.evaluate(results_datas, out_file, out_type)
      self.evaluate(results_datas, out_file, out_type, False)

  def evaluate(self, results_datas, out_file, out_type, optimize_graph=True):
    assert self.obj_rep  == 'XYZLgWsHA'
    self.optimize_graph = optimize_graph
    debug = 0

    self.num_img = len(results_datas)
    self.update_path(out_file)
    all_cor_nums_gt_pos_tp = defaultdict(list)
    all_line_nums_gt_pos_tp = defaultdict(list)
    catid_2_cat = results_datas[0]['catid_2_cat']

    self.is_pcl = 'input_style' in results_datas[0]['img_meta'] and results_datas[0]['img_meta']['input_style'] == 'pcl'
    if self.is_pcl:
      self._eval_img_size_aug = 20
      self._eval_img_scale_ratio = 1.5

    for i_img, res_data in enumerate(results_datas):
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
        for label in range(1, num_labels+1):
            cat = catid_2_cat[label]
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
              det_lines_merged, _ = post_process_bboxes_1cls(det_lines,
                  self._score_threshold, label, cat, self._opt_graph_cor_dis_thr,
                  self.dim_parse.OBJ_REP, self._min_out_length)
            else:
              det_lines_merged = det_lines

            if debug and 0:
              print('raw prediction')
              _show_lines_ls_points_ls(img[:,:,0], [det_lines[:,:5], gt_lines_l], line_colors=['green','red'])


            det_category_id = detections[label-1]['category_id']
            if det_category_id != 1:
              pass
              #raise NotImplementedError
            cor_nums_gt_pos_tp, line_nums_gt_pos_tp, eval_draws = self.eval_1img_1cls(img, det_lines_merged, gt_lines_l, scene_name, cat, det_points)
            all_cor_nums_gt_pos_tp[label].append(cor_nums_gt_pos_tp)
            all_line_nums_gt_pos_tp[label].append(line_nums_gt_pos_tp)
            eval_draws_ls.append(eval_draws)

            if debug or 0:
              print(f'optimize graph with self._opt_graph_cor_dis_thr= {self._opt_graph_cor_dis_thr}')
              _show_lines_ls_points_ls(img[:,:,0], [det_lines[:,:5], gt_lines_l], line_colors=['green','red'])
              _show_lines_ls_points_ls(img[:,:,0], [det_lines_merged[:,:5], gt_lines_l], line_colors=['green','red'])

            pass
        draw_eval_all_classes_1img(eval_draws_ls, self.obj_rep)
        pass

    corner_recall_precision_perimg = defaultdict(list)
    line_recall_precision_perimg = defaultdict(list)

    corner_recall_precision = {}
    line_recall_precision = {}
    cor_nums_sum = {}
    line_nums_sum = {}

    for label in all_cor_nums_gt_pos_tp:
      cat =  catid_2_cat[label]
      cor_nums = np.array(all_cor_nums_gt_pos_tp[label])
      line_nums = np.array(all_line_nums_gt_pos_tp[label])
      corner_recall_precision_perimg[cat].append( (cor_nums[:,2] / cor_nums[:,0], cor_nums[:,2] / cor_nums[:,1] ))
      line_recall_precision_perimg[cat].append( (line_nums[:,2] / line_nums[:,0], line_nums[:,2] / line_nums[:,1] ))

      cor = cor_nums.sum(axis=0)
      line = line_nums.sum(axis=0)

      corner_recall_precision[cat] = [cor[2]/cor[0], cor[2]/cor[1]]
      line_recall_precision[cat] = [line[2]/line[0], line[2]/line[1]]
      cor_nums_sum[cat] = cor
      line_nums_sum[cat] = line

    eval_res_str = self.get_eval_res_str(corner_recall_precision, line_recall_precision, img_meta, line_nums_sum, cor_nums_sum)
    path = os.path.dirname(out_file)
    eval_path = os.path.join(path, 'eval_res.txt')
    with open(eval_path, 'a') as f:
      f.write(eval_res_str)
    print(eval_res_str)

    # save eval res
    eval_res = dict( corner_recall_precision = corner_recall_precision,
                    line_recall_precision = line_recall_precision,
                    corner_recall_precision_perimg = corner_recall_precision_perimg,
                    line_recall_precision_perimg = line_recall_precision_perimg,
                    )
    s = 'OptimizeGraph' if self.optimize_graph else 'NoOptmizeGraph'
    eval_res_file = out_file.replace('.pickle', f'_EvalRes{s}.npy')
    np.save(eval_res_file, eval_res)
    return eval_res_str

  def get_eval_res_str(self, corner_recall_precision, line_recall_precision, img_meta, line_nums_sum, cor_nums_sum ):
    rotate = False
    eval_str = '\n\n--------------------------------------\n\n' + \
                str(self) + f'num_img: {self.num_img}\n'
    eval_str += f'optimize_graph: {self.optimize_graph}\n'

    eval_str += 'Precision-Recall\n\n'
    cats = corner_recall_precision.keys()
    eval_str += '| split |'
    for cat in cats:
      str_c = f'{cat} corner'
      str_e = f'{cat} edge '
      eval_str += f'{str_c:14}|{str_e:14}|'
    eval_str += '\n|-|'
    for cat in cats:
      eval_str += '-|-|'

    eval_str += '\n| eval  |'
    for cat in corner_recall_precision:
      cor_rec, cor_prec = corner_recall_precision[cat]
      line_rec, line_prec = line_recall_precision[cat]
      cor_str = f'{cor_prec:.3} - {cor_rec:.3}'
      line_str = f'{line_prec:.3} - {line_rec:.3}'
      eval_str += f'{cor_str:14}|{line_str:14}|'
      pass
    eval_str += '\n'

    eval_str += '|gt num |'
    for cat in cats:
      cor_num = cor_nums_sum[cat][0]
      line_num = line_nums_sum[cat][0]
      eval_str += f'{cor_num:14}|{line_num:14}|'
      pass
    eval_str += '\n'

    return eval_str

  def eval_1img_1cls(self, img, det_lines, gt_lines, scene_name, det_cat, det_points):
    num_gt = gt_lines.shape[0]

    det_corners, cor_scores, det_cor_ids_per_line,_ = gen_corners_from_lines_np(det_lines[:,:self.obj_dim],\
                                          None, self.obj_rep)
    gt_corners, _, gt_corIds_per_line,_ = gen_corners_from_lines_np(gt_lines, None, self.obj_rep)

    cor_nums_gt_pos_tp, cor_detIds_per_gt = self.eval_corners(gt_corners, det_corners)

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
      line_detIds_per_gt.append(det_lineIds)
    line_detIds_per_gt = np.array(line_detIds_per_gt)

    num_ture_pos_line = (line_detIds_per_gt>=0).sum()
    line_nums_gt_pos_tp = [gt_lines.shape[0], det_lines.shape[0], num_ture_pos_line]

    if 1:
      #det_lines_pos, det_lines_neg, det_corners_pos, det_corners_neg, gt_lines_true, gt_lines_false
      eval_draws = self.save_eval_res_img_1cls(img, det_lines, gt_lines, det_corners, gt_corners,
                            cor_detIds_per_gt, line_detIds_per_gt,
                            cor_nums_gt_pos_tp, scene_name, det_cat, det_points)
      #self.debug_line_eval(det_lines, gt_lines, line_detIds_per_gt)
    return cor_nums_gt_pos_tp, line_nums_gt_pos_tp, eval_draws

  def eval_corners(self, gt_corners, det_corners):
    '''
    gt_corners: [n,2]
    det_corners: [m,2]
    cor_scores: [m]

    A gt corner gt_i is successfully detected by det_j, when both of follownig two matches:
      1. det_j is the cloest to gt_i, and the distance is below corner_dis_threshold
      2. gt_i is the cloest to det_j

    nums_gt_pos_tp: [3]
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
    nums_gt_pos_tp = [num_gt, num_pos, num_ture_pos]
    return nums_gt_pos_tp, detIds_per_gt_2

  def debug_line_eval(self, det_lines, gt_lines, line_detIds_per_gt, obj_wise=1):
    pos_line_ids = line_detIds_per_gt[line_detIds_per_gt>=0]
    det_lines_pos = det_lines[pos_line_ids]
    #_show_lines_ls_points_ls((512,512), [gt_lines, det_lines[:,:5], det_lines_pos[:,:5]], line_colors=['white','yellow', 'green'], line_thickness=[1,1,2])
    if obj_wise:
      for i in range(line_detIds_per_gt.shape[0]):
        j = line_detIds_per_gt[i]
        if j>=0:
          _show_lines_ls_points_ls((512,512), [gt_lines[i:i+1], det_lines[j:j+1,:5]], line_colors=['white', 'green'])
    pass

  def save_eval_res_img_1cls(self, img, det_lines, gt_lines, det_corners, gt_corners,
                        cor_detIds_per_gt, line_detIds_per_gt,
                        cor_nums_gt_pos_tp,  scene_name, det_cat,
                        det_points, obj_wise=0):
    if cor_nums_gt_pos_tp[0] == 0:
      cor_recall = 1
    else:
      cor_recall = cor_nums_gt_pos_tp[2]/cor_nums_gt_pos_tp[0]
    if cor_nums_gt_pos_tp[1] == 0:
      cor_precision=1
    else:
      cor_precision = cor_nums_gt_pos_tp[2]/cor_nums_gt_pos_tp[1]
    print(f'\ncor_nums_gt_pos_tp: {cor_nums_gt_pos_tp}')
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
    det_points_pos = det_points[pos_line_ids]
    if len(neg_line_ids) == 0:
      det_lines_neg = det_lines[0:0]
      det_points_neg = det_points[0:0]
    else:
      det_lines_neg = det_lines[neg_line_ids]
      det_points_neg = det_points[neg_line_ids]


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
      _show_lines_ls_points_ls(img_size, [gt_lines_true, gt_lines_false],
                            points_ls=[gt_corners_true, gt_corners_false],
                            line_colors=['green','red'],
                            line_thickness=1,
                            point_colors=['blue', 'yellow'],
                            point_thickness=2, out_file=img_file, only_save=1)

    img_name = f'{scene_name}_{cat}_Recall_0d{r}_Precision_0d{p}_Det.png'
    img_file = os.path.join(self.eval_dir, img_name)
    if SHOW_EACH_CLASS:
      _show_lines_ls_points_ls(img_size, [det_lines], [det_corners],
                             line_colors='random', point_colors='random',
                             line_thickness=1, point_thickness=2,
                             out_file=img_file, only_save=1)

    # with input
    if  not self.is_pcl:
      img_name = f'{scene_name}_{cat}_Recall_0d{r}_Precision_0d{p}_EvalGt_wiht_input.png'
      img_file = os.path.join(self.eval_dir, img_name)
      if SHOW_EACH_CLASS:
        _show_lines_ls_points_ls(img[:,:,0], [gt_lines_true, gt_lines_false],
                              points_ls=[gt_corners_true, gt_corners_false],
                              line_colors=['green','red'],
                              line_thickness=1,
                              point_colors=['blue', 'yellow'],
                              point_thickness=2, out_file=img_file, only_save=1)

      img_name = f'{scene_name}_{cat}_Recall_0d{r}_Precision_0d{p}_EvalDet_with_input.png'
      img_file = os.path.join(self.eval_dir, img_name)
      if SHOW_EACH_CLASS:
        _show_lines_ls_points_ls(img[:,:,0], [det_lines_pos, det_lines_neg],
                                points_ls=[det_corners_pos, det_corners_neg],
                              line_colors=['green', 'red'], line_thickness=1,
                              point_colors=['blue', 'yellow'], point_thickness=2,
                              out_file=img_file, only_save=1)

    img_name = f'{scene_name}_{cat}_Recall_0d{r}_Precision_0d{p}'
    img_file_base_all_cls = os.path.join(self.eval_dir_all_cls, img_name)
    pass

    if obj_wise:
      for i in range(gt_corners.shape[0]):
        j = cor_detIds_per_gt[i]
        if j >= 0:
          _show_lines_ls_points_ls((512,512), [gt_lines], [gt_corners[i:i+1], det_corners[j:j+1]], line_colors=['white'], point_colors=['green', 'red'], point_thickness=2)
    pass
    return cat, img, img_file_base_all_cls, det_lines_pos, det_lines_neg, det_corners_pos, det_corners_neg, gt_lines_true, gt_lines_false, gt_corners_true, gt_corners_false, det_points_pos, det_points_neg

def get_z_by_iou(dets, gts, obj_rep):
  from mmdet.core.bbox.geometry import dsiou_rotated_3d_bbox
  import torch
  assert obj_rep ==  'XYZLgWsHA'
  dets_t = torch.from_numpy( dets[:,:7] ).to(torch.float32)
  gts_t = torch.from_numpy(gts).to(torch.float32)
  ious = dsiou_rotated_3d_bbox( dets_t, gts_t, 0.8, only_2d=True ).numpy()
  for i in range(dets.shape[0]):
    j = np.argmax(ious[i])
    print(ious[i,j])
    dets[i, [2,5]] = gts[j, [2,5]]
  return dets

def draw_eval_all_classes_1img(eval_draws_ls, obj_rep ):
  import mmcv
  colors_map = {'wall': 'gray', 'beam':'lime', 'column':'blue', 'door':'cyan',  'window':'purple',  'floor':'yellow', 'ceiling':'gray'}
  num_cats = len(eval_draws_ls)
  img_det = None
  img_det_pts = None
  img_gt = None
  obj_dim = OBJ_REPS_PARSE._obj_dims[obj_rep]

  det_bboxes = []
  gt_bboxes = []
  det_cats = []
  gt_cats = []
  for i in range(num_cats):
      cat, img, img_file_base_all_cls, det_lines_pos, det_lines_neg, det_corners_pos, \
        det_corners_neg, gt_lines_true, gt_lines_false, gt_corners_true, gt_corners_false,\
        det_points_pos, det_points_neg = eval_draws_ls[i]
      c = colors_map[cat]

      det_points_pos = det_points_pos.reshape(-1, 2)
      det_points_neg = det_points_neg.reshape(-1, 2)

      if cat == 'wall':
        walls = np.concatenate([det_lines_pos, det_lines_neg], 0)
      if cat in ['window', 'door']:
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

      if img_det is None:
        img_det = img.shape[:2]
        img_det_pts = img.shape[:2]
        det_file = img_file_base_all_cls + '_Det.png'
        det_pts_file = img_file_base_all_cls + '_DetPts.png'
      img_det = _draw_objs_ls_points_ls(img_det,
              [det_lines_pos[:,:obj_dim], det_lines_neg[:,:obj_dim]],
              obj_rep,
              [det_corners_pos, det_corners_neg],
              obj_colors=c,
              obj_scores_ls = [det_lines_pos[:,obj_dim], det_lines_neg[:,obj_dim]],
              obj_cats_ls = ['', 'F'],
              point_colors=['blue', 'yellow'],
              obj_thickness=[2,2],
              point_thickness=[3,3],
              out_file=None,
              text_colors_ls=['green', 'red'])

      img_det_pts = _draw_objs_ls_points_ls(img_det_pts,
              [det_lines_pos[:,:obj_dim], det_lines_neg[:,:obj_dim]],
              obj_rep,
              [det_points_pos, det_points_neg],
              obj_colors=c,
              obj_scores_ls = [det_lines_pos[:,obj_dim], det_lines_neg[:,obj_dim]],
              obj_cats_ls = ['', 'F'],
              point_colors='blue',
              obj_thickness=[2,2],
              point_thickness=[3,3],
              out_file=None,
              text_colors_ls=['green', 'red'])
      #mmcv.imshow(img_det_pts)

      if img_gt is None:
        img_gt = img.shape[:2]
        img_gt = img[:,:,0]
      gt_file = img_file_base_all_cls + '_Gt.png'
      img_gt = _draw_objs_ls_points_ls(img_gt,
              [gt_lines_true[:,:obj_dim], gt_lines_false[:,:obj_dim]],
              obj_rep,
              [gt_corners_true, gt_corners_false],
              obj_colors=c,
              obj_cats_ls = ['', 'F'],
              point_colors=['blue', 'yellow'],
              obj_thickness=[2,2],
              point_thickness=[3,3],
              out_file=None,
              text_colors_ls=['green', 'red'])
      #mmcv.imshow(img_gt)
      pass

  mmcv.imwrite(img_det, det_file)
  mmcv.imwrite(img_det_pts, det_pts_file)
  mmcv.imwrite(img_gt, gt_file)

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
  mmcv.imwrite(img_detgt, detgt_file)
  #mmcv.imshow(img_detgt)

  if SHOW_3D:
    det_bboxes = get_z_by_iou(det_bboxes, gt_bboxes, obj_rep)

    wall_mask = [c=='wall' for c in det_cats]
    floor_mask = [c=='floor' for c in det_cats]
    walls = det_bboxes[wall_mask][:,:-1]
    floors0 = det_bboxes[floor_mask][:,:-1]
    floors = get_cf_from_wall(floors0, walls, obj_rep, 'floor')
    _show_3d_points_objs_ls( objs_ls=[gt_bboxes, gt_bboxes], obj_rep=obj_rep, obj_colors=[gt_colors, 'maroon'], box_types= ['surface_mesh', 'line_mesh'] )
    _show_3d_points_objs_ls( objs_ls=[det_bboxes[:,:-1], det_bboxes[:,:-1]], obj_rep=obj_rep, obj_colors=[det_colors, 'maroon'], box_types=['surface_mesh', 'line_mesh'] )
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    pass

  pass

def show_2dlines_as_3d(lines_2d, labels):
  lines_3d = OBJ_REPS_PARSE.lines2d_to_lines3d(lines_2d)
  _show_3d_points_objs_ls(objs_ls=[lines_3d], obj_colors=[labels])
  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  pass

def filter_low_score_det(det_lines, score_threshold=0.5):
  mask = det_lines[:,-1] > score_threshold
  det_lines_ = det_lines[mask]
  return det_lines_

def apply_mask_on_ids(ids, mask):
  return (ids + 1) * mask - 1


def main():
  workdir = '/home/z/Research/mmdetection/work_dirs/'
  dirname = 'sTPV_r50_fpn_Rect4CornersZ0Z1_Apts4_stanford2d_wabecodowifl_bs7_lr10_LsW510R2P1N1_Rfiou741_Fpn44_Pbs1_Bp32_Fe_AreaL123456'
  filename = 'detection_11_Imgs.pickle'
  res_file = os.path.join( os.path.join(workdir, dirname), filename)
  eval_graph(res_file)

if __name__ == '__main__'  :
  main()




