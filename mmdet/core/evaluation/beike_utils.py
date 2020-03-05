import pickle
from beike_data_utils.beike_utils import load_gt_lines_bk
from mmdet.debug_tools import show_img_lines
from configs.common import clean_bboxes_out, OBJ_REP
from beike_data_utils.line_utils import gen_corners_from_lines_np, get_lineIdsPerCor_from_corIdsPerLine
from mmdet.debug_tools import _show_lines_ls_points_ls
from collections import defaultdict
import os
import numpy as np

class EvalParameter():
  score_threshold  = 0.5
  corner_dis_threshold = 3

def save_res_bk(dataset, data_loader, results, out_file):
    num_imgs = len(results)
    assert len(data_loader) == num_imgs
    results_datas = []
    for i_img, data in enumerate(data_loader):
        res_data = dict(  img_id = i_img,
                          img=data['img'][0][0].permute(1,2,0).cpu().data.numpy(),
                          img_meta=data['img_meta'][0].data[0][0],
                        )
        img_id = dataset.img_ids[i_img]
        assert img_id == i_img
        result = results[i_img]
        detections_all_labels = []
        for label in range(len(result)):
          det_lines_multi_stages = result[label]
          det_lines = clean_bboxes_out(det_lines_multi_stages, 'final')
          category_id = dataset.cat_ids[label]
          detection_l = {'det_lines': det_lines, 'category_id': category_id}
          detections_all_labels.append(detection_l)
        res_data['detections'] = detections_all_labels
        results_datas.append( res_data )

    with open(out_file, 'wb') as f:
      pickle.dump(results_datas, f)
      print(f'\nsave: {out_file}')
    return results_datas

def filter_low_score_det(det_lines, score_threshold=0.5):
  assert det_lines.shape[1] == 6
  mask = det_lines[:,-1] > score_threshold
  det_lines_ = det_lines[mask]
  return det_lines_

def eval_bk(results_datas, dataset, out_file, evalPara=EvalParameter()):
    all_cor_nums_gt_pos_tp = defaultdict(list)
    all_line_nums_gt_pos_tp = defaultdict(list)
    corner_recall_precision = defaultdict(list)
    line_recall_precision = defaultdict(list)

    for i_img, res_data in enumerate(results_datas):
        detections = res_data['detections']
        img = res_data['img']
        img_meta = res_data['img_meta']
        img_mean = img_meta['img_norm_cfg']['mean']
        img_std = img_meta['img_norm_cfg']['std']
        img = img * img_std + img_mean

        gt_lines = load_gt_lines_bk(img_meta, img)
        #show_img_lines(img[:,:,0], gt_lines)

        num_labels = len(detections)
        for label in range(num_labels):
            det_lines = detections[label]['det_lines']
            #show_img_lines(img[:,:,0], det_lines[:,:5])
            det_lines = filter_low_score_det(det_lines, evalPara.score_threshold)
            #show_img_lines(img[:,:,0], det_lines[:,:5])
            det_category_id = detections[label]['category_id']
            if det_category_id != 1:
              raise NotImplementedError
            cor_nums_gt_pos_tp, line_nums_gt_pos_tp = eval_1img_1cls(det_lines, gt_lines, evalPara)
            all_cor_nums_gt_pos_tp[label].append(cor_nums_gt_pos_tp)
            all_line_nums_gt_pos_tp[label].append(line_nums_gt_pos_tp)
            pass

    for label in all_cor_nums_gt_pos_tp:
      all_cor_nums_gt_pos_tp[label] = np.array(all_cor_nums_gt_pos_tp[label])
      all_line_nums_gt_pos_tp[label] = np.array(all_line_nums_gt_pos_tp[label])

      cor = all_cor_nums_gt_pos_tp[label].sum(axis=0)
      line = all_line_nums_gt_pos_tp[label].sum(axis=0)

      corner_recall_precision[label + 1] = [cor[2]/cor[0], cor[2]/cor[1]]
      line_recall_precision[label + 1] = [line[2]/line[0], line[2]/line[1]]

    eval_res_str = get_eval_res_str(corner_recall_precision, line_recall_precision, dataset)
    path = os.path.dirname(out_file)
    eval_path = os.path.join(path, 'eval_res.txt')
    with open(eval_path, 'w') as f:
      f.write(eval_res_str)
    print(eval_res_str)
    return eval_res_str

def get_eval_res_str(corner_recall_precision, line_recall_precision, dataset):
  eval_str = ''
  catid_2_cat = dataset.beike._catid_2_cat
  for label in corner_recall_precision:
    cat = catid_2_cat[label]
    recall, precision = corner_recall_precision[label]
    eval_str += f'{cat:6}\tcorner\trecall:{recall:.3}\tprecision:{precision:.3}\n'
  for label in line_recall_precision:
    cat = catid_2_cat[label]
    recall, precision = line_recall_precision[label]
    eval_str += f'{cat:6}\tline \trecall:{recall:.3}\tprecision:{precision:.3}\n'
  return eval_str

def apply_mask_on_ids(ids, mask):
  return (ids + 1) * mask - 1

def eval_1img_1cls(det_lines, gt_lines, evalPara):
  num_gt = gt_lines.shape[0]

  gt_corners, _, gt_corIds_per_line = gen_corners_from_lines_np(gt_lines, None, OBJ_REP)
  det_corners, cor_scores, det_cor_ids_per_line = gen_corners_from_lines_np(det_lines[:,:5], det_lines[:,-1:], OBJ_REP)

  cor_nums_gt_pos_tp, cor_detIds_per_gt = eval_corners(gt_corners, det_corners, evalPara.corner_dis_threshold)

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

  #debug_line_eval(det_lines, gt_lines, line_detIds_per_gt)
  #debug_corner_eval(det_lines, gt_lines, det_corners, gt_corners, cor_detIds_per_gt)
  return cor_nums_gt_pos_tp, line_nums_gt_pos_tp

def eval_corners(gt_corners, det_corners, corner_dis_threshold=5):
  '''
  gt_corners: [n,2]
  det_corners: [m,2]
  cor_scores: [m]

  A gt corner gt_i is successfully detected by det_j, when both of follownig two matches:
    1. det_j is the cloest to gt_i, and the distance is below corner_dis_threshold
    2. gt_i is the cloest to det_j
  '''
  diss = np.linalg.norm(gt_corners[:,None,:] - det_corners[None,:,:], axis=2)
  detIds_per_gt_0 = np.argmin(diss, axis=1)
  mindis_per_gt = diss.min(axis=1)
  dis_valid_mask_gt = (mindis_per_gt < corner_dis_threshold).astype(np.float32)
  detIds_per_gt_1 = apply_mask_on_ids(detIds_per_gt_0, dis_valid_mask_gt).astype(np.int32)

  # check if some gts are matched by multiple detections
  mindis_per_det = diss.min(axis=0)
  gt_ids_per_det = np.argmin(diss, axis=0)
  gt_ids_check = apply_mask_on_ids(gt_ids_per_det[detIds_per_gt_1], dis_valid_mask_gt)
  num_corner = gt_corners.shape[0]
  only_match_one_mask = gt_ids_check == np.arange(num_corner)

  detIds_per_gt_2 = apply_mask_on_ids(detIds_per_gt_1, only_match_one_mask)
  num_ture_pos = (detIds_per_gt_2 >=0).sum()
  num_gt = gt_corners.shape[0]
  num_pos = det_corners.shape[0]
  nums_gt_pos_tp = [num_gt, num_pos, num_ture_pos]
  return nums_gt_pos_tp, detIds_per_gt_2

def debug_line_eval(det_lines, gt_lines, line_detIds_per_gt, obj_wise=0):
  pos_line_ids = line_detIds_per_gt[line_detIds_per_gt>=0]
  det_lines_pos = det_lines[pos_line_ids]
  _show_lines_ls_points_ls((512,512), [gt_lines, det_lines[:,:5], det_lines_pos[:,:5]], line_colors=['white','yellow', 'green'], line_thickness=[1,1,2])
  if obj_wise:
    for i in range(line_detIds_per_gt.shape[0]):
      j = line_detIds_per_gt[i]
      if j>=0:
        _show_lines_ls_points_ls((512,512), [gt_lines[i:i+1], det_lines[j:j+1,:5]], line_colors=['white', 'green'])
  pass

def debug_corner_eval(det_lines, gt_lines, det_corners, gt_corners, cor_detIds_per_gt, obj_wise=0):
  num_det_cors = det_corners.shape[0]
  pos_det_cor_ids = cor_detIds_per_gt[cor_detIds_per_gt>=0]
  neg_det_cor_ids = np.array([i for i in range(num_det_cors) if i not in pos_det_cor_ids], dtype=np.int32)
  det_corners_pos = det_corners[pos_det_cor_ids]
  det_corners_neg = det_corners[neg_det_cor_ids]
  gt_corners_true = gt_corners[cor_detIds_per_gt >= 0]
  gt_corners_false = gt_corners[cor_detIds_per_gt < 0]

  tp_mask = cor_detIds_per_gt >= 0
  #_show_lines_ls_points_ls((512,512), [det_lines[:,:5], gt_lines], line_colors=['green', 'red'])
  print('gt  corners. green: true pos, red: false neg')
  _show_lines_ls_points_ls((512,512), [gt_lines, det_lines[:,:5]], points_ls=[gt_corners_true, gt_corners_false], line_colors=['white','yellow'], point_colors=['green', 'red'], point_thickness=2)
  print('det corners. green: true pos, red: false pos')
  _show_lines_ls_points_ls((512,512), [gt_lines, det_lines[:,:5]], points_ls=[det_corners_pos, det_corners_neg], line_colors=['white','yellow'], point_colors=['green', 'red'], point_thickness=2)

  if obj_wise:
    for i in range(gt_corners.shape[0]):
      j = cor_detIds_per_gt[i]
      if j >= 0:
        _show_lines_ls_points_ls((512,512), [gt_lines], [gt_corners[i:i+1], det_corners[j:j+1]], line_colors=['white'], point_colors=['green', 'red'], point_thickness=2)
  pass

