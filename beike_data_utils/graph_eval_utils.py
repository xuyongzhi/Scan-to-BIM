import pickle
from beike_data_utils.beike_utils import load_gt_lines_bk
from tools.debug_utils import _show_lines_ls_points_ls
from configs.common import clean_bboxes_out, OBJ_REP, IMAGE_SIZE, OPT_GRAPH_COR_DIS_THR
from beike_data_utils.line_utils import gen_corners_from_lines_np, get_lineIdsPerCor_from_corIdsPerLine, optimize_graph
from collections import defaultdict
import os
import numpy as np

def save_res_graph(dataset, data_loader, results, out_file):
    num_imgs = len(results)
    assert len(data_loader) == num_imgs
    results_datas = []
    for i_img, data in enumerate(data_loader):
        is_pcl = 'input_style' in data['img_meta'][0] and data['img_meta'][0]['input_style'] == 'pcl'
        if not is_pcl:
          img_i = data['img'][0][0].permute(1,2,0).cpu().data.numpy()
          img_meta_i = data['img_meta'][0].data[0][0]
        else:
          img_meta_i = data['img_meta'][0]
          img_shape = img_meta_i['img_shape']
          img_i = np.zeros(img_shape, dtype=np.int8)
        res_data = dict(  img_id = i_img,
                          img=img_i,
                          img_meta=img_meta_i,
                        )
        #img_id = dataset.img_ids[i_img]
        #assert img_id == i_img
        result = results[i_img]
        det_result = result['det_bboxes']
        res_data['gt_bboxes'] = [ b.cpu().data.numpy() for b in result['gt_bboxes']]
        res_data['gt_labels'] = [ b.cpu().data.numpy() for b in result['gt_labels']]

        detections_all_labels = []
        for label in range(len(det_result)):
          det_lines_multi_stages = det_result[label]
          det_lines = det_lines_multi_stages
          #det_lines = clean_bboxes_out(det_lines_multi_stages, 'final')
          category_id = dataset.cat_ids[label]
          detection_l = {'det_lines': det_lines, 'category_id': category_id}
          detections_all_labels.append(detection_l)
        res_data['detections'] = detections_all_labels
        results_datas.append( res_data )

    with open(out_file, 'wb') as f:
      pickle.dump(results_datas, f)
      print(f'\nsave: {out_file}')

    eval_graph(results_datas, dataset, out_file)
    return results_datas

def eval_graph(results_datas, dataset, out_file):
  graph_eval = GraphEval()
  graph_eval(results_datas, dataset, out_file)


class GraphEval():
  _eval_img_size = 512
  _all_out_types = [ 'composite', 'line_ave', 'score_refine' ]
  _score_threshold  = 0.4
  _corner_dis_threshold = 15
  _opt_graph_cor_dis_thr = OPT_GRAPH_COR_DIS_THR

  def __init__(self):
    pass

  def __str__(self):
    s = self._score_threshold
    par_str =  f'Eval parameters:\nimage size:{self._img_size}\n'
    par_str += f'Out type:{self.out_type}\n'
    par_str += f'Graph optimization corner distance threshold:{self._opt_graph_cor_dis_thr}\n'
    par_str += f'Positive score threshold:{s}\n'
    par_str += f'Positive corner distance threshold:{self._corner_dis_threshold}\n'
    par_str += '\n'
    return par_str

  def update_path(self, out_file):
    self.work_dir = os.path.dirname(out_file)
    s = int(self._score_threshold*10)
    self.par_nice = f'Score{s}_corDis{self._corner_dis_threshold}_optGraph{self._opt_graph_cor_dis_thr}_{self.out_type}'
    self.eval_dir = os.path.join(self.work_dir, 'Eval_' + self.par_nice + f'_{self.num_img}Imgs')
    if not os.path.exists(self.eval_dir):
      os.makedirs(self.eval_dir)

  def __call__(self, results_datas, dataset, out_file):
    for out_type in self._all_out_types:
      self.out_type = out_type
      self.evaluate(results_datas, dataset, out_file, out_type)

  def evaluate(self, results_datas, dataset, out_file, out_type):
    self.num_img = len(results_datas)
    self.update_path(out_file)
    all_cor_nums_gt_pos_tp = defaultdict(list)
    all_line_nums_gt_pos_tp = defaultdict(list)
    corner_recall_precision = defaultdict(list)
    line_recall_precision = defaultdict(list)
    self._catid_2_cat = dataset._catid_2_cat

    self.is_pcl = 'input_style' in results_datas[0]['img_meta'] and results_datas[0]['img_meta']['input_style'] == 'pcl'
    if self.is_pcl:
      self._img_size = tuple(results_datas[0]['img_meta']['img_shape'])
    else:
      self._img_size = (IMAGE_SIZE, IMAGE_SIZE, 3)
    self.scale_ratio = self._eval_img_size / self._img_size[0]

    for i_img, res_data in enumerate(results_datas):
        detections = res_data['detections']
        img = res_data['img']
        img_meta = res_data['img_meta']
        is_pcl = 'input_style' in img_meta and img_meta['input_style'] == 'pcl'
        if not is_pcl:
          img_mean = img_meta['img_norm_cfg']['mean']
          img_std = img_meta['img_norm_cfg']['std']
          img = img * img_std + img_mean

        filename =  img_meta['filename']
        scene_name = os.path.splitext(os.path.basename(filename))[0]

        #if not is_pcl:
        #  gt_lines = load_gt_lines_bk(img_meta, img)
        #else:
        #  import pdb; pdb.set_trace()  # XXX BREAKPOINT
        #  pass

        gt_lines = results_datas[i_img]['gt_bboxes'][0].copy()
        gt_lines[:,:4] *= self.scale_ratio
        #_show_lines_ls_points_ls(img[:,:,0], [gt_lines])

        num_labels = len(detections)
        for label in range(num_labels):
            det_lines_raw = detections[label]['det_lines'].copy()
            det_lines = clean_bboxes_out(det_lines_raw, stage='final', out_type=out_type)
            #_show_lines_ls_points_ls(img[:,:,0], det_lines[:,:5])
            det_lines = filter_low_score_det(det_lines, self._score_threshold)

            labels_i = np.ones(det_lines.shape[0])*label
            scores_i = det_lines[:,-1]
            det_lines[:,:4] *= self.scale_ratio
            det_lines_merged, scores_merged, _ = optimize_graph(det_lines[:,:5], scores_i, labels_i, OBJ_REP, opt_graph_cor_dis_thr=self._opt_graph_cor_dis_thr)
            det_lines_merged = np.concatenate([det_lines_merged, scores_merged], axis=1)

            #_show_lines_ls_points_ls(img[:,:,0], det_lines[:,:5])
            det_category_id = detections[label]['category_id']
            if det_category_id != 1:
              raise NotImplementedError
            cor_nums_gt_pos_tp, line_nums_gt_pos_tp = self.eval_1img_1cls(det_lines_merged, gt_lines, scene_name, det_category_id)
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

    eval_res_str = self.get_eval_res_str(corner_recall_precision, line_recall_precision, dataset)
    path = os.path.dirname(out_file)
    eval_path = os.path.join(path, 'eval_res.txt')
    with open(eval_path, 'a') as f:
      f.write(eval_res_str)
    print(eval_res_str)
    return eval_res_str

  def get_eval_res_str(self, corner_recall_precision, line_recall_precision, dataset):
    rotate = False
    dset_name = dataset.__class__.__name__
    if dset_name == 'BeikePclDataset':
      img_meta = dataset[0][3]['img_meta']
      data_aug = img_meta['data_aug']
      rotate_angles = data_aug['rotate_angles']
      rotate = rotate_angles if len(rotate_angles)>0 else False
      pass
    else:
      if len(dataset)>0 and 'rotate_angle' in dataset[0]['img_meta'][0].data:
        angle0 = dataset[0]['img_meta'][0].data['rotate_angle']
        rotate = abs(angle0) > 0 or rotate
      if len(dataset)>1 and 'rotate_angle' in dataset[1]['img_meta'][0].data:
        angle1 = dataset[1]['img_meta'][0].data['rotate_angle']
        rotate = abs(angle1) > 0 or rotate

    eval_str = '\n\n--------------------------------------\n\n' + \
                str(self) + f'num_img: {self.num_img}\n'
    catid_2_cat = dataset._catid_2_cat
    eval_str += f'rotate: {rotate}\n'
    for label in corner_recall_precision:
      cat = catid_2_cat[label]
      recall, precision = corner_recall_precision[label]
      eval_str += f'{cat:6} corner prec-recall: \t{precision:.3} - {recall:.3}\n'
    for label in line_recall_precision:
      cat = catid_2_cat[label]
      recall, precision = line_recall_precision[label]
      eval_str += f'{cat:6} line prec-recall: \t{precision:.3} - {recall:.3}\n'
    return eval_str

  def eval_1img_1cls(self, det_lines, gt_lines, scene_name, det_category_id):
    num_gt = gt_lines.shape[0]

    det_corners, cor_scores, det_cor_ids_per_line,_ = gen_corners_from_lines_np(det_lines[:,:5],\
                                          None, OBJ_REP)
    gt_corners, _, gt_corIds_per_line,_ = gen_corners_from_lines_np(gt_lines, None, OBJ_REP)

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
      self.save_eval_res_imgs(det_lines, gt_lines, det_corners, gt_corners,
                            cor_detIds_per_gt, line_detIds_per_gt,
                            cor_nums_gt_pos_tp, scene_name, det_category_id)
      #self.debug_line_eval(det_lines, gt_lines, line_detIds_per_gt)
    return cor_nums_gt_pos_tp, line_nums_gt_pos_tp

  def eval_corners(self, gt_corners, det_corners):
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
    dis_valid_mask_gt = (mindis_per_gt < self._corner_dis_threshold).astype(np.float32)
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

  def save_eval_res_imgs(self, det_lines, gt_lines, det_corners, gt_corners,
                        cor_detIds_per_gt, line_detIds_per_gt,
                        cor_nums_gt_pos_tp,  scene_name, det_category_id, obj_wise=0):
    cor_recall = cor_nums_gt_pos_tp[2]/cor_nums_gt_pos_tp[0]
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
    pos_line_ids = line_detIds_per_gt[line_detIds_per_gt>=0]
    neg_line_ids = np.array([ i for i in range(det_lines.shape[0] ) if i not in pos_line_ids])
    det_lines_pos = det_lines[pos_line_ids]
    if len(neg_line_ids) == 0:
      det_lines_neg = det_lines[0:0]
    else:
      det_lines_neg = det_lines[neg_line_ids]
    gt_line_true_ids = np.where(line_detIds_per_gt>=0)[0]
    gt_line_false_ids = np.where(line_detIds_per_gt<0)[0]
    gt_lines_true = gt_lines[gt_line_true_ids]
    gt_lines_false = gt_lines[gt_line_false_ids]

    r = int(cor_recall*100)
    p = int(cor_precision*100)
    cat = self._catid_2_cat[det_category_id]


    img_name = f'{scene_name}_{cat}_Recall_0d{r}_Precision_0d{p}_EvalDet.png'
    img_file = os.path.join(self.eval_dir, img_name)
    #print('det corners. green: true pos, red: false pos')
    img_size = (self._eval_img_size, self._eval_img_size)
    _show_lines_ls_points_ls(img_size, [det_lines_pos, det_lines_neg],
                              points_ls=[det_corners_pos, det_corners_neg],
                            line_colors=['green', 'red'], line_thickness=1,
                            point_colors=['blue', 'yellow'], point_thickness=2,
                            out_file=img_file, only_save=1)

    #print('gt  corners. green: true pos, red: false neg')
    img_name = f'{scene_name}_{cat}_Recall_0d{r}_Precision_0d{p}_EvalGt.png'
    img_file = os.path.join(self.eval_dir, img_name)
    _show_lines_ls_points_ls(img_size, [gt_lines_true, gt_lines_false],
                            points_ls=[gt_corners_true, gt_corners_false],
                            line_colors=['green','red'],
                            line_thickness=1,
                            point_colors=['blue', 'yellow'],
                            point_thickness=2, out_file=img_file, only_save=1)

    img_name = f'{scene_name}_{cat}_Recall_0d{r}_Precision_0d{p}_Det.png'
    img_file = os.path.join(self.eval_dir, img_name)
    _show_lines_ls_points_ls(img_size, [det_lines], [det_corners],
                             line_colors='random', point_colors='random',
                             line_thickness=1, point_thickness=2,
                             out_file=img_file, only_save=1)
    pass

    if obj_wise:
      for i in range(gt_corners.shape[0]):
        j = cor_detIds_per_gt[i]
        if j >= 0:
          _show_lines_ls_points_ls((512,512), [gt_lines], [gt_corners[i:i+1], det_corners[j:j+1]], line_colors=['white'], point_colors=['green', 'red'], point_thickness=2)
    pass


def filter_low_score_det(det_lines, score_threshold=0.5):
  assert det_lines.shape[1] == 6
  mask = det_lines[:,-1] > score_threshold
  det_lines_ = det_lines[mask]
  return det_lines_

def apply_mask_on_ids(ids, mask):
  return (ids + 1) * mask - 1

