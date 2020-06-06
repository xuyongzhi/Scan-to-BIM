import os
import open3d as o3d
import numpy as np
from tools.visual_utils import _show_3d_points_objs_ls

def show_1file(pred_f, gt_f, box_dir):
  from utils_dataset.stanford3d_utils.preprocess_stanford3d import gen_box_1_scene
  scene_name = os.path.basename(pred_f).split('.')[0]
  bbox_file = os.path.join(box_dir, scene_name + '.npy')

  preds = np.loadtxt(pred_f)
  xyz = preds[:,:3]
  colors = preds[:,3:6]
  scores =  preds[:,6]
  cls_labels = preds[:,7].astype(np.int)
  ins_labels =  preds[:,8].astype(np.int)
  #_show_3d_points_objs_ls([points[:,:3]], [cls_labels])

  max_num_points = None
  max_num_points = 10000 * 4
  gen_box_1_scene(xyz, colors, cls_labels, ins_labels, scene_name, max_num_points, bbox_file)
  pass

def main():
  path = '/home/z/Research/ASIS/models/ASIS/log2_test/test_results_0'
  path = './test_results_5_good_samples'
  box_dir = './bboxes'

  if not os.path.exists(box_dir):
    os.makedirs(box_dir)

  scenes = ['Area_2_conferenceRoom_1', 'Area_2_auditorium_1', 'Area_2_hallway_11', 'Area_4_hallway_3', 'Area_4_lobby_2', 'Area_3_office_4']
  scenes = scenes[ 1:2 ]
  for s in scenes:
    gt_f = os.path.join(path, s+'_gt.txt')
    pred_f = os.path.join(path, s+'_pred.txt')
    show_1file(pred_f, gt_f, box_dir)

if __name__ == '__main__':
  main()
