import torch
import numpy as np

#IMAGE_SIZE = 1024
IMAGE_SIZE = 512
#TRAIN_NUM=1
TRAIN_NUM=90

DATA = 'coco'
DATA = 'stanford_pcl_2d'
DATA = 'beike2d'
#DATA = 'beike_pcl_2d'

if DATA == 'coco':
  OBJ_REP = 'box_scope'
  NUM_CLASS = 80
else:
  OBJ_REP = 'lscope_istopleft'
  NUM_CLASS = 1
#*******************************************************************************
MAR8_VERSION = 1
OPT_GRAPH_COR_DIS_THR = 10
# net *******************************************************************************
SPARSE_BEV = 0
_all_obj_rep_dims = {'box_scope': 4, 'box3d_scope': 6, 'lscope_istopleft':5}
OBJ_DIM = _all_obj_rep_dims[OBJ_REP]
POINTS_NUM = 9
POINTS_DIM = POINTS_NUM * 2 * 2
SCORE_REFINE_DIM = 1
SCORE_FINAL_DIM = 1
OUT_EXTAR_DIM = POINTS_DIM + OBJ_DIM + SCORE_REFINE_DIM + SCORE_FINAL_DIM # see OUT_ORDER
#OUT_EXTAR_DIM = 0
CORNER_DIM = 4
AVE_LINE_SCORE = 1
COMPOSITE_SCORE = 1
IS_OUT_CORNER = 1
CORNER_DIM = CORNER_DIM * IS_OUT_CORNER
COMPOSITE_SCORE = COMPOSITE_SCORE * IS_OUT_CORNER
# IS_OUT_CORNER=1: 5+43+1+4+1 = 54
# IS_OUT_CORNER=0: 5+43+1+1 = 50
if DATA == 'coco':
  OUT_DIM_FINAL = OBJ_DIM + OUT_EXTAR_DIM + 1
else:
  OUT_DIM_BOX_INDEPENDENT_FINAL = OBJ_DIM + OUT_EXTAR_DIM + AVE_LINE_SCORE # 48
  OUT_DIM_FINAL = OUT_DIM_BOX_INDEPENDENT_FINAL + CORNER_DIM + COMPOSITE_SCORE

#OUT_SCORE_TYPE = ['Line_Ave', 'Corner_Ave', 'Corner_0', 'Corner_1'][1]


# see OUT_ORDER in strpoints_head.py/get_bboxes  L1029
OUT_ORDER = {'bbox_refine':[0,0+5], 'bbox_init':[5,5+5],
             'points_refine':[10,10+18], 'points_init':[28,28+18],
             'score_refine':[46,46+1], 'score_final':[47,47+1],
             'score_line_ave': [48,48+1],
             'corner0_score':[49,49+1],'corner1_score':[50,50+1],
             'corner0_center':[51,51+1],'corner1_center':[52,52+1],
             'score_composite': [53,53+1],
             }
#[4: cls, cen, ofs]
# [1,1,2]
OUT_CORNER_HM_ONLY = 0
LINE_CLS_WEIGHTS = {'refine':0.7, 'final':0.3}
# debug *******************************************************************************
MIN_BOX_SIZE = 5.0 * IMAGE_SIZE / 1024
PRINT_POINT_ASSIGNER = 0
PRINT_IOU_ASSIGNER = 0
VISUALIZE_POINT_ASSIGNER = 0
VISUALIZE_IOU_ASSIGNER = 0

CHECK_POINT_ASSIGN = False
LOAD_GT_TEST = True
VISUAL_RESNET_OUT = 0
VISUAL_TOPVIEW_INPUT = 0
#*******************************************************************************

OBJ_LEGEND = 'score' # score or rotation
#OBJ_LEGEND = 'rotation'
#*******************************************************************************

def parse_bboxes_out(bboxes_out, stage):
  assert stage in ['before_nms', 'before_cal_score_composite', 'final']
  if isinstance(bboxes_out, torch.Tensor):
    assert bboxes_out.dim() == 2
  else:
    assert bboxes_out.ndim == 2
  if stage == 'before_nms':
    assert bboxes_out.shape[1] == OUT_DIM_FINAL - 1 - CORNER_DIM - COMPOSITE_SCORE
  elif stage == 'before_cal_score_composite':
    assert bboxes_out.shape[1] == OUT_DIM_FINAL -1
  elif stage == 'final':
    assert bboxes_out.shape[1] == OUT_DIM_FINAL
  c = bboxes_out.shape[1]
  outs = {}
  for key in OUT_ORDER:
    s, e = OUT_ORDER[key]
    if bboxes_out.shape[1] < e:
      outs[key] = None
    else:
      outs[key] = bboxes_out[:, s:e]
  bboxes_refine, bboxes_init, points_refine, points_init, score_refine, \
      score_final, score_line_ave, corner0_score, corner1_score, \
      corner0_center, corner1_center, score_composite =  \
          outs['bbox_refine'], outs['bbox_init'], outs['points_refine'], \
          outs['points_init'], outs['score_refine'], outs['score_final'],\
          outs['score_line_ave'], outs['corner0_score'], outs['corner1_score'],\
          outs['corner0_center'], outs['corner1_center'], outs['score_composite']
  if score_composite is None:
    score_composite = score_line_ave
  return bboxes_refine, bboxes_init, points_refine, points_init, score_refine,\
    score_final, score_line_ave, corner0_score, corner1_score, corner0_center,\
    corner1_center, score_composite

def clean_bboxes_out(bboxes_out, stage, out_type='composite'):
  assert out_type in ['score_refine', 'score_final', 'line_ave', 'composite']
  bboxes_refine, bboxes_init, points_refine, points_init, score_refine, score_final,\
    score_line_ave, corner0_score, corner1_score, corner0_center, corner1_center,\
    score_composite = parse_bboxes_out(bboxes_out, stage)
  if out_type == 'composite':
    bboxes_clean = np.concatenate([ bboxes_refine, score_composite ], axis=1 )
  elif out_type == 'line_ave':
    bboxes_clean = np.concatenate([ bboxes_refine, score_line_ave ], axis=1 )
  elif out_type == 'score_final':
    bboxes_clean = np.concatenate([ bboxes_refine, score_final ], axis=1 )
  elif out_type == 'score_refine':
    bboxes_clean = np.concatenate([ bboxes_init, score_refine ], axis=1 )
  return bboxes_clean
