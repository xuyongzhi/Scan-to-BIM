import torch
#IMAGE_SIZE = 1024
IMAGE_SIZE = 512
#TRAIN_NUM=1
TRAIN_NUM=90

#DATA = 'coco'
DATA = 'beike'

if DATA == 'beike':
  OBJ_REP = 'lscope_istopleft'
  NUM_CLASS = 1
if DATA == 'coco':
  OBJ_REP = 'box_scope'
  NUM_CLASS = 80


#*******************************************************************************
_all_obj_rep_dims = {'box_scope': 4, 'line_scope': 4, 'lscope_istopleft':5}
OBJ_DIM = _all_obj_rep_dims[OBJ_REP]
POINTS_NUM = 9
POINTS_DIM = POINTS_NUM * 2 * 2
OUT_EXTAR_DIM = POINTS_DIM + OBJ_DIM + 2 # see OUT_ORDER
OUT_EXTAR_DIM = 0
CORNER_DIM = 4
AVE_LINE_SCORE = 1
COMPOSITE_SCORE = 1
# 5+43+1+4+1 = 54
if DATA == 'beike':
  OUT_DIM_FINAL = OBJ_DIM + OUT_EXTAR_DIM + AVE_LINE_SCORE + CORNER_DIM + COMPOSITE_SCORE
if DATA == 'coco':
  OUT_DIM_FINAL = OBJ_DIM + OUT_EXTAR_DIM + COMPOSITE_SCORE

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
#*******************************************************************************

MIN_BOX_SIZE = 5.0 * IMAGE_SIZE / 1024
PRINT_POINT_ASSIGNER = 0
PRINT_IOU_ASSIGNER = 0

CHECK_POINT_ASSIGN = True
LOAD_GT_TEST = True
#*******************************************************************************

OBJ_LEGEND = 'score' # score or rotation
#OBJ_LEGEND = 'rotation'
#*******************************************************************************

def parse_bboxes_out(bboxes_out, flag):
  assert flag in ['before_nms', 'after_nms', 'before_cal_score_composite']
  if isinstance(bboxes_out, torch.Tensor):
    assert bboxes_out.dim() == 2
  else:
    assert bboxes_out.ndim == 2
  if flag == 'before_nms':
    assert bboxes_out.shape[1] == OUT_DIM_FINAL - 1 - CORNER_DIM - 1
  elif flag == 'before_cal_score_composite':
    assert bboxes_out.shape[1] == OUT_DIM_FINAL -1
  elif flag == 'after_nms':
    assert bboxes_out.shape[1] == OUT_DIM_FINAL
  c = bboxes_out.shape[1]
  outs = {}
  for key in OUT_ORDER:
    s, e = OUT_ORDER[key]
    if bboxes_out.shape[1] < e:
      outs[key] = None
    else:
      outs[key] = bboxes_out[:, s:e]
  bboxes_refine, bboxes_init, points_refine, points_init, score_refine, score_final, score_line_ave, corner0_score, corner1_score, corner0_center, corner1_center, score_composite =  \
          outs['bbox_refine'], outs['bbox_init'], outs['points_refine'], \
          outs['points_init'], outs['score_refine'], outs['score_final'],\
          outs['score_line_ave'], outs['corner0_score'], outs['corner1_score'], outs['corner0_center'], outs['corner1_center'], outs['score_composite']
  return bboxes_refine, bboxes_init, points_refine, points_init, score_refine, score_final, score_line_ave, corner0_score, corner1_score, corner0_center, corner1_center, score_composite
