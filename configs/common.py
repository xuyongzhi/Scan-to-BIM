import torch
#IMAGE_SIZE = 1024
IMAGE_SIZE = 512
#TRAIN_NUM=1
TRAIN_NUM=90

#DATA = 'coco'
DATA = 'beike'

if DATA == 'beike':
  OBJ_REP = 'lscope_istopleft'
if DATA == 'coco':
  OBJ_REP = 'box_scope'


#*******************************************************************************
_all_obj_rep_dims = {'box_scope': 4, 'line_scope': 4, 'lscope_istopleft':5}
OBJ_DIM = _all_obj_rep_dims[OBJ_REP]
# 47: [bbox_refine, bbox_init, points_refine, points_init, score]
# ref: get_bboxes_single in strpoints_head.py
# [5,           5,        18,             18, 1]
POINTS_NUM = 9
POINTS_DIM = POINTS_NUM * 2 * 2
OUT_EXTAR_DIM = POINTS_DIM + 2 + OBJ_DIM # 43
# see OUT_ORDER in strpoints_head.py/get_bboxes  L1029
OUT_ORDER = {'bbox_refine':[0,0+5], 'bbox_init':[5,5+5],
             'points_refine':[10,10+18], 'points_init':[28,28+18],
             'score_refine':[46,46+1], 'score_final':[47,47+1],
             'score_ave': [48,48+1]}
#[4: cls, cen, ofs]
# [1,1,2]
OUT_CORNER_HM_ONLY = 0
LINE_CLS_WEIGHTS = {'refine':0.7, 'final':0.3}
#*******************************************************************************

CORNER_FLAG = 100
INCLUDE_CORNERS = False

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

def parse_bboxes_out(bboxes_out):
  if isinstance(bboxes_out, torch.Tensor):
    assert bboxes_out.dim() == 2
  else:
    assert bboxes_out.ndim == 2
  c = bboxes_out.shape[1]
  assert c == OUT_EXTAR_DIM + OBJ_DIM + 1 or c == OUT_EXTAR_DIM + OBJ_DIM
  outs = {}
  for key in OUT_ORDER:
    s, e = OUT_ORDER[key]
    if key == 'score_ave' and  c == OUT_EXTAR_DIM + OBJ_DIM:
      outs[key] = None
    else:
      outs[key] = bboxes_out[:, s:e]
  bboxes_refine, bboxes_init, points_refine, points_init, score_refine, score_final, score_ave =  \
          outs['bbox_refine'], outs['bbox_init'], outs['points_refine'], \
          outs['points_init'], outs['score_refine'], outs['score_final'],\
          outs['score_ave']
  return bboxes_refine, bboxes_init, points_refine, points_init, score_refine, score_final, score_ave
