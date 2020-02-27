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
OUT_EXTAR_DIM = 9*2 * 2 + OBJ_DIM #18  # 0 or 18 or 36
#*******************************************************************************

CORNER_FLAG = 100
INCLUDE_CORNERS = False

#*******************************************************************************
MIN_BOX_SIZE = 5.0 * IMAGE_SIZE / 1024
PRINT_POINT_ASSIGNER = 0
PRINT_IOU_ASSIGNER = 0

CHECK_POINT_ASSIGN = True
#*******************************************************************************

OBJ_LEGEND = 'score' # score or rotation
#OBJ_LEGEND = 'rotation'
#*******************************************************************************
