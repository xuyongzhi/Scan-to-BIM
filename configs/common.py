IMAGE_SIZE = 512
OBJ_REP = 'lscope_istopleft'

#OBJ_REP = 'line_scope'

OUT_EXTAR_DIM = 9*2 * 2 + 5 #18  # 0 or 18 or 36
#*******************************************************************************
_all_obj_rep_dims = {'box_scope': 4, 'line_scope': 4, 'lscope_istopleft':5}
OBJ_DIM = _all_obj_rep_dims[OBJ_REP]
#*******************************************************************************

CORNER_FLAG = 100
INCLUDE_CORNERS = False

#*******************************************************************************
MIN_BOX_SIZE = 5.0 * IMAGE_SIZE / 1024
PRINT_POINT_ASSIGNER = 1
PRINT_IOU_ASSIGNER = 1
MIN_POINT_ASSIGN_DIST = 2

# for coco
MIN_BOX_SIZE = 0
MIN_POINT_ASSIGN_DIST = 1000
#*******************************************************************************

