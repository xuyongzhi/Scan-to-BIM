
OBJ_REP = 'lscope_istopleft'

#OBJ_REP = 'line_scope'
OUT_PTS_DIM = 18  # 0 or 18
OUT_PTS_DIM = 0 if OUT_PTS_DIM < 0 else OUT_PTS_DIM
#*******************************************************************************
_all_obj_rep_dims = {'box_scope': 4, 'line_scope': 4, 'lscope_istopleft':5}
OBJ_DIM = _all_obj_rep_dims[OBJ_REP]


