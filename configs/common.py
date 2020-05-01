import torch
import numpy as np
from obj_geo_utils.obj_utils import OBJ_REPS_PARSE
#*******************************************************************************
class DIM_PARSE:
  IMAGE_SIZE = 512

  POINTS_NUM = 9
  POINTS_DIM = POINTS_NUM * 2 * 2
  # StrPointsHead.cal_test_score()
  LINE_CLS_WEIGHTS = {'refine':0.7, 'final':0.3}

  def __init__(self, obj_rep, num_classes_inc_bg, is_out_corner=False):
    '''
    num_classes: not include background
    '''
    assert num_classes_inc_bg >= 2, "include bg"
    self.OBJ_REP = obj_rep
    self.OBJ_DIM = OBJ_REPS_PARSE._obj_dims[obj_rep]
    self.NUM_CLASS = num_classes_inc_bg
    self.use_sigmoid_cls = True
    if self.use_sigmoid_cls:
      self.PRED_SCORES_DIM = self.NUM_CLASS - 1
    else:
      self.PRED_SCORES_DIM = self.NUM_CLASS
    AVE_LINE_SCORE_DIM = 1

    o = self.OBJ_DIM
    p = self.POINTS_NUM
    s = self.PRED_SCORES_DIM
    # OUT_DIMS and OUT_ORDER ref to
    # strpoints_head.py/StrPointsHead/get_bboxes/Line 1033
    self.OUT_DIMS = [('bbox_refine',   o),
                      ('bbox_init',     o),
                      ('points_refine', p*2),
                      ('points_init',   p*2),
                      ('score_refine',  s),
                      ('score_final',   s),
                      ('score_line_ave',AVE_LINE_SCORE_DIM),
                      ('corner0_score', s * is_out_corner),
                      ('corner1_score', s * is_out_corner),
                      ('corner0_center',s * is_out_corner),
                      ('corner1_center',s * is_out_corner),
                      ('score_composite',1 * is_out_corner),
                      ]
    self.OUT_ORDER = {}
    start_idx = 0
    for i in range(len(self.OUT_DIMS)):
      name, dim = self.OUT_DIMS[i]
      self.OUT_ORDER[name] = (start_idx, start_idx+dim)
      start_idx += dim
    #self.unused_OUT_ORDER = {'bbox_refine':[0,0+5], 'bbox_init':[5,5+5],
    #            'points_refine':[10,10+18], 'points_init':[28,28+18],
    #            'score_refine':[46,46+1], 'score_final':[47,47+1],
    #            'score_line_ave': [48,48+1],
    #            'corner0_score':[49,49+1],'corner1_score':[50,50+1],
    #            'corner0_center':[51,51+1],'corner1_center':[52,52+1],
    #            'score_composite': [53,53+1],
    #            }
    SCORE_REFINE_DIM = self.PRED_SCORES_DIM
    SCORE_FINAL_DIM = self.PRED_SCORES_DIM
    self.OUT_EXTAR_DIM = self.POINTS_DIM + self.OBJ_DIM + SCORE_REFINE_DIM + SCORE_FINAL_DIM # see OUT_ORDER
    self.NMS_IN_DIM = self.OBJ_DIM + self.OUT_EXTAR_DIM
    self.NMS_OUT_DIM = self.NMS_IN_DIM + 1 # add top score
    AVE_LINE_SCORE = 1

    # see OUT_ORDER in strpoints_head.py/get_bboxes  L1029
    self.CORNER_DIM = 4 * is_out_corner

    self.COMPOSITE_SCORE = 1 * is_out_corner
    #self.OUT_DIM_BOX_INDEPENDENT_FINAL = self.OBJ_DIM + self.OUT_EXTAR_DIM + AVE_LINE_SCORE # 48
    #self.OUT_DIM_FINAL = self.OUT_DIM_BOX_INDEPENDENT_FINAL + self.CORNER_DIM + self.COMPOSITE_SCORE
    self.OUT_DIM_FINAL = self.NMS_OUT_DIM + self.CORNER_DIM + self.COMPOSITE_SCORE

  def parse_bboxes_out(self, bboxes_out, stage):
    assert stage in ['before_nms', 'nms_out', 'before_cal_score_composite', 'final']
    if isinstance(bboxes_out, torch.Tensor):
      assert bboxes_out.dim() == 2
    else:
      assert bboxes_out.ndim == 2
    if stage == 'before_nms':
      assert bboxes_out.shape[1] == self.NMS_IN_DIM
    if stage == 'nms_out':
      assert bboxes_out.shape[1] == self.NMS_OUT_DIM
    elif stage == 'before_cal_score_composite':
      assert bboxes_out.shape[1] == self.OUT_DIM_FINAL -1
    elif stage == 'final':
      assert bboxes_out.shape[1] == self.OUT_DIM_FINAL
    c = bboxes_out.shape[1]
    outs = {}
    for key in self.OUT_ORDER:
      s, e = self.OUT_ORDER[key]
      if bboxes_out.shape[1] < e or s == e:
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

  def clean_bboxes_out(self, bboxes_out, stage, out_type='composite'):
    assert out_type in ['score_refine', 'score_final', 'line_ave', 'composite']
    bboxes_refine, bboxes_init, points_refine, points_init, score_refine, score_final,\
      score_line_ave, corner0_score, corner1_score, corner0_center, corner1_center,\
      score_composite = self.parse_bboxes_out(bboxes_out, stage)
    if out_type == 'composite':
      bboxes_clean = np.concatenate([ bboxes_refine, score_composite ], axis=1 )
    elif out_type == 'line_ave':
      bboxes_clean = np.concatenate([ bboxes_refine, score_line_ave ], axis=1 )
    elif out_type == 'score_final':
      score_final = score_final.max(1, keepdims=True)
      bboxes_clean = np.concatenate([ bboxes_refine, score_final ], axis=1 )
    elif out_type == 'score_refine':
      score_refine = score_refine.max(1, keepdims=True)
      bboxes_clean = np.concatenate([ bboxes_init, score_refine ], axis=1 )
    return bboxes_clean



class DEBUG_CFG:
  # tem
  IGNORE_Z = True
  SET_WIDTH_0 = True

  # debug in input
  LOAD_VOXELIZED_SPARSE = [False, 'raw','aug'][0]
  MIN_BOX_SIZE = 5.0 * DIM_PARSE.IMAGE_SIZE / 1024
  # mmdet/datasets$/custom.py
  VISUAL_TOPVIEW_INPUT = 0
  VISUAL_CONNECTIONS = 0

  # training debug inside net
  DISABLE_RESCALE = True      # single_stage.py/simple_test
  LOAD_GTS_IN_TEST = True

  PRINT_POINT_ASSIGNER = 0
  PRINT_IOU_ASSIGNER = 0
  VISUALIZE_POINT_ASSIGNER = 0
  VISUALIZE_IOU_ASSIGNER = 0

  VISUALIZE_VALID_LOSS_SAMPLES = 0

  VISUAL_RESNET_FEAT_OUT = 0
  SHOW_TRAIN_RES = 0
  SHOW_NMS_OUT = 0
  CHECK_POINT_ASSIGN = False
  SHOW_RELATION_IN_TRAIN = 0

  # debug in test
  OBJ_LEGEND = ['score', 'rotation'][0]


  SPARSE_BEV = 0

  OUT_CORNER_HM_ONLY = 0
