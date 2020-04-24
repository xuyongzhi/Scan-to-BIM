import torch.nn as nn
import torch
import numpy as np

from mmdet.core import bbox2result
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from utils_dataset.gen_sparse_input import update_img_shape_for_pcl

from tools import debug_utils
from tools.visual_utils import _show_objs_ls_points_ls
import time
from configs.common import DEBUG_CFG
SHOW_TRAIN_RES = DEBUG_CFG.SHOW_TRAIN_RES

RECORD_T = 0

@DETECTORS.register_module
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

        if 'stem_stride' in backbone:
          self.stem_stride = backbone['stem_stride']

        self.point_strides = bbox_head['point_strides']
        if 0:
          print('\n\nneck:')
          print(self.backbone)
          print('\n\nneck:')
          print(self.neck)
          print('\n\nbbox_head:')
          print(self.bbox_head)
          import pdb; pdb.set_trace()  # XXX BREAKPOINT
        pass

    def init_weights(self, pretrained=None):
        super(SingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img, gt_bboxes):
        """Directly extract features from the backbone+neck
        """
        if RECORD_T:
          t0 = time.time()
        x = self.backbone(img, gt_bboxes)
        if RECORD_T:
          t1 = time.time()
        if self.with_neck:
            x = self.neck(x)
        if RECORD_T:
          t2 = time.time()
          print(f'\n\n\tbackbone: {t1-t0:.3f}\tneck:{t2-t1:.3f}')
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmedetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def update_dynamic_shape(self, x, img_metas):
        input_style = img_metas[0]['input_style']
        assert input_style in ['pcl', 'bev_sparse', 'img']

        if not input_style == 'pcl':
          return
        feat_sizes = [ np.array([*xi.shape[2:]]) for xi in x]
        for img_meta in img_metas:
          img_meta['stem_stride'] = self.stem_stride
          img_meta['feat_sizes'] = feat_sizes
          img_meta['dynamic_img_shape'] = feat_sizes[0] * self.stem_stride

        update_img_shape_for_pcl(x, img_metas[0], self.point_strides)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_relations = None,
                      gt_bboxes_ignore=None):
        '''
        img: [6, 4, 512, 512]
        img_metas: [{}]*6
        gt_bboxes: [ni*5]*6
        gt_labels: [ni]*6
        '''
        if RECORD_T:
          t0 = time.time()
        #_show_objs_ls_points_ls(img[0].permute(1,2,0).cpu().data.numpy(), [gt_bboxes[0].cpu().data.numpy()], 'RoLine2D_UpRight_xyxy_sin2a')
        x = self.extract_feat(img, gt_bboxes)
        self.update_dynamic_shape(x, img_metas)
        #debug_utils._show_tensor_ls_shapes(x, 'single_stage forward_train - features')
        if RECORD_T:
          t1 = time.time()
        outs = self.bbox_head(x)
        if RECORD_T:
          t2 = time.time()
        losses = self.bbox_head.loss( *outs,
            gt_bboxes=gt_bboxes, gt_labels=gt_labels, gt_relations=gt_relations,
            img_metas=img_metas, cfg=self.train_cfg,  gt_bboxes_ignore=gt_bboxes_ignore)
        if RECORD_T:
          t3 = time.time()
          print(f'\textract feat:{t1-t0:.3f} head:{t2-t1:.3f}, loss:{t3-t2:.3f}')

        if 0:
          debug_utils._show_sparse_ls_shapes(img, 'single_stage forward_train - img')
          debug_utils._show_tensor_ls_shapes(x, 'single_stage forward_train - features')
          for i in range(len(outs)):
            for j in range(len(outs[i])):
              if isinstance(outs[i][j], torch.Tensor):
                debug_utils._show_tensor_ls_shapes(outs[i][j], f'single_stage forward_train - outs {i},{j}')
              elif isinstance(outs[i][j], dict):
                for key in outs[i][j]:
                  debug_utils._show_tensor_ls_shapes([outs[i][j][key]], f'single_stage forward_train - outs {i},{j},{key}')
              else:
                assert outs[i][j] is None
        if SHOW_TRAIN_RES:
          self.show_train_res(img, gt_bboxes, img_metas, outs, score_threshold=(0.3,1))
          pass
        return losses

    def show_train_res(self, img, gt_bboxes, img_metas, outs, score_threshold):
          from configs.common import DIM_PARSE
          img = img[0].permute(1,2,0).cpu().data.numpy()
          _gt_bboxes = [g.cpu().data.numpy() for g in gt_bboxes][0:1]
          rescale = False
          bbox_inputs = outs + (img_metas, self.test_cfg, rescale)
          bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
          det_bboxes, det_labels = bbox_list[0]
          _det_bboxes0 = det_bboxes.cpu().data.numpy()
          assert not 'background' in img_metas[0]['classes']
          dim_parse = DIM_PARSE( len(img_metas[0]['classes'])+1 )
          _det_bboxes1 = dim_parse.clean_bboxes_out(_det_bboxes0,'final', 'line_ave' )
          _det_bboxes = [_det_bboxes1]
          ngt = len(_gt_bboxes[0])
          ndt = len(_det_bboxes[0])
          print(f'gt num={ngt}, det num={ndt}')

          mask0 = _det_bboxes[0][:,5] >= score_threshold[0]
          mask1 = _det_bboxes[0][:,5] <= score_threshold[1]
          mask =  mask0 * mask1

          #debug_utils._show_lines_ls_points_ls((512,512), _det_bboxes)
          #debug_utils._show_lines_ls_points_ls((512,512), _gt_bboxes)
          debug_utils._show_lines_ls_points_ls((512,512), [_gt_bboxes[0], _det_bboxes[0][mask]], line_colors=['red','green'])
          _show_objs_ls_points_ls(img, [_gt_bboxes[0]], 'RoLine2D_UpRight_xyxy_sin2a')
          import pdb; pdb.set_trace()  # XXX BREAKPOINT
          pass

    def simple_test(self, img, img_meta, rescale=False, gt_bboxes=None, gt_labels=None):
        from configs.common import DIM_PARSE
        if DEBUG_CFG.DISABLE_RESCALE:
          rescale = False
        #_show_objs_ls_points_ls(img[0].permute(1,2,0).cpu().data.numpy(), [gt_bboxes[0][0].cpu().data.numpy()], 'RoLine2D_UpRight_xyxy_sin2a')
        x = self.extract_feat(img, gt_bboxes)
        self.update_dynamic_shape(x, img_meta)
        #update_img_shape_for_pcl(x, img_meta[0], self.point_strides)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]

        results = dict( det_bboxes=bbox_results[0], gt_bboxes=gt_bboxes, gt_labels=gt_labels, img = img)
        if 0:
          dim_parse = DIM_PARSE( len(img_meta[0]['classes'])+1 )
          det_bboxes = dim_parse.clean_bboxes_out( bbox_results[0][0],'final', 'line_ave' )[:,:5]
          _show_objs_ls_points_ls(img[0].permute(1,2,0).cpu().data.numpy(), [gt_bboxes[0][0].cpu().data.numpy(), det_bboxes], 'RoLine2D_UpRight_xyxy_sin2a')
          _show_objs_ls_points_ls(img[0].permute(1,2,0).cpu().data.numpy(), [gt_bboxes[0][0].cpu().data.numpy(), ], 'RoLine2D_UpRight_xyxy_sin2a')
        return results

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
