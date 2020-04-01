import torch.nn as nn
import torch
import numpy as np

from mmdet.core import bbox2result
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from utils_dataset.gen_sparse_input import update_img_shape_for_pcl

from tools import debug_utils
import time

RECORD_T = 0
SHOW_TRAIN_RES = 0

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

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        if RECORD_T:
          t0 = time.time()
        x = self.backbone(img)
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
        is_pcl =  'input_type' in img_metas[0] and img_metas[0]['input_type'] == 'pcl'
        if not is_pcl:
          return
        feat_sizes = [ np.array([*xi.shape[2:]]) for xi in x]
        for img_meta in img_metas:
          img_meta['stem_stride'] = self.stem_stride
          img_meta['feat_sizes'] = feat_sizes
          img_meta['dynamic_img_shape'] = feat_sizes[0] * self.stem_stride

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        '''
        img: [6, 4, 512, 512]
        img_metas: [{}]*6
        gt_bboxes: [ni*5]*6
        gt_labels: [ni]*6
        '''
        if RECORD_T:
          t0 = time.time()
        x = self.extract_feat(img)
        self.update_dynamic_shape(x, img_metas)
        #debug_utils._show_tensor_ls_shapes(x, 'single_stage forward_train - features')
        if RECORD_T:
          t1 = time.time()
        outs = self.bbox_head(x)
        if RECORD_T:
          t2 = time.time()
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
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
          _gt_bboxes = [g.cpu().data.numpy() for g in gt_bboxes][0:1]
          rescale = False
          bbox_inputs = outs + (img_metas, self.test_cfg, rescale)
          bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
          det_bboxes, det_labels = bbox_list[0]
          _det_bboxes0 = det_bboxes.cpu().data.numpy()
          from configs.common import clean_bboxes_out
          _det_bboxes1 = clean_bboxes_out(_det_bboxes0,'final', 'line_ave' )
          _det_bboxes = [_det_bboxes1]

          debug_utils._show_lines_ls_points_ls((512,512), _det_bboxes)
          debug_utils._show_lines_ls_points_ls((512,512), _gt_bboxes)
          debug_utils._show_lines_ls_points_ls((512,512), [_gt_bboxes[0], _det_bboxes[0]], line_colors=['red','green'])
        return losses

    def simple_test(self, img, img_meta, rescale=False, gt_bboxes=None, gt_labels=None):
        x = self.extract_feat(img)
        self.update_dynamic_shape(x, img_meta)
        update_img_shape_for_pcl(x, img_meta[0], self.point_strides)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]

        #debug_utils.show_shapes(img, 'single_stage simple_test img')
        #debug_utils.show_shapes(x, 'single_stage simple_test x')
        #debug_utils.show_shapes(outs, 'single_stage simple_test outs')
        #return bbox_results[0]
        results = dict( det_bboxes=bbox_results[0], gt_bboxes=gt_bboxes, gt_labels=gt_labels)
        return results

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
