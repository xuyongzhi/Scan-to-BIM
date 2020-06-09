from __future__ import division

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import normal_init
import mmcv

from mmdet.core import (PointGenerator, multi_apply, multiclass_nms,
                        point_target)
from mmdet.ops import DeformConv
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule, bias_init_with_prob, Scale

from obj_geo_utils.geometry_utils  import sin2theta, angle_from_vecs_to_vece, angle_with_x, four_corners_to_box, sort_four_corners, align_pred_gt_bboxes
#from obj_geo_utils.obj_utils import OBJ_REPS_PARSE
from obj_geo_utils.line_operations import decode_line_rep_th, gen_corners_from_lines_th

import torchvision as tcv
import cv2

from configs.common import DIM_PARSE, DEBUG_CFG
from tools.visual_utils import _show_objs_ls_points_ls_torch, _show_objs_ls_points_ls
import time

RECORD_TIME = 0

DEBUG = 0

@HEADS.register_module
class StrPointsHead(nn.Module):
    """StrPoint head.

    Args:
        num_classes: include background
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of channels of the feature map.
        point_feat_channels (int): Number of channels of points features.
        stacked_convs (int): How many conv layers are used.
        gradient_mul (float): The multiplier to gradients from
            points refinement and recognition.
        point_strides (Iterable): points strides.
        point_base_scale (int): bbox scale for assigning labels.
        loss_cls (dict): Config of classification loss.
        loss_bbox_init (dict): Config of initial points loss.
        loss_bbox_refine (dict): Config of points loss in refinement.
        use_grid_points (bool): If we use bounding box representation, the
        reppoints is represented as grid points on the bounding box.
        center_init (bool): Whether to use center point assignment.
        transform_method (str): The methods to transform StrPoints to bbox.
    """  # noqa: W605

    def __init__(self,
                 obj_rep,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 point_feat_channels=256,
                 stacked_convs=3,
                 num_points=9,
                 gradient_mul=0.1,
                 point_strides=[8, 16, 32, 64, 128],
                 point_base_scale=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0,),
                 loss_bbox_init=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.5),
                 loss_bbox_refine=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 use_grid_points=False,
                 center_init=True,
                 transform_method='moment',
                 moment_mul=0.01,
                 cls_types=['refine', 'final'],
                 dcn_zero_base=False,
                 corner_hm = True,
                 corner_hm_only = True,
                 loss_centerness=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 loss_cor_ofs=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 move_points_to_center = False,
                 relation_cfg=dict(
                    enable = 0,
                    stage = ['refine', 'final'][0],
                    score_threshold = 0.2,
                    max_relation_num = 120, ),
                 loss_relation=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0,),
                 num_ps_long_axis = 9,
                 adjust_5pts_by_4 = False,
                 cls_groups = None,
                 ):
        super(StrPointsHead, self).__init__()
        if cls_groups is None:
          self.cls_groups = [ list(range(1, num_classes)) ]
        else:
          self.cls_groups = cls_groups  # 0 is background
        assert 0 not in np.array(self.cls_groups)
        assert sum([len(g) for g in self.cls_groups]) == num_classes-1
        self.num_per_group = [len(g) for g in self.cls_groups]

        self.wall_label = 1
        self.line_constrain_loss = False
        self.obj_rep = obj_rep
        self.adjust_5pts_by_4 = False
        if obj_rep == 'XYXYSin2WZ0Z1':
            if transform_method == '4corners_to_rect':
              self.box_extra_dims = 2
            else:
              assert transform_method == 'moment_XYXYSin2WZ0Z1'
              self.box_extra_dims = 3
        elif obj_rep == 'XYXYSin2':
            self.box_extra_dims = 0
        elif obj_rep == 'XYXYSin2W':
            self.box_extra_dims = 0
        elif obj_rep == 'XYLgWsAbsSin2Z0Z1':
            if transform_method == 'minAreaRect':
              self.box_extra_dims = 3
            if transform_method == 'XYLgWsAbsSin2Z0Z1':
              self.box_extra_dims = 8
            self.line_constrain_loss = False
        elif obj_rep == 'XYDAsinAsinSin2Z0Z1':
            assert transform_method == '4corners_to_rect'
            self.box_extra_dims = 2
        elif obj_rep == 'Rect4CornersZ0Z1':
            assert transform_method == 'sort_4corners'
            self.box_extra_dims = 2
            self.adjust_5pts_by_4 = adjust_5pts_by_4

        self.dim_parse = DIM_PARSE(self.obj_rep, num_classes)
        self.obj_dim = self.dim_parse.OBJ_DIM

        self.move_points_to_center = move_points_to_center
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.point_feat_channels = point_feat_channels
        self.stacked_convs = stacked_convs
        self.num_points = num_points
        assert num_ps_long_axis <= num_points
        self.num_ps_long_axis = num_ps_long_axis
        self.gradient_mul = gradient_mul
        self.point_base_scale = point_base_scale
        self.point_strides = point_strides
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in ['FocalLoss']
        self.loss_cls = build_loss(loss_cls)
        self.cls_types = cls_types
        for c in self.cls_types:
          assert c in [ 'refine', 'final' ]
        self.cls_types = cls_types
        self.loss_bbox_init = build_loss(loss_bbox_init)
        self.loss_bbox_refine = build_loss(loss_bbox_refine)
        self.loss_line_constrain_init = build_loss(loss_bbox_init)
        self.loss_line_constrain_refine = build_loss(loss_bbox_refine)
        self.use_grid_points = use_grid_points
        self.center_init = center_init
        self.transform_method = transform_method
        if self.transform_method == 'moment' or \
            self.transform_method in ['moment_XYXYSin2', 'moment_XYXYSin2WZ0Z1']:
            self.moment_transfer = nn.Parameter(
                data=torch.zeros(2), requires_grad=True)
            self.moment_mul = moment_mul
        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes - 1
            assert sum(self.num_per_group) == self.cls_out_channels
        else:
            self.cls_out_channels = self.num_classes
            assert sum(self.num_per_group) == self.cls_out_channels-1
            self.num_per_group = [c+1 for c in self.num_per_group]
            self.cls_groups = [[0]+c for c in self.cls_groups ]
        self.cls_out_channels_g = self.cls_out_channels
        self.point_generators = [PointGenerator() for _ in self.point_strides]
        # we use deformable conv to extract points features
        self.dcn_kernel = int(np.sqrt(num_points))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        self.dcn_zero_base = dcn_zero_base
        assert self.dcn_kernel * self.dcn_kernel == num_points, \
            "The points number should be a square number."
        assert self.dcn_kernel % 2 == 1, \
            "The points number should be an odd square number."
        dcn_base = np.arange(-self.dcn_pad,
                             self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape(
            (-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)
        self.dcn_base_offset = self.dcn_base_offset.repeat(1,len(self.num_per_group),1,1)

        self.relation_cfg = relation_cfg
        if self.relation_cfg['enable']:
            self.loss_relation = build_loss(loss_relation)

        self.corner_hm = corner_hm
        self.corner_hm_only = corner_hm_only

        self.loss_centerness = build_loss(loss_centerness)
        self.loss_cor_ofs = build_loss(loss_cor_ofs)
        self._init_layers()

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        pts_out_dim = 4 if self.use_grid_points else 2 * self.num_points
        pts_out_dim *= len(self.num_per_group)
        self.reppoints_cls_conv = DeformConv(self.feat_channels,
                                             self.point_feat_channels,
                                             self.dcn_kernel, 1, self.dcn_pad)
        self.reppoints_cls_out = nn.Conv2d(self.point_feat_channels,
                                           self.cls_out_channels, 1, 1, 0)
        self.reppoints_pts_init_conv = nn.Conv2d(self.feat_channels,
                                                 self.point_feat_channels, 3,
                                                 1, 1)
        self.reppoints_pts_init_out = nn.Conv2d(self.point_feat_channels,
                                                pts_out_dim, 1, 1, 0)
        self.reppoints_pts_refine_conv = DeformConv(self.feat_channels,
                                                    self.point_feat_channels,
                                                    self.dcn_kernel, 1,
                                                    self.dcn_pad)
        self.reppoints_pts_refine_out = nn.Conv2d(self.point_feat_channels,
                                                  pts_out_dim, 1, 1, 0)

        if self.box_extra_dims >0:
          self.box_extra_init_out = nn.Conv2d(self.point_feat_channels,
                                                  self.box_extra_dims, 1, 1, 0)
          self.box_extra_refine_out = nn.Conv2d(self.point_feat_channels,
                                                  self.box_extra_dims, 1, 1, 0)
        #-----------------------------------------------------------------------
        # corner
        self.cor_cls_convs = nn.ModuleList()
        self.cor_reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cor_cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.cor_reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
        self.cor_fcos_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.cor_fcos_reg = nn.Conv2d(self.feat_channels, 2, 3, padding=1)
        self.cor_fcos_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)

        #-----------------------------------------------------------------------
        # relationship
        if self.relation_cfg['enable']:
            self.edge_relation_convs = nn.ModuleList()
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                self.edge_relation_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        bias=self.norm_cfg is None))

            self.edge_relation_cls_convs = nn.ModuleList()
            for i in range(3):
                chn = self.feat_channels * 2 if i == 0 else self.feat_channels
                self.edge_relation_cls_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        1,
                        stride=1,
                        padding=0,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        bias=self.norm_cfg is None))

            self.edge_relation_cls = nn.Conv2d(
                self.feat_channels, 1, 1, padding=0)

        # learnable scale
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.point_strides])


    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.reppoints_cls_conv, std=0.01)
        normal_init(self.reppoints_cls_out, std=0.01, bias=bias_cls)
        normal_init(self.reppoints_pts_init_conv, std=0.01)
        normal_init(self.reppoints_pts_init_out, std=0.01)
        normal_init(self.reppoints_pts_refine_conv, std=0.01)
        normal_init(self.reppoints_pts_refine_out, std=0.01)

        if self.corner_hm:
          for m in self.cor_cls_convs:
              normal_init(m.conv, std=0.01)
          for m in self.cor_reg_convs:
              normal_init(m.conv, std=0.01)
          bias_cls = bias_init_with_prob(0.01)
          normal_init(self.cor_fcos_cls, std=0.01, bias=bias_cls)
          normal_init(self.cor_fcos_reg, std=0.01)
          normal_init(self.cor_fcos_centerness, std=0.01)

        if self.relation_cfg['enable']:
          for m in self.edge_relation_convs:
            normal_init(m.conv, std=0.01)
          for m in self.edge_relation_cls_convs:
            normal_init(m.conv, std=0.01)
          bias_cls = bias_init_with_prob(0.01)
          normal_init(self.edge_relation_cls, std=0.01, bias=bias_cls)


    def points2bbox(self, pts, y_first=True, out_line_constrain=False, box_extra=None, stage=None, bbox_weights=None, bbox_gt=None):
        """
        Converting the points set into bounding box.
        :param pts: the input points sets (fields), each points
            set (fields) is represented as 2n scalar.
        :param y_first: if y_fisrt=True, the point set is represented as
            [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
            represented as [x1, y1, x2, y2 ... xn, yn].
        :return: each points set is converting to a bbox [x1, y1, x2, y2].

        bbox_weights: used for reducing processing time
        """
        assert pts.shape[1] == self.num_points * 2
        pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
        pts_y = pts_reshape[:, :, 0, ...] if y_first else pts_reshape[:, :, 1,
                                                                      ...]
        pts_x = pts_reshape[:, :, 1, ...] if y_first else pts_reshape[:, :, 0,
                                                                      ...]
        if self.transform_method == 'minmax':
            assert box_extra is None
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_up = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom],
                             dim=1)
        elif self.transform_method == 'XYLgWsAbsSin2Z0Z1':
            assert self.box_extra_dims == 8
            assert box_extra.shape[1] == 8
            bbox = box_extra
            if DEBUG_CFG.SET_WIDTH_0:
              bbox[:,3] *= 0
            pass

        elif self.transform_method == 'minAreaRect':
            assert self.box_extra_dims == 3
            assert box_extra.shape[1] == 3
            pts = torch.cat([pts_x[..., None], pts_y[...,None]], dim=-1)
            bs, npts, h,w,_ = pts.shape
            pts_flat = pts.permute(0, 2,3,1,4).reshape( bs*h*w, npts, 2 )
            t0 = time.time()
            boxes = []
            for i in range(pts_flat.shape[0]):
              pts_i = pts_flat[i].cpu().data.numpy()
              bi = cv2.minAreaRect(pts_i)
              boxes.append(bi)
            t1 = time.time()
            import pdb; pdb.set_trace()  # XXX BREAKPOINT
            if DEBUG_CFG.SET_WIDTH_0:
              bbox[:,3] *= 0
            pass

        elif self.transform_method == 'partial_minmax':
            assert box_extra is None
            pts_y = pts_y[:, :4, ...]
            pts_x = pts_x[:, :4, ...]
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_up = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom],
                             dim=1)
        elif self.transform_method == 'moment':
            assert box_extra is None
            pts_y_mean = pts_y.mean(dim=1, keepdim=True)
            pts_x_mean = pts_x.mean(dim=1, keepdim=True)
            pts_y_std = torch.std(pts_y - pts_y_mean, dim=1, keepdim=True)
            pts_x_std = torch.std(pts_x - pts_x_mean, dim=1, keepdim=True)
            moment_transfer = (self.moment_transfer * self.moment_mul) + (
                self.moment_transfer.detach() * (1 - self.moment_mul))
            moment_width_transfer = moment_transfer[0]
            moment_height_transfer = moment_transfer[1]
            half_width = pts_x_std * torch.exp(moment_width_transfer)
            half_height = pts_y_std * torch.exp(moment_height_transfer)
            bbox = torch.cat([
                pts_x_mean - half_width, pts_y_mean - half_height,
                pts_x_mean + half_width, pts_y_mean + half_height
            ],
                             dim=1)
        elif self.transform_method == 'moment_XYXYSin2':
            bbox = self.tran_fun_moment_XYXYSin2(pts_x, pts_y, box_extra, out_line_constrain)
        elif self.transform_method == 'moment_XYXYSin2WZ0Z1':
            bbox = self.tran_fun_moment_XYXYSin2WZ0Z1(pts_x, pts_y, box_extra, out_line_constrain)

        elif self.transform_method == '4corners_to_rect':
            assert box_extra.shape[1] == 2
            # pt_0: center
            # pt_1,2,3,4: the four corners
            # pt_5,6,7,8: half corners
            pts_xy = torch.cat([ pts_x[...,None], pts_y[...,None] ], dim=-1)

            bbox, rect_loss = four_corners_to_box( rect_corners = pts_xy[:,1:5], rect_center = pts_xy[:,0:1], stage=stage, bbox_weights=bbox_weights, bbox_gt = bbox_gt )
            if out_line_constrain:
              bbox = torch.cat([bbox, rect_loss], dim=1)

            pass

        elif self.transform_method == 'sort_4corners':
            pts_xy = torch.cat([ pts_x[...,None], pts_y[...,None] ], dim=-1)
            if bbox_gt is not None:
              gt_corners = bbox_gt[:,:8].reshape(-1,4,2)
            else:
              gt_corners = None
            #t0 = time.time()
            bbox, rect_loss = sort_four_corners( pred_corners = pts_xy[:,0:4])
            #bbox, rect_loss = sort_four_corners( pred_corners = pts_xy[:,0:4], pred_center = pts_xy[:,4:5])
            #t = time.time() - t0
            #print(f'\n\t sort t:{t}')
            z0z1 = box_extra * 0
            bbox = torch.cat([bbox, z0z1], dim=1)
            if out_line_constrain:
              bbox = torch.cat([bbox, rect_loss], dim=1)
            pass


        elif self.transform_method == 'moment_LWAsS2ZZ':
            pts_y_mean = pts_y.mean(dim=1, keepdim=True)
            pts_x_mean = pts_x.mean(dim=1, keepdim=True)
            pts_y_std = torch.std(pts_y - pts_y_mean, dim=1, keepdim=True)
            pts_x_std = torch.std(pts_x - pts_x_mean, dim=1, keepdim=True)
            moment_transfer = (self.moment_transfer * self.moment_mul) + (
                self.moment_transfer.detach() * (1 - self.moment_mul))
            moment_width_transfer = moment_transfer[0]
            moment_height_transfer = moment_transfer[1]
            half_width = pts_x_std * torch.exp(moment_width_transfer)
            half_height = pts_y_std * torch.exp(moment_height_transfer)

            length_greater = torch.max( half_width, half_height ) * 2
            width_smaller = torch.min( half_width, half_height ) * 2

            vec_pts_y = pts_y - pts_y_mean
            vec_pts_x = pts_x - pts_x_mean
            vec_pts = torch.cat([vec_pts_x.view(-1,1), vec_pts_y.view(-1,1)], dim=1)
            vec_start = torch.zeros_like(vec_pts)
            vec_start[:,1] = -1

            sin_2thetas, sin_thetas = sin2theta(vec_start, vec_pts)
            abs_sins = torch.abs(sin_thetas).view(pts_x.shape)
            sin_2thetas = sin_2thetas.view(pts_x.shape)

            npla = self.num_ps_long_axis
            sin_2theta = sin_2thetas[:,:npla].mean(dim=1, keepdim=True)
            abs_sin = abs_sins[:,:npla].mean(dim=1, keepdim=True)

            isaline_0 = sin_2thetas[:,:npla].std(dim=1, keepdim=True)
            isaline_1 = abs_sins[:,:npla].std(dim=1, keepdim=True)
            isaline = (isaline_0 + isaline_1) / 2

            z0z1 = torch.cat([torch.zeros_like(pts_x_mean)]*2, axis=1)

            bbox = torch.cat([
                pts_x_mean, pts_y_mean,
                length_greater, width_smaller,
                abs_sin, sin_2theta, z0z1
              ], dim=1)

            if out_line_constrain:
              bbox = torch.cat([bbox, isaline], dim=1)
            pass
        else:
          assert False
        return bbox

    def tran_fun_moment_XYXYSin2(self, pts_x, pts_y, box_extra, out_line_constrain):
            assert box_extra is None
            pts_y_mean = pts_y.mean(dim=1, keepdim=True)
            pts_x_mean = pts_x.mean(dim=1, keepdim=True)
            pts_y_std = torch.std(pts_y - pts_y_mean, dim=1, keepdim=True)
            pts_x_std = torch.std(pts_x - pts_x_mean, dim=1, keepdim=True)
            moment_transfer = (self.moment_transfer * self.moment_mul) + (
                self.moment_transfer.detach() * (1 - self.moment_mul))
            moment_width_transfer = moment_transfer[0]
            moment_height_transfer = moment_transfer[1]
            half_width = pts_x_std * torch.exp(moment_width_transfer)
            half_height = pts_y_std * torch.exp(moment_height_transfer)


            vec_pts_y = pts_y - pts_y_mean
            vec_pts_x = pts_x - pts_x_mean
            vec_pts = torch.cat([vec_pts_x.view(-1,1), vec_pts_y.view(-1,1)], dim=1)
            vec_start = torch.zeros_like(vec_pts)
            vec_start[:,1] = -1

            sin_2thetas, sin_thetas = sin2theta(vec_start, vec_pts)
            abs_sins = torch.abs(sin_thetas).view(pts_x.shape)
            sin_2thetas = sin_2thetas.view(pts_x.shape)

            npla = self.num_ps_long_axis
            sin_2theta = sin_2thetas[:,:npla].mean(dim=1, keepdim=True)
            abs_sin = abs_sins[:,:npla].mean(dim=1, keepdim=True)

            isaline_0 = sin_2thetas[:,:npla].std(dim=1, keepdim=True)
            isaline_1 = abs_sins[:,:npla].std(dim=1, keepdim=True)
            isaline = (isaline_0 + isaline_1) / 2

            bbox = torch.cat([
                pts_x_mean - half_width, pts_y_mean - half_height,
                pts_x_mean + half_width, pts_y_mean + half_height,
                sin_2theta
            ],
                             dim=1)
            if out_line_constrain:
              bbox = torch.cat([bbox, isaline], dim=1)
            return bbox
            pass

    def tran_fun_moment_XYXYSin2WZ0Z1(self, pts_x, pts_y, box_extra, out_line_constrain):
            assert self.box_extra_dims == 3 and box_extra.shape[1] == self.box_extra_dims
            if not box_extra.shape[2:] == pts_x.shape[2:]:
              import pdb; pdb.set_trace()  # XXX BREAKPOINT
              pass

            pts_y_mean = pts_y.mean(dim=1, keepdim=True)
            pts_x_mean = pts_x.mean(dim=1, keepdim=True)
            pts_y_std = torch.std(pts_y - pts_y_mean, dim=1, keepdim=True)
            pts_x_std = torch.std(pts_x - pts_x_mean, dim=1, keepdim=True)
            moment_transfer = (self.moment_transfer * self.moment_mul) + (
                self.moment_transfer.detach() * (1 - self.moment_mul))
            moment_width_transfer = moment_transfer[0]
            moment_height_transfer = moment_transfer[1]
            half_width = pts_x_std * torch.exp(moment_width_transfer)
            half_height = pts_y_std * torch.exp(moment_height_transfer)


            vec_pts_y = pts_y - pts_y_mean
            vec_pts_x = pts_x - pts_x_mean
            vec_pts = torch.cat([vec_pts_x.view(-1,1), vec_pts_y.view(-1,1)], dim=1)
            vec_start = torch.zeros_like(vec_pts)
            vec_start[:,1] = -1

            sin_2thetas, sin_thetas = sin2theta(vec_start, vec_pts)
            abs_sins = torch.abs(sin_thetas).view(pts_x.shape)
            sin_2thetas = sin_2thetas.view(pts_x.shape)

            npla = self.num_ps_long_axis
            sin_2theta = sin_2thetas[:,:npla].mean(dim=1, keepdim=True)
            abs_sin = abs_sins[:,:npla].mean(dim=1, keepdim=True)

            isaline_0 = sin_2thetas[:,:npla].std(dim=1, keepdim=True)
            isaline_1 = abs_sins[:,:npla].std(dim=1, keepdim=True)
            isaline = (isaline_0 + isaline_1) / 2

            width_z0_z1 = torch.zeros_like(pts_y)[:,:3]

            bbox = torch.cat([
                pts_x_mean - half_width, pts_y_mean - half_height,
                pts_x_mean + half_width, pts_y_mean + half_height,
                sin_2theta, width_z0_z1
            ],  dim=1)
            if out_line_constrain:
              bbox = torch.cat([bbox, isaline], dim=1)
            return bbox
            pass

    def Unused_tf_moment_XYXYSin2WZ0Z1(self, pts_x, pts_y, box_extra, out_line_constrain):
            assert self.box_extra_dims == 3 and box_extra.shape[1] == self.box_extra_dims
            if not box_extra.shape[2:] == pts_x.shape[2:]:
              import pdb; pdb.set_trace()  # XXX BREAKPOINT
              pass

            pts_y_mean = pts_y.mean(dim=1, keepdim=True)
            pts_x_mean = pts_x.mean(dim=1, keepdim=True)
            pts_y_std = torch.std(pts_y - pts_y_mean, dim=1, keepdim=True)
            pts_x_std = torch.std(pts_x - pts_x_mean, dim=1, keepdim=True)
            moment_transfer = (self.moment_transfer * self.moment_mul) + (
                self.moment_transfer.detach() * (1 - self.moment_mul))
            moment_width_transfer = moment_transfer[0]
            moment_height_transfer = moment_transfer[1]
            half_width = pts_x_std * torch.exp(moment_width_transfer)
            half_height = pts_y_std * torch.exp(moment_height_transfer)


            vec_pts_y = pts_y - pts_y_mean
            vec_pts_x = pts_x - pts_x_mean
            vec_pts = torch.cat([vec_pts_x.view(-1,1), vec_pts_y.view(-1,1)], dim=1)
            vec_start = torch.zeros_like(vec_pts)
            vec_start[:,1] = -1

            sin_2thetas, sin_thetas = sin2theta(vec_start, vec_pts)
            abs_sins = torch.abs(sin_thetas).view(pts_x.shape)
            sin_2thetas = sin_2thetas.view(pts_x.shape)

            npla = self.num_ps_long_axis
            sin_2theta = sin_2thetas[:,:npla].mean(dim=1, keepdim=True)
            abs_sin = abs_sins[:,:npla].mean(dim=1, keepdim=True)

            isaline_0 = sin_2thetas[:,:npla].std(dim=1, keepdim=True)
            isaline_1 = abs_sins[:,:npla].std(dim=1, keepdim=True)
            isaline = (isaline_0 + isaline_1) / 2

            width_z0_z1 = torch.zeros_like(pts_y)[:,:3]
            width_z0_z1[:,0] = box_extra[:,0]

            if DEBUG_CFG.SET_WIDTH_0:
              width_z0_z1[:,0] *= 0
            if DEBUG_CFG.SET_Z_0:
              width_z0_z1[:,1:] *= 0

            bbox = torch.cat([
                pts_x_mean - half_width, pts_y_mean - half_height,
                pts_x_mean + half_width, pts_y_mean + half_height,
                sin_2theta, width_z0_z1
            ], dim=1)
            if out_line_constrain:
              bbox = torch.cat([bbox, isaline], dim=1)
            return bbox

    def gen_grid_from_reg(self, reg, previous_boxes):
        """
        Base on the previous bboxes and regression values, we compute the
            regressed bboxes and generate the grids on the bboxes.
        :param reg: the regression value to previous bboxes.
        :param previous_boxes: previous bboxes.
        :return: generate grids on the regressed bboxes.
        """
        b, _, h, w = reg.shape
        bxy = (previous_boxes[:, :2, ...] + previous_boxes[:, 2:, ...]) / 2.
        bwh = (previous_boxes[:, 2:, ...] -
               previous_boxes[:, :2, ...]).clamp(min=1e-6)
        grid_topleft = bxy + bwh * reg[:, :2, ...] - 0.5 * bwh * torch.exp(
            reg[:, 2:, ...])
        grid_wh = bwh * torch.exp(reg[:, 2:, ...])
        grid_left = grid_topleft[:, [0], ...]
        grid_top = grid_topleft[:, [1], ...]
        grid_width = grid_wh[:, [0], ...]
        grid_height = grid_wh[:, [1], ...]
        intervel = torch.linspace(0., 1., self.dcn_kernel).view(
            1, self.dcn_kernel, 1, 1).type_as(reg)
        grid_x = grid_left + grid_width * intervel
        grid_x = grid_x.unsqueeze(1).repeat(1, self.dcn_kernel, 1, 1, 1)
        grid_x = grid_x.view(b, -1, h, w)
        grid_y = grid_top + grid_height * intervel
        grid_y = grid_y.unsqueeze(2).repeat(1, 1, self.dcn_kernel, 1, 1)
        grid_y = grid_y.view(b, -1, h, w)
        grid_yx = torch.stack([grid_y, grid_x], dim=2)
        grid_yx = grid_yx.view(b, -1, h, w)
        regressed_bbox = torch.cat([
            grid_left, grid_top, grid_left + grid_width, grid_top + grid_height
        ], 1)
        return grid_yx, regressed_bbox

    def auto_adjust_pts_by_partial(self, pts):
      if self.obj_rep not in ['Rect4CornersZ0Z1']:
        return pts
      bs, pts_dim, h, w = pts.shape
      assert pts_dim == 18
      rect_corners = pts[:,:8,:,:].detach().view(bs, 4, 2, h, w)
      center = rect_corners.mean(dim=1, keepdim=True)
      half_corners = (rect_corners - center)/2 + center
      half_corners = half_corners.view(bs, 8, h, w)
      new_pts = torch.cat( [pts[:,:8], center.squeeze(1), half_corners], dim = 1)
      return new_pts

    def forward_single(self, x, stride, scale_learn):
        dcn_base_offset = self.dcn_base_offset.type_as(x)
        # If we use center_init, the initial reppoints is from center points.
        # If we use bounding bbox representation, the initial reppoints is
        #   from regular grid placed on a pre-defined bbox.
        if self.use_grid_points or not self.center_init:
            scale = self.point_base_scale / 2
            points_init = dcn_base_offset / dcn_base_offset.max() * scale
            bbox_init = x.new_tensor([-scale, -scale, scale,
                                      scale]).view(1, 4, 1, 1)
        else:
            points_init = 0
        cls_feat = x
        pts_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            pts_feat = reg_conv(pts_feat)
        # initialize reppoints
        pts_feat_init = self.relu(self.reppoints_pts_init_conv(pts_feat))
        pts_out_init = self.reppoints_pts_init_out( pts_feat_init )
        if self.adjust_5pts_by_4:
          pts_out_init = self.auto_adjust_pts_by_partial(pts_out_init)
        if self.box_extra_dims >0:
          box_extra_init = self.box_extra_init_out(pts_feat_init)

        if self.use_grid_points:
            pts_out_init, bbox_out_init = self.gen_grid_from_reg(
                pts_out_init, bbox_init.detach())
        else:
            pts_out_init = pts_out_init + points_init
        # refine and classify reppoints
        pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach(
        ) + self.gradient_mul * pts_out_init
        if self.dcn_zero_base:
          dcn_offset = pts_out_init_grad_mul
        else:
          dcn_offset = pts_out_init_grad_mul - dcn_base_offset
        cls_out = {}
        npts = self.num_points * 2
        if 'refine' in self.cls_types:
          cls_ls = []
          pts_out_refine = []
          s = 0
          for i, num_cls in enumerate(self.num_per_group):
            cls_i = self.reppoints_cls_out(
               self.relu(self.reppoints_cls_conv(cls_feat, dcn_offset[:,i*npts:(i+1)*npts])))
            cls_ls.append(cls_i[:,s:s+num_cls])
            s += num_cls
            pts_feat_refine = self.relu(self.reppoints_pts_refine_conv(pts_feat, dcn_offset[:,i*npts:(i+1)*npts]))
            pts_out_refine.append( self.reppoints_pts_refine_out( pts_feat_refine )[:,i*npts:(i+1)*npts] )
            if self.adjust_5pts_by_4:
              pts_out_refine[-1] = self.auto_adjust_pts_by_partial(pts_out_refine[-1])
          cls_out['refine'] = torch.cat(cls_ls, 1)
          pts_out_refine = torch.cat(pts_out_refine, 1)

        if self.box_extra_dims >0:
          #print('Multi group version not implemented')
          box_extra_refine = self.box_extra_refine_out(pts_feat_refine)

        if self.use_grid_points:
            pts_out_refine, bbox_out_refine = self.gen_grid_from_reg(
                pts_out_refine, bbox_out_init.detach())
        else:
            pts_out_refine = pts_out_refine + pts_out_init.detach()


        if 'final' in self.cls_types:
          pts_out_refine_grad_mul = (1 - self.gradient_mul) * pts_out_refine.detach(
          ) + self.gradient_mul * pts_out_refine
          if self.dcn_zero_base:
            dcn_offset_refine = pts_out_refine_grad_mul
          else:
            dcn_offset_refine = pts_out_refine_grad_mul - dcn_base_offset
          s = 0
          cls_ls = []
          for i, num_cls in enumerate(self.num_per_group):
            cls_i = self.reppoints_cls_out(
              self.relu(self.reppoints_cls_conv(cls_feat, dcn_offset_refine[:,i*npts:(i+1)*npts])))
            cls_ls.append(cls_i[:,s:s+num_cls])
            s += num_cls
          cls_out['final'] = torch.cat(cls_ls, 1)

        if self.relation_cfg['enable']:
          rel_feat = self.forward_single_relation_feats(x)
        else:
          rel_feat = None

        if self.corner_hm and stride == self.point_strides[0]:
          corner_outs = self.forward_single_corner(x, scale_learn)
        else:
          corner_outs = None

        if not self.box_extra_dims >0:
          box_extra_init = None
          box_extra_refine = None

        #debug_utils.show_shapes(x, 'StrPointsHead input')
        #debug_utils.show_shapes(cls_feat, 'StrPointsHead cls_feat')
        #debug_utils.show_shapes(pts_feat, 'StrPointsHead pts_feat')
        #debug_utils.show_shapes(pts_out_init, 'StrPointsHead pts_out_init')
        #debug_utils.show_shapes(cls_out, 'StrPointsHead cls_out')
        #debug_utils.show_shapes(pts_out_refine, 'StrPointsHead pts_out_refine')
        return cls_out, pts_out_init, pts_out_refine, corner_outs, rel_feat, box_extra_init, box_extra_refine

    def forward_single_relation_feats(self, x):
        # use dcn later!
        rel_feat = x
        for rel_layer in self.edge_relation_convs:
          rel_feat = rel_layer(x)
        return rel_feat

    def forward_relation_cls(self, rel_feat_outs, pos_inds_list, gt_inds_per_pos_list = None, max_relation_num=None):
        '''
        Predict the relation of positive detections.
        '''
        num_levels = len(rel_feat_outs)
        batch_size = rel_feat_outs[0].shape[0]
        rel_feat_channel = rel_feat_outs[0].shape[1]
        assert len(pos_inds_list) == batch_size

        rel_feats_flat = [rel_feat_outs[l].view(batch_size, rel_feat_channel, -1) for l in range(num_levels)]
        rel_feats_flat = torch.cat(rel_feats_flat, dim=2)

        rel_scores_all = []
        rel_inds_all = []
        gt_inds_per_rel_all = []
        for i in range(batch_size):
          npos = pos_inds_list[i].numel()
          #print(f'num pos wall: {npos}, max set: {max_relation_num}')
          if max_relation_num is not None and npos > max_relation_num:
              choice = torch.randperm(npos)[:max_relation_num].sort()[0]
              valid_inds_i = pos_inds_list[i][choice]
              if gt_inds_per_pos_list is not None:
                gt_inds_per_rel = gt_inds_per_pos_list[i][choice]
          else:
              valid_inds_i = pos_inds_list[i]
              if gt_inds_per_pos_list is not None:
                gt_inds_per_rel = gt_inds_per_pos_list[i]
          rel_inds_all.append(valid_inds_i)
          if gt_inds_per_pos_list is not None:
            gt_inds_per_rel_all.append(gt_inds_per_rel)

          rel_feats = rel_feats_flat[i][None, :, valid_inds_i]
          ni = rel_feats.shape[2]
          x = rel_feats[:,:,None,:].repeat(1,1,ni,1)
          y = x.permute(0,1,3,2)
          rel_feats_matrix = torch.cat([x,y], dim=1)
          for rel_conv in self.edge_relation_cls_convs:
            rel_feats_matrix = rel_conv(rel_feats_matrix)
          rel_scores = self.edge_relation_cls(rel_feats_matrix)
          rel_scores_all.append(rel_scores)
        pass
        return rel_scores_all, rel_inds_all, gt_inds_per_rel_all

    def forward_single_corner(self, x, scale):
        cls_feat = x
        reg_feat = x

        for cls_layer in self.cor_cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.cor_fcos_cls(cls_feat)
        centerness = self.cor_fcos_centerness(cls_feat)

        for reg_layer in self.cor_reg_convs:
            reg_feat = reg_layer(reg_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        cor_ofs_pred = scale(self.cor_fcos_reg(reg_feat)).float()
        corner_outs = dict(
                    cor_scores = cls_score,
                    cor_centerness = centerness,
                    cor_ofs = cor_ofs_pred )
        return corner_outs

    def forward(self, feats):
        '''
        feats: (feats_level0, feats_level1...)
        outs: cls_out, pts_out_init, pts_out_refine, corner_outs, rel_feats
        len(outs[i]) = num_level, i=0:5
        len(outs[i][j]) = batch_size, j=0:num_level
        '''
        # forward per level
        outs =  multi_apply(self.forward_single, feats, self.point_strides, self.scales)
        return outs

    def get_points(self, featmap_sizes, img_metas):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: points of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # points center for one time
        multi_level_points = []
        for i in range(num_levels):
            points = self.point_generators[i].grid_points(
                featmap_sizes[i], self.point_strides[i])
            multi_level_points.append(points)
        points_list = [[point.clone() for point in multi_level_points]
                       for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level grids
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                point_stride = self.point_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                is_pcl = 'input_style' in img_meta and img_meta['input_style']=='pcl'
                if is_pcl:
                  valid_feat_h, valid_feat_w = feat_h, feat_w
                else:
                  h, w, _ = img_meta['pad_shape']
                  valid_feat_h = min(int(np.ceil(h / point_stride)), feat_h)
                  valid_feat_w = min(int(np.ceil(w / point_stride)), feat_w)
                flags = self.point_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w))
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)

        if self.move_points_to_center:
          self.do_move_points_to_center(points_list)
        debug_points = 0
        if debug_points:
          for i in  range(len(featmap_sizes)):
            for j in range(num_imgs):
              print(f'featmap_size: {featmap_sizes[i]}')
              print(points_list[j][i])
        return points_list, valid_flag_list

    def do_move_points_to_center(self, points_list):
      for i in range(len(points_list)):
        for j in range(len(points_list[i])):
          points_list[i][j][:,:2] += points_list[i][j][:,2:] * 0.5

    def centers_to_bboxes(self, point_list):
        """Get bboxes according to center points. Only used in MaxIOUAssigner.
        """
        bbox_list = []
        for i_img, point in enumerate(point_list):
            bbox = []
            for i_lvl in range(len(self.point_strides)):
                scale = self.point_base_scale * self.point_strides[i_lvl] * 0.5
                bbox_shift = torch.Tensor([-scale, -scale, scale,
                                           scale]).view(1, 4).type_as(point[0])
                bbox_center = torch.cat(
                    [point[i_lvl][:, :2], point[i_lvl][:, :2]], dim=1)
                bbox.append(bbox_center + bbox_shift)
            bbox_list.append(bbox)
        return bbox_list

    def offset_to_pts(self, center_list, pred_list, is_corner=False):
        """Change from point offset to point coordinate.
        Both center_list and pred_list should be  x_first
        """
        if is_corner:
          npts = 2
        else:
          npts = self.num_points
        pts_list = []
        for i_lvl in range(len(self.point_strides)):
            pts_lvl = []
            for i_img in range(len(center_list)):
                pts_center = center_list[i_img][i_lvl][:, :2].repeat(
                    1, npts)
                pts_shift = pred_list[i_lvl][i_img]
                yx_pts_shift = pts_shift.permute(1, 2, 0).view(-1, npts*2)
                y_pts_shift = yx_pts_shift[..., 0::2]
                x_pts_shift = yx_pts_shift[..., 1::2]
                xy_pts_shift = torch.stack([x_pts_shift, y_pts_shift], -1)
                xy_pts_shift = xy_pts_shift.view(*yx_pts_shift.shape[:-1], -1)
                pts = xy_pts_shift * self.point_strides[i_lvl] + pts_center
                pts_lvl.append(pts)
            pts_lvl = torch.stack(pts_lvl, 0)
            pts_list.append(pts_lvl)
        return pts_list


    def loss_single(self, cls_score, pts_pred_init, pts_pred_refine,
                    box_extra_init,  box_extra_refine,
                    labels,
                    label_weights, bbox_gt_init, bbox_weights_init,
                    bbox_gt_refine, bbox_weights_refine, stride,
                    num_total_samples_init, num_total_samples_refine,
                    num_total_samples_cls):
        '''
        n = batch_size * num_feature
        bbox_gt_init: [n,5/8]
        '''
        assert bbox_gt_init.ndim == 3
        assert pts_pred_init.ndim == 3
        batch_size, num_feats, obj_dim = bbox_gt_init.shape
        assert obj_dim == self.obj_dim
        assert pts_pred_init.shape == (batch_size, num_feats, self.num_points * 2)

        # classification loss
        loss_cls = {}
        for cls_type in cls_score:
          labels_i = labels[cls_type].reshape(-1)
          label_weights_i = label_weights[cls_type].reshape(-1)
          cls_score_i = cls_score[cls_type].permute(0, 2, 3,
                                        1).reshape(-1, self.cls_out_channels_g)
          if cls_score_i.shape[0] != labels_i.shape[0]:
            import pdb; pdb.set_trace()  # XXX BREAKPOINT
            pass
          loss_cls[cls_type] = self.loss_cls(
              cls_score_i,
              labels_i,
              label_weights_i,
              avg_factor=num_total_samples_cls[cls_type])

        # points loss
        bbox_gt_init = bbox_gt_init.reshape(-1, self.obj_dim)
        bbox_weights_init = bbox_weights_init.reshape(-1, self.obj_dim)
        if box_extra_init is not None:
          box_extra_init = box_extra_init.reshape(-1, self.box_extra_dims)
        pts_pred_init = pts_pred_init.reshape(-1, 2 * self.num_points)
        bbox_pred_init = self.points2bbox(
            pts_pred_init, y_first=False,
            out_line_constrain=self.line_constrain_loss,
            box_extra=box_extra_init,
            bbox_weights = bbox_weights_init,
            bbox_gt = bbox_gt_init,
            stage='loss_single_init')
        bbox_gt_refine = bbox_gt_refine.reshape(-1, self.obj_dim)
        bbox_weights_refine = bbox_weights_refine.reshape(-1, self.obj_dim)
        if box_extra_refine is not None:
            box_extra_refine = box_extra_refine.reshape(-1, self.box_extra_dims)
        pts_pred_refine = pts_pred_refine.reshape(-1, 2 * self.num_points)
        bbox_pred_refine = self.points2bbox(
            pts_pred_refine, y_first=False,
            out_line_constrain=self.line_constrain_loss,
            box_extra=box_extra_refine,
            bbox_weights = bbox_weights_refine,
            bbox_gt = bbox_gt_refine,
            stage='loss_single_refine')
        normalize_term = self.point_base_scale * stride

        if self.obj_rep == 'box_scope':
          bbox_pred_init_nm = bbox_pred_init / normalize_term
          bbox_gt_init_nm = bbox_gt_init / normalize_term
        elif self.obj_rep == 'XYXYSin2' or self.obj_rep == 'XYXYSin2WZ0Z1':
          bbox_pred_init_nm = bbox_pred_init / normalize_term
          bbox_gt_init_nm = bbox_gt_init / normalize_term
          bbox_pred_init_nm[:,4] = bbox_pred_init[:,4]
          bbox_gt_init_nm[:,4] = bbox_gt_init[:,4]
        elif self.obj_rep == 'Rect4CornersZ0Z1':
          bbox_pred_init_nm = bbox_pred_init / normalize_term
          bbox_gt_init_nm = bbox_gt_init / normalize_term
        elif self.obj_rep == 'XYLgWsAsinSin2Z0Z1' or self.obj_rep == 'XYLgWsAbsSin2Z0Z1':
          Xc, Yc, Lg, Ws, Asin, Sin2, Z0, Z1 = range(8)
          bbox_pred_init_nm = bbox_pred_init / normalize_term
          bbox_gt_init_nm = bbox_gt_init / normalize_term
          bbox_pred_init_nm[:,[Asin, Sin2]] = bbox_pred_init[:,[Asin, Sin2]]
          bbox_gt_init_nm[:,[Asin, Sin2]] = bbox_gt_init[:,[Asin, Sin2]]
        elif self.obj_rep == 'XYDAsinAsinSin2Z0Z1':
          Xc, Yc, Dl, AsinCor, Asin, Sin2, Z0, Z1 = range(8)
          bbox_pred_init_nm = bbox_pred_init / normalize_term
          bbox_gt_init_nm = bbox_gt_init / normalize_term
          bbox_pred_init_nm[:,[AsinCor, Asin, Sin2]] = bbox_pred_init[:,[AsinCor, Asin, Sin2]]
          bbox_gt_init_nm[:,[AsinCor, Asin, Sin2]] = bbox_gt_init[:,[AsinCor, Asin, Sin2]]

        loss_pts_init = cal_loss_bbox('init', self.obj_rep, self.loss_bbox_init,
                                  bbox_pred_init_nm, bbox_gt_init_nm,
                                  bbox_weights_init, num_total_samples_init,
                                  self.transform_method)

        if self.line_constrain_loss:
          assert bbox_pred_init.shape[1] == self.obj_dim + 1
          gt_line_constrain_init = torch.zeros_like(bbox_gt_init)[:,0:1]
          line_cons_weights_init = bbox_weights_init[:,0:1]
          loss_linec_init = self.loss_line_constrain_init(
              bbox_pred_init[:,self.obj_dim:],
              gt_line_constrain_init,
              line_cons_weights_init,
              avg_factor=num_total_samples_init)

        if self.obj_rep == 'box_scope':
          bbox_pred_refine_nm = bbox_pred_refine / normalize_term
          bbox_gt_refine_nm = bbox_gt_refine / normalize_term
        elif self.obj_rep == 'XYXYSin2' or self.obj_rep == 'XYXYSin2WZ0Z1':
          bbox_pred_refine_nm = bbox_pred_refine / normalize_term
          bbox_gt_refine_nm = bbox_gt_refine / normalize_term
          bbox_pred_refine_nm[:,4] = bbox_pred_refine[:,4]
          bbox_gt_refine_nm[:,4] = bbox_gt_refine[:,4]
          #bbox_pred_refine_nm[:, self.obj_dim] = bbox_pred_refine[:, self.obj_dim]
        elif self.obj_rep == 'Rect4CornersZ0Z1':
          bbox_pred_refine_nm = bbox_pred_refine / normalize_term
          bbox_gt_refine_nm = bbox_gt_refine / normalize_term
        elif self.obj_rep == 'XYLgWsAsinSin2Z0Z1' or self.obj_rep == 'XYLgWsAbsSin2Z0Z1':
          bbox_pred_refine_nm = bbox_pred_refine / normalize_term
          bbox_gt_refine_nm = bbox_gt_refine / normalize_term
          bbox_pred_refine_nm[:,[Asin, Sin2]] = bbox_pred_refine[:,[Asin, Sin2]]
          bbox_gt_refine_nm[:,[Asin, Sin2]] = bbox_gt_refine[:,[Asin, Sin2]]
        elif self.obj_rep == 'XYDAsinAsinSin2Z0Z1':
          bbox_pred_refine_nm = bbox_pred_refine / normalize_term
          bbox_gt_refine_nm = bbox_gt_refine / normalize_term
          bbox_pred_refine_nm[:,[AsinCor, Asin, Sin2]] = bbox_pred_refine[:,[AsinCor, Asin, Sin2]]
          bbox_gt_refine_nm[:,[AsinCor, Asin, Sin2]] = bbox_gt_refine[:,[AsinCor, Asin, Sin2]]


        loss_pts_refine = cal_loss_bbox('refine', self.obj_rep, self.loss_bbox_refine,
                                  bbox_pred_refine_nm, bbox_gt_refine_nm,
                                  bbox_weights_refine, num_total_samples_refine,
                                  self.transform_method)

        if self.line_constrain_loss:
          assert bbox_pred_refine.shape[1] == self.obj_dim + 1
          gt_line_constrain_refine = torch.zeros_like(bbox_gt_refine)[:,0:1]
          line_cons_weights_refine = bbox_weights_refine[:,0:1]
          loss_linec_refine = self.loss_line_constrain_refine(
              bbox_pred_refine[:,self.obj_dim:],
              gt_line_constrain_refine,
              line_cons_weights_refine,
              avg_factor=num_total_samples_refine)
        else:
          loss_linec_init  = None
          loss_linec_refine = None

        if DEBUG_CFG.VISUALIZE_VALID_LOSS_SAMPLES:
          print(f'num_total_samples_init:  {num_total_samples_init}\nnum_total_samples_refine: {num_total_samples_refine}')
          print(f'stride: {stride}')
          show_pred('init',self.obj_rep, bbox_pred_init, bbox_gt_init, bbox_weights_init, loss_pts_init, loss_linec_init, pts_pred_init)
          show_pred('refine',self.obj_rep, bbox_pred_refine, bbox_gt_refine, bbox_weights_refine, loss_pts_refine, loss_linec_refine, pts_pred_refine)


        return loss_cls, loss_pts_init, loss_pts_refine, loss_linec_init, loss_linec_refine

    def get_bbox_from_pts(self, center_list, pts_preds, box_extras, stage=None):
        if self.box_extra_dims > 0:
          assert len(box_extras) == len(pts_preds) and pts_preds[0].shape[2:] == box_extras[0].shape[2:]
        else:
          assert box_extras[0] is None
        bbox_list = []
        for i_img, center in enumerate(center_list):
            bbox = []
            for i_lvl in range(len(pts_preds)):
                if self.box_extra_dims >0:
                  box_extra_i = box_extras[i_lvl].detach()
                else:
                  box_extra_i = None
                bbox_preds_init = self.points2bbox(
                    pts_preds[i_lvl].detach(), box_extra=box_extra_i, stage=stage+'_get_bbox_from_pts')
                if self.transform_method == 'center_size_istopleft':
                  raise NotImplementedError
                elif self.obj_rep == 'XYXY':
                  assert self.transform_method == 'moment'
                  assert bbox_preds_init.shape[1] == 4
                  bbox_shift = bbox_preds_init * self.point_strides[i_lvl]
                  bbox_center = torch.cat(
                      [center[i_lvl][:, :2], center[i_lvl][:, :2]], dim=1)
                  bbox.append(bbox_center +
                              bbox_shift[i_img].permute(1, 2, 0).reshape(-1, 4))
                elif self.obj_rep == 'XYXYSin2':
                  assert self.transform_method == 'moment_XYXYSin2'
                  assert bbox_preds_init.shape[1] == 5
                  bbox_shift = bbox_preds_init[:,:4] * self.point_strides[i_lvl]
                  istopleft = bbox_preds_init[i_img,4:5]. permute(1,2,0).reshape(-1,1)
                  bbox_center = torch.cat([center[i_lvl][:, :2], center[i_lvl][:, :2]], dim=1)
                  bbox_i = bbox_center + bbox_shift[i_img].permute(1, 2, 0).reshape(-1, 4)
                  bbox_i = torch.cat([bbox_i, istopleft], dim=1)
                  bbox.append(bbox_i)
                elif self.obj_rep == 'Rect4CornersZ0Z1':
                  assert self.transform_method == 'sort_4corners'
                  assert bbox_preds_init.shape[1] == 10
                  bbox_i = bbox_preds_init[i_img].permute(1, 2, 0).reshape(-1,10)
                  bbox_i *=  self.point_strides[i_lvl]
                  bbox_center = center[i_lvl][:, :2].repeat(1,4)
                  bbox_i[:, :8] += bbox_center
                  bbox.append(bbox_i)
                elif self.obj_rep == 'XYXYSin2WZ0Z1':
                  assert self.transform_method == 'moment_XYXYSin2WZ0Z1'
                  assert bbox_preds_init.shape[1] == 8
                  bbox_preds_init_i = bbox_preds_init[i_img].permute(1,2,0).reshape(-1,8)
                  bbox_shift = bbox_preds_init_i[:,:4] * self.point_strides[i_lvl]
                  istopleft = bbox_preds_init_i[:,4:5]
                  bbox_center = torch.cat([center[i_lvl][:, :2], center[i_lvl][:, :2]], dim=1)
                  bbox_i = bbox_center + bbox_shift
                  wz0z1 =  bbox_preds_init_i[:,5:8] * self.point_strides[i_lvl]
                  bbox_i = torch.cat([bbox_i, istopleft, wz0z1], dim=1)
                  bbox.append(bbox_i)
                elif self.obj_rep == 'XYDAsinAsinSin2Z0Z1':
                  assert bbox_preds_init.shape[1] == 8
                  bbox_i = bbox_preds_init[i_img].permute(1, 2, 0).reshape(-1,8)
                  bbox_i[:,[0,1,2,6,7]] *=  self.point_strides[i_lvl]
                  bbox_center = center[i_lvl][:, :2]
                  bbox_i[:,[0,1]] += bbox_center
                  bbox.append(bbox_i)
                elif self.obj_rep == 'XYLgWsAbsSin2Z0Z1':
                  assert bbox_preds_init.shape[1] == 8
                  bbox_i = bbox_preds_init[i_img].permute(1, 2, 0).reshape(-1,8)
                  bbox_i[:,[0,1,2,3,6,7]] *=  self.point_strides[i_lvl]
                  bbox_center = center[i_lvl][:, :2]
                  bbox_i[:,[0,1]] += bbox_center
                  bbox.append(bbox_i)
                  pass
                else:
                  raise NotImplementedError
            bbox_list.append(bbox)
        return bbox_list


    def loss(self,
             cls_scores,
             pts_preds_init,
             pts_preds_refine,
             corner_outs,
             rel_feat_outs,
             box_extra_init,
             box_extra_refine,
             gt_bboxes,
             gt_labels,
             gt_relations,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
      npts = self.num_points * 2
      num_level = len(pts_preds_init)
      bs = len(gt_labels)
      s = 0
      gt_ids_mapping = [[] for _ in range(bs)]
      for i, ncg in enumerate(self.num_per_group):
        self.cls_out_channels_g = self.num_per_group[i]
        self.group_id = i
        cls_scores_i = []
        for cs in cls_scores:
          cls_scores_i.append( { e:cs[e][:,s:s+ncg] for e in cs } )
        s += ncg
        pts_preds_init_i = [pts_preds_init[j][:,i*npts:(i+1)*npts] for j in range(num_level)]
        pts_preds_refine_i = [pts_preds_refine[j][:,i*npts:(i+1)*npts] for j in range(num_level)]
        gt_bboxes_i = []
        gt_labels_i = []
        for j in range(bs):
          mask_ls = [gt_labels[j] == c for c in self.cls_groups[i] ]
          gt_mask = sum(mask_ls)>0
          gt_bboxes_i.append( gt_bboxes[j][gt_mask] )
          gt_labels_i.append( gt_labels[j][gt_mask] )
          gt_ids_mapping[j].append( torch.nonzero(gt_mask).squeeze(1) )
          #gt_nums_perg[j].append(gt_labels_i[-1].shape[0])

        loss_dict_i, pos_inds_list_refine_i, gt_inds_per_pos_list_refine_i  = self.loss_per_group(
              cls_scores_i,
              pts_preds_init_i,
              pts_preds_refine_i,
              corner_outs,
              rel_feat_outs,
              box_extra_init,
              box_extra_refine,
              gt_bboxes_i,
              gt_labels_i,
              gt_relations,
              img_metas,
              cfg,
              gt_bboxes_ignore)
        if i==0:
          loss_dict_all = loss_dict_i
          pos_inds_list_refine = pos_inds_list_refine_i
          gt_inds_per_pos_list_refine = gt_inds_per_pos_list_refine_i
        else:
          for e in loss_dict_i:
            loss_dict_all[e] += loss_dict_i[e]
          for bi in range(bs):
            tmp = gt_ids_mapping[bi][i]  [ gt_inds_per_pos_list_refine_i[bi]]
            gt_inds_per_pos_list_refine[bi] = torch.cat( [gt_inds_per_pos_list_refine[bi], tmp] )
            # directly overlap the inds of two groups
            pos_inds_list_refine[bi] = torch.cat( [pos_inds_list_refine[bi], pos_inds_list_refine_i[bi]] )

      #-----------------------------------------------------------------------
      # relation loss
      if self.relation_cfg['enable']:
          if self.relation_cfg['stage'] == 'refine':
              pos_inds_list = pos_inds_list_refine
              gt_inds_per_pos_list = gt_inds_per_pos_list_refine
          wall_pos_inds_list, wall_gt_inds_per_pos_list = self.get_wall_pos(
              gt_labels, pos_inds_list, gt_inds_per_pos_list)
          #num_flat = sum([cs['refine'].shape[2:].numel() for cs in cls_scores])
          #num_flats = (num_flat, ) * len(gt_labels)
          relation_scores, relation_inds, gt_inds_per_rel = \
            self.forward_relation_cls(rel_feat_outs,
                    pos_inds_list, gt_inds_per_pos_list,
                    self.relation_cfg['max_relation_num'])
          loss_relation_wall, = multi_apply(
              self.obj_relation_cls_loss,
              relation_scores,
              gt_inds_per_rel,
              gt_relations,
              gt_labels,
              gt_bboxes,)
          loss_dict_all['loss_relation'] = loss_relation_wall

      return loss_dict_all

    def loss_per_group(self,
             cls_scores,
             pts_preds_init,
             pts_preds_refine,
             corner_outs,
             rel_feat_outs,
             box_extra_init,
             box_extra_refine,
             gt_bboxes,
             gt_labels,
             gt_relations,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        if RECORD_TIME:
          t0 = time.time()

        #_show_objs_ls_points_ls_torch((512,512), gt_bboxes, self.obj_rep)

        if gt_relations is None:
          assert self.relation_cfg['enable'] == 0
          gt_relations = [None]*len(gt_bboxes)
        #-----------------------------------------------------------------------
        featmap_sizes = [featmap.size()[-2:] for featmap in pts_preds_init]
        assert len(featmap_sizes) == len(self.point_generators)
        label_channels = self.cls_out_channels_g if self.use_sigmoid_cls else 1
        #-----------------------------------------------------------------------
        # target for initial stage
        center_list, valid_flag_list = self.get_points(featmap_sizes,
                                                       img_metas)
        pts_coordinate_preds_init = self.offset_to_pts(center_list,
                                                       pts_preds_init)
        if cfg.init.assigner['type'] == 'PointAssigner':
            # Assign target for center list
            candidate_list = center_list # [ []*num_level ]*batch_size
        else:
            # transform center list to bbox list and
            #   assign target for bbox list
            bbox_list_org = self.centers_to_bboxes(center_list)
            candidate_list = bbox_list_org
        cls_reg_targets_init = point_target(
            candidate_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            cfg.init,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling,
            flag='init')
        ( labels_list_init, label_weights_list_init, bbox_gt_list_init,
          candidate_list_init, bbox_weights_list_init, num_total_pos_init,
          num_total_neg_init, pos_inds_list_init, gt_inds_per_pos_list_init
          ) = cls_reg_targets_init
        num_total_samples_init = (
            num_total_pos_init +
            num_total_neg_init if self.sampling else num_total_pos_init)

        #-----------------------------------------------------------------------
        # target for refinement stage
        center_list, valid_flag_list = self.get_points(featmap_sizes,
                                                       img_metas)
        pts_coordinate_preds_refine = self.offset_to_pts(
            center_list, pts_preds_refine)
        bbox_list_initres = self.get_bbox_from_pts(center_list, pts_preds_init, box_extra_init, stage='init')
        #debug_utils.show_shapes(pts_preds_init, 'StrPointsHead init')
        #debug_utils.show_shapes(bbox_list_initres, 'StrPointsHead init bbox')
        cls_reg_targets_refine = point_target(
            bbox_list_initres,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            cfg.refine,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling,
            flag='refine')
        (labels_list_refine, label_weights_list_refine, bbox_gt_list_refine,
         candidate_list_refine, bbox_weights_list_refine, num_total_pos_refine,
         num_total_neg_refine, pos_inds_list_refine,
         gt_inds_per_pos_list_refine) = cls_reg_targets_refine
        num_total_samples_refine = (
            num_total_pos_refine +
            num_total_neg_refine if self.sampling else num_total_pos_refine)

        #self.debug_pos_scores(cls_scores, pos_inds_list_refine)
        #-----------------------------------------------------------------------
        if 'final' in self.cls_types:
          center_list, valid_flag_list = self.get_points(featmap_sizes,
                                                       img_metas)
          bbox_list_refineres = self.get_bbox_from_pts(center_list, pts_preds_refine, box_extra_refine, stage='refine')
          #debug_utils.show_shapes(pts_preds_refine, 'StrPointsHead refine')
          #debug_utils.show_shapes(bbox_list_refineres, 'StrPointsHead refine bbox')
          cls_reg_targets_final = point_target(
              bbox_list_refineres,
              valid_flag_list,
              gt_bboxes,
              img_metas,
              cfg.refine,
              gt_bboxes_ignore_list=gt_bboxes_ignore,
              gt_labels_list=gt_labels,
              label_channels=label_channels,
              sampling=self.sampling,
              flag='final')
          (labels_list_final, label_weights_list_final, bbox_gt_list_final,
          candidate_list_final, bbox_weights_list_final, num_total_pos_final,
          num_total_neg_final, pos_inds_list_final, gt_inds_per_pos_list_final)\
              = cls_reg_targets_final
          num_total_samples_finale = (
              num_total_pos_final +
              num_total_neg_final if self.sampling else num_total_pos_final)


        #-----------------------------------------------------------------------
        labels_list = []
        label_weights_list = []
        num_total_samples_cls = {}
        if 'refine' in self.cls_types:
          num_total_samples_cls['refine'] = num_total_samples_refine
        if 'final' in self.cls_types:
          num_total_samples_cls['final'] = num_total_samples_finale
        for i in range(len(label_weights_list_refine)):
          labels_list_i = {}
          label_weights_list_i = {}
          if 'refine' in self.cls_types:
            labels_list_i['refine'] = labels_list_refine[i]
            label_weights_list_i['refine'] = label_weights_list_refine[i]
          if 'final' in self.cls_types:
            labels_list_i['final'] = labels_list_final[i]
            label_weights_list_i['final'] = label_weights_list_final[i]
          labels_list.append(labels_list_i)
          label_weights_list.append(label_weights_list_i)

        #-----------------------------------------------------------------------
        if self.box_extra_dims>0:
          assert box_extra_init[0].shape[1] == box_extra_refine[0].shape[1] == self.box_extra_dims
          bs = box_extra_init[0].shape[0]
          box_extra_init = [e.permute(0,2,3,1).reshape(bs, -1, self.box_extra_dims) for e in box_extra_init]
          box_extra_refine = [e.permute(0,2,3,1).reshape(bs, -1, self.box_extra_dims) for e in box_extra_refine]
        else:
          assert box_extra_init[0] is None
        if RECORD_TIME:
          t1 = time.time()
        #-----------------------------------------------------------------------
        # compute loss per level
        losses_cls, losses_pts_init, losses_pts_refine,\
          loss_linec_init, loss_linec_refine\
            = multi_apply(
            self.loss_single,
            cls_scores,
            pts_coordinate_preds_init,
            pts_coordinate_preds_refine,
            box_extra_init,
            box_extra_refine,
            labels_list,
            label_weights_list,
            bbox_gt_list_init,
            bbox_weights_list_init,
            bbox_gt_list_refine,
            bbox_weights_list_refine,
            self.point_strides,
            num_total_samples_init=num_total_samples_init,
            num_total_samples_refine=num_total_samples_refine,
            num_total_samples_cls=num_total_samples_cls)
        loss_dict_all = {}
        for ele in losses_pts_init[0].keys():
          loss_dict_all[ele] = [l[ele] for l in losses_pts_init]
        for ele in losses_pts_refine[0].keys():
          loss_dict_all[ele] = [l[ele] for l in losses_pts_refine]

        if self.line_constrain_loss:
          loss_dict_all['loss_clI'] = loss_linec_init
          loss_dict_all['loss_clR'] = loss_linec_refine
        for c in self.cls_types:
          cstr = {'init':'I', 'refine':'R', 'final':'F'}[c]
          loss_dict_all['loss_cls'+cstr] = []
          for lc in losses_cls:
            loss_dict_all['loss_cls'+cstr].append( lc[c] )

        ##-----------------------------------------------------------------------
        ## relation loss
        #if self.relation_cfg['enable']:
        #    if self.relation_cfg['stage'] == 'refine':
        #        pos_inds_list = pos_inds_list_refine
        #        gt_inds_per_pos_list = gt_inds_per_pos_list_refine
        #    wall_pos_inds_list, wall_gt_inds_per_pos_list = self.get_wall_pos(
        #        gt_labels, pos_inds_list, gt_inds_per_pos_list)
        #    #num_flat = sum([cs['refine'].shape[2:].numel() for cs in cls_scores])
        #    #num_flats = (num_flat, ) * len(gt_labels)
        #    relation_scores, relation_inds, gt_inds_per_rel = \
        #      self.forward_relation_cls(rel_feat_outs,
        #              wall_pos_inds_list, wall_gt_inds_per_pos_list,
        #              self.relation_cfg['max_relation_num'])
        #    import pdb; pdb.set_trace()  # XXX BREAKPOINT
        #    loss_relation_wall, = multi_apply(
        #        self.obj_relation_cls_loss,
        #        relation_scores,
        #        relation_inds,
        #        gt_inds_per_rel,
        #        gt_relations,
        #        gt_labels,
        #        gt_bboxes,)
        #    loss_dict_all['loss_relation'] = loss_relation_wall

        ##-----------------------------------------------------------------------
        # target for corner
        if self.corner_hm:
          loss_corner_hm = self.corner_loss(corner_outs, gt_bboxes,
                                 gt_labels, img_metas,cfg, gt_bboxes_ignore)
          if self.corner_hm_only:
            return loss_corner_hm
        if self.corner_hm:
          loss_dict_all.update(loss_corner_hm)

        if DEBUG and 0:
          loss_dict_all_new = {}
          #for e in ['loss_ptsR', 'loss_clsF']:
          for e in loss_dict_all.keys():
            if 'I'  in e:
              loss_dict_all_new[e] =   loss_dict_all[e]
          return loss_dict_all_new
        if RECORD_TIME:
          t_A = t1 - t0
          t_B = time.time() - t1
          print(f't_A: {t_A:.3f}\nt loss: {t_B:.3f}')
        return loss_dict_all, pos_inds_list_refine, gt_inds_per_pos_list_refine

    def get_wall_pos(self,  gt_labels, pos_inds_list, gt_inds_per_pos_list):
      batch_size = len(gt_labels)
      wall_pos_inds_list = []
      wall_gt_inds_per_pos_list = []
      for i in range(batch_size):
        # reamin only wall in positive detection
        pos_labels = gt_labels[i][ gt_inds_per_pos_list[i] ]
        wall_mask = pos_labels == self.wall_label
        wall_pos_inds_list.append( pos_inds_list[i][wall_mask] )
        wall_gt_inds_per_pos_list.append( gt_inds_per_pos_list[i][wall_mask] )

      return wall_pos_inds_list, wall_gt_inds_per_pos_list

    def obj_relation_cls_loss(self, relation_scores,
              gt_inds_per_rel, gt_relations, gt_labels, gt_bboxes=None ):
        '''
        Valid sample for relation classification: in both high_score_inds and pos_inds
        wall only input

        n: num of positive detections for all classes
        relation_scores: [1,1,n,n]
        relation_inds: [n]
        gt_inds_per_rel: [n]
        gt_relations: [n_gt_all, n_gt_wall]
        gt_labels: [n_gt_all]
        '''
        n = gt_inds_per_rel.numel()
        n_gt_all, n_gt_wall = gt_relations.shape
        assert relation_scores.shape == (1,1,n,n)
        assert gt_relations.shape[0] == gt_labels.shape[0]
        assert sum(gt_labels == self.wall_label) == n_gt_wall

        self_relation =  'True'
        weight = None
        if self_relation == 'True':
          gt_relations.fill_diagonal_(True)
        elif self_relation == 'False':
          gt_relations.fill_diagonal_(False)
        elif self_relation == 'Ignore':
          weight = torch.ones_like(relation_scores[0][0])
          weight.fill_diagonal_(0).view(-1)
        else:
          raise ValueError

        # only cal rel between wall and other categories (including wall)
        gt_labels_per_rel = gt_labels[gt_inds_per_rel]
        mask = gt_labels_per_rel == self.wall_label
        relation_scores = relation_scores[:,:,:, mask]
        non_wall_gt_inds_per_rel = gt_inds_per_rel[mask]

        c = relation_scores.shape[1]
        assert c==1, f"class num ={c}, but current version only allow wall"
        relation_scores_flat = relation_scores.permute(0,2,3,1).view(-1,c)
        gt_relations_rel = gt_relations[gt_inds_per_rel, :][:,non_wall_gt_inds_per_rel]
        gt_relation_labels = gt_relations_rel.view(-1).to(torch.long)
        num_rel_sample = gt_relation_labels.sum()

        # cal loss
        loss_relation_wall = self.loss_relation(
            relation_scores_flat,
            gt_relation_labels,
            weight=weight,
            avg_factor=num_rel_sample,)

        if DEBUG_CFG.SHOW_RELATION_IN_TRAIN:
          #show_relations(gt_bboxes, gt_relations)

          gt_rel_bboxes = gt_bboxes[gt_inds_per_rel,:]
          #show_relations(gt_rel_bboxes, gt_relations_rel)

          relation_scores_nm = relation_scores.sigmoid()[0,0]
          relation_mask = relation_scores_nm > 0.3
          #relation_mask.fill_diagonal_(False)
          show_relations(gt_rel_bboxes, relation_mask)

        return (loss_relation_wall, )

    def obj_relation_cls_loss_high_score(self, num_flat, relation_scores, high_score_inds,
                    gt_relations, pos_inds, gt_inds_per_pos, gt_labels ):
        '''
        Valid sample for relation classification: in both high_score_inds and pos_inds
        '''
        relation_scores  = relation_scores.squeeze(0).squeeze(0)
        # remain only wall in positive detection
        labels_pos = gt_labels[gt_inds_per_pos]
        pos_wall_mask = labels_pos == self.wall_label
        wall_pos_inds = pos_inds[pos_wall_mask]
        wall_num = wall_pos_inds.shape[0]
        gt_inds_per_pos = gt_inds_per_pos[pos_wall_mask]

        # remain only wall in gt
        gt_wall_mask = gt_labels == self.wall_label
        wall_gt_relations = gt_relations[gt_wall_mask,:]
        gt_wall_num = wall_gt_relations.shape[0]
        assert wall_gt_relations.shape[1] == gt_wall_num

        # find the inds in both high_score_inds and pos_inds
        tmp1 = torch.zeros([num_flat], dtype=torch.int32)
        tmp1[high_score_inds] = 1
        tmp2 = torch.zeros([num_flat], dtype=torch.int32)
        tmp2[wall_pos_inds] = 1
        tmp3 = tmp1 + tmp2
        valid_inds = torch.nonzero(tmp3 == 2).squeeze(1)
        tmp4 = tmp3[high_score_inds]
        valid_inds_in_highscore = torch.nonzero(tmp4 == 2).squeeze(1)
        tmp5 = tmp3[wall_pos_inds]
        valid_inds_in_pos = torch.nonzero(tmp5 == 2).squeeze(1)

        npos = pos_inds.shape[0]
        npos_wall = wall_pos_inds.shape[0]
        nhscore = high_score_inds.shape[0]
        nvalid = valid_inds_in_highscore.shape[0]
        #print(f'\n\n gt num: {gt_labels.shape[0]}\n wall gt num: {gt_wall_num}')
        #print(f'npos: {npos}\n wall pos: {npos_wall}\n nhscore: {nhscore}\n nvalid: {nvalid}\n')

        # get the final relations_scores for loss computation
        valid_relations_scores = relations_scores[valid_inds_in_highscore,:][:,valid_inds_in_highscore]
        valid_relations_scores = valid_relations_scores.view(-1,1)

        # get the relation gt
        gt_inds_per_valid = gt_inds_per_pos[valid_inds_in_pos]
        gt_relations_valid = wall_gt_relations[gt_inds_per_valid, :][:,gt_inds_per_valid]
        gt_relation_labels = gt_relations_valid.view(-1).to(torch.long)

        #gt_relation_labels[:] = 1
        valid_relations_scores[:] = -10

        # cal loss
        loss_relation_wall = self.loss_relation(
            valid_relations_scores,
            gt_relation_labels,
            weight=None,
            avg_factor=None,)
        return (loss_relation_wall, )

    def corner_loss(self, corner_outs, gt_bboxes,
                    gt_labels, img_metas,cfg, gt_bboxes_ignore):
        assert corner_outs[1] is None
        corner_outs = corner_outs[0]
        cor_scores = corner_outs['cor_scores']
        cor_centerness = corner_outs['cor_centerness']
        cor_ofs = corner_outs['cor_ofs']
        gt_corners_lab = [gen_corners_from_lines_th(gb, gl, OBJ_REP) for gb,gl in zip(gt_bboxes, gt_labels)]
        gt_corners = [d[0] for d in gt_corners_lab]
        gt_labels = [d[1] for d in gt_corners_lab]
        if gt_bboxes_ignore is None:
          gt_corners_ignore = None

        featmap_sizes =  [cor_scores.size()[-2:]]
        center_list, valid_flag_list = self.get_points(featmap_sizes, img_metas) # []*batch_size
        candidate_list = center_list # [ []*num_level ]*batch_size
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        cls_reg_targets_cor = point_target(
            candidate_list,
            valid_flag_list,
            gt_corners,
            img_metas,
            cfg.corner,
            gt_bboxes_ignore_list=gt_corners_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling,
            flag='corner')
        (labels_list, label_weights_list, cor_gt_list, candidate_list_neg0, cor_reg_weights_list,
         num_total_pos_cor, num_total_neg_cor) = cls_reg_targets_cor
        num_total_samples_cor = (
            num_total_pos_cor +
            num_total_neg_cor if self.sampling else num_total_pos_cor)


        cor_ofs_gt_list = [ gt-can[...,:2] for gt,can in zip(cor_gt_list, candidate_list_neg0) ]
        cor_centerness_gt_list = [pospro[...,3] for pospro in candidate_list_neg0]
        #-----------------------------------------------------------------------
        # compute loss per level
        losses_cls, losses_centerness, losses_ofs = multi_apply(
            self.loss_single_corner,
            [cor_scores],
            [cor_centerness],
            [cor_ofs],
            labels_list,
            label_weights_list,
            cor_centerness_gt_list,
            cor_ofs_gt_list,
            cor_reg_weights_list,
            self.point_strides[0:1],
            num_total_samples_cor=num_total_samples_cor,
        )
        loss_dict_all = {
            'loss_cor_cls': losses_cls,
            'loss_cor_cen': losses_centerness,
        }
        #loss_dict_all['loss_cor_ofs'] = losses_ofs
        return loss_dict_all

    def debug_pos_scores(self, cls_scores, pos_inds_list_refine):
        # debug: score of positive
        batch_size, num_classes = cls_scores[0]['refine'].shape[:2]
        cls_scores_flat = [cs['refine'].view(batch_size, num_classes, -1) for cs in cls_scores]
        cls_scores_flat = torch.cat(cls_scores_flat, axis=-1)
        if self.use_sigmoid_cls:
          cls_scores_flat = cls_scores_flat.sigmoid()
        else:
          cls_scores_flat = cls_scores_flat.softmax(-1)
        scores_pos = cls_scores_flat[0, :, pos_inds_list_refine[0]]
        scores_pos_wall = scores_pos[0]
        import pdb; pdb.set_trace()  # XXX BREAKPOINT

    def loss_single_corner(self, cor_score, cor_centerness, cor_ofs, labels,
                    label_weights, cor_centerness_gt, cor_ofs_gt, cor_reg_weights, stride,
                    num_total_samples_cor):
        '''
        fsfs = featmap_size * featmap_size
        cor_score:      [batch_size, self.cls_out_channels, featmap_size, featmap_size]
        cor_centerness: [batch_size, 1, featmap_size, featmap_size]
        cor_ofs: [batch_size, 2, featmap_size, featmap_size]
        labels: [batch_size, fsfs]
        label_weights: [batch_size, fsfs]
        cor_gt: [batch_size, fsfs, 2]
        cor_reg_weights: [batch_size, fsfs, 2]
        '''
        obj_dim = cor_ofs_gt.shape[-1]

        if 0:
          debug_utils.show_heatmap(labels[0].reshape(128,128), (512,512))
          debug_utils.show_heatmap(label_weights[0].reshape(128,128), (512,512))
          debug_utils.show_heatmap(cor_centerness_gt[0].reshape(128,128), (512,512))
          debug_utils.show_heatmap(cor_reg_weights[0,:,0].reshape(128,128), (512,512))
          ofs_img = (cor_ofs_gt.norm(dim=-1) / 10).clamp(max=1)
          debug_utils.show_heatmap(ofs_img[0].reshape(128,128), (512,512))


        cor_score = cor_score.permute(0,2,3,1).reshape(-1, self.cls_out_channels)
        cor_centerness = cor_centerness.permute(0,2,3,1).reshape(-1)
        cor_ofs = cor_ofs.permute(0,2,3,1).reshape(-1,obj_dim)

        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cor_reg_weights = cor_reg_weights.reshape(-1,obj_dim)
        cor_centerness_gt = cor_centerness_gt.reshape(-1)

        loss_cls = self.loss_cls(
            cor_score,
            labels,
            label_weights,
            avg_factor=num_total_samples_cor)

        normalize_term = self.point_base_scale * stride

        loss_centerness = self.loss_centerness(
            cor_centerness,
            cor_centerness_gt,
        )

        cor_ofs_nm = cor_ofs / normalize_term
        cor_ofs_gt_nm = cor_ofs_gt.reshape(-1,obj_dim) / normalize_term
        loss_ofs = self.loss_cor_ofs(
            cor_ofs_nm,
            cor_ofs_gt_nm,
            cor_reg_weights,
            avg_factor=num_total_samples_cor)
        return loss_cls, loss_centerness, loss_ofs


    def cor_fcos_target(self, points, gt_corners, gt_labels):
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)
        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list = multi_apply(
            self.fcos_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges)

        # split to per img, per level
        num_points = [center.size(0) for center in points]
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_bbox_targets.append(
                torch.cat(
                    [bbox_targets[i] for bbox_targets in bbox_targets_list]))
        return concat_lvl_labels, concat_lvl_bbox_targets

    def cal_test_score(self, cls_scores):
      assert 'refine' in cls_scores[0].keys()
      with_final = 'final' in cls_scores[0].keys()
      ave_cls_scores = []
      cls_scores_refine_final = []
      for i in range( len(cls_scores) ):
        s_refine = cls_scores[i]['refine']
        if with_final:
          s_final = cls_scores[i]['final']
          s = s_refine * self.dim_parse.LINE_CLS_WEIGHTS['refine']  + s_final * self.dim_parse.LINE_CLS_WEIGHTS['final']

          sff = torch.cat([cls_scores[i]['refine'], cls_scores[i]['final']], dim=1)
          cls_scores_refine_final.append(sff)
        else:
          s = s_refine
          cls_scores_refine_final.append(s.repeat(1,2,1,1))

        ave_cls_scores.append(s)

        #tmp = list(cls_scores[i].values())
        #ave_cls_scores.append( sum(tmp) / len(tmp) )
      return ave_cls_scores, cls_scores_refine_final

    def get_relations(self, rel_scores):
        if self.use_sigmoid_cls:
            scores = rel_scores.sigmoid()
        else:
            scores = rel_scores.softmax(-1)
        return scores

    def get_bboxes(self,
                   cls_scores,
                   pts_preds_init,
                   pts_preds_refine,
                   corner_outs,
                   rel_feat_outs,
                   box_extra_inits,
                   box_extra_refines,
                   img_metas,
                   cfg,
                   rescale=False,
                   nms=True):
        assert len(cls_scores) == len(pts_preds_refine)
        cls_scores, cls_scores_refine_final = self.cal_test_score(cls_scores)
        bbox_preds_refine = [
            self.points2bbox(pts_pred_refine, box_extra=box_extra_refine)
            for pts_pred_refine, box_extra_refine in zip(pts_preds_refine, box_extra_refines)
        ]
        num_levels = len(cls_scores)

        if self.dim_parse.OUT_EXTAR_DIM > 0:
          for i in range(num_levels):
            bbox_preds_init = [
                self.points2bbox(pts_pred_init, box_extra=box_extra_init)
                for pts_pred_init, box_extra_init in zip(pts_preds_init, box_extra_inits)
            ]
            # OUT_ORDER
            init_refine = torch.cat([bbox_preds_refine[i], bbox_preds_init[i],
                                     pts_preds_refine[i], pts_preds_init[i],
                                     cls_scores_refine_final[i], ], dim=1)
            assert bbox_preds_refine[i].shape[1] == self.dim_parse.OBJ_DIM
            assert init_refine.shape[1] == self.dim_parse.OBJ_DIM + self.dim_parse.OUT_EXTAR_DIM
            bbox_preds_refine[i] = init_refine

        mlvl_points = [
            self.point_generators[i].grid_points(cls_scores[i].size()[-2:],
                                                 self.point_strides[i])
            for i in range(num_levels)
        ]
        result_list = []
        det_inds_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds_refine[i][img_id].detach()
                for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals, det_inds = self.get_bboxes_single(cls_score_list,
                                               bbox_pred_list,
                                               mlvl_points, img_shape,
                                               scale_factor, cfg, rescale, nms)
            result_list.append(proposals)
            det_inds_list.append(det_inds)

        if self.relation_cfg['enable']:
            relation_scores, relation_inds, _ = \
              self.forward_relation_cls(rel_feat_outs, det_inds_list)
            for i in range(len(result_list)):
              result_list[i] += (relation_scores[i].squeeze(0).squeeze(0), )
        else:
            for i in range(len(result_list)):
              result_list[i] += (None, )

        if self.corner_hm:
          cor_heatmap_list = self.get_corners(corner_outs, img_metas, cfg)
          for i in range(len(cor_heatmap_list)):
            line_pred_with_corner_score = self.get_line_corner_cls_ofs(result_list[0][0], cor_heatmap_list[i][0], img_metas[i])
            line_label = result_list[0][1]
            result_list[0] = ( line_pred_with_corner_score, line_label )
          if OUT_CORNER_HM_ONLY:
            result_list =  cor_heatmap_list
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          mlvl_points,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False,
                          nms=True):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        obj_dim = bbox_preds[0].shape[0]
        assert obj_dim == self.dim_parse.NMS_IN_DIM
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_inds = []
        flat_inds_last = 0
        for i_lvl, (cls_score, bbox_pred, points) in enumerate(
                zip(cls_scores, bbox_preds, mlvl_points)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2, 0).\
                                  reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, obj_dim)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            else:
                topk_inds = torch.range(0, scores.shape[0]-1, dtype=torch.long).to(scores.device)
            mlvl_inds.append( topk_inds + flat_inds_last )
            flat_inds_last += cls_score.shape[0]

            #ze = torch.zeros_like(points[:,0:1])
            #bbox_pos_center = points[:, :2].repeat(1, 2)
            #xyxy = bbox_pred[:,:4] * self.point_strides[i_lvl]+ bbox_pos_center
            #x1 = xyxy[:, 0].clamp(min=0, max=img_shape[1])
            #y1 = xyxy[:, 1].clamp(min=0, max=img_shape[0])
            #x2 = xyxy[:, 2].clamp(min=0, max=img_shape[1])
            #y2 = xyxy[:, 3].clamp(min=0, max=img_shape[0])
            #bboxes = torch.stack([x1, y1, x2, y2], dim=-1)

            #if self.obj_rep == 'XYXYSin2':
            #  bboxes = torch.cat([bboxes, bbox_pred[:,4:5]], dim=1)
            #elif self.obj_rep == 'XYXYSin2WZ0Z1':
            #  wz0z1 = bbox_pred[:,5:8] * self.point_strides[i_lvl]
            #  bboxes = torch.cat([bboxes, bbox_pred[:,4:5], wz0z1], dim=1)

            if self.obj_rep == 'XYXYSin2' or self.obj_rep == 'XYXYSin2WZ0Z1':
              num_loc = 2
            elif self.obj_rep == 'Rect4CornersZ0Z1':
              num_loc = 4
            else:
              raise NotImplementedError
            bbox_pos_center = points[:, :2].repeat(1, num_loc)
            bboxes = bbox_pred[:, :self.obj_dim]
            bboxes[:,:num_loc*2] = bboxes[:,:num_loc*2] * self.point_strides[i_lvl] + bbox_pos_center

            if self.dim_parse.OUT_EXTAR_DIM > 0:
              _bboxes_refine, bboxes_init, points_refine, points_init, \
                    score_refine, score_final, score_ave, _, _,_,_,_ = \
                    self.dim_parse.parse_bboxes_out(bbox_pred, 'before_nms')
              #bboxes_init = bbox_pred[:, OBJ_DIM:OBJ_DIM*2]
              bboxes_init[:,:num_loc*2] = bboxes_init[:,:num_loc*2] * self.point_strides[i_lvl] + bbox_pos_center

              bbox_pos_center = points[:, :2].repeat(1, self.dim_parse.POINTS_DIM//2)
              bn = bboxes.shape[0]
              # key_points store in y-first, but box in x-first.
              # change key-points to x-first
              key_points = torch.cat([points_refine, points_init], dim=1)
              key_points = key_points.reshape(bn,-1,2)[:,:,[1,0]].reshape(bn,-1)

              key_points = key_points * self.point_strides[i_lvl] + bbox_pos_center
              for kp in range(0, key_points.shape[1], 2):
                key_points[:,kp] = key_points[:,kp].clamp(min=0, max=img_shape[1])
                key_points[:,kp+1] = key_points[:,kp+1].clamp(min=0, max=img_shape[0])

              if self.use_sigmoid_cls:
                  score_refine = score_refine.sigmoid()
                  score_final = score_final.sigmoid()
              else:
                  score_refine = score_refine.softmax(-1)
                  score_final = score_final.softmax(-1)
              bboxes = torch.cat([bboxes, bboxes_init, key_points, score_refine, score_final], dim=1) # [5,5,36]
              assert bboxes.shape[1] == self.dim_parse.NMS_IN_DIM
              pass

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_inds = torch.cat(mlvl_inds)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        if nms:
            # mlvl_scores: [3256, 2]
            # mlvl_bboxes: [3256, 46]
            # det_bboxes: [66, 47]
            det_bboxes, det_labels, nms_inds = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img,
                                                    obj_rep = self.obj_rep)
            det_inds = mlvl_inds[nms_inds]
            assert det_bboxes.shape[1] == self.dim_parse.NMS_OUT_DIM
            if DEBUG_CFG.SHOW_NMS_OUT:
              show_nms_out(det_bboxes, det_labels, self.obj_rep, self.num_classes)
            return (det_bboxes, det_labels), det_inds
        else:
            return (mlvl_bboxes, mlvl_scores), mlvl_inds

    def get_corners(self, corner_outs, img_metas, cfg):
        corner_outs = corner_outs[0]
        cor_hm_cls = corner_outs['cor_scores'].detach()
        cor_hm_ofs = corner_outs['cor_ofs'].detach()
        cor_centerness = corner_outs['cor_centerness'].detach()
        featmap_size = cor_hm_cls.size()[-2:]
        points = self.point_generators[0].grid_points(featmap_size, self.point_strides[0])
        cor_heatmap_list = []
        for img_id in range(len(img_metas)):
          img_shape = img_metas[img_id]['img_shape']
          scale_factor = img_metas[img_id]['scale_factor']
          labels, cor_cls_cen_ofs = self.get_corners_single(cor_hm_cls[img_id],
                                                   cor_centerness[img_id],
                                        cor_hm_ofs[img_id], points,
                                        img_shape, scale_factor, cfg)
          cor_heatmap_list.append((labels, cor_cls_cen_ofs ))
        return cor_heatmap_list

    def get_corners_single(self, cor_hm_score, cor_centerness, cor_hm_ofs, points, img_shape,
                           scale_factor, cfg, rescale=False, nms=False):
      featmap_size = cor_hm_score.shape[1:]
      cor_hm_score = cor_hm_score.permute(1,2,0).reshape(-1, self.cls_out_channels)
      cor_centerness = cor_centerness.permute(1,2,0).reshape(-1, 1)
      cor_hm_ofs = cor_hm_ofs.permute(1,2,0).reshape(-1, 2)
      if self.use_sigmoid_cls:
          cor_cls_score = cor_hm_score.sigmoid()
      else:
          cor_cls_score = cor_hm_score.softmax(-1)
      cor_centerness = cor_centerness.clamp(min=0, max=1)
      cor_cls_cen_ofs = torch.cat([cor_cls_score, cor_centerness, cor_hm_ofs], dim=1)

      scores = cor_cls_score * cor_centerness
      if self.use_sigmoid_cls:
          max_scores, labels = scores.max(dim=1)
      else:
          max_scores, labels = scores[:, 1:].max(dim=1)
      #assert labels.min() > 0 # all the pixels are outputed in heatmap
      #score_threshold = cfg['score_thr']
      #background_mask = (max_scores > score_threshold).to(labels.dtype)
      #labels *= background_mask


      if 0:
        from mmdet.debug_utils import show_heatmap
        show_heatmap(cor_cls_score.reshape( featmap_size ), (512,512) )
        #show_heatmap(cor_centerness.reshape( featmap_size), (512,512) )
        #show_heatmap(scores.reshape( featmap_size ), (512,512) )
        #ofs_img = (cor_hm_ofs.norm(dim=1)/10).clamp(max=1)
        #show_heatmap(ofs_img.reshape( featmap_size ), (512,512) )
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        pass
      return cor_cls_cen_ofs, labels

    def normalize_istopleft_loss(self, itl_loss):
        itl_loss_nm = 1 - torch.exp(-itl_loss)
        return itl_loss_nm

    def get_line_corner_cls_ofs(self, line_pred, cor_hm_pred, img_meta):
      '''
      line_pred: [n,47]
               [bbox_refine, bbox_init, points_refine, points_init, score]
      cor_hm_pred: [m,4]
      '''
      assert OBJ_REP == 'lscope_istopleft'
      bbox_refine = line_pred[:,:5]
      line_2p_refine_0 = decode_line_rep_th(bbox_refine, OBJ_REP)
      num_lines = line_2p_refine_0.shape[0]
      if 'pad_shape' in img_meta:
        pad_shape = img_meta['pad_shape'][:2]
      else:
        pad_shape = None
      m = cor_hm_pred.shape[0]
      assert cor_hm_pred.shape[1] == 4
      #featmap_size = int(np.sqrt(m))
      feat_size0, feat_size1 = img_meta['feat_sizes'][0]
      cor_hm0 = cor_hm_pred.reshape(feat_size0, feat_size1, cor_hm_pred.shape[1])
      cor_hm1 = torch.nn.functional.interpolate(cor_hm0.permute(2,0,1).unsqueeze(0), size=(pad_shape[0], pad_shape[1]), mode='bilinear')
      cor_hm2 = cor_hm1.squeeze(0).permute(1,2,0)

      line_2p_refine_1 = line_2p_refine_0.reshape(num_lines*2, 2)
      rr_inds = torch.ceil(line_2p_refine_1).to(torch.int64)
      ll_inds = torch.floor(line_2p_refine_1).to(torch.int64)

      rr_inds[:,0] = rr_inds[:,0].clamp(min=0, max=pad_shape[1]-1)
      rr_inds[:,1] = rr_inds[:,1].clamp(min=0, max=pad_shape[0]-1)
      ll_inds[:,0] = ll_inds[:,0].clamp(min=0, max=pad_shape[1]-1)
      ll_inds[:,1] = ll_inds[:,1].clamp(min=0, max=pad_shape[0]-1)

      rr_d = (line_2p_refine_1 - rr_inds.to(torch.float)).norm(dim=1)
      ll_d = (line_2p_refine_1 - ll_inds.to(torch.float)).norm(dim=1)
      sum_d = rr_d + ll_d
      rr_w = 1 - rr_d / sum_d
      ll_w = 1 - ll_d / sum_d

      # note: the index order in image: y first
      rr_cor = cor_hm2[ rr_inds[:,1], rr_inds[:,0] ]
      ll_cor = cor_hm2[ ll_inds[:,1], ll_inds[:,0] ]

      cor_2p_preds = rr_cor * rr_w.unsqueeze(1) + ll_cor * ll_w.unsqueeze(1)
      cor_2p_preds = cor_2p_preds.reshape(num_lines, 2, 4)

      #corner0_score = cor_2p_preds[:,0:1,:2].mean(dim=-1)
      #corner1_score = cor_2p_preds[:,1:2,:2].mean(dim=-1)
      #corner_ave_score = (corner0_score + corner1_score)/2

      corner0_score = cor_2p_preds[:,0,0:1]
      corner1_score = cor_2p_preds[:,1,0:1]
      corner0_center = cor_2p_preds[:,0,1:2]
      corner1_center = cor_2p_preds[:,1,1:2]
      line_preds_out = torch.cat([line_pred, corner0_score, corner1_score, corner0_center, corner1_center], dim=1)
      line_preds_out = cal_composite_score(line_preds_out)
      assert line_preds_out.shape[1] == OUT_DIM_FINAL


      if 0:
        from mmdet.debug_utils import show_lines, show_img_lines, show_heatmap
        #show_img_lines(cor_hm2.cpu().data.numpy()[:,:,0:1], bbox_refine.cpu().data.numpy())
        #show_img_lines(cor_hm2.cpu().data.numpy()[:,:,0:1], line_2p_refine_0.cpu().data.numpy())
        show_heatmap(cor_hm2.cpu().data.numpy()[:,:,0], (512,512), gt_corners = line_2p_refine_0[:,0:2].cpu().data.numpy())
        #show_lines(bbox_refine.cpu().data.numpy(), pad_shape)
        #show_lines(line_2p_refine.cpu().data.numpy(), pad_shape)

      return line_preds_out


def cal_loss_bbox(stage, obj_rep, loss_bbox_fun, bbox_pred_init_nm, bbox_gt_init_nm,
                  bbox_weights_init, num_total_samples_init, transform_method):
        assert stage in ['init', 'refine']
        s = {'init':'I', 'refine':'R'}[stage]

        if obj_rep == 'XYXYSin2' or obj_rep == 'XYXYSin2WZ0Z1':
            loss_pts_init_loc = loss_bbox_fun(
              bbox_pred_init_nm[:,:4],
              bbox_gt_init_nm[:,:4],
              bbox_weights_init[:,:4],
              avg_factor=num_total_samples_init)
            loss_pts_init_rotation = loss_bbox_fun(
              bbox_pred_init_nm[:,4:5],
              bbox_gt_init_nm[:,4:5],
              bbox_weights_init[:,4:5],
              avg_factor=num_total_samples_init)
            loss_pts_init ={
                f'loss_loc{s}': loss_pts_init_loc,
                f'loss_rot{s}': loss_pts_init_rotation
            }
            #if obj_rep == 'XYXYSin2WZ0Z1':
            #  loss_pts_init_width = loss_bbox_fun(
            #    bbox_pred_init_nm [:,[5]],
            #    bbox_gt_init_nm   [:,[5]],
            #    bbox_weights_init [:,[5]],
            #    avg_factor=num_total_samples_init)
            #  loss_pts_init_z = loss_bbox_fun(
            #    bbox_pred_init_nm [:,[6,7]],
            #    bbox_gt_init_nm   [:,[6,7]],
            #    bbox_weights_init [:,[6,7]],
            #    avg_factor=num_total_samples_init)
            #  loss_pts_init[f'loss_wd{s}'] = loss_pts_init_width
            #  #loss_pts_init[f'loss_z{s}'] = loss_pts_init_z
            pass

        elif obj_rep == 'Rect4CornersZ0Z1':
          #t0 = time.time()
          ids = torch.nonzero(bbox_weights_init[:,0]).squeeze().view(-1)
          if ids.numel() > 0:
            bbox_pred_init_nm_aligned = align_pred_gt_bboxes( bbox_pred_init_nm[ids][:,:10], bbox_gt_init_nm[ids], obj_rep )
            bbox_pred_init_nm[ids] = torch.cat([bbox_pred_init_nm_aligned, bbox_pred_init_nm[ids][:,10:]], dim=1)
          #t = time.time() - t0
          #print(f'\n\t align t:{t}')
          loss_pts_init_cor = loss_bbox_fun(
            bbox_pred_init_nm[:, :8],
            bbox_gt_init_nm  [:, :8],
            bbox_weights_init[:, :8],
            avg_factor=num_total_samples_init)
          loss_pts_init_z = loss_bbox_fun(
            bbox_pred_init_nm[:, 8:10],
            bbox_gt_init_nm  [:, 8:10],
            bbox_weights_init[:, 8:10],
            avg_factor=num_total_samples_init)

          loss_pts_init = {
                f'loss_cor{s}': loss_pts_init_cor,
          }
          pass

        elif obj_rep == 'XYLgWsAsinSin2Z0Z1' or obj_rep == 'XYLgWsAbsSin2Z0Z1':
            Xc, Yc, Lg, Ws, Asin, Sin2, Z0, Z1 = range(8)

            loss_pts_init_loc = loss_bbox_fun(
              bbox_pred_init_nm[:,[Xc, Yc]],
              bbox_gt_init_nm[:,[Xc, Yc]],
              bbox_weights_init[:,[Xc, Yc]],
              avg_factor=num_total_samples_init)
            loss_pts_init_lg = loss_bbox_fun(
              bbox_pred_init_nm[:,[Lg]],
              bbox_gt_init_nm[:,  [Lg]],
              bbox_weights_init[:,[Lg]],
              avg_factor=num_total_samples_init)
            loss_pts_init_ws = loss_bbox_fun(
              bbox_pred_init_nm[:,[ Ws]],
              bbox_gt_init_nm[:,  [ Ws]],
              bbox_weights_init[:,[ Ws]],
              avg_factor=num_total_samples_init)
            loss_pts_init_asin = loss_bbox_fun(
              bbox_pred_init_nm[:,[Asin]],
              bbox_gt_init_nm[:,  [Asin]],
              bbox_weights_init[:,[Asin]],
              avg_factor=num_total_samples_init)
            loss_pts_init_sin2 = loss_bbox_fun(
              bbox_pred_init_nm[:,[Sin2]],
              bbox_gt_init_nm[:,  [Sin2]],
              bbox_weights_init[:,[Sin2]],
              avg_factor=num_total_samples_init)

            loss_pts_init = {
              f'loss_loc{s}':  loss_pts_init_loc,
              f'loss_sin2{s}': loss_pts_init_sin2,
              f'loss_asin{s}': loss_pts_init_asin,
              f'loss_lg{s}': loss_pts_init_lg,
            }
            if not DEBUG_CFG.SET_WIDTH_0:
              loss_pts_init[f'loss_ws{s}'] = loss_pts_init_ws

        elif obj_rep == 'XYDAsinAsinSin2Z0Z1':
            Xc, Yc, Dl, AsinCor, Asin, Sin2, Z0, Z1 = range(8)

            loss_pts_init_loc = loss_bbox_fun(
              bbox_pred_init_nm[:,[Xc, Yc]],
              bbox_gt_init_nm[:,[Xc, Yc]],
              bbox_weights_init[:,[Xc, Yc]],
              avg_factor=num_total_samples_init)
            loss_pts_init_diag_len = loss_bbox_fun(
              bbox_pred_init_nm[:,[Dl]],
              bbox_gt_init_nm[:,  [Dl]],
              bbox_weights_init[:,[Dl]],
              avg_factor=num_total_samples_init)
            loss_pts_init_ascor = loss_bbox_fun(
              bbox_pred_init_nm[:,[AsinCor]],
              bbox_gt_init_nm[:,  [AsinCor]],
              bbox_weights_init[:,[AsinCor]],
              avg_factor=num_total_samples_init)

            # This is very important. Without the loss weight, the net will not
            # converge.
            #rotation_weight = bbox_weights_init[:,[Asin]]
            #pdl = bbox_pred_init_nm[:, [Dl]]
            #w = (pdl - 5).clamp(min=0)
            #rotation_weight *= w
            loss_pts_init_asin = loss_bbox_fun(
              bbox_pred_init_nm[:,[Asin]] * bbox_pred_init_nm[:,[Dl]],
              bbox_gt_init_nm[:,  [Asin]] * bbox_gt_init_nm[:,  [Dl]],
              bbox_weights_init[:,[Asin]],
              avg_factor=num_total_samples_init)
            loss_pts_init_sin2 = loss_bbox_fun(
              bbox_pred_init_nm[:,[Sin2]] * bbox_pred_init_nm[:,[Dl]],
              bbox_gt_init_nm[:,  [Sin2]] * bbox_gt_init_nm[:,  [Dl]],
              bbox_weights_init[:,[Sin2]],
              avg_factor=num_total_samples_init)

            loss_pts_init_asin *= 0.5
            loss_pts_init_sin2 *= 0.5

            loss_pts_init = {
              f'loss_loc{s}':  loss_pts_init_loc,
              f'loss_sin2{s}': loss_pts_init_sin2,
              f'loss_asin{s}': loss_pts_init_asin,
              f'loss_dial{s}': loss_pts_init_diag_len,
              f'loss_asc{s}': loss_pts_init_ascor,
            }
        else:
          raise NotImplementedError

        debug = 0
        if debug:
          ids = torch.nonzero(bbox_weights_init[:,0]).squeeze().view(-1)
          bbox_pred_init_nm_ = bbox_pred_init_nm[ids]
          bbox_gt_init_nm_ = bbox_gt_init_nm[ids]
          rotation_weight_ = rotation_weight[ids]
          import pdb; pdb.set_trace()  # XXX BREAKPOINT
          pass
        return loss_pts_init

def cal_composite_score(line_preds):
      assert line_preds.shape[1] == OUT_DIM_FINAL - 1
      bboxes_refine, bboxes_init, points_refine, points_init, score_refine, \
        score_final, score_line_ave, corner0_score, corner1_score, \
        corner0_center, corner1_center, _ = \
        parse_bboxes_out(line_preds, 'before_cal_score_composite')
      corner_score_min = torch.min(corner0_score, corner1_score) * 0.7 + torch.max(corner0_score, corner1_score) * 0.3
      score_composite = score_refine * 0.4 + score_final * 0.2 + corner_score_min * 0.4
      line_preds = torch.cat([line_preds, score_composite], dim=1)
      return line_preds


def convert_list_dict_order(f_ls_dict):
  '''
  in : [{}]
  out: {[]}
  '''
  num_level = len(f_ls_dict)
  f_dict_ls = {}
  for key in f_ls_dict[0].keys():
    f_dict_ls[key] = []
  for i in range(num_level):
    for key in f_ls_dict[i].keys():
      f_dict_ls[key].append( f_ls_dict[i][key] )
  return f_dict_ls


def show_pred(stage, obj_rep, bbox_pred, bbox_gt, bbox_weights, loss_pts,
              loss_linec_init, pts_pred
              ):
  np = bbox_pred.shape[0]
  if obj_rep == 'XYDAsinAsinSin2Z0Z1':
    npts = 5
    pts_pred = pts_pred.reshape(np,-1,2)[:,:npts].reshape(np,-1)

  print(f'\n------------------------------------')
  print(f'stage:{stage}')
  inds = torch.nonzero(bbox_weights.sum(dim=1)).squeeze()
  pos_n = inds.numel()
  print(f'loss_pts: \n{loss_pts}')
  print(f'num points: {np}, pos_n: {pos_n}')
  m = bbox_gt.shape[1]
  bbox_pred = bbox_pred[:,:m]
  bbox_pred_ = bbox_pred[inds].cpu().data.numpy().reshape(-1,m)
  bbox_gt_ = bbox_gt[inds].cpu().data.numpy().reshape(-1,m)
  pts_pred = pts_pred[inds].cpu().data.numpy().reshape(-1,2)

  errs = bbox_pred_ - bbox_gt_
  print(f'obj_rep:{obj_rep}\nerr:\n{errs}')
  print(f'bbox_pred_:\n{bbox_pred_}')
  print(f'bbox_gt_:\n{bbox_gt_}')

  #_show_objs_ls_points_ls( (512,512), [bbox_gt, ], obj_rep = obj_rep)
  _show_objs_ls_points_ls( (800, 800), [bbox_gt_, bbox_pred_], obj_rep = obj_rep,
                          points_ls = [pts_pred], point_colors='blue', point_thickness=2,
                          obj_colors=['red', 'green'], obj_thickness=[2,1])

  pass

def show_nms_out(det_bboxes, det_labels, obj_rep, num_classes):
  dim_parse = DIM_PARSE(obj_rep, num_classes)
  bboxes_refine, bboxes_init, points_refine, points_init, score_refine, score_final,\
    score_line_ave, corner0_score, corner1_score, corner0_center, corner1_center,\
    score_composite =\
      dim_parse.parse_bboxes_out(det_bboxes, 'nms_out')
  scores = score_refine.max(1)[0]
  mask = scores > 0.3
  _show_objs_ls_points_ls_torch( (512,512), [bboxes_refine[mask]], obj_rep, obj_scores_ls=[scores[mask]])
  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  pass

def show_relations(gt_bboxes, gt_relations):
  from tools.visual_utils import _show_objs_ls_points_ls
  gt_bboxes = gt_bboxes.cpu().data.numpy()
  gt_relations = gt_relations.cpu().data.numpy()
  n = gt_bboxes.shape[0]
  for i in range(n):
    cinds = np.where( gt_relations[i] )[0]
    _show_objs_ls_points_ls( (512,512), [gt_bboxes, gt_bboxes[i:i+1], gt_bboxes[cinds]], obj_colors=['white', 'blue', 'red'], obj_thickness=[1,3,1] )
    pass
  pass
