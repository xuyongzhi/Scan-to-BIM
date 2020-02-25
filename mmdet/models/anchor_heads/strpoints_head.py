from __future__ import division

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.core import (PointGenerator, multi_apply, multiclass_nms,
                        point_target)
from mmdet.ops import DeformConv
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule, bias_init_with_prob

from beike_data_utils.geometric_utils import angle_from_vecs_to_vece, sin2theta
from mmdet import debug_tools
from beike_data_utils.line_utils import decode_line_rep_th

from configs.common import OBJ_DIM, OBJ_REP, OUT_EXTAR_DIM

LINE_CONSTRAIN_LOSS = True
DEBUG = False

@HEADS.register_module
class StrPointsHead(nn.Module):
    """StrPoint head.

    Args:
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
                 cls_types=['refine'],
                 dcn_zero_base=False,
                 corcls = False,
                 corloc = False,
                 loss_cor_refine=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 corner_hm = True,
                 corner_hm_only = True,
                 ):
        super(StrPointsHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.point_feat_channels = point_feat_channels
        self.stacked_convs = stacked_convs
        self.num_points = num_points
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
        self.use_grid_points = use_grid_points
        self.center_init = center_init
        self.transform_method = transform_method
        if self.transform_method == 'moment' or \
                self.transform_method == 'moment_lscope_istopleft':
            self.moment_transfer = nn.Parameter(
                data=torch.zeros(2), requires_grad=True)
            self.moment_mul = moment_mul
        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes - 1
        else:
            self.cls_out_channels = self.num_classes
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
        self.corcls = corcls
        self.corloc = corloc
        self.loss_cor_refine = build_loss( loss_cor_refine )
        self.corner_hm = corner_hm
        self.corner_hm_only = corner_hm_only
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

        # layers for corner classificaiton and localization from line final
        if self.corcls or self.corloc:
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
                        norm_cfg=self.norm_cfg))
                self.cor_reg_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            self.corner_cls_conv = DeformConv(self.feat_channels,
                                                self.point_feat_channels,
                                                1, 1, 0)
            self.corner_cls_out = nn.Conv2d(self.point_feat_channels,
                                              self.cls_out_channels, 1, 1, 0)
            self.corner_loc_conv = DeformConv(self.feat_channels,
                                                self.point_feat_channels,
                                                1, 1, 0)
            self.corner_loc_out = nn.Conv2d(self.point_feat_channels,
                                                  2, 1, 1, 0)
        # layers for corner heat map from backbone
        if self.corner_hm:
              self.cor_hm_cls_convs = nn.ModuleList()
              self.cor_hm_ofs_convs = nn.ModuleList()
              for i in range(self.stacked_convs):
                  chn = self.in_channels if i == 0 else self.feat_channels
                  self.cor_hm_cls_convs.append(
                      ConvModule(
                          chn,
                          self.feat_channels,
                          3,
                          stride=1,
                          padding=1,
                          conv_cfg=self.conv_cfg,
                          norm_cfg=self.norm_cfg))
                  self.cor_hm_ofs_convs.append(
                      ConvModule(
                          chn,
                          self.feat_channels,
                          3,
                          stride=1,
                          padding=1,
                          conv_cfg=self.conv_cfg,
                          norm_cfg=self.norm_cfg))


              self.reppoints_pts_cor_conv = nn.Conv2d(self.feat_channels,
                                                      self.point_feat_channels, 3,
                                                      1, 1)
              self.reppoints_pts_cor_out = nn.Conv2d(self.point_feat_channels,
                                                      pts_out_dim, 1, 1, 0)
              self.cor_hm_cls_dconv = DeformConv(self.feat_channels,
                                                  self.point_feat_channels,
                                                  self.dcn_kernel, 1, self.dcn_pad)
              self.cor_hm_cls_out = nn.Conv2d(self.point_feat_channels,
                                                self.cls_out_channels, 1, 1, 0)
              self.cor_hm_ofs_dconv = DeformConv(self.feat_channels,
                                                  self.point_feat_channels,
                                                  self.dcn_kernel, 1, self.dcn_pad)
              self.cor_hm_ofs_out = nn.Conv2d(self.point_feat_channels,
                                                2, 1, 1, 0)



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

    def points2bbox(self, pts, y_first=True, out_line_constrain=False):
        """
        Converting the points set into bounding box.
        :param pts: the input points sets (fields), each points
            set (fields) is represented as 2n scalar.
        :param y_first: if y_fisrt=True, the point set is represented as
            [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
            represented as [x1, y1, x2, y2 ... xn, yn].
        :return: each points set is converting to a bbox [x1, y1, x2, y2].
        """
        assert pts.shape[1] == self.num_points * 2
        pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
        pts_y = pts_reshape[:, :, 0, ...] if y_first else pts_reshape[:, :, 1,
                                                                      ...]
        pts_x = pts_reshape[:, :, 1, ...] if y_first else pts_reshape[:, :, 0,
                                                                      ...]
        if self.transform_method == 'minmax':
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_up = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom],
                             dim=1)
        elif self.transform_method == 'partial_minmax':
            pts_y = pts_y[:, :4, ...]
            pts_x = pts_x[:, :4, ...]
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_up = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom],
                             dim=1)
        elif self.transform_method == 'moment':
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
        elif self.transform_method == 'moment_lscope_istopleft':
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
            istoplefts_0 = sin2theta(vec_start, vec_pts)

            istoplefts_1 = istoplefts_0.view(pts_x.shape)
            istopleft = istoplefts_1.mean(dim=1, keepdim=True)

            #isaline = istoplefts_1.max(dim=1, keepdim=True)[0] - \
            #          istoplefts_1.min(dim=1, keepdim=True)[0]

            isaline = istoplefts_1.std(dim=1, keepdim=True)

            bbox = torch.cat([
                pts_x_mean - half_width, pts_y_mean - half_height,
                pts_x_mean + half_width, pts_y_mean + half_height,
                istopleft
            ],
                             dim=1)
            if out_line_constrain:
              bbox = torch.cat([bbox, isaline], dim=1)
            pass
        elif self.transform_method == 'center_size_istopleft':
            raise NotImplementedError
        else:
            raise NotImplementedError
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

    def forward_single(self, x):
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
        pts_out_init = self.reppoints_pts_init_out(
            self.relu(self.reppoints_pts_init_conv(pts_feat)))
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
        if 'refine' in self.cls_types:
          cls_out['refine'] = self.reppoints_cls_out(
              self.relu(self.reppoints_cls_conv(cls_feat, dcn_offset)))

        pts_out_refine = self.reppoints_pts_refine_out(
            self.relu(self.reppoints_pts_refine_conv(pts_feat, dcn_offset)))
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
          cls_out['final'] = self.reppoints_cls_out(
              self.relu(self.reppoints_cls_conv(cls_feat, dcn_offset_refine)))

        is_process_cor = self.corcls or self.corloc
        if is_process_cor:
          corner_outs = self.get_corner_cls_reg_from_final(pts_out_refine, x)
        elif self.corner_hm:
          corner_outs = self.get_corner_hm(x)
        else:
          corner_outs = None
        # predict cls from the two end_points

        #debug_tools.show_shapes(x, 'StrPointsHead input')
        #debug_tools.show_shapes(cls_feat, 'StrPointsHead cls_feat')
        #debug_tools.show_shapes(pts_feat, 'StrPointsHead pts_feat')
        #debug_tools.show_shapes(pts_out_init, 'StrPointsHead pts_out_init')
        #debug_tools.show_shapes(cls_out, 'StrPointsHead cls_out')
        #debug_tools.show_shapes(pts_out_refine, 'StrPointsHead pts_out_refine')
        return cls_out, pts_out_init, pts_out_refine, corner_outs

    def get_corner_hm(self, x):
        if x.shape[2] != 128:
          # only use the largest level
          return {}
        assert self.center_init
        assert not self.use_grid_points
        cor_cls_feat = x
        cor_loc_feat = x
        for cls_conv in self.cor_hm_cls_convs:
            cor_cls_feat = cls_conv(cor_cls_feat)
        for reg_conv in self.cor_hm_ofs_convs:
            cor_loc_feat = reg_conv(cor_loc_feat)
        # get corner reppoints
        pts_out_cor = self.reppoints_pts_cor_out(
            self.relu(self.reppoints_pts_cor_conv(cor_loc_feat)))
        points_init = 0
        pts_out_cor = pts_out_cor + points_init
        # get corner heat map
        pts_out_cor_grad_mul = (1 - self.gradient_mul) * pts_out_cor.detach(
        ) + self.gradient_mul * pts_out_cor
        if self.dcn_zero_base:
          dcn_offset = pts_out_cor_grad_mul
        else:
          dcn_base_offset = self.dcn_base_offset.type_as(x)
          dcn_offset = pts_out_cor_grad_mul - dcn_base_offset
        cor_hm_cls = self.cor_hm_cls_out(
              self.relu(self.cor_hm_cls_dconv(cor_cls_feat, dcn_offset)))
        cor_hm_ofs = self.cor_hm_ofs_out(
            self.relu(self.cor_hm_ofs_dconv(cor_loc_feat, dcn_offset)))
        corner_outs = {'cor_hm_cls': cor_hm_cls, 'cor_hm_ofs':cor_hm_ofs}
        return corner_outs

    def get_corner_cls_reg_from_final(self, pts_out_refine, x):
        '''
          pts_out_refine: y_first
          deformable conv offset is also y_first
        '''
        assert OBJ_REP == 'lscope_istopleft'
        cor_cls_feat = x
        cor_loc_feat = x
        for cls_conv in self.cor_cls_convs:
            cor_cls_feat = cls_conv(cor_cls_feat)
        for reg_conv in self.cor_reg_convs:
            cor_loc_feat = reg_conv(cor_loc_feat)

        # pts_out_refine is y_first
        bbox_out_refine_xfirst = self.points2bbox(pts_out_refine)
        # bbox_out_refine and lines_out_refine0 is x_first,
        lines_out_refine_xfirst = decode_line_rep_th(bbox_out_refine_xfirst, OBJ_REP)
        # lines_out_refine is y_first
        lines_out_refine = lines_out_refine_xfirst[:,[1,0,3,2],:,:]

        lines_out_refine_grad_mul = (1 - self.gradient_mul) * lines_out_refine.detach(
        ) + self.gradient_mul * lines_out_refine
        corner0_dcn_offset = lines_out_refine_grad_mul[:,0:2,:,:] # y_first
        corner1_dcn_offset = lines_out_refine_grad_mul[:,2:4,:,:]

        cor0_cls_out = self.corner_cls_out(
            self.relu(self.corner_cls_conv(cor_cls_feat, corner0_dcn_offset)))
        cor1_cls_out = self.corner_cls_out(
            self.relu(self.corner_cls_conv(cor_cls_feat, corner1_dcn_offset)))

        cor0_refine = self.corner_loc_out(
            self.relu(self.corner_loc_conv(cor_loc_feat, corner0_dcn_offset)))
        cor1_refine = self.corner_loc_out(
            self.relu(self.corner_loc_conv(cor_loc_feat, corner1_dcn_offset)))
        corner_outs = dict(
                        corners_from_lines = lines_out_refine_xfirst,
                        cor0_cls_out = cor0_cls_out,
                        cor1_cls_out = cor1_cls_out,
                        cor0_refine = cor0_refine,
                        cor1_refine = cor1_refine )
        return corner_outs

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

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
                h, w, _ = img_meta['pad_shape']
                valid_feat_h = min(int(np.ceil(h / point_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / point_stride)), feat_w)
                flags = self.point_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w))
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)

        return points_list, valid_flag_list

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
          np = 2
        else:
          np = self.num_points
        pts_list = []
        for i_lvl in range(len(self.point_strides)):
            pts_lvl = []
            for i_img in range(len(center_list)):
                pts_center = center_list[i_img][i_lvl][:, :2].repeat(
                    1, np)
                pts_shift = pred_list[i_lvl][i_img]
                yx_pts_shift = pts_shift.permute(1, 2, 0).view(
                    -1, np*2)
                y_pts_shift = yx_pts_shift[..., 0::2]
                x_pts_shift = yx_pts_shift[..., 1::2]
                xy_pts_shift = torch.stack([x_pts_shift, y_pts_shift], -1)
                xy_pts_shift = xy_pts_shift.view(*yx_pts_shift.shape[:-1], -1)
                pts = xy_pts_shift * self.point_strides[i_lvl] + pts_center
                pts_lvl.append(pts)
            pts_lvl = torch.stack(pts_lvl, 0)
            pts_list.append(pts_lvl)
        return pts_list


    def loss_single(self, cls_score, pts_pred_init, pts_pred_refine, labels,
                    label_weights, bbox_gt_init, bbox_weights_init,
                    bbox_gt_refine, bbox_weights_refine, stride,
                    num_total_samples_init, num_total_samples_refine,
                    num_total_samples_cls):
        obj_dim = bbox_gt_init.shape[-1]
        # classification loss
        loss_cls = {}
        for cls_type in cls_score:
          labels_i = labels[cls_type].reshape(-1)
          label_weights_i = label_weights[cls_type].reshape(-1)
          cls_score_i = cls_score[cls_type].permute(0, 2, 3,
                                        1).reshape(-1, self.cls_out_channels)
          loss_cls[cls_type] = self.loss_cls(
              cls_score_i,
              labels_i,
              label_weights_i,
              avg_factor=num_total_samples_cls[cls_type])

        # points loss
        bbox_gt_init = bbox_gt_init.reshape(-1, obj_dim)
        bbox_weights_init = bbox_weights_init.reshape(-1, obj_dim)
        bbox_pred_init = self.points2bbox(
            pts_pred_init.reshape(-1, 2 * self.num_points), y_first=False,
            out_line_constrain=LINE_CONSTRAIN_LOSS)
        bbox_gt_refine = bbox_gt_refine.reshape(-1, obj_dim)
        bbox_weights_refine = bbox_weights_refine.reshape(-1, obj_dim)
        bbox_pred_refine = self.points2bbox(
            pts_pred_refine.reshape(-1, 2 * self.num_points), y_first=False,
            out_line_constrain=LINE_CONSTRAIN_LOSS)
        normalize_term = self.point_base_scale * stride

        if DEBUG:
          print(f'num_total_samples_init:  {num_total_samples_init}\nnum_total_samples_refine: {num_total_samples_refine}')
          print(f'stride: {stride}')
          show_pred(bbox_pred_init, bbox_gt_init, bbox_weights_init, f'S{stride}_init')
          show_pred(bbox_pred_refine, bbox_gt_refine, bbox_weights_refine, f'S{stride}_refine')

        if obj_dim == 4:
          bbox_pred_init_nm = bbox_pred_init / normalize_term
          bbox_gt_init_nm = bbox_gt_init / normalize_term
        elif obj_dim == 5:
          bbox_pred_init_nm = bbox_pred_init / normalize_term
          bbox_gt_init_nm = bbox_gt_init / normalize_term
          bbox_pred_init_nm[:,4] = bbox_pred_init[:,4]
          bbox_gt_init_nm[:,4] = bbox_gt_init[:,4]


        if LINE_CONSTRAIN_LOSS:
          tmp = torch.zeros_like(bbox_gt_init_nm)[:,0:1]
          bbox_gt_init_nm = torch.cat([bbox_gt_init_nm, tmp], dim=1)
          bbox_weights_init = torch.cat([bbox_weights_init, bbox_weights_init[:,0:1]], dim=1)

        loss_pts_init = self.loss_bbox_init(
            bbox_pred_init_nm,
            bbox_gt_init_nm,
            bbox_weights_init,
            avg_factor=num_total_samples_init)

        if obj_dim == 4:
          bbox_pred_refine_nm = bbox_pred_refine / normalize_term
          bbox_gt_refine_nm = bbox_gt_refine / normalize_term
        elif obj_dim == 5:
          bbox_pred_refine_nm = bbox_pred_refine / normalize_term
          bbox_gt_refine_nm = bbox_gt_refine / normalize_term
          bbox_pred_refine_nm[:,4] = bbox_pred_refine[:,4]
          bbox_gt_refine_nm[:,4] = bbox_gt_refine[:,4]

        if LINE_CONSTRAIN_LOSS:
          tmp = torch.zeros_like(bbox_gt_init_nm)[:,0:1]
          bbox_gt_refine_nm = torch.cat([bbox_gt_refine_nm, tmp], dim=1)
          bbox_weights_refine = torch.cat([bbox_weights_refine, bbox_weights_refine[:,0:1]], dim=1)

        loss_pts_refine = self.loss_bbox_refine(
            bbox_pred_refine_nm,
            bbox_gt_refine_nm,
            bbox_weights_refine,
            avg_factor=num_total_samples_refine)
        return loss_cls, loss_pts_init, loss_pts_refine

    def get_bbox_from_pts(self, center_list, pts_preds):
        bbox_list = []
        for i_img, center in enumerate(center_list):
            bbox = []
            for i_lvl in range(len(pts_preds)):
                bbox_preds_init = self.points2bbox(
                    pts_preds[i_lvl].detach())
                if self.transform_method == 'center_size_istopleft':
                  raise NotImplementedError
                elif self.transform_method == 'moment':
                  assert bbox_preds_init.shape[1] == 4
                  bbox_shift = bbox_preds_init * self.point_strides[i_lvl]
                  bbox_center = torch.cat(
                      [center[i_lvl][:, :2], center[i_lvl][:, :2]], dim=1)
                  bbox.append(bbox_center +
                              bbox_shift[i_img].permute(1, 2, 0).reshape(-1, 4))
                elif self.transform_method == 'moment_lscope_istopleft':
                  assert bbox_preds_init.shape[1] == 5
                  bbox_shift = bbox_preds_init[:,:4] * self.point_strides[i_lvl]
                  istopleft = bbox_preds_init[i_img,4:5].\
                                 permute(1,2,0).reshape(-1,1)
                  bbox_center = torch.cat(
                      [center[i_lvl][:, :2], center[i_lvl][:, :2]], dim=1)
                  bbox_i = bbox_center +\
                              bbox_shift[i_img].permute(1, 2, 0).reshape(-1, 4)
                  bbox_i = torch.cat([bbox_i, istopleft], dim=1)
                  bbox.append(bbox_i)
                else:
                  raise NotImplemented
            bbox_list.append(bbox)
        return bbox_list

    def loss_single_corner(self, stride, cor0_scores, cor1_scores, cor_labels, cor_label_weights, corners_gt, cor_preds, cor_refine_weights,  num_total_samples_cor):
      normalize_term = self.point_base_scale * stride
      cor0_scores = cor0_scores.permute(0, 2, 3,
                                        1).reshape(-1, self.cls_out_channels)
      cor1_scores = cor1_scores.permute(0, 2, 3,
                                        1).reshape(-1, self.cls_out_channels)
      cor0_loss_cls = self.loss_cls(
              cor0_scores,
              cor_labels.reshape(-1),
              cor_label_weights.reshape(-1),
              avg_factor=num_total_samples_cor)

      cor1_loss_cls = self.loss_cls(
              cor1_scores,
              cor_labels.reshape(-1),
              cor_label_weights.reshape(-1),
              avg_factor=num_total_samples_cor)
      loss_cor_cls = cor0_loss_cls + cor1_loss_cls

      corners_gt_nm = corners_gt / normalize_term
      cor_preds_nm = cor_preds / normalize_term
      cor_refine_weights = cor_refine_weights[:,:,0:4].reshape(-1,4)

      n = cor_preds_nm.shape[0]
      dif0 = (cor_preds_nm - corners_gt_nm).norm(dim=-1, keepdim=True)
      dif1 = (cor_preds_nm[:,[2,3, 0,1]] - corners_gt_nm).norm(dim=-1, keepdim=True)
      is_switch = (dif0 > dif1).to(corners_gt_nm.dtype)
      corners_gt_nm_ordered = corners_gt_nm * (1-is_switch) + corners_gt_nm[:,[2,3, 0,1]] * is_switch

      loss_cor_reg = self.loss_cor_refine(
            cor_preds_nm,
            corners_gt_nm_ordered,
            cor_refine_weights,
            avg_factor=num_total_samples_cor)
      return loss_cor_cls, loss_cor_reg

    def corner_loss(self, corner_outs, center_list, cor_labels_list, cor_label_weights_list, num_total_samples_finale, bbox_gt_list_final, bbox_weights_list_final):
      '''
        The target of predicted corners are directly achived from line final
        targets, instead of comparing pred_corners and gt_corners. We do not
        encourage a corner to be close to any one, but encourage a corner to
        be close to the one of coresponding gt line.
      '''
      corner_outs = convert_list_dict_order(corner_outs)
      cor0_scores = corner_outs['cor0_cls_out']
      cor1_scores = corner_outs['cor1_cls_out']

      corners_gt = [decode_line_rep_th(bg.reshape(-1,bg.shape[-1]), OBJ_REP) \
                    for bg in bbox_gt_list_final]

      corners_from_lines = corner_outs['corners_from_lines']
      cor_coordinates0 = self.offset_to_pts(center_list,
                                          corners_from_lines, is_corner=True)
      cor_preds = [cc.reshape(-1,4) for cc in cor_coordinates0]


      losses_cor_cls, losses_cor_reg = multi_apply(
                self.loss_single_corner,
                self.point_strides,
                cor0_scores,
                cor1_scores,
                cor_labels_list,
                cor_label_weights_list,
                corners_gt,
                cor_preds,
                bbox_weights_list_final,
                num_total_samples_cor = num_total_samples_finale)
      return losses_cor_cls, losses_cor_reg

    def loss_single_corner_hm(self, stride, cor_hm_scores, labels, label_weights,
              cor_hm_ofs, cor_gt_ofs, cor_reg_weights, num_total_samples_cor):
        cor_hm_scores = cor_hm_scores.permute(0,2,3,1).reshape(-1, self.cls_out_channels)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        loss_cor_hm_cls = self.loss_cls(
              cor_hm_scores,
              labels,
              label_weights,
              avg_factor=num_total_samples_cor)

        normalize_term = self.point_base_scale * stride
        cor_hm_ofs_nm = cor_hm_ofs.permute(0,2,3,1).reshape(-1,2) / normalize_term
        cor_gt_ofs_nm = cor_gt_ofs.reshape(-1,2) / normalize_term
        cor_reg_weights = cor_reg_weights.reshape(-1,2)
        loss_cor_hm_reg = self.loss_cor_refine(
              cor_hm_ofs_nm,
              cor_gt_ofs_nm,
              cor_reg_weights,
              avg_factor=num_total_samples_cor)
        return loss_cor_hm_cls, loss_cor_hm_reg

    def corner_heat_map_loss(self, corner_outs, gt_bboxes, gt_labels, img_metas, cfg, gt_bboxes_ignore):
        gt_corners = [self.get_corners_from_bboxes(gbs) for gbs in gt_bboxes]
        if gt_bboxes_ignore is None:
          gt_corners_ignore = None
        else:
          gt_corners_ignore = [self.get_corners_from_bboxes(gbs) for gbs in gt_bboxes_ignore]
        corner_outs = corner_outs[0] # use the largest level only
        cor_hm_cls = corner_outs['cor_hm_cls']
        cor_hm_ofs = corner_outs['cor_hm_ofs']
        featmap_size = cor_hm_cls.size()[-2:]
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        #-----------------------------------------------------------------------
        # target for corner heat map
        center_list, valid_flag_list = self.get_points([featmap_size], img_metas) # []*batch_size
        candidate_list = center_list

        cls_reg_targets_corner = point_target(
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
        (labels_list, label_weights_list, cor_gt_list, candidate_list_neg0,
         reg_weights_list, num_total_pos_cor, num_total_neg_cor) = cls_reg_targets_corner
        num_total_samples_cor = (
            num_total_pos_cor +
            num_total_neg_cor if self.sampling else num_total_pos_cor)

        cor_gt_ofs_list = [can[:,:2]-gt for can, gt in zip(candidate_list_neg0, cor_gt_list)]
        torch.nonzero( reg_weights_list[0] )

        losses_corhm_cls, losses_corhm_reg = multi_apply(
          self.loss_single_corner_hm,
          self.point_strides[0:1],
          [cor_hm_cls],
          labels_list,
          label_weights_list,
          [cor_hm_ofs],
          cor_gt_ofs_list,
          reg_weights_list,
          num_total_samples_cor = num_total_samples_cor
        )
        losses_cor_hm = {'loss_corhm_cls':losses_corhm_cls,
                         'loss_corhm_reg':losses_corhm_reg}
        #if DEBUG:
        #  pos_ids = torch.nonzero( reg_weights_list[0][0,:,0] ).squeeze()
        #  gt_ofs_pos = cor_gt_ofs_list[0][0, pos_ids]
        return losses_cor_hm

    def loss(self,
             cls_scores,
             pts_preds_init,
             pts_preds_refine,
             corner_outs,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):

        if self.corner_hm:
          loss_corner_hm = self.corner_heat_map_loss(corner_outs, gt_bboxes,
                                 gt_labels, img_metas,cfg, gt_bboxes_ignore)
          if self.corner_hm_only:
            return loss_corner_hm

        featmap_sizes = [featmap.size()[-2:] for featmap in pts_preds_init]
        assert len(featmap_sizes) == len(self.point_generators)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        #-----------------------------------------------------------------------
        # target for initial stage
        center_list, valid_flag_list = self.get_points(featmap_sizes,
                                                       img_metas)
        pts_coordinate_preds_init = self.offset_to_pts(center_list,
                                                       pts_preds_init)
        if cfg.init.assigner['type'] == 'PointAssigner':
            # Assign target for center list
            candidate_list = center_list
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
        (*_, bbox_gt_list_init, candidate_list_init, bbox_weights_list_init,
         num_total_pos_init, num_total_neg_init) = cls_reg_targets_init
        num_total_samples_init = (
            num_total_pos_init +
            num_total_neg_init if self.sampling else num_total_pos_init)

        #-----------------------------------------------------------------------
        # target for refinement stage
        center_list, valid_flag_list = self.get_points(featmap_sizes,
                                                       img_metas)
        pts_coordinate_preds_refine = self.offset_to_pts(
            center_list, pts_preds_refine)
        bbox_list_initres = self.get_bbox_from_pts(center_list, pts_preds_init)
        #debug_tools.show_shapes(pts_preds_init, 'StrPointsHead init')
        #debug_tools.show_shapes(bbox_list_initres, 'StrPointsHead init bbox')
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
         num_total_neg_refine) = cls_reg_targets_refine
        num_total_samples_refine = (
            num_total_pos_refine +
            num_total_neg_refine if self.sampling else num_total_pos_refine)

        #-----------------------------------------------------------------------
        is_process_cor = self.corcls or self.corloc
        if 'final' in self.cls_types or is_process_cor:
          center_list, valid_flag_list = self.get_points(featmap_sizes,
                                                       img_metas)
          bbox_list_refineres = self.get_bbox_from_pts(center_list, pts_preds_refine)
          #debug_tools.show_shapes(pts_preds_refine, 'StrPointsHead refine')
          #debug_tools.show_shapes(bbox_list_refineres, 'StrPointsHead refine bbox')
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
          num_total_neg_final) = cls_reg_targets_final
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
        # compute loss
        losses_cls, losses_pts_init, losses_pts_refine = multi_apply(
            self.loss_single,
            cls_scores,
            pts_coordinate_preds_init,
            pts_coordinate_preds_refine,
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
        loss_dict_all = {
            'loss_pts_init': losses_pts_init,
            'loss_pts_refine': losses_pts_refine
        }
        for c in self.cls_types:
          loss_dict_all['loss_cls_'+c] = []
          for lc in losses_cls:
            loss_dict_all['loss_cls_'+c].append( lc[c] )

        #-----------------------------------------------------------------------
        # target for corner
        if self.corner_hm:
          loss_dict_all.update(loss_corner_hm)
        if is_process_cor:
            assert not self.corner_hm
            center_list, valid_flag_list = self.get_points(featmap_sizes,
                                                          img_metas)
            losses_cor_cls, losses_cor_reg = self.corner_loss(corner_outs, center_list, labels_list_final, label_weights_list_final, num_total_samples_finale, bbox_gt_list_final, bbox_weights_list_final)
            loss_dict_all['loss_cor_cls'] = losses_cor_cls
            loss_dict_all['loss_cor_reg'] = losses_cor_reg

        return loss_dict_all


    def cal_test_score(self, cls_scores):
      ave_cls_scores = []
      for i in range( len(cls_scores) ):
        tmp = list(cls_scores[i].values())
        ave_cls_scores.append( sum(tmp) / len(tmp) )
      return ave_cls_scores

    def get_bboxes(self,
                   cls_scores,
                   pts_preds_init,
                   pts_preds_refine,
                   corner_outs,
                   img_metas,
                   cfg,
                   rescale=False,
                   nms=True):
        assert len(cls_scores) == len(pts_preds_refine)
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        cls_scores = self.cal_test_score(cls_scores)
        bbox_preds_refine = [
            self.points2bbox(pts_pred_refine)
            for pts_pred_refine in pts_preds_refine
        ]
        num_levels = len(cls_scores)

        if OUT_EXTAR_DIM > 0:
          for i in range(num_levels):
            bbox_preds_init = [
                self.points2bbox(pts_pred_init)
                for pts_pred_init in pts_preds_init
            ]
            init_refine = torch.cat([bbox_preds_refine[i], bbox_preds_init[i], pts_preds_refine[i], pts_preds_init[i]], dim=1)
            assert bbox_preds_refine[i].shape[1] == OBJ_DIM
            assert init_refine.shape[1] == OBJ_DIM + OUT_EXTAR_DIM
            bbox_preds_refine[i] = init_refine

        mlvl_points = [
            self.point_generators[i].grid_points(cls_scores[i].size()[-2:],
                                                 self.point_strides[i])
            for i in range(num_levels)
        ]
        result_list = []
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
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                               mlvl_points, img_shape,
                                               scale_factor, cfg, rescale, nms)
            result_list.append(proposals)
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
        assert obj_dim == OBJ_DIM + OUT_EXTAR_DIM
        mlvl_bboxes = []
        mlvl_scores = []
        for i_lvl, (cls_score, bbox_pred, points) in enumerate(
                zip(cls_scores, bbox_preds, mlvl_points)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
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
            bbox_pos_center = points[:, :2].repeat(1, OBJ_DIM//2)
            bboxes = bbox_pred[:,:4] * self.point_strides[i_lvl]+ bbox_pos_center
            x1 = bboxes[:, 0].clamp(min=0, max=img_shape[1])
            y1 = bboxes[:, 1].clamp(min=0, max=img_shape[0])
            x2 = bboxes[:, 2].clamp(min=0, max=img_shape[1])
            y2 = bboxes[:, 3].clamp(min=0, max=img_shape[0])
            bboxes = torch.stack([x1, y1, x2, y2], dim=-1)

            if OBJ_DIM == 5:
              bboxes = torch.cat([bboxes, bbox_pred[:,4:5]], dim=1)

            if OUT_EXTAR_DIM > 0:
              # bbox_preds: [bbox_refine, bbox_init, points_refine, points_init]
              bboxes_init = bbox_pred[:, OBJ_DIM:OBJ_DIM*2]
              bboxes_init[:,:4] = bboxes_init[:,:4] * self.point_strides[i_lvl] + bbox_pos_center

              POINTS_DIM = OUT_EXTAR_DIM - OBJ_DIM
              assert POINTS_DIM % 2 == 0
              bbox_pos_center = points[:, :2].repeat(1, POINTS_DIM//2)
              bn = bboxes.shape[0]
              # key_points store in y-first, but box in x-first.
              # change key-points to x-first
              key_points = bbox_pred[:,-POINTS_DIM:].\
                            reshape(bn,-1,2)[:,:,[1,0]].reshape(bn,-1)
              key_points = key_points * self.point_strides[i_lvl] + bbox_pos_center
              for kp in range(0, key_points.shape[1], 2):
                key_points[:,kp] = key_points[:,kp].clamp(min=0, max=img_shape[1])
                key_points[:,kp+1] = key_points[:,kp+1].clamp(min=0, max=img_shape[0])
              bboxes = torch.cat([bboxes, bboxes_init, key_points], dim=1)
              pass

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        if nms:
            det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores

    def normalize_istopleft_loss(self, itl_loss):
        itl_loss_nm = 1 - torch.exp(-itl_loss)
        return itl_loss_nm

    def get_corners_from_bboxes(self, bboxes):
      lines0 = decode_line_rep_th(bboxes, OBJ_REP)
      lines1 = lines0.reshape(-1,2)
      lines_uq = torch.unique(lines1, sorted=False, dim=0)
      return lines_uq

      if 0:
        n0 = lines1.shape[0]
        n1 = lines_uq.shape[0]
        print(f'{n0} -> {n1}')
        from mmdet.debug_tools import show_lines
        show_lines(lines0.cpu().data.numpy(), (512,512), points=lines_uq)
      pass

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


def show_pred(bbox_pred, bbox_gt, bbox_weights, flag):
  from mmdet.debug_tools import show_lines
  from configs.common import IMAGE_SIZE
  inds = torch.nonzero(bbox_weights.sum(dim=1)).squeeze()
  m = bbox_pred.shape[1]
  bbox_pred = bbox_pred[inds].cpu().data.numpy().reshape(-1,m)[:,:5]
  m = bbox_gt.shape[1]
  bbox_gt = bbox_gt[inds].cpu().data.numpy().reshape(-1,m)

  show_lines(bbox_gt, (IMAGE_SIZE, IMAGE_SIZE), lines_ref=bbox_pred, name=flag+'.png')
  pass
