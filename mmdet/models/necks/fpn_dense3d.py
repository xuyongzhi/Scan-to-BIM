import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import auto_fp16
from ..registry import NECKS
from ..utils import ConvModule
import math

from configs.common import SPARSE_BEV
from tools.debug_utils import _show_tensor_ls_shapes
SHOW_NET = 0

@NECKS.register_module
class FPN_Dense3D(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=dict(type='Conv3d'),
                 norm_cfg=None,
                 max_z_dim_start = None,
                 activation=None,
                 activation_bev_proj='relu'):
        super(FPN_Dense3D, self).__init__()
        assert isinstance(in_channels, list)
        conv_cfg_2d = dict(type='Conv')
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.activation_bev_proj = activation_bev_proj
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False


        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()


        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                activation=self.activation,
                inplace=False)

            fpn_conv_linear = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg_2d,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv_linear)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        self.extra_levels = extra_levels
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if SPARSE_BEV:
                  if i == 0 and self.extra_convs_on_inputs:
                      in_channels = self.in_channels[self.backbone_end_level - 1]
                  else:
                      in_channels = out_channels
                else:
                      in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg_2d,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

        stride_se = pow(2, self.backbone_end_level - self.start_level-1)
        self.zdim_end_level = int(math.ceil( max_z_dim_start / stride_se))

        if not SPARSE_BEV:
          self.build_project_layers()

    def build_project_layers(self,):
        self.project_layers = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
          self.project_layers.append( self.build_project_layer(i, self.out_channels) )

        for i in range(self.extra_levels):
          if i == 0 and self.extra_convs_on_inputs:
              in_channels = self.in_channels[self.backbone_end_level - 1]
          else:
              in_channels = out_channels
          self.project_layers.append( self.build_project_layer(self.backbone_end_level-1, in_channels) )
        #print(self.project_layers)
        pass

    def build_project_layer(self, level, in_channels):
      num_layer_to_end = self.backbone_end_level - level -1
      kernels_i =   [[3,3,3]] * num_layer_to_end
      strides_i =   [[1,1,2]] * num_layer_to_end
      kernels_i += [[1,1,self.zdim_end_level],]
      strides_i += [[1,1,self.zdim_end_level],]

      paddings_i = [[int(k[0]>1), int(k[1]>1), 1] for k in kernels_i]
      num_bev_project_layers = len(kernels_i)
      fpn_layer_i = []
      for j in range(num_bev_project_layers):
          fpn_conv_j = ConvModule(
              in_channels,
              self.out_channels,
              kernels_i[j],
              strides_i[j],
              padding=paddings_i[j],
              conv_cfg=self.conv_cfg,
              norm_cfg=self.norm_cfg,
              activation=self.activation_bev_proj,
              inplace=False)
          fpn_layer_i.append(fpn_conv_j)
          in_channels = self.out_channels

      part1 = nn.Sequential(*fpn_layer_i[:-1])
      part2 = fpn_layer_i[-1]
      return nn.ModuleList( [part1, part2] )

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        if SHOW_NET and 1:
          print('\n\n')
          _show_tensor_ls_shapes(inputs, 'inputs')

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            xd, yd, zd = laterals[i-1].shape[2:]
            upper_i = F.interpolate(laterals[i], scale_factor=2, mode='nearest')
            upper_i = upper_i[..., :xd, :yd, :zd]
            #_show_tensor_ls_shapes([laterals[i-1], upper_i], f'level {i}')
            laterals[i - 1] += upper_i


        # project to 2d
        bev_laterals = [self.forward_project(laterals[i], i) for i in range(used_backbone_levels)]

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](bev_laterals[i]) for i in range(used_backbone_levels)
        ]

        if SHOW_NET and 0:
          print('\n\n')
          _show_tensor_ls_shapes(laterals, 'laterals')
          _show_tensor_ls_shapes(bev_laterals, 'bev_laterals')
          _show_tensor_ls_shapes(outs, 'outs')

        assert inputs[0].shape[2:-1] == outs[0].shape[2:]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    bev_orig = self.forward_project(orig, used_backbone_levels)
                    outs.append(self.fpn_convs[used_backbone_levels](bev_orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        for i in range(len(outs)):
          oshape = outs[i].shape
          pad_sx = max(0, 3 - oshape[-2])
          pad_sy = max(0, 3 - oshape[-1])
          #print(f'pad: {pad_sx} {pad_sy}')
          if pad_sx > 0 or pad_sy > 0:
            outs[i] = F.pad(outs[i], (0, pad_sy, 0, pad_sx), "constant", 0)

        if SHOW_NET and 1:
          print('\n\n')
          _show_tensor_ls_shapes(outs, 'outs')
        return tuple(outs)


    @auto_fp16()
    def forward_project(self, laterals_i, i):
        if SPARSE_BEV:
          return laterals_i[...,0]

        bev_laterals_i = self.project_layers[i][0](laterals_i)
        bev_laterals_i = F.pad(bev_laterals_i, (0, self.zdim_end_level - bev_laterals_i.shape[-1]), "constant", 0)
        bev_laterals_i = self.project_layers[i][1](bev_laterals_i)
        bev_laterals_i = bev_laterals_i.max(4)[0]
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        return bev_laterals_i

