import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import auto_fp16
from ..registry import NECKS
from ..utils import ConvModule
import math
from ..utils import build_conv_layer, build_norm_layer

from configs.common import DEBUG_CFG
SPARSE_BEV = DEBUG_CFG.SPARSE_BEV
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
                 z_out_dims = None,
                 z_stride = 2,
                 activation=None,
                 activation_bev_proj='relu'):
        super(FPN_Dense3D, self).__init__()
        assert isinstance(in_channels, list)
        assert len(z_out_dims) == len(in_channels)
        self.proj_method = ['res_block', '2conv'][1]
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
        self.z_stride = z_stride

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
        self.z_out_dims = z_out_dims
        self.proj_z_strides = []
        for zdim in z_out_dims:
          s0 = 3 if zdim > 1 else 1
          s1 = 3 if zdim > 3 else 1
          #s1 = int(math.ceil(zdim/3))
          self.proj_z_strides.append( (s0,s1) )

        if not SPARSE_BEV:
          self.build_project_layers()

    def build_project_layers(self,):
        self.project_layers = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
          self.project_layers.append( self.build_project_layer(i, self.out_channels) )

        for i in range(self.extra_levels):
          self.z_out_dims += (self.z_out_dims[-1],)
          if i == 0 and self.extra_convs_on_inputs:
              in_channels = self.in_channels[self.backbone_end_level - 1]
          else:
              in_channels = out_channels
          self.project_layers.append( self.build_project_layer(self.backbone_end_level-1, in_channels) )
        #print(self.project_layers)
        pass

    def build_project_layer(self, level, in_channels):
      if self.proj_method == 'res_block':
        return self.build_project_layer_res_block(level, in_channels)
      else:
        return self.build_project_layer_2conv(level, in_channels)

    def build_project_layer_res_block(self, level, in_channels):
      '''
      part1 -> max pool -> part2
      '''
      s0, s1 = self.proj_z_strides[level]
      kernels = [ (3,3,s0), (1,1,s1) ] + [(3,3)]
      strides = [ (1,1,s0), (1,1,s1) ] + [(1,1)]
      proj_layer = []
      for j in range(3):
          if j < 2:
            conv_cfg = self.conv_cfg
          else:
            conv_cfg = dict(type='Conv')
          proj_j = make_res_layer_3d(
              BasicBlock3D,
              in_channels,
              self.out_channels,
              kernel=kernels[j],
              blocks=1,
              stride=strides[j],
              conv_cfg=conv_cfg,
              norm_cfg=self.norm_cfg,
              )
          proj_layer.append(proj_j)
          in_channels = self.out_channels

      part1 = nn.Sequential(*proj_layer[:-1])
      part2 = proj_layer[-1]
      return nn.ModuleList( [part1, part2] )

    def small_build_project_layer_2conv(self, level, in_channels):
      '''
      part1 -> max pool -> part2
      '''
      kernels_i =   [(1, 1, self.z_out_dims[level]),]
      strides_i =   [(1, 1, self.z_out_dims[level]),]
      n1 = len(kernels_i)
      kernels_i += [(3,3), (1,1)]
      strides_i += [(1,1), (1,1)]
      channel_rates = [4, 2, 1]

      paddings_i = [ get_padding_same_featsize(k) for k in kernels_i]
      npj = len(kernels_i)
      fpn_layer_i = []
      for j in range(npj):
          if j < n1:
            conv_cfg = self.conv_cfg
          else:
            conv_cfg = dict(type='Conv')
          fpn_conv_j = ConvModule(
              in_channels,
              self.out_channels * channel_rates[j],
              kernels_i[j],
              strides_i[j],
              padding=paddings_i[j],
              conv_cfg=conv_cfg,
              norm_cfg=self.norm_cfg,
              activation=self.activation_bev_proj,
              inplace=False)
          fpn_layer_i.append(fpn_conv_j)
          in_channels = self.out_channels * channel_rates[j]

      part1 = nn.Sequential(*fpn_layer_i[:n1])
      part2 = nn.Sequential(*fpn_layer_i[n1:])
      return nn.ModuleList( [part1, part2] )

    def build_project_layer_2conv(self, level, in_channels):
      '''
      part1 -> max pool -> part2
      '''
      from ..backbones.resnet import make_res_layer, BasicBlock, Bottleneck
      kernels_i =   [(1, 1, self.z_out_dims[level]),]
      strides_i =   [(1, 1, self.z_out_dims[level]),]

      paddings_i = [ get_padding_same_featsize(k) for k in kernels_i]
      channel_rate = 4
      part1 = ConvModule(
              in_channels,
              self.out_channels * channel_rate,
              kernels_i[0],
              strides_i[0],
              padding=paddings_i[0],
              conv_cfg=self.conv_cfg,
              norm_cfg=self.norm_cfg,
              activation=self.activation_bev_proj,
              inplace=False)

      part2 = make_res_layer( BasicBlock,
                              self.out_channels * channel_rate,
                              self.out_channels,
                              blocks=2,
                              conv_cfg=dict(type='Conv'),
                              norm_cfg=self.norm_cfg )

      return nn.ModuleList( [part1, part2] )


    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        if SHOW_NET and 0:
          print('\n\n')
          _show_tensor_ls_shapes(inputs, ' FPN_Dense3D inputs')

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            xd, yd, zd = laterals[i-1].shape[2:]
            upper_i = F.interpolate(laterals[i], scale_factor=(2, 2, self.z_stride), mode='nearest')
            #_show_tensor_ls_shapes([laterals[i-1], upper_i], f'level {i}')
            upper_i = upper_i[..., :xd, :yd, :zd]
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
          _show_tensor_ls_shapes(outs, 'FPN_Dense3D outs')
          print('\n\n')
        return tuple(outs)


    @auto_fp16()
    def forward_project(self, laterals_i, i):
        if SPARSE_BEV:
          return laterals_i[...,0]

        if self.proj_method == 'res_block':
          s0, s1 = self.proj_z_strides[i]
          zdim = s0 * s1
        else:
          zdim = self.z_out_dims[i]
        pad_size = zdim - laterals_i.shape[-1]
        if pad_size > 0:
          laterals_i = F.pad(laterals_i, (0, pad_size), "constant", 0)
        bev_laterals_i = self.project_layers[i][0](laterals_i)
        bev_laterals_i = bev_laterals_i.max(4)[0]
        bev_laterals_i = self.project_layers[i][1](bev_laterals_i)
        return bev_laterals_i

    @auto_fp16()
    def A_forward_project(self, laterals_i, i):
        if SPARSE_BEV:
          return laterals_i[...,0]

        bev_laterals_i = self.project_layers[i][0](laterals_i)
        bev_laterals_i = F.pad(bev_laterals_i, (0, self.zdim_end_level - bev_laterals_i.shape[-1]), "constant", 0)
        bev_laterals_i = self.project_layers[i][1](bev_laterals_i)
        bev_laterals_i = bev_laterals_i.max(4)[0]
        return bev_laterals_i


def make_res_layer_3d(block,
                   inplanes,
                   planes,
                   blocks,
                   kernel=3,
                   stride=1,
                   dilation=1,
                   style='pytorch',
                   with_cp=False,
                   conv_cfg=None,
                   norm_cfg=dict(type='BN'),
                   dcn=None,
                   gcb=None,
                   gen_attention=None,
                   gen_attention_blocks=[]):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False),
            build_norm_layer(norm_cfg, planes * block.expansion)[1],
        )

    layers = []
    layers.append(
        block(
            inplanes=inplanes,
            planes=planes,
            kernel=kernel,
            stride=stride,
            dilation=dilation,
            downsample=downsample,
            style=style,
            with_cp=with_cp,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            dcn=dcn,
            gcb=gcb,
            gen_attention=gen_attention if
            (0 in gen_attention_blocks) else None))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                kernel=kernel,
                stride=1,
                dilation=dilation,
                style=style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                gcb=gcb,
                gen_attention=gen_attention if
                (i in gen_attention_blocks) else None))

    return nn.Sequential(*layers)

def get_padding_same_featsize(kernel):
  if isinstance(kernel, tuple):
    padding = [int((k-1)/2) for k in kernel]
    padding = tuple(padding)
  else:
    padding = int((kernel-1)/2)
  return padding

class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 kernel=3,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 gcb=None,
                 gen_attention=None):
        super(BasicBlock3D, self).__init__()
        assert dcn is None, "Not implemented yet."
        assert gen_attention is None, "Not implemented yet."
        assert gcb is None, "Not implemented yet."

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)


        padding = get_padding_same_featsize(kernel)
        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, kernel, padding=padding, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        assert not with_cp

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

